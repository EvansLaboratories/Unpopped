// SPDX-License-Identifier: MIT OR Apache-2.0
//! Language-agnostic kernel plan — the schedule decision.
//!
//! [`build_plan`] turns an [`OpDef`] + a [`StructureKey`] cell into a neutral
//! [`KernelPlan`]: *what* to compute (the op body + dtype) and the *schedule*
//! (vectorized vs scalar) to compute it with. A [`crate::backend::Backend`]
//! lowers the plan to a concrete language. Choosing the schedule here, not in
//! the backend, keeps the decision shared across every backend.

use crate::ir::{
    Access, AccumSpec, BaseOffset, DagNode, OpDef, QualOp, ReadIndex, ReduceOp, ReduceStage,
    RoundMode, ScalarExpr, SortLimit, SortOrder, SortOut, View, WriteCombine, WriteIndex,
    is_admissible_int_reduction_operand,
};
use unpopped_vocab::{
    AxisMask, Contiguity, ElementKind, MAX_OPERANDS, OperandKey, StructureKey, VecWidth,
};

mod admissibility;
mod build;
mod contraction;
mod dtype_float;
mod dtype_int;
mod dtype_misc;
mod geometry;
mod out_post;
mod roles;
mod types;
mod views;
mod window;

pub(crate) use self::admissibility::*;
pub use self::build::*;
pub(crate) use self::contraction::*;
pub use self::dtype_float::*;
pub(crate) use self::dtype_int::*;
pub use self::dtype_misc::*;
pub(crate) use self::geometry::*;
use self::out_post::*;
pub use self::roles::*;
pub use self::types::*;
pub use self::views::*;
use self::window::*;

#[cfg(test)]
mod reduce_class_tests {
    use super::*;

    #[test]
    fn classify_axis_geometry() {
        // empty mask ⇒ legacy last-axis fast path
        assert_eq!(
            classify_reduce_axes(AxisMask::EMPTY, 3, true),
            ReduceAxisClass::InnerContig
        );
        // trailing axis (rank-1) over a contiguous input ⇒ fast path
        assert_eq!(
            classify_reduce_axes(AxisMask(0b100), 3, true),
            ReduceAxisClass::InnerContig
        );
        // outermost axis 0
        assert_eq!(
            classify_reduce_axes(AxisMask(0b001), 3, true),
            ReduceAxisClass::Outer
        );
        // interior axis 1
        assert_eq!(
            classify_reduce_axes(AxisMask(0b010), 3, true),
            ReduceAxisClass::Middle
        );
        // two reduced axes
        assert_eq!(
            classify_reduce_axes(AxisMask(0b011), 3, true),
            ReduceAxisClass::Multi
        );
        // trailing axis but a STRIDED input ⇒ no longer the contiguous fast path
        assert_eq!(
            classify_reduce_axes(AxisMask(0b100), 3, false),
            ReduceAxisClass::Middle
        );
    }
}

#[cfg(test)]
mod int_reduction_predicate_gate_validate {
    //! Rule 4 (`check_int_op_admissibility`, above): the int8 any/all/count
    //! admissibility lift. Two directions, both required: the elementwise int
    //! `Cmp*` rejection must survive UNCHANGED (negative control), and the
    //! reduction-predicate `Cmp*`/`Const` shape (`count = Sum(in != 0)`) must
    //! now be admitted (positive case). Calls `check_int_op_admissibility`
    //! directly rather than `build_plan`, to isolate the gate from unrelated
    //! key/shape plumbing.
    //!
    //! It used to call an `check_int_op_admissibility` wrapper and read
    //! rejection through `catch_unwind`. 0.6.0 removed the wrapper — each rule
    //! now has ONE implementation and one panicking caller — so these read the
    //! `Result` directly, which is both simpler and a stronger assertion:
    //! `catch_unwind` could not tell WHICH rule fired, only that something did.
    use super::{check_int_op_admissibility, check_int_op_admissibility_for};
    use crate::ir::{BinaryOp, Expr, OpDef, ReduceOp, ScalarExpr, input, konst, reduced};
    use unpopped_vocab::ElementKind;

    // The opt-in flag (`Backend::total_integer_ops`) is the ONLY thing that admits an Elementwise
    // integer `Div` or a `ConstInt` at an integer compute dtype: without it both stay refused by
    // THIS gate (the typed-expression gate would also refuse the constant, so the public path
    // cannot tell the two layers apart).
    #[test]
    fn total_ints_flag_alone_admits_int_div_and_const_int() {
        let div = OpDef::elementwise("d", 2, &[ElementKind::I32], input(0) / input(1));
        let c = Expr(ScalarExpr::ConstInt(3, ElementKind::I32));
        let cst = OpDef::elementwise("c", 1, &[ElementKind::I32], input(0) + c);
        let lt = ScalarExpr::Binary(BinaryOp::CmpLt, Box::new(input(0).0), Box::new(input(1).0));
        let sel = ScalarExpr::Select(
            Box::new(input(0).0),
            Box::new(input(0).0),
            Box::new(input(1).0),
        );
        let cmp = OpDef::elementwise("lt", 2, &[ElementKind::I32], Expr(lt));
        let sel = OpDef::elementwise("sel", 2, &[ElementKind::I32], Expr(sel));
        for op in [&div, &cst, &cmp, &sel] {
            let off = check_int_op_admissibility_for(op, ElementKind::I32, false);
            assert!(off.is_err(), "{}: refused without the flag", op.name);
            check_int_op_admissibility_for(op, ElementKind::I32, true)
                .unwrap_or_else(|e| panic!("{}: admitted with the flag: {e}", op.name));
        }
    }

    // ROADMAP U33: the flag admits an Elementwise integer `Max`/`Min` at the 32/64-bit dtypes and nothing else:
    // without it the gate still refuses it, and a narrower integer dtype stays refused even with it.
    #[test]
    fn total_ints_flag_admits_int_max_min_at_32_and_64_bit_only() {
        for bop in [BinaryOp::Max, BinaryOp::Min] {
            for t in [
                ElementKind::I32,
                ElementKind::I64,
                ElementKind::U32,
                ElementKind::U64,
            ] {
                let op = OpDef::elementwise("mm", 2, &[t], input(0).binary(bop, input(1)));
                assert!(
                    check_int_op_admissibility_for(&op, t, false).is_err(),
                    "{bop:?} {t:?}: refused without the flag"
                );
                check_int_op_admissibility_for(&op, t, true)
                    .unwrap_or_else(|e| panic!("{bop:?} {t:?}: admitted with the flag: {e}"));
            }
            let op =
                OpDef::elementwise("mm8", 2, &[ElementKind::I8], input(0).binary(bop, input(1)));
            assert!(
                check_int_op_admissibility_for(&op, ElementKind::I8, true).is_err(),
                "{bop:?} I8: stays refused with the flag"
            );
        }
    }

    // Negative control: an elementwise int Cmp* op (the FKC "comparison → U8
    // mask" shape, `OpDef::elementwise_pred`) must still decline at U8 — the
    // reduction-predicate lift must NOT loosen the Elementwise arm.
    #[test]
    fn int_elementwise_cmp_still_declines() {
        let op = OpDef::elementwise_pred(
            "elem_ne_u8",
            1,
            &[ElementKind::U8],
            input(0).binary(BinaryOp::CmpNe, konst(0.0)),
        );
        assert!(
            check_int_op_admissibility(&op, ElementKind::U8).is_err(),
            "elementwise int Cmp must still decline after the reduction-post lift"
        );
    }

    // Positive case: the `count` shape — `reduce-Sum` over a `Cmp*` predicate
    // (`in != 0`) on an S8 input, with a hetero `I64` count output (the
    // `assert_valid_out_dtype` I64 branch: `Sum` + identity `Reduced(0)`
    // post). Must be admitted — this is exactly what was blocked before the
    // rule-4 lift.
    #[test]
    fn int_reduction_predicate_cmp_admitted() {
        let mut op = OpDef::reduction(
            "count_s8",
            1,
            &[ElementKind::I8],
            input(0).binary(BinaryOp::CmpNe, konst(0.0)),
            ReduceOp::Sum,
        );
        op.out_dtype = Some(ElementKind::I64);
        check_int_op_admissibility(&op, ElementKind::I8).expect("admitted");
    }

    // Positive case, the other authored shape: `any` — a Max fold over a
    // per-element keep-mask predicate, with `post` casting the folded result
    // back through `Cmp*(Reduced(0), Const)` (the `assert_valid_out_dtype` U8
    // branch: post ROOT must be Cmp*). Exercises Cmp*/Const admission in BOTH
    // the reduction body AND post in the same op.
    #[test]
    fn int_reduction_any_body_and_post_cmp_admitted() {
        let mut op = OpDef::reduction_post(
            "any_u8",
            1,
            &[ElementKind::U8],
            input(0).binary(BinaryOp::CmpNe, konst(0.0)),
            ReduceOp::Max,
            reduced(0).binary(BinaryOp::CmpNe, konst(0.0)),
        );
        op.out_dtype = Some(ElementKind::U8);
        check_int_op_admissibility(&op, ElementKind::U8).expect("admitted");
    }

    // Double-math-hazard guard: a Cmp* Const in the admitted reduction
    // position must be exactly 0/1 — any other literal is still rejected,
    // even though the surrounding shape (Sum-fold over a Cmp* predicate) is
    // otherwise the admitted count/any/all pattern.
    #[test]
    fn int_reduction_predicate_cmp_rejects_non_01_const() {
        let op = OpDef::reduction(
            "bad_threshold_s8",
            1,
            &[ElementKind::I8],
            input(0).binary(BinaryOp::CmpGt, konst(5.0)),
            ReduceOp::Sum,
        );
        assert!(
            check_int_op_admissibility(&op, ElementKind::I8).is_err(),
            "a non-0/1 Const threshold must still decline — the leaf-or-{{0,1}} \
             pin is what keeps the lift from reopening the double-math hazard"
        );
    }

    // Composition guard: a Cmp* operand that is itself a composed expression
    // (not a leaf Input/Reduced or a 0/1 Const) must still decline in the
    // reduction predicate position — composition is out of scope for v1.
    #[test]
    fn int_reduction_predicate_cmp_rejects_composed_operand() {
        let op = OpDef::reduction(
            "composed_operand_s8",
            1,
            &[ElementKind::I8],
            (input(0) + input(0)).binary(BinaryOp::CmpNe, konst(0.0)),
            ReduceOp::Sum,
        );
        assert!(
            check_int_op_admissibility(&op, ElementKind::I8).is_err(),
            "a composed Cmp* operand must still decline in the reduction \
             predicate position — v1 pins operands to leaves or 0/1 Consts"
        );
    }

    // Root-only guard (whole-branch-review finding): a COMPOSED predicate —
    // two Cmp* nodes combined by Add, i.e. the Cmp is reached as a SUB-node
    // of the reduction body rather than the body's own root — must still
    // decline. Rule 4's lift is scoped to exactly the body/post ROOT (the
    // only shape `count = Sum(in != 0)` / `any`/`all` ever author); a Cmp
    // nested inside Add/Sub/Mul is NOT admitted by the emitter's
    // `int_reduction_predicate` (which only inspects the root node), so if
    // the gate admitted it here, the nested Cmp would fall through to the
    // FLOAT `binary_f32` speller and get folded into the `long long`
    // integer accumulator — the exact double-math hazard commit 6fdfe478
    // closed, reopened for the composed-Cmp case. Before this fix this
    // panic did NOT fire (the nested Cmp was wrongly admitted).
    #[test]
    fn int_reduction_predicate_rejects_composed_cmp_not_at_root() {
        let op = OpDef::reduction(
            "composed_predicate_s8",
            1,
            &[ElementKind::I8],
            input(0).binary(BinaryOp::CmpNe, konst(0.0))
                + input(0).binary(BinaryOp::CmpNe, konst(0.0)),
            ReduceOp::Sum,
        );
        assert!(
            check_int_op_admissibility(&op, ElementKind::I8).is_err(),
            "a Cmp* reached as a sub-node of Add/Sub/Mul (not the reduction \
             body/post ROOT) must decline — the emitter only integer-lowers a \
             Cmp* at the root, so a nested Cmp admitted here would silently \
             fall through to the float speller and launder float math into the \
             int accumulator"
        );
    }
}

#[cfg(test)]
mod contraction_role_validate {
    //! `validate_contraction_roles` unit tests: role-vector length, per-operand
    //! role-count legality (exactly one FreeM/FreeN on the correct side, exactly
    //! one ContractedK — multi-group deferred to v2), and lhs/rhs batch-count
    //! agreement. Pure predicate, wired into `build_plan`'s `Access::Contraction`
    //! arm (Task 4).
    use super::validate_contraction_roles;

    #[test]
    fn role_legality() {
        use crate::ir::{AxisRole::*, ContractionAxes};
        // canonical constructors pass
        assert!(validate_contraction_roles(&ContractionAxes::matmul(), 2, 2).is_ok());
        assert!(validate_contraction_roles(&ContractionAxes::batched_matmul(), 3, 3).is_ok());
        // transposed role order passes (rhs [N,K] → roles [FreeN, ContractedK])
        let t = ContractionAxes {
            lhs: vec![FreeM, ContractedK],
            rhs: vec![FreeN, ContractedK],
        };
        assert!(validate_contraction_roles(&t, 2, 2).is_ok());
        // batch axis in the middle passes
        let mid = ContractionAxes {
            lhs: vec![FreeM, Batch, ContractedK],
            rhs: vec![Batch, ContractedK, FreeN],
        };
        assert!(validate_contraction_roles(&mid, 3, 3).is_ok());
        // illegal: two FreeM
        let bad_m = ContractionAxes {
            lhs: vec![FreeM, FreeM],
            rhs: vec![ContractedK, FreeN],
        };
        assert!(validate_contraction_roles(&bad_m, 2, 2).is_err());
        // illegal: FreeN on lhs
        let bad_n = ContractionAxes {
            lhs: vec![FreeN, ContractedK],
            rhs: vec![ContractedK, FreeN],
        };
        assert!(validate_contraction_roles(&bad_n, 2, 2).is_err());
        // illegal: mismatched batch count
        let bad_b = ContractionAxes {
            lhs: vec![Batch, FreeM, ContractedK],
            rhs: vec![ContractedK, FreeN],
        };
        assert!(validate_contraction_roles(&bad_b, 3, 2).is_err());
        // illegal: two K (multi-group deferred)
        let bad_k = ContractionAxes {
            lhs: vec![FreeM, ContractedK, ContractedK],
            rhs: vec![ContractedK, ContractedK, FreeN],
        };
        assert!(validate_contraction_roles(&bad_k, 3, 3).is_err());
        // illegal: 2+ Batch axes (rank-4) — v1 lowers a SINGLE batch coordinate
        // (blockIdx.z); the old matched-count-only check wrongly accepted this.
        let two_batch = ContractionAxes {
            lhs: vec![Batch, Batch, FreeM, ContractedK],
            rhs: vec![Batch, Batch, ContractedK, FreeN],
        };
        assert!(
            validate_contraction_roles(&two_batch, 4, 4).is_err(),
            "2+ Batch (rank-4) is out of v1's single-batch-coord scope"
        );
        // illegal: role-vector length != rank
        assert!(validate_contraction_roles(&ContractionAxes::matmul(), 3, 2).is_err());
    }
}

#[cfg(test)]
mod multi_output_validate {
    //! Increment-1 multi-output gate-rejection tests + the hetero (dropout-class)
    //! per-output-dtype gates (G1/G3/G4). Per the house rule these call
    //! `build_plan` DIRECTLY (an emitter panic would mask a gate mutation).
    use super::{MAX_OPERANDS, Schedule, build_plan};
    use crate::ir::{Access, BinaryOp, OpDef, ReduceOp, ScalarExpr, input, konst, param};
    use unpopped_vocab::{
        ArchSku, ElementKind, OpCategory, OperandDesc, StructureKey, structure_key,
    };

    // n_operands contiguous 1D operands of `dtype`.
    fn key_dt(dtype: ElementKind, n_operands: usize) -> StructureKey {
        let a = OperandDesc::new(1, &[1024], &[1], dtype, 256);
        let ops: Vec<_> = std::iter::repeat_n(a, n_operands).collect();
        structure_key(OpCategory::BinaryElementwise, &ops, ArchSku::Sm89)
    }
    // F32 shorthand for the layout-shape rejection tests.
    fn key(n_operands: usize) -> StructureKey {
        key_dt(ElementKind::F32, n_operands)
    }

    fn mul_backward() -> OpDef {
        OpDef::elementwise_multi(
            "mul_backward",
            3,
            &[ElementKind::F32],
            vec![input(0) * input(2), input(0) * input(1)],
        )
    }

    // dropout_fw (the hetero v1 vehicle): inputs x=in0, rand=in1; params
    // keep_prob=p0, scale=p1. Output 0 (value, F32) = x·(rand<keep_prob ? scale : 0);
    // output 1 (mask, U8) = the SAME `rand<keep_prob` Cmp node (shared, hoisted by
    // cross-body CSE).
    fn dropout_fw() -> OpDef {
        let cond = || input(1).binary(BinaryOp::CmpLt, param(0));
        OpDef::elementwise_multi_hetero(
            "dropout",
            2,
            &[ElementKind::F32],
            vec![
                (input(0) * cond().select(param(1), konst(0.0)), None),
                (cond(), Some(ElementKind::U8)),
            ],
        )
    }

    // dropout key: inputs x,rand (F32) then outputs value (F32), mask (U8).
    // `even` picks the V4-vectorizable extent vs an odd extent (scalar).
    fn dropout_key_contig(even: bool) -> StructureKey {
        let ext: i64 = if even { 1 << 20 } else { 1_000_003 };
        let f = OperandDesc::new(1, &[ext], &[1], ElementKind::F32, 256);
        let u = OperandDesc::new(1, &[ext], &[1], ElementKind::U8, 256);
        structure_key(OpCategory::BinaryElementwise, &[f, f, f, u], ArchSku::Sm89)
    }
    // Transposed (column-major) rank-2 dropout key — all strided.
    fn dropout_key_strided() -> StructureKey {
        let f = OperandDesc::new(2, &[8, 4], &[1, 8], ElementKind::F32, 256);
        let u = OperandDesc::new(2, &[8, 4], &[1, 8], ElementKind::U8, 256);
        structure_key(OpCategory::BinaryElementwise, &[f, f, f, u], ArchSku::Sm89)
    }

    #[test]
    fn valid_multi_output_builds() {
        // The happy path: 3 inputs + 2 outputs = 5 operands.
        let _ = build_plan(&mul_backward(), &key(5));
    }

    #[test]
    #[should_panic(expected = "broadcast")]
    fn output_broadcast_aliases_its_writes_rejected() {
        // A broadcast (stride-0) OUTPUT aliases its own writes across iteration
        // coordinates (a write race) and is not the full output shape.
        let a = OperandDesc::new(2, &[8, 4], &[4, 1], ElementKind::F32, 256);
        // Output operand 4 (the last): broadcast inner axis (stride 0 → bcast).
        let bcast_out = OperandDesc::new(2, &[8, 4], &[1, 0], ElementKind::F32, 256);
        let ops = vec![a, a, a, a, bcast_out];
        let k = structure_key(OpCategory::BinaryElementwise, &ops, ArchSku::Sm89);
        let _ = build_plan(&mul_backward(), &k);
    }

    #[test]
    #[should_panic(expected = "shape mismatch")]
    fn operand_count_mismatch_rejected() {
        // The key must carry exactly n_inputs+n_outputs = 5 operands; a 4-operand
        // key is a declared shape/operand mismatch.
        let _ = build_plan(&mul_backward(), &key(4));
    }

    #[test]
    #[should_panic(expected = "exceeds MAX_OPERANDS")]
    fn n_outputs_overflow_max_operands_rejected() {
        // (MAX_OPERANDS - 2) inputs + 3 outputs = MAX_OPERANDS + 1 operands.
        let n_inputs = u8::try_from(MAX_OPERANDS - 2).expect("MAX_OPERANDS fits u8");
        let op = OpDef::elementwise_multi(
            "over",
            n_inputs,
            &[ElementKind::F32],
            vec![input(0), input(1), input(2)],
        );
        let _ = build_plan(&op, &key(MAX_OPERANDS));
    }

    #[test]
    #[should_panic(expected = "Reduced")]
    fn reduced_in_a_body_rejected() {
        // A multi-output body must not read Reduced (no reduction in an
        // elementwise map). Built directly (no constructor produces this).
        let mut op = OpDef::elementwise_multi(
            "bad",
            2,
            &[ElementKind::F32],
            vec![input(0) * input(1), input(0)],
        );
        op.extra_out_bodies[0] = ScalarExpr::Reduced(0);
        let _ = build_plan(&op, &key(4));
    }

    #[test]
    #[should_panic(expected = "Coord")]
    fn coord_in_a_body_rejected() {
        // A multi-output body must not read Coord (elementwise-map only in v1).
        let mut op = OpDef::elementwise_multi(
            "bad",
            2,
            &[ElementKind::F32],
            vec![input(0) * input(1), input(0)],
        );
        op.extra_out_bodies[0] = ScalarExpr::Coord(0);
        let _ = build_plan(&op, &key(4));
    }

    #[test]
    #[should_panic(expected = "Elementwise-only")]
    fn multi_output_on_a_reduction_rejected() {
        // extra_out_bodies on a non-Elementwise op — a fused reduction stores one
        // accumulator, not N bodies. Built directly by pushing onto the field.
        let mut op = OpDef::reduction("s", 1, &[ElementKind::F32], input(0), ReduceOp::Sum);
        op.extra_out_bodies.push(ScalarExpr::Input(0));
        assert!(matches!(op.access, Access::Reduction { .. }));
        // A reduction key = [input, output]; the Elementwise check fires first.
        let _ = build_plan(&op, &key(2));
    }

    // ==== Hetero multi-output (dropout-class) per-output-dtype gates ====

    #[test]
    fn dropout_hetero_multi_builds() {
        // G1 POSITIVE (build_plan-direct): dropout_fw — output 0 F32 value +
        // output 1 U8 keep-mask (a Cmp-root body) — builds. 2 inputs + 2 outputs =
        // 4 operands; the last is U8. Pins that a legal U8-with-Cmp-root hetero
        // output is admitted (kill target for M1: deleting the G1 loop / reverting
        // to `out_dtype.is_none()` makes this panic).
        let op = dropout_fw();
        let k = dropout_key_contig(false);
        let plan = build_plan(&op, &k);
        assert_eq!(plan.n_outputs, 2);
        assert_eq!(plan.out_dtype_of(0), ElementKind::F32);
        assert_eq!(plan.out_dtype_of(1), ElementKind::U8);
    }

    #[test]
    #[should_panic(expected = "the only legal per-output hetero dtype is U8")]
    fn hetero_output_non_u8_is_rejected() {
        // G1 (M3): a non-U8 hetero side-output (I32 count beside a float) has no
        // exact compute→out store conversion — rejected. Built directly (no
        // constructor produces an I32 elementwise side-output).
        let op = OpDef::elementwise_multi_hetero(
            "bad",
            2,
            &[ElementKind::F32],
            vec![
                (input(0) * input(1), None),
                (
                    input(0).binary(BinaryOp::CmpLt, input(1)),
                    Some(ElementKind::I32),
                ),
            ],
        );
        // Key output 1 keyed I32 so the operand shape checks pass; the per-output
        // dtype-legality gate (G1) is what must fire. (A key-side dtype cross-check
        // is not implementable — the structure key carries no per-operand dtype.)
        let f = OperandDesc::new(1, &[1024], &[1], ElementKind::F32, 256);
        let i = OperandDesc::new(1, &[1024], &[1], ElementKind::I32, 256);
        let k = structure_key(OpCategory::BinaryElementwise, &[f, f, f, i], ArchSku::Sm89);
        let _ = build_plan(&op, &k);
    }

    #[test]
    #[should_panic(expected = "body ROOT is not a comparison")]
    fn hetero_u8_output_with_noncmp_root_is_rejected() {
        // G1 (M2): a U8 output whose body ROOT is a Mul (not a Cmp*) would truncate
        // a raw float silently under the u8 store — rejected.
        let op = OpDef::elementwise_multi_hetero(
            "bad",
            2,
            &[ElementKind::F32],
            vec![
                (input(0), None),
                (input(0) * input(1), Some(ElementKind::U8)),
            ],
        );
        let f = OperandDesc::new(1, &[1024], &[1], ElementKind::F32, 256);
        let u = OperandDesc::new(1, &[1024], &[1], ElementKind::U8, 256);
        let k = structure_key(OpCategory::BinaryElementwise, &[f, f, f, u], ArchSku::Sm89);
        let _ = build_plan(&op, &k);
    }

    #[test]
    fn uniform_multi_output_plan_is_unchanged() {
        // G3: a UNIFORM multi-output op (empty extra_out_dtypes) resolves every
        // output to the compute dtype — `out_dtype_of(j) == dtype` for all j — so
        // the store path is byte-identical to pre-hetero. Pins the no-op guarantee.
        let op = mul_backward();
        let k = key(5);
        let plan = build_plan(&op, &k);
        assert!(plan.extra_out_dtypes.is_empty());
        for j in 0..plan.n_outputs as usize {
            assert_eq!(plan.out_dtype_of(j), plan.dtype);
        }
    }

    // ==== G4: schedule forces Scalar/Strided when any output is hetero ====

    #[test]
    fn dropout_contig_key_yields_scalar_schedule() {
        // G4: a contig V4-eligible dropout key still takes SCALAR — the U8 mask has
        // no packed vector store. (Kill target for M5: keeping `out_dtype.is_none()`
        // in the vectorize gate would route this to Vectorized.)
        let op = dropout_fw();
        let k = dropout_key_contig(true);
        let plan = build_plan(&op, &k);
        assert_eq!(
            plan.schedule,
            Schedule::Scalar,
            "hetero forces scalar on contig"
        );
    }

    #[test]
    fn dropout_strided_key_yields_strided() {
        // G4: a non-contiguous (transposed) dropout key takes STRIDED.
        let op = dropout_fw();
        let k = dropout_key_strided();
        let plan = build_plan(&op, &k);
        assert_eq!(plan.schedule, Schedule::Strided);
    }

    #[test]
    fn uniform_multi_vec4_key_still_vectorizes() {
        // G4 no-regression: a UNIFORM multi-output op on a V4 contig key STILL
        // vectorizes (all_outputs_uniform true) — the hetero gate does not steal the
        // uniform vectorized path.
        let f = OperandDesc::new(1, &[1 << 20], &[1], ElementKind::F32, 256);
        let ops: Vec<_> = std::iter::repeat_n(f, 5).collect();
        let k = structure_key(OpCategory::BinaryElementwise, &ops, ArchSku::Sm89);
        let op = mul_backward();
        let plan = build_plan(&op, &k);
        assert!(
            matches!(plan.schedule, Schedule::Vectorized { .. }),
            "uniform multi still vectorizes: {:?}",
            plan.schedule
        );
    }

    #[test]
    #[should_panic(expected = "finite")]
    fn non_finite_const_in_a_body_rejected() {
        let op = OpDef::elementwise_multi(
            "bad",
            2,
            &[ElementKind::F32],
            vec![input(0) * input(1), input(0) + konst(f64::INFINITY)],
        );
        let _ = build_plan(&op, &key(4));
    }

    // ---- Review-caught gate asymmetry: EXTRA output bodies must be walked by the
    // half-Nextafter and int-op-admissibility gates too (they seeded only op.body).

    #[test]
    #[should_panic(expected = "requires LEAF")]
    fn composed_int_op_operand_in_an_extra_body_rejected_at_the_plan_gate() {
        use crate::ir::BinaryOp;
        let op = OpDef::elementwise_multi(
            "bad",
            3,
            &[ElementKind::U8],
            vec![
                input(0) + input(1),
                (input(0) + input(1)).binary(BinaryOp::Shr, input(2)),
            ],
        );
        let _ = build_plan(&op, &key_dt(ElementKind::U8, 5));
    }

    #[test]
    #[should_panic(expected = "must miss honestly")]
    fn half_nextafter_in_an_extra_body_rejected_at_the_plan_gate() {
        use crate::ir::BinaryOp;
        let op = OpDef::elementwise_multi(
            "bad",
            1,
            &[ElementKind::F16],
            vec![
                input(0) * input(0),
                input(0).binary(BinaryOp::Nextafter, input(0)),
            ],
        );
        let _ = build_plan(&op, &key_dt(ElementKind::F16, 3));
    }
}

#[cfg(test)]
mod rowreduce_role_validate {
    //! Increment-2 RowReduce role tests: `rr_role` classification units + gate
    //! tests that call `build_plan` DIRECTLY (the house rule — an emitter panic
    //! would mask a gate mutation). Covers the new `RowScalar` role, the lifted
    //! "inputs>0 must be column-broadcast" restriction, and the rejected-ambiguous
    //! cases.
    use super::{RrRole, build_plan, rr_role};
    use crate::ir::{OpDef, ReduceOp, ReduceStage, input, reduced};
    use unpopped_vocab::{
        ArchSku, AxisMask, Contiguity, DivBucket, ElementKind, OpCategory, OperandDesc, OperandKey,
        VecWidth, structure_key,
    };

    // A minimal OperandKey carrying only the broadcast mask + flip (all rr_role /
    // validate read for classification); contig is irrelevant to the role.
    fn opkey(bcast: u8, flipped: bool) -> OperandKey {
        OperandKey::new(
            Contiguity::Broadcast,
            AxisMask(bcast),
            VecWidth::Scalar,
            DivBucket::Any,
            flipped,
        )
    }

    #[test]
    fn rr_role_classifies_the_three_geometries_and_the_ambiguous_case() {
        // rank 2 ⇒ feature (last) axis = 1.
        assert_eq!(rr_role(opkey(0b00, false), 1), RrRole::RowStreamed); // nothing bcast
        assert_eq!(rr_role(opkey(0b10, false), 1), RrRole::RowScalar); // feature bcast ⇒ per-row
        assert_eq!(rr_role(opkey(0b01, false), 1), RrRole::ColBroadcast); // outer bcast ⇒ per-col
        // A varying feature axis is NEVER a RowScalar (that is exactly ColBroadcast).
        assert_ne!(rr_role(opkey(0b01, false), 1), RrRole::RowScalar);
        // All-broadcast is ambiguous: classified RowScalar (last is set), then REJECTED
        // by validate's outer-axis-clear check (a true scalar is a Const, not an operand).
        assert_eq!(rr_role(opkey(0b11, false), 1), RrRole::RowScalar);
    }

    // A full-width row-streamed operand [256,128].
    fn stream() -> OperandDesc {
        OperandDesc::new(2, &[256, 128], &[128, 1], ElementKind::F32, 256)
    }
    // A per-row scalar: [n_out,k]-presented, strides [1,0] (feature-axis broadcast,
    // outer varies dense).
    fn rowscalar() -> OperandDesc {
        OperandDesc::new(2, &[256, 128], &[1, 0], ElementKind::F32, 256)
    }

    #[test]
    fn second_row_streamed_input_builds() {
        // softmax bw: y, dy both row-streamed [n,k] + dx. The former guard rejected
        // input>0 unless column-broadcast; the lift makes this the point — it PASSES.
        let op = OpDef::row_reduce(
            "softmax_bw",
            2,
            &[ElementKind::F32],
            vec![ReduceStage {
                pre: (input(0) * input(1)).0,
                op: ReduceOp::Sum,
            }],
            input(0) * (input(1) - reduced(0)),
        );
        let s = stream();
        let key = structure_key(OpCategory::Softmax, &[s, s, s], ArchSku::Sm89);
        let _ = build_plan(&op, &key);
    }

    #[test]
    fn row_scalar_inputs_build() {
        // layer_norm bw: x, dy row-streamed; mean, rstd per-row scalars (used INSIDE a
        // stage pre — x_hat — and the epilogue).
        let x_hat = (input(0) - input(2)) * input(3);
        let op = OpDef::row_reduce(
            "layer_norm_bw",
            4,
            &[ElementKind::F32],
            vec![
                ReduceStage {
                    pre: input(1).0,
                    op: ReduceOp::Mean,
                },
                ReduceStage {
                    pre: (input(1) * x_hat.clone()).0,
                    op: ReduceOp::Mean,
                },
            ],
            input(3) * (input(1) - reduced(0) - x_hat * reduced(1)),
        );
        let s = stream();
        let rs = rowscalar();
        let key = structure_key(OpCategory::Normalization, &[s, s, rs, rs, s], ArchSku::Sm89);
        let _ = build_plan(&op, &key);
    }

    // A 2-input op whose input 1 is the offending operand under test.
    fn probe_op() -> OpDef {
        OpDef::row_reduce(
            "probe",
            2,
            &[ElementKind::F32],
            vec![ReduceStage {
                pre: input(0).0,
                op: ReduceOp::Sum,
            }],
            input(0) + input(1),
        )
    }

    #[test]
    #[should_panic(expected = "true scalar")]
    fn all_broadcast_input_rejected_as_a_true_scalar() {
        // strides [0,0] ⇒ both axes broadcast ⇒ classified RowScalar (last set) then
        // rejected by the outer-axis-clear check — a genuinely ambiguous mask.
        let s = stream();
        let allb = OperandDesc::new(2, &[256, 128], &[0, 0], ElementKind::F32, 256);
        let key = structure_key(OpCategory::Softmax, &[s, allb, s], ArchSku::Sm89);
        let _ = build_plan(&probe_op(), &key);
    }

    #[test]
    #[should_panic(expected = "reversed")]
    fn row_scalar_flipped_rejected() {
        // feature-axis broadcast (RowScalar) but a NEGATIVE outer stride ⇒ flipped.
        let s = stream();
        let flipped_rs = OperandDesc::new(2, &[256, 128], &[-1, 0], ElementKind::F32, 256);
        let key = structure_key(OpCategory::Softmax, &[s, flipped_rs, s], ArchSku::Sm89);
        let _ = build_plan(&probe_op(), &key);
    }

    #[test]
    #[should_panic(expected = "reversed")]
    fn flipped_row_streamed_input_rejected() {
        // Review #2 CRITICAL: a dense-but-REVERSED second row-streamed input
        // (empty bcast ⇒ RowStreamed; strides [128,-1] ⇒ |stride|-contig=Contig
        // but flipped=true). Pre-fix the RowStreamed branch checked only Contig
        // and accepted it, then the emitter read it forward (mirrored/OOB). Now
        // rejected, matching the ColBroadcast/RowScalar branches. Via build_plan
        // DIRECTLY so only the plan gate can fire.
        let s = stream();
        let flipped = OperandDesc::new(2, &[256, 128], &[128, -1], ElementKind::F32, 256);
        let key = structure_key(OpCategory::Softmax, &[s, flipped, s], ArchSku::Sm89);
        let _ = build_plan(&probe_op(), &key);
    }

    #[test]
    #[should_panic(expected = "every outer")]
    fn rank3_partial_middle_broadcast_rejected() {
        // rank 3, only the MIDDLE axis broadcast: feature (last) varies ⇒ ColBroadcast,
        // but it fails "must broadcast every outer axis" (axis 0 is not broadcast) — an
        // ambiguous partial broadcast, neither a clean column nor a row-scalar.
        let x = OperandDesc::new(3, &[4, 8, 16], &[128, 16, 1], ElementKind::F32, 256);
        let mid = OperandDesc::new(3, &[4, 8, 16], &[16, 0, 1], ElementKind::F32, 256);
        let out = OperandDesc::new(3, &[4, 8, 16], &[128, 16, 1], ElementKind::F32, 256);
        let key = structure_key(OpCategory::Softmax, &[x, mid, out], ArchSku::Sm89);
        let _ = build_plan(&probe_op(), &key);
    }

    #[test]
    #[should_panic(expected = "Input0")]
    fn input0_as_row_scalar_rejected() {
        // Input 0 (the reduced tensor) presented feature-broadcast (a row-scalar) is
        // illegal — it must be the row-streamed x.
        let op = OpDef::row_reduce(
            "t",
            1,
            &[ElementKind::F32],
            vec![ReduceStage {
                pre: input(0).0,
                op: ReduceOp::Sum,
            }],
            reduced(0),
        );
        let key = structure_key(OpCategory::Softmax, &[rowscalar(), stream()], ArchSku::Sm89);
        let _ = build_plan(&op, &key);
    }
}

#[cfg(test)]
mod view_gate_validate {
    //! Item-01 layout-view gate-rejection + schedule-routing tests. Per the house
    //! rule these call `build_plan` DIRECTLY — an emitter panic would mask a gate
    //! mutation (the 0c lesson).
    use super::{Schedule, build_plan};
    use crate::ir::{OpDef, ReduceOp, View, input};
    use unpopped_vocab::{
        ArchSku, AxisMask, ElementKind, OpCategory, OperandDesc, StructureKey, structure_key,
    };

    // A rank-2 contiguous [128,256] f32 cell (1 input + 1 output) — the input keys
    // Contig + a vector width, so a view-free relu VECTORIZES here.
    fn contig_2d_key() -> StructureKey {
        let a = OperandDesc::new(2, &[128, 256], &[256, 1], ElementKind::F32, 256);
        structure_key(OpCategory::UnaryElementwise, &[a, a], ArchSku::Sm89)
    }

    fn relu() -> OpDef {
        OpDef::elementwise("relu", 1, &[ElementKind::F32], input(0).relu())
    }
    fn relu_t() -> OpDef {
        relu().with_views(vec![View::Permute { perm: vec![1, 0] }])
    }

    #[test]
    fn transpose_view_forces_strided_off_the_vectorized_path() {
        let key = contig_2d_key();
        // Baseline: the view-free relu vectorizes on this contiguous cell.
        assert!(
            matches!(
                build_plan(&relu(), &key).schedule,
                Schedule::Vectorized { .. }
            ),
            "precondition: the view-free relu must vectorize on a contiguous cell"
        );
        // A Permute view forces the STRIDED schedule (a transposed read is
        // non-contiguous; only the strided emitter folds the stride remap).
        assert_eq!(build_plan(&relu_t(), &key).schedule, Schedule::Strided);
        // And the plan carries the view through to the backend.
        assert_eq!(build_plan(&relu_t(), &key).views.len(), 1);
    }

    #[test]
    fn identity_views_route_exactly_like_view_free() {
        let key = contig_2d_key();
        // An all-Identity views vec is byte-identical to view-free: same schedule.
        let identated = OpDef::elementwise("relu", 1, &[ElementKind::F32], input(0).relu())
            .with_views(vec![View::Identity]);
        assert_eq!(
            build_plan(&relu(), &key).schedule,
            build_plan(&identated, &key).schedule,
            "all-Identity views must not change the schedule"
        );
        assert!(matches!(
            build_plan(&identated, &key).schedule,
            Schedule::Vectorized { .. }
        ));
    }

    #[test]
    fn same_rank_reshape_is_not_addressing_and_does_not_force_strided() {
        // A same-rank Reshape is an identity linear map (recognition/keying only) —
        // it must NOT force Strided (unlike Permute/Broadcast).
        let op = OpDef::elementwise("relu", 1, &[ElementKind::F32], input(0).relu())
            .with_views(vec![View::Reshape { producer_rank: 2 }]);
        assert!(matches!(
            build_plan(&op, &contig_2d_key()).schedule,
            Schedule::Vectorized { .. }
        ));
    }

    #[test]
    #[should_panic(expected = "true permutation")]
    fn invalid_permutation_rejected() {
        // perm [0,0] is not a permutation of 0..2 (duplicate axis).
        let op = relu().with_views(vec![View::Permute { perm: vec![0, 0] }]);
        let _ = build_plan(&op, &contig_2d_key());
    }

    #[test]
    #[should_panic(expected = "admitted only on Access::Elementwise")]
    fn permute_view_on_reduction_rejected() {
        // A non-Identity view on a Reduction op: rejected (reductions own their
        // axis machinery). Build the OpDef with a view via with_views. Task 4
        // widened Gate B's admitted set to Elementwise|Contraction and updated
        // its message accordingly — Reduction is still outside that set.
        let op = OpDef::reduction("sum", 1, &[ElementKind::F32], input(0), ReduceOp::Sum)
            .with_views(vec![View::Permute { perm: vec![1, 0] }]);
        let a = OperandDesc::new(2, &[256, 128], &[128, 1], ElementKind::F32, 256);
        let o = OperandDesc::new(1, &[256], &[1], ElementKind::F32, 256);
        let key = structure_key(OpCategory::Reduction, &[a, o], ArchSku::Sm89);
        let _ = build_plan(&op, &key);
    }

    #[test]
    #[should_panic(expected = "orthogonal")]
    fn permute_with_broadcast_operand_rejected() {
        // Input 0 is broadcast on an axis AND carries a Permute view — v1 keeps
        // them orthogonal.
        let bcast_in = OperandDesc::new(2, &[128, 256], &[0, 1], ElementKind::F32, 256);
        let out = OperandDesc::new(2, &[128, 256], &[256, 1], ElementKind::F32, 256);
        let key = structure_key(
            OpCategory::UnaryElementwise,
            &[bcast_in, out],
            ArchSku::Sm89,
        );
        let _ = build_plan(&relu_t(), &key);
    }

    #[test]
    #[should_panic(expected = "does not broadcast")]
    fn broadcast_view_disagreeing_with_key_rejected() {
        // The view declares axis 0 broadcast, but the key operand is dense (no
        // broadcast) — a lie the key-driven emitter would ignore.
        let op = relu().with_views(vec![View::Broadcast {
            bcast: AxisMask(0b01),
        }]);
        let _ = build_plan(&op, &contig_2d_key());
    }

    #[test]
    #[should_panic(expected = "rank-change")]
    fn rank_change_reshape_rejected() {
        // producer_rank 3 != iteration rank 2 — genuine rank-change emit (items
        // 03/10), out of item-01 scope.
        let op = relu().with_views(vec![View::Reshape { producer_rank: 3 }]);
        let _ = build_plan(&op, &contig_2d_key());
    }

    #[test]
    #[should_panic(expected = "deferred composition")]
    fn viewed_multi_output_rejected() {
        // A viewed input on a multi-output op is deferred in v1.
        let op = OpDef::elementwise_multi(
            "dual",
            1,
            &[ElementKind::F32],
            vec![input(0).relu(), input(0)],
        )
        .with_views(vec![View::Permute { perm: vec![1, 0] }]);
        let a = OperandDesc::new(2, &[128, 256], &[256, 1], ElementKind::F32, 256);
        let key = structure_key(OpCategory::UnaryElementwise, &[a, a, a], ArchSku::Sm89);
        let _ = build_plan(&op, &key);
    }

    #[test]
    #[should_panic(expected = "must equal n_inputs")]
    fn views_len_mismatch_rejected() {
        // Bypass the with_views debug_assert to prove the plan gate's own
        // release-path length check (views.len() != n_inputs).
        let mut op = OpDef::elementwise("add", 2, &[ElementKind::F32], input(0) + input(1));
        op.views = vec![View::Permute { perm: vec![1, 0] }]; // len 1, n_inputs 2
        let a = OperandDesc::new(2, &[128, 256], &[256, 1], ElementKind::F32, 256);
        let key = structure_key(OpCategory::BinaryElementwise, &[a, a, a], ArchSku::Sm89);
        let _ = build_plan(&op, &key);
    }

    #[test]
    fn reduction_with_identity_view_passes_through() {
        // A trivially-Identity view on a reduction is allowed (pass-through).
        let op = OpDef::reduction("sum", 1, &[ElementKind::F32], input(0), ReduceOp::Sum)
            .with_views(vec![View::Identity]);
        let a = OperandDesc::new(2, &[256, 128], &[128, 1], ElementKind::F32, 256);
        let o = OperandDesc::new(1, &[256], &[1], ElementKind::F32, 256);
        let key = structure_key(OpCategory::Reduction, &[a, o], ArchSku::Sm89);
        let _ = build_plan(&op, &key); // no panic — trivially-Identity pass-through
    }

    #[test]
    fn contraction_admits_permute_view_on_rhs() {
        use crate::ir::{ContractionAxes, View, reduced};
        // matmul with a Permute (transpose) view on rhs; lhs Identity.
        // views.len() MUST equal n_inputs (== 2 for a bias-free matmul).
        let op = OpDef::contraction(
            "matmul",
            &[ElementKind::F32],
            ContractionAxes::matmul(),
            reduced(0),
        )
        .with_views(vec![View::Identity, View::Permute { perm: vec![1, 0] }]);
        // Transposed-rhs key from Task 2's fixture: rhs physically [N,K] (K unit-stride).
        let lhs = OperandDesc::new(2, &[8, 16], &[16, 1], ElementKind::F32, 256);
        let rhs = OperandDesc::new(2, &[16, 4], &[1, 16], ElementKind::F32, 256);
        let out = OperandDesc::new(2, &[8, 4], &[4, 1], ElementKind::F32, 256);
        let key = structure_key(OpCategory::Gemm, &[lhs, rhs, out], ArchSku::Sm89);
        let plan = build_plan(&op, &key);
        assert!(
            matches!(plan.schedule, Schedule::Contraction),
            "permute-viewed contraction must route to Schedule::Contraction, got {:?}",
            plan.schedule
        );
    }

    #[test]
    #[should_panic(expected = "exactly one FreeM")]
    fn contraction_rejects_illegal_roles() {
        use crate::ir::{AxisRole::*, ContractionAxes, reduced};
        // Illegal: two FreeM on lhs. OpDef::contraction takes axes directly.
        let op = OpDef::contraction(
            "matmul",
            &[ElementKind::F32],
            ContractionAxes {
                lhs: vec![FreeM, FreeM],
                rhs: vec![ContractedK, FreeN],
            },
            reduced(0),
        );
        // Canonical key is fine — role validation panics before the key assert.
        let lhs = OperandDesc::new(2, &[8, 16], &[16, 1], ElementKind::F32, 256);
        let rhs = OperandDesc::new(2, &[16, 4], &[4, 1], ElementKind::F32, 256);
        let out = OperandDesc::new(2, &[8, 4], &[4, 1], ElementKind::F32, 256);
        let key = structure_key(OpCategory::Gemm, &[lhs, rhs, out], ArchSku::Sm89);
        let _ = build_plan(&op, &key);
    }
}

#[cfg(test)]
mod gather_gate_validate {
    //! Increment-4 GATHER gate-rejection + schedule-routing tests. Per the house
    //! rule these call `build_plan` DIRECTLY — an emitter panic would mask a gate
    //! mutation (the 0c lesson).
    use super::{Schedule, build_plan};
    use crate::ir::{OobPolicy, OpDef, ReadIndex, ReduceOp, input};
    use unpopped_vocab::{
        ArchSku, ElementKind, OpCategory, OperandDesc, StructureKey, structure_key,
    };

    // rank-2 gather cell: data [4,3] (input 0), index [4,3] (input 1), out [4,3].
    fn gather_key(idx_dt: ElementKind) -> StructureKey {
        let data = OperandDesc::new(2, &[4, 3], &[3, 1], ElementKind::F32, 256);
        let idx = OperandDesc::new(2, &[4, 3], &[3, 1], idx_dt, 256);
        let out = OperandDesc::new(2, &[4, 3], &[3, 1], ElementKind::F32, 256);
        structure_key(
            OpCategory::BinaryElementwise,
            &[data, idx, out],
            ArchSku::Sm89,
        )
    }

    #[test]
    fn gather_forces_the_strided_schedule() {
        let op = OpDef::gather(
            "g",
            &[ElementKind::F32],
            0,
            OobPolicy::Skip,
            ElementKind::I32,
        );
        assert_eq!(
            build_plan(&op, &gather_key(ElementKind::I32)).schedule,
            Schedule::Strided
        );
        // The plan carries the read_index through to the backend.
        assert_eq!(
            build_plan(&op, &gather_key(ElementKind::I32))
                .read_index
                .len(),
            2
        );
    }

    #[test]
    #[should_panic(expected = "index_dtype must be an integer")]
    fn non_integer_index_operand_rejected() {
        // A float index dtype is meaningless (the emitted load type must be int).
        let op = OpDef::gather(
            "g",
            &[ElementKind::F32],
            0,
            OobPolicy::Skip,
            ElementKind::F32,
        );
        let _ = build_plan(&op, &gather_key(ElementKind::F32));
    }

    #[test]
    #[should_panic(expected = "gather axis")]
    fn axis_ge_rank_rejected() {
        // axis 2 on a rank-2 cell.
        let op = OpDef::gather(
            "g",
            &[ElementKind::F32],
            2,
            OobPolicy::Skip,
            ElementKind::I32,
        );
        let _ = build_plan(&op, &gather_key(ElementKind::I32));
    }

    #[test]
    #[should_panic(expected = "index_operand")]
    fn index_operand_ge_n_inputs_rejected() {
        // index_operand 2 but only 2 inputs (valid indices 0,1).
        let op = OpDef::elementwise("g", 2, &[ElementKind::F32], input(0)).with_indexed(vec![
            ReadIndex::Indexed {
                index_operand: 2,
                axis: 0,
                oob: OobPolicy::Skip,
                index_dtype: ElementKind::I32,
            },
            ReadIndex::Direct,
        ]);
        let _ = build_plan(&op, &gather_key(ElementKind::I32));
    }

    #[test]
    #[should_panic(expected = "Elementwise-only")]
    fn gather_on_a_reduction_rejected() {
        // A gather (Indexed read) on a Reduction op: rejected (reductions own their
        // axis machinery). read_index length must equal n_inputs (1 here).
        let op = OpDef::reduction("sum", 1, &[ElementKind::F32], input(0), ReduceOp::Sum)
            .with_indexed(vec![ReadIndex::Indexed {
                index_operand: 0,
                axis: 0,
                oob: OobPolicy::Skip,
                index_dtype: ElementKind::I32,
            }]);
        let a = OperandDesc::new(2, &[256, 128], &[128, 1], ElementKind::F32, 256);
        let o = OperandDesc::new(1, &[256], &[1], ElementKind::F32, 256);
        let key = structure_key(OpCategory::Reduction, &[a, o], ArchSku::Sm89);
        let _ = build_plan(&op, &key);
    }

    #[test]
    #[should_panic(expected = "exactly one gathered input")]
    fn two_gathered_inputs_rejected() {
        // Two Indexed inputs — v1 emitter handles exactly one substituted axis.
        let op = OpDef::elementwise("g", 3, &[ElementKind::F32], input(0) + input(1)).with_indexed(
            vec![
                ReadIndex::Indexed {
                    index_operand: 2,
                    axis: 0,
                    oob: OobPolicy::Skip,
                    index_dtype: ElementKind::I32,
                },
                ReadIndex::Indexed {
                    index_operand: 2,
                    axis: 0,
                    oob: OobPolicy::Skip,
                    index_dtype: ElementKind::I32,
                },
                ReadIndex::Direct,
            ],
        );
        let data = OperandDesc::new(2, &[4, 3], &[3, 1], ElementKind::F32, 256);
        let idx = OperandDesc::new(2, &[4, 3], &[3, 1], ElementKind::I32, 256);
        let key = structure_key(
            OpCategory::BinaryElementwise,
            &[data, data, idx, data],
            ArchSku::Sm89,
        );
        let _ = build_plan(&op, &key);
    }

    #[test]
    #[should_panic(expected = "gather \u{22a5} view")]
    fn gather_plus_view_on_the_same_input_rejected() {
        use crate::ir::View;
        // The gathered input 0 also carries a Permute view — gather ⊥ view in v1.
        let op = OpDef::gather(
            "g",
            &[ElementKind::F32],
            0,
            OobPolicy::Skip,
            ElementKind::I32,
        )
        .with_views(vec![View::Permute { perm: vec![1, 0] }, View::Identity]);
        let _ = build_plan(&op, &gather_key(ElementKind::I32));
    }
}

#[cfg(test)]
mod scatter_gate_validate {
    //! Increment-5 SCATTER gate-rejection + schedule-routing tests. Per the house
    //! rule these call `build_plan` DIRECTLY — an emitter panic would mask a gate
    //! mutation (the 0c lesson).
    use super::{Schedule, build_plan};
    use crate::ir::{OobPolicy, OpDef, ReduceOp, WriteCombine, WriteIndex, input};
    use unpopped_vocab::{
        ArchSku, ElementKind, OpCategory, OperandDesc, StructureKey, structure_key,
    };

    // rank-2 scatter cell: updates [4,3] (input 0), index [4,3] (input 1), dst
    // [4,3] (out slot). The dst extent along the scattered axis rides `sext` at
    // launch; here the key dst just supplies the strides/broadcast facts.
    fn scatter_key(idx_dt: ElementKind) -> StructureKey {
        let upd = OperandDesc::new(2, &[4, 3], &[3, 1], ElementKind::F32, 256);
        let idx = OperandDesc::new(2, &[4, 3], &[3, 1], idx_dt, 256);
        let dst = OperandDesc::new(2, &[4, 3], &[3, 1], ElementKind::F32, 256);
        structure_key(
            OpCategory::BinaryElementwise,
            &[upd, idx, dst],
            ArchSku::Sm89,
        )
    }

    #[test]
    fn scatter_forces_the_strided_schedule() {
        let op = OpDef::scatter("s", &[ElementKind::F32], 0, ElementKind::I32);
        assert_eq!(
            build_plan(&op, &scatter_key(ElementKind::I32)).schedule,
            Schedule::Strided
        );
        // The plan carries the write role through to the backend.
        assert!(
            !build_plan(&op, &scatter_key(ElementKind::I32))
                .write_index
                .is_direct()
        );
    }

    #[test]
    #[should_panic(expected = "index_dtype must be an integer")]
    fn non_integer_scatter_index_rejected() {
        let op = OpDef::scatter("s", &[ElementKind::F32], 0, ElementKind::F32);
        let _ = build_plan(&op, &scatter_key(ElementKind::F32));
    }

    #[test]
    #[should_panic(expected = "scatter axis")]
    fn scatter_axis_ge_rank_rejected() {
        let op = OpDef::scatter("s", &[ElementKind::F32], 2, ElementKind::I32);
        let _ = build_plan(&op, &scatter_key(ElementKind::I32));
    }

    #[test]
    #[should_panic(expected = "index_operand")]
    fn scatter_index_operand_ge_n_inputs_rejected() {
        let op = OpDef::elementwise("s", 2, &[ElementKind::F32], input(0)).with_scatter(
            WriteIndex::ScatterIndexed {
                index_operand: 2,
                axis: 0,
                combine: WriteCombine::Assign,
                oob: OobPolicy::Skip,
                index_dtype: ElementKind::I32,
            },
        );
        let _ = build_plan(&op, &scatter_key(ElementKind::I32));
    }

    #[test]
    #[should_panic(expected = "Elementwise-only")]
    fn scatter_on_a_reduction_rejected() {
        let op = OpDef::reduction("sum", 1, &[ElementKind::F32], input(0), ReduceOp::Sum)
            .with_scatter(WriteIndex::ScatterIndexed {
                index_operand: 0,
                axis: 0,
                combine: WriteCombine::AtomicAdd,
                oob: OobPolicy::Skip,
                index_dtype: ElementKind::I32,
            });
        let a = OperandDesc::new(2, &[256, 128], &[128, 1], ElementKind::F32, 256);
        let o = OperandDesc::new(1, &[256], &[1], ElementKind::F32, 256);
        let key = structure_key(OpCategory::Reduction, &[a, o], ArchSku::Sm89);
        let _ = build_plan(&op, &key);
    }

    #[test]
    #[should_panic(expected = "not legal for output dtype")]
    fn float_atomic_max_rejected() {
        // AtomicMax on a float output is not native — integer-only in v1.
        let op = OpDef::elementwise("smax", 2, &[ElementKind::F32], input(0)).with_scatter(
            WriteIndex::ScatterIndexed {
                index_operand: 1,
                axis: 0,
                combine: WriteCombine::AtomicMax,
                oob: OobPolicy::Skip,
                index_dtype: ElementKind::I32,
            },
        );
        let _ = build_plan(&op, &scatter_key(ElementKind::I32));
    }

    #[test]
    #[should_panic(expected = "OOB policy must be Skip")]
    fn scatter_zerofill_rejected() {
        let op = OpDef::elementwise("s", 2, &[ElementKind::F32], input(0)).with_scatter(
            WriteIndex::ScatterIndexed {
                index_operand: 1,
                axis: 0,
                combine: WriteCombine::Assign,
                oob: OobPolicy::ZeroFill,
                index_dtype: ElementKind::I32,
            },
        );
        let _ = build_plan(&op, &scatter_key(ElementKind::I32));
    }

    #[test]
    #[should_panic(expected = "identity value read")]
    fn composed_scatter_body_rejected() {
        // Review #5 CRITICAL: a composed scatter body would be silently DROPPED by
        // the deterministic gather-sum base (it sums the value operand directly).
        // A fused `relu(updates)` scatter_add is a v1 deferral. Via build_plan
        // DIRECTLY so only the plan gate can fire (not an emitter panic).
        let op = OpDef::elementwise("s", 2, &[ElementKind::F32], input(0).relu()).with_scatter(
            WriteIndex::ScatterIndexed {
                index_operand: 1,
                axis: 0,
                combine: WriteCombine::AtomicAdd,
                oob: OobPolicy::Skip,
                index_dtype: ElementKind::I32,
            },
        );
        let _ = build_plan(&op, &scatter_key(ElementKind::I32));
    }

    #[test]
    fn integer_scatter_add_is_admitted() {
        // Integer AtomicAdd (bincount-class) is deterministic and legal.
        let op = OpDef::scatter_add("isa", &[ElementKind::I32], 0, ElementKind::I32);
        let iupd = OperandDesc::new(2, &[4, 3], &[3, 1], ElementKind::I32, 256);
        let idx = OperandDesc::new(2, &[4, 3], &[3, 1], ElementKind::I32, 256);
        let dst = OperandDesc::new(2, &[4, 3], &[3, 1], ElementKind::I32, 256);
        let key = structure_key(
            OpCategory::BinaryElementwise,
            &[iupd, idx, dst],
            ArchSku::Sm89,
        );
        assert_eq!(build_plan(&op, &key).schedule, Schedule::Strided);
    }
}

#[cfg(test)]
mod scan_gate_validate {
    //! Increment-6 SCAN gate-rejection tests. Per the house rule these call
    //! `build_plan` DIRECTLY — an emitter panic would mask a gate mutation (the 0c
    //! lesson). Every `validate_scan` (and `assert_valid_out_dtype`) rejection has a
    //! test here; each is mutation-checked both directions by a targeted reverse-edit.
    use super::{Schedule, build_plan};
    use crate::ir::{OpDef, ReduceOp, input, konst, reduced};
    use unpopped_vocab::{
        ArchSku, ElementKind, OpCategory, OperandDesc, StructureKey, structure_key,
    };

    // A rank-2 [256,128] scan cell: contiguous input + full-width contiguous output.
    fn scan_key(dt: ElementKind) -> StructureKey {
        let a = OperandDesc::new(2, &[256, 128], &[128, 1], dt, 256);
        let o = OperandDesc::new(2, &[256, 128], &[128, 1], dt, 256);
        structure_key(OpCategory::UnaryElementwise, &[a, o], ArchSku::Sm89)
    }

    #[test]
    fn valid_scan_builds_all_monoids_and_flags() {
        // Every v1 monoid × inclusive/exclusive × forward/reverse builds; the
        // schedule is the serial base (block=false), innermost axis (rank-1 = 1).
        for op in [ReduceOp::Sum, ReduceOp::Prod, ReduceOp::Max, ReduceOp::Min] {
            for reverse in [false, true] {
                for exclusive in [false, true] {
                    let sc =
                        OpDef::scan_simple("cum", &[ElementKind::F32], op, 1, reverse, exclusive);
                    let key = scan_key(ElementKind::F32);
                    let plan = build_plan(&sc, &key);
                    assert_eq!(
                        plan.schedule,
                        Schedule::Scan {
                            op,
                            axis: 1,
                            reverse,
                            exclusive,
                            block: false
                        }
                    );
                }
            }
        }
    }

    #[test]
    fn integer_scan_builds_sum_max_min() {
        // Integer Sum/Max/Min ride the serial base BitIdentical — validate_scan
        // does NOT copy check_row_reduce's float-only gate.
        for op in [ReduceOp::Sum, ReduceOp::Max, ReduceOp::Min] {
            let sc = OpDef::scan_simple("cumi", &[ElementKind::I32], op, 1, false, false);
            let _ = build_plan(&sc, &scan_key(ElementKind::I32));
        }
    }

    #[test]
    fn prod_is_admitted_unlike_rowreduce() {
        // Prod is admitted (cumprod).
        let sc = OpDef::scan_simple(
            "cumprod",
            &[ElementKind::F32],
            ReduceOp::Prod,
            1,
            false,
            false,
        );
        let key = scan_key(ElementKind::F32);
        let plan = build_plan(&sc, &key);
        assert!(matches!(
            plan.schedule,
            Schedule::Scan {
                op: ReduceOp::Prod,
                ..
            }
        ));
    }

    #[test]
    fn mean_combine_is_admitted_at_a_float_dtype() {
        // 0.42.10: a running mean plans (an Unpopped extension; KISS#766)
        let sc = OpDef::scan_simple(
            "cummean",
            &[ElementKind::F32],
            ReduceOp::Mean,
            1,
            false,
            false,
        );
        let _ = build_plan(&sc, &scan_key(ElementKind::F32));
    }

    #[test]
    #[should_panic(expected = "requires a float dtype")]
    fn mean_scan_at_an_integer_dtype_is_rejected() {
        let sc = OpDef::scan_simple(
            "cummean",
            &[ElementKind::I32],
            ReduceOp::Mean,
            1,
            false,
            false,
        );
        let _ = build_plan(&sc, &scan_key(ElementKind::I32));
    }

    #[test]
    fn a_non_inner_axis_plans_for_dense_inputs() {
        // 0.42.11: axis 0 on a rank-2 cell (a strided scan over dense operands)
        let sc = OpDef::scan_simple("cum0", &[ElementKind::F32], ReduceOp::Sum, 0, false, false);
        let key = scan_key(ElementKind::F32);
        let plan = build_plan(&sc, &key);
        assert!(matches!(plan.schedule, Schedule::Scan { axis: 0, .. }));
    }

    #[test]
    #[should_panic(expected = "outside the rank")]
    fn an_axis_outside_the_rank_is_rejected() {
        let sc = OpDef::scan_simple("cum9", &[ElementKind::F32], ReduceOp::Sum, 2, false, false);
        let _ = build_plan(&sc, &scan_key(ElementKind::F32));
    }

    #[test]
    #[should_panic(expected = "must be contiguous")]
    fn non_contig_scanned_input_rejected() {
        // A transposed (column-major) input keys non-Contig on the scanned axis —
        // base = row*k assumes a dense scanned axis, so reject.
        let a = OperandDesc::new(2, &[256, 128], &[1, 256], ElementKind::F32, 256);
        let o = OperandDesc::new(2, &[256, 128], &[128, 1], ElementKind::F32, 256);
        let key = structure_key(OpCategory::UnaryElementwise, &[a, o], ArchSku::Sm89);
        let sc = OpDef::scan_simple("cum", &[ElementKind::F32], ReduceOp::Sum, 1, false, false);
        let _ = build_plan(&sc, &key);
    }

    #[test]
    #[should_panic(expected = "must not be reversed")]
    fn flipped_operand_rejected() {
        // A dense-but-REVERSED input keys |stride|-Contig + flipped; idx = base+j
        // reads forward, so a flipped operand reads mirrored/OOB — reject (the
        // reverse SCAN is the `reverse` flag, never a flipped operand).
        let a = OperandDesc::new(2, &[256, 128], &[128, -1], ElementKind::F32, 256);
        let o = OperandDesc::new(2, &[256, 128], &[128, 1], ElementKind::F32, 256);
        let key = structure_key(OpCategory::UnaryElementwise, &[a, o], ArchSku::Sm89);
        let sc = OpDef::scan_simple("cum", &[ElementKind::F32], ReduceOp::Sum, 1, false, false);
        let _ = build_plan(&sc, &key);
    }

    #[test]
    #[should_panic(expected = "rejected under Scan")]
    fn hetero_out_dtype_rejected() {
        // A cumulative op is same-dtype as its input — a hetero out_dtype has no
        // exact store (assert_valid_out_dtype, runs before validate_scan).
        let mut sc = OpDef::scan_simple("cum", &[ElementKind::F32], ReduceOp::Sum, 1, false, false);
        sc.out_dtype = Some(ElementKind::U8);
        let _ = build_plan(&sc, &scan_key(ElementKind::F32));
    }

    // ---- pre/post expression-legality gate (the `check` closure). Reachable only
    // via the public OpDef::scan (arbitrary pre/post); scan_simple can't exercise
    // it. Review-caught: a mutation neutralizing BOTH `check` call sites passed the
    // whole suite — these four tests now kill that mutant. ----

    #[test]
    #[should_panic(expected = "must not read Reduced")]
    fn pre_map_reading_reduced_rejected() {
        // The pre-map runs BEFORE the fold — the running prefix does not exist yet,
        // so a `Reduced` read in `pre` would lower to an undefined register.
        let sc = OpDef::scan(
            "cum",
            1,
            &[ElementKind::F32],
            ReduceOp::Sum,
            1,
            false,
            false,
            reduced(0),
            reduced(0),
        );
        let _ = build_plan(&sc, &scan_key(ElementKind::F32));
    }

    #[test]
    #[should_panic(expected = "single Reduced(0) leaf")]
    fn post_reading_reduced_nonzero_rejected() {
        // The running prefix is the single Reduced(0) leaf; Reduced(1) has no source.
        let sc = OpDef::scan(
            "cum",
            1,
            &[ElementKind::F32],
            ReduceOp::Sum,
            1,
            false,
            false,
            input(0),
            reduced(1),
        );
        let _ = build_plan(&sc, &scan_key(ElementKind::F32));
    }

    #[test]
    #[should_panic(expected = ">= n_inputs")]
    fn pre_input_out_of_range_rejected() {
        // Input(5) with n_inputs = 1 — the kernel signature has no in5.
        let sc = OpDef::scan(
            "cum",
            1,
            &[ElementKind::F32],
            ReduceOp::Sum,
            1,
            false,
            false,
            input(5),
            reduced(0),
        );
        let _ = build_plan(&sc, &scan_key(ElementKind::F32));
    }

    #[test]
    #[should_panic(expected = "Const must be finite")]
    fn nonfinite_const_in_post_rejected() {
        // A non-finite Const in the epilogue (here NaN) has no valid emission.
        let sc = OpDef::scan(
            "cum",
            1,
            &[ElementKind::F32],
            ReduceOp::Sum,
            1,
            false,
            false,
            input(0),
            crate::ir::Expr(crate::ir::ScalarExpr::Add(
                Box::new(reduced(0).0),
                Box::new(konst(f64::NAN).0),
            )),
        );
        let _ = build_plan(&sc, &scan_key(ElementKind::F32));
    }

    // ---- operand-role / output-layout gates (review-caught: only the RowStreamed
    // contig + flip guards were tested). ----

    #[test]
    #[should_panic(expected = "full-width forward-dense")]
    fn flipped_output_rejected() {
        // A reversed OUTPUT keys |stride|-Contig + flipped; the scan store is
        // forward-dense (out[base+j]) — a flipped output would write mirrored.
        let a = OperandDesc::new(2, &[256, 128], &[128, 1], ElementKind::F32, 256);
        let o = OperandDesc::new(2, &[256, 128], &[128, -1], ElementKind::F32, 256);
        let key = structure_key(OpCategory::UnaryElementwise, &[a, o], ArchSku::Sm89);
        let sc = OpDef::scan_simple("cum", &[ElementKind::F32], ReduceOp::Sum, 1, false, false);
        let _ = build_plan(&sc, &key);
    }

    #[test]
    #[should_panic(expected = "Input0 must be the row-streamed")]
    fn input0_not_streamed_rejected() {
        // Input0 broadcast along the scanned (last) axis keys as a per-row scalar,
        // not the row-streamed scanned tensor — there is nothing to scan.
        let a = OperandDesc::new(2, &[256, 128], &[1, 0], ElementKind::F32, 256);
        let o = OperandDesc::new(2, &[256, 128], &[128, 1], ElementKind::F32, 256);
        let key = structure_key(OpCategory::UnaryElementwise, &[a, o], ArchSku::Sm89);
        let sc = OpDef::scan_simple("cum", &[ElementKind::F32], ReduceOp::Sum, 1, false, false);
        let _ = build_plan(&sc, &key);
    }
}

#[cfg(test)]
mod window_gate_validate {
    //! Increment-7 WINDOW gate-rejection tests. Per the house rule these call
    //! `build_plan` DIRECTLY — an emitter panic would mask a gate mutation (the 0c
    //! lesson). Every `validate_window` (and `assert_valid_out_dtype`) rejection
    //! has a test here; each window-specific gate is mutation-checked both
    //! directions by a targeted reverse-edit.
    use super::{Schedule, build_plan};
    use crate::ir::{OpDef, ReduceOp, input, konst, reduced};
    use unpopped_vocab::{
        ArchSku, ElementKind, OpCategory, OperandDesc, StructureKey, structure_key,
    };

    // A rank-2 window cell: contiguous input [256,128] + downsampled contiguous
    // output [256,64] (the extent is NOT keyed — only the layout class is — so any
    // Contig output stands in; the k_in→k_out arithmetic is a runtime precondition).
    fn window_key(dt: ElementKind) -> StructureKey {
        let a = OperandDesc::new(2, &[256, 128], &[128, 1], dt, 256);
        let o = OperandDesc::new(2, &[256, 64], &[64, 1], dt, 256);
        structure_key(OpCategory::UnaryElementwise, &[a, o], ArchSku::Sm89)
    }

    // A plain single-input pool: size/stride/dilation/pad_lo/pad_hi/cip.
    #[allow(
        clippy::too_many_arguments,
        reason = "one argument per declared window parameter"
    )]
    fn pool(
        op: ReduceOp,
        dt: ElementKind,
        axis: u8,
        size: u8,
        stride: u8,
        dilation: u8,
        pad_lo: u8,
        pad_hi: u8,
        cip: bool,
    ) -> OpDef {
        OpDef::window_simple(
            "pool",
            &[dt],
            op,
            axis,
            size,
            stride,
            dilation,
            pad_lo,
            pad_hi,
            cip,
        )
    }

    #[test]
    fn a_window_that_pools_a_row_to_one_element_builds() {
        // A global pool: the output row has extent 1 beside an input row of 128. The key's
        // mask is over the iteration frame, so that size-1 own axis reads as broadcast
        // though the output is a dense [256,1]; the gate must admit it.
        let a = OperandDesc::new(2, &[256, 128], &[128, 1], ElementKind::F32, 256);
        let o = OperandDesc::new(2, &[256, 1], &[1, 1], ElementKind::F32, 256);
        let key = structure_key(OpCategory::UnaryElementwise, &[a, o], ArchSku::Sm89);
        let p = pool(
            ReduceOp::Mean,
            ElementKind::F32,
            1,
            128,
            128,
            1,
            0,
            0,
            false,
        );
        let _ = build_plan(&p, &key);
    }

    #[test]
    fn valid_window_builds_all_combines_and_geometry() {
        // Max/Min/Sum/Mean × a spread of stride/dilation/pad build; the schedule is
        // the pooling Window on the innermost axis (rank-1 = 1).
        for op in [ReduceOp::Max, ReduceOp::Min, ReduceOp::Sum, ReduceOp::Mean] {
            for &(size, stride, dilation, pad) in &[
                (2u8, 2u8, 1u8, 0u8),
                (3, 1, 1, 1),
                (3, 2, 2, 2),
                (5, 3, 1, 2),
            ] {
                let p = pool(
                    op,
                    ElementKind::F32,
                    1,
                    size,
                    stride,
                    dilation,
                    pad,
                    pad,
                    false,
                );
                let key = window_key(ElementKind::F32);
                let plan = build_plan(&p, &key);
                assert_eq!(
                    plan.schedule,
                    Schedule::Window {
                        op,
                        axis: 1,
                        size,
                        stride,
                        dilation,
                        pad_lo: pad,
                        pad_hi: pad,
                        count_include_pad: false,
                    }
                );
            }
        }
    }

    #[test]
    fn integer_max_min_sum_pool_builds() {
        // Max/Min/Sum ride the integer base (select / wrapping sum); only Mean is
        // float-gated.
        for op in [ReduceOp::Max, ReduceOp::Min, ReduceOp::Sum] {
            let p = pool(op, ElementKind::I32, 1, 2, 2, 1, 0, 0, false);
            let _ = build_plan(&p, &window_key(ElementKind::I32));
        }
    }

    #[test]
    fn avg_pool_count_include_pad_flag_rides_schedule() {
        let p = pool(ReduceOp::Mean, ElementKind::F32, 1, 3, 1, 1, 1, 1, true);
        let key = window_key(ElementKind::F32);
        let plan = build_plan(&p, &key);
        assert!(matches!(
            plan.schedule,
            Schedule::Window {
                op: ReduceOp::Mean,
                count_include_pad: true,
                ..
            }
        ));
    }

    // ---- window-specific gates (each mutation-checked both directions) ----

    #[test]
    fn prod_combine_is_admitted_as_an_extension() {
        // 0.42.9: a windowed product plans (it is an Unpopped extension, not a KISS op)
        let p = pool(ReduceOp::Prod, ElementKind::F32, 1, 2, 2, 1, 0, 0, false);
        let _ = build_plan(&p, &window_key(ElementKind::F32));
    }

    #[test]
    #[should_panic(expected = "requires a float dtype")]
    fn mean_on_integer_rejected() {
        // avg_pool (Mean) at an integer dtype rounds — reject (Max/Min/Sum are OK).
        let p = pool(ReduceOp::Mean, ElementKind::I32, 1, 2, 2, 1, 0, 0, false);
        let _ = build_plan(&p, &window_key(ElementKind::I32));
    }

    #[test]
    fn a_non_inner_axis_plans_for_dense_inputs() {
        // 0.42.12: axis 0 on a rank-2 cell (a strided window over dense operands)
        let p = pool(ReduceOp::Max, ElementKind::F32, 0, 2, 2, 1, 0, 0, false);
        let key = window_key(ElementKind::F32);
        let plan = build_plan(&p, &key);
        assert!(matches!(plan.schedule, Schedule::Window { axis: 0, .. }));
    }

    #[test]
    #[should_panic(expected = "outside the rank")]
    fn a_window_axis_outside_the_rank_is_rejected() {
        let p = pool(ReduceOp::Max, ElementKind::F32, 2, 2, 2, 1, 0, 0, false);
        let _ = build_plan(&p, &window_key(ElementKind::F32));
    }

    #[test]
    #[should_panic(expected = "size must be >= 1")]
    fn zero_size_rejected() {
        let p = pool(ReduceOp::Max, ElementKind::F32, 1, 0, 2, 1, 0, 0, false);
        let _ = build_plan(&p, &window_key(ElementKind::F32));
    }

    #[test]
    #[should_panic(expected = "stride must be >= 1")]
    fn zero_stride_rejected() {
        let p = pool(ReduceOp::Max, ElementKind::F32, 1, 2, 0, 1, 0, 0, false);
        let _ = build_plan(&p, &window_key(ElementKind::F32));
    }

    #[test]
    #[should_panic(expected = "dilation must be >= 1")]
    fn zero_dilation_rejected() {
        let p = pool(ReduceOp::Max, ElementKind::F32, 1, 2, 2, 0, 0, 0, false);
        let _ = build_plan(&p, &window_key(ElementKind::F32));
    }

    #[test]
    #[should_panic(expected = "pad_lo")]
    fn pad_lo_over_half_span_rejected() {
        // span = dilation*(size-1)+1 = 1*(2-1)+1 = 2; pad_lo=2 ⇒ 2*2=4 > 2 → reject.
        let p = pool(ReduceOp::Max, ElementKind::F32, 1, 2, 2, 1, 2, 0, false);
        let _ = build_plan(&p, &window_key(ElementKind::F32));
    }

    #[test]
    #[should_panic(expected = "pad_hi")]
    fn pad_hi_over_half_span_rejected() {
        let p = pool(ReduceOp::Max, ElementKind::F32, 1, 2, 2, 1, 0, 2, false);
        let _ = build_plan(&p, &window_key(ElementKind::F32));
    }

    // ---- operand-role / output-layout / expr gates (mirror validate_scan) ----

    #[test]
    #[should_panic(expected = "must be contiguous")]
    fn non_contig_pooled_input_rejected() {
        let a = OperandDesc::new(2, &[256, 128], &[1, 256], ElementKind::F32, 256);
        let o = OperandDesc::new(2, &[256, 64], &[64, 1], ElementKind::F32, 256);
        let key = structure_key(OpCategory::UnaryElementwise, &[a, o], ArchSku::Sm89);
        let p = pool(ReduceOp::Max, ElementKind::F32, 1, 2, 2, 1, 0, 0, false);
        let _ = build_plan(&p, &key);
    }

    #[test]
    #[should_panic(expected = "must not be reversed")]
    fn flipped_input_rejected() {
        let a = OperandDesc::new(2, &[256, 128], &[128, -1], ElementKind::F32, 256);
        let o = OperandDesc::new(2, &[256, 64], &[64, 1], ElementKind::F32, 256);
        let key = structure_key(OpCategory::UnaryElementwise, &[a, o], ArchSku::Sm89);
        let p = pool(ReduceOp::Max, ElementKind::F32, 1, 2, 2, 1, 0, 0, false);
        let _ = build_plan(&p, &key);
    }

    #[test]
    #[should_panic(expected = "full-width forward-dense")]
    fn flipped_output_rejected() {
        let a = OperandDesc::new(2, &[256, 128], &[128, 1], ElementKind::F32, 256);
        let o = OperandDesc::new(2, &[256, 64], &[64, -1], ElementKind::F32, 256);
        let key = structure_key(OpCategory::UnaryElementwise, &[a, o], ArchSku::Sm89);
        let p = pool(ReduceOp::Max, ElementKind::F32, 1, 2, 2, 1, 0, 0, false);
        let _ = build_plan(&p, &key);
    }

    #[test]
    #[should_panic(expected = "Input0 must be the row-streamed")]
    fn input0_not_streamed_rejected() {
        // Input0 broadcast along the pooled (last) axis keys as a per-row scalar.
        let a = OperandDesc::new(2, &[256, 128], &[1, 0], ElementKind::F32, 256);
        let o = OperandDesc::new(2, &[256, 64], &[64, 1], ElementKind::F32, 256);
        let key = structure_key(OpCategory::UnaryElementwise, &[a, o], ArchSku::Sm89);
        let p = pool(ReduceOp::Max, ElementKind::F32, 1, 2, 2, 1, 0, 0, false);
        let _ = build_plan(&p, &key);
    }

    #[test]
    #[should_panic(expected = "rejected under Window")]
    fn hetero_out_dtype_rejected() {
        let mut p = pool(ReduceOp::Max, ElementKind::F32, 1, 2, 2, 1, 0, 0, false);
        p.out_dtype = Some(ElementKind::U8);
        let _ = build_plan(&p, &window_key(ElementKind::F32));
    }

    #[test]
    #[should_panic(expected = "must not read Reduced")]
    fn pre_map_reading_reduced_rejected() {
        let p = OpDef::window(
            "pool",
            1,
            &[ElementKind::F32],
            ReduceOp::Sum,
            1,
            2,
            2,
            1,
            0,
            0,
            false,
            reduced(0),
            reduced(0),
        );
        let _ = build_plan(&p, &window_key(ElementKind::F32));
    }

    #[test]
    #[should_panic(expected = "single Reduced(0) leaf")]
    fn post_reading_reduced_nonzero_rejected() {
        let p = OpDef::window(
            "pool",
            1,
            &[ElementKind::F32],
            ReduceOp::Sum,
            1,
            2,
            2,
            1,
            0,
            0,
            false,
            input(0),
            reduced(1),
        );
        let _ = build_plan(&p, &window_key(ElementKind::F32));
    }

    #[test]
    #[should_panic(expected = ">= n_inputs")]
    fn pre_input_out_of_range_rejected() {
        let p = OpDef::window(
            "pool",
            1,
            &[ElementKind::F32],
            ReduceOp::Sum,
            1,
            2,
            2,
            1,
            0,
            0,
            false,
            input(5),
            reduced(0),
        );
        let _ = build_plan(&p, &window_key(ElementKind::F32));
    }

    #[test]
    #[should_panic(expected = "Const must be finite")]
    fn nonfinite_const_in_post_rejected() {
        let p = OpDef::window(
            "pool",
            1,
            &[ElementKind::F32],
            ReduceOp::Sum,
            1,
            2,
            2,
            1,
            0,
            0,
            false,
            input(0),
            crate::ir::Expr(crate::ir::ScalarExpr::Add(
                Box::new(reduced(0).0),
                Box::new(konst(f64::NAN).0),
            )),
        );
        let _ = build_plan(&p, &window_key(ElementKind::F32));
    }

    // ---- review-caught coverage: pin the pad <= half-span EQUALITY boundary
    // from BOTH sides (a `<=`->`<` mutation previously survived — every positive
    // geometry had 2*pad strictly < span, yet size=2/pad=1 is the PyTorch-legal
    // kernel_size=2/padding=1 pool and MUST build). ----

    #[test]
    fn pad_equal_to_half_span_builds() {
        // span = dilation*(size-1)+1 = 2; 2*pad_lo = 2*pad_hi = 2 == span — the
        // boundary case is legal (each edge window still overlaps >= 1 tap).
        let p = pool(ReduceOp::Max, ElementKind::F32, 1, 2, 2, 1, 1, 1, false);
        let key = window_key(ElementKind::F32);
        let plan = build_plan(&p, &key);
        assert!(matches!(
            plan.schedule,
            Schedule::Window {
                pad_lo: 1,
                pad_hi: 1,
                ..
            }
        ));
    }

    #[test]
    #[should_panic(expected = "pad_lo 2 exceeds half the window span")]
    fn pad_lo_one_past_half_span_rejected() {
        // span = 2; 2*pad_lo = 4 > 2 — one past the boundary rejects.
        let p = pool(ReduceOp::Max, ElementKind::F32, 1, 2, 2, 1, 2, 0, false);
        let _ = build_plan(&p, &window_key(ElementKind::F32));
    }

    #[test]
    #[should_panic(expected = "pad_hi 2 exceeds half the window span")]
    fn pad_hi_one_past_half_span_rejected() {
        let p = pool(ReduceOp::Max, ElementKind::F32, 1, 2, 2, 1, 0, 2, false);
        let _ = build_plan(&p, &window_key(ElementKind::F32));
    }

    // ---- review-caught coverage: the ColBroadcast / RowScalar operand-role gates
    // were unreachable by any test (every pool was single-input). A 2-input window
    // (pre = input(0)*input(1), the per-column-weight shape validate_scan also
    // admits) reaches them for i >= 1. ----

    // A 2-input window cell: streamed input0 + a second operand with the given
    // key, then the downsampled output.
    fn two_input_pool(second: OperandDesc) -> (OpDef, StructureKey) {
        let a = OperandDesc::new(2, &[256, 128], &[128, 1], ElementKind::F32, 256);
        let o = OperandDesc::new(2, &[256, 64], &[64, 1], ElementKind::F32, 256);
        let key = structure_key(
            OpCategory::BinaryElementwise,
            &[a, second, o],
            ArchSku::Sm89,
        );
        let p = OpDef::window(
            "wpool",
            2,
            &[ElementKind::F32],
            ReduceOp::Sum,
            1,
            2,
            2,
            1,
            0,
            0,
            false,
            input(0) * input(1),
            reduced(0),
        );
        (p, key)
    }

    #[test]
    fn weighted_pool_with_column_weight_builds() {
        // input1 = a per-column weight broadcast over rows (stride 0 on the outer
        // axis, varying along the pooled axis) — the ColBroadcast happy path. Full
        // iteration shape with a 0 stride marks the broadcast (the scan-test
        // convention).
        let w = OperandDesc::new(2, &[256, 128], &[0, 1], ElementKind::F32, 256);
        let (p, key) = two_input_pool(w);
        let _ = build_plan(&p, &key);
    }

    #[test]
    #[should_panic(expected = "column input 1 must not be reversed")]
    fn flipped_column_weight_rejected() {
        // The same column weight REVERSED along the pooled axis: |stride|-varying +
        // flipped — the in_i[p] read would be mirrored.
        let w = OperandDesc::new(2, &[256, 128], &[0, -1], ElementKind::F32, 256);
        let (p, key) = two_input_pool(w);
        let _ = build_plan(&p, &key);
    }

    #[test]
    #[should_panic(expected = "row-scalar input 1 must not be reversed")]
    fn flipped_row_scalar_rejected() {
        // input1 = a per-row scalar (broadcast along the pooled axis) but REVERSED
        // along the outer axis — in_i[row] would read mirrored.
        let s = OperandDesc::new(2, &[256, 128], &[-1, 0], ElementKind::F32, 256);
        let (p, key) = two_input_pool(s);
        let _ = build_plan(&p, &key);
    }
}

#[cfg(test)]
mod im2col_gate_validate {
    //! Increment-11 IM2COL gate-rejection tests. Per the house rule these call
    //! `build_plan` DIRECTLY — an emitter panic would mask a gate mutation (the 0c
    //! lesson). Every `validate_im2col` (and `assert_valid_out_dtype`) rejection has
    //! a test here; each gate is mutation-checked both directions by a targeted
    //! reverse-edit.
    use super::{Schedule, build_plan};
    use crate::ir::{Access, OpDef};
    use unpopped_vocab::{
        ArchSku, ElementKind, OpCategory, OperandDesc, StructureKey, structure_key,
    };

    // A canonical im2col cell: rank-4 NCHW input [2,3,4,4] dense + rank-3 output
    // [2,8,8] dense. Extents are NOT keyed (only rank + layout), so any dense shapes
    // stand in; the (H,W)->(oH,oW) conv arithmetic is a runtime precondition.
    fn im2col_key(dt: ElementKind) -> StructureKey {
        let a = OperandDesc::new(4, &[2, 3, 4, 4], &[48, 16, 4, 1], dt, 256);
        let o = OperandDesc::new(3, &[2, 8, 8], &[64, 8, 1], dt, 256);
        structure_key(OpCategory::UnaryElementwise, &[a, o], ArchSku::Sm89)
    }

    fn im2col(kernel: (u8, u8), stride: (u8, u8), pad: (u8, u8), dilation: (u8, u8)) -> OpDef {
        OpDef::im2col_2d("unfold", ElementKind::F32, kernel, stride, pad, dilation)
    }

    // ---- G0: schedule threading ----

    #[test]
    fn im2col_cell_builds() {
        // A rank-4-in / rank-3-out Im2Col cell builds and threads the geometry into
        // Schedule::Im2Col (3x3, stride 1, pad 1, dilation 1 — the canonical conv).
        let p = im2col((3, 3), (1, 1), (1, 1), (1, 1));
        let key = im2col_key(ElementKind::F32);
        let plan = build_plan(&p, &key);
        assert_eq!(
            plan.schedule,
            Schedule::Im2Col {
                kernel: (3, 3),
                stride: (1, 1),
                pad: (1, 1),
                dilation: (1, 1),
            }
        );
        assert!(matches!(plan.access, Access::Im2Col { .. }));
    }

    #[test]
    fn im2col_dilated_cell_builds() {
        // A dilated, non-square (kh != kw), strided cell builds — the geometry rides
        // the schedule verbatim.
        let p = im2col((3, 5), (2, 1), (2, 0), (2, 3));
        let key = im2col_key(ElementKind::F32);
        let plan = build_plan(&p, &key);
        assert_eq!(
            plan.schedule,
            Schedule::Im2Col {
                kernel: (3, 5),
                stride: (2, 1),
                pad: (2, 0),
                dilation: (2, 3),
            }
        );
    }

    #[test]
    fn im2col_builds_when_an_own_axis_has_extent_one() {
        // The key's broadcast mask is over the iteration frame, and a size-1 own axis
        // counts as stride 0 wherever the frame is wider. Batch 1 beside two channels
        // (frame axis 1 is max(N, C) = 2) and a 1x1 output beside a wider input (frame
        // axis 3 is max(W, oH*oW)) are ordinary shapes, not broadcasts: the output is
        // still dense, so the gate must admit them.
        let p = im2col((3, 3), (1, 1), (0, 0), (1, 1));
        let a = OperandDesc::new(4, &[1, 2, 3, 3], &[18, 9, 3, 1], ElementKind::F32, 256);
        let o = OperandDesc::new(3, &[1, 18, 1], &[18, 1, 1], ElementKind::F32, 256);
        let key = structure_key(OpCategory::UnaryElementwise, &[a, o], ArchSku::Sm89);
        let plan = build_plan(&p, &key);
        assert!(matches!(plan.access, Access::Im2Col { .. }));

        let a = OperandDesc::new(4, &[1, 3, 4, 4], &[48, 16, 4, 1], ElementKind::F32, 256);
        let o = OperandDesc::new(3, &[1, 27, 4], &[108, 4, 1], ElementKind::F32, 256);
        let key = structure_key(OpCategory::UnaryElementwise, &[a, o], ArchSku::Sm89);
        let plan = build_plan(&p, &key);
        assert!(matches!(plan.access, Access::Im2Col { .. }));
    }

    #[test]
    fn im2col_builds_a_single_channel_batch() {
        // N = 2, C = 1: the input's channel axis has extent 1 beside a frame axis of 2
        // (the output's batch axis lands there), so its mask bit is set though no axis is
        // broadcast. Single-channel batches (MNIST-shaped) are ordinary.
        let p = im2col((3, 3), (1, 1), (0, 0), (1, 1));
        let a = OperandDesc::new(4, &[2, 1, 4, 4], &[16, 16, 4, 1], ElementKind::F32, 256);
        let o = OperandDesc::new(3, &[2, 9, 4], &[36, 4, 1], ElementKind::F32, 256);
        let key = structure_key(OpCategory::UnaryElementwise, &[a, o], ArchSku::Sm89);
        let plan = build_plan(&p, &key);
        assert!(matches!(plan.access, Access::Im2Col { .. }));
    }

    #[test]
    fn im2col_builds_every_dtype() {
        // A pure raw-bit gather is dtype-agnostic — f32/f64/f16/bf16/i32/i64 all build.
        for dt in [
            ElementKind::F32,
            ElementKind::F64,
            ElementKind::F16,
            ElementKind::Bf16,
            ElementKind::I32,
            ElementKind::I64,
        ] {
            let p = OpDef::im2col_2d("unfold", dt, (3, 3), (1, 1), (1, 1), (1, 1));
            let _ = build_plan(&p, &im2col_key(dt));
        }
    }

    // ---- G1: operand count + single input ----

    #[test]
    #[should_panic(expected = "expects n_inputs+1 = 2 operands")]
    fn im2col_key_needs_two_operands() {
        // A 3-operand key (input + two outputs) is not an im2col shape — reject.
        let a = OperandDesc::new(4, &[2, 3, 4, 4], &[48, 16, 4, 1], ElementKind::F32, 256);
        let o = OperandDesc::new(3, &[2, 8, 8], &[64, 8, 1], ElementKind::F32, 256);
        let key = structure_key(OpCategory::UnaryElementwise, &[a, o, o], ArchSku::Sm89);
        let p = im2col((3, 3), (1, 1), (1, 1), (1, 1));
        let _ = build_plan(&p, &key);
    }

    #[test]
    #[should_panic(expected = "takes a SINGLE input")]
    fn im2col_rejects_second_input() {
        // A weight operand (n_inputs == 2) is the deferred windowed-contraction, not
        // an im2col — reject.
        let mut p = im2col((3, 3), (1, 1), (1, 1), (1, 1));
        p.n_inputs = 2;
        let _ = build_plan(&p, &im2col_key(ElementKind::F32));
    }

    // ---- G2: input rank / layout (FIXED 4 -> 3) ----

    #[test]
    #[should_panic(expected = "pins a rank-4 NCHW input")]
    fn im2col_rejects_rank3_input() {
        // A rank-3 input drops key.rank below 4 (the input is the widest operand).
        let a = OperandDesc::new(3, &[2, 3, 16], &[48, 16, 1], ElementKind::F32, 256);
        let o = OperandDesc::new(3, &[2, 8, 8], &[64, 8, 1], ElementKind::F32, 256);
        let key = structure_key(OpCategory::UnaryElementwise, &[a, o], ArchSku::Sm89);
        let p = im2col((3, 3), (1, 1), (1, 1), (1, 1));
        let _ = build_plan(&p, &key);
    }

    #[test]
    #[should_panic(expected = "input 0 must be dense contiguous NCHW")]
    fn im2col_rejects_a_broadcast_input() {
        // A stride-0 axis of extent > 1 reads `Broadcast`, not `Contig`: the dense-input
        // clause alone rejects it (the key's frame mask is not consulted).
        let a = OperandDesc::new(4, &[2, 3, 4, 4], &[0, 16, 4, 1], ElementKind::F32, 256);
        let o = OperandDesc::new(3, &[2, 8, 8], &[64, 8, 1], ElementKind::F32, 256);
        let key = structure_key(OpCategory::UnaryElementwise, &[a, o], ArchSku::Sm89);
        let p = im2col((3, 3), (1, 1), (1, 1), (1, 1));
        let _ = build_plan(&p, &key);
    }

    #[test]
    #[should_panic(expected = "input 0 must be dense contiguous NCHW")]
    fn im2col_rejects_noncontig_input() {
        // Last-two-axes-swapped strides ([..,1,4] instead of [..,4,1]) — non-contig.
        let a = OperandDesc::new(4, &[2, 3, 4, 4], &[48, 16, 1, 4], ElementKind::F32, 256);
        let o = OperandDesc::new(3, &[2, 8, 8], &[64, 8, 1], ElementKind::F32, 256);
        let key = structure_key(OpCategory::UnaryElementwise, &[a, o], ArchSku::Sm89);
        let p = im2col((3, 3), (1, 1), (1, 1), (1, 1));
        let _ = build_plan(&p, &key);
    }

    #[test]
    #[should_panic(expected = "input 0 must not be reversed")]
    fn im2col_rejects_flipped_input() {
        // A dense-magnitude but inner-axis-reversed input (stride -1) keys Contig but
        // flipped — the source read would be mirrored.
        let a = OperandDesc::new(4, &[2, 3, 4, 4], &[48, 16, 4, -1], ElementKind::F32, 256);
        let o = OperandDesc::new(3, &[2, 8, 8], &[64, 8, 1], ElementKind::F32, 256);
        let key = structure_key(OpCategory::UnaryElementwise, &[a, o], ArchSku::Sm89);
        let p = im2col((3, 3), (1, 1), (1, 1), (1, 1));
        let _ = build_plan(&p, &key);
    }

    // ---- G3: output rank / layout ----

    #[test]
    #[should_panic(expected = "output must be forward-dense contiguous")]
    fn im2col_rejects_broadcast_output() {
        let a = OperandDesc::new(4, &[2, 3, 4, 4], &[48, 16, 4, 1], ElementKind::F32, 256);
        let o = OperandDesc::new(3, &[2, 8, 8], &[0, 8, 1], ElementKind::F32, 256);
        let key = structure_key(OpCategory::UnaryElementwise, &[a, o], ArchSku::Sm89);
        let p = im2col((3, 3), (1, 1), (1, 1), (1, 1));
        let _ = build_plan(&p, &key);
    }

    #[test]
    #[should_panic(expected = "output must be forward-dense contiguous")]
    fn im2col_rejects_flipped_output() {
        let a = OperandDesc::new(4, &[2, 3, 4, 4], &[48, 16, 4, 1], ElementKind::F32, 256);
        let o = OperandDesc::new(3, &[2, 8, 8], &[64, 8, -1], ElementKind::F32, 256);
        let key = structure_key(OpCategory::UnaryElementwise, &[a, o], ArchSku::Sm89);
        let p = im2col((3, 3), (1, 1), (1, 1), (1, 1));
        let _ = build_plan(&p, &key);
    }

    #[test]
    #[should_panic(expected = "output must be forward-dense contiguous")]
    fn im2col_rejects_strided_output() {
        // Last-two-axes-swapped output strides ([64,1,8]): the inner axis stride is
        // 8 != 1, so classify_contiguity keys it Strided/InnerContig (not Contig) —
        // but with NO broadcast (no 0 stride) and NO flip (all positive). This
        // ISOLATES the G3 `contig == Contig` clause: the broadcast test trips
        // `bcast.is_empty()` and the flipped test trips `!flipped`, so only a pure
        // strided output pins the middle clause (a strided output would gather into
        // the wrong slabs — the emitter assumes a dense row-major write).
        let a = OperandDesc::new(4, &[2, 3, 4, 4], &[48, 16, 4, 1], ElementKind::F32, 256);
        let o = OperandDesc::new(3, &[2, 8, 8], &[64, 1, 8], ElementKind::F32, 256);
        let key = structure_key(OpCategory::UnaryElementwise, &[a, o], ArchSku::Sm89);
        let p = im2col((3, 3), (1, 1), (1, 1), (1, 1));
        let _ = build_plan(&p, &key);
    }

    // ---- G4: geometry legality ----

    #[test]
    #[should_panic(expected = "kernel (kh,kw) = (0,3) must be >= 1")]
    fn im2col_rejects_zero_kernel() {
        let p = im2col((0, 3), (1, 1), (1, 1), (1, 1));
        let _ = build_plan(&p, &im2col_key(ElementKind::F32));
    }

    #[test]
    #[should_panic(expected = "stride (sh,sw) = (1,0) must be >= 1")]
    fn im2col_rejects_zero_stride() {
        let p = im2col((3, 3), (1, 0), (1, 1), (1, 1));
        let _ = build_plan(&p, &im2col_key(ElementKind::F32));
    }

    #[test]
    #[should_panic(expected = "dilation (dh,dw) = (0,1) must be >= 1")]
    fn im2col_rejects_zero_dilation() {
        let p = im2col((3, 3), (1, 1), (1, 1), (0, 1));
        let _ = build_plan(&p, &im2col_key(ElementKind::F32));
    }

    // ---- G5: out_dtype None + body == Input(0) ----

    #[test]
    #[should_panic(expected = "rejected under Im2Col")]
    fn im2col_rejects_out_dtype_some() {
        // A pure gather preserves dtype — a hetero out_dtype has no exact store
        // (double-gated: assert_valid_out_dtype fires first, then validate_im2col).
        let mut p = im2col((3, 3), (1, 1), (1, 1), (1, 1));
        p.out_dtype = Some(ElementKind::I32);
        let _ = build_plan(&p, &im2col_key(ElementKind::F32));
    }

    #[test]
    #[should_panic(expected = "body must be exactly Input(0)")]
    fn im2col_rejects_composed_body() {
        // v1 has no pre/post — the body must be exactly Input(0); a composed body
        // (Input(0)+Input(0)) has arithmetic the raw-bit copy does not emit.
        use crate::ir::input;
        let mut p = im2col((3, 3), (1, 1), (1, 1), (1, 1));
        p.body = (input(0) + input(0)).0;
        let _ = build_plan(&p, &im2col_key(ElementKind::F32));
    }
}

#[cfg(test)]
mod sort_gate_validate {
    //! Increment-8 SORT_PERM gate-rejection tests. Per the house rule these call
    //! `build_plan` DIRECTLY — an emitter panic would mask a gate mutation (the 0c
    //! lesson). Every `validate_row_sort` (and the argsort `assert_valid_out_dtype`
    //! coupling + `assert_valid_multi_output`) rejection has a test here; each
    //! sort-specific gate is mutation-checked both directions by a targeted
    //! reverse-edit.
    use super::{Schedule, access_tag, build_plan};
    use crate::ir::{Access, OpDef, ScalarExpr, SortLimit, SortOrder, SortOut, input};
    use unpopped_vocab::{
        ArchSku, ElementKind, OpCategory, OperandDesc, StructureKey, structure_key,
    };

    // A rank-2 [256,128] sort cell: contiguous input + full-width contiguous output.
    fn sort_key(dt: ElementKind) -> StructureKey {
        let a = OperandDesc::new(2, &[256, 128], &[128, 1], dt, 256);
        let o = OperandDesc::new(2, &[256, 128], &[128, 1], dt, 256);
        structure_key(OpCategory::UnaryElementwise, &[a, o], ArchSku::Sm89)
    }
    // The same cell with an I32 index output (argsort).
    fn argsort_key(dt: ElementKind) -> StructureKey {
        let a = OperandDesc::new(2, &[256, 128], &[128, 1], dt, 256);
        let o = OperandDesc::new(2, &[256, 128], &[128, 1], ElementKind::I32, 256);
        structure_key(OpCategory::UnaryElementwise, &[a, o], ArchSku::Sm89)
    }
    // Increment 9 FUSED_ARGSORT: the three-operand `Both` key — input, values
    // output (same dtype), I32 index output. `oi` overrides the index operand so
    // the layout-rejection tests can hand a broadcast/flipped out_idx.
    fn both_key_with(dt: ElementKind, oi: OperandDesc) -> StructureKey {
        let a = OperandDesc::new(2, &[256, 128], &[128, 1], dt, 256);
        let ov = OperandDesc::new(2, &[256, 128], &[128, 1], dt, 256);
        structure_key(OpCategory::UnaryElementwise, &[a, ov, oi], ArchSku::Sm89)
    }
    fn both_key(dt: ElementKind) -> StructureKey {
        let oi = OperandDesc::new(2, &[256, 128], &[128, 1], ElementKind::I32, 256);
        both_key_with(dt, oi)
    }

    // ---- positive gate tests ----

    #[test]
    fn valid_sort_builds_asc_and_desc_values() {
        for order in [SortOrder::Asc, SortOrder::Desc] {
            let sc = OpDef::row_sort("sort", ElementKind::F32, order);
            let key = sort_key(ElementKind::F32);
            let plan = build_plan(&sc, &key);
            assert_eq!(
                plan.schedule,
                Schedule::RowSort {
                    order,
                    stable: true,
                    out: SortOut::Values,
                    limit: SortLimit::Full
                }
            );
            assert_eq!(access_tag(&sc.access), "RowSort");
        }
    }

    #[test]
    fn valid_argsort_builds_desc_with_i32_out() {
        let sc = OpDef::row_argsort("argsort", ElementKind::F32, SortOrder::Desc);
        assert_eq!(sc.out_dtype, Some(ElementKind::I32));
        let key = argsort_key(ElementKind::F32);
        let plan = build_plan(&sc, &key);
        assert_eq!(
            plan.schedule,
            Schedule::RowSort {
                order: SortOrder::Desc,
                stable: true,
                out: SortOut::Indices,
                limit: SortLimit::Full
            }
        );
    }

    #[test]
    fn i64_and_f64_sort_build() {
        for dt in [ElementKind::I64, ElementKind::F64] {
            let sc = OpDef::row_sort("sort", dt, SortOrder::Asc);
            let _ = build_plan(&sc, &sort_key(dt));
        }
    }

    // ---- validate_row_sort rejections ----

    #[test]
    #[should_panic(expected = "unstable declined")]
    fn unstable_rejected() {
        // v1 emits only the stable pair-sort; stable=false is dead keying.
        let mut sc = OpDef::row_sort("sort", ElementKind::F32, SortOrder::Asc);
        sc.access = Access::RowSort {
            order: SortOrder::Asc,
            stable: false,
            out: SortOut::Values,
            limit: SortLimit::Full,
        };
        let _ = build_plan(&sc, &sort_key(ElementKind::F32));
    }

    #[test]
    #[should_panic(expected = "out of the v1 set")]
    fn s8_dtype_rejected() {
        // S8 is a v1 de-scope (the bespoke argsort covers small ints).
        let sc = OpDef::row_sort("sort", ElementKind::I8, SortOrder::Asc);
        let _ = build_plan(&sc, &sort_key(ElementKind::I8));
    }

    #[test]
    #[should_panic(expected = "out of the v1 set")]
    fn u32_dtype_rejected() {
        // U32 is index-only (no value dtype).
        let sc = OpDef::row_sort("sort", ElementKind::U32, SortOrder::Asc);
        let _ = build_plan(&sc, &sort_key(ElementKind::U32));
    }

    #[test]
    #[should_panic(expected = "must be exactly Input(0)")]
    fn composed_body_rejected() {
        // v1 has no pre/post — the body must be exactly Input(0); a composed body
        // (here Input(0)+Input(0)) rejects (replaces the recursive check machinery).
        let mut sc = OpDef::row_sort("sort", ElementKind::F32, SortOrder::Asc);
        sc.body = (input(0) + input(0)).0;
        let _ = build_plan(&sc, &sort_key(ElementKind::F32));
    }

    #[test]
    #[should_panic(expected = "must be exactly Input(0)")]
    fn param_body_rejected() {
        let mut sc = OpDef::row_sort("sort", ElementKind::F32, SortOrder::Asc);
        sc.body = ScalarExpr::Param(0);
        let _ = build_plan(&sc, &sort_key(ElementKind::F32));
    }

    #[test]
    #[should_panic(expected = "streams exactly one row operand")]
    fn multiple_inputs_rejected() {
        // A sort streams exactly one row operand.
        let mut sc = OpDef::row_sort("sort", ElementKind::F32, SortOrder::Asc);
        sc.n_inputs = 2;
        let a = OperandDesc::new(2, &[256, 128], &[128, 1], ElementKind::F32, 256);
        let o = OperandDesc::new(2, &[256, 128], &[128, 1], ElementKind::F32, 256);
        let key = structure_key(OpCategory::BinaryElementwise, &[a, a, o], ArchSku::Sm89);
        let _ = build_plan(&sc, &key);
    }

    #[test]
    #[should_panic(expected = "must be contiguous")]
    fn non_contig_input_rejected() {
        // A transposed (column-major) input keys non-Contig on the sorted axis.
        let a = OperandDesc::new(2, &[256, 128], &[1, 256], ElementKind::F32, 256);
        let o = OperandDesc::new(2, &[256, 128], &[128, 1], ElementKind::F32, 256);
        let key = structure_key(OpCategory::UnaryElementwise, &[a, o], ArchSku::Sm89);
        let sc = OpDef::row_sort("sort", ElementKind::F32, SortOrder::Asc);
        let _ = build_plan(&sc, &key);
    }

    #[test]
    #[should_panic(expected = "must not be reversed")]
    fn flipped_input_rejected() {
        // A dense-but-REVERSED input keys |stride|-Contig + flipped; idx = base+j
        // reads forward, so a flipped operand reads mirrored/OOB (use Desc order).
        let a = OperandDesc::new(2, &[256, 128], &[128, -1], ElementKind::F32, 256);
        let o = OperandDesc::new(2, &[256, 128], &[128, 1], ElementKind::F32, 256);
        let key = structure_key(OpCategory::UnaryElementwise, &[a, o], ArchSku::Sm89);
        let sc = OpDef::row_sort("sort", ElementKind::F32, SortOrder::Asc);
        let _ = build_plan(&sc, &key);
    }

    #[test]
    #[should_panic(expected = "Input0 must be the row-streamed")]
    fn broadcast_input_rejected() {
        // Input0 broadcast along the sorted (last) axis keys as a per-row scalar,
        // not the row-streamed sorted tensor — there is nothing to sort.
        let a = OperandDesc::new(2, &[256, 128], &[1, 0], ElementKind::F32, 256);
        let o = OperandDesc::new(2, &[256, 128], &[128, 1], ElementKind::F32, 256);
        let key = structure_key(OpCategory::UnaryElementwise, &[a, o], ArchSku::Sm89);
        let sc = OpDef::row_sort("sort", ElementKind::F32, SortOrder::Asc);
        let _ = build_plan(&sc, &key);
    }

    #[test]
    #[should_panic(expected = "forward-dense contiguous")]
    fn flipped_output_rejected() {
        // A reversed OUTPUT keys |stride|-Contig + flipped; the store is
        // forward-dense (out[base+r]) — a flipped output would write mirrored.
        let a = OperandDesc::new(2, &[256, 128], &[128, 1], ElementKind::F32, 256);
        let o = OperandDesc::new(2, &[256, 128], &[128, -1], ElementKind::F32, 256);
        let key = structure_key(OpCategory::UnaryElementwise, &[a, o], ArchSku::Sm89);
        let sc = OpDef::row_sort("sort", ElementKind::F32, SortOrder::Asc);
        let _ = build_plan(&sc, &key);
    }

    // ---- out_dtype ↔ argsort coupling ----

    #[test]
    #[should_panic(expected = "must be Some(I32)")]
    fn argsort_with_none_out_dtype_rejected() {
        // argsort=true but out_dtype None: assert_valid_out_dtype returns early on
        // None, so validate_row_sort's coupling is the layer that fires.
        let mut sc = OpDef::row_argsort("argsort", ElementKind::F32, SortOrder::Asc);
        sc.out_dtype = None;
        let _ = build_plan(&sc, &argsort_key(ElementKind::F32));
    }

    #[test]
    #[should_panic(expected = "admitted only as the argsort index output")]
    fn argsort_with_i64_out_dtype_rejected() {
        // argsort with a non-I32 index dtype: I64 indices are a v1 de-scope
        // (assert_valid_out_dtype fires first).
        let mut sc = OpDef::row_argsort("argsort", ElementKind::F32, SortOrder::Asc);
        sc.out_dtype = Some(ElementKind::I64);
        let _ = build_plan(&sc, &argsort_key(ElementKind::F32));
    }

    #[test]
    #[should_panic(expected = "admitted only as the argsort index output")]
    fn values_sort_with_i32_out_dtype_rejected() {
        // A values-sort is dtype-preserving — a Some(I32) out_dtype has no exact
        // store (assert_valid_out_dtype fires on the values-sort branch).
        let mut sc = OpDef::row_sort("sort", ElementKind::F32, SortOrder::Asc);
        sc.out_dtype = Some(ElementKind::I32);
        let _ = build_plan(&sc, &sort_key(ElementKind::F32));
    }

    #[test]
    #[should_panic(expected = "Elementwise-only")]
    fn extra_out_bodies_rejected() {
        // Multi-output does not ride RowSort (assert_valid_multi_output fires first;
        // validate_row_sort backstops it).
        let mut sc = OpDef::row_sort("sort", ElementKind::F32, SortOrder::Asc);
        sc.extra_out_bodies = vec![input(0).0];
        let _ = build_plan(&sc, &sort_key(ElementKind::F32));
    }

    // ---- review-caught coverage (both mutations EMPIRICALLY survived the suite):
    // the output-layout compound assert had only its `flipped` term tested, and the
    // n_operands == n_inputs+1 assert was deletable. Pin every term. ----

    #[test]
    #[should_panic(expected = "forward-dense contiguous")]
    fn non_contig_output_rejected() {
        // A transposed (column-major) output keys non-Contig — the sorted writeback
        // out[base + r] assumes a dense inner axis.
        let a = OperandDesc::new(2, &[256, 128], &[128, 1], ElementKind::F32, 256);
        let o = OperandDesc::new(2, &[256, 128], &[1, 256], ElementKind::F32, 256);
        let key = structure_key(OpCategory::UnaryElementwise, &[a, o], ArchSku::Sm89);
        let sc = OpDef::row_sort("sort", ElementKind::F32, SortOrder::Asc);
        let _ = build_plan(&sc, &key);
    }

    #[test]
    #[should_panic(expected = "forward-dense contiguous")]
    fn broadcast_output_rejected() {
        // A stride-0 (broadcast) output axis is not a full-width store target.
        let a = OperandDesc::new(2, &[256, 128], &[128, 1], ElementKind::F32, 256);
        let o = OperandDesc::new(2, &[256, 128], &[0, 1], ElementKind::F32, 256);
        let key = structure_key(OpCategory::UnaryElementwise, &[a, o], ArchSku::Sm89);
        let sc = OpDef::row_sort("sort", ElementKind::F32, SortOrder::Asc);
        let _ = build_plan(&sc, &key);
    }

    #[test]
    #[should_panic(expected = "expects n_inputs+1 operands")]
    fn wrong_operand_count_rejected() {
        // n_inputs=1 against a 3-operand key — the signature and the accept
        // predicate would describe different arities.
        let a = OperandDesc::new(2, &[256, 128], &[128, 1], ElementKind::F32, 256);
        let key = structure_key(OpCategory::BinaryElementwise, &[a, a, a], ArchSku::Sm89);
        let sc = OpDef::row_sort("sort", ElementKind::F32, SortOrder::Asc);
        let _ = build_plan(&sc, &key);
    }

    // ---- Increment 9 FUSED_ARGSORT: the `Both` two-output gates (G1-G3, G5) ----

    #[test]
    fn valid_both_builds_asc_and_desc() {
        // Positive: a fused two-output sort builds with the 3-operand key and the
        // Both schedule; out_dtype stays None (values output 0 is dtype-preserving,
        // the I32 out_idx is emitter-hardwired). Also the M2 kill (G2 relaxed to
        // n+1 for Both would reject this valid 3-operand build).
        for order in [SortOrder::Asc, SortOrder::Desc] {
            let sc = OpDef::row_sort_indices("sort_both", ElementKind::F32, order);
            assert_eq!(sc.out_dtype, None);
            let key = both_key(ElementKind::F32);
            let plan = build_plan(&sc, &key);
            assert_eq!(
                plan.schedule,
                Schedule::RowSort {
                    order,
                    stable: true,
                    out: SortOut::Both,
                    limit: SortLimit::Full
                }
            );
            // n_outputs() stays body-derived = 1 for Both (the corollary decision);
            // the second buffer is owned by the SortOut state + the 3-operand key.
            assert_eq!(sc.n_outputs(), 1);
        }
    }

    #[test]
    fn both_builds_for_all_v1_dtypes() {
        for dt in [
            ElementKind::F32,
            ElementKind::F32Strict,
            ElementKind::F64,
            ElementKind::F16,
            ElementKind::Bf16,
            ElementKind::I32,
            ElementKind::I64,
        ] {
            let sc = OpDef::row_sort_indices("sort_both", dt, SortOrder::Asc);
            let _ = build_plan(&sc, &both_key(dt));
        }
    }

    #[test]
    fn values_key_still_two_operands() {
        // Regression: a Values sort still requires exactly n_inputs+1 operands (the
        // n_outputs computation must stay 1 for non-Both). Positive build with the
        // 2-operand key; a 3-operand Values key is rejected by
        // `wrong_operand_count_rejected` above.
        let sc = OpDef::row_sort("sort", ElementKind::F32, SortOrder::Asc);
        let key = sort_key(ElementKind::F32);
        let plan = build_plan(&sc, &key);
        assert_eq!(
            plan.schedule,
            Schedule::RowSort {
                order: SortOrder::Asc,
                stable: true,
                out: SortOut::Values,
                limit: SortLimit::Full
            }
        );
    }

    // G1 — out_dtype ↔ state coupling. The `Both ⇒ None` arm lives in
    // `validate_row_sort`, but `assert_valid_out_dtype` (G5) rejects any `Some(_)`
    // for Both FIRST in `build_plan` (it runs before the schedule match). So this
    // test calls `validate_row_sort` DIRECTLY to isolate G1 (the mutation M1
    // target) — the build_plan-direct house rule guards against EMITTER masking,
    // which does not apply to a plan-gate-vs-plan-gate shadow; the G5 front gate is
    // pinned separately by `both_with_some_out_dtype_rejected_at_out_dtype_gate`.
    #[test]
    #[should_panic(expected = "dtype-preserving on output 0")]
    fn both_requires_out_dtype_none() {
        let mut sc = OpDef::row_sort_indices("sort_both", ElementKind::F32, SortOrder::Asc);
        sc.out_dtype = Some(ElementKind::I32);
        super::validate_row_sort(
            &sc,
            &both_key(ElementKind::F32),
            SortOrder::Asc,
            true,
            SortOut::Both,
            SortLimit::Full,
        );
    }

    // G5 — the `assert_valid_out_dtype` RowSort arm rejects a Both carrying any
    // Some(_) out_dtype (build_plan-direct: this is the reachable front gate).
    #[test]
    #[should_panic(expected = "admitted only as the argsort index output")]
    fn both_with_some_out_dtype_rejected_at_out_dtype_gate() {
        let mut sc = OpDef::row_sort_indices("sort_both", ElementKind::F32, SortOrder::Asc);
        sc.out_dtype = Some(ElementKind::I32);
        let _ = build_plan(&sc, &both_key(ElementKind::F32));
    }

    // G2 — Both needs a 3-operand key ([in0, out_val, out_idx]).
    #[test]
    #[should_panic(expected = "expects n_inputs+2 operands")]
    fn both_key_needs_three_operands() {
        // A 2-operand key against a Both op — the emitter's two-pointer signature
        // and the accept predicate would describe different arities.
        let sc = OpDef::row_sort_indices("sort_both", ElementKind::F32, SortOrder::Asc);
        let _ = build_plan(&sc, &sort_key(ElementKind::F32)); // 2-operand key
    }

    // G3 — the out_idx (operand 2) layout: full-width forward-dense contiguous.
    #[test]
    #[should_panic(expected = "out_idx (operand 2) must be forward-dense")]
    fn both_out_idx_broadcast_rejected() {
        // A stride-0 (broadcast) out_idx inner axis is not a full-width store target.
        let oi = OperandDesc::new(2, &[256, 128], &[0, 1], ElementKind::I32, 256);
        let key = both_key_with(ElementKind::F32, oi);
        let sc = OpDef::row_sort_indices("sort_both", ElementKind::F32, SortOrder::Asc);
        let _ = build_plan(&sc, &key);
    }

    #[test]
    #[should_panic(expected = "out_idx (operand 2) must be forward-dense")]
    fn both_out_idx_flipped_rejected() {
        // A reversed out_idx keys |stride|-Contig + flipped; out_idx[base+r] writes
        // forward-dense, so a flipped index output would write mirrored.
        let oi = OperandDesc::new(2, &[256, 128], &[128, -1], ElementKind::I32, 256);
        let key = both_key_with(ElementKind::F32, oi);
        let sc = OpDef::row_sort_indices("sort_both", ElementKind::F32, SortOrder::Asc);
        let _ = build_plan(&sc, &key);
    }

    // G4 — extra_out_bodies empty (a permutation is not a body). `build_plan`
    // rejects a Both+body at `assert_valid_multi_output` (Elementwise-only) FIRST,
    // so this calls `validate_row_sort` DIRECTLY to isolate G4 (the M4 target); the
    // front gate is the same one the existing `extra_out_bodies_rejected` pins.
    #[test]
    #[should_panic(expected = "extra_out_bodies must be empty")]
    fn both_rejects_extra_out_bodies() {
        let mut sc = OpDef::row_sort_indices("sort_both", ElementKind::F32, SortOrder::Asc);
        sc.extra_out_bodies = vec![input(0).0];
        super::validate_row_sort(
            &sc,
            &both_key(ElementKind::F32),
            SortOrder::Asc,
            true,
            SortOut::Both,
            SortLimit::Full,
        );
    }

    // ---- Increment 10 TOPK/BOTTOMK: the runtime-k cap + shrunk out-extent ----

    // The `Both` key whose OUTPUTS are NARROWER than the input (`k_out < k_in`) —
    // [256,128] input, [256,k_out] value + I32 index outputs. `validate_row_sort`'s
    // output gate is LAYOUT-only (it does NOT assert the width), so this narrower
    // output passes as written — no relaxation. The `k_out <= k_in` relationship is a
    // runtime launch precondition (the key carries no numeric extent).
    fn topk_key(dt: ElementKind, k_out: i64) -> StructureKey {
        let a = OperandDesc::new(2, &[256, 128], &[128, 1], dt, 256);
        let ov = OperandDesc::new(2, &[256, k_out], &[k_out, 1], dt, 256);
        let oi = OperandDesc::new(2, &[256, k_out], &[k_out, 1], ElementKind::I32, 256);
        structure_key(OpCategory::UnaryElementwise, &[a, ov, oi], ArchSku::Sm89)
    }

    // G0 — `limit` threading: a TopK cell with a narrower `[batch,k_out]` out
    // operand BUILDS and schedules `limit: TopK` with the direction `order` picks
    // (topk = Desc, bottomk = Asc). This is the "gate" for the runtime-k/shrunk
    // extent — there is NO plan-time `k_out <= k_in` assert (none is expressible;
    // the key carries no numeric extents); the shrink is admitted because the
    // layout gate is width-agnostic.
    #[test]
    fn a_top_one_cell_builds() {
        // k = 1 (an arg-max): both outputs have extent 1 beside an input row of 128. The
        // key's frame mask reads that size-1 own axis as broadcast; the outputs are dense.
        let key = topk_key(ElementKind::F32, 1);
        let _ = build_plan(&OpDef::row_topk("topk", ElementKind::F32), &key);
    }

    #[test]
    fn topk_both_cell_builds() {
        for (sc, want_order) in [
            (OpDef::row_topk("topk", ElementKind::F32), SortOrder::Desc),
            (
                OpDef::row_bottomk("bottomk", ElementKind::F32),
                SortOrder::Asc,
            ),
        ] {
            assert_eq!(sc.out_dtype, None);
            assert_eq!(sc.n_outputs(), 1); // increment-9 corollary: Both stays 1
            let key = topk_key(ElementKind::F32, 64);
            let plan = build_plan(&sc, &key);
            assert_eq!(
                plan.schedule,
                Schedule::RowSort {
                    order: want_order,
                    stable: true,
                    out: SortOut::Both,
                    limit: SortLimit::TopK
                }
            );
            assert_eq!(access_tag(&sc.access), "RowSort");
        }
    }

    // G0 regression — a `Full` sort still builds and schedules `limit: Full` (adding
    // the field did not disturb the existing three states).
    #[test]
    fn topk_full_regression_still_builds() {
        let sc = OpDef::row_sort_indices("fused", ElementKind::F32, SortOrder::Asc);
        let key = both_key(ElementKind::F32);
        let plan = build_plan(&sc, &key);
        assert_eq!(
            plan.schedule,
            Schedule::RowSort {
                order: SortOrder::Asc,
                stable: true,
                out: SortOut::Both,
                limit: SortLimit::Full
            }
        );
    }

    // G0 — TopK builds for every v1 dtype at a narrow k_out (the cap is orthogonal
    // to the dtype set; inherits the RowSort dtype gate verbatim).
    #[test]
    fn topk_builds_for_all_v1_dtypes() {
        for dt in [
            ElementKind::F32,
            ElementKind::F32Strict,
            ElementKind::F64,
            ElementKind::F16,
            ElementKind::Bf16,
            ElementKind::I32,
            ElementKind::I64,
        ] {
            let sc = OpDef::row_topk("topk", dt);
            let _ = build_plan(&sc, &topk_key(dt, 32));
        }
    }

    // G1 — out_dtype ↔ SortOut coupling is UNCHANGED under TopK: a TopK+Both with a
    // Some(I32) out_dtype rejects at `validate_row_sort`'s G1 (dtype-preserving on
    // output 0). Called DIRECTLY to isolate G1 from the `assert_valid_out_dtype`
    // front gate (mirrors `both_requires_out_dtype_none`).
    #[test]
    #[should_panic(expected = "dtype-preserving on output 0")]
    fn topk_both_requires_out_dtype_none() {
        let mut sc = OpDef::row_topk("topk", ElementKind::F32);
        sc.out_dtype = Some(ElementKind::I32);
        super::validate_row_sort(
            &sc,
            &topk_key(ElementKind::F32, 64),
            SortOrder::Desc,
            true,
            SortOut::Both,
            SortLimit::TopK,
        );
    }

    // G2 — operand count is extent-independent, so TopK+Both reuses it verbatim: a
    // 2-operand key against a Both op rejects (the two-pointer signature needs three).
    #[test]
    #[should_panic(expected = "expects n_inputs+2 operands")]
    fn topk_both_key_needs_three_operands() {
        let sc = OpDef::row_topk("topk", ElementKind::F32);
        let _ = build_plan(&sc, &sort_key(ElementKind::F32)); // 2-operand key
    }

    // G3 — output layout is UNCHANGED (LAYOUT-only, admits the narrower width): a
    // TopK+Both with a broadcast out_idx still rejects (the narrower width passes,
    // but the broadcast does not).
    #[test]
    #[should_panic(expected = "out_idx (operand 2) must be forward-dense")]
    fn topk_both_out_idx_broadcast_rejected() {
        let a = OperandDesc::new(2, &[256, 128], &[128, 1], ElementKind::F32, 256);
        let ov = OperandDesc::new(2, &[256, 64], &[64, 1], ElementKind::F32, 256);
        let oi = OperandDesc::new(2, &[256, 64], &[0, 1], ElementKind::I32, 256); // broadcast
        let key = structure_key(OpCategory::UnaryElementwise, &[a, ov, oi], ArchSku::Sm89);
        let sc = OpDef::row_topk("topk", ElementKind::F32);
        let _ = build_plan(&sc, &key);
    }

    // G4 — stable=false rejects under TopK too (the cap adds no exception to the
    // inherited RowSort gates).
    #[test]
    #[should_panic(expected = "unstable declined")]
    fn topk_rejects_unstable() {
        let mut sc = OpDef::row_topk("topk", ElementKind::F32);
        sc.access = Access::RowSort {
            order: SortOrder::Desc,
            stable: false,
            out: SortOut::Both,
            limit: SortLimit::TopK,
        };
        let _ = build_plan(&sc, &topk_key(ElementKind::F32, 64));
    }
}

#[cfg(test)]
mod select_gate_validate {
    //! WHERE/SELECT plan-gate tests (G1-G4). Per the house rule these call
    //! `build_plan` DIRECTLY (an emitter panic would mask a gate mutation);
    //! the emitter backstops are independent Tier-2 tests in `cuda`.
    use super::{Schedule, build_plan};
    use crate::ir::{BinaryOp, OpDef, ReduceOp, coord, input, konst};
    use unpopped_vocab::{
        ArchSku, ElementKind, OpCategory, OperandDesc, StructureKey, structure_key,
    };

    // n_operands contiguous 1D operands of `dtype` (V4-eligible at f32).
    fn key_dt(dtype: ElementKind, n_operands: usize) -> StructureKey {
        let a = OperandDesc::new(1, &[1024], &[1], dtype, 256);
        let ops: Vec<_> = std::iter::repeat_n(a, n_operands).collect();
        structure_key(OpCategory::TernaryElementwise, &ops, ArchSku::Sm89)
    }

    // G1: select is rejected OUTRIGHT at every int dtype (v1 float-only) —
    // the int-reject arm in `check_int_op_admissibility`.
    #[test]
    #[should_panic(expected = "Select has no integer lowering")]
    fn select_at_i32_is_rejected_at_the_plan_gate() {
        let op = OpDef::elementwise(
            "sel",
            3,
            &[ElementKind::I32],
            input(0).select(input(1), input(2)),
        );
        let _ = build_plan(&op, &key_dt(ElementKind::I32, 4));
    }

    #[test]
    #[should_panic(expected = "Select has no integer lowering")]
    fn select_at_i64_is_rejected_at_the_plan_gate() {
        let op = OpDef::elementwise(
            "sel",
            3,
            &[ElementKind::I64],
            input(0).select(input(1), input(2)),
        );
        let _ = build_plan(&op, &key_dt(ElementKind::I64, 4));
    }

    #[test]
    #[should_panic(expected = "Select has no integer lowering")]
    fn select_at_u8_is_rejected_at_the_plan_gate() {
        // U8 is exactly the 0c cond-observer dtype the outright reject sidesteps.
        let op = OpDef::elementwise(
            "sel",
            3,
            &[ElementKind::U8],
            input(0).select(input(1), input(2)),
        );
        let _ = build_plan(&op, &key_dt(ElementKind::U8, 4));
    }

    // G2: a Select ROOT never gets u8-out powers — `assert_valid_out_dtype`
    // admits only a `Binary(is_cmp)` root for `Some(U8)`, and a select stores
    // the ARM dtype. Already true without a code change; pinned here.
    #[test]
    #[should_panic(expected = "requires the body ROOT to be a comparison")]
    fn select_root_with_u8_out_is_rejected_at_the_plan_gate() {
        let mut op = OpDef::elementwise(
            "sel_mask",
            3,
            &[ElementKind::F32],
            input(0).select(input(1), input(2)),
        );
        op.out_dtype = Some(ElementKind::U8);
        let _ = build_plan(&op, &key_dt(ElementKind::F32, 4));
    }

    // G3: select is legal in every Access arm at float dtypes — the masked-sum
    // reduction (a select in the pre-fold body) builds a plan.
    #[test]
    fn masked_sum_reduction_with_a_select_body_builds() {
        let masked = OpDef::reduction(
            "masked_sum",
            1,
            &[ElementKind::F32],
            input(0)
                .binary(BinaryOp::CmpGt, konst(0.5))
                .select(input(0), konst(0.0)),
            ReduceOp::Sum,
        );
        let a = OperandDesc::new(2, &[256, 128], &[128, 1], ElementKind::F32, 256);
        let o = OperandDesc::new(1, &[256], &[1], ElementKind::F32, 256);
        let key = structure_key(OpCategory::Reduction, &[a, o], ArchSku::Sm89);
        let plan = build_plan(&masked, &key);
        assert!(matches!(plan.schedule, Schedule::Reduction { .. }));
    }

    // G4 (no new schedule machinery): a Coord-bearing cond rides the existing
    // Strided force — the triu body's route…
    #[test]
    fn triu_select_body_takes_the_strided_schedule() {
        let triu = OpDef::elementwise(
            "triu_sel",
            1,
            &[ElementKind::F32],
            coord(1)
                .binary(BinaryOp::CmpGe, coord(0) + konst(0.0))
                .select(input(0), konst(0.0)),
        );
        let a = OperandDesc::new(2, &[128, 256], &[256, 1], ElementKind::F32, 256);
        let key = structure_key(OpCategory::BinaryElementwise, &[a, a], ArchSku::Sm89);
        let plan = build_plan(&triu, &key);
        assert!(
            matches!(plan.schedule, Schedule::Strided),
            "Coord cond must force Strided, got {:?}",
            plan.schedule
        );
    }

    // …and an all-Input select on a vec-width-4 key vectorizes per-lane
    // (uniform-dtype select needs no schedule change).
    #[test]
    fn all_input_select_vectorizes_at_a_v4_cell() {
        let op = OpDef::elementwise(
            "sel",
            3,
            &[ElementKind::F32],
            input(0).select(input(1), input(2)),
        );
        let key = key_dt(ElementKind::F32, 4);
        let plan = build_plan(&op, &key);
        assert!(
            matches!(plan.schedule, Schedule::Vectorized { width: 4 }),
            "all-Input select at a V4 cell must vectorize, got {:?}",
            plan.schedule
        );
    }
}

#[cfg(test)]
mod half_arith_tests {
    //! U1 of the joint P40/RTX 4070 plan (`docs/joint-gpu-milestone-plan.md`): the
    //! first plan decision that differs by sm.
    use super::{HalfArith, build_plan, half_arith_for};
    use crate::ir::{OpDef, input};
    use unpopped_vocab::{ElementKind, OpCategory, OperandDesc, TargetId, structure_key};

    fn t(tok: &str) -> TargetId {
        TargetId::parse(tok).expect("grammar-valid token")
    }

    /// The P40 promotes; the Ampere and Ada parts keep native 16-bit math.
    #[test]
    fn sm61_promotes_and_sm80_sm89_stay_native() {
        assert_eq!(half_arith_for(t("cuda:sm61")), HalfArith::ViaF32);
        assert_eq!(half_arith_for(t("cuda:sm80")), HalfArith::Native);
        assert_eq!(half_arith_for(t("cuda:sm89")), HalfArith::Native);
    }

    /// Unsourced means today's behaviour. 8.7, 10.3, 11.0 and 12.1 have resource
    /// rows but no throughput column. sm107 has no row at all, and a `vulkan:`
    /// token is outside the CUDA table.
    #[test]
    fn unsourced_targets_stay_native() {
        for tok in [
            "cuda:sm87",
            "cuda:sm103",
            "cuda:sm110",
            "cuda:sm121",
            "cuda:sm107",
            "vulkan:spirv1.6",
        ] {
            assert_eq!(half_arith_for(t(tok)), HalfArith::Native, "{tok}");
        }
        // Positive control: the resource rows for the four sourced-less CUDA
        // capabilities DO exist, so `Native` here comes from the missing rate,
        // not from a missing row.
        for tok in ["cuda:sm87", "cuda:sm103", "cuda:sm110", "cuda:sm121"] {
            assert!(
                crate::capability::capabilities_for(t(tok)).is_some(),
                "{tok}"
            );
        }
    }

    /// The method a backend actually calls reads the plan's own `key.target`.
    /// One f16 op, keyed for two targets, must disagree.
    #[test]
    fn the_plan_method_reads_its_key_target() {
        let op = OpDef::elementwise("relu_f16", 1, &[ElementKind::F16], input(0).relu());
        let a = OperandDesc::new(1, &[1024], &[1], ElementKind::F16, 256);
        let k61 = structure_key(OpCategory::UnaryElementwise, &[a, a], t("cuda:sm61"));
        let k89 = structure_key(OpCategory::UnaryElementwise, &[a, a], t("cuda:sm89"));
        assert_eq!(build_plan(&op, &k61).half_arith(), HalfArith::ViaF32);
        assert_eq!(build_plan(&op, &k89).half_arith(), HalfArith::Native);
    }
}

#[cfg(test)]
mod contraction_check_pins {
    //! Pins found by the 2026-10-17 mutation audit of the contraction checks: each test
    //! kills a mutant the suite let survive, named in its comment.
    use super::{contraction_epilogue_admissible as adm, validate_contraction_roles as roles};
    use crate::ir::{AxisRole::*, ContractionAxes, ScalarExpr as E};

    fn sel(c: E, a: E, b: E) -> E {
        E::Select(Box::new(c), Box::new(a), Box::new(b))
    }

    #[test]
    fn a_fused_bias_leaf_is_input_2_up_to_but_excluding_n_inputs() {
        // `*i < n_inputs` -> `<=`: Input(n_inputs) is one past the last operand.
        assert!(adm(&E::Input(2), 3));
        assert!(!adm(&E::Input(3), 3), "Input(n_inputs) names no operand");
        assert!(
            !adm(&E::Input(2), 2),
            "a bias-free contraction admits no Input"
        );
        assert!(!adm(&E::Input(1), 3), "rhs is consumed by the contraction");
    }

    #[test]
    fn select_admits_only_when_all_three_children_do() {
        // `&&` -> `||` on each of the two joins: a bad child in each position alone.
        let ok = || E::Const(1.0);
        let bad = || E::Input(0);
        assert!(adm(&sel(ok(), ok(), ok()), 2));
        assert!(!adm(&sel(bad(), ok(), ok()), 2), "bad condition");
        assert!(!adm(&sel(ok(), bad(), ok()), 2), "bad then-arm");
        assert!(!adm(&sel(ok(), ok(), bad()), 2), "bad else-arm");
    }

    #[test]
    fn a_role_vector_of_the_wrong_length_names_its_length_on_either_side() {
        // `||` -> `&&` on the length check: with only one side wrong the later checks
        // would refuse it too, but under a different message.
        let msg = "role vector length must equal operand rank";
        let lhs_long = ContractionAxes {
            lhs: vec![FreeM, ContractedK, Batch],
            rhs: vec![ContractedK, FreeN],
        };
        assert_eq!(roles(&lhs_long, 2, 2).unwrap_err(), msg);
        let rhs_long = ContractionAxes {
            lhs: vec![FreeM, ContractedK],
            rhs: vec![ContractedK, FreeN, Batch],
        };
        assert_eq!(roles(&rhs_long, 2, 2).unwrap_err(), msg);
    }

    #[test]
    fn an_operand_without_a_contracted_k_is_refused_as_such_on_either_side() {
        // `||` -> `&&` on the K-count check: with only one side lacking K the batch
        // checks below would refuse it too, but under a different message.
        let msg = "v1: exactly one ContractedK per operand (multi-group deferred)";
        let lhs_no_k = ContractionAxes {
            lhs: vec![FreeM, Batch],
            rhs: vec![ContractedK, FreeN],
        };
        assert_eq!(roles(&lhs_no_k, 2, 2).unwrap_err(), msg);
        let rhs_no_k = ContractionAxes {
            lhs: vec![FreeM, ContractedK],
            rhs: vec![Batch, FreeN],
        };
        assert_eq!(roles(&rhs_no_k, 2, 2).unwrap_err(), msg);
    }

    #[test]
    fn a_rank_4_operand_on_either_side_is_refused_as_a_rank() {
        // `||` -> `&&` on the rank check: the batch-count check would refuse a lone
        // rank-4 side too, but as a batch disagreement, not as a rank.
        let msg = "v1: contraction operands must be rank-2 or rank-3";
        let lhs4 = ContractionAxes {
            lhs: vec![Batch, Batch, FreeM, ContractedK],
            rhs: vec![ContractedK, FreeN],
        };
        assert_eq!(roles(&lhs4, 4, 2).unwrap_err(), msg);
        let rhs4 = ContractionAxes {
            lhs: vec![FreeM, ContractedK],
            rhs: vec![Batch, Batch, ContractedK, FreeN],
        };
        assert_eq!(roles(&rhs4, 2, 4).unwrap_err(), msg);
    }
}
