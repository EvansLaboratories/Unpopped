// SPDX-License-Identifier: MIT OR Apache-2.0
//! **The layers agree on input views** (`docs/ir-kernel-layer.md` section 7): an Elementwise op
//! whose inputs are read through `View::Permute`, `Layout`, `Broadcast` or `Reshape`, lifted by
//! `lift_elementwise_opdef` and run by the kernel interpreter, produces byte for byte what
//! `oracle::evaluate` produces.
//!
//! In the kernel layer a `Place` goes through the bound buffer's own shape and strides, so:
//!
//! - **`Permute`** is lift code: iteration axis `d` indexes producer axis `perm[d]`, so the producer
//!   coordinate on axis `perm[d]` is the loop counter of axis `d`. The producer buffer is bound as
//!   it is.
//! - **`Layout`**, **`Broadcast`** and **`Reshape`** (and a flipped or broadcast operand) are binding
//!   contracts, like a base offset: the caller binds the view's extents and strides (a `Broadcast`
//!   as stride 0) on the same storage, and the lifted kernel is the identity one.
//!
//! The body `Input(0) * Input(1) + Input(0)` is not symmetric, so a view applied to the wrong
//! operand, or a permutation applied the wrong way round, disagrees.

use unpopped::ir::{Expr, OpDef, ScalarExpr, View, input};
use unpopped::kernel::check_kernel;
use unpopped::kernel_lift::{LiftDecline, lift_elementwise_opdef};
use unpopped::oracle::kernel_eval::{KernelArg, run_kernel};
use unpopped::oracle::{TypedBuffer, evaluate};
use unpopped::plan::try_build_plan;
use unpopped_vocab::{ArchSku, AxisMask, ElementKind, OpCategory, OperandDesc, structure_key};

fn width(dtype: ElementKind) -> usize {
    if dtype == ElementKind::F64 { 8 } else { 4 }
}

/// `len` storage elements of distinct small values.
fn storage(dtype: ElementKind, len: usize, seed: i64) -> Vec<u8> {
    (0..len as i64)
        .flat_map(|i| {
            let v = (i * 5 + seed) % 17 - 8;
            match dtype {
                ElementKind::F32 => (v as f32 * 0.5).to_le_bytes().to_vec(),
                ElementKind::F64 => (v as f64 * 0.5).to_le_bytes().to_vec(),
                _ => (v as i32 * 3).to_le_bytes().to_vec(),
            }
        })
        .collect()
}

fn buffer(
    dtype: ElementKind,
    shape: &[i64],
    strides: &[i64],
    base: i64,
    len: usize,
    seed: i64,
) -> TypedBuffer {
    let mut b = TypedBuffer::new(
        dtype,
        shape.to_vec(),
        strides.to_vec(),
        storage(dtype, len, seed),
    );
    b.base_offset = base;
    b
}

/// One view case: the iteration `shape`, the second input's view, the second input's buffer as the
/// oracle sees it and as the kernel is bound to it.
struct Case {
    shape: Vec<i64>,
    view: View,
    oracle_in1: TypedBuffer,
    kernel_in1: TypedBuffer,
}

/// `Input(0) * Input(1) + Input(0)`.
fn body_of() -> ScalarExpr {
    ScalarExpr::Add(
        Box::new(ScalarExpr::Mul(Box::new(input(0).0), Box::new(input(1).0))),
        Box::new(input(0).0),
    )
}

/// What the oracle writes for the viewed op on `case`.
fn oracle_out(dtype: ElementKind, case: &Case, op: &OpDef, in0: &TypedBuffer) -> Vec<u8> {
    let rank = case.shape.len();
    let d = |t: &TypedBuffer| OperandDesc::new(t.shape.len(), &t.shape, &t.strides, dtype, 256);
    let out_desc = OperandDesc::new(rank, &case.shape, &strides_of(&case.shape), dtype, 256);
    let ops = [d(in0), d(&case.oracle_in1), out_desc];
    let key = structure_key(OpCategory::BinaryElementwise, &ops, ArchSku::Sm89);
    let plan = try_build_plan(op, &key).expect("the viewed op plans");
    let len = case.shape.iter().product::<i64>() as usize * width(dtype);
    evaluate(&plan, &ops, &[in0.clone(), case.oracle_in1.clone()], &[])
        .remove(0)
        .bytes[..len]
        .to_vec()
}

/// What `op`, lifted, writes on the buffers the kernel is bound to; `Err` when it traps.
fn kernel_out(
    dtype: ElementKind,
    case: &Case,
    op: &OpDef,
    in0: &TypedBuffer,
) -> Result<Vec<u8>, String> {
    let rank = case.shape.len();
    let len = case.shape.iter().product::<i64>() as usize * width(dtype);
    let kernel = lift_elementwise_opdef(op, dtype, rank as u8).expect("the op lifts");
    assert_eq!(
        check_kernel(&kernel),
        Ok(()),
        "a lifted kernel is well formed"
    );
    let mut args = vec![
        KernelArg::Buffer(in0.clone()),
        KernelArg::Buffer(case.kernel_in1.clone()),
        KernelArg::Buffer(TypedBuffer::new(
            dtype,
            case.shape.clone(),
            strides_of(&case.shape),
            vec![0; len],
        )),
    ];
    args.extend(case.shape.iter().map(|&e| KernelArg::Int(i128::from(e))));
    run_kernel(&kernel, &mut args).map_err(|e| format!("{e:?}"))?;
    let KernelArg::Buffer(got) = &args[2] else {
        unreachable!()
    };
    Ok(got.bytes[..len].to_vec())
}

fn run(dtype: ElementKind, case: &Case) {
    let n: i64 = case.shape.iter().product();
    let in0 = buffer(
        dtype,
        &case.shape,
        &strides_of(&case.shape),
        0,
        n as usize,
        1,
    );
    let op = OpDef::elementwise("v", 2, &[dtype], Expr(body_of()))
        .with_views(vec![View::Identity, case.view.clone()]);
    let want = oracle_out(dtype, case, &op, &in0);
    let got = kernel_out(dtype, case, &op, &in0).expect("the kernel runs");
    assert_eq!(got, want, "{:?}", case.view);

    // Control (a Permute is lift code, the other views are bindings): the same bound buffers under
    // the view-free op either trap or give a different answer, so the agreement above depends on the
    // lift reordering the producer coordinates.
    if matches!(case.view, View::Permute { .. }) {
        let plain = OpDef::elementwise("v", 2, &[dtype], Expr(body_of()));
        let differs = kernel_out(dtype, case, &plain, &in0).map_or(true, |g| g != want);
        assert!(
            differs,
            "ignoring the permute must not reproduce the oracle"
        );
    }
}

fn strides_of(shape: &[i64]) -> Vec<i64> {
    (0..shape.len())
        .map(|d| shape[d + 1..].iter().product())
        .collect()
}

const DTYPES: [ElementKind; 3] = [ElementKind::F32, ElementKind::F64, ElementKind::I32];

#[test]
fn a_rank_2_permute_agrees() {
    for dtype in DTYPES {
        // The second input is stored [4, 3] and read as its transpose [3, 4].
        let producer = buffer(dtype, &[4, 3], &[3, 1], 0, 12, 4);
        run(
            dtype,
            &Case {
                shape: vec![3, 4],
                view: View::Permute { perm: vec![1, 0] },
                oracle_in1: producer.clone(),
                kernel_in1: producer,
            },
        );
    }
}

#[test]
fn a_rank_3_permute_agrees_in_both_directions() {
    let dtype = ElementKind::F32;
    // Iteration [2, 3, 4], perm [2, 0, 1]: axis 0 -> producer axis 2, 1 -> 0, 2 -> 1.
    let producer = buffer(dtype, &[3, 4, 2], &[8, 2, 1], 0, 24, 4);
    run(
        dtype,
        &Case {
            shape: vec![2, 3, 4],
            view: View::Permute {
                perm: vec![2, 0, 1],
            },
            oracle_in1: producer.clone(),
            kernel_in1: producer,
        },
    );
    // The inverse permutation over the matching producer [4, 2, 3] reads a different arrangement.
    let producer = buffer(dtype, &[4, 2, 3], &[6, 3, 1], 0, 24, 4);
    run(
        dtype,
        &Case {
            shape: vec![2, 3, 4],
            view: View::Permute {
                perm: vec![1, 2, 0],
            },
            oracle_in1: producer.clone(),
            kernel_in1: producer,
        },
    );
}

#[test]
fn a_layout_view_is_a_binding_of_its_strides() {
    for dtype in DTYPES {
        // A strided slice of a [6, 8] matrix starting at element 1: every other row and column.
        let storage_buf = buffer(dtype, &[6, 8], &[8, 1], 1, 48, 4);
        let view = View::Layout {
            shape: vec![3, 4],
            stride: vec![16, 2],
        };
        let mut kernel_in1 = storage_buf.clone();
        kernel_in1.shape = vec![3, 4];
        kernel_in1.strides = vec![16, 2];
        run(
            dtype,
            &Case {
                shape: vec![3, 4],
                view,
                oracle_in1: storage_buf,
                kernel_in1,
            },
        );
    }
}

#[test]
fn a_broadcast_view_is_a_stride_zero_binding() {
    let dtype = ElementKind::F32;
    let mut bcast = AxisMask::EMPTY;
    bcast.set(0);
    // A row of 4 broadcast down 3 rows.
    let row = buffer(dtype, &[3, 4], &[0, 1], 0, 4, 4);
    run(
        dtype,
        &Case {
            shape: vec![3, 4],
            view: View::Broadcast { bcast },
            oracle_in1: row.clone(),
            kernel_in1: row,
        },
    );
}

#[test]
fn a_same_rank_reshape_and_a_flipped_operand_agree() {
    let dtype = ElementKind::F32;
    let plain = buffer(dtype, &[3, 4], &[4, 1], 0, 12, 4);
    run(
        dtype,
        &Case {
            shape: vec![3, 4],
            view: View::Reshape { producer_rank: 2 },
            oracle_in1: plain.clone(),
            kernel_in1: plain,
        },
    );
    // Columns read right to left: stride -1 from the last column of each row.
    let flipped = buffer(dtype, &[3, 4], &[4, -1], 3, 12, 4);
    run(
        dtype,
        &Case {
            shape: vec![3, 4],
            view: View::Identity,
            oracle_in1: flipped.clone(),
            kernel_in1: flipped,
        },
    );
}

#[test]
fn forms_the_lift_does_not_cover_decline_with_a_reason() {
    let f32k = [ElementKind::F32];
    let base = OpDef::elementwise("e", 1, &f32k, Expr(input(0).0));
    // Control: an identity view lifts, and so does a permute.
    let ident = base.clone().with_views(vec![View::Identity]);
    assert!(lift_elementwise_opdef(&ident, ElementKind::F32, 2).is_ok());
    let perm = base
        .clone()
        .with_views(vec![View::Permute { perm: vec![1, 0] }]);
    assert!(lift_elementwise_opdef(&perm, ElementKind::F32, 2).is_ok());
    // A permutation that is not one of the iteration rank is malformed.
    let bad = base
        .clone()
        .with_views(vec![View::Permute { perm: vec![0, 0] }]);
    assert_eq!(
        lift_elementwise_opdef(&bad, ElementKind::F32, 2),
        Err(LiftDecline::UnsupportedForm("an ill-formed view"))
    );
    // Any other access declines a view, since it would ignore it.
    let mut scan = OpDef::scan_simple("s", &f32k, unpopped::ir::ReduceOp::Sum, 1, false, false);
    scan.views = vec![View::Permute { perm: vec![1, 0] }];
    assert_eq!(
        unpopped::kernel_lift::lift_scan_opdef(&scan, ElementKind::F32, 2),
        Err(LiftDecline::UnsupportedForm("a non-identity view"))
    );
}
