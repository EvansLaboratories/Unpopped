//! Every Qwen3 dense op that Unpopped's IR can express plans for `cuda:sm61`
//! exactly as it does for `cuda:sm89`, except for [`HalfArith`].
//!
//! # Why this test exists
//!
//! U2 of the joint P40/RTX 4070 plan (`docs/joint-gpu-milestone-plan.md`).
//! baracuda's sm_61 route is IR → `baracuda-cuda-emit` → NVRTC, and its gap
//! analysis (`docs/sm61-parse-emit-gap-analysis.md`) left one question for us
//! open: *does anything on `unpopped::generate()`'s path reject a `cuda:sm61`
//! target before the emitter sees it?* A grep says no. This makes it a checked
//! fact, op by op, for the op set the bring-up model needs.
//!
//! # What it does not prove
//!
//! Numerics. Each op's arithmetic is checked against the oracle elsewhere, and
//! the numbers on a real P40 are M4's. This file proves the plan exists and is
//! target-neutral where it must be.
//!
//! # The gap it records
//!
//! **Attention has no IR node** (`Access::Attention` is the last item in
//! `docs/idiom-lifting-design.md` §11). For bring-up, it composes from the ops
//! below: a `Contraction` for the scores, a `RowReduce` softmax with a causal
//! mask, and a `Contraction` with V.

mod common;

use common::StubBackend;
use unpopped::ir::{
    BaseOffset, ContractionAxes, OpDef, ReduceOp, ReduceStage, View, input, konst, reduced,
};
use unpopped::plan::try_build_plan;
use unpopped::{HalfArith, Schedule, try_generate};
use unpopped_vocab::{ElementKind, OpCategory, OperandDesc, StructureKey, TargetId, structure_key};

const F32: ElementKind = ElementKind::F32;
const F16: ElementKind = ElementKind::F16;

fn t(tok: &str) -> TargetId {
    TargetId::parse(tok).expect("grammar-valid token")
}

/// One Qwen3 op: its IR, plus a key builder parameterised by target.
struct Case {
    name: &'static str,
    op: OpDef,
    key: Box<dyn Fn(TargetId) -> StructureKey>,
    /// The schedule this op must route to, so a case that silently fell to a
    /// different path is caught.
    schedule: fn(&Schedule) -> bool,
}

fn contig1(dt: ElementKind) -> OperandDesc {
    OperandDesc::new(1, &[1024], &[1], dt, 256)
}
/// `[rows, hidden]`, row-streamed.
fn stream(dt: ElementKind) -> OperandDesc {
    OperandDesc::new(2, &[256, 128], &[128, 1], dt, 256)
}
/// A per-feature weight broadcast down the rows (RMSNorm's `w`).
fn row_broadcast(dt: ElementKind) -> OperandDesc {
    OperandDesc::new(2, &[256, 128], &[0, 1], dt, 256)
}

fn cases() -> Vec<Case> {
    let mut v = Vec::new();
    for dt in [F32, F16] {
        v.push(residual_add(dt));
        v.push(swiglu(dt));
    }
    v.extend([rms_norm(), softmax(), rope_pair(), embedding(), matmul()]);
    v
}

fn binary_elementwise_key(dt: ElementKind) -> Box<dyn Fn(TargetId) -> StructureKey> {
    Box::new(move |tg| {
        let a = contig1(dt);
        structure_key(OpCategory::BinaryElementwise, &[a, a, a], tg)
    })
}

fn vector_or_scalar(s: &Schedule) -> bool {
    matches!(s, Schedule::Vectorized { .. } | Schedule::Scalar)
}

/// Residual add.
fn residual_add(dt: ElementKind) -> Case {
    Case {
        name: if dt == F32 {
            "residual_add_f32"
        } else {
            "residual_add_f16"
        },
        op: OpDef::elementwise("residual_add", 2, &[dt], input(0) + input(1)),
        key: binary_elementwise_key(dt),
        schedule: vector_or_scalar,
    }
}

/// SwiGLU: silu(gate) * up.
fn swiglu(dt: ElementKind) -> Case {
    Case {
        name: if dt == F32 {
            "swiglu_f32"
        } else {
            "swiglu_f16"
        },
        op: OpDef::elementwise("swiglu", 2, &[dt], input(0).silu() * input(1)),
        key: binary_elementwise_key(dt),
        schedule: vector_or_scalar,
    }
}

/// RMSNorm (also Qwen3's per-head q/k norm): x * w / sqrt(mean(x²) + eps).
fn rms_norm() -> Case {
    Case {
        name: "rms_norm_f32",
        op: OpDef::row_reduce(
            "rms_norm",
            2,
            &[F32],
            vec![ReduceStage {
                pre: (input(0) * input(0)).0,
                op: ReduceOp::Mean,
            }],
            input(0) * input(1) / (reduced(0) + konst(1e-6)).sqrt(),
        ),
        key: Box::new(|tg| {
            structure_key(
                OpCategory::Normalization,
                &[stream(F32), row_broadcast(F32), stream(F32)],
                tg,
            )
        }),
        schedule: |s| matches!(s, Schedule::RowReduce { .. }),
    }
}

/// Softmax: exp(x - max) / sum(exp(x - max)).
fn softmax() -> Case {
    Case {
        name: "softmax_f32",
        op: OpDef::row_reduce(
            "softmax",
            1,
            &[F32],
            vec![
                ReduceStage {
                    pre: input(0).0,
                    op: ReduceOp::Max,
                },
                ReduceStage {
                    pre: (input(0) - reduced(0)).exp().0,
                    op: ReduceOp::Sum,
                },
            ],
            (input(0) - reduced(0)).exp() / reduced(1),
        ),
        key: Box::new(|tg| structure_key(OpCategory::Softmax, &[stream(F32), stream(F32)], tg)),
        schedule: |s| matches!(s, Schedule::RowReduce { .. }),
    }
}

/// RoPE, one lane of the rotate-half pair: x*cos + partner*sin, the partner
/// read through a runtime base offset (baracuda's rope pair kernels).
fn rope_pair() -> Case {
    Case {
        name: "rope_pair_f32",
        op: OpDef::elementwise(
            "rope_pair",
            4,
            &[F32],
            input(0) * input(2) + input(1) * input(3),
        )
        .with_base_offsets(
            vec![
                BaseOffset::Runtime,
                BaseOffset::Runtime,
                BaseOffset::Zero,
                BaseOffset::Zero,
            ],
            BaseOffset::Runtime,
        ),
        key: Box::new(|tg| {
            let a = contig1(F32);
            structure_key(OpCategory::BinaryElementwise, &[a, a, a, a, a], tg)
        }),
        schedule: |_| true,
    }
}

/// Token embedding: a gather on axis 0.
fn embedding() -> Case {
    Case {
        name: "embedding_f32",
        op: OpDef::embedding("embedding", &[F32], ElementKind::I32),
        key: Box::new(|tg| {
            let data = OperandDesc::new(2, &[4, 3], &[3, 1], F32, 256);
            let idx = OperandDesc::new(2, &[4, 3], &[3, 1], ElementKind::I32, 256);
            let out = OperandDesc::new(2, &[4, 3], &[3, 1], F32, 256);
            structure_key(OpCategory::BinaryElementwise, &[data, idx, out], tg)
        }),
        schedule: |s| matches!(s, Schedule::Strided),
    }
}

/// The projections: x @ Wᵀ (weights stored [N, K]).
fn matmul() -> Case {
    Case {
        name: "matmul_f32",
        op: OpDef::contraction("matmul", &[F32], ContractionAxes::matmul(), reduced(0))
            .with_views(vec![View::Identity, View::Permute { perm: vec![1, 0] }]),
        key: Box::new(|tg| {
            let lhs = OperandDesc::new(2, &[8, 16], &[16, 1], F32, 256);
            let rhs = OperandDesc::new(2, &[16, 4], &[1, 16], F32, 256);
            let out = OperandDesc::new(2, &[8, 4], &[4, 1], F32, 256);
            structure_key(OpCategory::Gemm, &[lhs, rhs, out], tg)
        }),
        schedule: |s| matches!(s, Schedule::Contraction),
    }
}

/// The headline: every case plans and generates for sm_61, on the same
/// schedule as sm_89, and the plans differ only where `HalfArith` says so.
#[test]
fn every_qwen3_dense_op_plans_and_generates_for_sm61() {
    let (sm61, sm89) = (t("cuda:sm61"), t("cuda:sm89"));
    let cases = cases();
    assert_eq!(cases.len(), 9, "the census changed; update the doc list");
    for c in &cases {
        let k61 = (c.key)(sm61);
        let k89 = (c.key)(sm89);
        let p61 = try_build_plan(&c.op, &k61)
            .unwrap_or_else(|e| panic!("{}: sm61 plan refused: {e}", c.name));
        let p89 = try_build_plan(&c.op, &k89)
            .unwrap_or_else(|e| panic!("{}: sm89 plan refused: {e}", c.name));

        assert!(
            (c.schedule)(&p61.schedule),
            "{}: unexpected schedule {:?}",
            c.name,
            p61.schedule
        );
        assert_eq!(
            p61.schedule, p89.schedule,
            "{}: schedule depends on target",
            c.name
        );
        assert_eq!(
            (p61.dtype, p61.out_dtype, p61.n_inputs, p61.n_outputs),
            (p89.dtype, p89.out_dtype, p89.n_inputs, p89.n_outputs),
            "{}",
            c.name
        );

        // The one per-sm decision: 16-bit math promotes on sm_61 only.
        assert_eq!(p61.half_arith(), HalfArith::ViaF32, "{}", c.name);
        assert_eq!(p89.half_arith(), HalfArith::Native, "{}", c.name);

        // The whole generate path, not just the planner.
        try_generate(&c.op, &k61, &StubBackend)
            .unwrap_or_else(|e| panic!("{}: sm61 generate refused: {e}", c.name));
    }
}

/// Positive control: the same harness DOES see a refusal, so "every case
/// planned" is not a harness that cannot fail. A gather with a float index is
/// refused by the plan gate on both targets.
#[test]
#[should_panic(expected = "index_dtype must be an integer")]
fn the_harness_sees_a_refusal() {
    let op = OpDef::embedding("bad_embedding", &[F32], F32);
    let data = OperandDesc::new(2, &[4, 3], &[3, 1], F32, 256);
    let out = data;
    let k = structure_key(
        OpCategory::BinaryElementwise,
        &[data, data, out],
        t("cuda:sm61"),
    );
    let _ = try_build_plan(&op, &k);
}
