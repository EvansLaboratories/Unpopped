// SPDX-License-Identifier: MIT OR Apache-2.0
//! **The layers agree on the total integer rule** (ROADMAP E16, `docs/int-div-shift-semantics.md`
//! is the single source of the semantics): an Elementwise integer `Div`, `Shl` and `Shr`, lifted into the kernel layer and run by the kernel interpreter, produce
//! byte for byte what `oracle::evaluate` produces for the `OpDef`.
//!
//! The plan is built for a backend that declares `Backend::total_integer_ops` (0.40.7): without
//! that the gate refuses integer `Div`. Every corpus carries the boundary cases the document
//! lists, as a cross product (a zero divisor, `MIN / -1`, an oversized and a negative shift count).
//! `Rem` is not here: its `Select` composition does not plan at an integer dtype (ROADMAP E16).

use unpopped::backend::{Backend, DtypeSupport, GeneratedKernel, LowerError};
use unpopped::ir::{BinaryOp, Expr, OpDef, ScalarExpr, input};
use unpopped::kernel_lift::lift_elementwise_opdef;
use unpopped::oracle::kernel_eval::{KernelArg, run_kernel};
use unpopped::oracle::{TypedBuffer, evaluate};
use unpopped::plan::{KernelPlan, try_build_plan_for};
use unpopped_vocab::{ArchSku, ElementKind, OpCategory, OperandDesc, TargetId, structure_key};

const N: i64 = 15;

/// Opts in to the total integer rule; never lowers.
struct Total;
impl Backend for Total {
    fn name(&self) -> &str {
        "total"
    }
    fn provider(&self) -> &str {
        "unpopped-tests"
    }
    fn supports_dtype(&self, _d: ElementKind, _t: TargetId) -> DtypeSupport {
        DtypeSupport::Supported
    }
    fn total_integer_ops(&self) -> bool {
        true
    }
    fn lower(&self, _p: &KernelPlan<'_>) -> Result<GeneratedKernel, LowerError> {
        Err(LowerError::UnsupportedOp {
            detail: "not lowered".into(),
        })
    }
}

fn bx(e: ScalarExpr) -> Box<ScalarExpr> {
    Box::new(e)
}

/// Little-endian bytes of `v` at `dtype`'s width (a value outside the range wraps).
fn bytes(dtype: ElementKind, v: i128) -> Vec<u8> {
    let w = if matches!(dtype, ElementKind::I64 | ElementKind::U64) {
        8
    } else {
        4
    };
    v.to_le_bytes()[..w].to_vec()
}

/// The boundary values of `dtype`: extremes, zero, minus one, ones, and counts around the width.
fn corpus(dtype: ElementKind) -> Vec<i128> {
    let (lo, hi): (i128, i128) = match dtype {
        ElementKind::I32 => (i128::from(i32::MIN), i128::from(i32::MAX)),
        ElementKind::I64 => (i128::from(i64::MIN), i128::from(i64::MAX)),
        ElementKind::U32 => (0, i128::from(u32::MAX)),
        _ => (0, i128::from(u64::MAX)),
    };
    let neg = if lo < 0 { -1 } else { 7 };
    vec![
        0,
        1,
        neg,
        lo,
        hi,
        2,
        31,
        32,
        33,
        63,
        64,
        65,
        -3,
        100,
        hi - 1,
    ]
}

/// The cross product of the corpus: element `[i][j]` of the first operand is `vals[i]`, of the second
/// `vals[j]`, so every pair (a zero divisor, `MIN / -1`, an oversized count) is present.
fn buffer(dtype: ElementKind, first: bool) -> TypedBuffer {
    let vals = corpus(dtype);
    let data: Vec<u8> = (0..vals.len())
        .flat_map(|i| (0..vals.len()).map(move |j| if first { i } else { j }))
        .flat_map(|k| bytes(dtype, vals[k]))
        .collect();
    TypedBuffer::new(dtype, vec![N, N], vec![N, 1], data)
}

/// Oracle bytes and lifted-kernel bytes for `body` over two corpus buffers.
fn both(name: &str, dtype: ElementKind, body: ScalarExpr) -> (Vec<u8>, Vec<u8>) {
    let op = OpDef::elementwise(name, 2, &[dtype], Expr(body));
    let (a, b) = (buffer(dtype, true), buffer(dtype, false));
    let desc = |t: &TypedBuffer| OperandDesc::new(2, &t.shape, &t.strides, dtype, 256);
    let out = OperandDesc::new(2, &[N, N], &[N, 1], dtype, 256);
    let descs = [desc(&a), desc(&b), out];
    let key = structure_key(OpCategory::BinaryElementwise, &descs, ArchSku::Sm89);
    let plan = try_build_plan_for(&op, &key, &Total).expect("the op plans for an opted-in backend");
    let want = evaluate(&plan, &descs, &[a.clone(), b.clone()], &[])
        .remove(0)
        .bytes;
    let kernel = lift_elementwise_opdef(&op, dtype, 2).expect("the op lifts");
    let zero = vec![0u8; want.len()];
    let sink = TypedBuffer::new(dtype, vec![N, N], vec![N, 1], zero);
    let mut args = vec![
        KernelArg::Buffer(a),
        KernelArg::Buffer(b),
        KernelArg::Buffer(sink),
        KernelArg::Int(i128::from(N)),
        KernelArg::Int(i128::from(N)),
    ];
    run_kernel(&kernel, &mut args).expect("the kernel runs");
    let KernelArg::Buffer(got) = &args[2] else {
        unreachable!()
    };
    (want, got.bytes.clone())
}

fn a() -> ScalarExpr {
    input(0).0
}
fn b() -> ScalarExpr {
    input(1).0
}

#[test]
fn integer_div_and_shifts_agree_at_every_defined_width() {
    for dt in [
        ElementKind::I32,
        ElementKind::I64,
        ElementKind::U32,
        ElementKind::U64,
    ] {
        let forms = [
            ("div", ScalarExpr::Div(bx(a()), bx(b()))),
            ("shl", ScalarExpr::Binary(BinaryOp::Shl, bx(a()), bx(b()))),
            ("shr", ScalarExpr::Binary(BinaryOp::Shr, bx(a()), bx(b()))),
        ];
        for (name, body) in forms {
            let (want, got) = both(name, dt, body);
            assert_eq!(got, want, "{name} at {dt:?}");
        }
    }
}
