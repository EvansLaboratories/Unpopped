// SPDX-License-Identifier: MIT OR Apache-2.0
//! **The layers agree on row reductions, in the mode each case allows** (`docs/ir-kernel-layer.md`
//! section 7). The oracle passes each stage's value on as an unrounded `f64` (`Reduced(s)`); the
//! kernel language cannot state an `f64` leaf inside a compute-dtype expression, so the lift rounds
//! each stage once to the compute dtype (`lift_row_reduce_opdef`, module docs). That gives two modes:
//!
//! * **byte-exact**: the epilogue is the identity (`Reduced(0)`), or the compute dtype is `F64`
//!   (the cast is then the identity). Compared with `==` on the output bytes.
//! * **tolerance**: an `F32` epilogue that computes with a stage value (softmax, a centred
//!   quotient). Compared to within [`MAX_ULPS`] ulps, NaN equal to NaN and infinities exact. The
//!   oracle is a *tolerance referee* here, as it is for `eval_contraction`'s fused epilogue.
//!
//! A tolerance case must also prove the tolerance is exercised: `the_tolerance_is_not_vacuous`
//! measures that the rounded and unrounded stage values really do differ on the corpus.

use unpopped::ir::{Expr, OpDef, ReduceOp, ReduceStage, UnaryOp, input, konst, reduced};
use unpopped::kernel::check_kernel;
use unpopped::kernel_lift::{LiftDecline, lift_row_reduce_opdef};
use unpopped::oracle::kernel_eval::{KernelArg, run_kernel};
use unpopped::oracle::{TypedBuffer, evaluate};
use unpopped::plan::try_build_plan;
use unpopped_vocab::{ArchSku, ElementKind, OpCategory, OperandDesc, try_structure_key};

/// The PM-approved bound for tolerance mode (2026-10-29).
const MAX_ULPS: u64 = 2;

const VALUES: [f64; 13] = [
    1.0e8,
    1.0,
    -1.0e8,
    0.5,
    3.25,
    -2.0,
    7.0,
    0.1,
    -0.75,
    12.0,
    -5.5,
    0.3,
    2.0,
];

fn buffer(dtype: ElementKind, rows: i64, k: i64, f: impl Fn(usize) -> f64) -> TypedBuffer {
    let bytes: Vec<u8> = (0..(rows * k) as usize)
        .flat_map(|i| match dtype {
            ElementKind::F32 => (f(i) as f32).to_le_bytes().to_vec(),
            _ => f(i).to_le_bytes().to_vec(),
        })
        .collect();
    TypedBuffer::new(dtype, vec![rows, k], vec![k, 1], bytes)
}

/// Run `op` through the oracle and through the lifted kernel on `src`.
fn both(op: &OpDef, dtype: ElementKind, src: &TypedBuffer) -> (Vec<u8>, Vec<u8>) {
    let (rows, k) = (src.shape[0], src.shape[1]);
    let desc = OperandDesc::new(2, &[rows, k], &[k, 1], dtype, 256);
    let ops = [desc, desc];
    let key = try_structure_key(OpCategory::Softmax, &ops, ArchSku::Sm89).expect("a key");
    let plan = try_build_plan(op, &key).expect("the row reduction plans");
    let want = evaluate(&plan, &ops, std::slice::from_ref(src), &[])
        .remove(0)
        .bytes;

    let kernel = lift_row_reduce_opdef(op, dtype, 2).expect("the row reduction lifts");
    assert_eq!(check_kernel(&kernel), Ok(()), "a lifted kernel is well formed");
    let mut args = vec![
        KernelArg::Buffer(src.clone()),
        KernelArg::Buffer(TypedBuffer::new(
            dtype,
            vec![rows, k],
            vec![k, 1],
            vec![0xAB; want.len()],
        )),
        KernelArg::Int(i128::from(rows)),
        KernelArg::Int(i128::from(k)),
    ];
    run_kernel(&kernel, &mut args).expect("the kernel runs");
    let KernelArg::Buffer(got) = &args[1] else {
        unreachable!()
    };
    (want.clone(), got.bytes[..want.len()].to_vec())
}

/// Distance in representable `f32` values; NaN equals NaN, infinities must match exactly.
fn ulps_f32(a: f32, b: f32) -> u64 {
    if a.is_nan() || b.is_nan() {
        return if a.is_nan() && b.is_nan() { 0 } else { u64::MAX };
    }
    let key = |x: f32| {
        let bits = i64::from(x.to_bits() as i32);
        if bits < 0 { i64::from(i32::MIN) - bits } else { bits }
    };
    key(a).abs_diff(key(b))
}

fn max_ulps_f32(want: &[u8], got: &[u8]) -> u64 {
    let f = |b: &[u8]| -> Vec<f32> {
        b.chunks(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
            .collect()
    };
    f(want)
        .into_iter()
        .zip(f(got))
        .map(|(a, b)| ulps_f32(a, b))
        .max()
        .unwrap_or(0)
}

fn stage(pre: Expr, op: ReduceOp) -> ReduceStage {
    ReduceStage { pre: pre.0, op }
}

fn rr(dtype: ElementKind, stages: Vec<ReduceStage>, epilogue: Expr) -> OpDef {
    OpDef::row_reduce("rr", 1, &[dtype], stages, epilogue)
}

fn softmax(dtype: ElementKind) -> OpDef {
    rr(
        dtype,
        vec![
            stage(input(0), ReduceOp::Max),
            stage((input(0) - reduced(0)).exp(), ReduceOp::Sum),
        ],
        (input(0) - reduced(0)).exp() / reduced(1),
    )
}

fn centred_quotient(dtype: ElementKind) -> OpDef {
    rr(
        dtype,
        vec![
            stage(input(0), ReduceOp::Mean),
            stage(input(0) - reduced(0), ReduceOp::Max),
        ],
        (input(0) - reduced(0)) / reduced(1),
    )
}

fn rms_norm(dtype: ElementKind) -> OpDef {
    rr(
        dtype,
        vec![stage(input(0) * input(0), ReduceOp::Mean)],
        input(0) / (reduced(0) + konst(1.0e-3)).unary(UnaryOp::Sqrt),
    )
}

fn data(rows: i64, k: i64, dtype: ElementKind) -> TypedBuffer {
    buffer(dtype, rows, k, |i| VALUES[i % VALUES.len()] * (1.0 + (i / 13) as f64 * 0.37))
}

#[test]
fn identity_epilogues_are_byte_exact() {
    for stage_op in [ReduceOp::Sum, ReduceOp::Mean, ReduceOp::Max, ReduceOp::Min] {
        let op = rr(
            ElementKind::F32,
            vec![stage(input(0), stage_op)],
            reduced(0),
        );
        let (want, got) = both(&op, ElementKind::F32, &data(3, 13, ElementKind::F32));
        assert_eq!(got, want, "{stage_op:?}");
    }
}

#[test]
fn f64_compute_is_byte_exact_for_every_epilogue() {
    for op in [softmax(ElementKind::F64), centred_quotient(ElementKind::F64), rms_norm(ElementKind::F64)] {
        let (want, got) = both(&op, ElementKind::F64, &data(3, 13, ElementKind::F64));
        assert_eq!(got, want);
    }
}

#[test]
fn f32_epilogues_are_within_the_tolerance() {
    for (name, op) in [
        ("softmax", softmax(ElementKind::F32)),
        ("centred quotient", centred_quotient(ElementKind::F32)),
        ("rms norm", rms_norm(ElementKind::F32)),
    ] {
        let (want, got) = both(&op, ElementKind::F32, &data(5, 13, ElementKind::F32));
        let worst = max_ulps_f32(&want, &got);
        assert!(worst <= MAX_ULPS, "{name}: {worst} ulps apart (bound {MAX_ULPS})");
    }
}

#[test]
fn the_tolerance_is_not_vacuous() {
    // Control: rounding a stage value to f32 really moves the answer on this corpus, so the
    // tolerance cases above are not secretly byte-exact. Softmax's sum stage is the probe: the
    // oracle divides by the unrounded f64 sum, the kernel by its f32 rounding.
    let mut moved = false;
    for rows in 1..40 {
        let src = buffer(ElementKind::F32, rows, 11, |i| ((i * 7 + 3) % 19) as f64 * 0.37 - 2.0);
        let (want, got) = both(&softmax(ElementKind::F32), ElementKind::F32, &src);
        assert!(max_ulps_f32(&want, &got) <= MAX_ULPS);
        moved |= want != got;
    }
    assert!(moved, "some softmax row differs between the unrounded and rounded stage value");
}

#[test]
fn a_wide_accumulator_is_required() {
    // 1e8 + 1 - 1e8 in a row: an f32 accumulator loses the 1. Identity epilogue, so byte-exact.
    let op = rr(
        ElementKind::F32,
        vec![stage(input(0), ReduceOp::Sum)],
        reduced(0),
    );
    let src = buffer(ElementKind::F32, 1, 5, |i| [1.0e8, 1.0, -1.0e8, 0.5, 2.0][i]);
    let (want, got) = both(&op, ElementKind::F32, &src);
    assert_eq!(got, want);
    let v = f32::from_le_bytes(got[..4].try_into().unwrap());
    assert_eq!(v, 3.5, "the f64 sum is 3.5");
}

#[test]
fn nan_and_infinity_rows_agree() {
    let src = buffer(ElementKind::F32, 3, 4, |i| {
        [
            1.0,
            f64::NAN,
            2.0,
            3.0,
            f64::INFINITY,
            1.0,
            2.0,
            -f64::INFINITY,
            0.0,
            -0.0,
            1.0,
            2.0,
        ][i]
    });
    for op in [softmax(ElementKind::F32), centred_quotient(ElementKind::F32)] {
        let (want, got) = both(&op, ElementKind::F32, &src);
        assert!(max_ulps_f32(&want, &got) <= MAX_ULPS);
        let f = |b: &[u8]| -> Vec<bool> {
            b.chunks(4)
                .map(|c| f32::from_le_bytes(c.try_into().unwrap()).is_nan())
                .collect()
        };
        assert_eq!(f(&want), f(&got), "the same cells are NaN");
    }
}

#[test]
fn forms_the_lift_does_not_cover_decline_with_a_reason() {
    assert!(lift_row_reduce_opdef(&softmax(ElementKind::F32), ElementKind::F32, 2).is_ok());
    let ew = OpDef::elementwise("e", 1, &[ElementKind::F32], Expr(input(0).0));
    assert_eq!(
        lift_row_reduce_opdef(&ew, ElementKind::F32, 2),
        Err(LiftDecline::NotRowReduce)
    );
    assert_eq!(
        lift_row_reduce_opdef(&softmax(ElementKind::I32), ElementKind::I32, 2),
        Err(LiftDecline::UnsupportedForm("a non-float row reduction"))
    );
    let prod = rr(
        ElementKind::F32,
        vec![stage(input(0), ReduceOp::Prod)],
        reduced(0),
    );
    assert_eq!(
        lift_row_reduce_opdef(&prod, ElementKind::F32, 2),
        Err(LiftDecline::UnsupportedForm("a Prod or empty stage list"))
    );
}

#[test]
fn extremum_rows_that_never_cross_zero_agree() {
    // A Max over an all-negative row and a Min over an all-positive row: an accumulator that
    // starts at the sum identity (0) instead of the first element returns 0 for both.
    let neg = buffer(ElementKind::F32, 2, 4, |i| -1.0 - i as f64 * 0.5);
    let pos = buffer(ElementKind::F32, 2, 4, |i| 1.0 + i as f64 * 0.5);
    for (stage_op, src) in [(ReduceOp::Max, &neg), (ReduceOp::Min, &pos)] {
        let op = rr(
            ElementKind::F32,
            vec![stage(input(0), stage_op)],
            reduced(0),
        );
        let (want, got) = both(&op, ElementKind::F32, src);
        assert_eq!(got, want, "{stage_op:?}");
        let first = f32::from_le_bytes(got[..4].try_into().unwrap());
        assert_ne!(first, 0.0, "{stage_op:?}: the extremum is not the zero identity");
    }
}
