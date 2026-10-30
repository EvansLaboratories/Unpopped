// SPDX-License-Identifier: MIT OR Apache-2.0
//! **The emitted C agrees with the kernel interpreter on row reductions**
//! (`docs/e1-emit-from-the-kernel-layer.md`, stage E1b): lifted `Access::RowReduce` kernels (the
//! identity epilogues, softmax, a centred quotient, an RMS norm) are lowered by
//! `CpuC::lower_kernel`, compiled, run, and compared with `run_kernel`: bit-exact except NaN
//! sign/payload for the identity epilogues and `F64`, within 3 ulps for an `F32` epilogue that
//! computes with a stage value (tolerance mode). The serial C is a correctness reference; nothing
//! is timed.

use unpopped::backend::Backend;
use unpopped::ir::{Expr, OpDef, ReduceOp, ReduceStage, UnaryOp, input, konst, reduced};
use unpopped::kernel::{KernelFn, check_kernel};
use unpopped::kernel_lift::lift_row_reduce_opdef;
use unpopped::oracle::TypedBuffer;
use unpopped::oracle::kernel_eval::{KernelArg, run_kernel};
use unpopped_cpu_c::CpuC;
use unpopped_vocab::ElementKind;

#[path = "support/cc.rs"]
mod cc;
#[path = "support/kernel_data.rs"]
mod kernel_data;
#[path = "support/kernel_run.rs"]
mod kernel_run;
use kernel_data::zeroed;
use kernel_run::{F32, F64, agree, agree_within, run_c_mutated, ulps};

const VALUES: [f64; 13] = [
    1.0e8, 1.0, -1.0e8, 0.5, 3.25, -2.0, 7.0, 0.1, -0.75, 12.0, -5.5, 0.3, 2.0,
];

fn buffer(dtype: ElementKind, rows: i64, k: i64, f: impl Fn(usize) -> f64) -> TypedBuffer {
    let bytes = (0..(rows * k) as usize)
        .flat_map(|i| match dtype {
            ElementKind::F32 => (f(i) as f32).to_le_bytes().to_vec(),
            _ => f(i).to_le_bytes().to_vec(),
        })
        .collect();
    TypedBuffer::new(dtype, vec![rows, k], vec![k, 1], bytes)
}

fn args(src: TypedBuffer) -> Vec<KernelArg> {
    let (rows, k) = (src.shape[0], src.shape[1]);
    vec![
        KernelArg::Buffer(src.clone()),
        KernelArg::Buffer(zeroed(src.dtype, &[rows, k])),
        KernelArg::Int(i128::from(rows)),
        KernelArg::Int(i128::from(k)),
    ]
}

fn data(rows: i64, k: i64, dtype: ElementKind) -> TypedBuffer {
    buffer(dtype, rows, k, |i| {
        VALUES[i % VALUES.len()] * (1.0 + (i / 13) as f64 * 0.37)
    })
}

fn stage(pre: Expr, op: ReduceOp) -> ReduceStage {
    ReduceStage { pre: pre.0, op }
}

fn lifted(dtype: ElementKind, stages: Vec<ReduceStage>, epilogue: Expr) -> KernelFn {
    let op = OpDef::row_reduce("rr", 1, &[dtype], stages, epilogue);
    let k = lift_row_reduce_opdef(&op, dtype, 2).expect("the row reduction lifts");
    assert_eq!(check_kernel(&k), Ok(()));
    k
}

fn softmax(dtype: ElementKind) -> KernelFn {
    lifted(
        dtype,
        vec![
            stage(input(0), ReduceOp::Max),
            stage((input(0) - reduced(0)).exp(), ReduceOp::Sum),
        ],
        (input(0) - reduced(0)).exp() / reduced(1),
    )
}

fn centred_quotient(dtype: ElementKind) -> KernelFn {
    lifted(
        dtype,
        vec![
            stage(input(0), ReduceOp::Mean),
            stage(input(0) - reduced(0), ReduceOp::Max),
        ],
        (input(0) - reduced(0)) / reduced(1),
    )
}

fn rms_norm(dtype: ElementKind) -> KernelFn {
    lifted(
        dtype,
        vec![stage(input(0) * input(0), ReduceOp::Mean)],
        input(0) / (reduced(0) + konst(1.0e-3)).unary(UnaryOp::Sqrt),
    )
}

fn identity(dtype: ElementKind, op: ReduceOp) -> KernelFn {
    lifted(dtype, vec![stage(input(0), op)], reduced(0))
}

#[test]
fn identity_epilogues_agree_exactly() {
    for op in [ReduceOp::Sum, ReduceOp::Mean, ReduceOp::Max, ReduceOp::Min] {
        let k = identity(F32, op);
        agree(&format!("rr_id_{op:?}"), &k, args(data(3, 13, F32)));
    }
    let k = identity(F64, ReduceOp::Sum);
    agree("rr_id_f64", &k, args(data(3, 13, F64)));
}

#[test]
fn extremum_rows_that_never_cross_zero_agree() {
    // A Max over an all-negative row and a Min over an all-positive row: an accumulator that
    // starts at the sum identity (0) returns 0 for both.
    let neg = buffer(F32, 2, 4, |i| -1.0 - i as f64 * 0.5);
    let pos = buffer(F32, 2, 4, |i| 1.0 + i as f64 * 0.5);
    agree("rr_neg_max", &identity(F32, ReduceOp::Max), args(neg));
    agree("rr_pos_min", &identity(F32, ReduceOp::Min), args(pos));
}

#[test]
fn epilogues_agree_within_the_tolerance() {
    // F32 epilogues use a stage value the oracle keeps at f64: 3 ulps (as for the elementwise
    // differential). F64 is exact apart from the libm `exp`, which both sides share in value.
    for (name, k) in [
        ("softmax", softmax(F32)),
        ("centred", centred_quotient(F32)),
        ("rms", rms_norm(F32)),
    ] {
        agree_within(&format!("rr_{name}"), &k, args(data(5, 13, F32)), 3);
    }
    for (name, k) in [("softmax", softmax(F64)), ("rms", rms_norm(F64))] {
        agree_within(&format!("rr_{name}_f64"), &k, args(data(3, 13, F64)), 0);
    }
}

#[test]
fn nan_and_infinity_rows_agree() {
    let vals = [
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
    ];
    let src = buffer(F32, 3, 4, |i| vals[i]);
    for (name, k) in [
        ("softmax", softmax(F32)),
        ("centred", centred_quotient(F32)),
    ] {
        agree_within(&format!("rr_nan_{name}"), &k, args(src.clone()), 3);
    }
}

/// The accumulator names of the emitted C: every `double` local declared with the fold's
/// identity (`= (double)(... )` initialisers are the stage accumulators, listed in order).
fn doubles(k: &KernelFn) -> Vec<String> {
    let src = CpuC.lower_kernel(k).unwrap().source;
    src.lines()
        .filter_map(|l| l.trim_start().strip_prefix("double l"))
        .filter_map(|l| l.split(" = ").next().map(|n| format!("l{n}")))
        .collect()
}

fn out(tag: &str, k: &KernelFn, a: &[KernelArg], mutate: Option<&str>) -> Vec<f32> {
    let ran = run_c_mutated(tag, k, a, |s| match mutate {
        Some(n) => s.replace(&format!("double {n} = "), &format!("float {n} = ")),
        None => s,
    })
    .expect("a host C compiler");
    ran.outs[0]
        .chunks(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect()
}

#[test]
fn the_row_accumulator_is_f64_in_the_emitted_c() {
    // (1e8, 1, -1e8, 0.5, 2) summed: f64 gives 3.5, a float accumulator 2.5.
    let k = identity(F32, ReduceOp::Sum);
    let a = args(buffer(F32, 1, 5, |i| [1.0e8, 1.0, -1.0e8, 0.5, 2.0][i]));
    let acc = doubles(&k).remove(0);
    assert_eq!(
        out("rr_sum_acc", &k, &a, None)[0],
        3.5,
        "the emitted f64 sum"
    );
    assert_eq!(
        out("rr_sum_acc_f", &k, &a, Some(&acc))[0],
        2.5,
        "the mutant sums in float"
    );
}

#[test]
fn the_tolerance_does_not_hide_a_float_stage_accumulator() {
    // Softmax over [0, -18.42 x 100]: exp(-18.42) ~ 1e-8 is below half an ulp of 1.0, so a float
    // sum stays 1.0 where the f64 sum is 1 + 1e-6, and the outputs move by far more than 3 ulps.
    let k = softmax(F32);
    let row = |i: usize| if i == 0 { 0.0 } else { -18.420_680_743_952_367 };
    let a = args(buffer(F32, 1, 101, row));
    let mut interp = a.clone();
    run_kernel(&k, &mut interp).expect("the interpreter runs");
    let KernelArg::Buffer(b) = &interp[1] else {
        unreachable!()
    };
    let want = f32::from_le_bytes(b.bytes[..4].try_into().unwrap());
    let accs = doubles(&k);
    let worst = |m: Option<&str>| {
        ulps(
            out(&format!("rr_sm_{}", m.unwrap_or("none")), &k, &a, m)[0],
            want,
        )
    };
    assert!(worst(None) <= 3, "the emitted C is inside the bound");
    assert!(
        accs.iter().any(|n| worst(Some(n)) > 3),
        "some float accumulator leaves the bound: {accs:?}"
    );
}
