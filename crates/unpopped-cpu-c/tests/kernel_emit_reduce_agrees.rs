// SPDX-License-Identifier: MIT OR Apache-2.0
//! **The emitted C agrees with the kernel interpreter on reductions**
//! (`docs/e1-emit-from-the-kernel-layer.md`, stage E1b): a lifted `Access::Reduction` is lowered
//! by `CpuC::lower_kernel`, compiled, run, and its output bytes compared with `run_kernel`.
//!
//! The corpus has NaN, signed zeros, infinities and a row that cancels in `f32` but not in `f64`:
//! friendly numbers would pass an emitter that folded in `float`, dropped the NaN rule or
//! reordered the fold. The serial C is a correctness reference; nothing here times anything.

use unpopped::backend::Backend;
use unpopped::ir::{Expr, OpDef, ReduceOp, ScalarExpr, UnaryOp, input};
use unpopped::kernel::{KernelFn, check_kernel};
use unpopped::kernel_lift::{lift_reduction_opdef, lift_scan_opdef};
use unpopped::oracle::TypedBuffer;
use unpopped::oracle::kernel_eval::{KernelArg, run_kernel};
use unpopped_cpu_c::CpuC;
use unpopped_vocab::{AxisMask, ElementKind};

#[path = "support/cc.rs"]
mod cc;
#[path = "support/kernel_run.rs"]
mod kernel_run;
use kernel_run::{F32, F64, agree, agree_within, run_c_mutated, ulps, zeroed};

const ROWS: i64 = 5;
const COLS: i64 = 6;

/// Row 0 cancels in f32 (1e8 + 1 - 1e8) but not f64; row 1 is all negative; row 2 has a NaN
/// first, row 3 one in the middle; row 4 is signed zeros.
fn corpus() -> Vec<f64> {
    vec![
        1.0e8,
        1.0,
        -1.0e8,
        0.5,
        2.0,
        -3.0, //
        -1.0,
        -2.0,
        -3.0,
        -4.0,
        -5.0,
        -6.0, //
        f64::NAN,
        1.0,
        2.0,
        3.0,
        -4.0,
        5.0, //
        f64::INFINITY,
        f64::NAN,
        f64::NEG_INFINITY,
        2.0,
        0.1,
        0.2, //
        0.0,
        -0.0,
        0.0,
        -0.0,
        0.0,
        -0.0,
    ]
}

fn input_buffer(dtype: ElementKind) -> TypedBuffer {
    let bytes = corpus()
        .iter()
        .flat_map(|&v| {
            if dtype == F64 {
                v.to_le_bytes().to_vec()
            } else {
                (v as f32).to_le_bytes().to_vec()
            }
        })
        .collect();
    TypedBuffer::new(dtype, vec![ROWS, COLS], vec![COLS, 1], bytes)
}

fn lifted(rop: ReduceOp, axis0: bool, keepdim: bool, dtype: ElementKind) -> (KernelFn, Vec<i64>) {
    let mut axes = AxisMask::EMPTY;
    if axis0 {
        axes.set(0);
    }
    let op = OpDef::reduction_axes("r", 1, &[dtype], Expr(input(0).0), rop, axes, keepdim);
    let k = lift_reduction_opdef(&op, dtype, 2).expect("the reduction lifts");
    assert_eq!(check_kernel(&k), Ok(()));
    let out = match (axis0, keepdim) {
        (false, false) => vec![ROWS],
        (false, true) => vec![ROWS, 1],
        (true, false) => vec![COLS],
        (true, true) => vec![1, COLS],
    };
    (k, out)
}

fn args(dtype: ElementKind, out_shape: &[i64]) -> Vec<KernelArg> {
    vec![
        KernelArg::Buffer(input_buffer(dtype)),
        KernelArg::Buffer(zeroed(dtype, out_shape)),
        KernelArg::Int(i128::from(ROWS)),
        KernelArg::Int(i128::from(COLS)),
    ]
}

const OPS: [ReduceOp; 5] = [
    ReduceOp::Sum,
    ReduceOp::Prod,
    ReduceOp::Max,
    ReduceOp::Min,
    ReduceOp::Mean,
];

#[test]
fn every_reduce_op_agrees() {
    // Last axis plain and outer axis with keepdim cover both loop nests and both output ranks;
    // each compile is seconds, so the matrix stops there.
    for rop in OPS {
        for (dtype, axis0, keep) in [(F32, false, false), (F32, true, true), (F64, false, false)] {
            let (k, out) = lifted(rop, axis0, keep, dtype);
            agree(
                &format!("red_{dtype:?}_{rop:?}_{axis0}_{keep}"),
                &k,
                args(dtype, &out),
            );
        }
    }
}

#[test]
fn the_accumulator_is_f64_not_f32_in_the_emitted_c() {
    // Row 0 is (1e8, 1, -1e8, 0.5, 2, -3): the f64 fold gives 0.5, an f32 fold -0.5. The mutant
    // is the emitted C with its accumulator declared `float`: it must give the f32 answer, or the
    // agreement above could not have noticed a narrower fold. (The kernel checker rejects an F32
    // accumulator, so the mutation is made on the generated source.)
    let (k, out) = lifted(ReduceOp::Sum, false, false, F32);
    let acc = "l5 = "; // extents, counters, one loaded input, then the accumulator
    let src = CpuC.lower_kernel(&k).unwrap().source;
    let decl = format!("double {acc}");
    assert_eq!(
        src.matches(&decl).count(),
        1,
        "the accumulator is a double:
{src}"
    );
    let row0 = |m: bool| {
        let ran = run_c_mutated("acc", &k, &args(F32, &out), |s| {
            if m {
                s.replace(&decl, &format!("float {acc}"))
            } else {
                s
            }
        })
        .expect("a host C compiler");
        f32::from_le_bytes(ran.outs[0][..4].try_into().unwrap())
    };
    let f32_fold = ((((1.0e8_f32 + 1.0) + -1.0e8) + 0.5) + 2.0) + -3.0;
    assert_eq!(row0(false), 0.5, "the emitted f64 fold");
    assert_eq!(row0(true), f32_fold, "the mutant folds in float");
}

fn scan_kernel(op: &OpDef, dtype: ElementKind) -> KernelFn {
    let k = lift_scan_opdef(op, dtype, 2).expect("the scan lifts");
    assert_eq!(check_kernel(&k), Ok(()));
    k
}

fn scan_args(dtype: ElementKind) -> Vec<KernelArg> {
    args(dtype, &[ROWS, COLS])
}

#[test]
fn every_scan_monoid_agrees() {
    // Forward-inclusive and reverse-exclusive bracket the two directions and both write orders.
    for rop in [ReduceOp::Sum, ReduceOp::Prod, ReduceOp::Max, ReduceOp::Min] {
        for (reverse, exclusive) in [(false, false), (true, true)] {
            let op = OpDef::scan_simple("s", &[F32], rop, 1, reverse, exclusive);
            agree(
                &format!("scan_{rop:?}_{reverse}_{exclusive}"),
                &scan_kernel(&op, F32),
                scan_args(F32),
            );
        }
    }
    let op = OpDef::scan_simple("s", &[F64], ReduceOp::Sum, 1, false, false);
    agree("scan_f64", &scan_kernel(&op, F64), scan_args(F64));
}

fn c(v: f64) -> Box<ScalarExpr> {
    Box::new(ScalarExpr::Const(v))
}

/// `|x| * 0.1 + 0.25` over the (rounded) fold: two roundings in `float` C, one in the oracle.
fn post(x: ScalarExpr) -> Expr {
    let abs = ScalarExpr::Unary(UnaryOp::Abs, Box::new(x));
    Expr(ScalarExpr::Add(
        Box::new(ScalarExpr::Mul(Box::new(abs), c(f64::from(0.1_f32)))),
        c(0.25),
    ))
}

#[test]
fn fused_posts_agree_within_the_tolerance() {
    // The post is compound, so F32 is tolerance mode (<= 3 ulps, as in the E1a elementwise
    // differential); F64 stays bit-exact except NaN sign/payload.
    for dtype in [F32, F64] {
        let tol = if dtype == F32 { 3 } else { 0 };
        let red = OpDef::reduction_post(
            "rp",
            1,
            &[dtype],
            Expr(input(0).0),
            ReduceOp::Sum,
            post(ScalarExpr::Reduced(0)),
        );
        let k = lift_reduction_opdef(&red, dtype, 2).expect("the reduction post lifts");
        agree_within(&format!("redpost_{dtype:?}"), &k, args(dtype, &[ROWS]), tol);
        let scan = OpDef::scan(
            "sp",
            1,
            &[dtype],
            ReduceOp::Sum,
            1,
            false,
            false,
            Expr(input(0).0),
            post(ScalarExpr::Reduced(0)),
        );
        agree_within(
            &format!("scanpost_{dtype:?}"),
            &scan_kernel(&scan, dtype),
            scan_args(dtype),
            tol,
        );
    }
}

#[test]
fn the_scan_prefix_is_f64_not_f32_in_the_emitted_c() {
    // Row 0 inclusive Sum: after (1e8, 1, -1e8) the f64 prefix is 1.0, the f32 prefix 0.0.
    let op = OpDef::scan_simple("s", &[F32], ReduceOp::Sum, 1, false, false);
    let k = scan_kernel(&op, F32);
    let src = CpuC.lower_kernel(&k).unwrap().source;
    assert_eq!(
        src.matches("double l5 = ").count(),
        1,
        "the prefix is a double:
{src}"
    );
    let third = |mutate: bool| {
        let ran = run_c_mutated("prefix", &k, &scan_args(F32), |s| {
            if mutate {
                s.replace("double l5 = ", "float l5 = ")
            } else {
                s
            }
        })
        .expect("a host C compiler");
        f32::from_le_bytes(ran.outs[0][8..12].try_into().unwrap())
    };
    assert_eq!(third(false), 1.0, "the emitted f64 prefix");
    assert_eq!(third(true), 0.0, "the mutant carries a float prefix");
}

#[test]
fn the_tolerance_does_not_hide_a_float_prefix_under_a_post() {
    // The post's tolerance is 3 ulps; a float prefix under it is off by far more at element 2 of
    // row 0 (f64 prefix 1.0 gives 0.35, float prefix 0.0 gives 0.25), so the bound is not vacuous.
    let scan = OpDef::scan(
        "sp",
        1,
        &[F32],
        ReduceOp::Sum,
        1,
        false,
        false,
        Expr(input(0).0),
        post(ScalarExpr::Reduced(0)),
    );
    let k = scan_kernel(&scan, F32);
    let mut interp = scan_args(F32);
    run_kernel(&k, &mut interp).expect("the interpreter runs");
    let KernelArg::Buffer(b) = &interp[1] else {
        unreachable!()
    };
    let want = f32::from_le_bytes(b.bytes[8..12].try_into().unwrap());
    let third = |mutate: bool| {
        let ran = run_c_mutated("post_prefix", &k, &scan_args(F32), |s| {
            if mutate {
                s.replace("double l5 = ", "float l5 = ")
            } else {
                s
            }
        })
        .expect("a host C compiler");
        f32::from_le_bytes(ran.outs[0][8..12].try_into().unwrap())
    };
    assert!(
        ulps(third(false), want) <= 3,
        "the emitted C is inside the bound"
    );
    assert!(ulps(third(true), want) > 3, "the mutant is outside it");
}
