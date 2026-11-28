// SPDX-License-Identifier: MIT OR Apache-2.0
//! `F128` as a kernel dtype (ROADMAP E3 slice 2b): the oracle runs a kernel over binary128 buffers with `soft_f128`. Each operation
//! is a row checked against the independent reference of `tests/data/soft_f128_vectors.txt` (`tools/gen_f128_vectors.py`, **UNVERIFIED
//! against a third-party binary128 library**), through a kernel (one thread per row) rather than by calling `soft_f128`, so the
//! wiring (load, the node, the store, the cast) is what is under test. The refusals are by name, never a panic.

use unpopped::exec::LaunchShape;
use unpopped::ir::{BinaryOp, ScalarExpr, UnaryOp};
use unpopped::kernel::{Block, BufferRole, KernelFn, LocalDecl, Param, Stmt, check_kernel};
use unpopped::oracle::kernel_eval::KernelArg;
use unpopped::oracle::kernel_launch::run_launch;
use unpopped_vocab::ElementKind::{F32, F128};

#[path = "support/f128.rs"]
#[allow(
    dead_code,
    reason = "each F128 test file uses some of the shared helpers"
)]
mod support;
use support::*;

fn check_op(op: &str, f: impl Fn(ScalarExpr, ScalarExpr) -> ScalarExpr) {
    let rows = rows(op);
    assert!(rows.len() > 150, "{op}: {} rows", rows.len());
    let (a, b): (Vec<u128>, Vec<u128>) = rows.iter().map(|r| (r.0, r.1)).unzip();
    let got = run_binary(&binary(rows.len(), f), &a, &b);
    let bad: Vec<String> = rows
        .iter()
        .zip(&got)
        .filter(|(r, g)| format!("{g:x}") != r.2)
        .map(|(r, g)| format!("{op} {:x} {:x}: got {g:x}, want {}", r.0, r.1, r.2))
        .collect();
    assert!(
        bad.is_empty(),
        "{} of {} differ, first: {:?}",
        bad.len(),
        rows.len(),
        &bad[..bad.len().min(3)]
    );
}

#[test]
fn add_sub_mul_div_run_through_a_kernel_against_the_independent_reference() {
    check_op("add", |x, y| ScalarExpr::Add(bx(x), bx(y)));
    check_op("sub", |x, y| ScalarExpr::Sub(bx(x), bx(y)));
    check_op("mul", |x, y| ScalarExpr::Mul(bx(x), bx(y)));
    check_op("div", |x, y| ScalarExpr::Div(bx(x), bx(y)));
}

#[test]
fn a_copy_moves_the_bytes_verbatim_including_a_nan_payload() {
    let odd_nan = 0x7fff_8000_0000_0000_0000_0000_dead_beef_u128;
    let a = [ONE, odd_nan, 1, 1 << 127];
    let k = binary(a.len(), |x, _| x);
    assert_eq!(run_binary(&k, &a, &a), a.to_vec());
}

#[test]
fn neg_and_abs_edit_the_sign_only() {
    let a = [
        ONE,
        1 << 127,
        0,
        0x7fff_u128 << 112,
        0xffff_u128 << 112,
        0x7fff_8000_0000_0000_0000_0000_0000_0001,
    ];
    let neg = run_binary(
        &binary(a.len(), |x, _| ScalarExpr::Unary(UnaryOp::Neg, bx(x))),
        &a,
        &a,
    );
    let abs = run_binary(
        &binary(a.len(), |x, _| ScalarExpr::Unary(UnaryOp::Abs, bx(x))),
        &a,
        &a,
    );
    let sign = 1u128 << 127;
    assert_eq!(neg, a.iter().map(|v| v ^ sign).collect::<Vec<_>>());
    assert_eq!(abs, a.iter().map(|v| v & !sign).collect::<Vec<_>>());
}

#[test]
fn comparisons_follow_ieee_with_nan_unordered_and_zeros_equal() {
    let nan = 0x7fff_8000_0000_0000_0000_0000_0000_0000_u128;
    let (one, two, minus_one) = (ONE, 0x4000_u128 << 112, ONE | 1 << 127);
    let (zero, minus_zero) = (0u128, 1u128 << 127);
    let (inf, minus_inf) = (0x7fff_u128 << 112, 0xffff_u128 << 112);
    // (a, b, [eq, ne, lt, le, gt, ge])
    let cases: &[(u128, u128, [bool; 6])] = &[
        (one, two, [false, true, true, true, false, false]),
        (two, one, [false, true, false, false, true, true]),
        (one, one, [true, false, false, true, false, true]),
        (zero, minus_zero, [true, false, false, true, false, true]),
        (minus_one, one, [false, true, true, true, false, false]),
        (nan, one, [false, true, false, false, false, false]),
        (one, nan, [false, true, false, false, false, false]),
        (nan, nan, [false, true, false, false, false, false]),
        (minus_inf, inf, [false, true, true, true, false, false]),
        (inf, inf, [true, false, false, true, false, true]),
    ];
    let ops = [
        BinaryOp::CmpEq,
        BinaryOp::CmpNe,
        BinaryOp::CmpLt,
        BinaryOp::CmpLe,
        BinaryOp::CmpGt,
        BinaryOp::CmpGe,
    ];
    let (a, b): (Vec<u128>, Vec<u128>) = cases.iter().map(|c| (c.0, c.1)).unzip();
    for (i, op) in ops.into_iter().enumerate() {
        let k = binary(a.len(), move |x, y| ScalarExpr::Binary(op, bx(x), bx(y)));
        let got = run_binary(&k, &a, &b);
        let want: Vec<u128> = cases.iter().map(|c| if c.2[i] { ONE } else { 0 }).collect();
        assert_eq!(got, want, "{op:?}");
    }
}

#[test]
fn select_takes_one_arm_by_an_f128_comparison_or_an_f128_condition() {
    let (a, b) = (
        [ONE, 0x4000_u128 << 112, 5, 7],
        [0x4000_u128 << 112, ONE, 5, 3],
    );
    // max-like: a < b ? b : a
    let k = binary(a.len(), |x, y| {
        ScalarExpr::Select(
            bx(ScalarExpr::Binary(
                BinaryOp::CmpLt,
                bx(x.clone()),
                bx(y.clone()),
            )),
            bx(y),
            bx(x),
        )
    });
    assert_eq!(run_binary(&k, &a, &b), vec![b[0], a[1], a[2], a[3]]);
    // an F128 condition is taken when it is not zero (-0 is zero; a NaN is taken)
    let cond = [
        0u128,
        1 << 127,
        ONE,
        0x7fff_8000_0000_0000_0000_0000_0000_0000,
    ];
    let k = binary(cond.len(), |c, x| {
        ScalarExpr::Select(bx(c), bx(x), bx(seven_f128()))
    });
    let seven = run_binary(&binary(1, |_, _| seven_f128()), &[0], &[0])[0];
    let got = run_binary(&k, &cond, &[ONE; 4]);
    assert_eq!(got, vec![seven, seven, ONE, ONE]);
}

#[test]
fn an_f128_scalar_argument_is_the_exact_binary128_of_the_f64_given() {
    // out[t] = s * s with s = 1.5: 2.25 = 1.125 * 2^1, exponent 0x4000, the fraction bit 2^-3
    let k = KernelFn::new(
        "scalar",
        F32,
        vec![
            Param::scalar("s", F128, 0),
            Param::buffer("out", F128, BufferRole::Write, Some(1)),
        ],
        vec![LocalDecl::scalar(F128)],
        Block(vec![Stmt::Store {
            place: at(1),
            value: ScalarExpr::Mul(bx(ScalarExpr::Local(0)), bx(ScalarExpr::Local(0))),
        }]),
    )
    .with_launch(LaunchShape::new([2, 1, 1], [1, 1, 1]));
    assert_eq!(check_kernel(&k), Ok(()));
    let mut args = vec![KernelArg::Float(1.5), buf(F128, vec![0; 32], 2)];
    run_launch(&k, &mut args).unwrap_or_else(|e| panic!("{e:?}"));
    let want = (0x4000u128 << 112) | (1 << 109);
    assert_eq!(words(&out_bytes(&args, 1)), vec![want, want]);
}
