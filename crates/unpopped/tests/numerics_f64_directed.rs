// SPDX-License-Identifier: MIT OR Apache-2.0
//! **Directed rounding at f64** (ROADMAP U43): `numerics_f64::directed_f64` against the independent Python exact-rational reference
//! (`tools/gen_f64_directed_vectors.py`, rows in `tests/data/numerics_f64_directed.txt`), plus the host's own rounding as a
//! cross-check where the host is correct (round to nearest equals the directed result when the exact value is on the grid).
//! Every class the reference visits (subnormal results, overflow, exact cancellation, signed zero) is counted and asserted non-empty,
//! so a corpus that stopped reaching one fails here rather than passing blind.

use unpopped::ir::{Numerics, QualOp, RoundMode};
use unpopped::numerics_f64::{directed_f64, eval_qualified_f64};

const DATA: &str = include_str!("data/numerics_f64_directed.txt");

struct Row {
    op: QualOp,
    mode: RoundMode,
    args: Vec<f64>,
    want: u64,
}

fn hex(s: &str) -> u64 {
    u64::from_str_radix(s, 16).expect("a hex word")
}

fn rows() -> Vec<Row> {
    DATA.lines()
        .map(|l| {
            let f: Vec<&str> = l.split_whitespace().collect();
            let op = match f[0] {
                "add" => QualOp::Add,
                "sub" => QualOp::Sub,
                "mul" => QualOp::Mul,
                "div" => QualOp::Div,
                "fma" => QualOp::Fma,
                "sqrt" => QualOp::Sqrt,
                other => panic!("unknown op {other}"),
            };
            let mode = match f[1] {
                "rz" => RoundMode::TowardZero,
                "rd" => RoundMode::TowardNegInf,
                "ru" => RoundMode::TowardPosInf,
                other => panic!("unknown mode {other}"),
            };
            let all: Vec<f64> = f[2..5].iter().map(|w| f64::from_bits(hex(w))).collect();
            Row {
                op,
                mode,
                args: all[..op.arity()].to_vec(),
                want: hex(f[5]),
            }
        })
        .collect()
}

#[test]
fn every_row_matches_the_independent_reference_bit_for_bit() {
    let rs = rows();
    assert!(rs.len() > 600, "the corpus is {} rows", rs.len());
    let mut bad = Vec::new();
    for r in &rs {
        let got = directed_f64(r.op, r.mode, &r.args).to_bits();
        if got != r.want {
            bad.push(format!(
                "{:?} {:?} {:016x?} got {got:016x} want {:016x}",
                r.op,
                r.mode,
                r.args.iter().map(|a| a.to_bits()).collect::<Vec<_>>(),
                r.want
            ));
        }
    }
    assert!(
        bad.is_empty(),
        "{} of {} rows differ, first: {:?}",
        bad.len(),
        rs.len(),
        &bad[..bad.len().min(3)]
    );
}

#[test]
fn the_corpus_reaches_each_hard_class() {
    let (mut subnormal, mut overflow, mut zero_out, mut neg_zero, mut inexact_dir) =
        (0, 0, 0, 0, 0);
    for r in rows() {
        let w = f64::from_bits(r.want);
        subnormal += usize::from(w.is_subnormal());
        overflow += usize::from(w.is_infinite() || w.abs() == f64::MAX);
        zero_out += usize::from(w == 0.0);
        neg_zero += usize::from(w == 0.0 && w.is_sign_negative());
        let near = directed_nearest(r.op, &r.args);
        inexact_dir += usize::from(near.to_bits() != r.want && near.is_finite());
    }
    assert!(subnormal >= 20, "subnormal results: {subnormal}");
    assert!(overflow >= 8, "overflow-region results: {overflow}");
    assert!(zero_out >= 20, "zero results: {zero_out}");
    assert!(neg_zero >= 4, "negative-zero results: {neg_zero}");
    assert!(
        inexact_dir >= 150,
        "rows where a directed result differs from round-to-nearest: {inexact_dir}"
    );
}

fn directed_nearest(op: QualOp, a: &[f64]) -> f64 {
    eval_qualified_f64(op, Numerics::default(), a)
}

#[test]
fn nearest_even_is_the_host_operation_and_a_directed_mode_brackets_it() {
    for r in rows() {
        let near = directed_nearest(r.op, &r.args);
        let got = f64::from_bits(r.want);
        if !near.is_finite() || !got.is_finite() {
            continue;
        }
        match r.mode {
            RoundMode::TowardPosInf => assert!(
                got >= near,
                "{:?} ru {:?}: {got} < nearest {near}",
                r.op,
                r.args
            ),
            RoundMode::TowardNegInf => assert!(
                got <= near,
                "{:?} rd {:?}: {got} > nearest {near}",
                r.op,
                r.args
            ),
            _ => assert!(
                got.abs() <= near.abs(),
                "{:?} rz {:?}: |{got}| > |nearest {near}|",
                r.op,
                r.args
            ),
        }
    }
}

#[test]
fn a_directed_result_is_within_one_neighbour_of_the_nearest_and_exact_results_agree() {
    // 1 + 2^-60: nearest is 1, ru is the next f64 up, rd and rz are 1
    let (one, tiny) = (1.0f64, 2f64.powi(-60));
    let up = directed_f64(QualOp::Add, RoundMode::TowardPosInf, &[one, tiny]);
    assert_eq!(up, f64::from_bits(1.0f64.to_bits() + 1));
    for m in [RoundMode::TowardZero, RoundMode::TowardNegInf] {
        assert_eq!(directed_f64(QualOp::Add, m, &[one, tiny]), one);
    }
    // an exact operation has one answer in every mode
    for m in [
        RoundMode::TowardZero,
        RoundMode::TowardNegInf,
        RoundMode::TowardPosInf,
    ] {
        assert_eq!(directed_f64(QualOp::Mul, m, &[3.0, 0.5]), 1.5);
        assert_eq!(directed_f64(QualOp::Sqrt, m, &[4.0]), 2.0);
        assert_eq!(directed_f64(QualOp::Div, m, &[1.0, 8.0]), 0.125);
    }
}

#[test]
fn overflow_follows_ieee_754_7_4_in_each_direction() {
    let big = f64::MAX;
    let want = |m, neg: bool| match (m, neg) {
        (RoundMode::TowardZero, false) => f64::MAX,
        (RoundMode::TowardZero, true) => -f64::MAX,
        (RoundMode::TowardPosInf, false) => f64::INFINITY,
        (RoundMode::TowardPosInf, true) => -f64::MAX,
        (_, false) => f64::MAX,
        (_, true) => f64::NEG_INFINITY,
    };
    for m in [
        RoundMode::TowardZero,
        RoundMode::TowardNegInf,
        RoundMode::TowardPosInf,
    ] {
        assert_eq!(
            directed_f64(QualOp::Add, m, &[big, big]),
            want(m, false),
            "{m:?}"
        );
        assert_eq!(
            directed_f64(QualOp::Add, m, &[-big, -big]),
            want(m, true),
            "{m:?}"
        );
        assert_eq!(
            directed_f64(QualOp::Mul, m, &[big, 2.0]),
            want(m, false),
            "{m:?}"
        );
    }
}

#[test]
fn non_finite_operands_and_zero_divisors_follow_the_host() {
    for m in [
        RoundMode::TowardZero,
        RoundMode::TowardNegInf,
        RoundMode::TowardPosInf,
    ] {
        assert!(directed_f64(QualOp::Add, m, &[f64::NAN, 1.0]).is_nan());
        assert_eq!(
            directed_f64(QualOp::Add, m, &[f64::INFINITY, 1.0]),
            f64::INFINITY
        );
        assert!(directed_f64(QualOp::Mul, m, &[f64::INFINITY, 0.0]).is_nan());
        assert_eq!(directed_f64(QualOp::Div, m, &[1.0, 0.0]), f64::INFINITY);
        assert!(directed_f64(QualOp::Sqrt, m, &[-1.0]).is_nan());
        assert_eq!(
            directed_f64(QualOp::Sqrt, m, &[f64::INFINITY]),
            f64::INFINITY
        );
    }
}
