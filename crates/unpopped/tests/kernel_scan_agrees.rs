// SPDX-License-Identifier: MIT OR Apache-2.0
//! **The layers agree on scans** (`docs/ir-kernel-layer.md` section 7): a float `Access::Scan`
//! lifted into the kernel layer and run by the kernel interpreter produces, byte for byte, what
//! `oracle::evaluate` produces, for every monoid, in both directions, inclusive and exclusive.
//!
//! The oracle keeps an `f64` prefix from the monoid identity, writes it before the fold
//! (exclusive) or after it (inclusive), and gives `Max` / `Min` the NaN-keeping rule. The corpus
//! has f32-cancelling magnitudes, an all-negative row (a `Max` scan whose prefix started at `0`
//! instead of `-inf` would answer `0`), a NaN first and a NaN mid-row, infinities and signed zeros.

use unpopped::ir::{Expr, OpDef, ReduceOp, ScalarExpr, input};
use unpopped::kernel::check_kernel;
use unpopped::kernel_lift::{LiftDecline, lift_scan_opdef};
use unpopped::oracle::kernel_eval::{KernelArg, run_kernel};
use unpopped::oracle::{TypedBuffer, evaluate};
use unpopped::plan::try_build_plan;
use unpopped_vocab::{ArchSku, ElementKind, OpCategory, OperandDesc, structure_key};

const ROWS: i64 = 5;
const COLS: i64 = 6;

fn f32_bytes(vs: &[f32]) -> Vec<u8> {
    vs.iter().flat_map(|v| v.to_le_bytes()).collect()
}

fn corpus() -> Vec<f32> {
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
        f32::NAN,
        1.0,
        2.0,
        3.0,
        -4.0,
        5.0, //
        f32::INFINITY,
        f32::NAN,
        f32::NEG_INFINITY,
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

fn input_buffer() -> TypedBuffer {
    TypedBuffer::new(
        ElementKind::F32,
        vec![ROWS, COLS],
        vec![COLS, 1],
        f32_bytes(&corpus()),
    )
}

fn both(op: &OpDef) -> (Vec<u8>, Vec<u8>) {
    let inp = input_buffer();
    let a = OperandDesc::new(2, &inp.shape, &inp.strides, ElementKind::F32, 256);
    let key = structure_key(OpCategory::Scan, &[a, a], ArchSku::Sm89);
    let plan = try_build_plan(op, &key).expect("the scan plans");
    let want = evaluate(&plan, &[a, a], std::slice::from_ref(&inp), &[])
        .remove(0)
        .bytes;

    let kernel = lift_scan_opdef(op, ElementKind::F32, 2).expect("the scan lifts");
    assert_eq!(
        check_kernel(&kernel),
        Ok(()),
        "a lifted kernel is well formed"
    );
    let mut args = vec![
        KernelArg::Buffer(inp),
        KernelArg::Buffer(TypedBuffer::new(
            ElementKind::F32,
            vec![ROWS, COLS],
            vec![COLS, 1],
            vec![0u8; 4 * (ROWS * COLS) as usize],
        )),
        KernelArg::Int(i128::from(ROWS)),
        KernelArg::Int(i128::from(COLS)),
    ];
    run_kernel(&kernel, &mut args).expect("the kernel runs");
    let KernelArg::Buffer(out) = &args[1] else {
        unreachable!()
    };
    (want, out.bytes[..].to_vec())
}

const OPS: [ReduceOp; 4] = [ReduceOp::Sum, ReduceOp::Prod, ReduceOp::Max, ReduceOp::Min];

#[test]
fn every_monoid_agrees_in_every_direction_and_exclusivity() {
    for rop in OPS {
        for reverse in [false, true] {
            for exclusive in [false, true] {
                let op = OpDef::scan_simple("s", &[ElementKind::F32], rop, 1, reverse, exclusive);
                let (want, got) = both(&op);
                assert_eq!(got, want, "{rop:?} reverse={reverse} exclusive={exclusive}");
            }
        }
    }
}

#[test]
fn a_pre_map_over_the_input_agrees() {
    let body = ScalarExpr::Mul(Box::new(input(0).0), Box::new(input(0).0));
    let op = OpDef::scan(
        "sq",
        1,
        &[ElementKind::F32],
        ReduceOp::Sum,
        1,
        false,
        false,
        Expr(body),
        Expr(ScalarExpr::Reduced(0)),
    );
    let (want, got) = both(&op);
    assert_eq!(got, want);
}

#[test]
fn the_max_scan_starts_at_negative_infinity_not_zero() {
    // Row 1 is all negative: an inclusive Max scan is its running maximum, never 0.
    let op = OpDef::scan_simple("m", &[ElementKind::F32], ReduceOp::Max, 1, false, false);
    let (want, got) = both(&op);
    assert_eq!(got, want);
    let at =
        |i: usize| f32::from_le_bytes([got[4 * i], got[4 * i + 1], got[4 * i + 2], got[4 * i + 3]]);
    assert_eq!(at(COLS as usize), -1.0, "row 1, position 0");
    assert_eq!(
        at(COLS as usize + 5),
        -1.0,
        "row 1, position 5 (running max)"
    );
}

#[test]
fn an_exclusive_scan_writes_the_identity_first() {
    let op = OpDef::scan_simple("m", &[ElementKind::F32], ReduceOp::Max, 1, false, true);
    let (want, got) = both(&op);
    assert_eq!(got, want);
    let first = f32::from_le_bytes([got[0], got[1], got[2], got[3]]);
    assert_eq!(
        first,
        f32::NEG_INFINITY,
        "the first position holds the identity"
    );
}

#[test]
fn a_scan_kernel_survives_a_json_round_trip() {
    // The infinite identities are built from bit patterns, not `Const(inf)`, so the kernel
    // serializes (a JSON number cannot be infinite).
    for rop in OPS {
        let op = OpDef::scan_simple("s", &[ElementKind::F32], rop, 1, false, false);
        let kernel = lift_scan_opdef(&op, ElementKind::F32, 2).expect("lifts");
        let text = serde_json::to_string(&kernel).expect("serializes");
        let back: unpopped::kernel::KernelFn = serde_json::from_str(&text).expect("parses");
        assert_eq!(back, kernel, "{rop:?}");
    }
}

#[test]
fn forms_the_lift_does_not_cover_decline_with_a_reason() {
    let f32k = [ElementKind::F32];
    // Control: the plain cumulative sum lifts.
    let plain = OpDef::scan_simple("p", &f32k, ReduceOp::Sum, 1, false, false);
    assert!(lift_scan_opdef(&plain, ElementKind::F32, 2).is_ok());
    let ew = OpDef::elementwise("e", 1, &f32k, Expr(input(0).0));
    assert_eq!(
        lift_scan_opdef(&ew, ElementKind::F32, 2),
        Err(LiftDecline::NotScan)
    );
    // A scan over an outer axis.
    let outer = OpDef::scan_simple("o", &f32k, ReduceOp::Sum, 0, false, false);
    assert_eq!(
        lift_scan_opdef(&outer, ElementKind::F32, 2),
        Err(LiftDecline::UnsupportedForm(
            "a scan over a non-innermost axis"
        ))
    );
    // A fused post-expression lifts in tolerance mode (`kernel_scan_post_agrees.rs`).
    let post = OpDef::scan(
        "q",
        1,
        &f32k,
        ReduceOp::Sum,
        1,
        false,
        false,
        Expr(input(0).0),
        Expr(ScalarExpr::Unary(
            unpopped::ir::UnaryOp::Sqrt,
            Box::new(ScalarExpr::Reduced(0)),
        )),
    );
    assert!(lift_scan_opdef(&post, ElementKind::F32, 2).is_ok());
    // An integer scan.
    let int = OpDef::scan_simple("i", &[ElementKind::I32], ReduceOp::Sum, 1, false, false);
    assert_eq!(
        lift_scan_opdef(&int, ElementKind::I32, 2),
        Err(LiftDecline::UnsupportedForm("a non-float scan"))
    );
}
