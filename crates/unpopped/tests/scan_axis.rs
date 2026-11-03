// SPDX-License-Identifier: MIT OR Apache-2.0
//! **A scan over any axis** (ROADMAP U2 D5, scan first; `docs/options-u2-lift-declines.md`): an `Access::Scan` whose axis
//! is not the innermost. The tensor splits as `(outer, k, inner)` around the scanned axis, and element `j` of a "row" sits
//! `inner` apart in the dense layout; every input must be full-width dense (no broadcast operand), which the plan gate
//! checks. The oracle is held to a scan written here in Rust over explicit coordinates; the lifted kernel (its counters
//! walk every axis, the scanned one innermost of the loop nest) is held to the oracle byte for byte; the launched lift
//! still scans the innermost axis only and declines the others, typed.
//!
//! The tensor is `[2, 3, 4]`, so axis 0 has `k = 2`, `inner = 12`; axis 1 has `k = 3`, `outer = 2`, `inner = 4`; axis 2 is
//! the old innermost walk (the control: `inner = 1`). A `Max` scan carries a NaN, and an exclusive scan its identity.

use unpopped::ir::{OpDef, ReduceOp};
use unpopped::kernel::check_kernel;
use unpopped::kernel_lift::{LiftDecline, lift_scan_opdef, lift_scan_opdef_launched};
use unpopped::oracle::kernel_eval::{KernelArg, run_kernel};
use unpopped::oracle::{TypedBuffer, evaluate};
use unpopped::plan::try_build_plan;
use unpopped_vocab::{ArchSku, ElementKind, OpCategory, OperandDesc, structure_key};

const SHAPE: [usize; 3] = [2, 3, 4];
const N: usize = 24;
const F64: ElementKind = ElementKind::F64;

fn values() -> Vec<f64> {
    (0..N)
        .map(|i| match i {
            5 => f64::NAN,
            _ => ((i * 7 + 3) % 11) as f64 - 4.5,
        })
        .collect()
}

fn buffer(vals: &[f64]) -> TypedBuffer {
    let bytes = vals.iter().flat_map(|v| v.to_le_bytes()).collect();
    TypedBuffer::new(F64, vec![2, 3, 4], vec![12, 4, 1], bytes)
}

fn op(rop: ReduceOp, axis: u8, reverse: bool, exclusive: bool) -> OpDef {
    OpDef::scan_simple("s", &[F64], rop, axis, reverse, exclusive)
}

fn oracle(o: &OpDef, vals: &[f64]) -> Vec<f64> {
    let src = buffer(vals);
    let a = OperandDesc::new(3, &src.shape, &src.strides, F64, 256);
    let key = structure_key(OpCategory::Scan, &[a, a], ArchSku::Sm89);
    let plan = try_build_plan(o, &key).expect("the scan plans");
    f64s(
        &evaluate(&plan, &[a, a], std::slice::from_ref(&src), &[])
            .remove(0)
            .bytes,
    )
}

fn lifted(o: &OpDef, vals: &[f64]) -> Vec<f64> {
    let kernel = lift_scan_opdef(o, F64, 3).expect("the scan lifts");
    assert_eq!(check_kernel(&kernel), Ok(()));
    let out = TypedBuffer::new(F64, vec![2, 3, 4], vec![12, 4, 1], vec![0xAB; 8 * N]);
    let mut args = vec![KernelArg::Buffer(buffer(vals)), KernelArg::Buffer(out)];
    args.extend(SHAPE.iter().map(|&e| KernelArg::Int(e as i128)));
    run_kernel(&kernel, &mut args).expect("the kernel runs");
    let KernelArg::Buffer(got) = &args[1] else {
        unreachable!()
    };
    f64s(&got.bytes)
}

fn f64s(b: &[u8]) -> Vec<f64> {
    b.chunks(8)
        .map(|c| f64::from_le_bytes(c.try_into().unwrap()))
        .collect()
}

fn same(a: &[f64], b: &[f64]) -> bool {
    a.len() == b.len()
        && a.iter()
            .zip(b)
            .all(|(x, y)| x.to_bits() == y.to_bits() || (x.is_nan() && y.is_nan()))
}

/// The reference: for each output coordinate, fold the elements along `axis` in scan order, with explicit coordinates.
fn reference(rop: ReduceOp, axis: usize, reverse: bool, exclusive: bool, vals: &[f64]) -> Vec<f64> {
    let at = |c: [usize; 3]| vals[(c[0] * SHAPE[1] + c[1]) * SHAPE[2] + c[2]];
    let mut out = vec![0.0; N];
    for a in 0..SHAPE[0] {
        for b in 0..SHAPE[1] {
            for c in 0..SHAPE[2] {
                let here = [a, b, c];
                let k = SHAPE[axis];
                let mine = here[axis];
                // the positions folded before (exclusive) or up to (inclusive) this one, in scan order
                let upto = if reverse { k - 1 - mine } else { mine };
                let stop = if exclusive { upto } else { upto + 1 };
                let mut acc: Option<f64> = None;
                for step in 0..stop {
                    let j = if reverse { k - 1 - step } else { step };
                    let mut p = here;
                    p[axis] = j;
                    let x = at(p);
                    acc = Some(match (acc, rop) {
                        (None, _) => x,
                        (Some(s), ReduceOp::Sum) => s + x,
                        (Some(s), _) => {
                            if x.is_nan() || s.is_nan() || x > s {
                                if s.is_nan() { s } else { x }
                            } else {
                                s
                            }
                        }
                    });
                }
                let ident = if rop == ReduceOp::Sum {
                    0.0
                } else {
                    f64::NEG_INFINITY
                };
                out[(a * SHAPE[1] + b) * SHAPE[2] + c] = acc.unwrap_or(ident);
            }
        }
    }
    out
}

#[test]
fn the_oracle_matches_a_scan_written_over_explicit_coordinates() {
    let v = values();
    for rop in [ReduceOp::Sum, ReduceOp::Max] {
        for axis in 0..3u8 {
            for (reverse, exclusive) in [(false, false), (false, true), (true, false), (true, true)]
            {
                let got = oracle(&op(rop, axis, reverse, exclusive), &v);
                let want = reference(rop, usize::from(axis), reverse, exclusive, &v);
                assert!(
                    same(&got, &want),
                    "{rop:?} axis={axis} reverse={reverse} exclusive={exclusive}: {got:?} vs {want:?}"
                );
            }
        }
    }
}

#[test]
fn a_hand_computed_axis_zero_scan() {
    // axis 0 has k = 2: out[1][b][c] = in[0][b][c] + in[1][b][c], out[0] = in[0]
    let v: Vec<f64> = (0..N).map(|i| i as f64).collect();
    let got = oracle(&op(ReduceOp::Sum, 0, false, false), &v);
    assert_eq!(&got[..12], &v[..12]);
    for i in 12..24 {
        assert_eq!(got[i], v[i] + v[i - 12]);
    }
}

#[test]
fn the_lifted_kernel_agrees_with_the_oracle_byte_for_byte() {
    let v = values();
    for rop in [ReduceOp::Sum, ReduceOp::Max] {
        for axis in 0..3u8 {
            for (reverse, exclusive) in [(false, false), (true, true)] {
                let o = op(rop, axis, reverse, exclusive);
                assert!(
                    same(&lifted(&o, &v), &oracle(&o, &v)),
                    "{rop:?} axis={axis} reverse={reverse} exclusive={exclusive}"
                );
            }
        }
    }
}

#[test]
fn the_launched_lift_scans_only_the_innermost_axis() {
    assert!(lift_scan_opdef_launched(&op(ReduceOp::Sum, 2, false, false), F64, 3, 8, 2).is_ok());
    assert!(matches!(
        lift_scan_opdef_launched(&op(ReduceOp::Sum, 1, false, false), F64, 3, 8, 2),
        Err(LiftDecline::UnsupportedForm(m)) if m.contains("non-innermost")
    ));
    assert!(matches!(
        lift_scan_opdef(&op(ReduceOp::Sum, 3, false, false), F64, 3),
        Err(LiftDecline::UnsupportedForm(m)) if m.contains("outside the rank")
    ));
}
