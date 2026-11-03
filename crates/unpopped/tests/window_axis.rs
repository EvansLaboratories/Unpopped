// SPDX-License-Identifier: MIT OR Apache-2.0
//! **A window over any axis** (ROADMAP U2 D5, window half; `docs/options-u2-lift-declines.md`): an `Access::Window` whose
//! pooled axis is not the innermost. The tensor splits as `(outer, k, inner)` around the pooled axis; a "row" is one
//! `(outer, inner)` pair and its elements sit `inner` apart; the output has the pooled axis's `k_out` and every other
//! extent unchanged. Every input must be full-width dense (the plan gate checks it). The oracle is held to a pool written
//! here in Rust over explicit coordinates; the lifted kernel (its loop nest is the same statements with the pooled axis in
//! the window position) is held to the oracle byte for byte.
//!
//! The tensor is `[2, 5, 4]`: axis 0 has `k = 2`, axis 1 has `k = 5`, axis 2 is the old innermost walk (the control). The
//! geometry pads, strides and dilates so edge windows hold fewer taps than `size`.

use unpopped::ir::{OpDef, ReduceOp};
use unpopped::kernel::check_kernel;
use unpopped::kernel_lift::lift_window_opdef;
use unpopped::oracle::kernel_eval::{KernelArg, run_kernel};
use unpopped::oracle::{TypedBuffer, evaluate};
use unpopped::plan::try_build_plan;
use unpopped_vocab::{ArchSku, ElementKind, OpCategory, OperandDesc, try_structure_key};

const F64: ElementKind = ElementKind::F64;
const IN: [i64; 3] = [2, 5, 4];

#[derive(Clone, Copy)]
struct Geo {
    size: i64,
    stride: i64,
    dilation: i64,
    pad: i64,
}

const G: [Geo; 2] = [
    Geo {
        size: 2,
        stride: 1,
        dilation: 1,
        pad: 1,
    },
    Geo {
        size: 3,
        stride: 2,
        dilation: 1,
        pad: 1,
    },
];

impl Geo {
    fn k_out(&self, k_in: i64) -> i64 {
        let span = self.dilation * (self.size - 1) + 1;
        (k_in + 2 * self.pad - span) / self.stride + 1
    }
}

fn op(rop: ReduceOp, axis: u8, g: Geo, include_pad: bool) -> OpDef {
    OpDef::window_simple(
        "win",
        &[F64],
        rop,
        axis,
        g.size as u8,
        g.stride as u8,
        g.dilation as u8,
        g.pad as u8,
        g.pad as u8,
        include_pad,
    )
}

fn out_shape(axis: usize, g: Geo) -> Vec<i64> {
    let mut s = IN.to_vec();
    s[axis] = g.k_out(IN[axis]);
    s
}

fn strides(shape: &[i64]) -> Vec<i64> {
    vec![shape[1] * shape[2], shape[2], 1]
}

fn values() -> Vec<f64> {
    (0..40)
        .map(|i| match i {
            13 => f64::NAN,
            _ => ((i * 5 + 2) % 9) as f64 - 3.5,
        })
        .collect()
}

fn buffer(vals: &[f64]) -> TypedBuffer {
    let bytes = vals.iter().flat_map(|v| v.to_le_bytes()).collect();
    TypedBuffer::new(F64, IN.to_vec(), strides(&IN), bytes)
}

fn descs(axis: usize, g: Geo) -> [OperandDesc; 2] {
    let o = out_shape(axis, g);
    [
        OperandDesc::new(3, &IN, &strides(&IN), F64, 256),
        OperandDesc::new(3, &o, &strides(&o), F64, 256),
    ]
}

fn oracle(o: &OpDef, axis: usize, g: Geo, vals: &[f64]) -> Vec<f64> {
    let d = descs(axis, g);
    let key = try_structure_key(OpCategory::Pooling, &d, ArchSku::Sm89).expect("a key");
    let plan = try_build_plan(o, &key).expect("the window plans");
    f64s(
        &evaluate(&plan, &d, std::slice::from_ref(&buffer(vals)), &[])
            .remove(0)
            .bytes,
    )
}

fn lifted(o: &OpDef, axis: usize, g: Geo, vals: &[f64]) -> Vec<f64> {
    let kernel = lift_window_opdef(o, F64, 3).expect("the window lifts");
    assert_eq!(check_kernel(&kernel), Ok(()));
    let shape = out_shape(axis, g);
    let n: i64 = shape.iter().product();
    let out = TypedBuffer::new(
        F64,
        shape.clone(),
        strides(&shape),
        vec![0xAB; 8 * n as usize],
    );
    let mut args = vec![KernelArg::Buffer(buffer(vals)), KernelArg::Buffer(out)];
    args.extend(IN.iter().map(|&e| KernelArg::Int(i128::from(e))));
    args.push(KernelArg::Int(i128::from(shape[axis])));
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

/// The reference: for each output coordinate, the in-bounds taps along `axis`, folded in order, with explicit coordinates.
fn reference(rop: ReduceOp, axis: usize, g: Geo, include_pad: bool, vals: &[f64]) -> Vec<f64> {
    let shape = out_shape(axis, g);
    let at = |c: [i64; 3]| vals[((c[0] * IN[1] + c[1]) * IN[2] + c[2]) as usize];
    let mut out = Vec::new();
    for a in 0..shape[0] {
        for b in 0..shape[1] {
            for c in 0..shape[2] {
                let here = [a, b, c];
                let o = here[axis];
                let taps: Vec<f64> = (0..g.size)
                    .map(|t| o * g.stride - g.pad + t * g.dilation)
                    .filter(|p| (0..IN[axis]).contains(p))
                    .map(|p| {
                        let mut q = here;
                        q[axis] = p;
                        at(q)
                    })
                    .collect();
                out.push(match rop {
                    ReduceOp::Sum => taps.iter().fold(0.0, |s, x| s + x),
                    ReduceOp::Mean => {
                        let d = if include_pad {
                            g.size
                        } else {
                            taps.len() as i64
                        };
                        taps.iter().fold(0.0, |s, x| s + x) / d as f64
                    }
                    _ => taps.iter().skip(1).fold(taps[0], |m, &x| {
                        if m.is_nan() {
                            m
                        } else if x.is_nan() || x > m {
                            x
                        } else {
                            m
                        }
                    }),
                });
            }
        }
    }
    out
}

#[test]
fn the_oracle_matches_a_pool_written_over_explicit_coordinates() {
    let v = values();
    for g in G {
        for axis in 0..3usize {
            for (rop, include) in [
                (ReduceOp::Sum, false),
                (ReduceOp::Max, false),
                (ReduceOp::Mean, false),
                (ReduceOp::Mean, true),
            ] {
                let got = oracle(&op(rop, axis as u8, g, include), axis, g, &v);
                let want = reference(rop, axis, g, include, &v);
                assert!(
                    same(&got, &want),
                    "{rop:?} include={include} axis={axis} size={}: {got:?} vs {want:?}",
                    g.size
                );
            }
        }
    }
}

#[test]
fn the_output_keeps_every_other_extent() {
    // axis 1 with size 2, stride 1, pad 1: k_out = (5 + 2 - 2) / 1 + 1 = 6; the output is [2, 6, 4]
    let v = values();
    let got = oracle(&op(ReduceOp::Sum, 1, G[0], false), 1, G[0], &v);
    assert_eq!(got.len(), 2 * 6 * 4);
    // output (0, 0, 0) pools input positions -1 and 0 along axis 1: only (0, 0, 0)
    assert_eq!(got[0], v[0]);
}

#[test]
fn the_lifted_kernel_agrees_with_the_oracle_byte_for_byte() {
    let v = values();
    for g in G {
        for axis in 0..3usize {
            for (rop, include) in [
                (ReduceOp::Sum, false),
                (ReduceOp::Max, false),
                (ReduceOp::Mean, true),
            ] {
                let o = op(rop, axis as u8, g, include);
                assert!(
                    same(&lifted(&o, axis, g, &v), &oracle(&o, axis, g, &v)),
                    "{rop:?} include={include} axis={axis} size={}",
                    g.size
                );
            }
        }
    }
}
