// SPDX-License-Identifier: MIT OR Apache-2.0
//! **E1d-2: a block-tree float `Sum`/`Mean` meets its declared bound and a documented order**
//! (PM ruling 2026-10-31; `docs/e1d-parallel-mapping.md`, "E1d-2 as built").
//!
//! A float sum depends on its order, so the launched kernel is not held to the serial kernel's bytes.
//! It is held to three things:
//!
//! 1. **The bound.** For every row, `|result - exact| <= gamma_(n-1) * sum|x_i|` with `u = 2^-53`
//!    (plus the half ulp of the final rounding to the compute dtype, and for `Mean` everything divided
//!    by `n`), and against the serial kernel at most twice that. The corpus has a cancelling row, so
//!    the bound is far from `|result|` there and a dropped partial cannot hide.
//! 2. **Reproducibility.** The same `(grid, block)` gives the same bytes every run, and the bytes do
//!    not depend on `grid` at all (a row's order is fixed by `block`).
//! 3. **The written order.** The kernel's bytes equal a plain-Rust reimplementation of the order the
//!    note states: thread `t` folds `t, t+B, t+2B, ...` ascending into an `f64`; then
//!    `acc[t] = acc[t] + acc[t+s]` for `s = B/2 ... 1`; one rounding at the end. At `block = 1` that
//!    is the serial order, so the bytes equal the serial kernel's.
//!
//! Broken copies (a dropped round, an unstrided slice) must be seen by the order check.

use unpopped::ir::{Expr, OpDef, ReduceOp, ScalarExpr, input};
use unpopped::kernel::{Block, KernelFn, Stmt};
use unpopped::kernel_lift::{lift_reduction_opdef, lift_reduction_opdef_launched};
use unpopped::oracle::TypedBuffer;
use unpopped::oracle::kernel_eval::{KernelArg, run_kernel};
use unpopped::oracle::kernel_launch::run_launch;
use unpopped_vocab::{AxisMask, ElementKind};

const ROWS: usize = 5;
const COLS: usize = 96;
const F32: ElementKind = ElementKind::F32;
const F64: ElementKind = ElementKind::F64;
const U: f64 = 1.0 / 9_007_199_254_740_992.0; // 2^-53

/// grid x block: serial order, two threads, a tree with fewer blocks than rows, a block wider than
/// the row (128 > 96, so threads with nothing to fold), and more blocks than rows.
const GEOMETRIES: [(u32, u32); 6] = [(1, 1), (1, 2), (3, 8), (4, 32), (1, 128), (7, 16)];

/// A deterministic spread of finite values, no `rand` dependency.
fn lcg(state: &mut u64) -> f64 {
    *state = state
        .wrapping_mul(6_364_136_223_846_793_005)
        .wrapping_add(1_442_695_040_888_963_407);
    ((*state >> 11) as f64) / ((1u64 << 53) as f64) // [0, 1)
}

/// Five rows of `COLS` values: benign positives, a cancelling row, huge/tiny alternation, a value
/// with no exact binary form, and mixed signs.
fn corpus() -> Vec<f64> {
    let mut s = 0x9E37_79B9_7F4A_7C15u64;
    let mut v = Vec::with_capacity(ROWS * COLS);
    for c in 0..COLS {
        v.push(1.0e-3 + 1.0e3 * lcg(&mut s));
        let big = if c % 2 == 0 { 1.0e8 } else { -1.0e8 };
        v.push(if c % 7 == 0 { 1.0 + lcg(&mut s) } else { big });
        let huge = if c % 2 == 0 {
            1.0e30
        } else {
            -1.0e30 * (1.0 + 1e-3 * lcg(&mut s))
        };
        v.push(huge * 1.0e-20);
        v.push(0.1);
        v.push((lcg(&mut s) - 0.5) * 2.0e5);
    }
    // The pushes above interleave the five rows column by column; regroup them row-major.
    (0..ROWS)
        .flat_map(|r| (0..COLS).map(move |c| (r, c)))
        .map(|(r, c)| v[c * ROWS + r])
        .collect()
}

/// The corpus at the compute dtype: an `F32` run reads the `f32` roundings, and so must the reference.
fn values(dtype: ElementKind) -> Vec<f64> {
    corpus()
        .into_iter()
        .map(|x| if dtype == F32 { f64::from(x as f32) } else { x })
        .collect()
}

fn to_bytes(dtype: ElementKind, xs: &[f64]) -> Vec<u8> {
    if dtype == F32 {
        xs.iter().flat_map(|&x| (x as f32).to_le_bytes()).collect()
    } else {
        xs.iter().flat_map(|x| x.to_le_bytes()).collect()
    }
}

fn from_bytes(dtype: ElementKind, b: &[u8]) -> Vec<f64> {
    if dtype == F32 {
        b.chunks(4)
            .map(|c| f64::from(f32::from_le_bytes(c.try_into().unwrap())))
            .collect()
    } else {
        b.chunks(8)
            .map(|c| f64::from_le_bytes(c.try_into().unwrap()))
            .collect()
    }
}

fn op(dtype: ElementKind, rop: ReduceOp) -> OpDef {
    OpDef::reduction_axes(
        "r",
        1,
        &[dtype],
        Expr(input(0).0),
        rop,
        AxisMask::EMPTY,
        false,
    )
}

/// Run `k` (serial or launched) over the corpus; the per-row results.
fn run(k: &KernelFn, dtype: ElementKind, launch: bool) -> Vec<f64> {
    let w = if dtype == F32 { 4 } else { 8 };
    let mut a = vec![
        KernelArg::Buffer(TypedBuffer::new(
            dtype,
            vec![ROWS as i64, COLS as i64],
            vec![COLS as i64, 1],
            to_bytes(dtype, &values(dtype)),
        )),
        KernelArg::Buffer(TypedBuffer::new(
            dtype,
            vec![ROWS as i64],
            vec![1],
            vec![0u8; ROWS * w],
        )),
        KernelArg::Int(ROWS as i128),
        KernelArg::Int(COLS as i128),
    ];
    if launch {
        run_launch(k, &mut a).expect("the launch runs");
    } else {
        run_kernel(k, &mut a).expect("the serial kernel runs");
    }
    let KernelArg::Buffer(out) = &a[1] else {
        unreachable!("the output follows the input")
    };
    from_bytes(dtype, &out.bytes)
}

fn launched(dtype: ElementKind, rop: ReduceOp, grid: u32, block: u32) -> KernelFn {
    lift_reduction_opdef_launched(&op(dtype, rop), dtype, 2, block, grid).expect("lifts")
}

/// The sum in the order the note states, one rounding at the end (`Mean` divides the `f64` total).
fn documented(xs: &[f64], block: usize, mean: bool, dtype: ElementKind) -> f64 {
    let mut acc: Vec<f64> = (0..block)
        .map(|t| xs.iter().skip(t).step_by(block).fold(0.0, |a, &x| a + x))
        .collect();
    let mut s = block / 2;
    while s > 0 {
        for t in 0..s {
            acc[t] += acc[t + s];
        }
        s /= 2;
    }
    let total = if mean {
        acc[0] / xs.len() as f64
    } else {
        acc[0]
    };
    if dtype == F32 {
        f64::from(total as f32)
    } else {
        total
    }
}

/// A near-exact sum (Neumaier): its own error is `O(u^2 n sum|x|)`, negligible against the bound.
fn exact(xs: &[f64]) -> f64 {
    let (mut s, mut c) = (0.0f64, 0.0f64);
    for &x in xs {
        let t = s + x;
        c += if s.abs() >= x.abs() {
            (s - t) + x
        } else {
            (x - t) + s
        };
        s = t;
    }
    s + c
}

fn ulp32(x: f64) -> f64 {
    let b = (x as f32).abs().to_bits();
    f64::from(f32::from_bits(b + 1)) - f64::from(f32::from_bits(b))
}

/// `gamma_k = k u / (1 - k u)`.
fn gamma(k: usize) -> f64 {
    let ku = k as f64 * U;
    ku / (1.0 - ku)
}

/// The declared bound on one row's result against the exact value.
fn bound(xs: &[f64], mean: bool, dtype: ElementKind, got: f64) -> f64 {
    let sum_abs: f64 = xs.iter().map(|x| x.abs()).sum();
    let scale = if mean { 1.0 / xs.len() as f64 } else { 1.0 };
    let rounding = if dtype == F32 { 0.5 * ulp32(got) } else { 0.0 };
    gamma(xs.len() - 1) * sum_abs * scale + rounding + 4.0 * U * exact(xs).abs() * scale
}

fn row(dtype: ElementKind, r: usize) -> Vec<f64> {
    values(dtype)[r * COLS..(r + 1) * COLS].to_vec()
}

#[test]
fn every_geometry_meets_the_declared_bound_and_the_corpus_makes_it_bite() {
    for dtype in [F32, F64] {
        for rop in [ReduceOp::Sum, ReduceOp::Mean] {
            let mean = rop == ReduceOp::Mean;
            let serial = run(
                &lift_reduction_opdef(&op(dtype, rop), dtype, 2).unwrap(),
                dtype,
                false,
            );
            for (grid, block) in GEOMETRIES {
                let got = run(&launched(dtype, rop, grid, block), dtype, true);
                for (r, &g) in got.iter().enumerate() {
                    let xs = row(dtype, r);
                    let b = bound(&xs, mean, dtype, g);
                    let want = exact(&xs) / if mean { COLS as f64 } else { 1.0 };
                    assert!(
                        (g - want).abs() <= b,
                        "{dtype:?} {rop:?} row {r} grid {grid} block {block}: {} vs exact {want} (bound {b})",
                        g
                    );
                    assert!((g - serial[r]).abs() <= 2.0 * b, "vs the serial kernel");
                }
            }
        }
    }
    // Non-vacuity: the benign row's bound is tiny against its value; the cancelling row's is not.
    let (benign, cancelling) = (row(F64, 0), row(F64, 1));
    assert!(bound(&benign, false, F64, 0.0) < 1e-12 * exact(&benign).abs());
    assert!(exact(&cancelling).abs() < 1e-6 * cancelling.iter().map(|x| x.abs()).sum::<f64>());
}

#[test]
fn the_result_is_reproducible_and_depends_on_block_not_grid() {
    for dtype in [F32, F64] {
        for block in [2u32, 8, 32] {
            let once = run(&launched(dtype, ReduceOp::Sum, 3, block), dtype, true);
            assert_eq!(
                once,
                run(&launched(dtype, ReduceOp::Sum, 3, block), dtype, true)
            );
            for grid in [1u32, 7] {
                let other = run(&launched(dtype, ReduceOp::Sum, grid, block), dtype, true);
                assert_eq!(
                    once, other,
                    "{dtype:?} block {block}: grid {grid} changed a row"
                );
            }
        }
    }
}

#[test]
fn the_kernel_follows_the_written_order_and_block_one_is_the_serial_order() {
    for dtype in [F32, F64] {
        for rop in [ReduceOp::Sum, ReduceOp::Mean] {
            let mean = rop == ReduceOp::Mean;
            for (grid, block) in GEOMETRIES {
                let got = run(&launched(dtype, rop, grid, block), dtype, true);
                for (r, &g) in got.iter().enumerate() {
                    let want = documented(&row(dtype, r), block as usize, mean, dtype);
                    assert_eq!(
                        g.to_bits(),
                        want.to_bits(),
                        "{dtype:?} {rop:?} row {r} block {block}"
                    );
                }
            }
            let serial = lift_reduction_opdef(&op(dtype, rop), dtype, 2).unwrap();
            assert_eq!(
                run(&launched(dtype, rop, 2, 1), dtype, true),
                run(&serial, dtype, false)
            );
        }
    }
}

/// The block that holds the `Barrier`s directly.
fn row_block(b: &mut Block) -> Option<&mut Block> {
    if b.0.iter().any(|s| matches!(s, Stmt::Barrier { .. })) {
        return Some(b);
    }
    for s in &mut b.0 {
        if let Stmt::For { body, .. } = s {
            if let Some(r) = row_block(body) {
                return Some(r);
            }
        }
    }
    None
}

fn barriers(b: &Block) -> Vec<usize> {
    (0..b.0.len())
        .filter(|&i| matches!(b.0[i], Stmt::Barrier { .. }))
        .collect()
}

/// Does `k` (a broken copy) produce bytes other than the documented order's, or leave the bound?
fn seen(k: &KernelFn, block: usize) -> bool {
    let got = run(k, F64, true);
    (0..ROWS).any(|r| {
        let xs = row(F64, r);
        got[r].to_bits() != documented(&xs, block, false, F64).to_bits()
            || (got[r] - exact(&xs)).abs() > bound(&xs, false, F64, got[r])
    })
}

#[test]
fn a_dropped_round_and_an_unstrided_slice_are_seen() {
    let k = launched(F64, ReduceOp::Sum, 2, 8);
    assert!(
        !seen(&k, 8),
        "the control: the unbroken kernel is not seen as broken"
    );
    let mut dropped = k.clone();
    {
        let rb = row_block(&mut dropped.body).expect("a row block");
        let last = *barriers(rb).last().expect("barriers");
        rb.0.drain(last - 1..=last);
    }
    assert!(seen(&dropped, 8), "a dropped last round loses a partial");
    let mut unstrided = k;
    {
        let rb = row_block(&mut unstrided.body).expect("a row block");
        let Some(Stmt::For { step, .. }) = rb.0.iter_mut().find(|s| matches!(s, Stmt::For { .. }))
        else {
            panic!("the slice loop")
        };
        *step = ScalarExpr::ConstInt(1, ElementKind::I32);
    }
    assert!(
        seen(&unstrided, 8),
        "every thread folding the whole row counts it B times"
    );
}
