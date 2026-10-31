// SPDX-License-Identifier: MIT OR Apache-2.0
//! **E1d-5: a launched `Access::Scan` (chunked three-phase block scan) agrees with the serial scan
//! in the mode each case allows** (`docs/e1d-parallel-mapping.md`, "E1d-5 as built").
//!
//! * Integer `Sum`/`Prod`/`Max`/`Min` and float `Max`/`Min` are order-independent: bit-exact against
//!   the serial scan, in both directions and both exclusivities.
//! * A float `Sum` meets the declared bound for every output, is reproducible, does not depend on
//!   `grid`, equals a plain-Rust reimplementation of the written order bit for bit, and at
//!   `block = 1` equals the serial scan.
//! * A fused post, and a scan with no leading axis (block 0 does the row), follow the same rules.
//!
//! Broken copies (a dropped tree round, chunks that run to the end of the row) must be seen.

use unpopped::ir::{Expr, OpDef, ReduceOp, ScalarExpr, input, konst, reduced};
use unpopped::kernel::{Block, KernelFn, Stmt, check_kernel};
use unpopped::kernel_lift::{LiftDecline, lift_scan_opdef, lift_scan_opdef_launched};
use unpopped::oracle::TypedBuffer;
use unpopped::oracle::kernel_eval::{KernelArg, run_kernel};
use unpopped::oracle::kernel_launch::run_launch;
use unpopped_vocab::ElementKind;

const ROWS: usize = 4;
const COLS: usize = 45;
const F32: ElementKind = ElementKind::F32;
const F64: ElementKind = ElementKind::F64;
const I32: ElementKind = ElementKind::I32;
const U: f64 = 1.0 / 9_007_199_254_740_992.0;
/// Serial order, two threads, fewer blocks than rows, a block wider than the row (empty chunks),
/// more blocks than rows.
const GEOMETRIES: [(u32, u32); 6] = [(1, 1), (1, 2), (3, 8), (4, 32), (1, 64), (7, 16)];

fn lcg(state: &mut u64) -> f64 {
    *state = state
        .wrapping_mul(6_364_136_223_846_793_005)
        .wrapping_add(1_442_695_040_888_963_407);
    ((*state >> 11) as f64) / ((1u64 << 53) as f64)
}

/// Rows of `COLS` values: benign positives, a cancelling row, huge/tiny, `0.1`, mixed signs.
fn float_values(dtype: ElementKind) -> Vec<f64> {
    let mut s = 0x9E37_79B9_7F4A_7C15u64;
    let mut cols: Vec<[f64; ROWS]> = Vec::new();
    for c in 0..COLS {
        let big = if c % 2 == 0 { 1.0e8 } else { -1.0e8 };
        cols.push([
            1.0e-3 + 1.0e3 * lcg(&mut s),
            if c % 7 == 0 { 1.0 + lcg(&mut s) } else { big },
            0.1,
            (lcg(&mut s) - 0.5) * 2.0e5,
        ]);
    }
    (0..ROWS)
        .flat_map(|r| (0..COLS).map(move |c| (r, c)))
        .map(|(r, c)| {
            let x = cols[c][r];
            if dtype == F32 { f64::from(x as f32) } else { x }
        })
        .collect()
}

fn int_values() -> Vec<f64> {
    (0..ROWS * COLS)
        .map(|i| ((i * 37 + 11) % 41) as f64 - 20.0)
        .collect()
}

fn to_bytes(dtype: ElementKind, xs: &[f64]) -> Vec<u8> {
    match dtype {
        F32 => xs.iter().flat_map(|&x| (x as f32).to_le_bytes()).collect(),
        I32 => xs.iter().flat_map(|&x| (x as i32).to_le_bytes()).collect(),
        _ => xs.iter().flat_map(|x| x.to_le_bytes()).collect(),
    }
}

fn from_bytes(dtype: ElementKind, b: &[u8]) -> Vec<f64> {
    match dtype {
        F32 => b
            .chunks(4)
            .map(|c| f64::from(f32::from_le_bytes(c.try_into().unwrap())))
            .collect(),
        I32 => b
            .chunks(4)
            .map(|c| f64::from(i32::from_le_bytes(c.try_into().unwrap())))
            .collect(),
        _ => b
            .chunks(8)
            .map(|c| f64::from_le_bytes(c.try_into().unwrap()))
            .collect(),
    }
}

fn scan(dtype: ElementKind, rop: ReduceOp, reverse: bool, exclusive: bool) -> OpDef {
    OpDef::scan_simple("s", &[dtype], rop, 1, reverse, exclusive)
}

/// Run `k` on a `rows x COLS` input (or `[COLS]` for rank 1); the output cells, or the error.
fn run(
    k: &KernelFn,
    dtype: ElementKind,
    xs: &[f64],
    rank1: bool,
    launch: bool,
) -> Result<Vec<f64>, String> {
    let shape: Vec<i64> = if rank1 {
        vec![xs.len() as i64]
    } else {
        vec![ROWS as i64, COLS as i64]
    };
    let strides: Vec<i64> = if rank1 { vec![1] } else { vec![COLS as i64, 1] };
    let w = if dtype == F64 { 8 } else { 4 };
    let mut a = vec![
        KernelArg::Buffer(TypedBuffer::new(
            dtype,
            shape.clone(),
            strides.clone(),
            to_bytes(dtype, xs),
        )),
        KernelArg::Buffer(TypedBuffer::new(
            dtype,
            shape.clone(),
            strides,
            vec![0xAB; xs.len() * w],
        )),
    ];
    a.extend(shape.iter().map(|&d| KernelArg::Int(i128::from(d))));
    if launch {
        run_launch(k, &mut a).map_err(|e| format!("{e:?}"))?;
    } else {
        run_kernel(k, &mut a).map_err(|e| format!("{e:?}"))?;
    }
    let KernelArg::Buffer(out) = &a[1] else {
        unreachable!("the output follows the input")
    };
    Ok(from_bytes(dtype, &out.bytes))
}

fn launched(op: &OpDef, dtype: ElementKind, rank: u8, block: u32, grid: u32) -> KernelFn {
    let k = lift_scan_opdef_launched(op, dtype, rank, block, grid).expect("lifts");
    assert_eq!(check_kernel(&k), Ok(()), "a lifted kernel is well formed");
    k
}

fn serial(op: &OpDef, dtype: ElementKind, rank: u8, xs: &[f64], rank1: bool) -> Vec<f64> {
    let k = lift_scan_opdef(op, dtype, rank).expect("the serial lift");
    run(&k, dtype, xs, rank1, false).expect("the serial kernel runs")
}

fn same(a: &[f64], b: &[f64]) -> bool {
    a.len() == b.len()
        && a.iter()
            .zip(b)
            .all(|(x, y)| x.to_bits() == y.to_bits() || (x.is_nan() && y.is_nan()))
}

#[test]
fn order_independent_scans_equal_the_serial_scan_bit_for_bit() {
    let cases = [
        (I32, ReduceOp::Sum),
        (I32, ReduceOp::Prod),
        (I32, ReduceOp::Max),
        (I32, ReduceOp::Min),
        (F32, ReduceOp::Max),
        (F64, ReduceOp::Min),
    ];
    for (dtype, rop) in cases {
        let xs = if dtype == I32 {
            int_values()
        } else {
            float_values(dtype)
        };
        for reverse in [false, true] {
            for exclusive in [false, true] {
                let op = scan(dtype, rop, reverse, exclusive);
                let want = serial(&op, dtype, 2, &xs, false);
                for (grid, block) in GEOMETRIES {
                    let got = run(
                        &launched(&op, dtype, 2, block, grid),
                        dtype,
                        &xs,
                        false,
                        true,
                    )
                    .unwrap();
                    assert!(
                        same(&got, &want),
                        "{dtype:?} {rop:?} reverse={reverse} exclusive={exclusive} grid {grid} block {block}"
                    );
                }
            }
        }
    }
}

/// The float `Sum` scan in the written order, in scan order (callers reverse for a reverse scan).
fn documented(xs: &[f64], block: usize, exclusive: bool, dtype: ElementKind) -> Vec<f64> {
    let n = xs.len();
    let c = n.div_ceil(block);
    let chunk = |t: usize| (t * c).min(n)..(t * c + c).min(n);
    let mut totals: Vec<f64> = (0..block)
        .map(|t| xs[chunk(t)].iter().fold(0.0, |a, &x| a + x))
        .collect();
    let mut d = 1;
    while d < block {
        let old = totals.clone();
        for t in d..block {
            totals[t] = old[t] + old[t - d];
        }
        d *= 2;
    }
    let mut out = vec![0.0; n];
    for t in 0..block {
        let mut acc = if t == 0 { 0.0 } else { totals[t - 1] };
        for j in chunk(t) {
            if exclusive {
                out[j] = acc;
                acc += xs[j];
            } else {
                acc += xs[j];
                out[j] = acc;
            }
        }
    }
    out.into_iter()
        .map(|x| if dtype == F32 { f64::from(x as f32) } else { x })
        .collect()
}

fn row(xs: &[f64], r: usize) -> &[f64] {
    &xs[r * COLS..(r + 1) * COLS]
}

fn documented_row(
    xs: &[f64],
    r: usize,
    block: usize,
    reverse: bool,
    exclusive: bool,
    dtype: ElementKind,
) -> Vec<f64> {
    let mut v = row(xs, r).to_vec();
    if reverse {
        v.reverse();
    }
    let mut out = documented(&v, block, exclusive, dtype);
    if reverse {
        out.reverse();
    }
    out
}

#[test]
fn a_float_sum_scan_follows_the_written_order_and_block_one_is_serial() {
    for dtype in [F32, F64] {
        let xs = float_values(dtype);
        for reverse in [false, true] {
            for exclusive in [false, true] {
                let op = scan(dtype, ReduceOp::Sum, reverse, exclusive);
                for (grid, block) in GEOMETRIES {
                    let got = run(
                        &launched(&op, dtype, 2, block, grid),
                        dtype,
                        &xs,
                        false,
                        true,
                    )
                    .unwrap();
                    for r in 0..ROWS {
                        let want =
                            documented_row(&xs, r, block as usize, reverse, exclusive, dtype);
                        assert!(
                            same(&got[r * COLS..(r + 1) * COLS], &want),
                            "{dtype:?} reverse={reverse} exclusive={exclusive} row {r} grid {grid} block {block}"
                        );
                    }
                }
                let one = run(&launched(&op, dtype, 2, 1, 3), dtype, &xs, false, true).unwrap();
                assert!(
                    same(&one, &serial(&op, dtype, 2, &xs, false)),
                    "block 1 is the serial order"
                );
            }
        }
    }
}

/// A near-exact sum (Neumaier).
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

fn gamma(k: usize) -> f64 {
    let ku = k as f64 * U;
    ku / (1.0 - ku)
}

#[test]
fn every_prefix_meets_the_declared_bound() {
    for dtype in [F32, F64] {
        let xs = float_values(dtype);
        let op = scan(dtype, ReduceOp::Sum, false, false);
        let ser = serial(&op, dtype, 2, &xs, false);
        for (grid, block) in GEOMETRIES {
            let got = run(
                &launched(&op, dtype, 2, block, grid),
                dtype,
                &xs,
                false,
                true,
            )
            .unwrap();
            for r in 0..ROWS {
                let v = row(&xs, r);
                for i in 0..COLS {
                    let g = got[r * COLS + i];
                    let prefix = &v[..=i];
                    let sum_abs: f64 = prefix.iter().map(|x| x.abs()).sum();
                    let rounding = if dtype == F32 { 0.5 * ulp32(g) } else { 0.0 };
                    let b = gamma(prefix.len().saturating_sub(1)) * sum_abs
                        + rounding
                        + 4.0 * U * exact(prefix).abs();
                    assert!(
                        (g - exact(prefix)).abs() <= b,
                        "{dtype:?} row {r} pos {i} grid {grid} block {block}"
                    );
                    assert!(
                        (g - ser[r * COLS + i]).abs() <= 2.0 * b + 1e-300,
                        "vs the serial scan"
                    );
                }
            }
        }
    }
    // Non-vacuity: on the cancelling row the bound is far from the prefix value.
    let v = float_values(F64);
    let cancelling = row(&v, 1);
    assert!(exact(cancelling).abs() < 1e-6 * cancelling.iter().map(|x| x.abs()).sum::<f64>());
}

#[test]
fn the_result_is_reproducible_and_depends_on_block_not_grid() {
    let xs = float_values(F64);
    let op = scan(F64, ReduceOp::Sum, false, true);
    for block in [2u32, 8, 32] {
        let once = run(&launched(&op, F64, 2, block, 3), F64, &xs, false, true).unwrap();
        for grid in [3u32, 1, 7] {
            assert_eq!(
                once,
                run(&launched(&op, F64, 2, block, grid), F64, &xs, false, true).unwrap()
            );
        }
    }
}

#[test]
fn a_scan_with_no_leading_axis_and_a_fused_post_follow_the_same_rules() {
    let n = COLS * 2;
    let xs: Vec<f64> = int_values().into_iter().take(n).collect();
    for rop in [ReduceOp::Sum, ReduceOp::Max] {
        let op = OpDef::scan_simple("s", &[I32], rop, 0, false, false);
        let want = serial(&op, I32, 1, &xs, true);
        for (grid, block) in GEOMETRIES {
            let got = run(&launched(&op, I32, 1, block, grid), I32, &xs, true, true).unwrap();
            assert_eq!(got, want, "{rop:?} grid {grid} block {block}");
        }
    }
    // A Max scan with a post is order-independent, so bit-exact against the serial scan.
    let fv = float_values(F32);
    let op = OpDef::scan(
        "p",
        1,
        &[F32],
        ReduceOp::Max,
        1,
        false,
        false,
        Expr(input(0).0),
        Expr((reduced(0) * konst(0.5) + konst(1.0)).0),
    );
    let want = serial(&op, F32, 2, &fv, false);
    for (grid, block) in GEOMETRIES {
        let got = run(&launched(&op, F32, 2, block, grid), F32, &fv, false, true).unwrap();
        assert!(same(&got, &want), "post grid {grid} block {block}");
    }
    let _ = ScalarExpr::Reduced(0);
}

#[test]
fn forms_the_lift_does_not_cover_decline_with_a_reason() {
    let ok = |op: &OpDef, dtype, b, g| lift_scan_opdef_launched(op, dtype, 2, b, g);
    let sum = scan(F32, ReduceOp::Sum, false, false);
    assert!(ok(&sum, F32, 8, 2).is_ok());
    assert_eq!(ok(&sum, F32, 0, 2), Err(LiftDecline::BadLaunch));
    assert_eq!(ok(&sum, F32, 6, 2), Err(LiftDecline::BadLaunch));
    assert_eq!(ok(&sum, F32, 8, 0), Err(LiftDecline::BadLaunch));
    assert!(matches!(
        ok(&scan(F32, ReduceOp::Prod, false, false), F32, 8, 2),
        Err(LiftDecline::UnsupportedForm(_))
    ));
    assert!(
        ok(&scan(I32, ReduceOp::Prod, false, false), I32, 8, 2).is_ok(),
        "an integer Prod wraps and is associative"
    );
    assert!(matches!(
        ok(&scan(F32, ReduceOp::Mean, false, false), F32, 8, 2),
        Err(LiftDecline::UnsupportedForm(_))
    ));
    let outer = OpDef::scan_simple("o", &[F32], ReduceOp::Sum, 0, false, false);
    assert!(matches!(
        ok(&outer, F32, 8, 2),
        Err(LiftDecline::UnsupportedForm(_))
    ));
    let ew = OpDef::elementwise("e", 1, &[F32], Expr(input(0).0));
    assert_eq!(ok(&ew, F32, 8, 2), Err(LiftDecline::NotScan));
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

/// Is a broken copy seen: it errors, or its bytes differ from the written order?
fn seen(k: &KernelFn, xs: &[f64], block: usize) -> bool {
    let Ok(got) = run(k, F64, xs, false, true) else {
        return true;
    };
    (0..ROWS).any(|r| {
        !same(
            &got[r * COLS..(r + 1) * COLS],
            &documented_row(xs, r, block, false, false, F64),
        )
    })
}

#[test]
fn a_dropped_tree_round_and_chunks_that_run_to_the_end_are_seen() {
    let xs = float_values(F64);
    let k = launched(&scan(F64, ReduceOp::Sum, false, false), F64, 2, 8, 2);
    assert!(
        !seen(&k, &xs, 8),
        "the control: the unbroken kernel is not seen as broken"
    );

    // Barriers in the row block: the publish barrier, then two per round (read, write), then the end.
    let mut dropped = k.clone();
    {
        let rb = row_block(&mut dropped.body).expect("a row block");
        let b = barriers(rb);
        // Drop the last round: its guarded read, barrier, guarded combine and barrier.
        let last = b[b.len() - 2];
        rb.0.drain(last - 3..=last);
    }
    assert!(
        seen(&dropped, &xs, 8),
        "a dropped last round loses a partial offset"
    );

    let mut to_end = k;
    {
        // body.0[0..3] are `chunk`, the counting loop, `lo`, `hi`; make `hi` the row length.
        let Stmt::Let { init, .. } = &mut to_end.body.0[3] else {
            panic!("the hi bound")
        };
        *init = ScalarExpr::Local(1);
    }
    assert!(
        seen(&to_end, &xs, 8),
        "every thread walking to the end of the row counts elements twice"
    );
}
