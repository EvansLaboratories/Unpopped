// SPDX-License-Identifier: MIT OR Apache-2.0
//! **E1d-3: a launched `Access::RowReduce` (one block per row, each stage a block-tree reduction,
//! then a strided epilogue) agrees with the serial lift in the mode each case allows**
//! (`docs/e1d-parallel-mapping.md`, "E1d-3 as built").
//!
//! * `Max`/`Min` stages are order-independent: bit-exact against the serial lift.
//! * A `Sum`/`Mean` stage meets the E1d-2 bound against the exact value, is reproducible, does not
//!   depend on `grid`, and its bytes equal a plain-Rust reimplementation of the written order.
//! * A two-stage op whose epilogue is plain arithmetic (a centred quotient) equals the written
//!   order bit for bit, which exercises the broadcast of a stage result into the next stage.
//! * Softmax and RMS norm (transcendental epilogues) stay within a few ulps of the serial lift.
//!
//! Broken copies (a dropped round, an unstrided slice, a missing broadcast `Barrier`) must be seen.

use unpopped::ir::{Expr, OpDef, ReduceOp, ReduceStage, UnaryOp, input, konst, reduced};
use unpopped::kernel::{Block, KernelFn, Stmt, check_kernel};
use unpopped::kernel_lift::{LiftDecline, lift_row_reduce_opdef, lift_row_reduce_opdef_launched};
use unpopped::oracle::TypedBuffer;
use unpopped::oracle::kernel_eval::{KernelArg, run_kernel};
use unpopped::oracle::kernel_launch::run_launch;
use unpopped_vocab::ElementKind;

const ROWS: usize = 5;
const COLS: usize = 96;
const F32: ElementKind = ElementKind::F32;
const F64: ElementKind = ElementKind::F64;
const U: f64 = 1.0 / 9_007_199_254_740_992.0; // 2^-53
const GEOMETRIES: [(u32, u32); 6] = [(1, 1), (1, 2), (3, 8), (4, 32), (1, 128), (7, 16)];

fn lcg(state: &mut u64) -> f64 {
    *state = state
        .wrapping_mul(6_364_136_223_846_793_005)
        .wrapping_add(1_442_695_040_888_963_407);
    ((*state >> 11) as f64) / ((1u64 << 53) as f64)
}

/// Five rows: benign positives, a cancelling row, huge/tiny alternation, `0.1`, mixed signs.
fn corpus() -> Vec<f64> {
    let mut s = 0x9E37_79B9_7F4A_7C15u64;
    let mut cols: Vec<[f64; ROWS]> = Vec::new();
    for c in 0..COLS {
        let big = if c % 2 == 0 { 1.0e8 } else { -1.0e8 };
        let huge = if c % 2 == 0 { 1.0e10 } else { -1.0e10 };
        cols.push([
            1.0e-3 + 1.0e3 * lcg(&mut s),
            if c % 7 == 0 { 1.0 + lcg(&mut s) } else { big },
            huge * (1.0 + 1e-3 * lcg(&mut s)),
            0.1,
            (lcg(&mut s) - 0.5) * 2.0e5,
        ]);
    }
    (0..ROWS)
        .flat_map(|r| (0..COLS).map(move |c| (r, c)))
        .map(|(r, c)| cols[c][r])
        .collect()
}

/// A gentle corpus for the transcendental epilogues: values in `[-2, 4]`.
fn gentle() -> Vec<f64> {
    (0..ROWS * COLS)
        .map(|i| ((i * 7 + 3) % 19) as f64 * 0.37 - 2.0)
        .collect()
}

fn at(dtype: ElementKind, xs: Vec<f64>) -> Vec<f64> {
    xs.into_iter()
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

fn stage(pre: Expr, op: ReduceOp) -> ReduceStage {
    ReduceStage { pre: pre.0, op }
}

fn rr(dtype: ElementKind, stages: Vec<ReduceStage>, epilogue: Expr) -> OpDef {
    OpDef::row_reduce("rr", 1, &[dtype], stages, epilogue)
}

fn identity(dtype: ElementKind, rop: ReduceOp) -> OpDef {
    rr(dtype, vec![stage(input(0), rop)], reduced(0))
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

fn rms_norm(dtype: ElementKind) -> OpDef {
    rr(
        dtype,
        vec![stage(input(0) * input(0), ReduceOp::Mean)],
        input(0) / (reduced(0) + konst(1.0e-3)).unary(UnaryOp::Sqrt),
    )
}

/// Run `k` over `xs` (`ROWS x COLS`); the output cells, or the launch error.
fn run(k: &KernelFn, dtype: ElementKind, xs: &[f64], launch: bool) -> Result<Vec<f64>, String> {
    let w = if dtype == F32 { 4 } else { 8 };
    let shape = vec![ROWS as i64, COLS as i64];
    let strides = vec![COLS as i64, 1];
    let mut a = vec![
        KernelArg::Buffer(TypedBuffer::new(
            dtype,
            shape.clone(),
            strides.clone(),
            to_bytes(dtype, xs),
        )),
        KernelArg::Buffer(TypedBuffer::new(
            dtype,
            shape,
            strides,
            vec![0xAB; ROWS * COLS * w],
        )),
        KernelArg::Int(ROWS as i128),
        KernelArg::Int(COLS as i128),
    ];
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

fn launched(op: &OpDef, dtype: ElementKind, block: u32, grid: u32) -> KernelFn {
    let k = lift_row_reduce_opdef_launched(op, dtype, 2, block, grid).expect("lifts");
    assert_eq!(check_kernel(&k), Ok(()), "a lifted kernel is well formed");
    k
}

fn serial(op: &OpDef, dtype: ElementKind, xs: &[f64]) -> Vec<f64> {
    let k = lift_row_reduce_opdef(op, dtype, 2).expect("the serial lift");
    run(&k, dtype, xs, false).expect("the serial kernel runs")
}

fn rnd(dtype: ElementKind, x: f64) -> f64 {
    if dtype == F32 { f64::from(x as f32) } else { x }
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
    rnd(
        dtype,
        if mean {
            acc[0] / xs.len() as f64
        } else {
            acc[0]
        },
    )
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

/// The declared bound on one stage result against the exact value.
fn bound(xs: &[f64], mean: bool, dtype: ElementKind, got: f64) -> f64 {
    let sum_abs: f64 = xs.iter().map(|x| x.abs()).sum();
    let scale = if mean { 1.0 / xs.len() as f64 } else { 1.0 };
    let rounding = if dtype == F32 { 0.5 * ulp32(got) } else { 0.0 };
    gamma(xs.len() - 1) * sum_abs * scale + rounding + 4.0 * U * exact(xs).abs() * scale
}

fn row(xs: &[f64], r: usize) -> &[f64] {
    &xs[r * COLS..(r + 1) * COLS]
}

fn ulps(dtype: ElementKind, a: f64, b: f64) -> u64 {
    if a.is_nan() || b.is_nan() {
        return u64::from(a.is_nan() != b.is_nan()) * u64::MAX;
    }
    let key = |x: f64| {
        let bits = if dtype == F32 {
            i64::from((x as f32).to_bits() as i32)
        } else {
            x.to_bits() as i64
        };
        if bits < 0 { i64::MIN - bits } else { bits }
    };
    key(a).abs_diff(key(b))
}

#[test]
fn max_and_min_stages_equal_the_serial_lift_bit_for_bit() {
    for dtype in [F32, F64] {
        let xs = at(dtype, corpus());
        for rop in [ReduceOp::Max, ReduceOp::Min] {
            let op = identity(dtype, rop);
            let want = serial(&op, dtype, &xs);
            for (grid, block) in GEOMETRIES {
                let got = run(&launched(&op, dtype, block, grid), dtype, &xs, true).unwrap();
                assert_eq!(got, want, "{dtype:?} {rop:?} grid {grid} block {block}");
            }
        }
    }
}

#[test]
fn sum_and_mean_stages_meet_the_declared_bound_and_the_corpus_makes_it_bite() {
    for dtype in [F32, F64] {
        let xs = at(dtype, corpus());
        for rop in [ReduceOp::Sum, ReduceOp::Mean] {
            let mean = rop == ReduceOp::Mean;
            let op = identity(dtype, rop);
            let ser = serial(&op, dtype, &xs);
            for (grid, block) in GEOMETRIES {
                let got = run(&launched(&op, dtype, block, grid), dtype, &xs, true).unwrap();
                for r in 0..ROWS {
                    let want = exact(row(&xs, r)) / if mean { COLS as f64 } else { 1.0 };
                    for c in 0..COLS {
                        let g = got[r * COLS + c];
                        let b = bound(row(&xs, r), mean, dtype, g);
                        assert!(
                            (g - want).abs() <= b,
                            "{dtype:?} {rop:?} row {r} grid {grid} block {block}: {g} vs {want} (bound {b})"
                        );
                        assert!(
                            (g - ser[r * COLS + c]).abs() <= 2.0 * b,
                            "vs the serial lift"
                        );
                    }
                }
            }
        }
    }
    // Non-vacuity: the benign row's bound is tiny against its value; the cancelling row's is not.
    let xs = at(F64, corpus());
    assert!(bound(row(&xs, 0), false, F64, 0.0) < 1e-12 * exact(row(&xs, 0)).abs());
    let cancelling = row(&xs, 1);
    assert!(exact(cancelling).abs() < 1e-6 * cancelling.iter().map(|x| x.abs()).sum::<f64>());
}

#[test]
fn the_written_order_is_followed_and_block_one_is_the_serial_order() {
    for dtype in [F32, F64] {
        let xs = at(dtype, corpus());
        for rop in [ReduceOp::Sum, ReduceOp::Mean] {
            let op = identity(dtype, rop);
            for (grid, block) in GEOMETRIES {
                let got = run(&launched(&op, dtype, block, grid), dtype, &xs, true).unwrap();
                for r in 0..ROWS {
                    let want =
                        documented(row(&xs, r), block as usize, rop == ReduceOp::Mean, dtype);
                    for c in 0..COLS {
                        assert_eq!(
                            got[r * COLS + c].to_bits(),
                            want.to_bits(),
                            "{rop:?} row {r}"
                        );
                    }
                }
            }
            let one = run(&launched(&op, dtype, 1, 2), dtype, &xs, true).unwrap();
            assert_eq!(one, serial(&op, dtype, &xs));
        }
    }
}

/// The centred quotient in the written order: stage 0 a documented mean, stage 1 the max of
/// `x - r0` (order-independent), the epilogue `(x - r0) / r1`; stage values and the output are
/// rounded to the compute dtype, the expressions between them are not.
fn centred_documented(xs: &[f64], block: usize, dtype: ElementKind) -> Vec<f64> {
    let r0 = documented(xs, block, true, dtype);
    // The kernel interpreter evaluates an expression in `f64` and rounds when a value is stored.
    let sub = |x: f64| x - r0;
    let r1 = rnd(dtype, xs.iter().map(|&x| sub(x)).fold(f64::MIN, f64::max));
    xs.iter().map(|&x| rnd(dtype, sub(x) / r1)).collect()
}

#[test]
fn a_two_stage_op_follows_the_written_order_bit_for_bit() {
    for dtype in [F32, F64] {
        let xs = at(dtype, gentle());
        let op = centred_quotient(dtype);
        for (grid, block) in GEOMETRIES {
            let got = run(&launched(&op, dtype, block, grid), dtype, &xs, true).unwrap();
            for r in 0..ROWS {
                let want = centred_documented(row(&xs, r), block as usize, dtype);
                let cells = &got[r * COLS..(r + 1) * COLS];
                assert!(
                    cells
                        .iter()
                        .zip(&want)
                        .all(|(g, w)| g.to_bits() == w.to_bits()),
                    "{dtype:?} row {r} grid {grid} block {block}: {:?} vs {:?}",
                    &cells[..4],
                    &want[..4]
                );
            }
        }
        let one = run(&launched(&op, dtype, 1, 3), dtype, &xs, true).unwrap();
        assert_eq!(one, serial(&op, dtype, &xs), "block 1 is the serial lift");
    }
}

#[test]
fn transcendental_epilogues_stay_within_a_few_ulps_of_the_serial_lift() {
    for dtype in [F32, F64] {
        let xs = at(dtype, gentle());
        for (name, op) in [("softmax", softmax(dtype)), ("rms norm", rms_norm(dtype))] {
            let want = serial(&op, dtype, &xs);
            for (grid, block) in GEOMETRIES {
                let got = run(&launched(&op, dtype, block, grid), dtype, &xs, true).unwrap();
                let worst = got
                    .iter()
                    .zip(&want)
                    .map(|(&g, &w)| ulps(dtype, g, w))
                    .max();
                assert!(
                    worst <= Some(4),
                    "{dtype:?} {name} grid {grid} block {block}: {worst:?} ulps"
                );
            }
        }
    }
}

#[test]
fn the_result_is_reproducible_and_depends_on_block_not_grid() {
    for dtype in [F32, F64] {
        let xs = at(dtype, gentle());
        for block in [2u32, 8, 32] {
            let once = run(
                &launched(&softmax(dtype), dtype, block, 3),
                dtype,
                &xs,
                true,
            )
            .unwrap();
            for grid in [3u32, 1, 7] {
                let again = run(
                    &launched(&softmax(dtype), dtype, block, grid),
                    dtype,
                    &xs,
                    true,
                )
                .unwrap();
                assert_eq!(once, again, "{dtype:?} block {block} grid {grid}");
            }
        }
    }
}

#[test]
fn forms_the_lift_does_not_cover_decline_with_a_reason() {
    let sm = softmax(F32);
    let ok = |b, g| lift_row_reduce_opdef_launched(&sm, F32, 2, b, g);
    assert!(ok(8, 2).is_ok());
    assert_eq!(ok(0, 2), Err(LiftDecline::BadLaunch));
    assert_eq!(ok(6, 2), Err(LiftDecline::BadLaunch));
    assert_eq!(ok(8, 0), Err(LiftDecline::BadLaunch));
    assert_eq!(
        lift_row_reduce_opdef_launched(&sm, F32, 1, 8, 2),
        Err(LiftDecline::UnsupportedForm(
            "a launched row reduction needs a leading axis to stride blocks over"
        ))
    );
    let ew = OpDef::elementwise("e", 1, &[F32], Expr(input(0).0));
    assert_eq!(
        lift_row_reduce_opdef_launched(&ew, F32, 2, 8, 2),
        Err(LiftDecline::NotRowReduce)
    );
    assert_eq!(
        lift_row_reduce_opdef_launched(&softmax(ElementKind::I32), ElementKind::I32, 2, 8, 2),
        Err(LiftDecline::UnsupportedForm("a non-float row reduction"))
    );
    assert_eq!(
        lift_row_reduce_opdef_launched(&identity(F32, ReduceOp::Prod), F32, 2, 8, 2),
        Err(LiftDecline::UnsupportedForm("a Prod or empty stage list"))
    );
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

/// Is `k` (a broken copy) seen: it errors, or its bytes differ from the written order's?
fn seen(k: &KernelFn, xs: &[f64], block: usize) -> bool {
    let Ok(got) = run(k, F64, xs, true) else {
        return true;
    };
    (0..ROWS).any(|r| {
        let want = centred_documented(row(xs, r), block, F64);
        got[r * COLS..(r + 1) * COLS]
            .iter()
            .zip(&want)
            .any(|(g, w)| g.to_bits() != w.to_bits())
    })
}

#[test]
fn a_dropped_round_an_unstrided_slice_and_a_missing_broadcast_barrier_are_seen() {
    let xs = at(F64, gentle());
    let k = launched(&centred_quotient(F64), F64, 8, 2);
    assert!(
        !seen(&k, &xs, 8),
        "the control: the unbroken kernel is not seen as broken"
    );

    // Layout of stage 0 in the row block: publish barrier, three round barriers, broadcast barrier.
    let mut dropped = k.clone();
    {
        let rb = row_block(&mut dropped.body).expect("a row block");
        let b = barriers(rb);
        rb.0.drain(b[3] - 1..=b[3]);
    }
    assert!(
        seen(&dropped, &xs, 8),
        "a dropped last round loses a partial"
    );

    let mut unstrided = k.clone();
    {
        let rb = row_block(&mut unstrided.body).expect("a row block");
        let Some(Stmt::For { step, .. }) = rb.0.iter_mut().find(|s| matches!(s, Stmt::For { .. }))
        else {
            panic!("the slice loop")
        };
        *step = unpopped::ir::ScalarExpr::ConstInt(1, ElementKind::I32);
    }
    assert!(
        seen(&unstrided, &xs, 8),
        "every thread folding the whole row counts it B times"
    );

    let mut unbroadcast = k;
    {
        let rb = row_block(&mut unbroadcast.body).expect("a row block");
        let b = barriers(rb);
        rb.0.remove(b[4]);
    }
    assert!(
        seen(&unbroadcast, &xs, 8),
        "readers of the stage result need the barrier after thread 0's store"
    );
}
