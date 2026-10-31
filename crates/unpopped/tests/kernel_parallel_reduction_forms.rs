// SPDX-License-Identifier: MIT OR Apache-2.0
//! **E1d-4: the launched reduction takes several reduced axes, no kept axis, and a fused post**
//! (`docs/e1d-parallel-mapping.md`, "E1d-4 as built").
//!
//! * **Several reduced axes:** thread `t` takes the indices `t, t+B, ...` of the outermost reduced
//!   axis and folds everything under each in row-major order. Order-independent folds equal the
//!   serial kernel's bytes; a float `Sum`/`Mean` equals a plain-Rust reimplementation of that order.
//! * **No kept axis** (one output): block 0 does the row, other blocks idle.
//! * **A fused post** runs on thread 0 on the finalised fold rounded once, as the serial lift does:
//!   `Max` is bit-exact against the serial kernel, a `Sum`/`Mean` post equals the written order
//!   followed by the same post, and at `block = 1` every post equals the serial bytes.

use unpopped::ir::{Expr, OpDef, ReduceOp, ScalarExpr, input, konst, reduced};
use unpopped::kernel::{KernelFn, check_kernel};
use unpopped::kernel_lift::{lift_reduction_opdef, lift_reduction_opdef_launched};
use unpopped::oracle::TypedBuffer;
use unpopped::oracle::kernel_eval::{KernelArg, run_kernel};
use unpopped::oracle::kernel_launch::run_launch;
use unpopped_vocab::{AxisMask, ElementKind};

const F32: ElementKind = ElementKind::F32;
const F64: ElementKind = ElementKind::F64;
const I32: ElementKind = ElementKind::I32;
const GEOMETRIES: [(u32, u32); 5] = [(1, 1), (1, 2), (3, 8), (4, 32), (7, 16)];

fn mask(axes: &[u8]) -> AxisMask {
    let mut m = AxisMask::EMPTY;
    for &a in axes {
        m.set(a);
    }
    m
}

fn red(dtype: ElementKind, rop: ReduceOp, axes: &[u8], keepdim: bool) -> OpDef {
    OpDef::reduction_axes("r", 1, &[dtype], Expr(input(0).0), rop, mask(axes), keepdim)
}

fn values(dtype: ElementKind, n: usize) -> Vec<f64> {
    (0..n)
        .map(|i| {
            let v = ((i * 37 + 11) % 23) as f64 * 0.731 - 7.5 + 1.0e6 * ((i % 5) as f64 - 2.0);
            match dtype {
                ElementKind::I32 => ((i * 37 + 11) % 41) as f64 - 20.0,
                F32 => f64::from(v as f32),
                _ => v,
            }
        })
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

fn strides(shape: &[i64]) -> Vec<i64> {
    (0..shape.len())
        .map(|d| shape[d + 1..].iter().product())
        .collect()
}

/// Run `k` on a dense input of `shape`; `out_shape` is the output's. The output cells.
fn run(
    k: &KernelFn,
    dtype: ElementKind,
    shape: &[i64],
    out_shape: &[i64],
    launch: bool,
) -> Vec<f64> {
    let n: i64 = shape.iter().product();
    let out_n: i64 = out_shape.iter().product();
    let w = if dtype == F64 { 8 } else { 4 };
    let mut a = vec![
        KernelArg::Buffer(TypedBuffer::new(
            dtype,
            shape.to_vec(),
            strides(shape),
            to_bytes(dtype, &values(dtype, n as usize)),
        )),
        KernelArg::Buffer(TypedBuffer::new(
            dtype,
            out_shape.to_vec(),
            strides(out_shape),
            vec![0xAB; out_n as usize * w],
        )),
    ];
    a.extend(shape.iter().map(|&d| KernelArg::Int(i128::from(d))));
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

fn launched(op: &OpDef, dtype: ElementKind, rank: u8, block: u32, grid: u32) -> KernelFn {
    let k = lift_reduction_opdef_launched(op, dtype, rank, block, grid).expect("lifts");
    assert_eq!(check_kernel(&k), Ok(()), "a lifted kernel is well formed");
    k
}

/// The sum in the written order over `groups` (each thread's elements in row-major order): thread
/// `t` folds `groups[t], groups[t+B], ...`, the tree adds `acc[t] += acc[t+s]`, one rounding.
fn documented(groups: &[Vec<f64>], block: usize, divisor: f64, dtype: ElementKind) -> f64 {
    let mut acc: Vec<f64> = (0..block)
        .map(|t| {
            groups
                .iter()
                .skip(t)
                .step_by(block)
                .flatten()
                .fold(0.0, |a, &x| a + x)
        })
        .collect();
    let mut s = block / 2;
    while s > 0 {
        for t in 0..s {
            acc[t] += acc[t + s];
        }
        s /= 2;
    }
    let total = acc[0] / divisor;
    if dtype == F32 {
        f64::from(total as f32)
    } else {
        total
    }
}

const SHAPE3: [i64; 3] = [4, 6, 5];

#[test]
fn several_reduced_axes_equal_the_serial_kernel_for_order_independent_folds() {
    for (dtype, rop) in [
        (I32, ReduceOp::Sum),
        (I32, ReduceOp::Max),
        (F32, ReduceOp::Max),
        (F64, ReduceOp::Min),
    ] {
        for (axes, keepdim, out) in [
            (&[1u8, 2][..], false, vec![4i64]),
            (&[1, 2], true, vec![4, 1, 1]),
            (&[0, 1], false, vec![5]),
            (&[0, 2], false, vec![6]),
        ] {
            let op = red(dtype, rop, axes, keepdim);
            let want = run(
                &lift_reduction_opdef(&op, dtype, 3).unwrap(),
                dtype,
                &SHAPE3,
                &out,
                false,
            );
            for (grid, block) in GEOMETRIES {
                let got = run(
                    &launched(&op, dtype, 3, block, grid),
                    dtype,
                    &SHAPE3,
                    &out,
                    true,
                );
                assert_eq!(
                    got, want,
                    "{dtype:?} {rop:?} axes {axes:?} grid {grid} block {block}"
                );
            }
        }
    }
}

#[test]
fn a_float_sum_over_two_axes_follows_the_written_order() {
    let xs = values(F64, 4 * 6 * 5);
    for rop in [ReduceOp::Sum, ReduceOp::Mean] {
        let op = red(F64, rop, &[1, 2], false);
        let div = if rop == ReduceOp::Mean { 30.0 } else { 1.0 };
        for (grid, block) in GEOMETRIES {
            let got = run(
                &launched(&op, F64, 3, block, grid),
                F64,
                &SHAPE3,
                &[4],
                true,
            );
            for (a, g) in got.iter().enumerate() {
                // Group b holds that slab's 5 elements; thread t takes b = t, t+B, ...
                let groups: Vec<Vec<f64>> = (0..6)
                    .map(|b| (0..5).map(|c| xs[a * 30 + b * 5 + c]).collect())
                    .collect();
                let want = documented(&groups, block as usize, div, F64);
                assert_eq!(
                    g.to_bits(),
                    want.to_bits(),
                    "{rop:?} out {a} grid {grid} block {block}"
                );
            }
        }
        let serial = run(
            &lift_reduction_opdef(&op, F64, 3).unwrap(),
            F64,
            &SHAPE3,
            &[4],
            false,
        );
        assert_eq!(
            run(&launched(&op, F64, 3, 1, 2), F64, &SHAPE3, &[4], true),
            serial
        );
    }
}

#[test]
fn a_reduction_with_no_kept_axis_has_one_output_whatever_the_grid() {
    let n = 53i64;
    for (dtype, rop) in [
        (I32, ReduceOp::Sum),
        (F32, ReduceOp::Max),
        (F64, ReduceOp::Min),
    ] {
        let op = red(dtype, rop, &[], false);
        let want = run(
            &lift_reduction_opdef(&op, dtype, 1).unwrap(),
            dtype,
            &[n],
            &[],
            false,
        );
        assert_eq!(want.len(), 1);
        for (grid, block) in GEOMETRIES {
            let got = run(
                &launched(&op, dtype, 1, block, grid),
                dtype,
                &[n],
                &[],
                true,
            );
            assert_eq!(got, want, "{dtype:?} {rop:?} grid {grid} block {block}");
        }
    }
    let xs = values(F64, n as usize);
    let groups: Vec<Vec<f64>> = xs.iter().map(|&x| vec![x]).collect();
    for rop in [ReduceOp::Sum, ReduceOp::Mean] {
        let op = red(F64, rop, &[], false);
        let div = if rop == ReduceOp::Mean { n as f64 } else { 1.0 };
        for (grid, block) in GEOMETRIES {
            let got = run(&launched(&op, F64, 1, block, grid), F64, &[n], &[], true);
            let want = documented(&groups, block as usize, div, F64);
            assert_eq!(
                got[0].to_bits(),
                want.to_bits(),
                "{rop:?} grid {grid} block {block}"
            );
        }
    }
}

fn with_post(dtype: ElementKind, rop: ReduceOp, post: Expr) -> OpDef {
    OpDef::reduction_post("p", 1, &[dtype], input(0) * input(0), rop, post)
}

const POST_SHAPE: [i64; 2] = [9, 37];
const POST_OUT: [i64; 1] = [9];

fn serial_cells(op: &OpDef, dtype: ElementKind) -> Vec<f64> {
    let k = lift_reduction_opdef(op, dtype, 2).unwrap();
    run(&k, dtype, &POST_SHAPE, &POST_OUT, false)
}

fn launched_cells(op: &OpDef, dtype: ElementKind, block: u32, grid: u32) -> Vec<f64> {
    let k = launched(op, dtype, 2, block, grid);
    run(&k, dtype, &POST_SHAPE, &POST_OUT, true)
}

#[test]
fn a_max_post_is_bit_exact_against_the_serial_kernel() {
    for dtype in [F32, F64] {
        // Max is order-independent, so the whole op (fold, then post) equals the serial kernel.
        let smax = OpDef::reduction_post(
            "smax",
            1,
            &[dtype],
            Expr(input(0).0),
            ReduceOp::Max,
            reduced(0) * konst(0.3) - konst(1.0),
        );
        let want = serial_cells(&smax, dtype);
        for (grid, block) in GEOMETRIES {
            let got = launched_cells(&smax, dtype, block, grid);
            assert_eq!(got, want, "{dtype:?} smax grid {grid} block {block}");
        }
    }
}

#[test]
fn a_sum_or_mean_post_is_serial_at_block_one_and_close_elsewhere() {
    // Block 1 is the serial order; other blocks stay within a relative tolerance of it (the Sum
    // differs by the E1d-2 bound, and the sqrt halves the relative difference).
    for dtype in [F32, F64] {
        let tol = if dtype == F32 { 1e-5 } else { 1e-12 };
        for (rop, post) in [
            (ReduceOp::Sum, reduced(0).sqrt()),
            (ReduceOp::Mean, (reduced(0) + konst(1.0e-3)).sqrt()),
        ] {
            let op = with_post(dtype, rop, post);
            let serial = serial_cells(&op, dtype);
            assert_eq!(launched_cells(&op, dtype, 1, 3), serial);
            for (grid, block) in GEOMETRIES {
                let got = launched_cells(&op, dtype, block, grid);
                for (g, w) in got.iter().zip(&serial) {
                    assert!(
                        (g - w).abs() <= tol * w.abs().max(1.0),
                        "{dtype:?} {rop:?} {g} vs {w}"
                    );
                }
            }
        }
    }
}

#[test]
fn a_post_that_is_not_the_identity_is_actually_applied() {
    // Control: the post changes the answer (so the tests above are not secretly post-free).
    let plain = red(F64, ReduceOp::Max, &[], false);
    let posted = OpDef::reduction_post(
        "p",
        1,
        &[F64],
        Expr(input(0).0),
        ReduceOp::Max,
        reduced(0) * konst(2.0),
    );
    let a = run(&launched(&plain, F64, 1, 8, 2), F64, &[53], &[], true);
    let b = run(&launched(&posted, F64, 1, 8, 2), F64, &[53], &[], true);
    assert_eq!(b[0], 2.0 * a[0]);
    let _ = ScalarExpr::Reduced(0);
}
