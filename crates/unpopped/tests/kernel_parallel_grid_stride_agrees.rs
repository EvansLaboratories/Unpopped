// SPDX-License-Identifier: MIT OR Apache-2.0
//! **E1d-6: a grid-stride launched Window and Im2Col are bit-exact against the serial kernels**
//! (`docs/e1d-parallel-mapping.md`, "E1d-6 as built").
//!
//! The outermost output axis becomes a grid-stride loop; every output cell is computed from its
//! own taps in the serial lift's order, so no result depends on `grid` or `block`: float pools
//! (with a NaN, an infinity and a negative zero planted) and the raw-bit im2col gather must equal
//! the serial kernel's bytes on every geometry, including more threads than output cells and a
//! block size that is not a power of two. Two broken copies (a step of 1 and a wrong start) must
//! disagree.

use unpopped::ir::{OpDef, ReduceOp};
use unpopped::kernel::{KernelFn, Stmt, check_kernel};
use unpopped::kernel_lift::{
    LiftDecline, lift_im2col_opdef, lift_im2col_opdef_launched, lift_window_opdef,
    lift_window_opdef_launched,
};
use unpopped::oracle::TypedBuffer;
use unpopped::oracle::kernel_eval::{KernelArg, run_kernel};
use unpopped::oracle::kernel_launch::run_launch;
use unpopped_vocab::ElementKind;

const F32: ElementKind = ElementKind::F32;
/// One thread; two blocks of three (a block that is not a power of two); fewer threads than output
/// cells; more threads than output cells.
const GEOMETRIES: [(u32, u32); 5] = [(1, 1), (2, 3), (3, 8), (4, 32), (16, 64)];
const VALUES: [f64; 9] = [
    1.0e8,
    1.0,
    -1.0e8,
    0.5,
    f64::NAN,
    -0.0,
    f64::INFINITY,
    -3.0,
    0.1,
];

fn strides(shape: &[i64]) -> Vec<i64> {
    (0..shape.len())
        .map(|i| shape[i + 1..].iter().product())
        .collect()
}

fn f32_buffer(shape: &[i64], value: impl Fn(usize) -> f64) -> TypedBuffer {
    let n: i64 = shape.iter().product();
    let bytes = (0..n as usize)
        .flat_map(|i| (value(i) as f32).to_le_bytes())
        .collect();
    TypedBuffer::new(F32, shape.to_vec(), strides(shape), bytes)
}

/// Run `k` on `[input, zeroed output, extents...]`; the output bytes, or the launch error.
fn try_run(
    k: &KernelFn,
    src: &TypedBuffer,
    out_shape: &[i64],
    extents: &[i64],
    launch: bool,
) -> Result<Vec<u8>, String> {
    let n: i64 = out_shape.iter().product();
    let zeroed = TypedBuffer::new(
        F32,
        out_shape.to_vec(),
        strides(out_shape),
        vec![0xAB; n as usize * 4],
    );
    let mut args = vec![KernelArg::Buffer(src.clone()), KernelArg::Buffer(zeroed)];
    args.extend(extents.iter().map(|&e| KernelArg::Int(i128::from(e))));
    if launch {
        run_launch(k, &mut args).map_err(|e| format!("{e:?}"))?;
    } else {
        run_kernel(k, &mut args).map_err(|e| format!("{e:?}"))?;
    }
    let KernelArg::Buffer(out) = &args[1] else {
        unreachable!("the output follows the input")
    };
    Ok(out.bytes.clone())
}

fn run(
    k: &KernelFn,
    src: &TypedBuffer,
    out_shape: &[i64],
    extents: &[i64],
    launch: bool,
) -> Vec<u8> {
    try_run(k, src, out_shape, extents, launch).expect("the kernel runs")
}

const ROWS: i64 = 7;
const K_IN: i64 = 19;
/// size 3, stride 2, dilation 1, pad 1: `(19 + 2 - 3) / 2 + 1`.
const K_OUT: i64 = 10;

fn pool(wop: ReduceOp, include_pad: bool) -> OpDef {
    OpDef::window_simple("w", &[F32], wop, 1, 3, 2, 1, 1, 1, include_pad)
}

fn pool_launched(op: &OpDef, block: u32, grid: u32) -> KernelFn {
    let k = lift_window_opdef_launched(op, F32, 2, block, grid).expect("lifts");
    assert_eq!(check_kernel(&k), Ok(()), "a lifted kernel is well formed");
    k
}

fn pool_input() -> TypedBuffer {
    f32_buffer(&[ROWS, K_IN], |i| VALUES[i % VALUES.len()])
}

#[test]
fn every_pool_is_bit_exact_on_every_geometry() {
    let src = pool_input();
    for wop in [ReduceOp::Sum, ReduceOp::Mean, ReduceOp::Max, ReduceOp::Min] {
        for include_pad in [false, true] {
            let op = pool(wop, include_pad);
            let serial = lift_window_opdef(&op, F32, 2).unwrap();
            let want = run(&serial, &src, &[ROWS, K_OUT], &[ROWS, K_IN, K_OUT], false);
            for (grid, block) in GEOMETRIES {
                let got = run(
                    &pool_launched(&op, block, grid),
                    &src,
                    &[ROWS, K_OUT],
                    &[ROWS, K_IN, K_OUT],
                    true,
                );
                assert_eq!(
                    got, want,
                    "{wop:?} include_pad={include_pad} grid {grid} block {block}"
                );
            }
        }
    }
}

fn im2col_case() -> (OpDef, TypedBuffer, Vec<i64>, Vec<i64>) {
    // [N, C, H, W] = [2, 3, 5, 6]; a 3x2 window, stride (2, 1), pad (1, 1), dilation (1, 2).
    let shape = [2i64, 3, 5, 6];
    let (kh, kw, sh, sw, ph, pw, dh, dw) = (3i64, 2i64, 2i64, 1i64, 1i64, 1i64, 1i64, 2i64);
    let oh = (shape[2] + 2 * ph - dh * (kh - 1) - 1) / sh + 1;
    let ow = (shape[3] + 2 * pw - dw * (kw - 1) - 1) / sw + 1;
    let op = OpDef::im2col_2d("i", F32, (3, 2), (2, 1), (1, 1), (1, 2));
    let src = f32_buffer(&shape, |i| match i {
        2 => f64::NAN,
        5 => -0.0,
        _ => (i + 1) as f64,
    });
    let out = vec![shape[0], shape[1] * kh * kw, oh * ow];
    let extents = vec![shape[0], shape[1], shape[2], shape[3], oh, ow];
    (op, src, out, extents)
}

#[test]
fn im2col_is_bit_exact_on_every_geometry() {
    let (op, src, out, extents) = im2col_case();
    let serial = lift_im2col_opdef(&op, F32).unwrap();
    let want = run(&serial, &src, &out, &extents, false);
    assert!(
        want.chunks(4).any(|c| c == 0.0f32.to_le_bytes()),
        "the case reaches padded taps"
    );
    for (grid, block) in GEOMETRIES {
        let k = lift_im2col_opdef_launched(&op, F32, block, grid).expect("lifts");
        assert_eq!(check_kernel(&k), Ok(()), "a lifted kernel is well formed");
        assert_eq!(
            run(&k, &src, &out, &extents, true),
            want,
            "grid {grid} block {block}"
        );
    }
}

#[test]
fn bad_launches_and_other_families_decline() {
    let op = pool(ReduceOp::Max, false);
    assert!(matches!(
        lift_window_opdef_launched(&op, F32, 2, 0, 2),
        Err(LiftDecline::BadLaunch)
    ));
    assert!(matches!(
        lift_window_opdef_launched(&op, F32, 2, 8, 0),
        Err(LiftDecline::BadLaunch)
    ));
    let ew = OpDef::elementwise("e", 1, &[F32], unpopped::ir::Expr(unpopped::ir::input(0).0));
    assert_eq!(
        lift_window_opdef_launched(&ew, F32, 2, 8, 2),
        Err(LiftDecline::NotWindow)
    );
    assert!(lift_im2col_opdef_launched(&ew, F32, 8, 2).is_err());
}

/// Is a broken copy of the launched pool seen: the launch errors (the race checker) or its bytes
/// differ from the serial kernel's?
fn seen(k: &KernelFn, src: &TypedBuffer, want: &[u8]) -> bool {
    try_run(k, src, &[ROWS, K_OUT], &[ROWS, K_IN, K_OUT], true).map_or(true, |got| got != want)
}

#[test]
fn an_unstrided_loop_and_a_wrong_start_are_seen() {
    let src = pool_input();
    let op = pool(ReduceOp::Sum, false);
    let want = run(
        &lift_window_opdef(&op, F32, 2).unwrap(),
        &src,
        &[ROWS, K_OUT],
        &[ROWS, K_IN, K_OUT],
        false,
    );
    let k = pool_launched(&op, 2, 2);
    assert!(
        !seen(&k, &src, &want),
        "the control: the unbroken kernel is not seen as broken"
    );

    let mut unstrided = k.clone();
    let Some(Stmt::For { step, .. }) = unstrided.body.0.first_mut() else {
        panic!("the outer loop")
    };
    *step = unpopped::ir::ScalarExpr::ConstInt(1, ElementKind::I32);
    assert!(
        seen(&unstrided, &src, &want),
        "every thread doing every row writes the same cells (a data race)"
    );

    let mut offset = k;
    let Some(Stmt::For { start, .. }) = offset.body.0.first_mut() else {
        panic!("the outer loop")
    };
    *start = unpopped::ir::ScalarExpr::ConstInt(1, ElementKind::I32);
    assert!(
        seen(&offset, &src, &want),
        "a start that skips row 0 leaves it unwritten"
    );
}
