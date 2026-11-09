// SPDX-License-Identifier: MIT OR Apache-2.0
//! **The implicit-GEMM convolution family in the oracle** (ROADMAP R5.7 F7): forward, data gradient and filter gradient of an
//! NHWC convolution with stride `(3, 1)`, dilation `(2, 3)` and padding `(1, 1)`, written with the kernel layer alone (the
//! kernels are `support/conv_family.rs`). Each is checked against a host **scatter** reference, whose loop order differs
//! from the kernel's gather, on data where an index mix-up changes the answer.

use unpopped::kernel::check_kernel;
use unpopped::oracle::TypedBuffer;
use unpopped::oracle::kernel_eval::KernelArg;
use unpopped::oracle::kernel_launch::run_launch;

#[path = "support/conv_family.rs"]
mod conv_family;
use conv_family::{
    C, DH, DW, H, I32, K, N, P, PH, PW, Q, R, S, SH, SW, W, data, dgrad, fprop, reference, wgrad,
};

fn buf(shape: &[i64], v: &[i32]) -> KernelArg {
    let strides: Vec<i64> = (0..shape.len())
        .map(|d| shape[d + 1..].iter().product())
        .collect();
    KernelArg::Buffer(TypedBuffer::new(
        I32,
        shape.to_vec(),
        strides,
        v.iter().flat_map(|x| x.to_le_bytes()).collect(),
    ))
}

fn words(a: &KernelArg) -> Vec<i32> {
    let KernelArg::Buffer(b) = a else {
        unreachable!("a buffer")
    };
    b.bytes
        .chunks(4)
        .map(|c| i32::from_le_bytes(c.try_into().unwrap()))
        .collect()
}

const IN: [i64; 4] = [N, H, W, C];
const FILT: [i64; 4] = [R, S, C, K];
const OUT: [i64; 4] = [N, P, Q, K];

fn zeros(shape: &[i64]) -> KernelArg {
    buf(shape, &vec![0; shape.iter().product::<i64>() as usize])
}

#[test]
fn the_forward_convolution_matches_the_scatter_reference() {
    let (input, filter, d_out) = data();
    let (want, _, _) = reference(&input, &filter, &d_out);
    let k = fprop();
    assert_eq!(check_kernel(&k), Ok(()));
    let mut args = vec![buf(&IN, &input), buf(&FILT, &filter), zeros(&OUT)];
    run_launch(&k, &mut args).unwrap();
    assert_eq!(words(&args[2]), want);
    assert!(want.iter().any(|v| *v != 0) && want.windows(2).any(|w| w[0] != w[1]));
}

#[test]
fn the_data_gradient_matches_the_scatter_reference() {
    let (input, filter, d_out) = data();
    let (_, want, _) = reference(&input, &filter, &d_out);
    let k = dgrad();
    assert_eq!(check_kernel(&k), Ok(()));
    let mut args = vec![buf(&OUT, &d_out), buf(&FILT, &filter), zeros(&IN)];
    run_launch(&k, &mut args).unwrap();
    assert_eq!(words(&args[2]), want);
    assert!(want.iter().any(|v| *v != 0) && want.windows(2).any(|w| w[0] != w[1]));
}

#[test]
fn the_filter_gradient_matches_the_scatter_reference() {
    let (input, filter, d_out) = data();
    let (_, _, want) = reference(&input, &filter, &d_out);
    let k = wgrad();
    assert_eq!(check_kernel(&k), Ok(()));
    let mut args = vec![buf(&IN, &input), buf(&OUT, &d_out), zeros(&FILT)];
    run_launch(&k, &mut args).unwrap();
    assert_eq!(words(&args[2]), want);
    assert!(want.iter().any(|v| *v != 0) && want.windows(2).any(|w| w[0] != w[1]));
}

#[test]
fn the_geometry_exercises_stride_dilation_and_padding_at_both_ends() {
    // the output is 2 x 5 from a 6 x 6 image: a stride of 3 and a dilation of 2 on H shrink it, a dilation of 3 on W with
    // padding 1 takes one column off
    assert_eq!((P, Q), (2, 5));
    // the row and column parameters differ (and a stride is not the dilation beside it), so a mix-up changes the answer
    let differ = |a: i64, b: i64| a != b;
    assert!(differ(SH, SW) && differ(DH, DW) && differ(DH, SH));
    // a tap that falls off the image exists at both ends of each axis, so a dropped bounds check changes the answer
    let pos = |o: i64, t: i64, stride: i64, dil: i64, pad: i64| o * stride + t * dil - pad;
    let rows: Vec<i64> = (0..P)
        .flat_map(|p| (0..R).map(move |r| pos(p, r, SH, DH, PH)))
        .collect();
    assert!(
        rows.iter().any(|&i| i < 0) && rows.iter().any(|&i| i >= H),
        "off the top and the bottom: {rows:?}"
    );
    let cols: Vec<i64> = (0..Q)
        .flat_map(|q| (0..S).map(move |s| pos(q, s, SW, DW, PW)))
        .collect();
    assert!(
        cols.iter().any(|&i| i < 0) && cols.iter().any(|&i| i >= W),
        "off the left and the right: {cols:?}"
    );
}
