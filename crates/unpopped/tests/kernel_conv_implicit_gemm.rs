// SPDX-License-Identifier: MIT OR Apache-2.0
//! **Implicit-GEMM convolution in the oracle** (ROADMAP U9 K7, `docs/options-u9-gemm-family.md`): NHWC forward
//! convolution as a GEMM over `(r, s, c)` with the im2col index math in the loop, no staged matrix. The shapes make an
//! index mix-up loud (non-square image and filter, padding on height only, several channels). Checked against a direct
//! host convolution.

use unpopped::kernel::check_kernel;
use unpopped::oracle::TypedBuffer;
use unpopped::oracle::kernel_eval::KernelArg;
use unpopped::oracle::kernel_launch::run_launch;

#[path = "support/conv.rs"]
mod conv;
use conv::{BATCH, C, H, I32, KOUT, P, Q, R, S, W, conv_data, conv_implicit_gemm, conv_ref};

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

/// `(args, expected out)`.
fn args() -> (Vec<KernelArg>, Vec<i32>) {
    let (input, filter) = conv_data();
    let want = conv_ref(&input, &filter);
    (
        vec![
            buf(&[BATCH, H, W, C], &input),
            buf(&[R, S, C, KOUT], &filter),
            buf(&[BATCH, P, Q, KOUT], &vec![0; want.len()]),
        ],
        want,
    )
}

#[test]
fn the_implicit_gemm_convolution_matches_the_direct_host_loop() {
    let k = conv_implicit_gemm();
    assert_eq!(check_kernel(&k), Ok(()));
    let (mut args, want) = args();
    run_launch(&k, &mut args).unwrap();
    assert_eq!(words(&args[2]), want);
    assert!(want.iter().any(|v| *v != 0));
}
