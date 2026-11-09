// SPDX-License-Identifier: MIT OR Apache-2.0
//! **The C of the implicit-GEMM convolution family agrees with the oracle** (ROADMAP R5.7 F7; the kernels and the host
//! scatter references are `support/conv_family.rs`, checked in `crates/unpopped/tests/kernel_conv_family.rs`): forward, data
//! gradient and filter gradient with stride `(3, 1)`, dilation `(2, 3)` and padding `(1, 1)`, lowered by
//! `unpopped_cpu_c::CpuC::lower_kernel`, compiled and run with a host C compiler, and compared byte for byte with
//! `run_launch`. The integer divisions of the data gradient go through the C helpers the oracle's totality rule defines.

use unpopped::oracle::TypedBuffer;
use unpopped::oracle::kernel_eval::KernelArg;
use unpopped_cpu_c as _;

#[path = "support/cc.rs"]
mod cc;
// The shared compile-and-run drivers; this file uses a subset of them.
// The host references of the shared file are used by the oracle test only.
#[allow(dead_code)]
#[path = "support/conv_family.rs"]
mod conv_family;
#[allow(dead_code)]
#[path = "support/kernel_run.rs"]
mod kernel_run;
#[allow(dead_code)]
#[path = "support/launch_run.rs"]
mod launch_run;
use conv_family::{C, H, I32, K, N, P, Q, R, S, W, data, dgrad, fprop, wgrad};
use launch_run::agree_launch;

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

fn zeros(shape: &[i64]) -> KernelArg {
    buf(shape, &vec![0; shape.iter().product::<i64>() as usize])
}

const IN: [i64; 4] = [N, H, W, C];
const FILT: [i64; 4] = [R, S, C, K];
const OUT: [i64; 4] = [N, P, Q, K];

#[test]
fn the_forward_convolution_agrees() {
    let (input, filter, _) = data();
    agree_launch(
        "conv_fprop",
        &fprop(),
        vec![buf(&IN, &input), buf(&FILT, &filter), zeros(&OUT)],
    );
}

#[test]
fn the_data_gradient_agrees() {
    let (_, filter, d_out) = data();
    agree_launch(
        "conv_dgrad",
        &dgrad(),
        vec![buf(&OUT, &d_out), buf(&FILT, &filter), zeros(&IN)],
    );
}

#[test]
fn the_filter_gradient_agrees() {
    let (input, _, d_out) = data();
    agree_launch(
        "conv_wgrad",
        &wgrad(),
        vec![buf(&IN, &input), buf(&OUT, &d_out), zeros(&FILT)],
    );
}
