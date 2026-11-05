// SPDX-License-Identifier: MIT OR Apache-2.0
//! **The serial C of the implicit-GEMM convolution agrees with the launch interpreter** (ROADMAP U9 K7,
//! `docs/options-u9-gemm-family.md`): compiled with a host C compiler, run, and `out` compared byte for byte with
//! `run_launch` and with the direct host convolution.

use unpopped::backend::Backend;
use unpopped::oracle::TypedBuffer;
use unpopped::oracle::kernel_eval::KernelArg;
use unpopped::oracle::kernel_launch::run_launch;
use unpopped_cpu_c::CpuC;

#[path = "support/cc.rs"]
mod cc;
#[path = "support/conv.rs"]
mod conv;
#[expect(
    dead_code,
    reason = "the shared data builders; this file builds its own"
)]
#[path = "support/kernel_data.rs"]
mod kernel_data;
#[expect(
    dead_code,
    reason = "the shared driver; this file uses the part that runs a launched kernel"
)]
#[path = "support/kernel_run.rs"]
mod kernel_run;
#[path = "support/launch_run.rs"]
mod launch_run;
use conv::{BATCH, C, H, I32, KOUT, P, Q, R, S, W, conv_data, conv_implicit_gemm, conv_ref};
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
fn the_convolution_agrees_with_the_oracle_and_the_direct_host_loop() {
    let k = conv_implicit_gemm();
    let (a, want) = args();
    agree_launch("conv_igemm", &k, a);
    let (mut run, _) = args();
    run_launch(&k, &mut run).expect("the oracle runs it");
    assert_eq!(words(&run[2]), want);
}

#[test]
fn the_c_has_the_three_nested_loops() {
    let g = CpuC
        .lower_kernel(&conv_implicit_gemm())
        .expect("the kernel lowers");
    assert!(g.source.matches("for (").count() >= 3, "{}", g.source);
}
