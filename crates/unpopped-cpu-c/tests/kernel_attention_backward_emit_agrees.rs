// SPDX-License-Identifier: MIT OR Apache-2.0
//! **The serial C of the attention backward kernels agrees with the launch interpreter** (ROADMAP U10 A5,
//! `docs/options-u10-attention.md`): compiled with a host C compiler and run, `dQ`, `dK` and `dV` match `run_launch` within
//! 3 ulps (the oracle rounds once per statement, C per operation; the kernels are multi-operation).

use unpopped::backend::Backend;
use unpopped::oracle::TypedBuffer;
use unpopped::oracle::kernel_eval::KernelArg;
use unpopped_cpu_c::CpuC;

#[expect(
    dead_code,
    reason = "the shared forward kernels; this file uses their shapes and data"
)]
#[path = "support/attention.rs"]
mod attention;
// The kernels are written once, in the oracle crate's support file, so the two tests cannot drift.
#[expect(
    dead_code,
    reason = "the host reference and loss belong to the oracle test"
)]
#[path = "../../unpopped/tests/support/attention_backward.rs"]
mod attention_backward;
#[path = "support/cc.rs"]
mod cc;
#[expect(
    dead_code,
    reason = "the shared driver; this file uses the part that runs a launched kernel"
)]
#[path = "support/kernel_run.rs"]
mod kernel_run;
#[expect(
    dead_code,
    reason = "the shared runner; this file uses the tolerance form"
)]
#[path = "support/launch_run.rs"]
mod launch_run;
use attention::{D, DV, F32, LK, LQ};
use attention_backward::{backward_dkv, backward_dq, backward_inputs};
use launch_run::agree_launch_within;

fn f32_buf(v: &[f32]) -> KernelArg {
    KernelArg::Buffer(TypedBuffer::new(
        F32,
        vec![v.len() as i64],
        vec![1],
        v.iter().flat_map(|x| x.to_le_bytes()).collect(),
    ))
}

fn args(outputs: &[usize]) -> Vec<KernelArg> {
    let mut a: Vec<KernelArg> = backward_inputs().iter().map(|v| f32_buf(v)).collect();
    a.extend(outputs.iter().map(|&n| f32_buf(&vec![0.0; n])));
    a
}

#[test]
fn the_c_dq_kernel_agrees_with_the_oracle() {
    let k = backward_dq();
    assert!(CpuC.lower_kernel(&k).is_ok(), "the kernel lowers");
    agree_launch_within("attention backward dQ", &k, args(&[LQ * D]), 3);
}

#[test]
fn the_c_dkv_kernel_agrees_with_the_oracle() {
    let k = backward_dkv();
    assert!(CpuC.lower_kernel(&k).is_ok(), "the kernel lowers");
    agree_launch_within("attention backward dK/dV", &k, args(&[LK * D, LK * DV]), 3);
}
