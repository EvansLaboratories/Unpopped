// SPDX-License-Identifier: MIT OR Apache-2.0
//! **The serial C of flash-style attention agrees with the launch interpreter** (ROADMAP U10 A2,
//! `docs/options-u10-attention.md`): compiled with a host C compiler and run, the output matches `run_launch` within 2 ulps.

use unpopped::backend::Backend;
use unpopped::oracle::TypedBuffer;
use unpopped::oracle::kernel_eval::KernelArg;
use unpopped_cpu_c::CpuC;

#[expect(
    dead_code,
    reason = "the shared kernels and data; this file uses the flash kernel"
)]
#[path = "support/attention.rs"]
mod attention;
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
use attention::{DV, F32, LQ, attention_data, flash_attention};
use launch_run::agree_launch_within;

fn f32_buf(v: &[f32]) -> KernelArg {
    KernelArg::Buffer(TypedBuffer::new(
        F32,
        vec![v.len() as i64],
        vec![1],
        v.iter().flat_map(|x| x.to_le_bytes()).collect(),
    ))
}

#[test]
fn the_c_flash_attention_agrees_with_the_oracle() {
    let k = flash_attention();
    assert!(CpuC.lower_kernel(&k).is_ok(), "the kernel lowers");
    let (q, kk, v) = attention_data();
    let args = vec![
        f32_buf(&q),
        f32_buf(&kk),
        f32_buf(&v),
        f32_buf(&[0.0; LQ * DV]),
    ];
    agree_launch_within("flash attention", &k, args, 2);
}
