// SPDX-License-Identifier: MIT OR Apache-2.0
//! **The serial C of naive attention agrees with the launch interpreter** (ROADMAP U10 A1, `docs/options-u10-attention.md`):
//! compiled with a host C compiler and run, the scores and the output match `run_launch` within 2 ulps (the C library's `exp`
//! need not equal the interpreter's bit for bit).

use unpopped::backend::Backend;
use unpopped::oracle::TypedBuffer;
use unpopped::oracle::kernel_eval::KernelArg;
use unpopped_cpu_c::CpuC;

#[expect(
    dead_code,
    reason = "the shared kernel and data; the host reference is the oracle test's"
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
use attention::{DV, F32, LK, LQ, attention_data, naive_attention};
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
fn the_c_naive_attention_agrees_with_the_oracle() {
    let k = naive_attention();
    assert!(CpuC.lower_kernel(&k).is_ok(), "the kernel lowers");
    let (q, kk, v) = attention_data();
    let args = vec![
        f32_buf(&q),
        f32_buf(&kk),
        f32_buf(&v),
        f32_buf(&[0.0; LQ * LK]),
        f32_buf(&[0.0; LQ * DV]),
    ];
    agree_launch_within("naive attention", &k, args, 2);
}
