// SPDX-License-Identifier: MIT OR Apache-2.0
//! **The serial C of paged-KV decode attention agrees with the launch interpreter** (ROADMAP U10 A4,
//! `docs/options-u10-attention.md`): compiled with a host C compiler and run, the output matches `run_launch` within 2 ulps,
//! with the `I32` block table and lengths read as buffers.

use unpopped::backend::Backend;
use unpopped::oracle::TypedBuffer;
use unpopped::oracle::kernel_eval::KernelArg;
use unpopped_cpu_c::CpuC;
use unpopped_vocab::ElementKind;

#[expect(
    dead_code,
    reason = "the shared kernels and data; this file uses the paged kernel"
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
use attention::{DV, F32, I32, LENS, NSEQ, TABLE, paged_attention, paged_data};
use launch_run::agree_launch_within;

fn buf(ty: ElementKind, bytes: Vec<u8>, n: usize) -> KernelArg {
    KernelArg::Buffer(TypedBuffer::new(ty, vec![n as i64], vec![1], bytes))
}

fn f32_buf(v: &[f32]) -> KernelArg {
    buf(
        F32,
        v.iter().flat_map(|x| x.to_le_bytes()).collect(),
        v.len(),
    )
}

fn i32_buf(v: &[i32]) -> KernelArg {
    buf(
        I32,
        v.iter().flat_map(|x| x.to_le_bytes()).collect(),
        v.len(),
    )
}

#[test]
fn the_c_paged_attention_agrees_with_the_oracle() {
    let k = paged_attention();
    assert!(CpuC.lower_kernel(&k).is_ok(), "the kernel lowers");
    let (q, kp, vp) = paged_data();
    for lens in [LENS, [0, 2, 5]] {
        let args = vec![
            f32_buf(&q),
            f32_buf(&kp),
            f32_buf(&vp),
            i32_buf(&TABLE),
            i32_buf(&lens),
            f32_buf(&[0.0; NSEQ * DV]),
        ];
        agree_launch_within(&format!("paged attention {lens:?}"), &k, args, 2);
    }
}
