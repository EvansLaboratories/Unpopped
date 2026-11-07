// SPDX-License-Identifier: MIT OR Apache-2.0
//! **The serial C of tensor-core attention agrees with the launch interpreter** (ROADMAP U10 A6,
//! `docs/options-u10-attention.md`): two `Stmt::Mma` for the scores, a scalar row softmax between block barriers, and one
//! more `Stmt::Mma` for the output, compiled with a host C compiler and run, match `run_launch` within 2 ulps (the C library's
//! `exp` need not equal the interpreter's bit for bit). The shipped half-precision atoms have no C element type here, so the
//! atom is the `F32` toy `m4n4k8`.

use unpopped::backend::Backend;
use unpopped::oracle::TypedBuffer;
use unpopped::oracle::kernel_eval::KernelArg;
use unpopped_cpu_c::CpuC;

#[expect(
    dead_code,
    reason = "the shared kernel and data; the host reference is the oracle test's"
)]
#[path = "support/attention_mma.rs"]
mod attention_mma;
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
use attention_mma::{F32, K, M, N, U32, id_table, logical_buffers, mma_attention};
use launch_run::agree_launch_within;

fn f32_buf(v: &[f32]) -> KernelArg {
    KernelArg::Buffer(TypedBuffer::new(
        F32,
        vec![v.len() as i64],
        vec![1],
        v.iter().flat_map(|x| x.to_le_bytes()).collect(),
    ))
}

fn u32_buf(v: &[u32]) -> KernelArg {
    KernelArg::Buffer(TypedBuffer::new(
        U32,
        vec![v.len() as i64],
        vec![1],
        v.iter().flat_map(|x| x.to_le_bytes()).collect(),
    ))
}

#[test]
fn the_c_mma_attention_agrees_with_the_oracle() {
    let k = mma_attention();
    assert!(CpuC.lower_kernel(&k).is_ok(), "the kernel lowers");
    let [q, k0, k1, v] = logical_buffers();
    let args = vec![
        f32_buf(&q),
        f32_buf(&k0),
        f32_buf(&k1),
        f32_buf(&v),
        u32_buf(&id_table(0)),
        u32_buf(&id_table(1)),
        u32_buf(&id_table(2)),
        f32_buf(&[0.0; M * N]),
        f32_buf(&[0.0; M * N]),
        f32_buf(&[0.0; M * K]),
        f32_buf(&[0.0; M * N]),
    ];
    agree_launch_within("mma attention", &k, args, 2);
}
