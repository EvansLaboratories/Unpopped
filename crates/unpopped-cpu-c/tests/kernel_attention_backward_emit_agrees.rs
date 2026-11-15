// SPDX-License-Identifier: MIT OR Apache-2.0
//! **The serial C of the attention backward kernels agrees with the launch interpreter** (ROADMAP U10 A5,
//! `docs/options-u10-attention.md`): compiled with a host C compiler and run, `dQ`, `dK` and `dV` match `run_launch` within
//! an absolute tolerance of 1e-4 of the largest gradient. A gradient is a cancelling sum (`dS = P (dP - delta)` over terms
//! far larger than the result), so an ulp bound means nothing; the oracle rounds once per statement and C per operation.

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
use kernel_run::run_c;
use launch_run::written;
use unpopped::kernel::KernelFn;
use unpopped::oracle::kernel_launch::run_launch;

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

/// Run `k` as C and as the launch interpreter; every written element agrees within `1e-4` of the largest of its buffer.
fn agree_abs(tag: &str, k: &KernelFn, args: Vec<KernelArg>) {
    let Some(c) = run_c(tag, k, &args) else {
        return;
    };
    assert_eq!(c.status, 0, "{tag}: the C reported {}", c.status);
    let mut interp = args;
    run_launch(k, &mut interp).expect("the oracle runs it");
    let (want, _) = written(k, &interp);
    assert_eq!(c.outs.len(), want.len(), "{tag}");
    for (b, (got, want)) in c.outs.iter().zip(&want).enumerate() {
        let f = |x: &[u8]| -> Vec<f32> {
            x.chunks(4)
                .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
                .collect()
        };
        let (got, want) = (f(got), f(want));
        let scale = want.iter().fold(1e-3f32, |a, w| a.max(w.abs()));
        for (i, (g, w)) in got.iter().zip(&want).enumerate() {
            assert!(
                (g - w).abs() <= 1e-4 * scale,
                "{tag}: buffer {b} element {i}: C {g} vs interpreter {w}"
            );
        }
    }
}

#[test]
fn the_c_dq_kernel_agrees_with_the_oracle() {
    let k = backward_dq();
    assert!(CpuC.lower_kernel(&k).is_ok(), "the kernel lowers");
    agree_abs("attention backward dQ", &k, args(&[LQ * D]));
}

#[test]
fn the_c_dkv_kernel_agrees_with_the_oracle() {
    let k = backward_dkv();
    assert!(CpuC.lower_kernel(&k).is_ok(), "the kernel lowers");
    agree_abs("attention backward dKdV", &k, args(&[LK * D, LK * DV]));
}
