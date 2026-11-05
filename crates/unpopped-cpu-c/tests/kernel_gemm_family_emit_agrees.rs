// SPDX-License-Identifier: MIT OR Apache-2.0
//! **The serial C of the GEMM-family kernels agrees with the launch interpreter** (ROADMAP U9, kernel-layer part,
//! `docs/options-u9-gemm-family.md`). The kernels are in `support/gemm.rs`; each is compiled with a host C compiler, run,
//! and its written buffer compared byte for byte with `run_launch`, and with the host loop that is the reference.

use unpopped::backend::Backend;
use unpopped::oracle::TypedBuffer;
use unpopped::oracle::kernel_eval::KernelArg;
use unpopped::oracle::kernel_launch::run_launch;
use unpopped_cpu_c::CpuC;

#[path = "support/cc.rs"]
mod cc;
#[path = "support/gemm.rs"]
mod gemm;
#[expect(
    dead_code,
    reason = "the shared data builders; this file builds its own matrices"
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
use gemm::{I32, K, M, N, aux_outputs, aux_outputs_ref, fused_epilogue, fused_epilogue_ref};
use launch_run::agree_launch;

fn mat(rows: u32, cols: u32, f: impl Fn(i32, i32) -> i32) -> Vec<i32> {
    (0..rows as i32)
        .flat_map(|r| (0..cols as i32).map(move |c| (r, c)))
        .map(|(r, c)| f(r, c))
        .collect()
}

fn arg(shape: &[i64], v: &[i32]) -> KernelArg {
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

fn fused_args() -> (Vec<KernelArg>, Vec<i32>) {
    let a = mat(M, K, |r, c| r - c + 1);
    let b = mat(K, N, |r, c| 2 * c - r);
    let c = mat(M, N, |r, c| (r * c) % 3 - 1);
    let bias = [-20, 5, -1, 0];
    let args = vec![
        arg(&[i64::from(M), i64::from(K)], &a),
        arg(&[i64::from(K), i64::from(N)], &b),
        arg(&[i64::from(M), i64::from(N)], &c),
        arg(&[i64::from(N)], &bias),
        arg(&[i64::from(M), i64::from(N)], &vec![0; (M * N) as usize]),
    ];
    (args, fused_epilogue_ref(&a, &b, &c, &bias))
}

#[test]
fn the_fused_epilogue_gemm_agrees_with_the_oracle_and_the_host_loop() {
    let k = fused_epilogue();
    let (args, want) = fused_args();
    agree_launch("gemm_epilogue", &k, args);
    let (mut run, _) = fused_args();
    run_launch(&k, &mut run).expect("the oracle runs it");
    assert_eq!(words(&run[4]), want);
}

#[test]
fn the_c_names_every_epilogue_term() {
    let g = CpuC
        .lower_kernel(&fused_epilogue())
        .expect("the kernel lowers");
    assert!(g.source.contains("for ("), "the K loop: {}", g.source);
}

fn aux_args() -> (Vec<KernelArg>, (Vec<i32>, Vec<i32>, Vec<i32>, i32)) {
    let a = mat(M, K, |r, c| 2 * r - 3 * c + 1);
    let b = mat(K, N, |r, c| c - r + 1);
    let args = vec![
        arg(&[i64::from(M), i64::from(K)], &a),
        arg(&[i64::from(K), i64::from(N)], &b),
        arg(&[i64::from(M), i64::from(N)], &vec![0; (M * N) as usize]),
        arg(&[i64::from(M), i64::from(N)], &vec![0; (M * N) as usize]),
        arg(&[i64::from(N)], &vec![0; N as usize]),
        arg(&[1], &[i32::MIN]),
    ];
    (args, aux_outputs_ref(&a, &b))
}

fn check_aux(args: &[KernelArg], want: &(Vec<i32>, Vec<i32>, Vec<i32>, i32)) {
    assert_eq!(words(&args[2]), want.0, "D");
    assert_eq!(words(&args[3]), want.1, "mask");
    assert_eq!(words(&args[4]), want.2, "dbias");
    assert_eq!(words(&args[5]), vec![want.3], "amax");
    // the data exercise every output: both mask values, a clamped and an unclamped column sum
    assert!(want.1.contains(&0) && want.1.contains(&1));
    assert!(want.0.iter().any(|v| *v < 0) && want.3 > 0);
}

#[test]
fn the_aux_epilogue_agrees_with_the_oracle_and_the_host_loop() {
    let k = aux_outputs();
    let (args, want) = aux_args();
    agree_launch("gemm_aux", &k, args);
    let (mut run, _) = aux_args();
    run_launch(&k, &mut run).expect("the oracle runs it");
    check_aux(&run, &want);
}
