// SPDX-License-Identifier: MIT OR Apache-2.0
//! **The serial C of the grouped GEMM agrees with the launch interpreter** (ROADMAP U9 K5,
//! `docs/options-u9-gemm-family.md`): compiled with a host C compiler, run, and its `d` compared byte for byte with
//! `run_launch` and with the host loop.

use unpopped::backend::Backend;
use unpopped::oracle::TypedBuffer;
use unpopped::oracle::kernel_eval::KernelArg;
use unpopped::oracle::kernel_launch::run_launch;
use unpopped_cpu_c::CpuC;
use unpopped_vocab::ElementKind;

#[path = "support/cc.rs"]
mod cc;
#[path = "support/grouped.rs"]
mod grouped;
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
use grouped::{COLS, I32, PROBLEMS, U32, grouped_data, grouped_gemm};
use launch_run::agree_launch;

fn buf(dtype: ElementKind, shape: &[i64], bytes: Vec<u8>) -> KernelArg {
    let strides: Vec<i64> = (0..shape.len())
        .map(|d| shape[d + 1..].iter().product())
        .collect();
    KernelArg::Buffer(TypedBuffer::new(dtype, shape.to_vec(), strides, bytes))
}

fn i32s(v: &[i32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
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

/// `(args, expected D)`: the table, `a`, `b`, and a zeroed `d` one element longer than needed on each side is not used;
/// `d` is exactly the packed outputs, so an out-of-problem write is out of bounds or overwrites a neighbour.
fn args() -> (Vec<KernelArg>, Vec<i32>) {
    let (tbl, a, b, dlen, d) = grouped_data();
    let tbl_bytes: Vec<u8> = tbl.iter().flat_map(|x| x.to_le_bytes()).collect();
    (
        vec![
            buf(U32, &[PROBLEMS.len() as i64, COLS as i64], tbl_bytes),
            buf(I32, &[a.len() as i64], i32s(&a)),
            buf(I32, &[b.len() as i64], i32s(&b)),
            buf(I32, &[dlen as i64], i32s(&vec![0; dlen])),
        ],
        d,
    )
}

#[test]
fn the_grouped_gemm_agrees_with_the_oracle_and_the_host_loop() {
    let k = grouped_gemm();
    let (a, want) = args();
    agree_launch("gemm_grouped", &k, a);
    let (mut run, _) = args();
    run_launch(&k, &mut run).expect("the oracle runs it");
    assert_eq!(words(&run[3]), want);
}

#[test]
fn the_c_loops_over_k_from_the_table() {
    let g = CpuC
        .lower_kernel(&grouped_gemm())
        .expect("the kernel lowers");
    assert!(g.source.contains("for ("), "{}", g.source);
}
