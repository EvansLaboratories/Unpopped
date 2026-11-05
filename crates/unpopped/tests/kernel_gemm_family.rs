// SPDX-License-Identifier: MIT OR Apache-2.0
//! **GEMM-family kernels in the oracle** (ROADMAP U9, kernel-layer part, `docs/options-u9-gemm-family.md`): an integer
//! GEMM with a fused epilogue (`relu(alpha * acc + beta * C + bias[col])`), written with the kernel layer alone and
//! checked against a plain host loop. The matrices include negatives so that the `relu` and the `bias` matter.

use unpopped::kernel::check_kernel;
use unpopped::oracle::TypedBuffer;
use unpopped::oracle::kernel_eval::KernelArg;
use unpopped::oracle::kernel_launch::run_launch;

#[path = "support/gemm.rs"]
mod gemm;
use gemm::{I32, K, M, N, aux_outputs, aux_outputs_ref, fused_epilogue, fused_epilogue_ref};

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

#[test]
fn the_fused_epilogue_gemm_matches_the_host_loop() {
    let k = fused_epilogue();
    assert_eq!(check_kernel(&k), Ok(()));
    let a = mat(M, K, |r, c| r - c + 1);
    let b = mat(K, N, |r, c| 2 * c - r);
    let c = mat(M, N, |r, c| (r * c) % 3 - 1);
    let bias = [-20, 5, -1, 0];
    let mut args = [
        arg(&[M as i64, K as i64], &a),
        arg(&[K as i64, N as i64], &b),
        arg(&[M as i64, N as i64], &c),
        arg(&[N as i64], &bias),
        arg(&[M as i64, N as i64], &vec![0; (M * N) as usize]),
    ];
    run_launch(&k, &mut args).unwrap();
    let want = fused_epilogue_ref(&a, &b, &c, &bias);
    assert_eq!(words(&args[4]), want);
    // the data really exercise the epilogue: some outputs are clamped, some are not
    assert!(want.contains(&0), "a relu-clamped output");
    assert!(want.iter().any(|v| *v > 0), "an unclamped output");
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
fn the_aux_epilogue_writes_d_the_mask_dbias_and_amax() {
    let k = aux_outputs();
    assert_eq!(check_kernel(&k), Ok(()));
    let (mut args, want) = aux_args();
    run_launch(&k, &mut args).unwrap();
    check_aux(&args, &want);
}
