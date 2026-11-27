// SPDX-License-Identifier: MIT OR Apache-2.0
//! **GEMM over arbitrary strides at the kernel layer** (CUTLASS survey row 7: `lda`, non-contiguous and transposed operands).
//! `A` is a row-major view padded to `lda = K + 3`, `B` is column-major (`strides = [1, ldb]`, `ldb = K + 2`), and `C` is padded to
//! `ldc = N + 2`. The storage is poisoned outside the views (a read of padding shows by value), the reference is computed from
//! the logical matrices, and the output storage outside `C`'s view must come back untouched. A swapped `B` index and a shifted
//! output row are mutants that stay in range, so they fail by value.

use unpopped::kernel::{KernelFn, check_kernel};
use unpopped::oracle::TypedBuffer;
use unpopped::oracle::kernel_eval::KernelArg;
use unpopped::oracle::kernel_launch::run_launch;

#[path = "support/strided_gemm.rs"]
mod fixture;
use fixture::{F32, K, M, Mutant, N, strided_gemm};

const LDA: usize = K + 3;
const LDB: usize = K + 2;
const LDC: usize = N + 2;
const POISON: f32 = 12345.0;
const CANARY: f32 = -777.0;

fn view(shape: [i64; 2], strides: [i64; 2], storage: &[f32]) -> KernelArg {
    KernelArg::Buffer(TypedBuffer::new(
        F32,
        shape.to_vec(),
        strides.to_vec(),
        storage.iter().flat_map(|x| x.to_le_bytes()).collect(),
    ))
}

/// Row-major `A` storage with `a[m][k]` at `m * LDA + k`, the padding poisoned.
fn a_storage(a: &[Vec<f32>]) -> Vec<f32> {
    let mut s = vec![POISON; M * LDA];
    for (m, row) in a.iter().enumerate() {
        s[m * LDA..m * LDA + K].copy_from_slice(row);
    }
    s
}

/// Column-major `B` storage with `b[k][n]` at `k + n * LDB`, the padding poisoned.
fn b_storage(b: &[Vec<f32>]) -> Vec<f32> {
    let mut s = vec![POISON; N * LDB];
    for (k, row) in b.iter().enumerate() {
        for (n, v) in row.iter().enumerate() {
            s[k + n * LDB] = *v;
        }
    }
    s
}

fn run(mutant: Mutant, a: &[Vec<f32>], b: &[Vec<f32>]) -> (Vec<Vec<f32>>, Vec<f32>) {
    let k: KernelFn = strided_gemm(mutant);
    assert_eq!(check_kernel(&k), Ok(()));
    let mut args = vec![
        view([M as i64, K as i64], [LDA as i64, 1], &a_storage(a)),
        view([K as i64, N as i64], [1, LDB as i64], &b_storage(b)),
        view([M as i64, N as i64], [LDC as i64, 1], &[CANARY; M * LDC]),
    ];
    run_launch(&k, &mut args).expect("the oracle runs it");
    let KernelArg::Buffer(c) = &args[2] else {
        unreachable!("a buffer")
    };
    let s: Vec<f32> = c
        .bytes
        .chunks(4)
        .map(|x| f32::from_le_bytes(x.try_into().unwrap()))
        .collect();
    let logical = (0..M).map(|m| s[m * LDC..m * LDC + N].to_vec()).collect();
    let padding = (0..M)
        .flat_map(|m| s[m * LDC + N..(m + 1) * LDC].to_vec())
        .collect();
    (logical, padding)
}

fn reference(a: &[Vec<f32>], b: &[Vec<f32>]) -> Vec<Vec<f32>> {
    (0..M)
        .map(|m| {
            (0..N)
                .map(|n| (0..K).fold(0.0f32, |acc, k| acc + a[m][k] * b[k][n]))
                .collect()
        })
        .collect()
}

fn same(got: &[Vec<f32>], want: &[Vec<f32>]) -> bool {
    got.iter()
        .flatten()
        .zip(want.iter().flatten())
        .all(|(g, w)| g.to_bits() == w.to_bits() || (g.is_nan() && w.is_nan()))
}

fn corpus() -> (Vec<Vec<f32>>, Vec<Vec<f32>>) {
    let a = (0..M)
        .map(|m| {
            (0..K)
                .map(|k| ((m * 5 + k * 3) % 11) as f32 - 5.0)
                .collect()
        })
        .collect();
    let b = (0..K)
        .map(|k| {
            (0..N)
                .map(|n| ((k * 7 + n * 2) % 13) as f32 - 6.0)
                .collect()
        })
        .collect();
    (a, b)
}

fn corpus_nan() -> (Vec<Vec<f32>>, Vec<Vec<f32>>) {
    let (mut a, b) = corpus();
    a[1][2] = f32::NAN;
    (a, b)
}

fn corpus_negative() -> (Vec<Vec<f32>>, Vec<Vec<f32>>) {
    let (a, b) = corpus();
    let neg = |v: Vec<Vec<f32>>, d: f32| {
        v.iter()
            .map(|r| r.iter().map(|x| -x.abs() - d).collect())
            .collect()
    };
    (neg(a, 1.0), neg(b, 0.5))
}

#[test]
fn the_strided_gemm_matches_the_logical_reference_and_leaves_the_padding_alone() {
    for (name, (a, b)) in [
        ("ordinary", corpus()),
        ("nan", corpus_nan()),
        ("negative", corpus_negative()),
    ] {
        let (got, padding) = run(Mutant::None, &a, &b);
        let want = reference(&a, &b);
        assert!(same(&got, &want), "{name}: {got:?} vs {want:?}");
        assert!(
            padding.iter().all(|x| *x == CANARY),
            "{name}: padding written"
        );
    }
}

#[test]
fn the_corpus_is_not_blind() {
    // the NaN reaches exactly row 1, B is not symmetric (a transposed read shows), and the padding differs from the data
    let (a, b) = corpus_nan();
    let rows: Vec<bool> = reference(&a, &b)
        .iter()
        .map(|r| r.iter().any(|x| x.is_nan()))
        .collect();
    assert_eq!(rows, vec![false, true, false]);
    let (_, b) = corpus();
    assert!((0..K).any(|k| (0..N).any(|n| b[k][n] != b[n][k])));
}

#[test]
fn the_mutants_fail_by_value() {
    let (a, b) = corpus();
    let want = reference(&a, &b);
    for mutant in [Mutant::SwappedB, Mutant::ShiftedRow] {
        let (got, _) = run(mutant, &a, &b);
        assert!(got.iter().flatten().all(|x| x.is_finite()), "{mutant:?}");
        assert!(!same(&got, &want), "{mutant:?} must differ");
    }
}
