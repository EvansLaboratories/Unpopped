// SPDX-License-Identifier: MIT OR Apache-2.0
//! **Complex GEMM at the kernel layer** (CUTLASS survey rows 26 and 82). Interleaved `(re, im)` `F32` operands, a four-multiply
//! and a three-multiply (Gaussian) kernel, each run on the oracle against an independently written complex reference over small
//! integers (exact in `f32`), with a NaN corpus and an all-negative corpus. A conjugated operand and an imaginary-sign slip are
//! mutants that stay finite and in range, so they fail by value.

use unpopped::kernel::{KernelFn, check_kernel};
use unpopped::oracle::TypedBuffer;
use unpopped::oracle::kernel_eval::KernelArg;
use unpopped::oracle::kernel_launch::run_launch;

#[path = "support/complex_gemm.rs"]
mod fixture;
use fixture::{F32, Form, K, M, Mutant, N, complex_gemm};

type Z = (f32, f32);

fn buf(v: &[f32]) -> KernelArg {
    KernelArg::Buffer(TypedBuffer::new(
        F32,
        vec![v.len() as i64],
        vec![1],
        v.iter().flat_map(|x| x.to_le_bytes()).collect(),
    ))
}

fn flatten(z: &[Z]) -> Vec<f32> {
    z.iter().flat_map(|&(re, im)| [re, im]).collect()
}

fn run(k: &KernelFn, a: &[Z], b: &[Z]) -> Vec<f32> {
    assert_eq!(check_kernel(k), Ok(()));
    let mut args = vec![buf(&flatten(a)), buf(&flatten(b)), buf(&[0.0; M * N * 2])];
    run_launch(k, &mut args).expect("the oracle runs it");
    let KernelArg::Buffer(o) = &args[2] else {
        unreachable!("a buffer")
    };
    o.bytes
        .chunks(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect()
}

fn same(got: &[f32], want: &[f32]) -> bool {
    got.len() == want.len()
        && got
            .iter()
            .zip(want)
            .all(|(g, w)| g.to_bits() == w.to_bits() || (g.is_nan() && w.is_nan()))
}

/// The reference: a complex multiply-accumulate with its own type, not the kernels' real expansion.
fn reference(a: &[Z], b: &[Z]) -> Vec<f32> {
    let mut out = Vec::new();
    for m in 0..M {
        for n in 0..N {
            let mut acc: Z = (0.0, 0.0);
            for k in 0..K {
                let (x, y) = (a[m * K + k], b[k * N + n]);
                acc = (
                    acc.0 + (x.0 * y.0 - x.1 * y.1),
                    acc.1 + (x.0 * y.1 + x.1 * y.0),
                );
            }
            out.extend([acc.0, acc.1]);
        }
    }
    out
}

fn corpus() -> (Vec<Z>, Vec<Z>) {
    let a = (0..M * K)
        .map(|i| (((i * 5) % 11) as f32 - 5.0, ((i * 3) % 7) as f32 - 3.0))
        .collect();
    let b = (0..K * N)
        .map(|i| (((i * 7) % 13) as f32 - 6.0, ((i * 2) % 9) as f32 - 4.0))
        .collect();
    (a, b)
}

fn corpus_nan() -> (Vec<Z>, Vec<Z>) {
    let (mut a, b) = corpus();
    a[K + 2].0 = f32::NAN;
    (a, b)
}

fn corpus_negative() -> (Vec<Z>, Vec<Z>) {
    let (a, b) = corpus();
    let neg = |v: Vec<Z>| {
        v.iter()
            .map(|z| (-z.0.abs() - 1.0, -z.1.abs() - 1.0))
            .collect()
    };
    (neg(a), neg(b))
}

#[test]
fn both_forms_match_the_complex_reference_on_every_corpus() {
    for form in [Form::Four, Form::Three] {
        for (name, (a, b)) in [
            ("ordinary", corpus()),
            ("nan", corpus_nan()),
            ("negative", corpus_negative()),
        ] {
            let got = run(&complex_gemm(form, Mutant::None), &a, &b);
            let want = reference(&a, &b);
            assert!(same(&got, &want), "{form:?} {name}: {got:?} vs {want:?}");
        }
    }
}

#[test]
fn the_corpus_is_not_blind() {
    // the NaN reaches exactly row 1 (both parts), and the imaginary parts take part (a real-only product would differ)
    let (a, b) = corpus_nan();
    let want = reference(&a, &b);
    let rows: Vec<bool> = (0..M)
        .map(|m| want[m * N * 2..(m + 1) * N * 2].iter().any(|x| x.is_nan()))
        .collect();
    assert_eq!(rows, vec![false, true, false]);
    let (a, b) = corpus();
    let real_only: Vec<f32> = reference(
        &a.iter().map(|z| (z.0, 0.0)).collect::<Vec<_>>(),
        &b.iter().map(|z| (z.0, 0.0)).collect::<Vec<_>>(),
    );
    assert!(!same(&real_only, &reference(&a, &b)));
}

#[test]
fn the_mutants_fail_by_value() {
    let (a, b) = corpus();
    let want = reference(&a, &b);
    for form in [Form::Four, Form::Three] {
        for mutant in [Mutant::ConjugateB, Mutant::ImagSign] {
            let got = run(&complex_gemm(form, mutant), &a, &b);
            assert!(got.iter().all(|x| x.is_finite()), "{form:?} {mutant:?}");
            assert!(!same(&got, &want), "{form:?} {mutant:?} must differ");
        }
    }
}
