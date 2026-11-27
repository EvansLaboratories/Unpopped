// SPDX-License-Identifier: MIT OR Apache-2.0
//! **Nine-term `bf16` emulation of an `f32` GEMM at the kernel layer** (CUTLASS survey row 27). The kernel splits each operand
//! into three `bf16` pieces and accumulates the nine products in `f32`. The reference is written independently: it rounds to
//! `bf16` by bit manipulation (round to nearest even on the top 16 bits), not through the IR's `Cast`, and is compared bit for
//! bit. A second check shows the emulation is close to the exact `f64` product, where a plain `bf16` product is not. Corpora:
//! non-integer values (the low pieces matter), a NaN, and all-negative operands.

use unpopped::kernel::{KernelFn, check_kernel};
use unpopped::oracle::TypedBuffer;
use unpopped::oracle::kernel_eval::KernelArg;
use unpopped::oracle::kernel_launch::run_launch;

#[path = "support/bf16x9.rs"]
mod fixture;
use fixture::{F32, K, M, Mutant, N, bf16x9_gemm};

fn buf(v: &[f32]) -> KernelArg {
    KernelArg::Buffer(TypedBuffer::new(
        F32,
        vec![v.len() as i64],
        vec![1],
        v.iter().flat_map(|x| x.to_le_bytes()).collect(),
    ))
}

fn run(mutant: Mutant, a: &[f32], b: &[f32]) -> Vec<f32> {
    let k: KernelFn = bf16x9_gemm(mutant);
    assert_eq!(check_kernel(&k), Ok(()));
    let mut args = vec![buf(a), buf(b), buf(&[0.0; M * N])];
    run_launch(&k, &mut args).expect("the oracle runs it");
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

/// Round to `bf16` (nearest even on the top 16 bits) and widen back, by bits.
fn bf16(x: f32) -> f32 {
    if x.is_nan() {
        return x;
    }
    let bits = x.to_bits();
    let rounded = bits.wrapping_add(0x7FFF + ((bits >> 16) & 1)) & 0xFFFF_0000;
    f32::from_bits(rounded)
}

fn pieces(x: f32) -> [f32; 3] {
    let x0 = bf16(x);
    let x1 = bf16(x - x0);
    [x0, x1, bf16(x - x0 - x1)]
}

fn reference(a: &[f32], b: &[f32]) -> Vec<f32> {
    let mut out = Vec::new();
    for m in 0..M {
        for n in 0..N {
            let mut acc = 0.0f32;
            for k in 0..K {
                let (pa, pb) = (pieces(a[m * K + k]), pieces(b[k * N + n]));
                for x in pa {
                    for y in pb {
                        acc += x * y;
                    }
                }
            }
            out.push(acc);
        }
    }
    out
}

fn exact(a: &[f32], b: &[f32]) -> Vec<f64> {
    let mut out = Vec::new();
    for m in 0..M {
        for n in 0..N {
            out.push(
                (0..K)
                    .map(|k| f64::from(a[m * K + k]) * f64::from(b[k * N + n]))
                    .sum(),
            );
        }
    }
    out
}

fn corpus() -> (Vec<f32>, Vec<f32>) {
    let a = (0..M * K)
        .map(|i| 0.37 * i as f32 - 1.713 + 0.0191 * (i * i) as f32)
        .collect();
    let b = (0..K * N)
        .map(|i| 1.1 - 0.293 * i as f32 + 0.0077 * (i * i) as f32)
        .collect();
    (a, b)
}

fn corpus_nan() -> (Vec<f32>, Vec<f32>) {
    let (mut a, b) = corpus();
    a[K + 1] = f32::NAN;
    (a, b)
}

fn corpus_negative() -> (Vec<f32>, Vec<f32>) {
    let (a, b) = corpus();
    (
        a.iter().map(|x| -x.abs() - 0.5).collect(),
        b.iter().map(|x| -x.abs() - 0.25).collect(),
    )
}

#[test]
fn the_nine_term_gemm_matches_the_bitwise_reference_on_every_corpus() {
    for (name, (a, b)) in [
        ("ordinary", corpus()),
        ("nan", corpus_nan()),
        ("negative", corpus_negative()),
    ] {
        let (got, want) = (run(Mutant::None, &a, &b), reference(&a, &b));
        assert!(same(&got, &want), "{name}: {got:?} vs {want:?}");
    }
}

#[test]
fn the_emulation_tracks_the_exact_product_and_plain_bf16_does_not() {
    let (a, b) = corpus();
    let got = run(Mutant::None, &a, &b);
    let hi = run(Mutant::HiOnly, &a, &b);
    let want = exact(&a, &b);
    let rel = |v: &[f32]| {
        v.iter()
            .zip(&want)
            .map(|(g, w)| (f64::from(*g) - w).abs() / w.abs().max(1.0))
            .fold(0.0, f64::max)
    };
    assert!(rel(&got) < 1e-6, "nine terms: {}", rel(&got));
    assert!(
        rel(&hi) > 1e-4,
        "a plain bf16 product is far off: {}",
        rel(&hi)
    );
}

#[test]
fn the_corpus_is_not_blind() {
    // the NaN reaches exactly row 1, and the low pieces are non-zero (a split that did nothing would show)
    let (a, b) = corpus_nan();
    let want = reference(&a, &b);
    let rows: Vec<bool> = (0..M)
        .map(|m| want[m * N..(m + 1) * N].iter().any(|x| x.is_nan()))
        .collect();
    assert_eq!(rows, vec![false, true]);
    let (a, _) = corpus();
    assert!(
        a.iter()
            .all(|x| pieces(*x)[1] != 0.0 && pieces(*x)[2] != 0.0 || *x == 0.0)
    );
}

#[test]
fn the_mutants_fail_by_value() {
    let (a, b) = corpus();
    let want = reference(&a, &b);
    for mutant in [
        Mutant::HiOnly,
        Mutant::UnchainedResidual,
        Mutant::ReversedPieces,
    ] {
        let got = run(mutant, &a, &b);
        assert!(got.iter().all(|x| x.is_finite()), "{mutant:?}");
        assert!(!same(&got, &want), "{mutant:?} must differ");
    }
}
