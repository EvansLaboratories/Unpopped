// SPDX-License-Identifier: MIT OR Apache-2.0
//! **Gather/scatter GEMM at the kernel layer** (CUTLASS survey row 13). `D[idx_d[m]][n] = sum_k A[idx_a[m]][k] B[k][n]`, with a
//! repeated gather row, a skipped (negative) scatter index and three rows of `D` that no problem row reaches (filled with a
//! poison value, so a stray write shows). Compared with a host reference written as a scatter into a poisoned `D`. Three corpora:
//! ordinary, an all-negative `A`, and a NaN in the row `A` that is gathered twice (it must reach exactly the two problem rows
//! that gather it, and only the ones that are scattered). Mutants (gather index used for the scatter, scatter ignored, a
//! negative index clamped) fail by value.

use unpopped::kernel::check_kernel;
use unpopped::oracle::TypedBuffer;
use unpopped::oracle::kernel_eval::KernelArg;
use unpopped::oracle::kernel_launch::run_launch;
use unpopped_vocab::ElementKind;

#[path = "support/gather_gemm.rs"]
mod fixture;
use fixture::{F32, I32, IDX_A, IDX_D, K, M, Mutant, N, POISON, RA, RD, gather_scatter_gemm};

fn buf(kind: ElementKind, shape: &[i64], bytes: Vec<u8>) -> KernelArg {
    let strides: Vec<i64> = (0..shape.len())
        .map(|d| shape[d + 1..].iter().product())
        .collect();
    KernelArg::Buffer(TypedBuffer::new(kind, shape.to_vec(), strides, bytes))
}

fn f32s(shape: &[i64], v: &[f32]) -> KernelArg {
    buf(F32, shape, v.iter().flat_map(|x| x.to_le_bytes()).collect())
}

fn i32s(v: &[i32]) -> KernelArg {
    buf(
        I32,
        &[v.len() as i64],
        v.iter().flat_map(|x| x.to_le_bytes()).collect(),
    )
}

struct Corpus {
    a: Vec<f32>,
    b: Vec<f32>,
}

fn ordinary() -> Corpus {
    Corpus {
        a: (0..RA * K)
            .map(|i| ((i * 5 + 2) % 9) as f32 - 4.0)
            .collect(),
        b: (0..K * N).map(|i| ((i * 3 + 1) % 7) as f32 - 3.0).collect(),
    }
}

fn negative() -> Corpus {
    let mut c = ordinary();
    c.a.iter_mut().for_each(|x| *x = -x.abs() - 1.0);
    c.b.iter_mut().for_each(|x| *x = x.abs() + 1.0);
    c
}

/// A NaN in row 5 of `A`, which problem rows 2 and 3 gather; row 2 is not scattered (its index is -1).
fn nan() -> Corpus {
    let mut c = ordinary();
    c.a[5 * K + 1] = f32::NAN;
    c
}

fn reference(c: &Corpus) -> Vec<f32> {
    let mut d = vec![POISON; RD * N];
    for m in 0..M {
        if IDX_D[m] < 0 {
            continue;
        }
        for n in 0..N {
            let mut acc = 0.0f32;
            for k in 0..K {
                acc += c.a[IDX_A[m] as usize * K + k] * c.b[k * N + n];
            }
            d[IDX_D[m] as usize * N + n] = acc;
        }
    }
    d
}

fn try_run(
    c: &Corpus,
    mutant: Mutant,
) -> Result<Vec<f32>, unpopped::oracle::kernel_eval::KernelTrap> {
    let k = gather_scatter_gemm(mutant);
    assert_eq!(check_kernel(&k), Ok(()));
    let mut args = [
        f32s(&[RA as i64, K as i64], &c.a),
        f32s(&[K as i64, N as i64], &c.b),
        i32s(&IDX_A),
        i32s(&IDX_D),
        f32s(&[RD as i64, N as i64], &[POISON; RD * N]),
    ];
    run_launch(&k, &mut args)?;
    let KernelArg::Buffer(b) = &args[4] else {
        unreachable!("a buffer")
    };
    Ok(b.bytes
        .chunks(4)
        .map(|v| f32::from_le_bytes(v.try_into().unwrap()))
        .collect())
}

fn run(c: &Corpus, mutant: Mutant) -> Vec<f32> {
    try_run(c, mutant).expect("the oracle runs it")
}

fn same(got: &[f32], want: &[f32]) -> bool {
    got.len() == want.len()
        && got
            .iter()
            .zip(want)
            .all(|(g, w)| g.to_bits() == w.to_bits() || (g.is_nan() && w.is_nan()))
}

#[test]
fn the_gather_scatter_gemm_matches_the_scatter_reference_on_every_corpus() {
    for (name, c) in [
        ("ordinary", ordinary()),
        ("negative", negative()),
        ("nan", nan()),
    ] {
        let (got, want) = (run(&c, Mutant::None), reference(&c));
        assert!(same(&got, &want), "{name}: {got:?} vs {want:?}");
    }
}

#[test]
fn the_index_pattern_is_not_blind() {
    let d = reference(&ordinary());
    // rows 0, 3 and 5 of D are never written; row 3 of A is gathered by problem row 0 and row 5 twice
    for row in [0usize, 3, 5] {
        assert!(
            d[row * N..(row + 1) * N].iter().all(|x| *x == POISON),
            "D row {row} untouched"
        );
    }
    assert_eq!(IDX_A[2], IDX_A[3], "a repeated gather row");
    // the NaN reaches only D row 4 (problem row 3); problem row 2 gathers it too but is not scattered
    let d = reference(&nan());
    let nan_rows: Vec<usize> = (0..RD)
        .filter(|r| d[r * N..(r + 1) * N].iter().any(|x| x.is_nan()))
        .collect();
    assert_eq!(nan_rows, vec![4]);
}

#[test]
fn the_gather_and_scatter_mutants_fail_by_value() {
    let c = ordinary();
    let want = reference(&c);
    for m in [
        Mutant::MirroredScatter,
        Mutant::NoScatter,
        Mutant::NegativeNotSkipped,
    ] {
        assert!(!same(&run(&c, m), &want), "{m:?}");
    }
}

#[test]
fn applying_the_gather_index_to_the_scatter_is_a_data_race_the_oracle_traps() {
    // problem rows 2 and 3 gather the same row, so they would both write D row 5: caught as a trap, not a wrong number
    assert_eq!(
        try_run(&ordinary(), Mutant::GatherIndexForScatter),
        Err(unpopped::oracle::kernel_eval::KernelTrap::DataRace)
    );
}
