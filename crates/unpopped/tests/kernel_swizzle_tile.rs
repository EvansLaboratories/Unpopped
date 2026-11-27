// SPDX-License-Identifier: MIT OR Apache-2.0
//! **A swizzled shared-memory tile at the kernel layer** (CUTLASS survey row 66). An 8 x 8 transpose through a shared tile whose
//! columns are XOR-swizzled by the row, run on the oracle against a plain transpose. The input is distinct small integers (a
//! wrong slot shows), with a NaN corpus and an all-negative corpus. A load that forgets the swizzle and a store with a narrowed
//! mask are mutants that stay in range and race-free, so they fail by value.

use unpopped::kernel::{KernelFn, check_kernel};
use unpopped::oracle::TypedBuffer;
use unpopped::oracle::kernel_eval::KernelArg;
use unpopped::oracle::kernel_launch::run_launch;

#[path = "support/swizzle_tile.rs"]
mod fixture;
use fixture::{F32, Mutant, T, swizzle_transpose};

fn buf(v: &[f32]) -> KernelArg {
    KernelArg::Buffer(TypedBuffer::new(
        F32,
        vec![v.len() as i64],
        vec![1],
        v.iter().flat_map(|x| x.to_le_bytes()).collect(),
    ))
}

fn run(mutant: Mutant, input: &[f32]) -> Vec<f32> {
    let k: KernelFn = swizzle_transpose(mutant);
    assert_eq!(check_kernel(&k), Ok(()));
    let mut args = vec![buf(input), buf(&[0.0; T * T])];
    run_launch(&k, &mut args).expect("the oracle runs it");
    let KernelArg::Buffer(o) = &args[1] else {
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

fn transpose(a: &[f32]) -> Vec<f32> {
    (0..T * T).map(|i| a[(i % T) * T + i / T]).collect()
}

fn corpus() -> Vec<f32> {
    (0..T * T).map(|i| ((i * 7) % 61) as f32 - 30.0).collect()
}

fn corpus_nan() -> Vec<f32> {
    let mut a = corpus();
    a[3 * T + 5] = f32::NAN;
    a
}

fn corpus_negative() -> Vec<f32> {
    corpus().iter().map(|x| -x.abs() - 1.0).collect()
}

#[test]
fn the_swizzled_transpose_matches_a_plain_transpose_on_every_corpus() {
    for (name, a) in [
        ("ordinary", corpus()),
        ("nan", corpus_nan()),
        ("negative", corpus_negative()),
    ] {
        let (got, want) = (run(Mutant::None, &a), transpose(&a));
        assert!(same(&got, &want), "{name}: {got:?} vs {want:?}");
    }
}

#[test]
fn the_corpus_is_not_blind() {
    // the NaN lands at (5, 3) of the output only, and the transpose moves the data (a no-op kernel would show)
    let want = transpose(&corpus_nan());
    let nans: Vec<usize> = (0..T * T).filter(|i| want[*i].is_nan()).collect();
    assert_eq!(nans, vec![5 * T + 3]);
    assert!(!same(&transpose(&corpus()), &corpus()));
}

#[test]
fn the_mutants_fail_by_value() {
    let a = corpus();
    let want = transpose(&a);
    for mutant in [Mutant::ReadUnswizzled, Mutant::NarrowMask] {
        let got = run(mutant, &a);
        assert!(got.iter().all(|x| x.is_finite()), "{mutant:?}");
        assert!(!same(&got, &want), "{mutant:?} must differ");
    }
}
