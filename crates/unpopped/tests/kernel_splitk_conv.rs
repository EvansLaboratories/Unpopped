// SPDX-License-Identifier: MIT OR Apache-2.0
//! **Split-K convolution, serial-reduction form, at the kernel layer** (CUTLASS survey row 50). The channel reduction of a 1-D
//! convolution is cut into four slices: launch 1 writes one partial per slice to a workspace, launch 2 sums the partials in slice
//! order. The result is compared with a plain convolution over distinct small integers (exact in `f32`), with a NaN corpus and an
//! all-negative corpus. A slice that reads the wrong channels and a reduction that drops a slice are mutants that stay in range,
//! so they fail by value. The stream-K division of one tile between blocks is held and not modelled.

use unpopped::kernel::{KernelFn, check_kernel};
use unpopped::oracle::TypedBuffer;
use unpopped::oracle::kernel_eval::KernelArg;
use unpopped::oracle::kernel_launch::run_launch;

#[path = "support/splitk_conv.rs"]
mod fixture;
use fixture::{C, F32, Mutant, O, R, S, W, WO, partial_conv, reduce_slices};

fn buf(v: &[f32]) -> KernelArg {
    KernelArg::Buffer(TypedBuffer::new(
        F32,
        vec![v.len() as i64],
        vec![1],
        v.iter().flat_map(|x| x.to_le_bytes()).collect(),
    ))
}

fn floats(a: &KernelArg) -> Vec<f32> {
    let KernelArg::Buffer(b) = a else {
        unreachable!("a buffer")
    };
    b.bytes
        .chunks(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect()
}

/// Both launches in order, the workspace carried from the first to the second.
fn run(mutant: Mutant, x: &[f32], w: &[f32]) -> Vec<f32> {
    let (first, second) = (partial_conv(mutant), reduce_slices(mutant));
    assert_eq!(check_kernel(&first), Ok(()));
    assert_eq!(check_kernel(&second), Ok(()));
    let mut args = vec![buf(x), buf(w), buf(&[0.0; S * O * WO])];
    run_launch(&first, &mut args).expect("the oracle runs the partials");
    let mut args2 = vec![args.remove(2), buf(&[0.0; O * WO])];
    run_launch(&second, &mut args2).expect("the oracle runs the reduction");
    floats(&args2[1])
}

fn same(got: &[f32], want: &[f32]) -> bool {
    got.len() == want.len()
        && got
            .iter()
            .zip(want)
            .all(|(g, w)| g.to_bits() == w.to_bits() || (g.is_nan() && w.is_nan()))
}

/// The plain convolution over the whole channel axis, written without slices.
fn reference(x: &[f32], w: &[f32]) -> Vec<f32> {
    let mut out = Vec::new();
    for o in 0..O {
        for p in 0..WO {
            let mut acc = 0.0f32;
            for c in 0..C {
                for r in 0..R {
                    acc += x[c * W + p + r] * w[(o * C + c) * R + r];
                }
            }
            out.push(acc);
        }
    }
    out
}

fn corpus() -> (Vec<f32>, Vec<f32>) {
    let x = (0..C * W).map(|i| ((i * 5) % 17) as f32 - 8.0).collect();
    let w = (0..O * C * R).map(|i| ((i * 3) % 7) as f32 - 3.0).collect();
    (x, w)
}

fn corpus_nan() -> (Vec<f32>, Vec<f32>) {
    let (mut x, w) = corpus();
    x[5 * W + 2] = f32::NAN; // channel 5 is in slice 2
    (x, w)
}

fn corpus_negative() -> (Vec<f32>, Vec<f32>) {
    let (x, w) = corpus();
    (
        x.iter().map(|v| -v.abs() - 1.0).collect(),
        w.iter().map(|v| v.abs() + 1.0).collect(),
    )
}

#[test]
fn the_split_k_conv_matches_the_plain_conv_on_every_corpus() {
    for (name, (x, w)) in [
        ("ordinary", corpus()),
        ("nan", corpus_nan()),
        ("negative", corpus_negative()),
    ] {
        let (got, want) = (run(Mutant::None, &x, &w), reference(&x, &w));
        assert!(same(&got, &want), "{name}: {got:?} vs {want:?}");
    }
}

#[test]
fn the_corpus_is_not_blind() {
    // the NaN at (channel 5, position 2) reaches exactly the outputs whose window covers position 2: p in {0, 1, 2}
    let (x, w) = corpus_nan();
    let want = reference(&x, &w);
    let nan_positions: Vec<usize> = (0..WO).filter(|p| want[*p].is_nan()).collect();
    assert_eq!(nan_positions, vec![0, 1, 2]);
    assert!(!want[WO - 1].is_nan());
}

#[test]
fn the_mutants_fail_by_value() {
    let (x, w) = corpus();
    let want = reference(&x, &w);
    for mutant in [Mutant::SliceZero, Mutant::DropLastSlice] {
        let got = run(mutant, &x, &w);
        assert!(got.iter().all(|v| v.is_finite()), "{mutant:?}");
        assert!(!same(&got, &want), "{mutant:?} must differ");
    }
}
