// SPDX-License-Identifier: MIT OR Apache-2.0
//! **Hillis-Steele prefix sum with per-group named barriers** (survey row 34). A block of 64 threads is two 32-lane groups, each
//! scanned in shared memory with its own named barrier (`bar.sync id, 32`). The result is compared with a serial inclusive prefix
//! sum per group over small integers (exact in `f32`, so the different association does not show), a NaN corpus (the NaN
//! reaches exactly the lanes at and after it, inside its own group) and an all-negative corpus. Dropping the last step and an
//! off-by-one guard fail by value; dropping the sync between a step's read and its write is a `DataRace` trap (last test).

use unpopped::kernel::{KernelFn, check_kernel};
use unpopped::oracle::TypedBuffer;
use unpopped::oracle::kernel_eval::{KernelArg, KernelTrap};
use unpopped::oracle::kernel_launch::run_launch;

#[path = "support/hillis_steele.rs"]
mod fixture;
use fixture::{F32, GROUP, Mutant, THREADS, hillis_steele};

fn buf(v: &[f32]) -> KernelArg {
    KernelArg::Buffer(TypedBuffer::new(
        F32,
        vec![v.len() as i64],
        vec![1],
        v.iter().flat_map(|x| x.to_le_bytes()).collect(),
    ))
}

fn run(mutant: Mutant, input: &[f32]) -> Result<Vec<f32>, KernelTrap> {
    let k: KernelFn = hillis_steele(mutant);
    assert_eq!(check_kernel(&k), Ok(()));
    let mut args = vec![buf(input), buf(&[0.0; THREADS])];
    run_launch(&k, &mut args)?;
    let KernelArg::Buffer(o) = &args[1] else {
        unreachable!("a buffer")
    };
    Ok(o.bytes
        .chunks(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect())
}

fn same(got: &[f32], want: &[f32]) -> bool {
    got.len() == want.len()
        && got
            .iter()
            .zip(want)
            .all(|(g, w)| g.to_bits() == w.to_bits() || (g.is_nan() && w.is_nan()))
}

/// A serial inclusive prefix sum within each group of `GROUP` lanes.
fn reference(x: &[f32]) -> Vec<f32> {
    x.chunks(GROUP)
        .flat_map(|g| {
            g.iter()
                .scan(0.0f32, |acc, v| {
                    *acc += v;
                    Some(*acc)
                })
                .collect::<Vec<_>>()
        })
        .collect()
}

fn corpus() -> Vec<f32> {
    (0..THREADS).map(|i| ((i * 7) % 13) as f32 - 6.0).collect()
}

fn corpus_nan() -> Vec<f32> {
    let mut x = corpus();
    x[5] = f32::NAN; // group 0
    x
}

fn corpus_negative() -> Vec<f32> {
    corpus().iter().map(|v| -v.abs() - 1.0).collect()
}

#[test]
fn the_group_scan_matches_a_serial_prefix_sum_on_every_corpus() {
    for (name, x) in [
        ("ordinary", corpus()),
        ("nan", corpus_nan()),
        ("negative", corpus_negative()),
    ] {
        let got = run(Mutant::None, &x).expect("the scan runs");
        assert!(same(&got, &reference(&x)), "{name}: {got:?}");
    }
}

#[test]
fn the_corpus_is_not_blind() {
    // the NaN reaches lanes 5..32 of group 0 and nothing of group 1; the groups scan independently
    let want = reference(&corpus_nan());
    assert!((0..5).all(|i| !want[i].is_nan()));
    assert!((5..GROUP).all(|i| want[i].is_nan()));
    assert!((GROUP..THREADS).all(|i| !want[i].is_nan()));
    assert_ne!(
        reference(&corpus())[GROUP],
        reference(&corpus())[GROUP - 1] + corpus()[GROUP]
    );
}

#[test]
fn the_value_mutants_fail_by_value() {
    let x = corpus();
    for mutant in [Mutant::DropLastStep, Mutant::OffByOneGuard] {
        let got = run(mutant, &x).expect("race-free, so it runs");
        assert!(got.iter().all(|v| v.is_finite()), "{mutant:?}");
        assert!(!same(&got, &reference(&x)), "{mutant:?} must differ");
    }
}

#[test]
fn the_missing_sync_between_a_steps_read_and_write_is_a_data_race() {
    // the neighbour's write and this lane's read are unordered, so the oracle traps instead of running on with wrong values
    // (it used to run to completion: a named-barrier waiter acquired a later phase's release; fixed in 0.60.31)
    assert_eq!(
        run(Mutant::NoSecondSync, &corpus()),
        Err(KernelTrap::DataRace)
    );
}
