// SPDX-License-Identifier: MIT OR Apache-2.0
//! **A two-stage full/empty `mbarrier` ring in one block** (CUTLASS survey row 88: pipelines, producer/consumer, the multi-stage
//! ring the earlier rows lacked). Six items pass through two slots, so each barrier completes three phases and the wait parity
//! flips twice. The result is compared with `2 * x + 1` over ordinary, NaN and all-negative corpora. A consumer that reads the
//! wrong item counter fails by value; a consumer that reads the slot being refilled is a data race, and a consumer that never
//! releases a slot, or a producer that refills without waiting, deadlocks: those are typed traps, pinned as such. In-block roles, separate init/arrive/wait
//! statements, no transaction count (see `support/mbar_ring.rs` for the patent-chart line).

use unpopped::kernel::check_kernel;
use unpopped::oracle::TypedBuffer;
use unpopped::oracle::kernel_eval::{KernelArg, KernelTrap};
use unpopped::oracle::kernel_launch::run_launch;

#[path = "support/mbar_ring.rs"]
mod fixture;
use fixture::{F32, ITEMS, Mutant, ring_kernel};

fn buf(v: &[f32]) -> KernelArg {
    KernelArg::Buffer(TypedBuffer::new(
        F32,
        vec![v.len() as i64],
        vec![1],
        v.iter().flat_map(|x| x.to_le_bytes()).collect(),
    ))
}

fn run(mutant: Mutant, input: &[f32]) -> Result<Vec<f32>, KernelTrap> {
    let k = ring_kernel(mutant);
    assert_eq!(check_kernel(&k), Ok(()));
    let mut args = vec![buf(input), buf(&[0.0; ITEMS])];
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

fn reference(x: &[f32]) -> Vec<f32> {
    x.iter().map(|v| v * 2.0 + 1.0).collect()
}

fn corpus() -> Vec<f32> {
    vec![3.0, -7.5, 0.25, 11.0, -0.0, 6.5]
}

fn corpus_nan() -> Vec<f32> {
    let mut x = corpus();
    x[4] = f32::NAN;
    x
}

fn corpus_negative() -> Vec<f32> {
    corpus().iter().map(|v| -v.abs() - 1.0).collect()
}

#[test]
fn the_ring_delivers_every_item_in_order_on_every_corpus() {
    for (name, x) in [
        ("ordinary", corpus()),
        ("nan", corpus_nan()),
        ("negative", corpus_negative()),
    ] {
        let got = run(Mutant::None, &x).expect("the ring runs");
        assert!(same(&got, &reference(&x)), "{name}: {got:?}");
    }
}

#[test]
fn the_corpus_is_not_blind() {
    // distinct items (a slot read twice or skipped shows) and the NaN reaches exactly item 4
    let want = reference(&corpus_nan());
    assert_eq!(
        (0..ITEMS).filter(|i| want[*i].is_nan()).collect::<Vec<_>>(),
        vec![4]
    );
    let mut sorted = reference(&corpus());
    sorted.sort_by(f32::total_cmp);
    sorted.dedup();
    assert_eq!(sorted.len(), ITEMS);
}

#[test]
fn a_consumer_with_an_off_by_one_item_counter_fails_by_value() {
    let x = corpus();
    let got = run(Mutant::ShiftedItem, &x).expect("race-free, so it runs");
    assert!(got.iter().all(|v| v.is_finite()));
    assert!(!same(&got, &reference(&x)), "{got:?}");
}

#[test]
fn ring_protocol_violations_are_typed_traps() {
    let x = corpus();
    // reading a slot the producer is refilling is a data race; a slot never released deadlocks the producer's refill wait
    let observed = |m| run(m, &x).expect_err("a trap");
    assert!(matches!(
        observed(Mutant::ReadOtherSlot),
        KernelTrap::DataRace { .. }
    ));
    assert!(matches!(
        observed(Mutant::NoEmptyArrive),
        KernelTrap::MbarrierDeadlock
    ));
    // the producer that refills without waiting runs a phase ahead of the consumer, whose wait parity then never matches
    assert!(matches!(
        observed(Mutant::NoEmptyWait),
        KernelTrap::MbarrierDeadlock
    ));
}
