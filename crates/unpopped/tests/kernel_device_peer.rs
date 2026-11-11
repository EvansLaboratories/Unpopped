// SPDX-License-Identifier: MIT OR Apache-2.0
//! ROADMAP U12 slice 2 (`docs/options-u12-peer-multicast.md`): a launch spans several **devices**. Every device holds its
//! own copy of each buffer and runs the whole grid (`LaunchShape::devices`, `ExecId::DeviceRank`/`DeviceCount`);
//! `Place::Peer` reaches another device's copy; a `System`-scope barrier, atomic or fence orders accesses across devices,
//! and an access nothing orders is a data race. The expected arrays are worked by hand; the fixtures are
//! `tests/support/device_peer.rs`.

use unpopped::exec::{LaunchShape, Scope};
use unpopped::kernel::{KernelError, KernelFn, MemoryOrder, check_kernel};
use unpopped::oracle::TypedBuffer;
use unpopped::oracle::kernel_eval::{KernelArg, KernelTrap};
use unpopped::oracle::kernel_launch::{run_launch, run_launch_devices};

#[allow(dead_code)]
#[path = "support/device_peer.rs"]
mod fixture;
use fixture::{D, I32, all_gather, c, chain, kernel, load, peer};

fn buffer(vals: &[i32]) -> KernelArg {
    KernelArg::Buffer(TypedBuffer::new(
        I32,
        vec![vals.len() as i64],
        vec![1],
        vals.iter().flat_map(|v| v.to_le_bytes()).collect(),
    ))
}

fn ints(a: &KernelArg) -> Vec<i32> {
    let KernelArg::Buffer(b) = a else {
        unreachable!()
    };
    b.bytes
        .chunks(4)
        .map(|w| i32::from_le_bytes(w.try_into().unwrap()))
        .collect()
}

/// Run `k` on `devices` (one argument vector per device); every device's buffers afterwards, or the trap.
fn run(k: &KernelFn, mut devices: Vec<Vec<KernelArg>>) -> Result<Vec<Vec<Vec<i32>>>, KernelTrap> {
    assert_eq!(check_kernel(k), Ok(()), "well formed");
    run_launch_devices(k, &mut devices)?;
    Ok(devices
        .iter()
        .map(|d| d.iter().map(ints).collect())
        .collect())
}

/// The arguments of every device: `inp` is `base + 10 * (device + 1) + i`, the other buffers are `len` zeros each.
fn devices(base: i32, inp_len: usize, others: &[usize]) -> Vec<Vec<KernelArg>> {
    (0..D as i32)
        .map(|d| {
            let inp: Vec<i32> = (0..inp_len as i32)
                .map(|i| base + 10 * (d + 1) + i)
                .collect();
            let mut args = vec![buffer(&inp)];
            args.extend(others.iter().map(|n| buffer(&vec![0; *n])));
            args
        })
        .collect()
}

#[test]
fn every_device_gets_its_own_rank_and_the_launch_count() {
    let out = run(&fixture::rank_and_count(), devices(0, 1, &[1])).unwrap();
    assert_eq!(
        out.iter().map(|d| d[1].clone()).collect::<Vec<_>>(),
        [[3], [13], [23]]
    );
}

#[test]
fn an_all_gather_is_ordered_by_a_system_barrier() {
    // slot `d * 4 + i` of every device's `out` is device d's `inp[i] = 10 * (d + 1) + i`, and `sum[i]` adds the three
    let out = run(&all_gather(Some(Scope::System)), devices(0, 4, &[12, 4])).unwrap();
    for d in out {
        assert_eq!(d[1], [10, 11, 12, 13, 20, 21, 22, 23, 30, 31, 32, 33]);
        assert_eq!(d[2], [60, 63, 66, 69]);
    }
}

#[test]
fn an_all_gather_races_without_a_system_barrier() {
    // no barrier, and a barrier of one device only: a peer's write and the owner's read are unordered
    // a `Cluster` barrier orders the blocks of one device's cluster only, the clusters of other devices are not in it
    for sync in [None, Some(Scope::Device), Some(Scope::Cluster)] {
        assert_eq!(
            run(&all_gather(sync), devices(0, 4, &[12, 4])).unwrap_err(),
            KernelTrap::DataRace,
            "{sync:?}"
        );
    }
}

#[test]
fn a_system_release_and_acquire_chain_a_reduction_across_devices() {
    // device r receives the partial sum of the devices before it: 0, 10, then 10 + 20 + 30 on the last
    let sys = Scope::System;
    let out = run(
        &chain((MemoryOrder::Release, sys), (MemoryOrder::Acquire, sys)),
        devices(0, 1, &[1, 1]),
    )
    .unwrap();
    assert_eq!(
        out.iter().map(|d| d[1].clone()).collect::<Vec<_>>(),
        [[0], [10], [60]]
    );
}

#[test]
fn a_release_or_acquire_that_is_relaxed_or_below_system_scope_orders_nothing() {
    let sys = Scope::System;
    let (rel, acq) = (MemoryOrder::Release, MemoryOrder::Acquire);
    for (release, acquire) in [
        ((MemoryOrder::Relaxed, sys), (acq, sys)),
        ((rel, sys), (MemoryOrder::Relaxed, sys)),
        // a device-scope atomic does not reach another device
        ((rel, Scope::Device), (acq, sys)),
        ((rel, sys), (acq, Scope::Device)),
    ] {
        assert_eq!(
            run(&chain(release, acquire), devices(0, 1, &[1, 1])).unwrap_err(),
            KernelTrap::DataRace,
            "{release:?} / {acquire:?}"
        );
    }
}

#[test]
fn a_device_barrier_releases_each_device_on_its_own() {
    let mut devs = devices(0, 1, &[2, 2]);
    devs.truncate(2);
    let out = run(&fixture::device_barrier(), devs).unwrap();
    assert_eq!(out[0][1], [1, 2]);
    assert_eq!(out[0][2], [2, 1]);
    assert_eq!(out[1][1], [101, 102]);
    assert_eq!(out[1][2], [102, 101]);
}

#[test]
fn a_peer_read_names_the_device_by_rank_and_an_own_rank_is_the_own_copy() {
    // inp[0] of device d is 100 + d (the base 90 plus 10 * (d + 1))
    let two = |d: Vec<Vec<KernelArg>>| d.into_iter().take(2).collect::<Vec<_>>();
    let args = || two(devices(90, 1, &[1]));
    for (which, want) in [(0, [100, 100]), (1, [110, 110])] {
        let out = run(&fixture::read_device(c(which)), args()).unwrap();
        assert_eq!(out[0][1], [want[0]], "device 0 reads {which}");
        assert_eq!(out[1][1], [want[1]], "device 1 reads {which}");
    }
}

#[test]
fn a_device_rank_outside_the_launch_is_a_trap() {
    for rank in [2, -1] {
        let d = devices(0, 1, &[1]).into_iter().take(2).collect();
        assert_eq!(
            run(&fixture::read_device(c(rank)), d).unwrap_err(),
            KernelTrap::DeviceRankOutOfRange {
                rank: i128::from(rank)
            }
        );
    }
}

#[test]
fn a_multi_device_launch_needs_one_argument_vector_per_device() {
    let k = fixture::rank_and_count();
    let mut one = devices(0, 1, &[1]).into_iter().take(1).collect::<Vec<_>>();
    assert!(matches!(
        run_launch_devices(&k, &mut one),
        Err(KernelTrap::ArgMismatch(_))
    ));
    // the single-device entry point refuses a launch of several
    let mut args = devices(0, 1, &[1]).remove(0);
    assert!(matches!(
        run_launch(&k, &mut args),
        Err(KernelTrap::ArgMismatch(_))
    ));
}

#[test]
fn the_checker_refuses_a_peer_place_and_a_system_scope_in_a_launch_of_one_device() {
    let one = LaunchShape::new([1, 1, 1], [1, 1, 1]);
    let k = kernel(
        "one_device",
        one,
        &["inp", "out"],
        (1, vec![load(0, peer(0, c(0), c(0)))]),
    );
    assert!(matches!(
        check_kernel(&k),
        Err(KernelError::PeerUnsupported { .. })
    ));
    let k = kernel(
        "system_barrier",
        one,
        &["inp", "out"],
        (0, vec![fixture::barrier(Scope::System)]),
    );
    assert_eq!(check_kernel(&k), Err(KernelError::BadBarrierScope));
}

#[test]
fn a_value_read_from_a_peer_device_and_the_device_rank_are_divergent_but_the_count_is_not() {
    let guarded = |cond: unpopped::ir::ScalarExpr, pre: Vec<unpopped::kernel::Stmt>| {
        let mut body = pre;
        body.push(unpopped::kernel::Stmt::If {
            cond,
            then: unpopped::kernel::Block(vec![fixture::barrier(Scope::Device)]),
            els: unpopped::kernel::Block(vec![]),
        });
        kernel(
            "guarded",
            LaunchShape::new([1, 1, 1], [1, 1, 1]).with_devices(2),
            &["inp", "out"],
            (1, body),
        )
    };
    // a Device barrier under a condition on a value another device holds, or on the rank, is divergent
    let k = guarded(fixture::local(0), vec![load(0, peer(0, c(1), c(0)))]);
    assert_eq!(check_kernel(&k), Err(KernelError::BarrierDivergent));
    let k = guarded(fixture::rank(), vec![load(0, fixture::at(0, c(0)))]);
    assert_eq!(check_kernel(&k), Err(KernelError::BarrierDivergent));
    // a System barrier spans blocks, so a condition on the block index is divergent for it
    let k = kernel(
        "block_guard",
        LaunchShape::new([1, 1, 1], [2, 1, 1]).with_devices(2),
        &["inp", "out"],
        (
            0,
            vec![unpopped::kernel::Stmt::If {
                cond: fixture::block(),
                then: unpopped::kernel::Block(vec![fixture::barrier(Scope::System)]),
                els: unpopped::kernel::Block(vec![]),
            }],
        ),
    );
    assert_eq!(check_kernel(&k), Err(KernelError::BarrierDivergent));
    // the count is the same for every thread
    let k = guarded(fixture::count(), vec![load(0, fixture::at(0, c(0)))]);
    assert_eq!(check_kernel(&k), Ok(()));
}

#[test]
fn the_checker_refuses_a_device_rank_that_is_not_an_integer_and_zero_devices() {
    let float = unpopped::ir::ScalarExpr::Cast(
        Box::new(fixture::rank()),
        unpopped::ir::CastSpec {
            to: unpopped_vocab::ElementKind::F32,
            round: unpopped::ir::RoundMode::NearestEven,
            sat: unpopped::ir::Saturation::Wrap,
        },
    );
    let k = kernel(
        "float_rank",
        fixture::gather_launch(),
        &["inp", "out"],
        (1, vec![load(0, peer(0, float, c(0)))]),
    );
    assert!(matches!(
        check_kernel(&k),
        Err(KernelError::BadPlace { buffer: 0 })
    ));
    let k = kernel(
        "no_devices",
        LaunchShape::new([1, 1, 1], [1, 1, 1]).with_devices(0),
        &["inp", "out"],
        (0, vec![]),
    );
    assert_eq!(check_kernel(&k), Err(KernelError::BadLaunch));
}
