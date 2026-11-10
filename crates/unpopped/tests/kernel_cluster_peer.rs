// SPDX-License-Identifier: MIT OR Apache-2.0
//! ROADMAP U12 slice 1 (`docs/options-u12-peer-multicast.md`): **cluster-peer shared memory**. `Place::ClusterPeer`
//! names the `Shared` array of another block of the launch's cluster by its rank in the cluster. A peer access is a
//! `Shared` access like any other: ordered by a `Cluster` barrier or by a cluster-scope release/acquire pair, a data race
//! otherwise. The expected arrays are worked by hand from `inp[i] = i + 1` over 4 blocks of 4 threads in clusters of 2
//! blocks (`{0, 1}` and `{2, 3}`); the fixtures are `tests/support/cluster_peer.rs`.

use unpopped::exec::{LaunchShape, MemorySpace};
use unpopped::kernel::{KernelError, MemoryOrder, Stmt, check_kernel};
use unpopped::oracle::TypedBuffer;
use unpopped::oracle::kernel_eval::{KernelArg, KernelTrap};
use unpopped::oracle::kernel_launch::run_launch;

#[allow(dead_code)]
#[path = "support/cluster_peer.rs"]
mod fixture;
use fixture::{I32, N, block, c, kernel, launch, load, local, peer, rank, sh, thread};

fn buffer(vals: &[i32]) -> KernelArg {
    KernelArg::Buffer(TypedBuffer::new(
        I32,
        vec![vals.len() as i64],
        vec![1],
        vals.iter().flat_map(|v| v.to_le_bytes()).collect(),
    ))
}

/// Run `k` over `inp[i] = i + 1` and a zeroed `out`; the `out` array, or the trap.
fn run(k: &unpopped::kernel::KernelFn) -> Result<Vec<i32>, KernelTrap> {
    assert_eq!(check_kernel(k), Ok(()), "well formed");
    let inp: Vec<i32> = (1..=N as i32).collect();
    let mut args = vec![buffer(&inp), buffer(&[0; N])];
    run_launch(k, &mut args)?;
    let KernelArg::Buffer(out) = &args[1] else {
        unreachable!()
    };
    Ok(out
        .bytes
        .chunks(4)
        .map(|w| i32::from_le_bytes(w.try_into().unwrap()))
        .collect())
}

#[test]
fn a_block_reads_its_peers_shared_array_after_a_cluster_barrier() {
    // each thread's own value plus the value of the same thread in the other block of its cluster
    assert_eq!(
        run(&fixture::exchange()).unwrap(),
        [6, 8, 10, 12, 6, 8, 10, 12, 22, 24, 26, 28, 22, 24, 26, 28]
    );
}

#[test]
fn the_rank_picks_the_block_and_the_own_rank_names_the_own_array() {
    // the peer's values alone
    assert_eq!(
        run(&fixture::read_rank(fixture::other())).unwrap(),
        [5, 6, 7, 8, 1, 2, 3, 4, 13, 14, 15, 16, 9, 10, 11, 12]
    );
    // the rank of the running block is its own array
    assert_eq!(
        run(&fixture::read_rank(rank())).unwrap(),
        (1..=N as i32).collect::<Vec<_>>()
    );
    // a fixed rank 0: every block reads the first block of its cluster
    assert_eq!(
        run(&fixture::read_rank(c(0))).unwrap(),
        [1, 2, 3, 4, 1, 2, 3, 4, 9, 10, 11, 12, 9, 10, 11, 12]
    );
}

#[test]
fn a_rank_outside_the_cluster_is_a_trap() {
    assert_eq!(
        run(&fixture::read_rank(c(2))).unwrap_err(),
        KernelTrap::ClusterRankOutOfRange { rank: 2 }
    );
    assert_eq!(
        run(&fixture::read_rank(c(-1))).unwrap_err(),
        KernelTrap::ClusterRankOutOfRange { rank: -1 }
    );
}

#[test]
fn a_peer_write_is_seen_by_the_owner_after_a_barrier_and_races_without_one() {
    // each thread's input lands in the peer block's array: the owner reads the peer's inputs
    assert_eq!(
        run(&fixture::push(true)).unwrap(),
        [5, 6, 7, 8, 1, 2, 3, 4, 13, 14, 15, 16, 9, 10, 11, 12]
    );
    // without the second barrier the peer's write and the owner's read are unordered
    assert_eq!(
        run(&fixture::push(false)).unwrap_err(),
        KernelTrap::DataRace
    );
}

#[test]
fn a_cluster_scope_atomic_on_a_peer_element_counts_every_thread() {
    // block rank r adds r + 1 four times (one per thread) to its peer's counter: rank 0 receives 4 * 2, rank 1 receives 4 * 1
    let mut want = [0; N];
    for (b, v) in [8, 4, 8, 4].into_iter().enumerate() {
        want[b * 4] = v;
    }
    assert_eq!(run(&fixture::count(MemoryOrder::Relaxed)).unwrap(), want);
}

#[test]
fn a_cluster_scope_release_and_acquire_orders_a_peer_write_without_a_barrier() {
    // rank 0 writes into rank 1's array and releases a flag; rank 1 acquires it and reads its own array
    let mut want = [0; N];
    want[4..8].copy_from_slice(&[1, 2, 3, 4]);
    want[12..16].copy_from_slice(&[9, 10, 11, 12]);
    assert_eq!(
        run(&fixture::handoff(
            MemoryOrder::Release,
            MemoryOrder::Acquire
        ))
        .unwrap(),
        want
    );
    // either side relaxed: the write and the read are unordered
    for (rel, acq) in [
        (MemoryOrder::Relaxed, MemoryOrder::Acquire),
        (MemoryOrder::Release, MemoryOrder::Relaxed),
    ] {
        assert_eq!(
            run(&fixture::handoff(rel, acq)).unwrap_err(),
            KernelTrap::DataRace,
            "{rel:?} / {acq:?}"
        );
    }
}

#[test]
fn the_checker_refuses_a_peer_place_without_cluster_dimensions_or_outside_shared() {
    // no cluster dimensions
    let k = kernel(
        "no_cluster",
        LaunchShape::new([4, 1, 1], [4, 1, 1]),
        2,
        vec![load(0, peer(c(0), thread()))],
    );
    assert!(matches!(
        check_kernel(&k),
        Err(KernelError::ClusterPeerUnsupported { .. })
    ));
    // a Private allocation
    let mut k = fixture::exchange();
    k.allocs[0].space = MemorySpace::Private;
    assert!(matches!(
        check_kernel(&k),
        Err(KernelError::ClusterPeerUnsupported { .. })
    ));
}

#[test]
fn the_checker_refuses_a_peer_rank_that_is_not_an_integer_and_an_unknown_allocation() {
    // a rank that is not an integer
    let float = unpopped::ir::ScalarExpr::Cast(
        Box::new(rank()),
        unpopped::ir::CastSpec {
            to: unpopped_vocab::ElementKind::F32,
            round: unpopped::ir::RoundMode::NearestEven,
            sat: unpopped::ir::Saturation::Wrap,
        },
    );
    let k = kernel(
        "float_rank",
        launch(),
        2,
        vec![load(0, peer(float, thread()))],
    );
    assert!(matches!(
        check_kernel(&k),
        Err(KernelError::BadPlace { buffer: 0 })
    ));
    // an unknown allocation
    let k = kernel(
        "no_alloc",
        launch(),
        2,
        vec![load(
            0,
            unpopped::kernel::Place::ClusterPeer {
                alloc: 3,
                block: c(0),
                index: c(0),
            },
        )],
    );
    assert!(matches!(
        check_kernel(&k),
        Err(KernelError::BadPlace { buffer: 3 })
    ));
}

#[test]
fn a_value_read_from_a_peer_is_divergent() {
    // a Block barrier under a condition that reads a peer's array is under divergent control
    let k = kernel(
        "divergent",
        launch(),
        2,
        vec![
            load(1, peer(c(0), c(0))),
            Stmt::If {
                cond: local(1),
                then: unpopped::kernel::Block(vec![Stmt::Barrier {
                    scope: unpopped::exec::Scope::Block,
                    kind: unpopped::kernel::BarrierKind::Full,
                }]),
                els: unpopped::kernel::Block(vec![]),
            },
        ],
    );
    assert_eq!(check_kernel(&k), Err(KernelError::BarrierDivergent));
}

#[test]
fn an_own_array_read_by_rank_is_not_a_peer_access_of_another_block() {
    // block() and sh() are used so the fixture's helpers stay exercised; the own-rank read equals the plain read
    let plain = kernel(
        "plain",
        launch(),
        2,
        vec![
            load(0, fixture::at(0, fixture::gid())),
            fixture::store(sh(thread()), local(0)),
            fixture::cluster_barrier(),
            load(1, sh(thread())),
            fixture::store(fixture::at(1, fixture::gid()), local(1)),
        ],
    );
    assert_eq!(
        run(&plain).unwrap(),
        run(&fixture::read_rank(rank())).unwrap()
    );
    let _ = block();
}

#[test]
fn a_release_and_acquire_on_a_peers_own_flag_orders_a_peer_write() {
    // the flag is an element of the peer's Shared array: rank 0 releases on it through the peer place, rank 1 acquires on
    // it through its own place; the two spell one element, so the edge reaches the read
    let mut want = [0; N];
    want[4] = 1;
    want[12] = 9;
    assert_eq!(
        run(&fixture::peer_flag(
            MemoryOrder::Release,
            MemoryOrder::Acquire
        ))
        .unwrap(),
        want
    );
    for (rel, acq) in [
        (MemoryOrder::Relaxed, MemoryOrder::Acquire),
        (MemoryOrder::Release, MemoryOrder::Relaxed),
    ] {
        assert_eq!(
            run(&fixture::peer_flag(rel, acq)).unwrap_err(),
            KernelTrap::DataRace,
            "{rel:?} / {acq:?}"
        );
    }
}

#[test]
fn ranks_count_x_fastest_across_a_two_dimensional_cluster() {
    // 8 blocks of one thread in a 4 by 2 grid, clusters of 2 by 2 blocks: block `x + 4y`; the block of rank `r` in a
    // cluster is at x = 2 * (x / 2) + r % 2, y = r / 2, so it holds the id 2 * (x / 2) + r % 2 + 4 * (r / 2)
    for (rank, per_cluster) in [(0, [0, 2]), (1, [1, 3]), (2, [4, 6]), (3, [5, 7])] {
        let k = fixture::rank_probe(rank);
        assert_eq!(check_kernel(&k), Ok(()), "well formed");
        let mut args = vec![buffer(&[0; N]), buffer(&[0; N])];
        run_launch(&k, &mut args).unwrap();
        let KernelArg::Buffer(out) = &args[1] else {
            unreachable!()
        };
        let got: Vec<i32> = out
            .bytes
            .chunks(4)
            .take(8)
            .map(|w| i32::from_le_bytes(w.try_into().unwrap()))
            .collect();
        let want: Vec<i32> = (0..8).map(|b| per_cluster[(b % 4) / 2]).collect();
        assert_eq!(got, want, "rank {rank}");
    }
}
