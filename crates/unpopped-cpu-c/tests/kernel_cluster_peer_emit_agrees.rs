// SPDX-License-Identifier: MIT OR Apache-2.0
//! **The serial C of a cluster-peer access agrees with the launch interpreter** (ROADMAP U12 slice 1, the cpu-c
//! reference): `Place::ClusterPeer` is spelled as an element of the grid-wide `Shared` array of the block with the given
//! rank in this block's cluster (`sh<i>_all[peer * len + index]`), a kernel that reaches a peer keeps every block
//! resident, and the blocks run in ascending order as the interpreter runs them. The semantics are
//! `crates/unpopped/tests/kernel_cluster_peer.rs`; this file compiles the C with a host compiler, runs it, and compares
//! the written bytes with `run_launch`. The seeded mutants of the emitted C each have to change them.

use unpopped::kernel::{KernelFn, MemoryOrder, Module};
use unpopped::oracle::TypedBuffer;
use unpopped::oracle::kernel_eval::KernelArg;
use unpopped::oracle::kernel_launch::run_launch;
use unpopped_cpu_c::lower_module;

#[path = "support/cc.rs"]
mod cc;
// The shared compile-and-run drivers; this file uses a subset of them.
#[allow(dead_code)]
#[path = "support/kernel_run.rs"]
mod kernel_run;
#[allow(dead_code)]
#[path = "support/launch_run.rs"]
mod launch_run;
// The fixture shared with the oracle's test; each file uses part of it.
#[allow(dead_code)]
#[path = "support/cluster_peer.rs"]
mod fixture;
use fixture::{I32, N, c, rank};
use kernel_run::run_generated;
use launch_run::agree_launch;

fn buffer(vals: &[i32]) -> KernelArg {
    KernelArg::Buffer(TypedBuffer::new(
        I32,
        vec![vals.len() as i64],
        vec![1],
        vals.iter().flat_map(|v| v.to_le_bytes()).collect(),
    ))
}

/// `inp[i] = i + 1` and a zeroed `out`.
fn args() -> Vec<KernelArg> {
    let inp: Vec<i32> = (1..=N as i32).collect();
    vec![buffer(&inp), buffer(&[0; N])]
}

/// Every kernel of the fixture, by name.
fn kernels() -> Vec<(&'static str, KernelFn)> {
    vec![
        ("peer_exchange", fixture::exchange()),
        ("peer_other", fixture::read_rank(fixture::other())),
        ("peer_own", fixture::read_rank(rank())),
        ("peer_zero", fixture::read_rank(c(0))),
        ("peer_push", fixture::push(true)),
        ("peer_count", fixture::count(MemoryOrder::Relaxed)),
        (
            "peer_handoff",
            fixture::handoff(MemoryOrder::Release, MemoryOrder::Acquire),
        ),
        (
            "peer_flag",
            fixture::peer_flag(MemoryOrder::Release, MemoryOrder::Acquire),
        ),
        // a two-dimensional cluster: the rank order of every rank
        ("peer_rank0", fixture::rank_probe(0)),
        ("peer_rank1", fixture::rank_probe(1)),
        ("peer_rank2", fixture::rank_probe(2)),
        ("peer_rank3", fixture::rank_probe(3)),
    ]
}

#[test]
fn a_cluster_peer_access_agrees_with_the_launch_interpreter() {
    for (tag, k) in kernels() {
        agree_launch(tag, &k, args());
    }
}

/// A seeded mutant of the emitted C: its name, the kernel, and the text to replace and what replaces it.
struct Mutant {
    tag: &'static str,
    kernel: KernelFn,
    from: &'static str,
    to: &'static str,
}

fn mutants() -> Vec<Mutant> {
    vec![
        // the peer is the block's own: the rank is ignored
        Mutant {
            tag: "peer_mut_rank",
            kernel: fixture::exchange(),
            from: "pr2 % 2)",
            to: "blk0 % 2)",
        },
        // the cluster origin is dropped: ranks count from block 0 of the grid
        Mutant {
            tag: "peer_mut_origin",
            kernel: fixture::exchange(),
            from: "(blk0 / 2 * 2 + ",
            to: "(",
        },
        // the row of the cluster is ignored: a rank on the second row of a two-dimensional cluster lands on the first
        Mutant {
            tag: "peer_mut_row",
            kernel: fixture::rank_probe(2),
            from: "(blk1 / 2 * 2 + (pr1 / 2) % 2)",
            to: "(blk1 / 2 * 2 + pr1 % 2)",
        },
        // the index is taken from the wrong place: always element 0 of the peer's array
        Mutant {
            tag: "peer_mut_index",
            kernel: fixture::exchange(),
            from: "pb2 * 4 + o2]",
            to: "pb2 * 4]",
        },
    ]
}

#[test]
fn the_corpus_catches_seeded_mutants_of_the_emitted_peer_address() {
    for Mutant {
        tag,
        kernel,
        from,
        to,
    } in mutants()
    {
        let mut want = args();
        run_launch(&kernel, &mut want).expect("the oracle runs it");
        let want = written(&want);
        let m = Module::new(vec![kernel.clone()]);
        let g = lower_module(&m, 0).expect("lowers");
        assert_eq!(
            g.source.matches(from).count(),
            1,
            "{tag}: the anchor `{from}` is in the emitted C once:\n{}",
            g.source
        );
        let Some(ran) = run_generated(tag, &kernel, &g, &args(), |s| s.replacen(from, to, 1))
        else {
            return;
        };
        assert_ne!(ran.outs, want, "{tag}: the mutant went unnoticed");
    }
}

/// The `out` buffer's bytes after a run.
fn written(args: &[KernelArg]) -> Vec<Vec<u8>> {
    let KernelArg::Buffer(out) = &args[1] else {
        unreachable!()
    };
    vec![out.bytes.clone()]
}
