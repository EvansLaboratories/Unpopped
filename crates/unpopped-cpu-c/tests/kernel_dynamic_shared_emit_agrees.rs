// SPDX-License-Identifier: MIT OR Apache-2.0
//! **The serial C of a dynamic-extent allocation agrees with the launch interpreter** (ROADMAP U16): an `Extent::Param`
//! `Shared` or `Private` array (CUDA's `extern __shared__`), sized by a scalar launch argument, is spelled as a body that
//! takes its arrays and an entry point that `calloc`s one zeroed copy per block, calls it and frees them. Compiled with a
//! host C compiler and run, it must write the interpreter's bytes for several counts, report an index past the count as
//! the interpreter does (status 1), and refuse a count that is not positive. The seeded mutants of the emitted C break the
//! sizing, the zeroing and the per-block slicing, and each must change the bytes or the status.

use unpopped::exec::{AllocDecl, ExecId, Extent, LaunchShape, MemorySpace, Scope};
use unpopped::ir::ScalarExpr;
use unpopped::kernel::{
    BarrierKind, Block, BufferRole, KernelFn, LocalDecl, Param, Place, Stmt, check_kernel,
};
use unpopped::oracle::kernel_eval::{KernelArg, KernelTrap};
use unpopped::oracle::kernel_launch::run_launch;
use unpopped_vocab::ElementKind;

#[path = "support/cc.rs"]
mod cc;
#[allow(dead_code)]
#[path = "support/cluster_peer.rs"]
mod cp;
#[path = "support/kernel_data.rs"]
mod kernel_data;
#[allow(dead_code)]
#[path = "support/kernel_run.rs"]
mod kernel_run;
#[path = "support/launch_run.rs"]
mod launch_run;
use kernel_data::{dense, zeroed};
use kernel_run::{run_c, run_c_mutated};
use launch_run::agree_launch;

const I32: ElementKind = ElementKind::I32;
const U32: ElementKind = ElementKind::U32;
const THREADS: u32 = 4;
const BLOCKS: u32 = 2;

fn tid() -> ScalarExpr {
    ScalarExpr::Exec(ExecId::ThreadIdx(0))
}

fn global() -> ScalarExpr {
    ScalarExpr::Add(
        Box::new(ScalarExpr::Mul(
            Box::new(ScalarExpr::Exec(ExecId::BlockIdx(0))),
            Box::new(ScalarExpr::ConstInt(i64::from(THREADS), U32)),
        )),
        Box::new(tid()),
    )
}

fn at(buffer: u32, i: ScalarExpr) -> Place {
    Place::Logical {
        buffer,
        coords: vec![i],
    }
}

fn tile(i: ScalarExpr) -> Place {
    Place::Alloc { alloc: 0, index: i }
}

/// `tile[tid] = in[global]`, a barrier, then `out[global] = tile[THREADS - 1 - tid]`: the count `n` decides whether the
/// mirror index is in range.
fn mirrored(space: MemorySpace) -> KernelFn {
    let mirror = ScalarExpr::Sub(
        Box::new(ScalarExpr::ConstInt(i64::from(THREADS) - 1, U32)),
        Box::new(tid()),
    );
    // a `Private` array is each thread's own, so it holds the thread's value at its own index
    let read = if space == MemorySpace::Private {
        tid()
    } else {
        mirror
    };
    KernelFn::new(
        "dyn_mirror",
        I32,
        vec![
            Param::buffer("in", I32, BufferRole::Read, Some(1)),
            Param::buffer("out", I32, BufferRole::Write, Some(1)),
            Param::scalar("n", I32, 0),
        ],
        vec![
            LocalDecl::scalar(I32),
            LocalDecl::scalar(I32),
            LocalDecl::scalar(I32),
        ],
        Block(vec![
            Stmt::Load {
                local: 1,
                place: at(0, global()),
            },
            Stmt::Store {
                place: tile(tid()),
                value: ScalarExpr::Local(1),
            },
            Stmt::Barrier {
                scope: Scope::Block,
                kind: BarrierKind::Full,
            },
            Stmt::Load {
                local: 2,
                place: tile(read),
            },
            Stmt::Store {
                place: at(1, global()),
                value: ScalarExpr::Local(2),
            },
        ]),
    )
    .with_allocs(vec![AllocDecl::new("tile", space, I32, Extent::Param(0))])
    .with_launch(LaunchShape::new([THREADS, 1, 1], [BLOCKS, 1, 1]))
}

fn args(n: i128) -> Vec<KernelArg> {
    let len = (THREADS * BLOCKS) as i64;
    vec![
        KernelArg::Buffer(dense(I32, &[len], 5)),
        KernelArg::Buffer(zeroed(I32, &[len])),
        KernelArg::Int(n),
    ]
}

#[test]
fn a_dynamic_shared_array_agrees_for_each_count() {
    let k = mirrored(MemorySpace::Shared);
    assert_eq!(check_kernel(&k), Ok(()), "well formed");
    for n in [4, 9] {
        agree_launch(&format!("dyn_shared_{n}"), &k, args(n));
    }
}

#[test]
fn an_index_past_the_count_is_reported_as_the_interpreter_does() {
    let k = mirrored(MemorySpace::Shared);
    for n in [3, 1] {
        let mut interp = args(n);
        let trap = run_launch(&k, &mut interp);
        assert!(
            matches!(trap, Err(KernelTrap::AllocOutOfBounds { alloc: 0 })),
            "n = {n}: {trap:?}"
        );
        let Some(ran) = run_c("dyn_past_count", &k, &args(n)) else {
            return;
        };
        assert_eq!(ran.status, 1, "n = {n}");
    }
}

#[test]
fn a_dynamic_private_array_agrees() {
    let k = mirrored(MemorySpace::Private);
    assert_eq!(check_kernel(&k), Ok(()), "well formed");
    for n in [4, 6] {
        agree_launch(&format!("dyn_private_{n}"), &k, args(n));
    }
}

#[test]
fn the_emitted_text_has_a_body_and_an_entry_point_that_allocates_and_frees() {
    use unpopped::backend::Backend;
    let g = unpopped_cpu_c::CpuC
        .lower_kernel(&mirrored(MemorySpace::Shared))
        .expect("lowers");
    for needle in [
        "#include <stdlib.h>",
        "static int unpopped_cpu_kernel_dyn_mirror_impl(",
        "int* sh0_dyn)",
        "calloc(",
        "free(sh0_dyn);",
    ] {
        assert!(g.source.contains(needle), "{needle}:\n{}", g.source);
    }
    // a kernel with only static allocations is spelled as it was: no allocation, no extra include
    let plain = unpopped_cpu_c::CpuC
        .lower_kernel(
            &KernelFn::new(
                "plain",
                I32,
                vec![Param::buffer("out", I32, BufferRole::Write, Some(1))],
                vec![],
                Block(vec![]),
            )
            .with_launch(LaunchShape::new([1, 1, 1], [1, 1, 1])),
        )
        .expect("lowers");
    assert!(!plain.source.contains("calloc"), "{}", plain.source);
}

#[test]
fn a_count_that_is_not_positive_is_refused_by_the_entry_point() {
    let k = mirrored(MemorySpace::Shared);
    for n in [0, -3] {
        let Some(ran) = run_c("dyn_nonpositive", &k, &args(n)) else {
            return;
        };
        assert_eq!(ran.status, 2, "n = {n}");
    }
}

/// A seeded mutant of the emitted C: its tag, the text to replace (everywhere) and what replaces it.
const MUTANTS: [(&str, &str, &str); 2] = [
    // the bounds check admits one element past the count
    ("dyn_mut_bound", ">= ((long long)p2)", "> ((long long)p2)"),
    // a zero count is no longer refused
    ("dyn_mut_zero", "dn0 <= 0", "dn0 < 0"),
];

#[test]
fn the_corpus_catches_seeded_mutants_of_the_emitted_sizing() {
    let k = mirrored(MemorySpace::Shared);
    let g = unpopped::backend::Backend::lower_kernel(&unpopped_cpu_c::CpuC, &k).expect("lowers");
    for (tag, from, _) in MUTANTS {
        assert!(
            g.source.contains(from),
            "{tag}: the anchor `{from}` is in the emitted C:
{}",
            g.source
        );
    }
    // with a count of 3 the last thread's write is out of range (status 1); the mutant lets it through
    let (tag, from, to) = MUTANTS[0];
    let past = args(3);
    let got = run_c_mutated(tag, &k, &past, |s| s.replace(from, to)).expect("a C compiler");
    assert_ne!(got.status, 1, "{tag}: the check must no longer fire");
    // a count of 0 is refused (status 2); the mutant lets it through
    let (tag, from, to) = MUTANTS[1];
    let zero = args(0);
    let got = run_c_mutated(tag, &k, &zero, |s| s.replace(from, to)).expect("a C compiler");
    assert_ne!(
        got.status, 2,
        "{tag}: a zero count must no longer be refused"
    );
}

/// Each thread publishes `in[gid]` in its block's dynamic `Shared` array, the cluster of 2 blocks meets, and each thread
/// reads the same slot of the other block: the array of the whole grid is kept resident (`sh0_all`), sized by the count.
fn cluster_exchange() -> KernelFn {
    let body = vec![
        cp::load(1, cp::at(0, cp::gid())),
        cp::store(cp::sh(cp::thread()), cp::local(1)),
        cp::cluster_barrier(),
        cp::load(2, cp::peer(cp::other(), cp::thread())),
        cp::store(cp::at(1, cp::gid()), cp::local(2)),
    ];
    KernelFn::new(
        "dyn_cluster",
        I32,
        vec![
            Param::buffer("in", I32, BufferRole::Read, Some(1)),
            Param::buffer("out", I32, BufferRole::Write, Some(1)),
            Param::scalar("n", I32, 0),
        ],
        vec![
            LocalDecl::scalar(I32),
            LocalDecl::scalar(I32),
            LocalDecl::scalar(I32),
        ],
        Block(body),
    )
    .with_allocs(vec![AllocDecl::new(
        "sh",
        MemorySpace::Shared,
        I32,
        Extent::Param(0),
    )])
    .with_launch(cp::launch())
}

#[test]
fn a_dynamic_array_a_cluster_peer_reads_agrees() {
    let k = cluster_exchange();
    assert_eq!(check_kernel(&k), Ok(()), "well formed");
    let args = |n| {
        vec![
            KernelArg::Buffer(dense(I32, &[16], 3)),
            KernelArg::Buffer(zeroed(I32, &[16])),
            KernelArg::Int(n),
        ]
    };
    for n in [4, 7] {
        agree_launch(&format!("dyn_cluster_{n}"), &k, args(n));
    }
}
