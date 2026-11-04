// SPDX-License-Identifier: MIT OR Apache-2.0
//! **A `Private` array in the serialized thread-loop C agrees with the launch interpreter** (ROADMAP U7 S2,
//! the groundwork for `Stmt::Mma` fragments). A `Private` allocation is one instance per thread: the C spells
//! it as one slice of `n` elements per thread of the block, so thread `t`'s element `j` is `t * n + j`. A
//! wrong stride or a slice shared between threads makes two threads read each other's values, which the
//! per-thread-distinct data below turns into a different sum. Compared byte for byte with `run_launch`.

use unpopped::exec::{AllocDecl, ExecId, Extent, LaunchShape, MemorySpace, Scope};
use unpopped::ir::ScalarExpr;
use unpopped::kernel::{Block, BufferRole, KernelFn, LocalDecl, Param, Place, Stmt};
use unpopped::oracle::TypedBuffer;
use unpopped::oracle::kernel_eval::KernelArg;
use unpopped_vocab::ElementKind;

#[path = "support/cc.rs"]
mod cc;
// The shared compile-and-run drivers; this file uses a subset of them.
#[allow(dead_code)]
#[path = "support/kernel_run.rs"]
mod kernel_run;
#[allow(dead_code)]
#[path = "support/launch_run.rs"]
mod launch_run;
use launch_run::agree_launch;

const U32: ElementKind = ElementKind::U32;

fn u(v: i64) -> ScalarExpr {
    ScalarExpr::ConstInt(v, U32)
}

fn add(a: ScalarExpr, b: ScalarExpr) -> ScalarExpr {
    ScalarExpr::Add(Box::new(a), Box::new(b))
}

fn mul(a: ScalarExpr, b: ScalarExpr) -> ScalarExpr {
    ScalarExpr::Mul(Box::new(a), Box::new(b))
}

fn scratch(j: i64) -> Place {
    Place::Alloc {
        alloc: 0,
        index: u(j),
    }
}

/// local 0 = running sum of the scratch, locals 1..=n = the elements read.
fn sum_scratch(n: i64) -> Vec<Stmt> {
    let mut out = vec![Stmt::Let {
        local: 0,
        init: u(0),
    }];
    for j in 0..n {
        out.push(Stmt::Load {
            local: 1 + j as u32,
            place: scratch(j),
        });
        out.push(Stmt::Assign {
            local: 0,
            value: add(ScalarExpr::Local(0), ScalarExpr::Local(1 + j as u32)),
        });
    }
    out
}

/// Thread `g` (global index) fills `scratch[j] = g * 10 + j`, optionally a Device barrier, then sums its own
/// scratch into `out[g]`. `N` is the scratch length.
fn kernel(threads: u32, blocks: u32, n: i64, device_barrier: bool) -> KernelFn {
    let tid = ScalarExpr::Exec(ExecId::ThreadIdx(0));
    let bid = ScalarExpr::Exec(ExecId::BlockIdx(0));
    let g = add(mul(bid, u(i64::from(threads))), tid);
    let mut body = vec![];
    for j in 0..n {
        body.push(Stmt::Store {
            place: scratch(j),
            value: add(mul(g.clone(), u(10)), u(j)),
        });
    }
    body.push(Stmt::barrier(Scope::Block));
    if device_barrier {
        body.push(Stmt::barrier(Scope::Device));
    }
    body.extend(sum_scratch(n));
    body.push(Stmt::Store {
        place: Place::Logical {
            buffer: 0,
            coords: vec![g],
        },
        value: ScalarExpr::Local(0),
    });
    let mut k = KernelFn::new(
        "p",
        U32,
        vec![Param::buffer("out", U32, BufferRole::ReadWrite, Some(1))],
        (0..=n).map(|_| LocalDecl::scalar(U32)).collect(),
        Block(body),
    )
    .with_allocs(vec![AllocDecl::new(
        "scratch",
        MemorySpace::Private,
        U32,
        Extent::Static(n as u32),
    )]);
    k.launch = Some(LaunchShape::new([threads, 1, 1], [blocks, 1, 1]));
    k
}

fn out(len: usize) -> TypedBuffer {
    TypedBuffer::new(U32, vec![len as i64], vec![1], vec![0; len * 4])
}

#[test]
fn a_private_array_is_one_instance_per_thread() {
    for (threads, blocks, n) in [(8, 1, 3), (5, 3, 1), (4, 2, 4)] {
        let k = kernel(threads, blocks, n, false);
        agree_launch(
            &format!("private_{threads}x{blocks}x{n}"),
            &k,
            vec![KernelArg::Buffer(out((threads * blocks) as usize))],
        );
    }
}

#[test]
fn a_private_array_survives_a_device_barrier() {
    for (threads, blocks, n) in [(4, 3, 3), (2, 2, 2)] {
        let k = kernel(threads, blocks, n, true);
        agree_launch(
            &format!("private_dev_{threads}x{blocks}x{n}"),
            &k,
            vec![KernelArg::Buffer(out((threads * blocks) as usize))],
        );
    }
}

/// `Private` is spelled now; `Constant` still is not, and the refusal says which spaces are.
#[test]
fn a_constant_allocation_is_still_refused_with_the_spelled_spaces_named() {
    use unpopped::backend::{Backend, LowerError};
    let mut k = kernel(4, 1, 2, false);
    // the checker refuses a store to a Constant array first, so lower a body that stores nothing
    k.body = Block(vec![]);
    k.allocs[0].space = MemorySpace::Constant;
    match unpopped_cpu_c::CpuC.lower_kernel(&k) {
        Err(LowerError::UnsupportedKernel { detail }) => {
            assert!(detail.contains("only Shared and Private"), "{detail}");
        }
        other => panic!("expected a typed refusal, got {other:?}"),
    }
}
