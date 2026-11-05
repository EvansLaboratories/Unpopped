// SPDX-License-Identifier: MIT OR Apache-2.0
//! **The serial C of a `CopyAsync` agrees with the launch interpreter** (ROADMAP U8 S1,
//! `docs/options-u8-copy-atoms.md`). cpu-c performs the copy at issue and spells commit and wait as nothing; a kernel
//! that waits before it reads cannot tell, so the C, compiled with a host compiler and run, must write the same bytes
//! as `run_launch`. Kernels: a waited copy read across a barrier, the 8-byte and 4-byte forms, two groups waited in
//! turn, and a copy whose source is offset by a loop variable.

use unpopped::exec::{AllocDecl, ExecId, Extent, LaunchShape, MemorySpace, Scope};
use unpopped::ir::ScalarExpr;
use unpopped::kernel::{
    BarrierKind, Block, BufferRole, KernelFn, LocalDecl, Param, Place, Stmt, check_kernel,
};
use unpopped::oracle::kernel_eval::KernelArg;
use unpopped_vocab::ElementKind;

#[path = "support/cc.rs"]
mod cc;
#[path = "support/kernel_data.rs"]
mod kernel_data;
#[expect(
    dead_code,
    reason = "the shared driver; this file uses the part that runs a launched kernel"
)]
#[path = "support/kernel_run.rs"]
mod kernel_run;
#[path = "support/launch_run.rs"]
mod launch_run;
use kernel_data::{dense, zeroed};
use launch_run::agree_launch;

const I32: ElementKind = ElementKind::I32;
const U32: ElementKind = ElementKind::U32;
const THREADS: u32 = 4;

fn cu(v: i64) -> ScalarExpr {
    ScalarExpr::ConstInt(v, U32)
}

fn tid() -> ScalarExpr {
    ScalarExpr::Exec(ExecId::ThreadIdx(0))
}

fn row(off: i64) -> ScalarExpr {
    ScalarExpr::Add(
        Box::new(ScalarExpr::Mul(Box::new(tid()), Box::new(cu(4)))),
        Box::new(cu(off)),
    )
}

fn mirror(off: i64) -> ScalarExpr {
    ScalarExpr::Add(
        Box::new(ScalarExpr::Mul(
            Box::new(ScalarExpr::Sub(
                Box::new(cu(i64::from(THREADS) - 1)),
                Box::new(tid()),
            )),
            Box::new(cu(4)),
        )),
        Box::new(cu(off)),
    )
}

fn shared(i: ScalarExpr) -> Place {
    Place::Alloc { alloc: 0, index: i }
}

fn global(buffer: u32, i: ScalarExpr) -> Place {
    Place::Logical {
        buffer,
        coords: vec![i],
    }
}

fn copy(dst: Place, src: Place, elems: u32) -> Stmt {
    Stmt::CopyAsync { dst, src, elems }
}

fn barrier() -> Stmt {
    Stmt::Barrier {
        scope: Scope::Block,
        kind: BarrierKind::Full,
    }
}

/// Load the mirror thread's `off` element of the `Shared` array and store it at `out[tid]`.
fn read_mirror(off: i64) -> Vec<Stmt> {
    vec![
        Stmt::Load {
            local: 0,
            place: shared(mirror(off)),
        },
        Stmt::Store {
            place: global(1, tid()),
            value: ScalarExpr::Local(0),
        },
    ]
}

fn kernel(body: Vec<Stmt>) -> KernelFn {
    KernelFn::new(
        "cp",
        I32,
        vec![
            Param::buffer("src", I32, BufferRole::Read, Some(1)),
            Param::buffer("out", I32, BufferRole::ReadWrite, Some(1)),
        ],
        vec![LocalDecl::scalar(I32)],
        Block(body),
    )
    .with_allocs(vec![AllocDecl::new(
        "s",
        MemorySpace::Shared,
        I32,
        Extent::Static(16),
    )])
    .with_launch(LaunchShape::new([THREADS, 1, 1], [1, 1, 1]))
}

fn check(tag: &str, body: Vec<Stmt>) {
    let k = kernel(body);
    assert_eq!(check_kernel(&k), Ok(()), "{tag}: well formed");
    let args = vec![
        KernelArg::Buffer(dense(I32, &[16], 7)),
        KernelArg::Buffer(zeroed(I32, &[4])),
    ];
    agree_launch(tag, &k, args);
}

fn wait(pending: u32) -> Stmt {
    Stmt::AsyncWait { pending }
}

#[test]
fn a_waited_16_byte_copy_read_across_a_barrier() {
    let mut body = vec![
        copy(shared(row(0)), global(0, row(0)), 4),
        Stmt::AsyncCommit,
        wait(0),
        barrier(),
    ];
    body.extend(read_mirror(2));
    check("cp_16", body);
}

#[test]
fn the_8_and_4_byte_forms() {
    let mut body = vec![
        copy(shared(row(0)), global(0, row(2)), 2),
        copy(shared(row(2)), global(0, row(1)), 1),
        Stmt::AsyncCommit,
        wait(0),
        barrier(),
    ];
    body.extend(read_mirror(0));
    check("cp_8_4", body);
    let mut body = vec![
        copy(shared(row(0)), global(0, row(2)), 2),
        copy(shared(row(2)), global(0, row(1)), 1),
        Stmt::AsyncCommit,
        wait(0),
        barrier(),
    ];
    body.extend(read_mirror(2));
    check("cp_4_back", body);
}

#[test]
fn two_groups_waited_in_turn() {
    let mut body = vec![
        copy(shared(row(0)), global(0, row(0)), 2),
        Stmt::AsyncCommit,
        copy(shared(row(2)), global(0, row(2)), 2),
        Stmt::AsyncCommit,
        wait(1),
        wait(0),
        barrier(),
    ];
    body.extend(read_mirror(3));
    check("cp_groups", body);
}

#[test]
fn a_copy_inside_a_loop_whose_source_moves() {
    // for i in 0..2: copy 2 elements from src[row + 2*i] to shared[row + 2*i], commit, wait
    let i = ScalarExpr::Local(1);
    let at = |base: ScalarExpr| {
        ScalarExpr::Add(
            Box::new(base),
            Box::new(ScalarExpr::Mul(Box::new(i.clone()), Box::new(cu(2)))),
        )
    };
    let mut k = kernel(vec![]);
    k.locals.push(LocalDecl::scalar(U32));
    k.body = Block(vec![Stmt::For {
        local: 1,
        start: cu(0),
        end: cu(2),
        step: cu(1),
        body: Block(vec![
            copy(shared(at(row(0))), global(0, at(row(0))), 2),
            Stmt::AsyncCommit,
            wait(0),
        ]),
    }]);
    k.body.0.push(barrier());
    k.body.0.extend(read_mirror(1));
    assert_eq!(check_kernel(&k), Ok(()), "well formed");
    let args = vec![
        KernelArg::Buffer(dense(I32, &[16], 11)),
        KernelArg::Buffer(zeroed(I32, &[4])),
    ];
    agree_launch("cp_loop", &k, args);
}
