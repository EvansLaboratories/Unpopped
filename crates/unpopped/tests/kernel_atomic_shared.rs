// SPDX-License-Identifier: MIT OR Apache-2.0
//! **`Stmt::AtomicRmw` on a `Shared` element** (`docs/ir-kernel-layer.md`, section 5a): a block
//! histogram. Threads count into a `Shared` array with atomics, then add the block's counts to a
//! global array with atomics. The init, the counting and the read need barriers between them (an
//! atomic races with a plain access, never with another atomic); only `Shared` is admitted.

use unpopped::exec::{AllocDecl, ExecId, Extent, LaunchShape, MemorySpace, Scope};
use unpopped::ir::{BinaryOp, ScalarExpr};
use unpopped::kernel::{
    AtomicOp, Block, BufferRole, KernelError, KernelFn, LocalDecl, Param, Place, Stmt, check_kernel,
};
use unpopped::oracle::TypedBuffer;
use unpopped::oracle::kernel_eval::{KernelArg, KernelTrap};
use unpopped::oracle::kernel_launch::run_launch;
use unpopped_vocab::ElementKind;

const I32: ElementKind = ElementKind::I32;
const U32: ElementKind = ElementKind::U32;

fn u(v: i64) -> ScalarExpr {
    ScalarExpr::ConstInt(v, U32)
}

fn tid() -> ScalarExpr {
    ScalarExpr::Exec(ExecId::ThreadIdx(0))
}

fn hist(index: ScalarExpr) -> Place {
    Place::Alloc { alloc: 0, index }
}

fn out(index: ScalarExpr) -> Place {
    Place::Logical {
        buffer: 0,
        coords: vec![index],
    }
}

/// `if tid < 1 { then }`.
fn thread_zero(then: Vec<Stmt>) -> Stmt {
    Stmt::If {
        cond: ScalarExpr::Binary(BinaryOp::CmpLt, Box::new(tid()), Box::new(u(1))),
        then: Block(then),
        els: Block(vec![]),
    }
}

fn barrier() -> Stmt {
    Stmt::Barrier {
        scope: Scope::Block,
    }
}

fn count_into_bin() -> Stmt {
    Stmt::AtomicRmw {
        op: AtomicOp::Add,
        place: hist(u(0)),
        value: ScalarExpr::ConstInt(1, I32),
    }
}

fn init() -> Stmt {
    thread_zero(vec![Stmt::Store {
        place: hist(u(0)),
        value: ScalarExpr::ConstInt(0, I32),
    }])
}

fn publish() -> Stmt {
    thread_zero(vec![
        Stmt::Load {
            local: 0,
            place: hist(u(0)),
        },
        Stmt::AtomicRmw {
            op: AtomicOp::Add,
            place: out(u(0)),
            value: ScalarExpr::Local(0),
        },
    ])
}

fn kernel(space: MemorySpace, body: Vec<Stmt>) -> KernelFn {
    let mut k = KernelFn::new(
        "h",
        I32,
        vec![Param::buffer("out", I32, BufferRole::ReadWrite, Some(1))],
        vec![LocalDecl::scalar(I32)],
        Block(body),
    )
    .with_allocs(vec![AllocDecl::new("hist", space, I32, Extent::Static(1))]);
    k.launch = Some(LaunchShape::new([4, 1, 1], [2, 1, 1]));
    k
}

fn run(k: &KernelFn) -> Result<Vec<i32>, KernelTrap> {
    let buf = TypedBuffer::new(I32, vec![1], vec![1], vec![0; 4]);
    let mut args = vec![KernelArg::Buffer(buf)];
    run_launch(k, &mut args)?;
    let KernelArg::Buffer(b) = &args[0] else {
        unreachable!()
    };
    Ok(b.bytes
        .chunks(4)
        .map(|c| i32::from_le_bytes(c.try_into().unwrap()))
        .collect())
}

fn histogram(body: Vec<Stmt>) -> KernelFn {
    kernel(MemorySpace::Shared, body)
}

#[test]
fn eight_threads_count_into_one_shared_cell() {
    let k = histogram(vec![
        init(),
        barrier(),
        count_into_bin(),
        barrier(),
        publish(),
    ]);
    assert_eq!(check_kernel(&k), Ok(()));
    assert_eq!(run(&k), Ok(vec![8]));
}

#[test]
fn dropping_the_barrier_after_the_init_is_a_race() {
    let k = histogram(vec![init(), count_into_bin(), barrier(), publish()]);
    assert_eq!(run(&k), Err(KernelTrap::DataRace));
}

#[test]
fn dropping_the_barrier_before_the_read_is_a_race() {
    let k = histogram(vec![init(), barrier(), count_into_bin(), publish()]);
    assert_eq!(run(&k), Err(KernelTrap::DataRace));
}

#[test]
fn an_atomic_on_an_element_nothing_stored_traps() {
    let k = histogram(vec![count_into_bin()]);
    assert_eq!(run(&k), Err(KernelTrap::UninitializedShared { alloc: 0 }));
}

#[test]
fn only_shared_is_admitted() {
    for space in [MemorySpace::Private, MemorySpace::Constant] {
        let k = kernel(space, vec![count_into_bin()]);
        assert!(
            matches!(check_kernel(&k), Err(KernelError::AtomicUnsupported { .. })),
            "{space:?}"
        );
    }
}
