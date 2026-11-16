// SPDX-License-Identifier: MIT OR Apache-2.0
//! **`CopyBulk` in the launch interpreter** (ROADMAP U8c slice 2, `docs/options-u8-copy-atoms.md` S3): a one-thread bulk
//! copy from a global buffer into `Shared` that completes on an `mbarrier` (`cp.async.bulk ... mbarrier::complete_tx`).
//! Each behaviour has a program that shows it and a control that does not: a consumer that waits on the barrier sees the
//! copied data without a `DataRace`; a copy whose bytes were announced but not all delivered leaves the phase open
//! (`MbarrierDeadlock`); a consumer that does not wait races the copy; the checker refuses what a bulk copy cannot be.
//! The consumer is thread 0 and runs first, so every successful wait here really blocked and was woken.

use unpopped::exec::{AllocDecl, ExecId, Extent, LaunchShape, MemorySpace, Scope};
use unpopped::ir::{BinaryOp, ScalarExpr};
use unpopped::kernel::{
    Block, BufferRole, KernelError, KernelFn, LocalDecl, Param, Place, Stmt, check_kernel,
};
use unpopped::oracle::TypedBuffer;
use unpopped::oracle::kernel_eval::{KernelArg, KernelTrap};
use unpopped::oracle::kernel_launch::run_launch;
use unpopped_vocab::ElementKind;

const U32: ElementKind = ElementKind::U32;

fn u(v: i64) -> ScalarExpr {
    ScalarExpr::ConstInt(v, U32)
}
fn bar() -> Place {
    Place::Alloc {
        alloc: 0,
        index: u(0),
    }
}
fn tile(i: i64) -> Place {
    Place::Alloc {
        alloc: 1,
        index: u(i),
    }
}
fn input(i: i64) -> Place {
    Place::Logical {
        buffer: 0,
        coords: vec![u(i)],
    }
}
fn on(t: i64, body: Vec<Stmt>) -> Stmt {
    Stmt::If {
        cond: ScalarExpr::Binary(
            BinaryOp::CmpEq,
            Box::new(ScalarExpr::Exec(ExecId::ThreadIdx(0))),
            Box::new(u(t)),
        ),
        then: Block(body),
        els: Block(vec![]),
    }
}
fn copy(elems: u32) -> Stmt {
    Stmt::CopyBulk {
        dst: tile(0),
        src: input(0),
        elems,
        bar: bar(),
    }
}
/// `out[0] = tile[0] + tile[1] + tile[2] + tile[3]`.
fn publish() -> Vec<Stmt> {
    let mut v: Vec<Stmt> = (0..4)
        .map(|i| Stmt::Load {
            local: i as u32,
            place: tile(i),
        })
        .collect();
    let sum = (1..4).fold(ScalarExpr::Local(0), |a, i| {
        ScalarExpr::Add(Box::new(a), Box::new(ScalarExpr::Local(i)))
    });
    v.push(Stmt::Store {
        place: Place::Logical {
            buffer: 1,
            coords: vec![u(0)],
        },
        value: sum,
    });
    v
}

/// Two threads; the barrier (Shared `u64`) is allocation 0 and four `u32` tile cells allocation 1.
fn kernel(body: Vec<Stmt>) -> KernelFn {
    let mut k = KernelFn::new(
        "bulk",
        U32,
        vec![
            Param::buffer("in", U32, BufferRole::Read, Some(1)),
            Param::buffer("out", U32, BufferRole::ReadWrite, Some(1)),
        ],
        (0..4).map(|_| LocalDecl::scalar(U32)).collect(),
        Block(body),
    )
    .with_launch(LaunchShape::new([2, 1, 1], [1, 1, 1]));
    k.allocs = vec![
        AllocDecl::new(
            "bar",
            MemorySpace::Shared,
            ElementKind::U64,
            Extent::Static(1),
        ),
        AllocDecl::new("tile", MemorySpace::Shared, U32, Extent::Static(4)),
    ];
    k
}

fn run(k: &KernelFn) -> Result<u32, KernelTrap> {
    assert_eq!(check_kernel(k), Ok(()));
    let words: Vec<u8> = [1u32, 2, 4, 8]
        .iter()
        .flat_map(|w| w.to_le_bytes())
        .collect();
    let mut args = [
        KernelArg::Buffer(TypedBuffer::new(U32, vec![4], vec![1], words)),
        KernelArg::Buffer(TypedBuffer::new(U32, vec![1], vec![1], vec![0; 4])),
    ];
    run_launch(k, &mut args)?;
    let KernelArg::Buffer(b) = &args[1] else {
        unreachable!("a buffer")
    };
    Ok(u32::from_le_bytes(b.bytes[..4].try_into().unwrap()))
}

fn init() -> Stmt {
    on(
        0,
        vec![Stmt::MbarInit {
            bar: bar(),
            count: u(1),
        }],
    )
}
fn arrive(bytes: i64) -> Stmt {
    Stmt::MbarArrive {
        bar: bar(),
        tx: Some(u(bytes)),
    }
}
fn consumer() -> Stmt {
    let mut then = vec![Stmt::MbarWait {
        bar: bar(),
        parity: u(0),
    }];
    then.extend(publish());
    on(0, then)
}

/// Thread 0 consumes; thread 1 announces `announce` bytes and copies `elems` elements.
fn scenario(announce: i64, elems: u32) -> KernelFn {
    kernel(vec![
        init(),
        Stmt::barrier(Scope::Block),
        consumer(),
        on(1, vec![arrive(announce), copy(elems)]),
    ])
}

#[test]
fn a_consumer_that_waits_sees_the_copied_tile() {
    assert_eq!(run(&scenario(16, 4)), Ok(15));
}

#[test]
fn bytes_announced_but_not_delivered_leave_the_phase_open() {
    // 32 bytes announced, 16 delivered
    assert_eq!(run(&scenario(32, 4)), Err(KernelTrap::MbarrierDeadlock));
}

#[test]
fn a_consumer_that_does_not_wait_races_the_copy() {
    let k = kernel(vec![
        init(),
        Stmt::barrier(Scope::Block),
        on(0, publish()),
        on(1, vec![arrive(16), copy(4)]),
    ]);
    assert!(matches!(
        run(&k),
        Err(KernelTrap::DataRace | KernelTrap::UninitializedShared { .. })
    ));
}

#[test]
fn a_copy_that_delivers_before_the_announcement_still_completes_the_phase() {
    // complete_tx may run ahead of expect_tx within a phase; the arrival then closes it
    let k = kernel(vec![
        init(),
        Stmt::barrier(Scope::Block),
        consumer(),
        on(1, vec![copy(4), arrive(16)]),
    ]);
    assert_eq!(run(&k), Ok(15));
}

#[test]
fn a_copy_to_an_uninitialised_barrier_is_a_misuse_trap() {
    let k = kernel(vec![on(1, vec![copy(4)])]);
    assert!(matches!(run(&k), Err(KernelTrap::MbarrierMisuse { .. })));
}

#[test]
fn the_checker_refuses_what_a_bulk_copy_cannot_be() {
    let refused = |k: KernelFn| {
        matches!(
            check_kernel(&k),
            Err(KernelError::BulkCopyUnsupported { .. })
        )
    };
    // 8 bytes is not a multiple of 16
    assert!(refused(kernel(vec![copy(2)])), "8 bytes");
    let mut narrow = kernel(vec![copy(4)]);
    narrow.allocs[0] = AllocDecl::new("bar", MemorySpace::Shared, U32, Extent::Static(1));
    assert!(refused(narrow), "32-bit barrier");
    let to_global = kernel(vec![Stmt::CopyBulk {
        dst: input(0),
        src: input(0),
        elems: 4,
        bar: bar(),
    }]);
    assert!(refused(to_global), "global destination");
    assert_eq!(check_kernel(&kernel(vec![copy(4)])), Ok(()));
    assert_eq!(check_kernel(&kernel(vec![copy(8)])), Ok(()));
}
