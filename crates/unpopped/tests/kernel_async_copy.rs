// SPDX-License-Identifier: MIT OR Apache-2.0
//! **`cp.async` in the oracle** (ROADMAP U8 S1, `docs/options-u8-copy-atoms.md`): `CopyAsync` reads its source at
//! issue and lands on the `Shared` destination only at the `AsyncWait` that covers its committed group. The oracle is
//! stricter than hardware, on purpose: touching a destination element before it lands is the typed `AsyncRace`, a copy
//! never waited for is `AsyncUnwaited`, and a landed copy is an ordinary `Shared` write, so another thread reads it
//! only after a barrier (`DataRace` otherwise). The checker refuses a destination that is not a `Shared` allocation, a
//! source that is not a readable global buffer, mismatched element types, and sizes other than 4, 8 or 16 bytes.

use unpopped::exec::{AllocDecl, ExecId, Extent, LaunchShape, MemorySpace, Scope};
use unpopped::ir::ScalarExpr;
use unpopped::kernel::{
    BarrierKind, Block, BufferRole, KernelError, KernelFn, LocalDecl, Param, Place, Stmt,
    check_kernel,
};
use unpopped::oracle::TypedBuffer;
use unpopped::oracle::kernel_eval::{KernelArg, KernelTrap};
use unpopped::oracle::kernel_launch::run_launch;
use unpopped_vocab::ElementKind;

const I32: ElementKind = ElementKind::I32;
const U32: ElementKind = ElementKind::U32;
const THREADS: u32 = 4;

fn cu(v: i64) -> ScalarExpr {
    ScalarExpr::ConstInt(v, U32)
}

fn tid() -> ScalarExpr {
    ScalarExpr::Exec(ExecId::ThreadIdx(0))
}

fn mul(a: ScalarExpr, b: ScalarExpr) -> ScalarExpr {
    ScalarExpr::Mul(Box::new(a), Box::new(b))
}

fn add(a: ScalarExpr, b: ScalarExpr) -> ScalarExpr {
    ScalarExpr::Add(Box::new(a), Box::new(b))
}

/// `tid * 4 + off`: thread `t`'s 4-element row, `off` into it.
fn row(off: i64) -> ScalarExpr {
    add(mul(tid(), cu(4)), cu(off))
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

fn wait(pending: u32) -> Stmt {
    Stmt::AsyncWait { pending }
}

fn barrier() -> Stmt {
    Stmt::Barrier {
        scope: Scope::Block,
        kind: BarrierKind::Full,
    }
}

fn load(local: u32, place: Place) -> Stmt {
    Stmt::Load { local, place }
}

fn store_out(local: u32) -> Stmt {
    Stmt::Store {
        place: global(1, tid()),
        value: ScalarExpr::Local(local),
    }
}

/// `k(src: I32 read, out: I32 read-write)` on four threads, with a 16-element `Shared` array of I32.
fn kernel(body: Vec<Stmt>) -> KernelFn {
    KernelFn::new(
        "k",
        I32,
        vec![
            Param::buffer("src", I32, BufferRole::Read, Some(1)),
            Param::buffer("out", I32, BufferRole::ReadWrite, Some(1)),
        ],
        vec![LocalDecl::scalar(I32), LocalDecl::scalar(I32)],
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

fn buffer(values: &[i32]) -> KernelArg {
    KernelArg::Buffer(TypedBuffer::new(
        I32,
        vec![values.len() as i64],
        vec![1],
        values.iter().flat_map(|v| v.to_le_bytes()).collect(),
    ))
}

/// `src[i] = 100 + i`, sixteen of them, and a zeroed `out` of four.
fn args() -> [KernelArg; 2] {
    let src: Vec<i32> = (0..16).map(|i| 100 + i).collect();
    [buffer(&src), buffer(&[0; 4])]
}

fn out_of(a: &KernelArg) -> Vec<i32> {
    let KernelArg::Buffer(b) = a else {
        unreachable!("a buffer")
    };
    b.bytes
        .chunks(4)
        .map(|c| i32::from_le_bytes(c.try_into().unwrap()))
        .collect()
}

fn run(k: &KernelFn) -> Result<Vec<i32>, KernelTrap> {
    let mut a = args();
    run_launch(k, &mut a)?;
    Ok(out_of(&a[1]))
}

/// Each thread copies its row, commits, waits for everything, passes a barrier, then reads its mirror thread's
/// element 1 (thread `3 - t`) and stores it.
fn neighbour(extra_before_wait: Vec<Stmt>, wait_n: u32, with_barrier: bool) -> KernelFn {
    let mut body = vec![copy(shared(row(0)), global(0, row(0)), 4)];
    body.push(Stmt::AsyncCommit);
    body.extend(extra_before_wait);
    body.push(wait(wait_n));
    if with_barrier {
        body.push(barrier());
    }
    // the mirror thread's row, element 1: shared[(3 - tid) * 4 + 1]
    body.push(load(
        0,
        shared(add(
            mul(
                ScalarExpr::Sub(Box::new(cu(i64::from(THREADS) - 1)), Box::new(tid())),
                cu(4),
            ),
            cu(1),
        )),
    ));
    body.push(store_out(0));
    kernel(body)
}

#[test]
fn a_waited_copy_lands_and_a_neighbour_reads_it_after_a_barrier() {
    let k = neighbour(vec![], 0, true);
    assert_eq!(check_kernel(&k), Ok(()));
    // thread t reads shared[(3 - t) * 4 + 1] = src[same] = 100 + that index
    assert_eq!(run(&k), Ok(vec![113, 109, 105, 101]));
}

#[test]
fn without_the_barrier_the_mirror_thread_has_not_landed_its_copy_yet() {
    // thread 0 runs its phase first and reads a row nobody has landed: Shared is never zero-filled
    let k = neighbour(vec![], 0, false);
    assert_eq!(run(&k), Err(KernelTrap::UninitializedShared { alloc: 0 }));
}

#[test]
fn two_threads_landing_on_one_element_without_a_barrier_is_a_data_race() {
    let k = kernel(vec![
        copy(shared(cu(0)), global(0, tid()), 1),
        Stmt::AsyncCommit,
        wait(0),
    ]);
    assert_eq!(run(&k), Err(KernelTrap::DataRace));
}

#[test]
fn reading_the_destination_before_the_wait_is_an_async_race() {
    let k = neighbour(vec![load(1, shared(row(0)))], 0, true);
    assert_eq!(run(&k), Err(KernelTrap::AsyncRace { alloc: 0 }));
    // control: the same read after the wait is fine
    let ok = kernel(vec![
        copy(shared(row(0)), global(0, row(0)), 4),
        Stmt::AsyncCommit,
        wait(0),
        load(0, shared(row(0))),
        store_out(0),
    ]);
    assert_eq!(run(&ok), Ok(vec![100, 104, 108, 112]));
}

#[test]
fn storing_or_copying_onto_an_in_flight_destination_is_an_async_race() {
    let store = kernel(vec![
        copy(shared(row(0)), global(0, row(0)), 4),
        Stmt::AsyncCommit,
        Stmt::Store {
            place: shared(row(0)),
            value: ScalarExpr::ConstInt(7, I32),
        },
        wait(0),
    ]);
    assert_eq!(run(&store), Err(KernelTrap::AsyncRace { alloc: 0 }));
    let again = kernel(vec![
        copy(shared(row(0)), global(0, row(0)), 4),
        copy(shared(row(1)), global(0, row(0)), 4),
    ]);
    assert_eq!(run(&again), Err(KernelTrap::AsyncRace { alloc: 0 }));
}

#[test]
fn a_copy_never_waited_for_is_reported_when_the_thread_ends() {
    // committed but never waited
    let committed = kernel(vec![
        copy(shared(row(0)), global(0, row(0)), 4),
        Stmt::AsyncCommit,
    ]);
    assert_eq!(run(&committed), Err(KernelTrap::AsyncUnwaited));
    // not even committed
    let open = kernel(vec![copy(shared(row(0)), global(0, row(0)), 4)]);
    assert_eq!(run(&open), Err(KernelTrap::AsyncUnwaited));
    // control: a wait on the committed group finishes cleanly
    let ok = kernel(vec![
        copy(shared(row(0)), global(0, row(0)), 4),
        Stmt::AsyncCommit,
        wait(0),
    ]);
    assert_eq!(run(&ok), Ok(vec![0; 4]));
}

#[test]
fn a_wait_does_not_cover_a_copy_that_is_not_committed() {
    let k = kernel(vec![
        copy(shared(row(0)), global(0, row(0)), 4),
        wait(0),
        load(0, shared(row(0))),
    ]);
    assert_eq!(run(&k), Err(KernelTrap::AsyncRace { alloc: 0 }));
}

#[test]
fn wait_pending_lands_only_the_older_groups() {
    // group A -> row element 0..4 of the thread's row, group B -> the same row's copy at +0 of the second half is not
    // available (16 elements / 4 threads = 4 each), so B copies two elements to the front of the array's tail.
    let two_groups = |first_read: i64| {
        kernel(vec![
            copy(shared(row(0)), global(0, row(0)), 2),
            Stmt::AsyncCommit,
            copy(shared(row(2)), global(0, row(2)), 2),
            Stmt::AsyncCommit,
            wait(1),
            load(0, shared(row(first_read))),
            store_out(0),
            wait(0),
        ])
    };
    // after wait(1) the older group (elements 0..2 of the row) has landed
    assert_eq!(run(&two_groups(1)), Ok(vec![101, 105, 109, 113]));
    // the newer group (elements 2..4) is still in flight
    assert_eq!(run(&two_groups(2)), Err(KernelTrap::AsyncRace { alloc: 0 }));
}

#[test]
fn an_empty_commit_is_a_group_so_wait_counts_line_up() {
    // one copy, then an empty commit: wait(1) leaves one group in flight, and that one is the EMPTY newest, so the
    // copy's group (the older) lands
    let k = kernel(vec![
        copy(shared(row(0)), global(0, row(0)), 4),
        Stmt::AsyncCommit,
        Stmt::AsyncCommit,
        wait(1),
        load(0, shared(row(3))),
        store_out(0),
        wait(0),
    ]);
    assert_eq!(run(&k), Ok(vec![103, 107, 111, 115]));
}

#[test]
fn a_copy_off_either_end_is_a_typed_trap() {
    // the last thread's first element (15) is inside the source, the fourth is not
    let src = kernel(vec![
        copy(shared(row(0)), global(0, row(3)), 4),
        Stmt::AsyncCommit,
        wait(0),
    ]);
    assert!(matches!(run(&src), Err(KernelTrap::OutOfBounds { .. })));
    // the last thread's first destination element (14) is inside the array, the fourth is not
    // (no wait: it is refused at issue, not when the landing would have failed)
    let last = ScalarExpr::Binary(
        unpopped::ir::BinaryOp::CmpEq,
        Box::new(tid()),
        Box::new(cu(i64::from(THREADS) - 1)),
    );
    let dst = kernel(vec![Stmt::If {
        cond: last,
        then: Block(vec![copy(shared(row(2)), global(0, row(0)), 4)]),
        els: Block(vec![]),
    }]);
    assert_eq!(run(&dst), Err(KernelTrap::AllocOutOfBounds { alloc: 0 }));
}

#[test]
fn the_issue_is_a_logged_write_so_a_refill_races_a_reader_even_if_a_barrier_precedes_the_landing() {
    // every thread fills its slot, barrier, reads its mirror's slot, then issues a copy onto its OWN slot with nothing
    // ordering that issue after the mirror's read; the barrier before the wait would order only the LANDING. Hardware
    // may write the destination any time after issue, so the issue itself races the read.
    let body = |read_barrier: bool| {
        let mut b = vec![
            Stmt::Store {
                place: shared(tid()),
                value: ScalarExpr::ConstInt(7, I32),
            },
            barrier(),
            load(
                1,
                shared(ScalarExpr::Sub(
                    Box::new(cu(i64::from(THREADS) - 1)),
                    Box::new(tid()),
                )),
            ),
        ];
        if read_barrier {
            b.push(barrier());
        }
        b.extend([
            copy(shared(tid()), global(0, tid()), 1),
            Stmt::AsyncCommit,
            barrier(),
            wait(0),
        ]);
        kernel(b)
    };
    assert_eq!(
        run(&body(true)),
        Ok(vec![0; 4]),
        "control: a barrier after the read"
    );
    assert_eq!(run(&body(false)), Err(KernelTrap::DataRace));
}

fn refused(body: Vec<Stmt>) -> KernelError {
    check_kernel(&kernel(body)).expect_err("the checker refuses it")
}

#[test]
fn the_checker_refuses_each_malformed_copy_by_its_text() {
    let ok = |elems: u32| {
        check_kernel(&kernel(vec![copy(
            shared(row(0)),
            global(0, row(0)),
            elems,
        )]))
    };
    assert_eq!(ok(1), Ok(()), "4 bytes");
    assert_eq!(ok(2), Ok(()), "8 bytes");
    assert_eq!(ok(4), Ok(()), "16 bytes");
    for bad in [3, 5, 8] {
        assert!(
            matches!(ok(bad), Err(KernelError::AsyncUnsupported { what }) if what.contains("4, 8 or 16 bytes")),
            "{bad} elements"
        );
    }
    // a global destination
    let to_global = refused(vec![copy(global(1, row(0)), global(0, row(0)), 4)]);
    assert!(
        matches!(to_global, KernelError::AsyncUnsupported { what } if what.contains("destination")),
        "{to_global:?}"
    );
    // a source that is a Shared allocation
    let from_shared = refused(vec![copy(shared(row(0)), shared(row(0)), 4)]);
    assert!(
        matches!(from_shared, KernelError::AsyncUnsupported { what } if what.contains("source")),
        "{from_shared:?}"
    );
    // a write-only source
    let mut k = kernel(vec![copy(shared(row(0)), global(1, row(0)), 4)]);
    k.params[1] = Param::buffer("out", I32, BufferRole::Write, Some(1));
    assert!(matches!(
        check_kernel(&k),
        Err(KernelError::ReadFromWriteOnly { buffer: 1 })
    ));
    // an element type that differs
    let mut k = kernel(vec![copy(shared(row(0)), global(0, row(0)), 4)]);
    k.params[0] = Param::buffer("src", ElementKind::F32, BufferRole::Read, Some(1));
    assert!(matches!(
        check_kernel(&k),
        Err(KernelError::TypeMismatch { .. })
    ));
}

#[test]
fn a_copy_overlapping_an_in_flight_one_from_below_is_an_async_race() {
    // elements 2..4 are in flight; the second copy starts at 0, so its first element is free but 2..4 are not
    let k = kernel(vec![
        copy(shared(row(2)), global(0, row(2)), 2),
        copy(shared(row(0)), global(0, row(0)), 4),
    ]);
    assert_eq!(run(&k), Err(KernelTrap::AsyncRace { alloc: 0 }));
}

#[test]
fn a_destination_in_another_space_is_refused() {
    let mut k = kernel(vec![copy(shared(row(0)), global(0, row(0)), 4)]);
    k.allocs[0] = AllocDecl::new("p", MemorySpace::Private, I32, Extent::Static(16));
    assert!(
        matches!(check_kernel(&k), Err(KernelError::AsyncUnsupported { what }) if what.contains("Shared")),
        "{:?}",
        check_kernel(&k)
    );
}

#[test]
fn the_source_read_is_logged_so_a_racing_write_to_it_is_a_data_race() {
    let body = |with_store: bool| {
        let mut b = Vec::new();
        if with_store {
            b.push(Stmt::Store {
                place: global(0, tid()),
                value: ScalarExpr::ConstInt(1, I32),
            });
        }
        b.extend([
            copy(shared(tid()), global(0, cu(0)), 1),
            Stmt::AsyncCommit,
            wait(0),
        ]);
        let mut k = kernel(b);
        k.params[0] = Param::buffer("src", I32, BufferRole::ReadWrite, Some(1));
        k
    };
    assert_eq!(run(&body(false)), Ok(vec![0; 4]), "control: no writer");
    assert_eq!(run(&body(true)), Err(KernelTrap::DataRace));
}
