// SPDX-License-Identifier: MIT OR Apache-2.0
//! ROADMAP U4, slice S3 (`docs/options-u4-sync-and-atomics.md`): `Stmt::AtomicCas` compares an element with
//! `expected` and stores `desired` if they are equal, as one indivisible step, and hands back the value it
//! found either way. The shape was reserved by 0.41.0 and refused; this lifts the refusal.
//!
//! Which thread wins a contended CAS depends on the order threads arrive, which the IR does not specify, so
//! these tests assert properties (exactly one winner, a CAS loop reaching the extremum), never the winner.

use unpopped::exec::{AllocDecl, ExecId, Extent, LaunchShape, MemorySpace, Scope};
use unpopped::ir::{BinaryOp, CastSpec, RoundMode, Saturation, ScalarExpr};
use unpopped::kernel::{
    AtomicOp, Block, BufferRole, KernelError, KernelFn, LocalDecl, MemoryOrder, Param, Place, Stmt,
    check_kernel,
};
use unpopped::oracle::TypedBuffer;
use unpopped::oracle::kernel_eval::{KernelArg, KernelTrap};
use unpopped::oracle::kernel_launch::run_launch;
use unpopped_vocab::ElementKind;

const I32: ElementKind = ElementKind::I32;
const U32: ElementKind = ElementKind::U32;
const N: usize = 12;

fn tid() -> ScalarExpr {
    ScalarExpr::Exec(ExecId::ThreadIdx(0))
}

fn c(v: i64) -> ScalarExpr {
    ScalarExpr::ConstInt(v, I32)
}

fn tid_i32() -> ScalarExpr {
    let spec = CastSpec {
        to: I32,
        round: RoundMode::NearestEven,
        sat: Saturation::Wrap,
    };
    ScalarExpr::Cast(Box::new(tid()), spec)
}

fn cell() -> Place {
    Place::Logical {
        buffer: 0,
        coords: vec![ScalarExpr::ConstInt(0, U32)],
    }
}

fn slot() -> Place {
    Place::Logical {
        buffer: 1,
        coords: vec![tid()],
    }
}

fn cas(place: Place, expected: ScalarExpr, desired: ScalarExpr, old: Option<u32>) -> Stmt {
    Stmt::AtomicCas {
        place,
        expected,
        desired,
        old,
        order: MemoryOrder::Relaxed,
        scope: Scope::Device,
    }
}

fn kernel(elem: ElementKind, locals: usize, body: Vec<Stmt>) -> KernelFn {
    let params = vec![
        Param::buffer("c", elem, BufferRole::ReadWrite, Some(1)),
        Param::buffer("o", I32, BufferRole::ReadWrite, Some(1)),
    ];
    let decls = (0..locals).map(|_| LocalDecl::scalar(I32)).collect();
    let mut k = KernelFn::new("k", I32, params, decls, Block(body));
    k.launch = Some(LaunchShape::new([N as u32, 1, 1], [1, 1, 1]));
    k
}

fn word(a: &KernelArg, i: usize) -> i32 {
    let KernelArg::Buffer(b) = a else {
        unreachable!()
    };
    i32::from_le_bytes(b.bytes[4 * i..4 * i + 4].try_into().unwrap())
}

/// `(final cell, per-thread slot)` after running on cell value `start`.
fn run(k: &KernelFn, start: i32) -> Result<(i32, Vec<i32>), KernelTrap> {
    let cellb = TypedBuffer::new(I32, vec![1], vec![1], start.to_le_bytes().to_vec());
    let outb = TypedBuffer::new(I32, vec![N as i64], vec![1], vec![0; 4 * N]);
    let mut args = vec![KernelArg::Buffer(cellb), KernelArg::Buffer(outb)];
    run_launch(k, &mut args)?;
    Ok((
        word(&args[0], 0),
        (0..N).map(|i| word(&args[1], i)).collect(),
    ))
}

fn record(old_to_slot: bool) -> Stmt {
    Stmt::Store {
        place: slot(),
        value: if old_to_slot {
            ScalarExpr::Local(0)
        } else {
            c(0)
        },
    }
}

#[test]
fn exactly_one_thread_wins_a_cas_from_zero() {
    let win = cas(
        cell(),
        c(0),
        ScalarExpr::Add(Box::new(tid_i32()), Box::new(c(1))),
        Some(0),
    );
    let k = kernel(I32, 1, vec![win, record(true)]);
    assert_eq!(check_kernel(&k), Ok(()));
    let (fin, olds) = run(&k, 0).expect("no race");
    let winners: Vec<usize> = (0..N).filter(|i| olds[*i] == 0).collect();
    assert_eq!(winners.len(), 1, "exactly one thread saw 0: {olds:?}");
    assert_eq!(
        fin,
        winners[0] as i32 + 1,
        "the cell holds the winner's value"
    );
    assert!(
        olds.iter()
            .enumerate()
            .all(|(i, o)| i == winners[0] || *o == fin)
    );
}

#[test]
fn a_failed_cas_leaves_the_cell_and_reports_what_it_found() {
    let k = kernel(I32, 1, vec![cas(cell(), c(9), c(1), Some(0)), record(true)]);
    let (fin, olds) = run(&k, 5).expect("no race");
    assert_eq!(fin, 5, "the compare failed, so nothing was stored");
    assert!(olds.iter().all(|o| *o == 5));
}

/// The classic CAS loop: read, compute the new value, CAS, retry if another thread got in between.
/// The read is an atomic add of 0, because a plain load of the element would race with the CASes.
#[test]
fn a_cas_loop_reaches_an_atomic_maximum() {
    let read = Stmt::AtomicRmw {
        op: AtomicOp::Add,
        place: cell(),
        value: c(0),
        old: Some(0),
        order: MemoryOrder::Relaxed,
        scope: Scope::Device,
    };
    let mine = ScalarExpr::Binary(
        BinaryOp::Min,
        Box::new(ScalarExpr::Binary(
            BinaryOp::Max,
            Box::new(tid_i32()),
            Box::new(c(3)),
        )),
        Box::new(c(100)),
    );
    let want = ScalarExpr::Binary(
        BinaryOp::Max,
        Box::new(ScalarExpr::Local(0)),
        Box::new(mine),
    );
    let attempt = vec![
        read,
        cas(cell(), ScalarExpr::Local(0), want, Some(1)),
        Stmt::Assign {
            local: 2,
            value: ScalarExpr::Binary(
                BinaryOp::CmpNe,
                Box::new(ScalarExpr::Local(1)),
                Box::new(ScalarExpr::Local(0)),
            ),
        },
    ];
    let body = vec![
        Stmt::Let {
            local: 2,
            init: c(1),
        },
        Stmt::While {
            cond: ScalarExpr::Local(2),
            body: Block(attempt),
        },
    ];
    let k = kernel(I32, 3, body);
    let (fin, _) = run(&k, 0).unwrap_or_else(|e| panic!("{e:?}"));
    assert_eq!(fin, N as i32 - 1, "the largest thread id, via a CAS loop");
}

#[test]
fn a_shared_cas_has_one_winner_too() {
    let sh = Place::Alloc {
        alloc: 0,
        index: ScalarExpr::ConstInt(0, U32),
    };
    let init = Stmt::If {
        cond: ScalarExpr::Binary(
            BinaryOp::CmpLt,
            Box::new(tid()),
            Box::new(ScalarExpr::ConstInt(1, U32)),
        ),
        then: Block(vec![Stmt::Store {
            place: sh.clone(),
            value: c(0),
        }]),
        els: Block(vec![]),
    };
    let body = vec![
        init,
        Stmt::barrier(Scope::Block),
        cas(sh, c(0), c(1), Some(0)),
        record(true),
    ];
    let k = kernel(I32, 1, body).with_allocs(vec![AllocDecl::new(
        "lock",
        MemorySpace::Shared,
        I32,
        Extent::Static(1),
    )]);
    assert_eq!(check_kernel(&k), Ok(()));
    let (_, olds) = run(&k, 0).expect("no race");
    assert_eq!(olds.iter().filter(|o| **o == 0).count(), 1, "{olds:?}");
}

#[test]
fn the_checker_admits_only_the_integer_relaxed_device_form() {
    let ok = |s| check_kernel(&kernel(I32, 1, vec![s]));
    assert_eq!(ok(cas(cell(), c(0), c(1), Some(0))), Ok(()));
    let float = kernel(
        ElementKind::F32,
        1,
        vec![cas(
            cell(),
            ScalarExpr::Const(0.0),
            ScalarExpr::Const(1.0),
            None,
        )],
    );
    assert!(matches!(
        check_kernel(&float),
        Err(KernelError::AtomicUnsupported { .. })
    ));
    let wide = ScalarExpr::ConstInt(1, ElementKind::I64);
    for bad in [
        cas(cell(), wide.clone(), c(1), None),
        cas(cell(), c(0), wide, None),
    ] {
        assert!(matches!(ok(bad), Err(KernelError::TypeMismatch { .. })));
    }
    for (order, scope) in [
        (MemoryOrder::Relaxed, Scope::Thread),
        (MemoryOrder::SeqCst, Scope::Cluster),
    ] {
        let s = Stmt::AtomicCas {
            place: cell(),
            expected: c(0),
            desired: c(1),
            old: None,
            order,
            scope,
        };
        assert!(
            matches!(ok(s), Err(KernelError::AtomicUnsupported { .. })),
            "{order:?} {scope:?}"
        );
    }
}

#[test]
fn a_cas_and_a_plain_access_to_one_element_race() {
    let plain = Stmt::Store {
        place: cell(),
        value: c(1),
    };
    let k = kernel(I32, 1, vec![plain, cas(cell(), c(0), c(2), None)]);
    assert_eq!(check_kernel(&k), Ok(()));
    assert_eq!(run(&k, 0), Err(KernelTrap::DataRace));
}
