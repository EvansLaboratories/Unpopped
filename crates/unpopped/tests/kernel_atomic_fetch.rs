// SPDX-License-Identifier: MIT OR Apache-2.0
//! ROADMAP U4, first slice (`docs/options-u4-sync-and-atomics.md`, option B): `Stmt::AtomicRmw` can return
//! the old value and has `And`/`Or`/`Xor`/`Exchange`; the reserved shapes (`order`, `scope`, `Fence`; `AtomicCas` was reserved
//! until `kernel_atomic_cas.rs`) are refused by the checker with a typed error instead of being run.
//!
//! The old value depends on the order the threads reach the element, which the IR does not specify, so
//! these tests assert **properties** (a permutation, a multiset) and never the interpreter's own order.

use unpopped::exec::{AllocDecl, ExecId, Extent, LaunchShape, MemorySpace, Scope};
use unpopped::ir::{BinaryOp, ScalarExpr};
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

fn fetch(op: AtomicOp, value: ScalarExpr, old: Option<u32>) -> Stmt {
    Stmt::AtomicRmw {
        op,
        place: cell(),
        value,
        old,
        order: MemoryOrder::Relaxed,
        scope: Scope::Device,
    }
}

/// `old = fetch_op(cell, value); out[tid] = old`.
fn kernel_with(elem: ElementKind, body: Vec<Stmt>) -> KernelFn {
    let params = vec![
        Param::buffer("c", elem, BufferRole::ReadWrite, Some(1)),
        Param::buffer("o", elem, BufferRole::ReadWrite, Some(1)),
    ];
    let mut k = KernelFn::new(
        "f",
        elem,
        params,
        vec![LocalDecl::scalar(elem)],
        Block(body),
    );
    k.launch = Some(LaunchShape::new([N as u32, 1, 1], [1, 1, 1]));
    k
}

fn record(op: AtomicOp, value: ScalarExpr) -> Vec<Stmt> {
    vec![
        fetch(op, value, Some(0)),
        Stmt::Store {
            place: slot(),
            value: ScalarExpr::Local(0),
        },
    ]
}

/// Run on `start`; return `(final cell, per-thread old values)`.
fn run(k: &KernelFn, start: i32) -> Result<(i32, Vec<i32>), KernelTrap> {
    let cellb = TypedBuffer::new(I32, vec![1], vec![1], start.to_le_bytes().to_vec());
    let outb = TypedBuffer::new(I32, vec![N as i64], vec![1], vec![0; 4 * N]);
    let mut args = vec![KernelArg::Buffer(cellb), KernelArg::Buffer(outb)];
    run_launch(k, &mut args)?;
    let word = |a: &KernelArg, i: usize| {
        let KernelArg::Buffer(b) = a else {
            unreachable!()
        };
        i32::from_le_bytes(b.bytes[4 * i..4 * i + 4].try_into().unwrap())
    };
    Ok((
        word(&args[0], 0),
        (0..N).map(|i| word(&args[1], i)).collect(),
    ))
}

#[test]
fn a_fetch_add_of_one_hands_every_thread_a_distinct_slot() {
    let k = kernel_with(I32, record(AtomicOp::Add, c(1)));
    assert_eq!(check_kernel(&k), Ok(()));
    let (fin, mut olds) = run(&k, 100).expect("no race");
    assert_eq!(fin, 100 + N as i32);
    olds.sort_unstable();
    let want: Vec<i32> = (100..100 + N as i32).collect();
    assert_eq!(
        olds, want,
        "the old values are a permutation of start..start+n"
    );
}

#[test]
fn a_fetch_max_returns_a_value_not_greater_than_the_result() {
    let k = kernel_with(
        I32,
        record(
            AtomicOp::Max,
            ScalarExpr::Cast(Box::new(tid()), cast_to_i32()),
        ),
    );
    assert_eq!(check_kernel(&k), Ok(()));
    let (fin, olds) = run(&k, -5).expect("no race");
    assert_eq!(fin, N as i32 - 1);
    assert!(olds.iter().all(|o| *o <= fin));
    assert!(
        olds.contains(&-5),
        "the first thread to arrive saw the start value"
    );
}

fn cast_to_i32() -> unpopped::ir::CastSpec {
    unpopped::ir::CastSpec {
        to: I32,
        round: unpopped::ir::RoundMode::NearestEven,
        sat: unpopped::ir::Saturation::Wrap,
    }
}

#[test]
fn bitwise_combines_reach_their_closed_form_whatever_the_order() {
    // Or of 1 << tid over 12 threads sets the low 12 bits; Xor of the same value twice cancels.
    let bit = ScalarExpr::Binary(
        BinaryOp::Shl,
        Box::new(c(1)),
        Box::new(ScalarExpr::Cast(Box::new(tid()), cast_to_i32())),
    );
    let or = kernel_with(I32, vec![fetch(AtomicOp::Or, bit.clone(), None)]);
    assert_eq!(run(&or, 0).expect("no race").0, (1 << N) - 1);
    let xor_twice = kernel_with(
        I32,
        vec![
            fetch(AtomicOp::Xor, bit.clone(), None),
            fetch(AtomicOp::Xor, bit, None),
        ],
    );
    assert_eq!(run(&xor_twice, 0x55).expect("no race").0, 0x55);
    let and = kernel_with(I32, vec![fetch(AtomicOp::And, c(0b1010), None)]);
    assert_eq!(run(&and, 0b0110).expect("no race").0, 0b0010);
}

#[test]
fn exchange_conserves_every_value_written() {
    // Thread t exchanges t + 1 into the cell: the values read back, plus the final cell, are exactly
    // the start value and every value written, once each.
    let v = ScalarExpr::Add(
        Box::new(ScalarExpr::Cast(Box::new(tid()), cast_to_i32())),
        Box::new(c(1)),
    );
    let k = kernel_with(I32, record(AtomicOp::Exchange, v));
    assert_eq!(check_kernel(&k), Ok(()));
    let (fin, mut seen) = run(&k, 1000).expect("no race");
    seen.push(fin);
    seen.sort_unstable();
    let mut want: Vec<i32> = (1..=N as i32).collect();
    want.push(1000);
    want.sort_unstable();
    assert_eq!(seen, want);
}

#[test]
fn a_float_exchange_is_admitted_and_float_and_or_xor_are_not() {
    let f = |op| {
        let body = vec![fetch(op, ScalarExpr::Const(1.0), None)];
        check_kernel(&kernel_with(ElementKind::F32, body))
    };
    assert_eq!(f(AtomicOp::Exchange), Ok(()));
    for op in [AtomicOp::And, AtomicOp::Or, AtomicOp::Xor, AtomicOp::Max] {
        assert!(
            matches!(f(op), Err(KernelError::AtomicUnsupported { .. })),
            "{op:?}"
        );
    }
}

#[test]
fn the_old_local_must_have_the_element_type_and_be_declared_once() {
    let mut wrong = kernel_with(I32, vec![fetch(AtomicOp::Add, c(1), Some(0))]);
    wrong.locals = vec![LocalDecl::scalar(ElementKind::I64)];
    assert!(matches!(
        check_kernel(&wrong),
        Err(KernelError::TypeMismatch { .. })
    ));
    let twice = kernel_with(
        I32,
        vec![
            fetch(AtomicOp::Add, c(1), Some(0)),
            fetch(AtomicOp::Add, c(1), Some(0)),
        ],
    );
    assert!(matches!(
        check_kernel(&twice),
        Err(KernelError::Redeclared { local: 0 })
    ));
}

#[test]
fn the_reserved_shapes_are_refused_with_a_typed_error() {
    let with = |order, scope| Stmt::AtomicRmw {
        op: AtomicOp::Add,
        place: cell(),
        value: c(1),
        old: None,
        order,
        scope,
    };
    for st in [
        with(MemoryOrder::Relaxed, Scope::Thread),
        with(MemoryOrder::Acquire, Scope::Cluster),
    ] {
        let r = check_kernel(&kernel_with(I32, vec![st]));
        assert!(
            matches!(r, Err(KernelError::AtomicUnsupported { .. })),
            "{r:?}"
        );
    }
    let fence = Stmt::Fence {
        scope: Scope::Device,
        order: MemoryOrder::SeqCst,
    };
    assert!(matches!(
        check_kernel(&kernel_with(I32, vec![fence])),
        Err(KernelError::SyncUnsupported { .. })
    ));
}

#[test]
fn a_barrier_under_a_branch_on_the_old_value_is_divergent() {
    // `old` differs between threads whatever the operands, so the uniformity analysis must mark it.
    let guarded = Stmt::If {
        cond: ScalarExpr::Binary(
            BinaryOp::CmpEq,
            Box::new(ScalarExpr::Local(0)),
            Box::new(c(0)),
        ),
        then: Block(vec![Stmt::barrier(Scope::Block)]),
        els: Block(vec![]),
    };
    let k = kernel_with(I32, vec![fetch(AtomicOp::Add, c(1), Some(0)), guarded]);
    assert_eq!(check_kernel(&k), Err(KernelError::BarrierDivergent));
}

#[test]
fn a_shared_fetch_add_hands_out_distinct_slots_too() {
    // Thread 0 zeroes a Shared cell, a barrier orders that before the atomics, every thread
    // fetch-adds 1 and writes the old value to its own slot of `o`.
    let cell = Place::Alloc {
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
            place: cell.clone(),
            value: c(0),
        }]),
        els: Block(vec![]),
    };
    let body = vec![
        init,
        Stmt::barrier(Scope::Block),
        Stmt::AtomicRmw {
            op: AtomicOp::Add,
            place: cell,
            value: c(1),
            old: Some(0),
            order: MemoryOrder::Relaxed,
            scope: Scope::Device,
        },
        Stmt::Store {
            place: slot(),
            value: ScalarExpr::Local(0),
        },
    ];
    let k = kernel_with(I32, body).with_allocs(vec![AllocDecl::new(
        "cell",
        MemorySpace::Shared,
        I32,
        Extent::Static(1),
    )]);
    assert_eq!(check_kernel(&k), Ok(()));
    let (_, mut olds) = run(&k, 0).expect("no race");
    olds.sort_unstable();
    assert_eq!(olds, (0..N as i32).collect::<Vec<_>>());
}
