// SPDX-License-Identifier: MIT OR Apache-2.0
//! **64-bit integer atomics** (survey row 50 of `docs/ir-union-evidence/conformance-idiom-gaps-2026-11-27.md`, read from the
//! HLSL Shader Model 6.6 `Interlocked*64` page, `docs/ir-union-evidence/conformance-idiom-gaps-verified-2026-11-27.md`). The
//! checker admits any integer element, but every other atomic test runs a 32-bit cell, so a result that dropped the high word
//! would pass them all. Twelve threads each apply one `AtomicRmw` with an operand above `2^32` to an `I64` or `U64` cell; the
//! expected cell is written by Rust's `u64`/`i64` operators, not by the oracle's combine. Each mutant (a narrowed operand, a
//! different op) changes the cell by VALUE.

use unpopped::exec::{LaunchShape, Scope};
use unpopped::ir::ScalarExpr;
use unpopped::kernel::{
    AtomicOp, Block, BufferRole, KernelFn, MemoryOrder, Param, Place, Stmt, check_kernel,
};
use unpopped::oracle::TypedBuffer;
use unpopped::oracle::kernel_eval::KernelArg;
use unpopped::oracle::kernel_launch::run_launch;
use unpopped_vocab::ElementKind;

const THREADS: u64 = 12;
const BIG: i64 = 0x0000_0003_8000_0005;
const START: u64 = 0x0000_0001_0000_00F0;

fn kernel(elem: ElementKind, op: AtomicOp, operand: i64) -> KernelFn {
    let cell = Place::Logical {
        buffer: 0,
        coords: vec![ScalarExpr::ConstInt(0, ElementKind::I32)],
    };
    let body = vec![Stmt::AtomicRmw {
        op,
        place: cell,
        value: ScalarExpr::ConstInt(operand, elem),
        old: None,
        order: MemoryOrder::Relaxed,
        scope: Scope::Device,
    }];
    let mut k = KernelFn::new(
        "wide",
        ElementKind::I32,
        vec![Param::buffer("c", elem, BufferRole::ReadWrite, Some(1))],
        vec![],
        Block(body),
    );
    k.launch = Some(LaunchShape::new([4, 1, 1], [3, 1, 1]));
    k
}

fn run(k: &KernelFn, elem: ElementKind, start: u64) -> u64 {
    assert_eq!(check_kernel(k), Ok(()), "well formed");
    let buf = TypedBuffer::new(elem, vec![1], vec![1], start.to_le_bytes().to_vec());
    let mut args = vec![KernelArg::Buffer(buf)];
    run_launch(k, &mut args).expect("runs");
    let KernelArg::Buffer(b) = &args[0] else {
        unreachable!()
    };
    u64::from_le_bytes(b.bytes[..8].try_into().unwrap())
}

/// The cell after `THREADS` applications of `op`, by Rust's operators.
fn reference(op: AtomicOp, start: u64, v: u64) -> u64 {
    (0..THREADS).fold(start, |c, _| match op {
        AtomicOp::Add => c.wrapping_add(v),
        AtomicOp::Max => c.max(v),
        AtomicOp::Min => c.min(v),
        AtomicOp::And => c & v,
        AtomicOp::Or => c | v,
        AtomicOp::Xor => c ^ v,
        AtomicOp::Exchange => v,
        _ => unreachable!(),
    })
}

const OPS: [AtomicOp; 7] = [
    AtomicOp::Add,
    AtomicOp::Max,
    AtomicOp::Min,
    AtomicOp::And,
    AtomicOp::Or,
    AtomicOp::Xor,
    AtomicOp::Exchange,
];

#[test]
fn every_integer_op_keeps_the_high_word_on_an_i64_cell() {
    for op in OPS {
        let got = run(&kernel(ElementKind::I64, op, BIG), ElementKind::I64, START);
        assert_eq!(got, reference(op, START, BIG as u64), "{op:?}");
    }
}

#[test]
fn every_integer_op_keeps_the_high_word_on_a_u64_cell() {
    for op in OPS {
        let got = run(&kernel(ElementKind::U64, op, BIG), ElementKind::U64, START);
        assert_eq!(got, reference(op, START, BIG as u64), "{op:?}");
    }
}

#[test]
fn a_signed_min_and_max_order_a_negative_operand_below_zero() {
    let neg = -0x0000_0002_0000_0001_i64;
    let start = 7_u64;
    for (op, want) in [
        (AtomicOp::Min, neg),
        (AtomicOp::Max, i64::try_from(start).unwrap()),
    ] {
        let got = run(&kernel(ElementKind::I64, op, neg), ElementKind::I64, start);
        assert_eq!(got as i64, want, "{op:?} on an I64 cell");
    }
}

#[test]
fn the_mutants_change_the_cell_by_value() {
    let good = run(
        &kernel(ElementKind::I64, AtomicOp::Add, BIG),
        ElementKind::I64,
        START,
    );
    // a narrowed operand (the low word only), the way a 32-bit implementation would read it
    let narrowed = run(
        &kernel(ElementKind::I64, AtomicOp::Add, BIG & 0xFFFF_FFFF),
        ElementKind::I64,
        START,
    );
    // the wrong op
    let wrong_op = run(
        &kernel(ElementKind::I64, AtomicOp::Or, BIG),
        ElementKind::I64,
        START,
    );
    assert_ne!(narrowed, good);
    assert_ne!(wrong_op, good);
    assert_eq!(good, reference(AtomicOp::Add, START, BIG as u64));
}
