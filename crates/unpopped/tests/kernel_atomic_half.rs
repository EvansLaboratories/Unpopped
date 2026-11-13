// SPDX-License-Identifier: MIT OR Apache-2.0
//! **`Stmt::AtomicRmw` `Add` and `Exchange` on `F16` and `Bf16`** (ROADMAP U12b): the checker admits them (and no
//! `Max`/`Min`), and the launch interpreter rounds each add once to the element, so `2048 + 1` stays `2048` in `F16`
//! (a tie at spacing 2 goes to the even mantissa) and twelve such adds leave 2048.

use unpopped::exec::{LaunchShape, Scope};
use unpopped::ir::ScalarExpr;
use unpopped::kernel::{
    AtomicOp, Block, BufferRole, KernelFn, LocalDecl, MemoryOrder, Param, Place, Stmt, check_kernel,
};
use unpopped::oracle::TypedBuffer;
use unpopped::oracle::kernel_eval::KernelArg;
use unpopped::oracle::kernel_launch::run_launch;
use unpopped_vocab::ElementKind;

const I32: ElementKind = ElementKind::I32;

fn cell(buffer: u32) -> Place {
    Place::Logical {
        buffer,
        coords: vec![ScalarExpr::ConstInt(0, I32)],
    }
}

/// Every thread of 4 x 3 loads `inp[0]` and applies `op` of it to `acc[0]`.
fn kernel(t: ElementKind, op: AtomicOp) -> KernelFn {
    let body = vec![
        Stmt::Load {
            local: 0,
            place: cell(0),
        },
        Stmt::AtomicRmw {
            op,
            place: cell(1),
            value: ScalarExpr::Local(0),
            old: None,
            order: MemoryOrder::Relaxed,
            scope: Scope::Device,
        },
    ];
    let params = vec![
        Param::buffer("inp", t, BufferRole::Read, Some(1)),
        Param::buffer("acc", t, BufferRole::ReadWrite, Some(1)),
    ];
    KernelFn::new(
        "half_atomic",
        ElementKind::F32,
        params,
        vec![LocalDecl::scalar(t)],
        Block(body),
    )
    .with_launch(LaunchShape::new([4, 1, 1], [3, 1, 1]))
}

fn run(t: ElementKind, op: AtomicOp, inp: u16, acc: u16) -> u16 {
    let k = kernel(t, op);
    assert_eq!(check_kernel(&k), Ok(()), "{t:?} {op:?}");
    let one = |bits: u16| {
        KernelArg::Buffer(TypedBuffer::new(
            t,
            vec![1],
            vec![1],
            bits.to_le_bytes().to_vec(),
        ))
    };
    let mut args = vec![one(inp), one(acc)];
    run_launch(&k, &mut args).unwrap();
    let KernelArg::Buffer(b) = &args[1] else {
        unreachable!()
    };
    u16::from_le_bytes(b.bytes[..2].try_into().unwrap())
}

#[test]
fn twelve_adds_of_one_to_a_tie_each_round_to_even_and_leave_it() {
    // F16: 2048 is 0x6800, 1.0 is 0x3C00; Bf16: 256 is 0x4380, 1.0 is 0x3F80
    assert_eq!(run(ElementKind::F16, AtomicOp::Add, 0x3C00, 0x6800), 0x6800);
    assert_eq!(
        run(ElementKind::Bf16, AtomicOp::Add, 0x3F80, 0x4380),
        0x4380
    );
}

#[test]
fn twelve_exact_adds_sum_and_an_exchange_takes_the_operand() {
    // 12 * 0.5 = 6.0 (F16 0x4600, Bf16 0x40C0); the exchange leaves the operand's bits, a NaN payload included
    assert_eq!(run(ElementKind::F16, AtomicOp::Add, 0x3800, 0), 0x4600);
    assert_eq!(run(ElementKind::Bf16, AtomicOp::Add, 0x3F00, 0), 0x40C0);
    assert_eq!(
        run(ElementKind::F16, AtomicOp::Exchange, 0x7E55, 0x3C00),
        0x7E55
    );
    assert_eq!(
        run(ElementKind::Bf16, AtomicOp::Exchange, 0x7FC5, 0x3F80),
        0x7FC5
    );
}

#[test]
fn max_and_min_on_a_16_bit_float_are_still_refused() {
    for t in [ElementKind::F16, ElementKind::Bf16] {
        for op in [AtomicOp::Max, AtomicOp::Min, AtomicOp::And] {
            assert!(check_kernel(&kernel(t, op)).is_err(), "{t:?} {op:?}");
        }
    }
}
