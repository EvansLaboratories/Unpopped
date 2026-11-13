// SPDX-License-Identifier: MIT OR Apache-2.0
//! **A launched `AtomicRmw` `Add`/`Exchange` on `F16` and `Bf16` in C agrees with the interpreter** (ROADMAP U26): the
//! combine decodes both patterns to `float`, adds, and rounds once to the element on every add (the tie at spacing 2
//! goes to the even mantissa, so twelve `2048 + 1` leave 2048), and an exchange moves the operand's bits, a NaN payload
//! included.

use unpopped::backend::Backend;
use unpopped::exec::{LaunchShape, Scope};
use unpopped::ir::ScalarExpr;
use unpopped::kernel::{
    AtomicOp, Block, BufferRole, KernelFn, LocalDecl, MemoryOrder, Param, Place, Stmt,
};
use unpopped::oracle::TypedBuffer;
use unpopped::oracle::kernel_eval::KernelArg;
use unpopped_cpu_c::CpuC;
use unpopped_vocab::ElementKind;

#[path = "support/cc.rs"]
mod cc;
#[allow(dead_code)]
#[path = "support/kernel_run.rs"]
mod kernel_run;
#[path = "support/launch_run.rs"]
mod launch_run;
use launch_run::agree_launch;

const I32: ElementKind = ElementKind::I32;

fn cell(buffer: u32) -> Place {
    Place::Logical {
        buffer,
        coords: vec![ScalarExpr::ConstInt(0, I32)],
    }
}

/// Every thread of 4 x 3 loads `inp[0]` and applies `op` of it to `acc[0]`.
fn kernel(tag: &str, t: ElementKind, op: AtomicOp) -> KernelFn {
    KernelFn::new(
        tag,
        ElementKind::F32,
        vec![
            Param::buffer("inp", t, BufferRole::Read, Some(1)),
            Param::buffer("acc", t, BufferRole::ReadWrite, Some(1)),
        ],
        vec![LocalDecl::scalar(t)],
        Block(vec![
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
        ]),
    )
    .with_launch(LaunchShape::new([4, 1, 1], [3, 1, 1]))
}

fn one(t: ElementKind, bits: u16) -> KernelArg {
    KernelArg::Buffer(TypedBuffer::new(
        t,
        vec![1],
        vec![1],
        bits.to_le_bytes().to_vec(),
    ))
}

#[test]
fn narrow_atomic_add_and_exchange_agree() {
    // (type, op, operand, accumulator): F16 2048 = 0x6800, 1.0 = 0x3C00, 0.5 = 0x3800; Bf16 256 = 0x4380, 1.0 = 0x3F80
    let cases = [
        (ElementKind::F16, AtomicOp::Add, 0x3C00u16, 0x6800u16),
        (ElementKind::Bf16, AtomicOp::Add, 0x3F80, 0x4380),
        (ElementKind::F16, AtomicOp::Add, 0x3800, 0),
        (ElementKind::Bf16, AtomicOp::Add, 0x3F00, 0),
        (ElementKind::F16, AtomicOp::Exchange, 0x7E55, 0x3C00),
        (ElementKind::Bf16, AtomicOp::Exchange, 0x7FC5, 0x3F80),
    ];
    for (i, (t, op, inp, acc)) in cases.into_iter().enumerate() {
        let tag = format!("atomic_{i}");
        let k = kernel(&tag, t, op);
        let src = CpuC.lower_kernel(&k).expect("the combine is spelled").source;
        assert!(!src.is_empty());
        agree_launch(&tag, &k, vec![one(t, inp), one(t, acc)]);
    }
}
