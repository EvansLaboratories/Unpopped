// SPDX-License-Identifier: MIT OR Apache-2.0
//! `AtomicOp::Inc` / `AtomicOp::Dec` (CUDA `atomicInc` / `atomicDec`, PTX `atom.inc.u32` / `atom.dec.u32`): the wrapping
//! increment `old >= v ? 0 : old + 1` and decrement `(old == 0 || old > v) ? v : old - 1`, on `U32` only.
//!
//! Every thread applies the same function with the same operand, so the final cell after `N` threads is `f^N(start)`
//! whatever the order; the expected value here is an independent loop written from the CUDA definition.

use unpopped::exec::LaunchShape;
use unpopped::ir::ScalarExpr;
use unpopped::kernel::{
    AtomicOp, Block, BufferRole, KernelError, KernelFn, Param, Place, Stmt, check_kernel,
};
use unpopped::oracle::TypedBuffer;
use unpopped::oracle::kernel_eval::KernelArg;
use unpopped::oracle::kernel_launch::run_launch;
use unpopped_vocab::ElementKind;

const U32: ElementKind = ElementKind::U32;
const I32: ElementKind = ElementKind::I32;
const N: u32 = 12;

fn kernel(op: AtomicOp, elem: ElementKind, limit: i64) -> KernelFn {
    let cell = Place::Logical {
        buffer: 0,
        coords: vec![ScalarExpr::ConstInt(0, U32)],
    };
    let mut k = KernelFn::new(
        "incdec",
        elem,
        vec![Param::buffer("c", elem, BufferRole::ReadWrite, Some(1))],
        vec![],
        Block(vec![Stmt::atomic_rmw(
            op,
            cell,
            ScalarExpr::ConstInt(limit, elem),
        )]),
    );
    k.launch = Some(LaunchShape::new([1, 1, 1], [N, 1, 1]));
    k
}

fn run(op: AtomicOp, limit: u32, start: u32) -> u32 {
    let k = kernel(op, U32, i64::from(limit));
    assert_eq!(check_kernel(&k), Ok(()));
    let buf = TypedBuffer::new(U32, vec![1], vec![1], start.to_le_bytes().to_vec());
    let mut args = vec![KernelArg::Buffer(buf)];
    run_launch(&k, &mut args).expect("runs");
    let KernelArg::Buffer(b) = &args[0] else {
        unreachable!()
    };
    u32::from_le_bytes(b.bytes[..4].try_into().unwrap())
}

fn inc(old: u32, v: u32) -> u32 {
    if old >= v { 0 } else { old + 1 }
}

fn dec(old: u32, v: u32) -> u32 {
    if old == 0 || old > v { v } else { old - 1 }
}

#[test]
fn inc_and_dec_wrap_as_cuda_defines() {
    for (limit, start) in [(7, 3), (4, 9), (0, 0), (5, 5), (3, 0)] {
        let (mut i, mut d) = (start, start);
        for _ in 0..N {
            i = inc(i, limit);
            d = dec(d, limit);
        }
        assert_eq!(run(AtomicOp::Inc, limit, start), i, "inc {limit} {start}");
        assert_eq!(run(AtomicOp::Dec, limit, start), d, "dec {limit} {start}");
    }
}

#[test]
fn inc_and_dec_are_refused_off_u32_with_the_u32_control() {
    for op in [AtomicOp::Inc, AtomicOp::Dec] {
        assert!(matches!(
            check_kernel(&kernel(op, I32, 7)),
            Err(KernelError::AtomicUnsupported { .. })
        ));
        assert_eq!(check_kernel(&kernel(op, U32, 7)), Ok(()));
    }
}
