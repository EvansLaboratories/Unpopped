// SPDX-License-Identifier: MIT OR Apache-2.0
//! Helpers shared by `kernel_wide_int.rs`, `kernel_wide_int_256.rs` and `kernel_wide_int_casts.rs`: a 256-bit pattern as
//! `(hi, lo)`, the one-block kernels that apply an operator or a cast to a grid of operands, and the operand grid.

use unpopped::exec::{ExecId, LaunchShape};
use unpopped::ir::{BinaryOp, CastSpec, RoundMode, Saturation, ScalarExpr};
use unpopped::kernel::{Block, BufferRole, KernelFn, LocalDecl, Param, Place, Stmt, check_kernel};
use unpopped::oracle::TypedBuffer;
use unpopped::oracle::kernel_eval::KernelArg;
use unpopped::oracle::kernel_launch::run_launch;
use unpopped_vocab::ElementKind::{self, I32, I256, U256};

/// A 256-bit pattern as `(hi, lo)`.
pub(crate) type P = (u128, u128);

pub(crate) fn cs(to: ElementKind, sat: Saturation) -> CastSpec {
    CastSpec {
        to,
        round: RoundMode::NearestEven,
        sat,
    }
}

pub(crate) fn at_tid(buffer: u32) -> Place {
    Place::Logical {
        buffer,
        coords: vec![ScalarExpr::Exec(ExecId::ThreadIdx(0))],
    }
}

pub(crate) fn bytes_of(t: ElementKind, p: P) -> Vec<u8> {
    let mut b = p.1.to_le_bytes().to_vec();
    if matches!(t, I256 | U256) {
        b.extend(p.0.to_le_bytes());
    }
    b
}

pub(crate) fn width(t: ElementKind) -> usize {
    if matches!(t, I256 | U256) { 32 } else { 16 }
}

pub(crate) fn pattern(t: ElementKind, c: &[u8]) -> P {
    let lo = u128::from_le_bytes(c[..16].try_into().unwrap());
    if width(t) == 32 {
        (u128::from_le_bytes(c[16..].try_into().unwrap()), lo)
    } else {
        (0, lo)
    }
}

/// `out[tid] = a[tid] <op> b[tid]` at `t`, the result stored at `out_t` (the compute type I32 for a comparison).
pub(crate) fn binary_kernel(op: BinaryOp, t: ElementKind, out_t: ElementKind, n: u32) -> KernelFn {
    let local = |i| Box::new(ScalarExpr::Local(i));
    let mut k = KernelFn::new(
        "wide",
        I32,
        vec![
            Param::buffer("a", t, BufferRole::Read, Some(1)),
            Param::buffer("b", t, BufferRole::Read, Some(1)),
            Param::buffer("out", out_t, BufferRole::Write, Some(1)),
        ],
        vec![LocalDecl::scalar(t), LocalDecl::scalar(t)],
        Block(vec![
            Stmt::Load {
                local: 0,
                place: at_tid(0),
            },
            Stmt::Load {
                local: 1,
                place: at_tid(1),
            },
            Stmt::Store {
                place: at_tid(2),
                value: ScalarExpr::Binary(op, local(0), local(1)),
            },
        ]),
    );
    k.launch = Some(LaunchShape::new([n, 1, 1], [1, 1, 1]));
    k
}

pub(crate) fn arith_kernel(
    make: fn(Box<ScalarExpr>, Box<ScalarExpr>) -> ScalarExpr,
    t: ElementKind,
    n: u32,
) -> KernelFn {
    let mut k = binary_kernel(BinaryOp::Max, t, t, n);
    let Stmt::Store { value, .. } = &mut k.body.0[2] else {
        unreachable!()
    };
    *value = make(
        Box::new(ScalarExpr::Local(0)),
        Box::new(ScalarExpr::Local(1)),
    );
    k
}

pub(crate) fn run(k: &KernelFn, t: ElementKind, out_t: ElementKind, a: &[P], b: &[P]) -> Vec<P> {
    assert_eq!(check_kernel(k), Ok(()), "well formed at {t:?}");
    let n = a.len() as i64;
    let buf =
        |ty: ElementKind, v: Vec<u8>| KernelArg::Buffer(TypedBuffer::new(ty, vec![n], vec![1], v));
    let mut args = vec![
        buf(t, a.iter().flat_map(|p| bytes_of(t, *p)).collect()),
        buf(t, b.iter().flat_map(|p| bytes_of(t, *p)).collect()),
        buf(out_t, vec![0u8; width(out_t).max(4) * a.len()]),
    ];
    run_launch(k, &mut args).unwrap_or_else(|e| panic!("{t:?}: {e:?}"));
    let KernelArg::Buffer(out) = &args[2] else {
        unreachable!()
    };
    out.bytes
        .chunks(if matches!(out_t, I32) {
            4
        } else {
            width(out_t)
        })
        .map(|c| {
            if matches!(out_t, I32) {
                (0, u128::from(u32::from_le_bytes(c.try_into().unwrap())))
            } else {
                pattern(out_t, c)
            }
        })
        .collect()
}

/// The operand grid: every pair of the interesting patterns.
pub(crate) fn grid(t: ElementKind) -> (Vec<P>, Vec<P>) {
    let wide = width(t) == 32;
    let top = if wide { u128::MAX } else { 0 };
    let vals: Vec<P> = vec![
        (0, 0),
        (0, 1),
        (top, u128::MAX),
        (
            if wide { 1 << 127 } else { 0 },
            if wide { 0 } else { 1 << 127 },
        ),
        (
            if wide { 1 } else { 0 },
            if wide {
                0
            } else {
                0x0123_4567_89ab_cdef_fedc_ba98_7654_3210
            },
        ),
        (0, 0x69),
        (0, u128::MAX),
        (
            if wide { 0x1234 } else { 0 },
            0x8000_0000_0000_0000_0000_0000_0000_0001,
        ),
    ];
    let (mut a, mut b) = (Vec::new(), Vec::new());
    for x in &vals {
        for y in &vals {
            a.push(*x);
            b.push(*y);
        }
    }
    (a, b)
}
