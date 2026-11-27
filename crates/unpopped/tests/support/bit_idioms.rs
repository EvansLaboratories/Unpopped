// SPDX-License-Identifier: MIT OR Apache-2.0
//! Two integer idioms from the public docs of other kernel languages, built from leaves already on main (no new statement or
//! operator): a carry-less (GF(2)) multiply (Triton's `tl.clmul`) and a four-lane byte dot product with accumulate (CUDA
//! `__dp4a`, TileLang `T.dp4a`). Both are written from their published definitions, not from any source code, and their meaning
//! here is the arithmetic only; whether the target hardware has a single instruction for either is not modelled. Survey rows 20
//! and 28 of `docs/ir-union-evidence/conformance-idiom-gaps-2026-11-27.md`. One thread per element, `ThreadIdx(0)`.

use unpopped::exec::{ExecId, LaunchShape};
use unpopped::ir::{BinaryOp, CastSpec, RoundMode, Saturation, ScalarExpr};
use unpopped::kernel::{Block, BufferRole, KernelFn, LocalDecl, Param, Place, Stmt, Unroll};
use unpopped_vocab::ElementKind;

pub(crate) const F32: ElementKind = ElementKind::F32;
pub(crate) const U32: ElementKind = ElementKind::U32;
pub(crate) const U64: ElementKind = ElementKind::U64;
pub(crate) const I32: ElementKind = ElementKind::I32;
pub(crate) const N: usize = 8;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum Mutant {
    None,
    /// `clmul` adds the partial products (ordinary multiplication, carries) instead of XOR-ing them.
    Carries,
    /// `clmul` shifts `a` by `i + 1` instead of `i`.
    ShiftedByOne,
    /// `dp4a` treats every byte of both operands as unsigned.
    UnsignedBytes,
    /// `dp4a` drops the accumulator.
    NoAccumulator,
}

fn l(i: u32) -> ScalarExpr {
    ScalarExpr::Local(i)
}
fn ci(v: i64, t: ElementKind) -> ScalarExpr {
    ScalarExpr::ConstInt(v, t)
}
fn bin(op: BinaryOp, a: ScalarExpr, b: ScalarExpr) -> ScalarExpr {
    ScalarExpr::Binary(op, Box::new(a), Box::new(b))
}
fn add(a: ScalarExpr, b: ScalarExpr) -> ScalarExpr {
    ScalarExpr::Add(Box::new(a), Box::new(b))
}
fn sub(a: ScalarExpr, b: ScalarExpr) -> ScalarExpr {
    ScalarExpr::Sub(Box::new(a), Box::new(b))
}
fn mul(a: ScalarExpr, b: ScalarExpr) -> ScalarExpr {
    ScalarExpr::Mul(Box::new(a), Box::new(b))
}
fn cast(x: ScalarExpr, to: ElementKind) -> ScalarExpr {
    ScalarExpr::Cast(
        Box::new(x),
        CastSpec {
            to,
            round: RoundMode::NearestEven,
            sat: Saturation::Wrap,
        },
    )
}
fn at(buffer: u32) -> Place {
    Place::Logical {
        buffer,
        coords: vec![ScalarExpr::Exec(ExecId::ThreadIdx(0))],
    }
}
fn counted(local: u32, end: i64, body: Vec<Stmt>) -> Stmt {
    Stmt::For {
        unroll: Unroll::None,
        local,
        start: ci(0, U32),
        end: ci(end, U32),
        step: ci(1, U32),
        body: Block(body),
    }
}
fn params(inputs: &[&str], outputs: &[&str]) -> Vec<Param> {
    let ins = inputs
        .iter()
        .map(|n| Param::buffer(*n, U32, BufferRole::Read, Some(1)));
    let outs = outputs
        .iter()
        .map(|n| Param::buffer(*n, U32, BufferRole::Write, Some(1)));
    ins.chain(outs).collect()
}
fn launch(k: KernelFn) -> KernelFn {
    k.with_launch(LaunchShape::new([N as u32, 1, 1], [1, 1, 1]))
}

/// The 64-bit carry-less product of two `u32`s, as two `u32` halves. Buffers: 0 `a`, 1 `b`, 2 `lo`, 3 `hi`. Locals: 0 `a`,
/// 1 `b`, 2 `a` widened (`U64`), 3 the accumulator (`U64`), 4 the bit index `i`.
pub(crate) fn clmul(mutant: Mutant) -> KernelFn {
    let amount = if mutant == Mutant::ShiftedByOne {
        add(l(4), ci(1, U32))
    } else {
        l(4)
    };
    let part = bin(BinaryOp::Shl, l(2), cast(amount, U64));
    let folded = if mutant == Mutant::Carries {
        add(l(3), part)
    } else {
        bin(BinaryOp::BitXor, l(3), part)
    };
    let bit = bin(BinaryOp::BitAnd, bin(BinaryOp::Shr, l(1), l(4)), ci(1, U32));
    let step = Stmt::If {
        cond: bit,
        then: Block(vec![Stmt::Assign {
            local: 3,
            value: folded,
        }]),
        els: Block(vec![]),
    };
    let load = |local, buffer| Stmt::Load {
        local,
        place: at(buffer),
    };
    let body = vec![
        load(0, 0),
        load(1, 1),
        Stmt::Let {
            local: 2,
            init: cast(l(0), U64),
        },
        Stmt::Let {
            local: 3,
            init: ci(0, U64),
        },
        counted(4, 32, vec![step]),
        Stmt::Store {
            place: at(2),
            value: cast(l(3), U32),
        },
        Stmt::Store {
            place: at(3),
            value: cast(bin(BinaryOp::Shr, l(3), ci(32, U64)), U32),
        },
    ];
    let locals = [U32, U32, U64, U64, U32].map(LocalDecl::scalar).to_vec();
    launch(KernelFn::new(
        "clmul",
        F32,
        params(&["a", "b"], &["lo", "hi"]),
        locals,
        Block(body),
    ))
}

/// A byte of `x` (local `x`), sign-extended to `I32` when `signed`: `byte - 256 * (byte >> 7)`.
fn byte(x: u32, k: i64, signed: bool) -> ScalarExpr {
    let raw = cast(
        bin(
            BinaryOp::BitAnd,
            bin(BinaryOp::Shr, l(x), ci(8 * k, U32)),
            ci(255, U32),
        ),
        I32,
    );
    if signed {
        sub(
            raw.clone(),
            mul(ci(256, I32), bin(BinaryOp::Shr, raw, ci(7, I32))),
        )
    } else {
        raw
    }
}

/// `c + sum over four byte lanes of a_k * b_k`, `a` signed and `b` signed (`__dp4a` of two `s8` operands). Buffers: 0 `a`,
/// 1 `b`, 2 `c` (all `u32` bit patterns), 3 `out` (the `i32` result as a `u32` bit pattern). Locals: 0 `a`, 1 `b`, 2 `c`, 3 the sum (`I32`).
pub(crate) fn dp4a(mutant: Mutant) -> KernelFn {
    let signed = mutant != Mutant::UnsignedBytes;
    let mut sum = if mutant == Mutant::NoAccumulator {
        ci(0, I32)
    } else {
        cast(l(2), I32)
    };
    for k in 0..4 {
        sum = add(sum, mul(byte(0, k, signed), byte(1, k, signed)));
    }
    let load = |local, buffer| Stmt::Load {
        local,
        place: at(buffer),
    };
    let body = vec![
        load(0, 0),
        load(1, 1),
        load(2, 2),
        Stmt::Let {
            local: 3,
            init: sum,
        },
        Stmt::Store {
            place: at(3),
            value: cast(l(3), U32),
        },
    ];
    let locals = [U32, U32, U32, I32].map(LocalDecl::scalar).to_vec();
    launch(KernelFn::new(
        "dp4a",
        F32,
        params(&["a", "b", "c"], &["out"]),
        locals,
        Block(body),
    ))
}
