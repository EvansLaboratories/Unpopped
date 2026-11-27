// SPDX-License-Identifier: MIT OR Apache-2.0
//! Philox4x32-10, a counter-based random generator (Salmon, Moraes, Dror and Shaw, "Parallel random numbers: as easy as 1, 2,
//! 3", SC11), at the kernel layer from leaves already on main: `U64` multiply, shifts and XOR. It is what Triton's `tl.rand`
//! family and ThunderKittens' dropout use (survey rows 22 and 33 of `docs/ir-union-evidence/conformance-idiom-gaps-2026-11-27.md`).
//! One thread per counter block `(c0, c1, c2, c3)` with key `(k0, k1)`: ten rounds of two 32 x 32 to 64-bit multiplies by
//! `0xD2511F53` and `0xCD9E8D57`, the key bumped by `0x9E3779B9` and `0xBB67AE85` after each round. The constants and the round
//! shape are written from memory of the published algorithm and checked only by the known answers in the test file:
//! UNVERIFIED against the paper or the Random123 source, which were not read.

use unpopped::exec::{ExecId, LaunchShape};
use unpopped::ir::{BinaryOp, CastSpec, RoundMode, Saturation, ScalarExpr};
use unpopped::kernel::{Block, BufferRole, KernelFn, LocalDecl, Param, Place, Stmt, Unroll};
use unpopped_vocab::ElementKind;

pub(crate) const F32: ElementKind = ElementKind::F32;
pub(crate) const U32: ElementKind = ElementKind::U32;
pub(crate) const U64: ElementKind = ElementKind::U64;
pub(crate) const N: usize = 4;
pub(crate) const ROUNDS: i64 = 10;
pub(crate) const M0: u32 = 0xD251_1F53;
pub(crate) const M1: u32 = 0xCD9E_8D57;
pub(crate) const W0: u32 = 0x9E37_79B9;
pub(crate) const W1: u32 = 0xBB67_AE85;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum Mutant {
    None,
    /// Nine rounds.
    NineRounds,
    /// The two multipliers are swapped.
    SwappedMultipliers,
    /// The key is never bumped.
    NoKeyBump,
}

fn l(i: u32) -> ScalarExpr {
    ScalarExpr::Local(i)
}
fn cu(v: u32) -> ScalarExpr {
    ScalarExpr::ConstInt(i64::from(v), U32)
}
fn bin(op: BinaryOp, a: ScalarExpr, b: ScalarExpr) -> ScalarExpr {
    ScalarExpr::Binary(op, Box::new(a), Box::new(b))
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
fn xor(a: ScalarExpr, b: ScalarExpr) -> ScalarExpr {
    bin(BinaryOp::BitXor, a, b)
}
/// The high and low 32 bits of the `U64` local `p`.
fn hi(p: u32) -> ScalarExpr {
    cast(bin(BinaryOp::Shr, l(p), ScalarExpr::ConstInt(32, U64)), U32)
}
fn lo(p: u32) -> ScalarExpr {
    cast(l(p), U32)
}
/// Element `j` of block `i` of buffer `b`, `width` words per block.
fn word(buffer: u32, width: i64, j: i64) -> Place {
    let block = ScalarExpr::Exec(ExecId::ThreadIdx(0));
    Place::Logical {
        buffer,
        coords: vec![ScalarExpr::Add(
            Box::new(ScalarExpr::Mul(
                Box::new(block),
                Box::new(ScalarExpr::ConstInt(width, U32)),
            )),
            Box::new(ScalarExpr::ConstInt(j, U32)),
        )],
    }
}

/// One round on locals 0..=3 (the counter) and 4, 5 (the key): products in locals 7 and 8 (`U64`), results in 9..=12.
fn round(mutant: Mutant) -> Vec<Stmt> {
    let (m0, m1) = if mutant == Mutant::SwappedMultipliers {
        (M1, M0)
    } else {
        (M0, M1)
    };
    let product = |m: u32, c: u32| {
        ScalarExpr::Mul(
            Box::new(ScalarExpr::ConstInt(i64::from(m), U64)),
            Box::new(cast(l(c), U64)),
        )
    };
    let let_ = |local, init| Stmt::Let { local, init };
    let set = |local, value| Stmt::Assign { local, value };
    let mut out = vec![
        let_(7, product(m0, 0)),
        let_(8, product(m1, 2)),
        let_(9, xor(xor(hi(8), l(1)), l(4))),
        let_(10, lo(8)),
        let_(11, xor(xor(hi(7), l(3)), l(5))),
        let_(12, lo(7)),
        set(0, l(9)),
        set(1, l(10)),
        set(2, l(11)),
        set(3, l(12)),
    ];
    if mutant != Mutant::NoKeyBump {
        out.push(set(4, ScalarExpr::Add(Box::new(l(4)), Box::new(cu(W0)))));
        out.push(set(5, ScalarExpr::Add(Box::new(l(5)), Box::new(cu(W1)))));
    }
    out
}

/// Buffers: 0 `ctr` `[N * 4]`, 1 `key` `[N * 2]` (read), 2 `out` `[N * 4]` (write); all `u32`. Locals: 0..=3 the counter, 4, 5
/// the key, 6 the round index, 7 and 8 the products (`U64`), 9..=12 the new counter.
pub(crate) fn philox(mutant: Mutant) -> KernelFn {
    let rounds = if mutant == Mutant::NineRounds {
        ROUNDS - 1
    } else {
        ROUNDS
    };
    let load = |local: u32, buffer, width, j| Stmt::Load {
        local,
        place: word(buffer, width, j),
    };
    let mut body: Vec<Stmt> = (0..4).map(|j| load(j as u32, 0, 4, j)).collect();
    body.extend((0..2).map(|j| load(4 + j as u32, 1, 2, j)));
    body.push(Stmt::For {
        unroll: Unroll::None,
        local: 6,
        start: cu(0),
        end: ScalarExpr::ConstInt(rounds, U32),
        step: cu(1),
        body: Block(round(mutant)),
    });
    body.extend((0..4).map(|j| Stmt::Store {
        place: word(2, 4, j),
        value: l(j as u32),
    }));
    let params = vec![
        Param::buffer("ctr", U32, BufferRole::Read, Some(1)),
        Param::buffer("key", U32, BufferRole::Read, Some(1)),
        Param::buffer("out", U32, BufferRole::Write, Some(1)),
    ];
    let mut locals = vec![LocalDecl::scalar(U32); 7];
    locals.extend([U64, U64].map(LocalDecl::scalar));
    locals.extend(vec![LocalDecl::scalar(U32); 4]);
    KernelFn::new("philox4x32_10", F32, params, locals, Block(body))
        .with_launch(LaunchShape::new([N as u32, 1, 1], [1, 1, 1]))
}
