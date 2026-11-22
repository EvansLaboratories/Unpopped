// SPDX-License-Identifier: MIT OR Apache-2.0
//! **Integer `Div`, shifts and bitwise operators at any integer type in a kernel** (U19 finding, pm-kernels option A,
//! 2026-11-22). The kernel layer's type rule (`types::infer_ty_in`) used to refuse them on any integer type other than the
//! kernel's compute type, so a float kernel could not shift an index; the plan layer's rule is unchanged. Each operation runs here
//! on every integer width, `I8` to `U64`, signed and unsigned, in a kernel whose compute type is `F32`, against an independent
//! reference written from `docs/int-div-shift-semantics.md`: `Div` is total (a zero divisor and `MIN / -1` give the dividend); a
//! shift count is read as unsigned bits modulo the promoted width (64 for a 64-bit type, 32 for every narrower one); `<<`
//! discards what leaves the type, `>>` is arithmetic on a signed type and logical on an unsigned one. Operands are asymmetric
//! (negative with positive, a high bit alone, counts past the width) so a swapped operand or a wrong width shows.

use unpopped::ir::{BinaryOp, ScalarExpr};
use unpopped::kernel::{
    Block, BufferRole, KernelFn, LocalDecl, Param, Place, Stmt, Unroll, check_kernel,
};
use unpopped::oracle::TypedBuffer;
use unpopped::oracle::kernel_eval::{KernelArg, run_kernel};
use unpopped::types::{int_range, typed_bits};
use unpopped_vocab::ElementKind as E;

const INTS: [E; 8] = [E::I8, E::U8, E::I16, E::U16, E::I32, E::U32, E::I64, E::U64];

#[derive(Copy, Clone, Debug, PartialEq)]
enum Op {
    Div,
    Shl,
    Shr,
    And,
    Or,
    Xor,
    /// `(a << b) >> b`: the shifted value is used again, so a `<<` that did not wrap into the type shows here.
    ShlThenShr,
}

const OPS: [Op; 7] = [
    Op::Div,
    Op::Shl,
    Op::Shr,
    Op::And,
    Op::Or,
    Op::Xor,
    Op::ShlThenShr,
];

fn local(i: u32) -> Box<ScalarExpr> {
    Box::new(ScalarExpr::Local(i))
}

fn at_i(buffer: u32) -> Place {
    Place::Logical {
        buffer,
        coords: vec![ScalarExpr::Local(0)],
    }
}

/// `out[i] = a[i] <op> b[i]` for `i` in `0..n`, operands and result of type `t`, kernel compute type `F32`.
fn kernel(t: E, op: Op, n: usize) -> KernelFn {
    let value = match op {
        Op::Div => ScalarExpr::Div(local(1), local(2)),
        Op::Shl => ScalarExpr::Binary(BinaryOp::Shl, local(1), local(2)),
        Op::Shr => ScalarExpr::Binary(BinaryOp::Shr, local(1), local(2)),
        Op::And => ScalarExpr::Binary(BinaryOp::BitAnd, local(1), local(2)),
        Op::Or => ScalarExpr::Binary(BinaryOp::BitOr, local(1), local(2)),
        Op::Xor => ScalarExpr::Binary(BinaryOp::BitXor, local(1), local(2)),
        Op::ShlThenShr => ScalarExpr::Binary(
            BinaryOp::Shr,
            Box::new(ScalarExpr::Binary(BinaryOp::Shl, local(1), local(2))),
            local(2),
        ),
    };
    let body = Block(vec![
        Stmt::Load {
            local: 1,
            place: at_i(0),
        },
        Stmt::Load {
            local: 2,
            place: at_i(1),
        },
        Stmt::Store {
            place: at_i(2),
            value,
        },
    ]);
    KernelFn::new(
        "int_op",
        E::F32,
        vec![
            Param::buffer("a", t, BufferRole::Read, Some(1)),
            Param::buffer("b", t, BufferRole::Read, Some(1)),
            Param::buffer("out", t, BufferRole::Write, Some(1)),
        ],
        vec![
            LocalDecl::scalar(E::I32),
            LocalDecl::scalar(t),
            LocalDecl::scalar(t),
        ],
        Block(vec![Stmt::For {
            local: 0,
            start: ScalarExpr::ConstInt(0, E::I32),
            end: ScalarExpr::ConstInt(n as i64, E::I32),
            step: ScalarExpr::ConstInt(1, E::I32),
            unroll: Unroll::None,
            body,
        }]),
    )
}

fn width(t: E) -> usize {
    typed_bits(t).expect("an integer type") as usize / 8
}

/// `v` wrapped into `t`'s range (two's complement for signed, modulo for unsigned).
fn wrap(t: E, v: i128) -> i128 {
    let (lo, hi) = int_range(t).expect("an integer type");
    let m = hi - lo + 1;
    (v - lo).rem_euclid(m) + lo
}

/// The reference, written from the ruling and nothing else.
fn reference(t: E, op: Op, x: i128, y: i128) -> i128 {
    let (lo, _) = int_range(t).expect("an integer type");
    let promoted: u32 = if typed_bits(t) == Some(64) { 64 } else { 32 };
    let count = (y as u128 & u128::from(promoted - 1)) as u32;
    match op {
        Op::Div if y == 0 || (lo < 0 && x == lo && y == -1) => x,
        Op::Div => x / y,
        Op::Shl => wrap(t, x << count),
        Op::Shr => x >> count,
        Op::And => x & y,
        Op::Or => x | y,
        Op::Xor => x ^ y,
        Op::ShlThenShr => reference(t, Op::Shr, reference(t, Op::Shl, x, y), y),
    }
}

/// Dividends and bit patterns: both signs, the extremes, a lone high bit, an alternating pattern.
fn lhs(t: E) -> Vec<i128> {
    let (lo, hi) = int_range(t).expect("an integer type");
    let bits = i128::from(typed_bits(t).expect("an integer type"));
    let mut v = vec![0, 1, lo, hi, hi - 1, 7, 85, 1 << (bits - 2), -1, -7, -85];
    v.retain(|x| (lo..=hi).contains(x));
    v.sort_unstable();
    v.dedup();
    v
}

/// The right operand: divisors for `Div`, counts (past the width, and negative) for a shift, patterns for a bit operation.
fn rhs(t: E, op: Op) -> Vec<i128> {
    let (lo, hi) = int_range(t).expect("an integer type");
    let bits = i128::from(typed_bits(t).expect("an integer type"));
    let mut v = match op {
        Op::Div => vec![0, 1, -1, lo, hi, 2, -3, 7],
        Op::Shl | Op::Shr | Op::ShlThenShr => {
            vec![0, 1, bits - 1, bits, bits + 1, 31, 32, 33, 63, 64, 255, -1]
        }
        _ => vec![0, 1, lo, hi, 85, -1, -86],
    };
    v.retain(|x| (lo..=hi).contains(x));
    v.sort_unstable();
    v.dedup();
    v
}

fn buffer(t: E, vs: &[i128]) -> TypedBuffer {
    let bytes = vs
        .iter()
        .flat_map(|v| v.to_le_bytes()[..width(t)].to_vec())
        .collect();
    TypedBuffer::new(t, vec![vs.len() as i64], vec![1], bytes)
}

/// The value of each little-endian element of `b`, sign-extended for a signed type.
fn decode(t: E, b: &TypedBuffer) -> Vec<i128> {
    let (lo, _) = int_range(t).expect("an integer type");
    b.bytes
        .chunks(width(t))
        .map(|c| {
            let mut raw = [0u8; 16];
            raw[..c.len()].copy_from_slice(c);
            let u = u128::from_le_bytes(raw);
            let v = u as i128;
            if lo < 0 && v >= (1i128 << (8 * c.len() - 1)) {
                v - (1i128 << (8 * c.len()))
            } else {
                v
            }
        })
        .collect()
}

fn inputs(t: E, op: Op) -> (Vec<i128>, Vec<i128>) {
    let (l, r) = (lhs(t), rhs(t, op));
    let a = l.iter().flat_map(|&x| r.iter().map(move |_| x)).collect();
    let b = l.iter().flat_map(|_| r.clone()).collect();
    (a, b)
}

fn run(t: E, op: Op) -> (Vec<i128>, Vec<i128>, Vec<i128>) {
    let (a, b) = inputs(t, op);
    let k = kernel(t, op, a.len());
    check_kernel(&k)
        .unwrap_or_else(|e| panic!("{t:?} {op:?} must type-check in a float kernel: {e:?}"));
    let mut args = vec![
        KernelArg::Buffer(buffer(t, &a)),
        KernelArg::Buffer(buffer(t, &b)),
        KernelArg::Buffer(buffer(t, &vec![0; a.len()])),
    ];
    run_kernel(&k, &mut args).unwrap_or_else(|e| panic!("{t:?} {op:?} runs: {e:?}"));
    let KernelArg::Buffer(out) = &args[2] else {
        unreachable!()
    };
    (a, b, decode(t, out))
}

#[test]
fn every_operation_at_every_width_agrees_with_the_ruling_in_a_float_kernel() {
    for t in INTS {
        for op in OPS {
            let (a, b, got) = run(t, op);
            let want: Vec<i128> = a
                .iter()
                .zip(&b)
                .map(|(&x, &y)| reference(t, op, x, y))
                .collect();
            assert_eq!(got.len(), want.len());
            for i in 0..got.len() {
                assert_eq!(got[i], want[i], "{t:?} {op:?}: {} {op:?} {}", a[i], b[i]);
            }
        }
    }
}

#[test]
fn the_vectors_are_not_trivially_symmetric() {
    // Asymmetric on purpose: a wrong operand order or a wrong width changes some result for every operation that has one.
    let r = |op, x, y| reference(E::I8, op, x, y);
    assert_ne!(r(Op::Shl, 3, 1), r(Op::Shl, 1, 3));
    assert_ne!(r(Op::Shr, -64, 2), r(Op::Shr, 2, -64));
    assert_ne!(r(Op::Div, 7, -3), r(Op::Div, -3, 7));
    // a narrow signed shift keeps the sign, an unsigned one does not
    assert_eq!(reference(E::I8, Op::Shr, -128, 3), -16);
    assert_eq!(reference(E::U8, Op::Shr, 128, 3), 16);
    // a count of the promoted width wraps to 0 (32 for a narrow type, 64 for a wide one)
    assert_eq!(reference(E::I8, Op::Shl, 1, 32), 1);
    assert_eq!(reference(E::U64, Op::Shl, 1, 64), 1);
    assert_eq!(reference(E::U64, Op::Shl, 1, 63), 1 << 63);
}

#[test]
fn the_logical_operators_and_float_only_operators_stay_refused() {
    // The relaxation is the five integer operators and `Div`; nothing else.
    for op in [
        BinaryOp::LogicalAnd,
        BinaryOp::Pow,
        BinaryOp::Rem,
        BinaryOp::RemTrunc,
    ] {
        let mut k = kernel(E::I32, Op::And, 2);
        if let Stmt::For { body, .. } = &mut k.body.0[0] {
            if let Stmt::Store { value, .. } = &mut body.0[2] {
                *value = ScalarExpr::Binary(op, local(1), local(2));
            }
        }
        assert!(
            check_kernel(&k).is_err(),
            "{op:?} on I32 in a float kernel must stay refused"
        );
    }
}
