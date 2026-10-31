// SPDX-License-Identifier: MIT OR Apache-2.0
//! **Integer division and shifts are total in the kernel interpreter**
//! (`docs/int-div-shift-semantics.md`, PM ruling 2026-10-31).
//!
//! Rules under test, for `I32`, `I64`, `U32` and `U64`: `a / 0` gives `a`; signed `MIN / -1` gives
//! `MIN`; otherwise the quotient truncated toward zero; a shift count is read as unsigned bits
//! modulo the width (`<<` defined for a negative or overflowing `a`, `>>` arithmetic on signed and
//! logical on unsigned). The vectors are the ruling's: divisors and dividends over `0, 1, -1, MIN,
//! MAX` and more, counts `0, 1, N-1, N, N+1, 255, -1`.
//!
//! The expectation is computed with the host's native integer types, written out case by case from
//! the rule (`wrapping_shl`/`wrapping_shr` mask the count by the width, as the rule says), so it
//! shares no code with the interpreter.

use unpopped::ir::{BinaryOp, ScalarExpr};
use unpopped::kernel::{Block, BufferRole, KernelFn, LocalDecl, Param, Place, Stmt};
use unpopped::oracle::TypedBuffer;
use unpopped::oracle::kernel_eval::{KernelArg, KernelTrap, run_kernel};
use unpopped::types::{int_range, typed_bits};
use unpopped_vocab::ElementKind;

const I32: ElementKind = ElementKind::I32;
const INTS: [ElementKind; 4] = [I32, ElementKind::I64, ElementKind::U32, ElementKind::U64];

#[derive(Copy, Clone, Debug, PartialEq)]
enum Op {
    Div,
    Shl,
    Shr,
}

fn local(i: u32) -> Box<ScalarExpr> {
    Box::new(ScalarExpr::Local(i))
}

/// Element `i` of buffer parameter `buffer`.
fn at_i(buffer: u32) -> Place {
    Place::Logical {
        buffer,
        coords: vec![ScalarExpr::Local(1)],
    }
}

/// `out[i] = a[i] <op> b[i]` for `i` in `0..n`, at compute dtype `t`.
fn kernel(t: ElementKind, op: Op) -> KernelFn {
    let value = match op {
        Op::Div => ScalarExpr::Div(local(2), local(3)),
        Op::Shl => ScalarExpr::Binary(BinaryOp::Shl, local(2), local(3)),
        Op::Shr => ScalarExpr::Binary(BinaryOp::Shr, local(2), local(3)),
    };
    let body = Block(vec![
        Stmt::Load {
            local: 2,
            place: at_i(0),
        },
        Stmt::Load {
            local: 3,
            place: at_i(1),
        },
        Stmt::Store {
            place: at_i(2),
            value,
        },
    ]);
    let looped = Stmt::For {
        local: 1,
        start: ScalarExpr::ConstInt(0, I32),
        end: ScalarExpr::Local(0),
        step: ScalarExpr::ConstInt(1, I32),
        body,
    };
    KernelFn::new(
        "int_op",
        t,
        vec![
            Param::buffer("a", t, BufferRole::Read, Some(1)),
            Param::buffer("b", t, BufferRole::Read, Some(1)),
            Param::buffer("out", t, BufferRole::Write, Some(1)),
            Param::scalar("n", I32, 0),
        ],
        vec![
            LocalDecl::scalar(I32),
            LocalDecl::scalar(I32),
            LocalDecl::scalar(t),
            LocalDecl::scalar(t),
        ],
        Block(vec![looped]),
    )
}

fn width(t: ElementKind) -> usize {
    typed_bits(t).expect("an integer type") as usize / 8
}

/// `v` as the dtype's raw bits: `-1` on an unsigned type means all ones.
fn raw(t: ElementKind, v: i128) -> i128 {
    let (lo, hi) = int_range(t).expect("an integer type");
    if v < 0 && lo == 0 { hi } else { v }
}

fn bytes(t: ElementKind, vs: &[i128]) -> Vec<u8> {
    vs.iter()
        .flat_map(|&v| raw(t, v).to_le_bytes()[..width(t)].to_vec())
        .collect()
}

fn read(t: ElementKind, b: &[u8]) -> Vec<i128> {
    let (lo, _) = int_range(t).expect("an integer type");
    b.chunks(width(t))
        .map(|c| {
            let mut w = [0u8; 16];
            w[..c.len()].copy_from_slice(c);
            let u = u128::from_le_bytes(w) as i128;
            if lo < 0 && c.last().is_some_and(|m| m & 0x80 != 0) {
                u - (1i128 << (8 * c.len()))
            } else {
                u
            }
        })
        .collect()
}

/// The ruling's vectors as parallel `(a, b)` lists: every dividend against every divisor or count.
fn pairs(t: ElementKind, op: Op) -> (Vec<i128>, Vec<i128>) {
    let (lo, hi) = int_range(t).expect("an integer type");
    let n = i128::from(typed_bits(t).expect("an integer type"));
    let mut dividends = vec![0, 1, -1, lo, hi, hi - 1, 7, -7, 1 << (n - 2)];
    let mut rhs = match op {
        Op::Div => vec![0, 1, -1, lo, hi, 2, -3, 7],
        _ => vec![0, 1, n - 1, n, n + 1, 255, -1, 32, 31, 63, 64],
    };
    for v in dividends.iter_mut().chain(rhs.iter_mut()) {
        *v = raw(t, *v);
    }
    dividends.sort_unstable();
    dividends.dedup();
    rhs.sort_unstable();
    rhs.dedup();
    let a = dividends
        .iter()
        .flat_map(|&d| rhs.iter().map(move |_| d))
        .collect();
    let b = dividends.iter().flat_map(|_| rhs.clone()).collect();
    (a, b)
}

macro_rules! native {
    ($ty:ty, $op:expr, $a:expr, $b:expr) => {{
        let (a, b) = ($a as $ty, $b as $ty);
        let r: $ty = match $op {
            Op::Div => {
                let minus_one = (0 as $ty).wrapping_sub(1);
                if b == 0 || (<$ty>::MIN != 0 && a == <$ty>::MIN && b == minus_one) {
                    a
                } else {
                    a / b
                }
            }
            Op::Shl => a.wrapping_shl(b as u32),
            Op::Shr => a.wrapping_shr(b as u32),
        };
        r as i128
    }};
}

/// The expected result from the host's own integer types, case by case from the rule.
fn expected(t: ElementKind, op: Op, a: i128, b: i128) -> i128 {
    match t {
        ElementKind::I32 => native!(i32, op, a, b),
        ElementKind::I64 => native!(i64, op, a, b),
        ElementKind::U32 => native!(u32, op, a, b),
        _ => native!(u64, op, a, b),
    }
}

fn run(t: ElementKind, op: Op, a: &[i128], b: &[i128]) -> Result<Vec<i128>, KernelTrap> {
    let n = a.len();
    let buf = |bytes: Vec<u8>| TypedBuffer::new(t, vec![n as i64], vec![1], bytes);
    let mut args = vec![
        KernelArg::Buffer(buf(bytes(t, a))),
        KernelArg::Buffer(buf(bytes(t, b))),
        KernelArg::Buffer(buf(vec![0u8; n * width(t)])),
        KernelArg::Int(n as i128),
    ];
    run_kernel(&kernel(t, op), &mut args)?;
    let KernelArg::Buffer(out) = &args[2] else {
        unreachable!("the output is the third argument")
    };
    Ok(read(t, &out.bytes))
}

#[test]
fn division_and_shifts_are_total_and_match_the_rule() {
    for t in INTS {
        for op in [Op::Div, Op::Shl, Op::Shr] {
            let (a, b) = pairs(t, op);
            let got = run(t, op, &a, &b).unwrap_or_else(|e| panic!("{t:?} {op:?}: {e:?}"));
            for (i, (&x, &y)) in a.iter().zip(&b).enumerate() {
                assert_eq!(got[i], expected(t, op, x, y), "{t:?} {op:?} a={x} b={y}");
            }
        }
    }
}

/// The named cases of the ruling, written as numbers so a mistaken reference cannot hide them.
#[test]
fn the_ruled_boundary_cases_by_value() {
    let one = |t, op, a: i128, b: i128| run(t, op, &[a], &[b]).expect("runs")[0];
    let (i32_min, i64_min) = (i128::from(i32::MIN), i128::from(i64::MIN));
    assert_eq!(one(I32, Op::Div, 7, 0), 7, "x / 0 is x");
    assert_eq!(one(I32, Op::Div, i32_min, -1), i32_min, "MIN / -1 is MIN");
    assert_eq!(one(I32, Op::Div, -7, 2), -3, "truncates toward zero");
    assert_eq!(one(ElementKind::I64, Op::Div, i64_min, -1), i64_min);
    assert_eq!(one(ElementKind::U32, Op::Div, 9, 0), 9);
    assert_eq!(one(I32, Op::Shl, 1, 33), 2, "count 33 is 1 modulo 32");
    assert_eq!(one(I32, Op::Shl, 1, -1), i32_min, "count -1 is 31");
    assert_eq!(one(I32, Op::Shr, -8, 1), -4, "arithmetic on signed");
    assert_eq!(
        one(ElementKind::U32, Op::Shr, 0x8000_0000, 31),
        1,
        "logical"
    );
    assert_eq!(
        one(ElementKind::U64, Op::Shl, 1, 65),
        2,
        "count 65 is 1 modulo 64"
    );
    assert_eq!(one(I32, Op::Shl, -1, 1), -2, "<< of a negative is defined");
}

/// `out[0] = (a <op> b) < 0` at `I32`: a comparison reads the operation's value itself, where a
/// store would truncate an unwrapped result to the same four bytes. This is what tells `MIN / -1 =
/// MIN` (negative) from an unwrapped `2^31`, and a wrapped `1 << 31` from the same.
fn negative(op: Op, a: i128, b: i128) -> i128 {
    let value = match op {
        Op::Div => ScalarExpr::Div(local(2), local(3)),
        Op::Shl => ScalarExpr::Binary(BinaryOp::Shl, local(2), local(3)),
        Op::Shr => ScalarExpr::Binary(BinaryOp::Shr, local(2), local(3)),
    };
    let mut k = kernel(I32, op);
    let Some(Stmt::For { body, .. }) = k.body.0.first_mut() else {
        unreachable!("the loop")
    };
    let Some(Stmt::Store { value: v, .. }) = body.0.last_mut() else {
        unreachable!("the store")
    };
    *v = ScalarExpr::Binary(
        BinaryOp::CmpLt,
        Box::new(value),
        Box::new(ScalarExpr::ConstInt(0, I32)),
    );
    let buf = |x: i128| TypedBuffer::new(I32, vec![1], vec![1], bytes(I32, &[x]));
    let mut args = vec![
        KernelArg::Buffer(buf(a)),
        KernelArg::Buffer(buf(b)),
        KernelArg::Buffer(buf(0)),
        KernelArg::Int(1),
    ];
    run_kernel(&k, &mut args).expect("runs");
    let KernelArg::Buffer(out) = &args[2] else {
        unreachable!("the output is the third argument")
    };
    read(I32, &out.bytes)[0]
}

#[test]
fn the_wrapped_value_not_just_its_low_bytes_is_what_the_operation_yields() {
    let min = i128::from(i32::MIN);
    assert_eq!(negative(Op::Div, min, -1), 1, "MIN / -1 is MIN, negative");
    assert_eq!(
        negative(Op::Shl, 1, 31),
        1,
        "1 << 31 wraps to MIN, negative"
    );
    assert_eq!(negative(Op::Shl, 1, 63), 1, "count 63 is 31 modulo 32");
    assert_eq!(negative(Op::Shr, min, 31), 1, "arithmetic: stays negative");
    assert_eq!(negative(Op::Div, 5, 0), 0, "5 / 0 is 5, not negative");
}
