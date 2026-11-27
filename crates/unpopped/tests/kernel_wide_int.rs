// SPDX-License-Identifier: MIT OR Apache-2.0
//! The wide integer dtypes `I128`, `U128`, `I256`, `U256` in a kernel (ROADMAP E2): wrapping arithmetic at the type's width, the
//! total `Div`, shifts with the count modulo the width, ordering, comparisons, and integer casts to and from the narrow integers
//! (`docs/int-div-shift-semantics.md` at the wide width).
//!
//! The 128-bit types are compared against Rust's own `i128`/`u128` operators (independent of the oracle's 256-bit limb type);
//! the 256-bit types against a `(hi, lo)` pair of `u128` model written here, and against hand-pinned values. Every operand set
//! includes 0, 1, the all-ones pattern, the sign bit alone and a pattern that crosses the 128-bit boundary.

use unpopped::exec::{ExecId, LaunchShape};
use unpopped::ir::{BinaryOp, CastSpec, RoundMode, Saturation, ScalarExpr};
use unpopped::kernel::{
    Block, BufferRole, KernelError, KernelFn, LocalDecl, Param, Place, Stmt, check_kernel,
};
use unpopped::oracle::TypedBuffer;
use unpopped::oracle::kernel_eval::KernelArg;
use unpopped::oracle::kernel_launch::run_launch;
use unpopped_vocab::ElementKind::{self, I32, I64, I128, I256, U8, U128, U256};

/// A 256-bit pattern as `(hi, lo)`.
type P = (u128, u128);

fn cs(to: ElementKind, sat: Saturation) -> CastSpec {
    CastSpec {
        to,
        round: RoundMode::NearestEven,
        sat,
    }
}

fn at_tid(buffer: u32) -> Place {
    Place::Logical {
        buffer,
        coords: vec![ScalarExpr::Exec(ExecId::ThreadIdx(0))],
    }
}

fn bytes_of(t: ElementKind, p: P) -> Vec<u8> {
    let mut b = p.1.to_le_bytes().to_vec();
    if matches!(t, I256 | U256) {
        b.extend(p.0.to_le_bytes());
    }
    b
}

fn width(t: ElementKind) -> usize {
    if matches!(t, I256 | U256) { 32 } else { 16 }
}

fn pattern(t: ElementKind, c: &[u8]) -> P {
    let lo = u128::from_le_bytes(c[..16].try_into().unwrap());
    if width(t) == 32 {
        (u128::from_le_bytes(c[16..].try_into().unwrap()), lo)
    } else {
        (0, lo)
    }
}

/// `out[tid] = a[tid] <op> b[tid]` at `t`, the result stored at `out_t` (the compute type I32 for a comparison).
fn binary_kernel(op: BinaryOp, t: ElementKind, out_t: ElementKind, n: u32) -> KernelFn {
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

fn arith_kernel(
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

fn run(k: &KernelFn, t: ElementKind, out_t: ElementKind, a: &[P], b: &[P]) -> Vec<P> {
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
fn grid(t: ElementKind) -> (Vec<P>, Vec<P>) {
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

// ---- 128-bit reference: Rust's own operators ------------------------------------------------------------------------

fn ref128(op: BinaryOp, signed: bool, x: u128, y: u128) -> u128 {
    let (sx, sy) = (x as i128, y as i128);
    let sh = (y & 127) as u32;
    match op {
        BinaryOp::BitAnd => x & y,
        BinaryOp::BitOr => x | y,
        BinaryOp::BitXor => x ^ y,
        BinaryOp::Shl => x << sh,
        BinaryOp::Shr if signed => (sx >> sh) as u128,
        BinaryOp::Shr => x >> sh,
        BinaryOp::Max if signed => sx.max(sy) as u128,
        BinaryOp::Max => x.max(y),
        BinaryOp::Min if signed => sx.min(sy) as u128,
        BinaryOp::Min => x.min(y),
        _ => unreachable!(),
    }
}

#[test]
fn the_128_bit_types_match_rusts_own_integers() {
    for (t, signed) in [(I128, true), (U128, false)] {
        let (a, b) = grid(t);
        for op in [
            BinaryOp::BitAnd,
            BinaryOp::BitOr,
            BinaryOp::BitXor,
            BinaryOp::Shl,
            BinaryOp::Shr,
            BinaryOp::Max,
            BinaryOp::Min,
        ] {
            let got = run(&binary_kernel(op, t, t, a.len() as u32), t, t, &a, &b);
            let want: Vec<P> = a
                .iter()
                .zip(&b)
                .map(|(x, y)| (0, ref128(op, signed, x.1, y.1)))
                .collect();
            assert_eq!(got, want, "{op:?} at {t:?}");
        }
        let add = run(
            &arith_kernel(|x, y| ScalarExpr::Add(x, y), t, a.len() as u32),
            t,
            t,
            &a,
            &b,
        );
        let sub = run(
            &arith_kernel(|x, y| ScalarExpr::Sub(x, y), t, a.len() as u32),
            t,
            t,
            &a,
            &b,
        );
        let mul = run(
            &arith_kernel(|x, y| ScalarExpr::Mul(x, y), t, a.len() as u32),
            t,
            t,
            &a,
            &b,
        );
        let div = run(
            &arith_kernel(|x, y| ScalarExpr::Div(x, y), t, a.len() as u32),
            t,
            t,
            &a,
            &b,
        );
        for (i, (x, y)) in a.iter().zip(&b).enumerate() {
            let (x, y) = (x.1, y.1);
            assert_eq!(add[i].1, x.wrapping_add(y), "add {x:#x} {y:#x} {t:?}");
            assert_eq!(sub[i].1, x.wrapping_sub(y), "sub {x:#x} {y:#x} {t:?}");
            assert_eq!(mul[i].1, x.wrapping_mul(y), "mul {x:#x} {y:#x} {t:?}");
            let q = if y == 0 {
                x // the total rule: a zero divisor gives the dividend
            } else if signed {
                (x as i128).wrapping_div(y as i128) as u128 // MIN / -1 wraps to MIN, the total rule's answer
            } else {
                x / y
            };
            assert_eq!(div[i].1, q, "div {x:#x} {y:#x} {t:?}");
        }
        for (op, f) in [
            (
                BinaryOp::CmpLt,
                (|o| o == std::cmp::Ordering::Less) as fn(std::cmp::Ordering) -> bool,
            ),
            (BinaryOp::CmpEq, |o| o == std::cmp::Ordering::Equal),
            (BinaryOp::CmpGe, |o| o != std::cmp::Ordering::Less),
        ] {
            let got = run(&binary_kernel(op, t, I32, a.len() as u32), t, I32, &a, &b);
            for (i, (x, y)) in a.iter().zip(&b).enumerate() {
                let o = if signed {
                    (x.1 as i128).cmp(&(y.1 as i128))
                } else {
                    x.1.cmp(&y.1)
                };
                assert_eq!(
                    got[i].1,
                    u128::from(f(o)),
                    "{op:?} {:#x} {:#x} at {t:?}",
                    x.1,
                    y.1
                );
            }
        }
    }
}

// ---- 256-bit reference: a (hi, lo) pair of u128 -----------------------------------------------------------------------

fn add256(a: P, b: P) -> P {
    let (lo, c) = a.1.overflowing_add(b.1);
    (a.0.wrapping_add(b.0).wrapping_add(u128::from(c)), lo)
}

fn not256(a: P) -> P {
    (!a.0, !a.1)
}

fn neg256(a: P) -> P {
    add256(not256(a), (0, 1))
}

fn shl256(a: P, n: u32) -> P {
    match n {
        0 => a,
        1..=127 => ((a.0 << n) | (a.1 >> (128 - n)), a.1 << n),
        128 => (a.1, 0),
        _ => (a.1 << (n - 128), 0),
    }
}

fn shr256(a: P, n: u32, arith: bool) -> P {
    let fill = if arith && (a.0 >> 127) == 1 {
        u128::MAX
    } else {
        0
    };
    match n {
        0 => a,
        1..=127 => (
            (a.0 >> n) | if fill != 0 { fill << (128 - n) } else { 0 },
            (a.1 >> n) | (a.0 << (128 - n)),
        ),
        128 => (fill, a.0),
        _ => (
            fill,
            (a.0 >> (n - 128))
                | if fill != 0 && n > 128 {
                    fill << (256 - n)
                } else {
                    0
                },
        ),
    }
}

fn cmp256(a: P, b: P, signed: bool) -> std::cmp::Ordering {
    if signed {
        ((a.0 as i128), a.1).cmp(&((b.0 as i128), b.1))
    } else {
        a.cmp(&b)
    }
}

#[test]
fn the_256_bit_types_match_the_limb_pair_model() {
    for (t, signed) in [(I256, true), (U256, false)] {
        let (a, b) = grid(t);
        let n = a.len() as u32;
        let check = |name: &str, got: Vec<P>, f: &dyn Fn(P, P) -> P| {
            let want: Vec<P> = a.iter().zip(&b).map(|(x, y)| f(*x, *y)).collect();
            assert_eq!(got, want, "{name} at {t:?}");
        };
        check(
            "add",
            run(
                &arith_kernel(|x, y| ScalarExpr::Add(x, y), t, n),
                t,
                t,
                &a,
                &b,
            ),
            &add256,
        );
        check(
            "sub",
            run(
                &arith_kernel(|x, y| ScalarExpr::Sub(x, y), t, n),
                t,
                t,
                &a,
                &b,
            ),
            &|x, y| add256(x, neg256(y)),
        );
        check(
            "and",
            run(&binary_kernel(BinaryOp::BitAnd, t, t, n), t, t, &a, &b),
            &|x, y| (x.0 & y.0, x.1 & y.1),
        );
        check(
            "xor",
            run(&binary_kernel(BinaryOp::BitXor, t, t, n), t, t, &a, &b),
            &|x, y| (x.0 ^ y.0, x.1 ^ y.1),
        );
        check(
            "shl",
            run(&binary_kernel(BinaryOp::Shl, t, t, n), t, t, &a, &b),
            &|x, y| shl256(x, (y.1 & 255) as u32),
        );
        check(
            "shr",
            run(&binary_kernel(BinaryOp::Shr, t, t, n), t, t, &a, &b),
            &|x, y| shr256(x, (y.1 & 255) as u32, signed),
        );
        check(
            "max",
            run(&binary_kernel(BinaryOp::Max, t, t, n), t, t, &a, &b),
            &|x, y| {
                if cmp256(x, y, signed) == std::cmp::Ordering::Less {
                    y
                } else {
                    x
                }
            },
        );
        let lt = run(&binary_kernel(BinaryOp::CmpLt, t, I32, n), t, I32, &a, &b);
        for (i, (x, y)) in a.iter().zip(&b).enumerate() {
            let want = u128::from(cmp256(*x, *y, signed) == std::cmp::Ordering::Less);
            assert_eq!(lt[i].1, want, "lt {x:x?} {y:x?} at {t:?}");
        }
    }
}

#[test]
fn the_256_bit_product_and_quotient_are_pinned_by_hand() {
    let (x, one) = ((1u128, 0u128), (0u128, 1u128)); // 2^128 and 1
    let mul = |t, a: P, b: P| {
        run(
            &arith_kernel(|x, y| ScalarExpr::Mul(x, y), t, 1),
            t,
            t,
            &[a],
            &[b],
        )[0]
    };
    let div = |t, a: P, b: P| {
        run(
            &arith_kernel(|x, y| ScalarExpr::Div(x, y), t, 1),
            t,
            t,
            &[a],
            &[b],
        )[0]
    };
    for t in [I256, U256] {
        assert_eq!(mul(t, x, x), (0, 0), "2^128 * 2^128 wraps to 0 at 256 bits");
        assert_eq!(mul(t, x, one), x);
        assert_eq!(
            mul(t, (0, u128::MAX), (0, u128::MAX)),
            (u128::MAX - 1, 1),
            "(2^128-1)^2 = 2^256 - 2^129 + 1"
        );
        assert_eq!(div(t, x, one), x);
        assert_eq!(div(t, x, (0, 2)), (0, 1 << 127), "2^128 / 2");
        assert_eq!(
            div(t, (5, 7), (0, 0)),
            (5, 7),
            "a zero divisor gives the dividend"
        );
    }
    let all_ones = (u128::MAX, u128::MAX);
    assert_eq!(
        div(U256, all_ones, (0, 2)),
        (u128::MAX >> 1, u128::MAX),
        "2^256-1 / 2 as unsigned"
    );
    assert_eq!(
        div(I256, all_ones, (0, 2)),
        (0, 0),
        "-1 / 2 truncates toward zero"
    );
    let min = (1u128 << 127, 0u128);
    assert_eq!(div(I256, min, all_ones), min, "MIN / -1 gives MIN");
}

#[test]
fn casts_between_wide_and_narrow_integers_wrap_and_clamp() {
    let cast = |from: ElementKind, to: ElementKind, spec: CastSpec, v: P| -> P {
        let mut k = KernelFn::new(
            "cast",
            I32,
            vec![
                Param::buffer("in", from, BufferRole::Read, Some(1)),
                Param::buffer("out", to, BufferRole::Write, Some(1)),
            ],
            vec![LocalDecl::scalar(from)],
            Block(vec![
                Stmt::Load {
                    local: 0,
                    place: at_tid(0),
                },
                Stmt::Store {
                    place: at_tid(1),
                    value: ScalarExpr::Cast(Box::new(ScalarExpr::Local(0)), spec),
                },
            ]),
        );
        k.launch = Some(LaunchShape::new([1, 1, 1], [1, 1, 1]));
        assert_eq!(check_kernel(&k), Ok(()), "{from:?} to {to:?}");
        let mut args = vec![
            KernelArg::Buffer(TypedBuffer::new(from, vec![1], vec![1], bytes_of(from, v))),
            KernelArg::Buffer(TypedBuffer::new(
                to,
                vec![1],
                vec![1],
                vec![0u8; width(to).max(8)],
            )),
        ];
        run_launch(&k, &mut args).unwrap_or_else(|e| panic!("{from:?} to {to:?}: {e:?}"));
        let KernelArg::Buffer(out) = &args[1] else {
            unreachable!()
        };
        let c = &out.bytes;
        match to {
            I64 => (
                0,
                u128::from(u64::from_le_bytes(c[..8].try_into().unwrap())),
            ),
            U8 => (0, u128::from(c[0])),
            _ => pattern(to, &c[..width(to)]),
        }
    };
    let spec = |to, sat: &str| {
        cs(
            to,
            if sat == "w" {
                Saturation::Wrap
            } else {
                Saturation::Clamp
            },
        )
    };
    let minus_one_64 = (0, u128::from(u64::MAX));
    // I64 -1 -> U128: wraps to 2^128 - 1; clamps to 0.
    assert_eq!(
        cast(I64, U128, spec(U128, "w"), minus_one_64),
        (0, u128::MAX)
    );
    assert_eq!(cast(I64, U128, spec(U128, "c"), minus_one_64), (0, 0));
    // U256 all-ones -> I128: wraps to -1; clamps to the I128 maximum.
    let all_ones = (u128::MAX, u128::MAX);
    assert_eq!(cast(U256, I128, spec(I128, "w"), all_ones), (0, u128::MAX));
    assert_eq!(
        cast(U256, I128, spec(I128, "c"), all_ones),
        (0, u128::MAX >> 1)
    );
    // I256 minimum -> I128 clamps to the I128 minimum.
    assert_eq!(
        cast(I256, I128, spec(I128, "c"), (1u128 << 127, 0)),
        (0, 1 << 127)
    );
    // I128 300 -> U8: wraps to 44, clamps to 255.
    assert_eq!(cast(I128, U8, spec(U8, "w"), (0, 300)), (0, 44));
    assert_eq!(cast(I128, U8, spec(U8, "c"), (0, 300)), (0, 255));
    // I64 -1 -> I256 sign-extends over all 256 bits.
    assert_eq!(
        cast(I64, I256, spec(I256, "w"), minus_one_64),
        (u128::MAX, u128::MAX)
    );
}

#[test]
fn a_float_cast_and_a_float_operand_are_refused_with_the_integer_control() {
    let kernel = |from: ElementKind, to: ElementKind| {
        let mut k = KernelFn::new(
            "c",
            I32,
            vec![
                Param::buffer("in", from, BufferRole::Read, Some(1)),
                Param::buffer("out", to, BufferRole::Write, Some(1)),
            ],
            vec![LocalDecl::scalar(from)],
            Block(vec![
                Stmt::Load {
                    local: 0,
                    place: at_tid(0),
                },
                Stmt::Store {
                    place: at_tid(1),
                    value: ScalarExpr::Cast(
                        Box::new(ScalarExpr::Local(0)),
                        cs(to, Saturation::Clamp),
                    ),
                },
            ]),
        );
        k.launch = Some(LaunchShape::new([1, 1, 1], [1, 1, 1]));
        check_kernel(&k)
    };
    assert_eq!(kernel(I64, I128), Ok(()), "the integer control is accepted");
    assert!(matches!(
        kernel(ElementKind::F32, I128),
        Err(KernelError::Type(_))
    ));
    assert!(matches!(
        kernel(I256, ElementKind::F64),
        Err(KernelError::Type(_))
    ));
}
