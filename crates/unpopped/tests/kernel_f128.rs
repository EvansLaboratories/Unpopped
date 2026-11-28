// SPDX-License-Identifier: MIT OR Apache-2.0
//! `F128` as a kernel dtype (ROADMAP E3 slice 2b): the oracle runs a kernel over binary128 buffers with `soft_f128`. Each operation
//! is a row checked against the independent reference of `tests/data/soft_f128_vectors.txt` (`tools/gen_f128_vectors.py`, **UNVERIFIED
//! against a third-party binary128 library**), through a kernel (one thread per row) rather than by calling `soft_f128`, so the
//! wiring (load, the node, the store, the cast) is what is under test. The refusals are by name, never a panic.

use unpopped::exec::{ExecId, LaunchShape};
use unpopped::ir::{BinaryOp, CastSpec, RoundMode, Saturation, ScalarExpr, UnaryOp};
use unpopped::kernel::{Block, BufferRole, KernelFn, LocalDecl, Param, Place, Stmt, check_kernel};
use unpopped::oracle::TypedBuffer;
use unpopped::oracle::kernel_eval::KernelArg;
use unpopped::oracle::kernel_launch::run_launch;
use unpopped_vocab::ElementKind::{self, F32, F64, F128, I8, I64};

const DATA: &str = include_str!("data/soft_f128_vectors.txt");
const ONE: u128 = 0x3fff << 112;

fn hex(s: &str) -> u128 {
    u128::from_str_radix(s, 16).expect("hex")
}

/// The `(a, b, expected)` columns of the rows of `op`.
fn rows(op: &str) -> Vec<(u128, u128, String)> {
    DATA.lines()
        .filter(|l| l.starts_with(op) && l[op.len()..].starts_with(' '))
        .map(|l| {
            let mut it = l.split_whitespace().skip(1);
            let (a, b) = (hex(it.next().unwrap()), hex(it.next().unwrap()));
            (a, b, it.next().unwrap().to_string())
        })
        .collect()
}

fn at(buffer: u32) -> Place {
    Place::Logical {
        buffer,
        coords: vec![ScalarExpr::Exec(ExecId::ThreadIdx(0))],
    }
}

fn buf(dt: ElementKind, bytes: Vec<u8>, n: usize) -> KernelArg {
    KernelArg::Buffer(TypedBuffer::new(dt, vec![n as i64], vec![1], bytes))
}

fn le16(v: &[u128]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

fn out_bytes(args: &[KernelArg], i: usize) -> Vec<u8> {
    let KernelArg::Buffer(b) = &args[i] else {
        unreachable!()
    };
    b.bytes.clone()
}

fn words(bytes: &[u8]) -> Vec<u128> {
    bytes
        .chunks(16)
        .map(|c| u128::from_le_bytes(c.try_into().unwrap()))
        .collect()
}

/// `out[t] = f(a[t], b[t])` over `F128` buffers, `n` threads.
fn binary(n: usize, f: impl Fn(ScalarExpr, ScalarExpr) -> ScalarExpr) -> KernelFn {
    KernelFn::new(
        "f128",
        F32,
        vec![
            Param::buffer("a", F128, BufferRole::Read, Some(1)),
            Param::buffer("b", F128, BufferRole::Read, Some(1)),
            Param::buffer("out", F128, BufferRole::Write, Some(1)),
        ],
        vec![LocalDecl::scalar(F128), LocalDecl::scalar(F128)],
        Block(vec![
            Stmt::Load {
                local: 0,
                place: at(0),
            },
            Stmt::Load {
                local: 1,
                place: at(1),
            },
            Stmt::Store {
                place: at(2),
                value: f(ScalarExpr::Local(0), ScalarExpr::Local(1)),
            },
        ]),
    )
    .with_launch(LaunchShape::new([n as u32, 1, 1], [1, 1, 1]))
}

fn run_binary(k: &KernelFn, a: &[u128], b: &[u128]) -> Vec<u128> {
    assert_eq!(check_kernel(k), Ok(()), "well formed");
    let n = a.len();
    let mut args = vec![
        buf(F128, le16(a), n),
        buf(F128, le16(b), n),
        buf(F128, vec![0; 16 * n], n),
    ];
    run_launch(k, &mut args).unwrap_or_else(|e| panic!("{e:?}"));
    words(&out_bytes(&args, 2))
}

/// `7.0` as an `F128`: a literal has the compute type (`F32` here), so it is cast.
fn seven_f128() -> ScalarExpr {
    ScalarExpr::Cast(
        Box::new(ScalarExpr::Const(7.0)),
        CastSpec {
            to: F128,
            round: RoundMode::NearestEven,
            sat: Saturation::Wrap,
        },
    )
}

fn bx(e: ScalarExpr) -> Box<ScalarExpr> {
    Box::new(e)
}

fn check_op(op: &str, f: impl Fn(ScalarExpr, ScalarExpr) -> ScalarExpr) {
    let rows = rows(op);
    assert!(rows.len() > 150, "{op}: {} rows", rows.len());
    let (a, b): (Vec<u128>, Vec<u128>) = rows.iter().map(|r| (r.0, r.1)).unzip();
    let got = run_binary(&binary(rows.len(), f), &a, &b);
    let bad: Vec<String> = rows
        .iter()
        .zip(&got)
        .filter(|(r, g)| format!("{g:x}") != r.2)
        .map(|(r, g)| format!("{op} {:x} {:x}: got {g:x}, want {}", r.0, r.1, r.2))
        .collect();
    assert!(
        bad.is_empty(),
        "{} of {} differ, first: {:?}",
        bad.len(),
        rows.len(),
        &bad[..bad.len().min(3)]
    );
}

#[test]
fn add_sub_mul_div_run_through_a_kernel_against_the_independent_reference() {
    check_op("add", |x, y| ScalarExpr::Add(bx(x), bx(y)));
    check_op("sub", |x, y| ScalarExpr::Sub(bx(x), bx(y)));
    check_op("mul", |x, y| ScalarExpr::Mul(bx(x), bx(y)));
    check_op("div", |x, y| ScalarExpr::Div(bx(x), bx(y)));
}

#[test]
fn a_copy_moves_the_bytes_verbatim_including_a_nan_payload() {
    let odd_nan = 0x7fff_8000_0000_0000_0000_0000_dead_beef_u128;
    let a = [ONE, odd_nan, 1, 1 << 127];
    let k = binary(a.len(), |x, _| x);
    assert_eq!(run_binary(&k, &a, &a), a.to_vec());
}

#[test]
fn neg_and_abs_edit_the_sign_only() {
    let a = [
        ONE,
        1 << 127,
        0,
        0x7fff_u128 << 112,
        0xffff_u128 << 112,
        0x7fff_8000_0000_0000_0000_0000_0000_0001,
    ];
    let neg = run_binary(
        &binary(a.len(), |x, _| ScalarExpr::Unary(UnaryOp::Neg, bx(x))),
        &a,
        &a,
    );
    let abs = run_binary(
        &binary(a.len(), |x, _| ScalarExpr::Unary(UnaryOp::Abs, bx(x))),
        &a,
        &a,
    );
    let sign = 1u128 << 127;
    assert_eq!(neg, a.iter().map(|v| v ^ sign).collect::<Vec<_>>());
    assert_eq!(abs, a.iter().map(|v| v & !sign).collect::<Vec<_>>());
}

#[test]
fn comparisons_follow_ieee_with_nan_unordered_and_zeros_equal() {
    let nan = 0x7fff_8000_0000_0000_0000_0000_0000_0000_u128;
    let (one, two, minus_one) = (ONE, 0x4000_u128 << 112, ONE | 1 << 127);
    let (zero, minus_zero) = (0u128, 1u128 << 127);
    let (inf, minus_inf) = (0x7fff_u128 << 112, 0xffff_u128 << 112);
    // (a, b, [eq, ne, lt, le, gt, ge])
    let cases: &[(u128, u128, [bool; 6])] = &[
        (one, two, [false, true, true, true, false, false]),
        (two, one, [false, true, false, false, true, true]),
        (one, one, [true, false, false, true, false, true]),
        (zero, minus_zero, [true, false, false, true, false, true]),
        (minus_one, one, [false, true, true, true, false, false]),
        (nan, one, [false, true, false, false, false, false]),
        (one, nan, [false, true, false, false, false, false]),
        (nan, nan, [false, true, false, false, false, false]),
        (minus_inf, inf, [false, true, true, true, false, false]),
        (inf, inf, [true, false, false, true, false, true]),
    ];
    let ops = [
        BinaryOp::CmpEq,
        BinaryOp::CmpNe,
        BinaryOp::CmpLt,
        BinaryOp::CmpLe,
        BinaryOp::CmpGt,
        BinaryOp::CmpGe,
    ];
    let (a, b): (Vec<u128>, Vec<u128>) = cases.iter().map(|c| (c.0, c.1)).unzip();
    for (i, op) in ops.into_iter().enumerate() {
        let k = binary(a.len(), move |x, y| ScalarExpr::Binary(op, bx(x), bx(y)));
        let got = run_binary(&k, &a, &b);
        let want: Vec<u128> = cases.iter().map(|c| if c.2[i] { ONE } else { 0 }).collect();
        assert_eq!(got, want, "{op:?}");
    }
}

#[test]
fn select_takes_one_arm_by_an_f128_comparison_or_an_f128_condition() {
    let (a, b) = (
        [ONE, 0x4000_u128 << 112, 5, 7],
        [0x4000_u128 << 112, ONE, 5, 3],
    );
    // max-like: a < b ? b : a
    let k = binary(a.len(), |x, y| {
        ScalarExpr::Select(
            bx(ScalarExpr::Binary(
                BinaryOp::CmpLt,
                bx(x.clone()),
                bx(y.clone()),
            )),
            bx(y),
            bx(x),
        )
    });
    assert_eq!(run_binary(&k, &a, &b), vec![b[0], a[1], a[2], a[3]]);
    // an F128 condition is taken when it is not zero (-0 is zero; a NaN is taken)
    let cond = [
        0u128,
        1 << 127,
        ONE,
        0x7fff_8000_0000_0000_0000_0000_0000_0000,
    ];
    let k = binary(cond.len(), |c, x| {
        ScalarExpr::Select(bx(c), bx(x), bx(seven_f128()))
    });
    let seven = run_binary(&binary(1, |_, _| seven_f128()), &[0], &[0])[0];
    let got = run_binary(&k, &cond, &[ONE; 4]);
    assert_eq!(got, vec![seven, seven, ONE, ONE]);
}

/// `out[t] = (to) a[t]` for `a` of type `from`.
fn convert(from: ElementKind, to: ElementKind, spec: CastSpec, n: usize) -> KernelFn {
    KernelFn::new(
        "cvt",
        F32,
        vec![
            Param::buffer("a", from, BufferRole::Read, Some(1)),
            Param::buffer("out", to, BufferRole::Write, Some(1)),
        ],
        vec![LocalDecl::scalar(from)],
        Block(vec![
            Stmt::Load {
                local: 0,
                place: at(0),
            },
            Stmt::Store {
                place: at(1),
                value: ScalarExpr::Cast(bx(ScalarExpr::Local(0)), spec),
            },
        ]),
    )
    .with_launch(LaunchShape::new([n as u32, 1, 1], [1, 1, 1]))
}

fn spec(to: ElementKind, round: RoundMode, sat: Saturation) -> CastSpec {
    CastSpec { to, round, sat }
}

fn run_convert(
    k: &KernelFn,
    from: ElementKind,
    to: ElementKind,
    input: Vec<u8>,
    n: usize,
) -> Vec<u8> {
    assert_eq!(check_kernel(k), Ok(()), "well formed");
    let width = |t| match t {
        F128 => 16,
        F64 | I64 => 8,
        I8 => 1,
        _ => 4,
    };
    let mut args = vec![buf(from, input, n), buf(to, vec![0; width(to) * n], n)];
    run_launch(k, &mut args).unwrap_or_else(|e| panic!("{e:?}"));
    out_bytes(&args, 1)
}

#[test]
fn narrowing_to_f32_and_f64_matches_the_reference_with_one_rounding() {
    for (op, to, width) in [("tof64", F64, 8usize), ("tof32", F32, 4)] {
        let rows = rows(op);
        assert!(rows.len() > 25, "{op}");
        let a: Vec<u128> = rows.iter().map(|r| r.0).collect();
        let k = convert(
            F128,
            to,
            spec(to, RoundMode::NearestEven, Saturation::Wrap),
            a.len(),
        );
        let out = run_convert(&k, F128, to, le16(&a), a.len());
        for (i, r) in rows.iter().enumerate() {
            let mut b = [0u8; 16];
            b[..width].copy_from_slice(&out[i * width..(i + 1) * width]);
            assert_eq!(
                format!("{:x}", u128::from_le_bytes(b)),
                r.2,
                "{op} {:x}",
                r.0
            );
        }
    }
}

#[test]
fn widening_from_f32_f64_is_exact() {
    for (op, from, width) in [("fromf64", F64, 8usize), ("fromf32", F32, 4)] {
        let rows = rows(op);
        assert!(rows.len() > 25, "{op}");
        let input: Vec<u8> = rows
            .iter()
            .flat_map(|r| r.0.to_le_bytes()[..width].to_vec())
            .collect();
        let k = convert(
            from,
            F128,
            spec(F128, RoundMode::NearestEven, Saturation::Wrap),
            rows.len(),
        );
        let out = words(&run_convert(&k, from, F128, input, rows.len()));
        for (g, r) in out.iter().zip(&rows) {
            assert_eq!(format!("{g:x}"), r.2, "{op} {:x}", r.0);
        }
    }
}

/// The `i64` the reference's 256-bit expected word saturates to (hex of the two's complement, no leading zeros).
fn saturate_i64(word: &str) -> i64 {
    let padded = format!("{word:0>64}");
    let (hi, lo) = (hex(&padded[..32]), hex(&padded[32..]));
    let negative = hi >> 127 == 1;
    let fits = if negative {
        hi == u128::MAX && (lo as i128) >= i128::from(i64::MIN)
    } else {
        hi == 0 && lo <= i64::MAX as u128
    };
    match (fits, negative) {
        (true, _) => lo as i128 as i64,
        (false, true) => i64::MIN,
        (false, false) => i64::MAX,
    }
}

#[test]
fn an_integer_cast_rounds_under_each_mode_and_saturates() {
    let modes = [
        ("int_trunc", RoundMode::TowardZero),
        ("int_floor", RoundMode::TowardNegInf),
        ("int_ceil", RoundMode::TowardPosInf),
        ("int_rne", RoundMode::NearestEven),
        ("int_rna", RoundMode::NearestAway),
    ];
    for (op, mode) in modes {
        let rows = rows(op);
        assert!(rows.len() > 25, "{op}");
        let a: Vec<u128> = rows.iter().map(|r| r.0).collect();
        let k = convert(F128, I64, spec(I64, mode, Saturation::Clamp), a.len());
        let out = run_convert(&k, F128, I64, le16(&a), a.len());
        for (i, r) in rows.iter().enumerate() {
            let got = i64::from_le_bytes(out[i * 8..i * 8 + 8].try_into().unwrap());
            assert_eq!(got, saturate_i64(&r.2), "{op} {:x}", r.0);
        }
    }
}

#[test]
fn an_integer_cast_saturates_to_the_destination_and_a_nan_is_zero() {
    let big = [
        ONE + (70u128 << 112),
        (ONE + (70u128 << 112)) | 1 << 127,
        0x7fff_u128 << 112,
        0xffff_u128 << 112,
        0x7fff_8000_0000_0000_0000_0000_0000_0000,
    ];
    let k = convert(
        F128,
        I8,
        spec(I8, RoundMode::NearestEven, Saturation::Clamp),
        big.len(),
    );
    let out = run_convert(&k, F128, I8, le16(&big), big.len());
    assert_eq!(
        out,
        vec![127u8, 0x80, 127, 0x80, 0],
        "2^71 and +inf to i8::MAX, -2^71 and -inf to i8::MIN, NaN to 0"
    );
}

#[test]
fn an_integer_into_f128_rounds_to_nearest_even_past_113_bits() {
    // i64 values: 0, 1, -1, i64::MAX (63 bits: exact), i64::MIN
    let v: [i64; 5] = [0, 1, -1, i64::MAX, i64::MIN];
    let input: Vec<u8> = v.iter().flat_map(|x| x.to_le_bytes()).collect();
    let k = convert(
        I64,
        F128,
        spec(F128, RoundMode::NearestEven, Saturation::Wrap),
        v.len(),
    );
    let out = words(&run_convert(&k, I64, F128, input, v.len()));
    assert_eq!(out[0], 0);
    assert_eq!(out[1], ONE);
    assert_eq!(out[2], ONE | 1 << 127);
    // 2^63 - 1 is 1.111..1 (63 ones) x 2^62: exponent 0x3fff + 62, fraction the top 62 bits of the mantissa set
    assert_eq!(out[3] >> 112, 0x3fff + 62);
    assert_eq!(
        out[4],
        ((0x3fff + 63) << 112) | 1 << 127,
        "-2^63 is -1.0 x 2^63"
    );
}

#[test]
fn an_f128_scalar_argument_is_the_exact_binary128_of_the_f64_given() {
    // out[t] = s * s with s = 1.5: 2.25 = 1.125 * 2^1, exponent 0x4000, the fraction bit 2^-3
    let k = KernelFn::new(
        "scalar",
        F32,
        vec![
            Param::scalar("s", F128, 0),
            Param::buffer("out", F128, BufferRole::Write, Some(1)),
        ],
        vec![LocalDecl::scalar(F128)],
        Block(vec![Stmt::Store {
            place: at(1),
            value: ScalarExpr::Mul(bx(ScalarExpr::Local(0)), bx(ScalarExpr::Local(0))),
        }]),
    )
    .with_launch(LaunchShape::new([2, 1, 1], [1, 1, 1]));
    assert_eq!(check_kernel(&k), Ok(()));
    let mut args = vec![KernelArg::Float(1.5), buf(F128, vec![0; 32], 2)];
    run_launch(&k, &mut args).unwrap_or_else(|e| panic!("{e:?}"));
    let want = (0x4000u128 << 112) | (1 << 109);
    assert_eq!(words(&out_bytes(&args, 1)), vec![want, want]);
}

#[test]
fn narrowing_overflow_is_an_infinity_under_wrap_and_the_largest_finite_under_clamp_finite() {
    // 2^200 overflows f32 (max about 2^128) but not f64; 2^2000 overflows both; an infinity stays an infinity under Wrap
    let two_pow = |e: u128| (0x3fff + e) << 112;
    let a = [
        two_pow(200),
        two_pow(2000),
        0x7fff_u128 << 112,
        0xffff_u128 << 112,
        two_pow(200) | 1 << 127,
    ];
    let to_f32 = |sat| {
        let k = convert(F128, F32, spec(F32, RoundMode::NearestEven, sat), a.len());
        run_convert(&k, F128, F32, le16(&a), a.len())
            .chunks(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
            .collect::<Vec<f32>>()
    };
    assert_eq!(
        to_f32(Saturation::Wrap),
        vec![
            f32::INFINITY,
            f32::INFINITY,
            f32::INFINITY,
            f32::NEG_INFINITY,
            f32::NEG_INFINITY
        ]
    );
    assert_eq!(
        to_f32(Saturation::ClampFinite),
        vec![f32::MAX, f32::MAX, f32::MAX, f32::MIN, f32::MIN]
    );
    let to_f64 = |sat| {
        let k = convert(F128, F64, spec(F64, RoundMode::NearestEven, sat), a.len());
        run_convert(&k, F128, F64, le16(&a), a.len())
            .chunks(8)
            .map(|c| f64::from_le_bytes(c.try_into().unwrap()))
            .collect::<Vec<f64>>()
    };
    assert_eq!(to_f64(Saturation::Wrap)[0], 2f64.powi(200));
    assert_eq!(to_f64(Saturation::Wrap)[1], f64::INFINITY);
    assert_eq!(to_f64(Saturation::ClampFinite)[1], f64::MAX);
    assert_eq!(
        to_f64(Saturation::ClampFinite)[0],
        2f64.powi(200),
        "an in-range value is untouched by the clamp"
    );
}
