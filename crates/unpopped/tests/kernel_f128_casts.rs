// SPDX-License-Identifier: MIT OR Apache-2.0
//! `F128` conversions in a kernel (ROADMAP E3 slice 2b): `Cast` to and from `F32`, `F64` and the integers, each against the independent
//! reference rows of `tests/data/soft_f128_vectors.txt` (**UNVERIFIED against a third-party binary128 library**).

use unpopped::exec::LaunchShape;
use unpopped::ir::{CastSpec, RoundMode, Saturation, ScalarExpr};
use unpopped::kernel::{Block, BufferRole, KernelFn, LocalDecl, Param, Stmt, check_kernel};
use unpopped::oracle::kernel_launch::run_launch;
use unpopped_vocab::ElementKind::{self, F32, F64, F128, I8, I64};

#[path = "support/f128.rs"]
#[allow(
    dead_code,
    reason = "each F128 test file uses some of the shared helpers"
)]
mod support;
use support::*;

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
