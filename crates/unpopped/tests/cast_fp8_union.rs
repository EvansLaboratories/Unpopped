// SPDX-License-Identifier: MIT OR Apache-2.0
//! Plan E1b (KISS 0.24.0 union batch 1): `Cast` between `F32` and `F8E3M4` / `F8UE4M3`, run on the oracle through
//! `Cast(Cast(x, fp8), F32)`. The expected values are KISS's own decode vectors (`conformance/tests/classify_union_tokens.rs`:
//! `test_classify_f8e3m4_format`, `test_classify_ue4m3_format`, KISS `a9ad8cf`), plus an exhaustive round trip of all 256 `F8E3M4`
//! codes and all 127 finite `F8UE4M3` codes against a reference decode written here from the §6.1-0019 / §6.1-0020 format
//! statements. Rounding is round to nearest even; `F8E3M4` follows IEEE overflow (infinity under `Wrap`, the largest finite value
//! under `ClampFinite`); `F8UE4M3` saturates at 448, sends NaN to its one NaN code `0x7F` and anything at or below zero to 0
//! (the IR's rule for the last two: UNVERIFIED against a vendor conversion, stated in `cast_ref::encode_ue4m3`).

use unpopped::exec::{ExecId, LaunchShape};
use unpopped::ir::{CastSpec, RoundMode, Saturation, ScalarExpr};
use unpopped::kernel::{Block, BufferRole, KernelFn, LocalDecl, Param, Place, Stmt, check_kernel};
use unpopped::oracle::TypedBuffer;
use unpopped::oracle::kernel_eval::KernelArg;
use unpopped::oracle::kernel_launch::run_launch;
use unpopped_vocab::ElementKind;

const F32: ElementKind = ElementKind::F32;
const E3M4: ElementKind = ElementKind::F8E3M4;
const UE4M3: ElementKind = ElementKind::F8UE4M3;

fn cast(e: ScalarExpr, to: ElementKind, sat: Saturation) -> ScalarExpr {
    ScalarExpr::Cast(
        Box::new(e),
        CastSpec {
            to,
            round: RoundMode::NearestEven,
            sat,
        },
    )
}

/// `out[tid] = (f32) (fp8) in[tid]` over `n` threads.
fn round_trip_kernel(fp8: ElementKind, sat: Saturation, n: u32) -> KernelFn {
    let tid = ScalarExpr::Exec(ExecId::ThreadIdx(0));
    let at = |b: u32| Place::Logical {
        buffer: b,
        coords: vec![tid.clone()],
    };
    KernelFn::new(
        "rt",
        F32,
        vec![
            Param::buffer("in", F32, BufferRole::Read, Some(1)),
            Param::buffer("out", F32, BufferRole::ReadWrite, Some(1)),
        ],
        vec![LocalDecl::scalar(F32)],
        Block(vec![
            Stmt::Load {
                local: 0,
                place: at(0),
            },
            Stmt::Store {
                place: at(1),
                value: cast(cast(ScalarExpr::Local(0), fp8, sat), F32, Saturation::Wrap),
            },
        ]),
    )
    .with_launch(LaunchShape::new([n, 1, 1], [1, 1, 1]))
}

fn round_trip(fp8: ElementKind, sat: Saturation, input: &[f32]) -> Vec<f32> {
    let n = input.len();
    let k = round_trip_kernel(fp8, sat, n as u32);
    assert_eq!(check_kernel(&k), Ok(()), "well formed");
    let mut args = vec![
        KernelArg::Buffer(TypedBuffer::from_f32(&[n as i64], input)),
        KernelArg::Buffer(TypedBuffer::from_f32(&[n as i64], &vec![0.0; n])),
    ];
    run_launch(&k, &mut args).expect("runs");
    let KernelArg::Buffer(o) = &args[1] else {
        unreachable!()
    };
    o.to_f64_vec().into_iter().map(|v| v as f32).collect()
}

/// The `F8E3M4` code's value from the format statement: sign 1, exponent 3 (bias 3), mantissa 4, infinity at exponent 7 with
/// mantissa 0, NaN above.
fn e3m4(code: u8) -> f32 {
    let sign = if code & 0x80 != 0 { -1.0f32 } else { 1.0 };
    let e = i32::from((code >> 4) & 7);
    let m = f32::from(code & 15);
    sign * match e {
        0 => m * 2f32.powi(1 - 3 - 4),
        7 if m == 0.0 => f32::INFINITY,
        7 => f32::NAN,
        _ => (1.0 + m / 16.0) * 2f32.powi(e - 3),
    }
}

/// The `F8UE4M3` code's value (code at most 0x7F): exponent 4 (bias 7), mantissa 3, no sign, `0x7F` NaN.
fn ue4m3(code: u8) -> f32 {
    assert!(code <= 0x7F);
    if code == 0x7F {
        return f32::NAN;
    }
    let e = i32::from(code >> 3);
    let m = f32::from(code & 7);
    if e == 0 {
        m * 2f32.powi(1 - 7 - 3)
    } else {
        (1.0 + m / 8.0) * 2f32.powi(e - 7)
    }
}

#[test]
fn the_reference_decode_agrees_with_kiss_s_pinned_vectors() {
    // KISS `test_classify_f8e3m4_format` and `test_classify_ue4m3_format`
    for (code, want) in [
        (0x00u8, 0.0f32),
        (0x01, 0.015625),
        (0x0F, 0.234375),
        (0x10, 0.25),
        (0x30, 1.0),
        (0x6F, 15.5),
        (0xEF, -15.5),
    ] {
        assert_eq!(e3m4(code), want, "e3m4 {code:#04x}");
    }
    assert!(e3m4(0x80).is_sign_negative() && e3m4(0x80) == 0.0);
    assert_eq!(e3m4(0x70), f32::INFINITY);
    assert_eq!(e3m4(0xF0), f32::NEG_INFINITY);
    for (code, want) in [
        (0x00u8, 0.0f32),
        (0x01, 0.001953125),
        (0x08, 0.015625),
        (0x38, 1.0),
        (0x7E, 448.0),
    ] {
        assert_eq!(ue4m3(code), want, "ue4m3 {code:#04x}");
    }
    assert!(ue4m3(0x7F).is_nan());
}

#[test]
fn every_f8e3m4_code_round_trips_through_the_oracle() {
    let finite: Vec<u8> = (0..=255u8).filter(|c| e3m4(*c).is_finite()).collect();
    let input: Vec<f32> = finite.iter().map(|c| e3m4(*c)).collect();
    let out = round_trip(E3M4, Saturation::Wrap, &input);
    for ((code, want), got) in finite.iter().zip(&input).zip(&out) {
        assert_eq!(
            got.to_bits(),
            want.to_bits(),
            "e3m4 code {code:#04x}: {want} came back as {got}"
        );
    }
    // the infinities and a NaN survive too
    let out = round_trip(
        E3M4,
        Saturation::Wrap,
        &[f32::INFINITY, f32::NEG_INFINITY, f32::NAN],
    );
    assert_eq!(out[0], f32::INFINITY);
    assert_eq!(out[1], f32::NEG_INFINITY);
    assert!(out[2].is_nan());
}

#[test]
fn f8e3m4_rounds_to_nearest_even_and_overflows_like_ieee() {
    // 1 + 1/32 is half way between 1.0 (code 0x30, even mantissa 0) and 1 + 1/16: ties to even gives 1.0; 1 + 3/32 is half way
    // between mantissa 1 and 2: ties to even gives mantissa 2 = 1 + 1/8
    let out = round_trip(
        E3M4,
        Saturation::Wrap,
        &[1.0 + 1.0 / 32.0, 1.0 + 3.0 / 32.0],
    );
    assert_eq!(out, [1.0, 1.125]);
    // overflow: 15.5 is the largest finite value; 15.74 still rounds down to it, 15.75 and beyond round to infinity under Wrap
    let out = round_trip(E3M4, Saturation::Wrap, &[15.74, 15.75, 100.0, -100.0]);
    assert_eq!(out, [15.5, f32::INFINITY, f32::INFINITY, f32::NEG_INFINITY]);
    // ClampFinite stops at the largest finite value instead; NaN stays NaN
    let out = round_trip(
        E3M4,
        Saturation::ClampFinite,
        &[15.75, 100.0, -100.0, f32::INFINITY, f32::NAN],
    );
    assert_eq!(&out[..4], &[15.5, 15.5, -15.5, 15.5]);
    assert!(out[4].is_nan());
}

#[test]
fn every_finite_f8ue4m3_code_round_trips_through_the_oracle() {
    let finite: Vec<u8> = (0..0x7Fu8).collect();
    let input: Vec<f32> = finite.iter().map(|c| ue4m3(*c)).collect();
    let out = round_trip(UE4M3, Saturation::ClampFinite, &input);
    for ((code, want), got) in finite.iter().zip(&input).zip(&out) {
        assert_eq!(
            got.to_bits(),
            want.to_bits(),
            "ue4m3 code {code:#04x}: {want} came back as {got}"
        );
    }
}

#[test]
fn f8ue4m3_saturates_at_448_sends_nan_to_its_code_and_negatives_to_zero() {
    let out = round_trip(
        UE4M3,
        Saturation::ClampFinite,
        &[
            448.0,
            460.0,
            1.0e6,
            f32::INFINITY,
            -1.0,
            -0.0,
            f32::NEG_INFINITY,
            0.001_953_125,
            0.000_976_562_5,
            0.003_906_25 * 1.5,
        ],
    );
    // 448 stays; 460 (above the largest finite value) and beyond, and infinity, saturate at 448; a negative value and -0 are 0;
    // the minimum subnormal 2^-9 is exact; half of it ties to even (0); 1.5 * 2^-8 = 3 * 2^-9 is an exact subnormal
    assert_eq!(
        &out[..8],
        &[448.0, 448.0, 448.0, 448.0, 0.0, 0.0, 0.0, 0.001_953_125]
    );
    assert_eq!(
        out[8], 0.0,
        "half of the minimum subnormal ties to even, which is zero"
    );
    assert_eq!(
        out[9], 0.005_859_375,
        "3 * 2^-9 is an exact subnormal and survives"
    );
    let nan = round_trip(UE4M3, Saturation::ClampFinite, &[f32::NAN]);
    assert!(nan[0].is_nan());
}

#[test]
fn an_integer_casts_into_f8ue4m3_and_f8e3m4() {
    // `(f32) (fp8) 7` and `(f32) (fp8) 1000` (an integer constant): 7 is exact in both; 1000 saturates at 448 in f8ue4m3 and is
    // infinity in f8e3m4 under Wrap
    for (fp8, sat, value, want) in [
        (UE4M3, Saturation::ClampFinite, 7, 7.0f32),
        (UE4M3, Saturation::ClampFinite, 1000, 448.0),
        (E3M4, Saturation::Wrap, 7, 7.0),
        (E3M4, Saturation::Wrap, 1000, f32::INFINITY),
    ] {
        let k = KernelFn::new(
            "int",
            F32,
            vec![Param::buffer("out", F32, BufferRole::ReadWrite, Some(1))],
            vec![],
            Block(vec![Stmt::Store {
                place: Place::Logical {
                    buffer: 0,
                    coords: vec![ScalarExpr::ConstInt(0, ElementKind::I64)],
                },
                value: cast(
                    cast(ScalarExpr::ConstInt(value, ElementKind::I32), fp8, sat),
                    F32,
                    Saturation::Wrap,
                ),
            }]),
        )
        .with_launch(LaunchShape::new([1, 1, 1], [1, 1, 1]));
        assert_eq!(check_kernel(&k), Ok(()), "{fp8:?} {value}");
        let mut args = vec![KernelArg::Buffer(TypedBuffer::from_f32(&[1], &[0.0]))];
        run_launch(&k, &mut args).expect("runs");
        let KernelArg::Buffer(o) = &args[0] else {
            unreachable!()
        };
        assert_eq!(o.to_f64_vec()[0] as f32, want, "{fp8:?} {value}");
    }
}

#[test]
fn a_wrap_cast_into_f8ue4m3_is_refused() {
    use unpopped::types::TypeError;
    // the control: a Cast into f8e3m4 takes Wrap; into f8ue4m3 only ClampFinite
    let wrap = round_trip_kernel(UE4M3, Saturation::Wrap, 1);
    assert!(
        matches!(
            check_kernel(&wrap),
            Err(unpopped::kernel::KernelError::Type(
                TypeError::IllegalCast { .. }
            ))
        ),
        "{:?}",
        check_kernel(&wrap)
    );
    assert_eq!(
        check_kernel(&round_trip_kernel(E3M4, Saturation::Wrap, 1)),
        Ok(())
    );
    assert_eq!(
        check_kernel(&round_trip_kernel(UE4M3, Saturation::ClampFinite, 1)),
        Ok(())
    );
}
