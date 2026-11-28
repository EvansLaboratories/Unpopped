// SPDX-License-Identifier: MIT OR Apache-2.0
//! The OCP FP8 formats as kernel-layer cast endpoints (ROADMAP U68): `Fp8E4M3FN` and `Fp8E5M2` buffers a kernel loads and stores, and
//! the `Cast` between each and `F32`. The conversion is checked against the formats' own definitions (hand-worked codes: E4M3FN has
//! bias 7, no infinity, one NaN per sign `S.1111.111`, largest finite `0x7E` = 448; E5M2 has bias 15, IEEE infinity and NaN, largest
//! finite `0x7B` = 57344), and the decode side runs through the oracle's own independent FP8 decoder (`float_codec`), so the encoder
//! in `cast_ref` and the decoder in the oracle are two derivations of one format, not one.

use unpopped::exec::{ExecId, LaunchShape};
use unpopped::ir::{CastSpec, RoundMode, Saturation, ScalarExpr};
use unpopped::kernel::{Block, BufferRole, KernelFn, LocalDecl, Param, Place, Stmt, check_kernel};
use unpopped::oracle::TypedBuffer;
use unpopped::oracle::kernel_eval::KernelArg;
use unpopped::oracle::kernel_launch::run_launch;
use unpopped_vocab::ElementKind::{self, F32, Fp8E4M3FN, Fp8E5M2};

fn at(buffer: u32) -> Place {
    Place::Logical {
        buffer,
        coords: vec![ScalarExpr::Exec(ExecId::ThreadIdx(0))],
    }
}

fn cast(to: ElementKind, round: RoundMode, sat: Saturation) -> impl Fn(ScalarExpr) -> ScalarExpr {
    move |x| ScalarExpr::Cast(Box::new(x), CastSpec { to, round, sat })
}

/// `out[t] = convert(in[t])` where `in` is a `from` buffer, `out` a `to` buffer, one thread per element.
fn convert(
    (from, to): (ElementKind, ElementKind),
    round: RoundMode,
    sat: Saturation,
    lanes: u32,
) -> KernelFn {
    KernelFn::new(
        "fp8",
        F32,
        vec![
            Param::buffer("in", from, BufferRole::Read, Some(1)),
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
                value: cast(to, round, sat)(ScalarExpr::Local(0)),
            },
        ]),
    )
    .with_launch(LaunchShape::new([lanes, 1, 1], [1, 1, 1]))
}

fn run(k: &KernelFn, from: ElementKind, input: Vec<u8>, to: ElementKind, n: usize) -> Vec<u8> {
    assert_eq!(check_kernel(k), Ok(()), "well formed");
    let width = |t| if t == F32 { 4 } else { 1 };
    let out_len = n * width(to);
    let mut args = vec![
        KernelArg::Buffer(TypedBuffer::new(from, vec![n as i64], vec![1], input)),
        KernelArg::Buffer(TypedBuffer::new(
            to,
            vec![n as i64],
            vec![1],
            vec![0; out_len],
        )),
    ];
    run_launch(k, &mut args).unwrap_or_else(|e| panic!("{e:?}"));
    let KernelArg::Buffer(out) = &args[1] else {
        unreachable!()
    };
    out.bytes.clone()
}

fn f32s(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

fn encode(to: ElementKind, round: RoundMode, sat: Saturation, v: &[f32]) -> Vec<u8> {
    run(
        &convert((F32, to), round, sat, v.len() as u32),
        F32,
        f32s(v),
        to,
        v.len(),
    )
}

fn decode(from: ElementKind, codes: &[u8]) -> Vec<f32> {
    let k = convert(
        (from, F32),
        RoundMode::NearestEven,
        Saturation::Wrap,
        codes.len() as u32,
    );
    run(&k, from, codes.to_vec(), F32, codes.len())
        .chunks(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect()
}

/// `2^e` as an `f32` (exact), so the tests name the powers of two they mean instead of a long decimal.
fn two_pow(e: i32) -> f32 {
    2f32.powi(e)
}

const NEAREST: RoundMode = RoundMode::NearestEven;
const WRAP: Saturation = Saturation::Wrap;
const CLAMP: Saturation = Saturation::ClampFinite;

#[test]
fn e4m3fn_encodes_the_hand_worked_codes() {
    // 1.0 = 0x38 (exp 7, mant 0); 448 = 0x7E; 2^-6 = 0x08 (smallest normal); 2^-9 = 0x01 (smallest subnormal); -1.5 = 0xBC; 0 and -0
    let v = [1.0, 448.0, 0.015625, 0.001953125, -1.5, 0.0, -0.0];
    assert_eq!(
        encode(Fp8E4M3FN, NEAREST, WRAP, &v),
        vec![0x38, 0x7E, 0x08, 0x01, 0xBC, 0x00, 0x80]
    );
    // a tie rounds to even: 2^-10 is half of the smallest subnormal (0 is even), 3 * 2^-10 is a tie between 0x01 and 0x02 (0x02 is even)
    assert_eq!(
        encode(Fp8E4M3FN, NEAREST, WRAP, &[0.0009765625, 0.0029296875]),
        vec![0x00, 0x02]
    );
}

#[test]
fn e4m3fn_overflow_is_nan_unless_clamped_and_has_no_infinity() {
    // 464 is the tie between 448 (0x7E, mant 110, even) and the NaN-valued 480 (0x7F): ties to even keeps 448; above it overflows
    let v = [
        464.0,
        465.0,
        1.0e9,
        f32::INFINITY,
        f32::NEG_INFINITY,
        f32::NAN,
        -465.0,
    ];
    assert_eq!(
        encode(Fp8E4M3FN, NEAREST, WRAP, &v),
        vec![0x7E, 0x7F, 0x7F, 0x7F, 0xFF, 0x7F, 0xFF],
        "Wrap: an overflow (infinities included) is the NaN of its sign"
    );
    assert_eq!(
        encode(Fp8E4M3FN, NEAREST, CLAMP, &v),
        vec![0x7E, 0x7E, 0x7E, 0x7E, 0xFE, 0x7F, 0xFE],
        "ClampFinite: the largest finite value of its sign; a NaN stays NaN"
    );
    // a directed mode pointing away from the overflow stops at 448 instead of producing the NaN
    assert_eq!(
        encode(Fp8E4M3FN, RoundMode::TowardZero, WRAP, &[500.0, -500.0]),
        vec![0x7E, 0xFE]
    );
}

#[test]
fn e5m2_encodes_with_ieee_infinity_and_nan() {
    // 1.0 = 0x3C; 57344 = 0x7B; 2^-14 = 0x04 (smallest normal); 2^-16 = 0x01; -2 = 0xC0
    let v = [1.0, 57344.0, two_pow(-14), two_pow(-16), -2.0];
    assert_eq!(
        encode(Fp8E5M2, NEAREST, WRAP, &v),
        vec![0x3C, 0x7B, 0x04, 0x01, 0xC0]
    );
    // 61440 is the tie between 57344 (0x7B, odd mantissa) and 65536 (infinity): ties to even is the infinity
    let big = [61440.0, 1.0e9, f32::INFINITY, f32::NEG_INFINITY];
    assert_eq!(
        encode(Fp8E5M2, NEAREST, WRAP, &big),
        vec![0x7C, 0x7C, 0x7C, 0xFC],
        "Wrap: overflow is infinity"
    );
    assert_eq!(
        encode(Fp8E5M2, NEAREST, CLAMP, &big),
        vec![0x7B, 0x7B, 0x7B, 0xFB],
        "ClampFinite: the largest finite value"
    );
    assert!(decode(Fp8E5M2, &encode(Fp8E5M2, NEAREST, WRAP, &[f32::NAN]))[0].is_nan());
}

#[test]
fn every_code_decodes_through_the_oracles_independent_decoder_and_re_encodes_to_itself() {
    // all 256 codes: decode (the oracle's float_codec) then encode (cast_ref) must be the identity, except a NaN, which stays a NaN of
    // its sign (E4M3FN: 0x7F/0xFF; E5M2: every NaN code collapses to the quiet 0x7F/0xFF)
    for fmt in [Fp8E4M3FN, Fp8E5M2] {
        let codes: Vec<u8> = (0..=255u8).collect();
        let back = encode(fmt, NEAREST, WRAP, &decode(fmt, &codes));
        for (c, b) in codes.iter().zip(&back) {
            let nan = if fmt == Fp8E4M3FN {
                c & 0x7F == 0x7F
            } else {
                c & 0x7C == 0x7C && c & 0x03 != 0
            };
            if nan {
                assert_eq!(b & 0x80, c & 0x80, "{fmt:?} {c:#04x}: a NaN keeps its sign");
                assert!(
                    decode(fmt, &[*b])[0].is_nan(),
                    "{fmt:?} {c:#04x}: still a NaN"
                );
            } else {
                assert_eq!(b, c, "{fmt:?} {c:#04x} did not round-trip");
            }
        }
    }
}

#[test]
fn the_decoder_values_are_the_formats_documented_extremes() {
    assert_eq!(
        decode(Fp8E4M3FN, &[0x7E, 0x01, 0x08, 0xFE]),
        vec![448.0, 0.001953125, 0.015625, -448.0]
    );
    assert_eq!(
        decode(Fp8E5M2, &[0x7B, 0x01, 0x04, 0x7C]),
        vec![57344.0, two_pow(-16), two_pow(-14), f32::INFINITY]
    );
}

#[test]
fn the_other_rounding_modes_round_toward_their_side() {
    // 1.0625 is between 1.0 (0x38) and 1.125 (0x39) in E4M3FN: halfway; away from it each directed mode picks a side
    let x = [1.03125f32];
    assert_eq!(
        encode(Fp8E4M3FN, RoundMode::TowardZero, WRAP, &x),
        vec![0x38]
    );
    assert_eq!(
        encode(Fp8E4M3FN, RoundMode::TowardPosInf, WRAP, &x),
        vec![0x39]
    );
    assert_eq!(
        encode(Fp8E4M3FN, RoundMode::TowardPosInf, WRAP, &[-1.03125]),
        vec![0xB8]
    );
    assert_eq!(
        encode(Fp8E4M3FN, RoundMode::TowardNegInf, WRAP, &[-1.03125]),
        vec![0xB9]
    );
}
