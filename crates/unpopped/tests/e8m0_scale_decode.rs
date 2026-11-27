// SPDX-License-Identifier: MIT OR Apache-2.0
//! **The oracle decodes an `F8E8M0` MX block scale** (CUDA audit row 60). E8M0 is exponent-only, unsigned, bias 127: code `c`
//! is `2^(c - 127)`, and `0xFF` is NaN; there is no zero and no infinity (OCP Microscaling Formats Specification v1.0; the ONNX
//! `float8` note's E8M0 table). It is a scale type, never a compute dtype (KISS-CLASSIFY 6.1-0013), so the plan gate still
//! refuses it as one; this only reads a scale operand's value. The expected values are built by repeated doubling, not by
//! the decoder's bit construction.

use unpopped::oracle::TypedBuffer;
use unpopped_vocab::ElementKind;

fn decode(codes: &[u8]) -> Vec<f64> {
    TypedBuffer::new(
        ElementKind::F8E8M0,
        vec![codes.len() as i64],
        vec![1],
        codes.to_vec(),
    )
    .to_f64_vec()
}

/// `2^e` by repeated doubling or halving (exact in f64 over this range).
fn pow2(e: i32) -> f64 {
    let mut v = 1.0f64;
    for _ in 0..e.unsigned_abs() {
        v = if e >= 0 { v * 2.0 } else { v / 2.0 };
    }
    v
}

#[test]
fn every_code_decodes_to_two_to_the_code_minus_127_and_ff_is_nan() {
    let all: Vec<u8> = (0..=255u8).collect();
    let got = decode(&all);
    for c in 0..=254usize {
        assert_eq!(got[c], pow2(c as i32 - 127), "code {c}");
    }
    assert!(got[255].is_nan(), "0xFF is NaN");
    // the corners by hand: 127 -> 1, 128 -> 2, 126 -> 0.5; code 0 is the smallest scale, not zero
    assert_eq!((got[127], got[128], got[126]), (1.0, 2.0, 0.5));
    assert!(got[0] > 0.0 && got[0] == pow2(-127));
}

#[test]
fn the_control_an_fp8_e5m2_code_still_decodes_as_a_float() {
    // 0x3C is 1.0 in E5M2 (bias 15, exponent 15): the same path decodes the computable FP8 dtypes
    let b = TypedBuffer::new(ElementKind::Fp8E5M2, vec![1], vec![1], vec![0x3C]);
    assert_eq!(b.to_f64_vec(), vec![1.0]);
}
