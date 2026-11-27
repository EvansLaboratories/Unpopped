// SPDX-License-Identifier: MIT OR Apache-2.0
//! **Blockwise-scaled FP8 GEMM** (CUTLASS survey row 24; the kernel is `support/fp8_blockwise.rs`): `A` scaled 1x128 (per row, per
//! 128-block of `K`), `B` scaled 128x128, FP8-valued `F32` operands (the oracle does not compute an FP8 dtype yet: finding), `F32` scales and accumulation. The operands are built as FP8 CODES
//! (bytes) and DECODED with the test's own `E3M4` formula (1 sign, 3 exponent bits with bias 3, 4 mantissa bits,
//! subnormals at exponent 0, infinity and NaN at exponent 7), so it does not use the oracle's codec. Three corpora: ordinary,
//! all-negative, and a NaN code in one `A` row that must reach exactly that output row. Mutants (swapped K-block scales, a
//! parity-chosen `B` scale block) fail by value.

use unpopped::kernel::check_kernel;
use unpopped::oracle::TypedBuffer;
use unpopped::oracle::kernel_eval::KernelArg;
use unpopped::oracle::kernel_launch::run_launch;
use unpopped_vocab::ElementKind;

#[path = "support/fp8_blockwise.rs"]
mod fixture;
use fixture::{BLK, F32, K, KB, M, Mutant, N, NB, fp8_blockwise_gemm};

/// The value of an `E3M4` code.
fn decode(code: u8) -> f32 {
    let (sign, exp, man) = (code >> 7, (code >> 4) & 7, code & 15);
    let mag = match exp {
        7 if man == 0 => f32::INFINITY,
        7 => return f32::NAN,
        0 => f32::from(man) / 16.0 * 2f32.powi(-2),
        _ => (1.0 + f32::from(man) / 16.0) * 2f32.powi(i32::from(exp) - 3),
    };
    if sign == 1 { -mag } else { mag }
}

struct Case {
    a: Vec<u8>,
    b: Vec<u8>,
    sfa: Vec<f32>,
    sfb: Vec<f32>,
}

/// Codes with exponents 1..=5 (values 0.25 to 31) and every mantissa, signs varying.
fn code(i: usize, salt: usize) -> u8 {
    let (exp, man) = (1 + (i * 5 + salt) % 5, (i * 3 + salt / 2) % 16);
    let sign = usize::from((i / 7 + salt) % 3 == 0);
    (sign << 7 | exp << 4 | man) as u8
}

fn ordinary() -> Case {
    let scales = [0.5f32, 1.0, 2.0, 0.25, 4.0];
    Case {
        a: (0..M * K).map(|i| code(i, 1)).collect(),
        b: (0..N * K).map(|i| code(i, 3)).collect(),
        sfa: (0..M * KB).map(|i| scales[i % 5]).collect(),
        sfb: (0..NB * KB).map(|i| scales[(i * 2 + 1) % 5]).collect(),
    }
}

fn negative() -> Case {
    let mut c = ordinary();
    c.a.iter_mut().for_each(|x| *x |= 0x80);
    c.b.iter_mut().for_each(|x| *x &= 0x7f);
    c
}

fn nan() -> Case {
    let mut c = ordinary();
    c.a[K + 150] = 0x7f; // exponent 7, mantissa 15: a NaN
    c
}

fn reference(c: &Case) -> Vec<f32> {
    let mut out = vec![0.0; M * N];
    for m in 0..M {
        for n in 0..N {
            let mut acc = 0.0f32;
            for kb in 0..KB {
                let mut inner = 0.0f32;
                for j in 0..BLK {
                    let k = kb * BLK + j;
                    inner += decode(c.a[m * K + k]) * decode(c.b[n * K + k]);
                }
                acc += c.sfa[m * KB + kb] * c.sfb[n / BLK * KB + kb] * inner;
            }
            out[m * N + n] = acc;
        }
    }
    out
}

fn bytes(kind: ElementKind, v: Vec<u8>) -> KernelArg {
    KernelArg::Buffer(TypedBuffer::new(kind, vec![v.len() as i64], vec![1], v))
}

fn floats(v: &[f32]) -> KernelArg {
    bytes(F32, v.iter().flat_map(|x| x.to_le_bytes()).collect())
}

fn decoded(codes: &[u8]) -> Vec<f32> {
    codes.iter().map(|c| decode(*c)).collect()
}

fn run(c: &Case, mutant: Mutant) -> Vec<f32> {
    let k = fp8_blockwise_gemm(mutant);
    assert_eq!(check_kernel(&k), Ok(()));
    let mut args = [
        floats(&decoded(&c.a)),
        floats(&decoded(&c.b)),
        floats(&c.sfa),
        floats(&c.sfb),
        floats(&vec![0.0; M * N]),
    ];
    run_launch(&k, &mut args).expect("the oracle runs it");
    let KernelArg::Buffer(b) = &args[4] else {
        unreachable!("a buffer")
    };
    b.bytes
        .chunks(4)
        .map(|v| f32::from_le_bytes(v.try_into().unwrap()))
        .collect()
}

fn same(got: &[f32], want: &[f32]) -> bool {
    got.len() == want.len()
        && got
            .iter()
            .zip(want)
            .all(|(g, w)| (g.is_nan() && w.is_nan()) || (g - w).abs() <= 1e-5 * w.abs().max(1.0))
}

#[test]
fn the_blockwise_fp8_gemm_matches_the_decoded_reference_on_every_corpus() {
    for (name, c) in [
        ("ordinary", ordinary()),
        ("negative", negative()),
        ("nan", nan()),
    ] {
        assert!(same(&run(&c, Mutant::None), &reference(&c)), "{name}");
    }
}

#[test]
fn the_decoder_and_the_corpus_are_not_blind() {
    // the format's own anchors: 1.0 is exponent 3, the largest finite value is exponent 6 mantissa 15, exponent 7 is inf or NaN,
    // and the smallest subnormal is 2^-6
    assert_eq!(decode(0b0_011_0000), 1.0);
    assert_eq!(decode(0b0_110_1111), 15.5);
    assert_eq!(decode(0b0_111_0000), f32::INFINITY);
    assert!(decode(0x7f).is_nan());
    assert_eq!(decode(0x01), 2f32.powi(-6));
    assert_eq!(decode(0x80 | 0b0_011_1000), -1.5);
    // the NaN reaches exactly output row 1
    let want = reference(&nan());
    let rows: Vec<usize> = (0..M)
        .filter(|m| want[m * N..(m + 1) * N].iter().any(|x| x.is_nan()))
        .collect();
    assert_eq!(rows, vec![1]);
    // both scale blocks of B and both K-blocks matter
    let mut c = ordinary();
    c.sfb[KB] = 8.0;
    assert!(
        !same(&reference(&c), &reference(&ordinary())),
        "the second N-block's scale"
    );
}

#[test]
fn the_scale_block_mutants_fail_by_value() {
    let c = ordinary();
    let want = reference(&c);
    for m in [Mutant::SwappedKBlock, Mutant::NBlockByParity] {
        assert!(!same(&run(&c, m), &want), "{m:?}");
    }
}
