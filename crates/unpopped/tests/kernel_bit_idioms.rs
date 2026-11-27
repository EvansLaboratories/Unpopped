// SPDX-License-Identifier: MIT OR Apache-2.0
//! **Carry-less multiply and four-lane byte dot product from leaves already on main** (survey rows 20 and 28). Run on the oracle
//! against references written independently: the carry-less product by the parity of the pairs of set bits whose indices sum to
//! each output bit (not by the kernel's shift-and-XOR loop), the byte dot product with Rust's `as i8` and `as u8` casts. The
//! corpora are boundary values (zero, all ones, a high bit alone, adjacent bits that would carry, signed extremes). Integers
//! have no NaN, so the "non-blind" check is that each mutant changes the result on the corpus.

use unpopped::kernel::{KernelFn, check_kernel};
use unpopped::oracle::TypedBuffer;
use unpopped::oracle::kernel_eval::KernelArg;
use unpopped::oracle::kernel_launch::run_launch;

#[path = "support/bit_idioms.rs"]
mod fixture;
use fixture::{Mutant, N, U32, clmul, dp4a};

fn buf(v: &[u32]) -> KernelArg {
    KernelArg::Buffer(TypedBuffer::new(
        U32,
        vec![v.len() as i64],
        vec![1],
        v.iter().flat_map(|x| x.to_le_bytes()).collect(),
    ))
}

fn words(a: &KernelArg) -> Vec<u32> {
    let KernelArg::Buffer(b) = a else {
        unreachable!("a buffer")
    };
    b.bytes
        .chunks(4)
        .map(|c| u32::from_le_bytes(c.try_into().unwrap()))
        .collect()
}

fn run(k: &KernelFn, inputs: &[&[u32]], outputs: usize) -> Vec<Vec<u32>> {
    assert_eq!(check_kernel(k), Ok(()));
    let mut args: Vec<KernelArg> = inputs.iter().map(|v| buf(v)).collect();
    args.extend((0..outputs).map(|_| buf(&[0; N])));
    run_launch(k, &mut args).expect("the oracle runs it");
    args[inputs.len()..].iter().map(words).collect()
}

const A: [u32; N] = [
    0,
    0xFFFF_FFFF,
    0x8000_0000,
    0x0000_0003,
    0xDEAD_BEEF,
    0x0000_0001,
    0x7FFF_FFFF,
    0x1234_5678,
];
const B: [u32; N] = [
    0xFFFF_FFFF,
    0xFFFF_FFFF,
    0x0000_0002,
    0x0000_0003,
    0xCAFE_BABE,
    0x8000_0000,
    0x0000_0001,
    0x0F0F_0F0F,
];

/// The carry-less product by coefficient parity: bit `k` of the result is the parity of the pairs `(i, j)`, `i + j = k`, with
/// bit `i` of `a` and bit `j` of `b` both set.
fn clmul_reference(a: u32, b: u32) -> (u32, u32) {
    let mut bits = [false; 64];
    for i in 0..32 {
        for j in 0..32 {
            if (a >> i) & 1 == 1 && (b >> j) & 1 == 1 {
                bits[i + j] ^= true;
            }
        }
    }
    let pack = |range: std::ops::Range<usize>| {
        range
            .enumerate()
            .fold(0u32, |acc, (n, k)| acc | (u32::from(bits[k]) << n))
    };
    (pack(0..32), pack(32..64))
}

fn dp4a_reference(a: u32, b: u32, c: u32) -> u32 {
    let sum: i32 = (0..4)
        .map(|k| i32::from((a >> (8 * k)) as u8 as i8) * i32::from((b >> (8 * k)) as u8 as i8))
        .sum();
    (c as i32).wrapping_add(sum) as u32
}

#[test]
fn the_carry_less_multiply_matches_the_parity_reference() {
    let got = run(&clmul(Mutant::None), &[&A, &B], 2);
    let want: Vec<(u32, u32)> = (0..N).map(|i| clmul_reference(A[i], B[i])).collect();
    assert_eq!(got[0], want.iter().map(|w| w.0).collect::<Vec<_>>());
    assert_eq!(got[1], want.iter().map(|w| w.1).collect::<Vec<_>>());
    // 3 (x) 3 = 5 in GF(2), where ordinary multiplication gives 9: the carry shows
    assert_eq!(clmul_reference(3, 3), (5, 0));
}

#[test]
fn the_clmul_mutants_fail_by_value() {
    let want: Vec<u32> = (0..N).map(|i| clmul_reference(A[i], B[i]).0).collect();
    for mutant in [Mutant::Carries, Mutant::ShiftedByOne] {
        let got = run(&clmul(mutant), &[&A, &B], 2);
        assert_ne!(got[0], want, "{mutant:?}");
    }
}

const C: [u32; N] = [
    0,
    1,
    0xFFFF_FFFF,
    100,
    0x8000_0000,
    7,
    0x7FFF_FFFF,
    0xFFFF_FF00,
];
const BYTES_A: [u32; N] = [
    0x0000_0000,
    0xFFFF_FFFF,
    0x8080_8080,
    0x7F7F_7F7F,
    0x0102_0304,
    0x80FF_017F,
    0xFF00_FF00,
    0x1234_5678,
];
const BYTES_B: [u32; N] = [
    0xFFFF_FFFF,
    0xFFFF_FFFF,
    0x8080_8080,
    0x0101_0101,
    0xFFFE_FDFC,
    0x7F01_FF80,
    0x00FF_00FF,
    0x8765_4321,
];

#[test]
fn the_byte_dot_product_matches_the_cast_reference() {
    let got = run(&dp4a(Mutant::None), &[&BYTES_A, &BYTES_B, &C], 1);
    let want: Vec<u32> = (0..N)
        .map(|i| dp4a_reference(BYTES_A[i], BYTES_B[i], C[i]))
        .collect();
    assert_eq!(got[0], want);
}

#[test]
fn the_dp4a_mutants_fail_by_value() {
    let want: Vec<u32> = (0..N)
        .map(|i| dp4a_reference(BYTES_A[i], BYTES_B[i], C[i]))
        .collect();
    for mutant in [Mutant::UnsignedBytes, Mutant::NoAccumulator] {
        let got = run(&dp4a(mutant), &[&BYTES_A, &BYTES_B, &C], 1);
        assert_ne!(got[0], want, "{mutant:?}");
    }
}
