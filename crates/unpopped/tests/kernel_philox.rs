// SPDX-License-Identifier: MIT OR Apache-2.0
//! **Philox4x32-10 at the kernel layer** (survey rows 22 and 33). Run on the oracle against a reference written separately in
//! Rust (`u64` multiply and a `u32` array, a different shape from the kernel's locals), and against three known answers
//! recalled from the Random123 distribution's test vectors (counter and key all zero, all ones, and the digits of pi). Those
//! known answers are from memory and UNVERIFIED against the source: if the recalled vectors and the independent reference
//! agree, that corroborates both; they would not agree by accident. Integers have no NaN; the mutants (nine rounds, swapped
//! multipliers, an unbumped key) must change the output on every counter block.

use unpopped::kernel::{KernelFn, check_kernel};
use unpopped::oracle::TypedBuffer;
use unpopped::oracle::kernel_eval::KernelArg;
use unpopped::oracle::kernel_launch::run_launch;

#[path = "support/philox.rs"]
mod fixture;
use fixture::{M0, M1, Mutant, N, U32, W0, W1, philox};

fn buf(v: &[u32]) -> KernelArg {
    KernelArg::Buffer(TypedBuffer::new(
        U32,
        vec![v.len() as i64],
        vec![1],
        v.iter().flat_map(|x| x.to_le_bytes()).collect(),
    ))
}

fn run(mutant: Mutant, ctr: &[u32], key: &[u32]) -> Vec<u32> {
    let k: KernelFn = philox(mutant);
    assert_eq!(check_kernel(&k), Ok(()));
    let mut args = vec![buf(ctr), buf(key), buf(&[0; N * 4])];
    run_launch(&k, &mut args).expect("the oracle runs it");
    let KernelArg::Buffer(o) = &args[2] else {
        unreachable!("a buffer")
    };
    o.bytes
        .chunks(4)
        .map(|c| u32::from_le_bytes(c.try_into().unwrap()))
        .collect()
}

/// The reference: the multiplies as `u128`, the counter as an array rotated through a new array each round.
fn reference(mut c: [u32; 4], mut k: [u32; 2]) -> [u32; 4] {
    for _ in 0..10 {
        let p0 = u128::from(M0) * u128::from(c[0]);
        let p1 = u128::from(M1) * u128::from(c[2]);
        c = [
            ((p1 >> 32) as u32) ^ c[1] ^ k[0],
            p1 as u32,
            ((p0 >> 32) as u32) ^ c[3] ^ k[1],
            p0 as u32,
        ];
        k = [k[0].wrapping_add(W0), k[1].wrapping_add(W1)];
    }
    c
}

/// Four counter blocks with their keys: the three recalled known-answer inputs and an asymmetric block.
const CTR: [[u32; 4]; N] = [
    [0, 0, 0, 0],
    [0xFFFF_FFFF; 4],
    [0x243F_6A88, 0x85A3_08D3, 0x1319_8A2E, 0x0370_7344],
    [1, 2, 3, 0x8000_0000],
];
const KEY: [[u32; 2]; N] = [
    [0, 0],
    [0xFFFF_FFFF; 2],
    [0xA409_3822, 0x299F_31D0],
    [0xDEAD_BEEF, 7],
];

fn flat<const M: usize>(v: &[[u32; M]]) -> Vec<u32> {
    v.iter().flatten().copied().collect()
}

#[test]
fn the_kernel_matches_the_independent_reference_on_every_block() {
    let got = run(Mutant::None, &flat(&CTR), &flat(&KEY));
    let want: Vec<u32> = (0..N).flat_map(|i| reference(CTR[i], KEY[i])).collect();
    assert_eq!(got, want);
}

#[test]
fn the_recalled_known_answers_agree_with_the_reference_and_the_kernel() {
    let known: [[u32; 4]; 3] = [
        [0x6627_E8D5, 0xE169_C58D, 0xBC57_AC4C, 0x9B00_DBD8],
        [0x408F_276D, 0x41C8_3B0E, 0xA20B_C7C6, 0x6D54_51FD],
        [0xD16C_FE09, 0x94FD_CCEB, 0x5001_E420, 0x2412_6EA1],
    ];
    let got = run(Mutant::None, &flat(&CTR), &flat(&KEY));
    for (i, want) in known.iter().enumerate() {
        assert_eq!(&reference(CTR[i], KEY[i]), want, "reference, vector {i}");
        assert_eq!(&got[i * 4..i * 4 + 4], want, "kernel, vector {i}");
    }
}

#[test]
fn every_mutant_changes_every_block() {
    let want: Vec<u32> = (0..N).flat_map(|i| reference(CTR[i], KEY[i])).collect();
    for mutant in [
        Mutant::NineRounds,
        Mutant::SwappedMultipliers,
        Mutant::NoKeyBump,
    ] {
        let got = run(mutant, &flat(&CTR), &flat(&KEY));
        for i in 0..N {
            assert_ne!(
                got[i * 4..i * 4 + 4],
                want[i * 4..i * 4 + 4],
                "{mutant:?}, block {i}"
            );
        }
    }
}
