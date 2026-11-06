// SPDX-License-Identifier: MIT OR Apache-2.0
//! **Naive attention at the kernel layer** (ROADMAP U10 A1, `docs/options-u10-attention.md`): scores into a scratch buffer, a row
//! max and sum, then the weighted values. The oracle's result must match a host loop in the same `f32` order, and the data must be
//! able to tell the mutants apart: some score passes 89 (an `exp` without the max subtraction overflows), the row maximum sits
//! at different positions, and the output is not constant across rows.
//!
//! Seeded mutants (anchor count fixed, tree committed first), all caught here: scale dropped, max subtraction dropped in the sums, in
//! the weights or in both, normalizer dropped, the row max a min, a key read d-major, the last key left out of the sum, and the `V`
//! row stride wrong (`DV` differs from `D` on purpose). One is equivalent and documented: "row max starts at the last score" (the loop
//! still visits every score). The overflow mutants need the oracle to hold an `F32` local as an `f32` (fixed with this kernel's
//! base, `tests/kernel_f32_local_rounding.rs`): before it, only the C differential caught them.

use unpopped::kernel::check_kernel;
use unpopped::oracle::TypedBuffer;
use unpopped::oracle::kernel_eval::KernelArg;
use unpopped::oracle::kernel_launch::run_launch;

#[path = "support/attention.rs"]
mod attention;
use attention::{D, DV, F32, LK, LQ, attention_data, attention_reference, naive_attention};

fn f32_buf(v: &[f32]) -> KernelArg {
    KernelArg::Buffer(TypedBuffer::new(
        F32,
        vec![v.len() as i64],
        vec![1],
        v.iter().flat_map(|x| x.to_le_bytes()).collect(),
    ))
}

fn floats(a: &KernelArg) -> Vec<f32> {
    let KernelArg::Buffer(b) = a else {
        unreachable!("a buffer")
    };
    b.bytes
        .chunks(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect()
}

fn close(got: &[f32], want: &[f32], what: &str) {
    assert_eq!(got.len(), want.len(), "{what}");
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
        assert!(
            (g - w).abs() <= 1e-4 * w.abs().max(1.0),
            "{what}[{i}]: got {g}, want {w}"
        );
    }
}

#[test]
fn the_naive_attention_kernel_matches_the_host_loop() {
    let k = naive_attention();
    assert_eq!(check_kernel(&k), Ok(()));
    let (q, kk, v) = attention_data();
    let mut args = [
        f32_buf(&q),
        f32_buf(&kk),
        f32_buf(&v),
        f32_buf(&[0.0; LQ * LK]),
        f32_buf(&[0.0; LQ * DV]),
    ];
    run_launch(&k, &mut args).expect("the oracle runs it");
    let (s, out) = attention_reference();
    close(&floats(&args[3]), &s, "scores");
    close(&floats(&args[4]), &out, "out");
}

#[test]
fn the_data_can_tell_the_mutants_apart() {
    let (s, out) = attention_reference();
    assert!(
        s.iter().any(|&x| x > 89.0),
        "some score overflows exp, so dropping the max subtraction is visible"
    );
    let argmax = |i: usize| {
        (0..LK)
            .max_by(|&a, &b| s[i * LK + a].total_cmp(&s[i * LK + b]))
            .unwrap()
    };
    let positions: std::collections::BTreeSet<usize> = (0..LQ).map(argmax).collect();
    assert!(positions.len() >= 2, "the maximum moves: {positions:?}");
    assert!(
        positions.contains(&(LK - 1)),
        "a row peaks in the last position"
    );
    assert!(
        out.chunks(DV).any(|r| r != &out[..DV]),
        "rows of the output differ"
    );
    assert!(out.iter().all(|x| x.is_finite()));
    let _ = D;
}
