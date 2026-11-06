// SPDX-License-Identifier: MIT OR Apache-2.0
//! **Tensor-core attention at the kernel layer** (ROADMAP U10 A6, `docs/options-u10-attention.md`): `S = scale * Q K^T` as two
//! `Stmt::Mma` over the two key tiles, the row softmax between them in scalar code (the lanes exchange `S` and `P` through
//! global scratch and block barriers), and `O = P V` as a third `Stmt::Mma`, on an `F32` toy `m4n4k8` atom. The oracle's
//! result must match a host loop in the same `f32` order. Scores pass 89, the row maximum sits at different keys, and the key
//! tiles differ, so a dropped max subtraction, a swapped tile, a wrong fragment table or a missing barrier changes the answer.

use unpopped::kernel::check_kernel;
use unpopped::oracle::TypedBuffer;
use unpopped::oracle::kernel_eval::KernelArg;
use unpopped::oracle::kernel_launch::run_launch;

#[path = "support/attention_mma.rs"]
mod attention_mma;
use attention_mma::{
    F32, K, M, N, U32, attention_reference, id_table, logical_buffers, mma_attention,
};

fn f32_buf(v: &[f32]) -> KernelArg {
    KernelArg::Buffer(TypedBuffer::new(
        F32,
        vec![v.len() as i64],
        vec![1],
        v.iter().flat_map(|x| x.to_le_bytes()).collect(),
    ))
}

fn u32_buf(v: &[u32]) -> KernelArg {
    KernelArg::Buffer(TypedBuffer::new(
        U32,
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

/// The kernel's arguments, in buffer order.
fn args() -> Vec<KernelArg> {
    let [q, k0, k1, v] = logical_buffers();
    vec![
        f32_buf(&q),
        f32_buf(&k0),
        f32_buf(&k1),
        f32_buf(&v),
        u32_buf(&id_table(0)),
        u32_buf(&id_table(1)),
        u32_buf(&id_table(2)),
        f32_buf(&[0.0; M * N]),
        f32_buf(&[0.0; M * N]),
        f32_buf(&[0.0; M * K]),
        f32_buf(&[0.0; M * N]),
    ]
}

#[test]
fn the_mma_attention_kernel_matches_the_host_loop() {
    let k = mma_attention();
    assert_eq!(check_kernel(&k), Ok(()));
    let mut a = args();
    run_launch(&k, &mut a).expect("the oracle runs it");
    let (s, out) = attention_reference();
    // The scratch holds S tile by tile, column-major; compare it as the host's row-major `M x 8`.
    let (s0, s1) = (floats(&a[7]), floats(&a[8]));
    let got_s: Vec<f32> = (0..M * K)
        .map(|i| {
            let (row, key) = (i / K, i % K);
            let tile = if key < N { &s0 } else { &s1 };
            tile[row + M * (key % N)]
        })
        .collect();
    close(&got_s, &s, "scores");
    // out is column-major `M x N`.
    let got: Vec<f32> = (0..M * N)
        .map(|i| floats(&a[10])[(i / N) + M * (i % N)])
        .collect();
    close(&got, &out, "out");
}

#[test]
fn the_data_can_tell_the_mutants_apart() {
    let (s, out) = attention_reference();
    assert!(
        s.iter().any(|&x| x > 89.0),
        "some score overflows exp, so dropping the max subtraction is visible"
    );
    let argmax = |i: usize| {
        (0..K)
            .max_by(|&a, &b| s[i * K + a].total_cmp(&s[i * K + b]))
            .unwrap()
    };
    let positions: std::collections::BTreeSet<usize> = (0..M).map(argmax).collect();
    assert!(positions.len() >= 2, "the maximum moves: {positions:?}");
    assert!(
        positions.iter().any(|&p| p >= N) && positions.iter().any(|&p| p < N),
        "the maximum falls in both key tiles: {positions:?}"
    );
    assert!(
        out.chunks(N).any(|r| r != &out[..N]),
        "rows of the output differ"
    );
    assert!(out.iter().all(|x| x.is_finite()));
}

#[test]
fn the_a_and_b_id_tables_are_the_same_for_this_atom() {
    // Documents the equivalent mutant "P gathered with the B table": for the toy atom the two fragment layouts coincide.
    assert_eq!(id_table(0), id_table(1));
}
