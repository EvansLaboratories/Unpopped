// SPDX-License-Identifier: MIT OR Apache-2.0
//! **GQA and MLA decode attention at the kernel layer** (CUTLASS survey row 92, "decode/GQA", "MLA"; ROADMAP U9 scope A style).
//! The kernels are `support/attention_gqa_mla.rs`. The references here are written independently: GQA is a two-pass softmax
//! (the kernel streams it), and MLA is the EXPANDED form (`k_nope[j] = W_UK c[j]`, `v[j] = W_UV c[j]`, a plain softmax over the
//! full keys) while the kernel takes the absorbed query and attends in the latent space, so the test also checks the absorption
//! identity. Mutants (a head mapping, a dropped rope term, a transposed up-projection) each fail a named test.

use unpopped::kernel::{KernelFn, check_kernel};
use unpopped::oracle::TypedBuffer;
use unpopped::oracle::kernel_eval::KernelArg;
use unpopped::oracle::kernel_launch::run_launch;

#[path = "support/attention_gqa_mla.rs"]
mod fixture;
use fixture::{D, DV, F32, H, HKV, HQ, L, Mutant, R, RP, SCALE, gqa_decode, mla_decode, pattern};

fn buf(v: &[f32]) -> KernelArg {
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

fn run(k: &KernelFn, mut args: Vec<KernelArg>, out: usize) -> Vec<f32> {
    assert_eq!(check_kernel(k), Ok(()));
    run_launch(k, &mut args).expect("the oracle runs it");
    floats(&args[out])
}

fn close(got: &[f32], want: &[f32], what: &str) {
    assert_eq!(got.len(), want.len(), "{what}");
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
        let same = (g.is_nan() && w.is_nan()) || (g - w).abs() <= 1e-4 * w.abs().max(1.0);
        assert!(same, "{what}[{i}]: got {g}, want {w}");
    }
}

fn agrees(got: &[f32], want: &[f32]) -> bool {
    got.len() == want.len()
        && got
            .iter()
            .zip(want)
            .all(|(g, w)| (g.is_nan() && w.is_nan()) || (g - w).abs() <= 1e-4 * w.abs().max(1.0))
}

/// `softmax(scores) . values`, two-pass.
fn softmax_mix(scores: &[f32], value: impl Fn(usize) -> f32) -> f32 {
    let m = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let e: Vec<f32> = scores.iter().map(|s| (s - m).exp()).collect();
    let z: f32 = e.iter().sum();
    e.iter().enumerate().map(|(j, p)| p / z * value(j)).sum()
}

struct Gqa {
    q: Vec<f32>,
    k: Vec<f32>,
    v: Vec<f32>,
}

fn gqa_data() -> Gqa {
    Gqa {
        q: pattern(HQ * D, 3, 7, 0),
        k: pattern(L * HKV * D, 5, 9, 1),
        v: pattern(L * HKV * DV, 7, 5, 2),
    }
}

/// Every score negative: a positive key and a negative query.
fn gqa_all_negative() -> Gqa {
    let mut g = gqa_data();
    g.q.iter_mut().for_each(|x| *x = -x.abs() - 0.25);
    g.k.iter_mut().for_each(|x| *x = x.abs() + 0.25);
    g
}

/// One NaN in a value of KV head 1: only the query heads that read KV head 1 may see it.
fn gqa_nan() -> Gqa {
    let mut g = gqa_data();
    g.v[(2 * HKV + 1) * DV + 1] = f32::NAN;
    g
}

fn gqa_reference(g: &Gqa) -> Vec<f32> {
    let group = HQ / HKV;
    let mut out = vec![0.0; HQ * DV];
    for h in 0..HQ {
        let kv = h / group;
        let scores: Vec<f32> = (0..L)
            .map(|j| {
                SCALE
                    * (0..D)
                        .map(|e| g.q[h * D + e] * g.k[(j * HKV + kv) * D + e])
                        .sum::<f32>()
            })
            .collect();
        for d in 0..DV {
            out[h * DV + d] = softmax_mix(&scores, |j| g.v[(j * HKV + kv) * DV + d]);
        }
    }
    out
}

fn gqa_run(m: Mutant, g: &Gqa) -> Vec<f32> {
    run(
        &gqa_decode(m),
        vec![buf(&g.q), buf(&g.k), buf(&g.v), buf(&[0.0; HQ * DV])],
        3,
    )
}

const DN: usize = 3;

struct Mla {
    q_nope: Vec<f32>,
    q_rope: Vec<f32>,
    c: Vec<f32>,
    k_rope: Vec<f32>,
    w_uk: Vec<f32>,
    w_uv: Vec<f32>,
}

fn mla_data() -> Mla {
    Mla {
        q_nope: pattern(H * DN, 3, 7, 0),
        q_rope: pattern(H * RP, 5, 5, 1),
        c: pattern(L * R, 5, 9, 1),
        k_rope: pattern(L * RP, 3, 7, 3),
        w_uk: pattern(H * DN * R, 7, 5, 2),
        w_uv: pattern(H * DV * R, 3, 7, 4),
    }
}

/// The reference: expand the latent to full keys and values, then a plain softmax.
fn mla_reference(m: &Mla) -> Vec<f32> {
    let mut out = vec![0.0; H * DV];
    for h in 0..H {
        let up = |w: &[f32], rows: usize, row: usize, j: usize| -> f32 {
            (0..R)
                .map(|r| w[(h * rows + row) * R + r] * m.c[j * R + r])
                .sum()
        };
        let scores: Vec<f32> = (0..L)
            .map(|j| {
                let nope: f32 = (0..DN)
                    .map(|e| m.q_nope[h * DN + e] * up(&m.w_uk, DN, e, j))
                    .sum();
                let rope: f32 = (0..RP)
                    .map(|e| m.q_rope[h * RP + e] * m.k_rope[j * RP + e])
                    .sum();
                SCALE * (nope + rope)
            })
            .collect();
        for d in 0..DV {
            out[h * DV + d] = softmax_mix(&scores, |j| up(&m.w_uv, DV, d, j));
        }
    }
    out
}

/// `q_abs[h][r] = sum_e W_UK[h][e][r] q_nope[h][e]`.
fn absorb(m: &Mla) -> Vec<f32> {
    (0..H * R)
        .map(|i| {
            let (h, r) = (i / R, i % R);
            (0..DN)
                .map(|e| m.w_uk[(h * DN + e) * R + r] * m.q_nope[h * DN + e])
                .sum()
        })
        .collect()
}

fn mla_run(mutant: Mutant, m: &Mla) -> Vec<f32> {
    run(
        &mla_decode(mutant),
        vec![
            buf(&absorb(m)),
            buf(&m.q_rope),
            buf(&m.c),
            buf(&m.k_rope),
            buf(&m.w_uv),
            buf(&[0.0; H * R]),
            buf(&[0.0; H * DV]),
        ],
        6,
    )
}

#[test]
fn the_gqa_kernel_matches_the_grouped_reference_on_every_corpus() {
    for (name, g) in [
        ("ordinary", gqa_data()),
        ("all negative", gqa_all_negative()),
        ("nan", gqa_nan()),
    ] {
        close(&gqa_run(Mutant::None, &g), &gqa_reference(&g), name);
    }
}

#[test]
fn the_gqa_nan_reaches_only_the_query_heads_that_read_its_kv_head() {
    let out = gqa_reference(&gqa_nan());
    let nan_heads = |o: &[f32]| {
        (0..HQ)
            .map(|h| o[h * DV..(h + 1) * DV].iter().any(|x| x.is_nan()))
            .collect::<Vec<_>>()
    };
    assert_eq!(nan_heads(&out), vec![false, false, true, true]);
    assert_eq!(
        nan_heads(&gqa_run(Mutant::None, &gqa_nan())),
        vec![false, false, true, true]
    );
}

#[test]
fn the_gqa_head_mapping_mutants_fail() {
    for m in [Mutant::InterleavedKvHead, Mutant::SingleKvHead] {
        let caught = [gqa_data(), gqa_all_negative(), gqa_nan()]
            .iter()
            .any(|g| !agrees(&gqa_run(m, g), &gqa_reference(g)));
        assert!(caught, "{m:?} agrees with the grouped reference everywhere");
    }
}

#[test]
fn the_mla_absorbed_kernel_matches_the_expanded_reference() {
    let m = mla_data();
    close(&mla_run(Mutant::None, &m), &mla_reference(&m), "mla");
}

#[test]
fn the_mla_mutants_fail() {
    let m = mla_data();
    for mutant in [Mutant::NoRope, Mutant::TransposedUp] {
        assert!(
            !agrees(&mla_run(mutant, &m), &mla_reference(&m)),
            "{mutant:?} agrees with the expanded reference"
        );
    }
    // the rope term matters: with it zeroed the reference changes too, so the data can see it
    let mut no_rope = mla_data();
    no_rope.q_rope.iter_mut().for_each(|x| *x = 0.0);
    assert!(!agrees(&mla_reference(&no_rope), &mla_reference(&m)));
}
