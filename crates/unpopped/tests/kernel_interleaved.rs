// SPDX-License-Identifier: MIT OR Apache-2.0
//! **Interleaved-layout contractions at the kernel layer** (CUTLASS survey row 6: `ColumnMajorInterleaved`, `NCxHWx`). A GEMM
//! with `A` column-major-interleaved and `B` row-major-interleaved (interleave 4 along `K`), and a 1x1 convolution over an
//! `NC4HW4` input, run on the oracle against references computed from the LOGICAL matrices. The operands are packed here by
//! flat-offset formulas written independently of the kernels' coordinate form, from distinct small integers (exact in `f32`),
//! with a NaN corpus and an all-negative corpus. A swapped interleave coordinate and a wrong minor coordinate are mutants that
//! stay in range, so they fail by value, not by a trap.

use unpopped::kernel::{KernelFn, check_kernel};
use unpopped::oracle::TypedBuffer;
use unpopped::oracle::kernel_eval::KernelArg;
use unpopped::oracle::kernel_launch::run_launch;

#[path = "support/interleaved.rs"]
mod fixture;
use fixture::{C, F32, HH, I, K, M, Mutant, N, NB, O, WW, conv1x1_nc4hw4, gemm_interleaved};

fn buf(shape: &[i64], v: &[f32]) -> KernelArg {
    let mut strides = vec![1; shape.len()];
    for d in (0..shape.len() - 1).rev() {
        strides[d] = strides[d + 1] * shape[d + 1];
    }
    KernelArg::Buffer(TypedBuffer::new(
        F32,
        shape.to_vec(),
        strides,
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

fn same(got: &[f32], want: &[f32]) -> bool {
    got.len() == want.len()
        && got
            .iter()
            .zip(want)
            .all(|(g, w)| g.to_bits() == w.to_bits() || (g.is_nan() && w.is_nan()))
}

/// Logical `A[m][k]`, `B[k][n]`: distinct small integers, so a wrong element shows.
fn a_at(m: usize, k: usize) -> f32 {
    ((m * 5 + k * 3) % 11) as f32 - 5.0
}
fn b_at(k: usize, n: usize) -> f32 {
    ((k * 7 + n * 2) % 13) as f32 - 6.0
}

#[derive(Clone)]
struct Gemm {
    a: Vec<f32>, // logical [M][K]
    b: Vec<f32>, // logical [K][N]
}

fn gemm_data() -> Gemm {
    Gemm {
        a: (0..M * K).map(|i| a_at(i / K, i % K)).collect(),
        b: (0..K * N).map(|i| b_at(i / N, i % N)).collect(),
    }
}

fn gemm_nan() -> Gemm {
    let mut g = gemm_data();
    g.a[2 * K + 5] = f32::NAN;
    g
}

fn gemm_negative() -> Gemm {
    let mut g = gemm_data();
    g.a.iter_mut().for_each(|x| *x = -x.abs() - 1.0);
    g.b.iter_mut().for_each(|x| *x = x.abs() + 1.0);
    g
}

/// `ColumnMajorInterleaved<I>` of an `M x K` matrix, by its flat offset formula.
fn pack_cmi(a: &[f32]) -> Vec<f32> {
    let mut p = vec![0.0; M * K];
    for r in 0..M {
        for c in 0..K {
            p[(c / I) * (M * I) + r * I + c % I] = a[r * K + c];
        }
    }
    p
}

/// `RowMajorInterleaved<I>` of a `K x N` matrix.
fn pack_rmi(b: &[f32]) -> Vec<f32> {
    let mut p = vec![0.0; K * N];
    for r in 0..K {
        for c in 0..N {
            p[(r / I) * (N * I) + c * I + r % I] = b[r * N + c];
        }
    }
    p
}

fn gemm_reference(g: &Gemm) -> Vec<f32> {
    let mut c = vec![0.0; M * N];
    for m in 0..M {
        for n in 0..N {
            let mut acc = 0.0f32;
            for k in 0..K {
                acc += g.a[m * K + k] * g.b[k * N + n];
            }
            c[m * N + n] = acc;
        }
    }
    c
}

fn gemm_run(mutant: Mutant, g: &Gemm) -> Vec<f32> {
    run(
        &gemm_interleaved(mutant),
        vec![
            buf(&[(K / I) as i64, M as i64, I as i64], &pack_cmi(&g.a)),
            buf(&[(K / I) as i64, N as i64, I as i64], &pack_rmi(&g.b)),
            buf(&[M as i64, N as i64], &[0.0; M * N]),
        ],
        2,
    )
}

#[test]
fn the_interleaved_gemm_matches_the_logical_reference_on_every_corpus() {
    for (name, g) in [
        ("ordinary", gemm_data()),
        ("nan", gemm_nan()),
        ("negative", gemm_negative()),
    ] {
        let (got, want) = (gemm_run(Mutant::None, &g), gemm_reference(&g));
        assert!(same(&got, &want), "{name}: {got:?} vs {want:?}");
    }
}

#[test]
fn the_gemm_corpus_is_not_blind() {
    // the NaN reaches exactly row 2, and the packed buffers differ from the logical ones (a layout that did nothing shows)
    let want = gemm_reference(&gemm_nan());
    let rows_nan: Vec<bool> = (0..M)
        .map(|m| want[m * N..(m + 1) * N].iter().any(|x| x.is_nan()))
        .collect();
    assert_eq!(rows_nan, vec![false, false, true, false]);
    let g = gemm_data();
    assert_ne!(pack_cmi(&g.a), g.a);
    assert_ne!(pack_rmi(&g.b), g.b);
}

#[test]
fn the_swapped_interleave_gemm_mutant_fails_by_value() {
    let g = gemm_data();
    let got = gemm_run(Mutant::SwappedMinor, &g);
    assert!(
        got.iter().all(|x| x.is_finite()),
        "in range, so no trap and no NaN"
    );
    assert!(!same(&got, &gemm_reference(&g)));
}

fn x_at(n: usize, c: usize, h: usize, w: usize) -> f32 {
    ((n * 3 + c * 5 + h * 7 + w * 11) % 17) as f32 - 8.0
}
fn w_at(o: usize, c: usize) -> f32 {
    ((o * 4 + c * 3) % 7) as f32 - 3.0
}

fn conv_reference(x: &dyn Fn(usize, usize, usize, usize) -> f32) -> Vec<f32> {
    let mut out = vec![0.0; NB * O * HH * WW];
    for n in 0..NB {
        for o in 0..O {
            for h in 0..HH {
                for w in 0..WW {
                    let mut acc = 0.0f32;
                    for c in 0..C {
                        acc += x(n, c, h, w) * w_at(o, c);
                    }
                    out[((n * O + o) * HH + h) * WW + w] = acc;
                }
            }
        }
    }
    out
}

/// `NCxHWx` packing by its flat offset: `[N][C/x][H][W][x]`.
fn pack_nchwx(x: &dyn Fn(usize, usize, usize, usize) -> f32) -> Vec<f32> {
    let mut p = vec![0.0; NB * C * HH * WW];
    for n in 0..NB {
        for c in 0..C {
            for h in 0..HH {
                for w in 0..WW {
                    p[(((n * (C / I) + c / I) * HH + h) * WW + w) * I + c % I] = x(n, c, h, w);
                }
            }
        }
    }
    p
}

fn conv_run(mutant: Mutant, x: &dyn Fn(usize, usize, usize, usize) -> f32) -> Vec<f32> {
    let wgt: Vec<f32> = (0..O * C).map(|i| w_at(i / C, i % C)).collect();
    run(
        &conv1x1_nc4hw4(mutant),
        vec![
            buf(
                &[NB as i64, (C / I) as i64, HH as i64, WW as i64, I as i64],
                &pack_nchwx(x),
            ),
            buf(&[O as i64, C as i64], &wgt),
            buf(
                &[NB as i64, O as i64, HH as i64, WW as i64],
                &[0.0; NB * O * HH * WW],
            ),
        ],
        2,
    )
}

#[test]
fn the_nc4hw4_conv_matches_the_logical_reference_including_a_nan_channel() {
    let (got, want) = (conv_run(Mutant::None, &x_at), conv_reference(&x_at));
    assert!(same(&got, &want), "{got:?} vs {want:?}");
    // a NaN at one (n, c, h, w) reaches exactly the outputs of that pixel
    let nan = |n, c, h, w| {
        if (n, c, h, w) == (1, 6, 1, 2) {
            f32::NAN
        } else {
            x_at(n, c, h, w)
        }
    };
    let (got, want) = (conv_run(Mutant::None, &nan), conv_reference(&nan));
    assert!(same(&got, &want));
    assert_eq!(
        want.iter().filter(|x| x.is_nan()).count(),
        O,
        "one pixel, every output channel"
    );
}

#[test]
fn the_wrong_minor_conv_mutant_fails_by_value() {
    let got = conv_run(Mutant::WrongMinor, &x_at);
    assert!(got.iter().all(|x| x.is_finite()));
    assert!(!same(&got, &conv_reference(&x_at)));
}
