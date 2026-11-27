// SPDX-License-Identifier: MIT OR Apache-2.0
//! **Grouped, depthwise and transposed convolution at the kernel layer** (CUTLASS survey rows 46 and 48; the kernels are
//! `support/conv_grouped.rs`). A grouped forward convolution (G = 2) and a depthwise one (G = C = K), both with stride
//! `(2, 1)`, dilation `(1, 2)` and ASYMMETRIC padding (top 1, bottom 0, left 2, right 1), and a transposed convolution with an
//! output padding that enlarges the output. Each is run on the oracle and compared with a host SCATTER reference (the kernels
//! gather) over integer data. Mutants (interleaved groups, first group only, a wrong padding sign) fail by value.

use unpopped::kernel::{KernelFn, check_kernel};
use unpopped::oracle::TypedBuffer;
use unpopped::oracle::kernel_eval::KernelArg;
use unpopped::oracle::kernel_launch::run_launch;

#[path = "support/conv_grouped.rs"]
mod fixture;
use fixture::{
    DEPTHWISE, DH, DW, GROUPED, Geo, H, HO, I32, Mutant, N, OPH, OPW, P, PB, PL, PR, PT, Q, R, S,
    SH, SW, TC, TH, TK, TPAD, TW, W, WO, grouped_fprop, transposed_fprop,
};

fn buf(shape: &[i64], v: &[i32]) -> KernelArg {
    let strides: Vec<i64> = (0..shape.len())
        .map(|d| shape[d + 1..].iter().product())
        .collect();
    KernelArg::Buffer(TypedBuffer::new(
        I32,
        shape.to_vec(),
        strides,
        v.iter().flat_map(|x| x.to_le_bytes()).collect(),
    ))
}

fn words(a: &KernelArg) -> Vec<i32> {
    let KernelArg::Buffer(b) = a else {
        unreachable!("a buffer")
    };
    b.bytes
        .chunks(4)
        .map(|c| i32::from_le_bytes(c.try_into().unwrap()))
        .collect()
}

fn run(k: &KernelFn, input: KernelArg, filter: KernelArg, out_shape: &[i64]) -> Vec<i32> {
    assert_eq!(check_kernel(k), Ok(()));
    let n: i64 = out_shape.iter().product();
    let mut args = [input, filter, buf(out_shape, &vec![0; n as usize])];
    run_launch(k, &mut args).expect("the oracle runs it");
    words(&args[2])
}

fn idx(d: &[i64], i: &[i64]) -> usize {
    d.iter().zip(i).fold(0, |a, (d, i)| a * d + i) as usize
}

fn data(n: i64, salt: i64, modulus: i64) -> Vec<i32> {
    (0..n)
        .map(|i| ((i * (salt + 4) + salt) % modulus) as i32 - (modulus / 2) as i32)
        .collect()
}

fn grouped_inputs(geo: Geo) -> (Vec<i32>, Vec<i32>) {
    (
        data(N * H * W * geo.c, 3, 11),
        data(geo.k * R * S * (geo.c / geo.g), 5, 7),
    )
}

/// Scatter reference of the grouped forward convolution.
fn grouped_reference(geo: Geo, x: &[i32], f: &[i32]) -> Vec<i32> {
    let (cg, kg) = (geo.c / geo.g, geo.k / geo.g);
    let mut out = vec![0; (N * P * Q * geo.k) as usize];
    for n in 0..N {
        for p in 0..P {
            for q in 0..Q {
                for r in 0..R {
                    for s in 0..S {
                        let (ih, iw) = (p * SH + r * DH - PT, q * SW + s * DW - PL);
                        if !(0..H).contains(&ih) || !(0..W).contains(&iw) {
                            continue;
                        }
                        for k in 0..geo.k {
                            for c in 0..cg {
                                let ch = (k / kg) * cg + c;
                                out[idx(&[N, P, Q, geo.k], &[n, p, q, k])] += x
                                    [idx(&[N, H, W, geo.c], &[n, ih, iw, ch])]
                                    * f[idx(&[geo.k, R, S, cg], &[k, r, s, c])];
                            }
                        }
                    }
                }
            }
        }
    }
    out
}

fn grouped_run(geo: Geo, mutant: Mutant) -> Vec<i32> {
    let (x, f) = grouped_inputs(geo);
    run(
        &grouped_fprop(geo, mutant),
        buf(&[N, H, W, geo.c], &x),
        buf(&[geo.k, R, S, geo.c / geo.g], &f),
        &[N, P, Q, geo.k],
    )
}

#[test]
fn the_grouped_and_depthwise_convolutions_match_the_scatter_reference() {
    for (name, geo) in [("grouped", GROUPED), ("depthwise", DEPTHWISE)] {
        let (x, f) = grouped_inputs(geo);
        assert_eq!(
            grouped_run(geo, Mutant::None),
            grouped_reference(geo, &x, &f),
            "{name}"
        );
    }
}

#[test]
fn the_asymmetric_padding_is_exercised_at_every_edge() {
    // the extents come out as the asymmetric padding says (a symmetric 1/1 would give Q = 5), and taps fall outside at top,
    // left and right but never at the bottom, where the padding is 0
    assert_eq!((P, Q), (2, 6));
    assert_eq!((PT, PB, PL, PR), (1, 0, 2, 1));
    let top = (0..R).any(|r| 0 * SH + r * DH - PT < 0);
    let bottom = (0..R).any(|r| (P - 1) * SH + r * DH - PT >= H);
    let left = (0..S).any(|s| 0 * SW + s * DW - PL < 0);
    let right = (0..S).any(|s| (Q - 1) * SW + s * DW - PL >= W);
    assert!(top && left && right && !bottom);
}

#[test]
fn the_group_mapping_mutants_fail_by_value() {
    for geo in [GROUPED] {
        let (x, f) = grouped_inputs(geo);
        let want = grouped_reference(geo, &x, &f);
        for m in [Mutant::InterleavedGroups, Mutant::FirstGroupOnly] {
            assert_ne!(grouped_run(geo, m), want, "{m:?} on {geo:?}");
        }
    }
}

fn transposed_inputs() -> (Vec<i32>, Vec<i32>) {
    (data(N * TH * TW * TC, 2, 9), data(R * S * TC * TK, 6, 5))
}

/// Scatter reference: each input element adds its filter taps into the output positions it reaches.
fn transposed_reference(x: &[i32], f: &[i32]) -> Vec<i32> {
    let mut out = vec![0; (N * HO * WO * TK) as usize];
    for n in 0..N {
        for ih in 0..TH {
            for iw in 0..TW {
                for r in 0..R {
                    for s in 0..S {
                        let (oh, ow) = (ih * SH + r * DH - TPAD, iw * SW + s * DW - TPAD);
                        if !(0..HO).contains(&oh) || !(0..WO).contains(&ow) {
                            continue;
                        }
                        for c in 0..TC {
                            for k in 0..TK {
                                out[idx(&[N, HO, WO, TK], &[n, oh, ow, k])] += x
                                    [idx(&[N, TH, TW, TC], &[n, ih, iw, c])]
                                    * f[idx(&[R, S, TC, TK], &[r, s, c, k])];
                            }
                        }
                    }
                }
            }
        }
    }
    out
}

fn transposed_run(mutant: Mutant) -> Vec<i32> {
    let (x, f) = transposed_inputs();
    run(
        &transposed_fprop(mutant),
        buf(&[N, TH, TW, TC], &x),
        buf(&[R, S, TC, TK], &f),
        &[N, HO, WO, TK],
    )
}

#[test]
fn the_transposed_convolution_matches_the_scatter_reference_including_output_padding() {
    let (x, f) = transposed_inputs();
    let want = transposed_reference(&x, &f);
    assert_eq!(transposed_run(Mutant::None), want);
    // the output padding adds real positions: the last row and column are reached by a tap, so dropping them would lose data
    let last_row = (0..N * WO * TK).any(|i| {
        want[(HO - 1) as usize * (WO * TK) as usize + i as usize % (WO * TK) as usize] != 0
    });
    assert!(last_row, "the output-padding row is not all zeros");
    assert_eq!((OPH, OPW), (1, 1));
}

#[test]
fn the_transposed_padding_sign_mutant_fails_by_value() {
    let (x, f) = transposed_inputs();
    assert_ne!(
        transposed_run(Mutant::PadSign),
        transposed_reference(&x, &f)
    );
}
