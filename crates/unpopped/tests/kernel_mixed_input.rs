// SPDX-License-Identifier: MIT OR Apache-2.0
//! **Mixed-input GEMM with floating-point scales** (CUTLASS survey rows 16 and 17; the kernel is `support/mixed_input.rs`).
//! `I8` and packed `I4` weights against `F16`, `Bf16` and `F32` activations, with `F16`, `Bf16` and `F32` scales, in the
//! scale-only and scale-plus-zero-point modes, group size 4. The reference dequantizes the WHOLE weight matrix first and then
//! multiplies (the kernel dequantizes inside the loop), decoding the narrow types with the `half` crate, a codec independent of the
//! oracle's. The corpora include the `I8` extremes (-128, 127), a NaN activation, and an all-negative weight block. Mutants
//! (scale per row, zero after the scale, zero sign) fail by value.

use half::{bf16, f16};
use unpopped::kernel::check_kernel;
use unpopped::oracle::TypedBuffer;
use unpopped::oracle::kernel_eval::KernelArg;
use unpopped::oracle::kernel_launch::run_launch;
use unpopped_vocab::ElementKind;

#[path = "support/mixed_input.rs"]
mod fixture;
use fixture::{C, F32, G, GROUP, K, Mode, Mutant, R, mixed_input_gemm};

fn enc(kind: ElementKind, v: f32) -> Vec<u8> {
    match kind {
        ElementKind::F16 => f16::from_f32(v).to_le_bytes().to_vec(),
        ElementKind::Bf16 => bf16::from_f32(v).to_le_bytes().to_vec(),
        ElementKind::F32 => v.to_le_bytes().to_vec(),
        other => panic!("{other:?}"),
    }
}

/// The value a narrow float holds after rounding `v` to it.
fn round(kind: ElementKind, v: f32) -> f32 {
    match kind {
        ElementKind::F16 => f16::from_f32(v).to_f32(),
        ElementKind::Bf16 => bf16::from_f32(v).to_f32(),
        _ => v,
    }
}

fn floats(kind: ElementKind, v: &[f32]) -> KernelArg {
    let bytes = v.iter().flat_map(|x| enc(kind, *x)).collect();
    KernelArg::Buffer(TypedBuffer::new(kind, vec![v.len() as i64], vec![1], bytes))
}

struct Case {
    /// Integer weights, logical values.
    w: Vec<i32>,
    scale: Vec<f32>,
    zero: Vec<f32>,
    x: Vec<f32>,
}

fn case(wk: ElementKind, nan: bool, negative: bool) -> Case {
    let (lo, span) = if wk == ElementKind::I4 {
        (-8, 16)
    } else {
        (-128, 256)
    };
    let mut w: Vec<i32> = (0..R * K)
        .map(|i| ((i * 37 + 11) % span as usize) as i32 + lo)
        .collect();
    w[0] = lo;
    w[R * K - 1] = lo + span - 1;
    if negative {
        w.iter_mut().for_each(|v| *v = -(v.abs().max(1)).min(-lo));
    }
    Case {
        w,
        scale: (0..R * G).map(|i| [0.5, 0.25, 1.5, 0.375][i % 4]).collect(),
        zero: (0..R * G).map(|i| [1.0, -2.0, 0.0, 3.0][i % 4]).collect(),
        x: (0..K * C)
            .map(|i| {
                if nan && i == 5 * C + 2 {
                    f32::NAN
                } else {
                    ((i * 7 % 9) as f32 - 4.0) * 0.5
                }
            })
            .collect(),
    }
}

fn weights(wk: ElementKind, w: &[i32]) -> KernelArg {
    let bytes: Vec<u8> = if wk == ElementKind::I4 {
        w.chunks(2)
            .map(|p| (p[0] & 0xf) as u8 | (((p[1] & 0xf) as u8) << 4))
            .collect()
    } else {
        w.iter().map(|v| *v as i8 as u8).collect()
    };
    KernelArg::Buffer(TypedBuffer::new(wk, vec![w.len() as i64], vec![1], bytes))
}

/// Dequantize the whole matrix, then multiply.
fn reference(c: &Case, sk: ElementKind, ak: ElementKind, mode: Mode) -> Vec<f32> {
    let dq: Vec<f32> = (0..R * K)
        .map(|i| {
            let g = i / K * G + i % K / GROUP;
            let (s, z) = (round(sk, c.scale[g]), round(sk, c.zero[g]));
            match mode {
                Mode::Scale => c.w[i] as f32 * s,
                Mode::ScaleZero => (c.w[i] as f32 - z) * s,
            }
        })
        .collect();
    let mut out = vec![0.0; R * C];
    for r in 0..R {
        for col in 0..C {
            let mut acc = 0.0f32;
            for k in 0..K {
                acc += dq[r * K + k] * round(ak, c.x[k * C + col]);
            }
            out[r * C + col] = acc;
        }
    }
    out
}

fn run(
    c: &Case,
    (wk, sk, ak): (ElementKind, ElementKind, ElementKind),
    mode: Mode,
    mutant: Mutant,
) -> Vec<f32> {
    let k = mixed_input_gemm(wk, sk, ak, mode, mutant);
    assert_eq!(check_kernel(&k), Ok(()), "{wk:?} {sk:?} {ak:?} {mode:?}");
    let mut args = vec![
        weights(wk, &c.w),
        floats(sk, &c.scale),
        floats(ak, &c.x),
        floats(F32, &[0.0; R * C]),
    ];
    if mode == Mode::ScaleZero {
        args.push(floats(sk, &c.zero));
    }
    run_launch(&k, &mut args).expect("the oracle runs it");
    let KernelArg::Buffer(b) = &args[3] else {
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

const COMBOS: [(ElementKind, ElementKind, ElementKind); 5] = [
    (ElementKind::I8, ElementKind::F16, ElementKind::F16),
    (ElementKind::I8, ElementKind::Bf16, ElementKind::F32),
    (ElementKind::I8, ElementKind::F32, ElementKind::Bf16),
    (ElementKind::I4, ElementKind::F32, ElementKind::F16),
    (ElementKind::I4, ElementKind::F16, ElementKind::Bf16),
];

#[test]
fn every_weight_scale_activation_combination_matches_the_dequantize_first_reference() {
    for combo in COMBOS {
        for mode in [Mode::Scale, Mode::ScaleZero] {
            for (name, nan, neg) in [
                ("ordinary", false, false),
                ("nan", true, false),
                ("negative", false, true),
            ] {
                let c = case(combo.0, nan, neg);
                let (got, want) = (
                    run(&c, combo, mode, Mutant::None),
                    reference(&c, combo.1, combo.2, mode),
                );
                assert!(
                    same(&got, &want),
                    "{combo:?} {mode:?} {name}: {got:?} vs {want:?}"
                );
            }
        }
    }
}

#[test]
fn the_corpus_is_not_blind() {
    let c = case(ElementKind::I8, false, false);
    assert!(c.w.contains(&-128) && c.w.contains(&127));
    // the NaN reaches exactly one output column
    let n = case(ElementKind::I8, true, false);
    let want = reference(&n, ElementKind::F16, ElementKind::F16, Mode::Scale);
    let cols: Vec<usize> = (0..C)
        .filter(|col| (0..R).any(|r| want[r * C + col].is_nan()))
        .collect();
    assert_eq!(cols, vec![2]);
    // the zero point and the scale both move the result
    let base = reference(&c, ElementKind::F32, ElementKind::F32, Mode::ScaleZero);
    assert!(!same(
        &base,
        &reference(&c, ElementKind::F32, ElementKind::F32, Mode::Scale)
    ));
}

#[test]
fn the_scale_and_zero_mutants_fail_by_value() {
    let combo = COMBOS[0];
    let c = case(combo.0, false, false);
    let want = reference(&c, combo.1, combo.2, Mode::ScaleZero);
    for m in [
        Mutant::ScalePerRow,
        Mutant::ZeroAfterScale,
        Mutant::ZeroSign,
    ] {
        assert!(!same(&run(&c, combo, Mode::ScaleZero, m), &want), "{m:?}");
    }
    let want = reference(&c, combo.1, combo.2, Mode::Scale);
    assert!(
        !same(&run(&c, combo, Mode::Scale, Mutant::ScalePerRow), &want),
        "per-row scale, scale-only"
    );
}
