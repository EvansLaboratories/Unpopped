// SPDX-License-Identifier: MIT OR Apache-2.0
//! **The emitted C agrees with the kernel interpreter on contractions and windows**
//! (`docs/e1-emit-from-the-kernel-layer.md`, stage E1b): lifted matmul / batched matmul and 1-D
//! pooling kernels go through `CpuC::lower_kernel`, are compiled and run, and their output bytes
//! are compared with `run_kernel`. The serial C is a correctness reference; nothing is timed.

use unpopped::backend::Backend;
use unpopped::ir::{ContractionAxes, OpDef, ReduceOp, reduced};
use unpopped::kernel::{KernelFn, check_kernel};
use unpopped::kernel_lift::{lift_contraction_opdef, lift_window_opdef};
use unpopped::oracle::TypedBuffer;
use unpopped::oracle::kernel_eval::KernelArg;
use unpopped_cpu_c::CpuC;
use unpopped_vocab::ElementKind;

#[path = "support/cc.rs"]
mod cc;
#[path = "support/kernel_data.rs"]
mod kernel_data;
#[path = "support/kernel_run.rs"]
mod kernel_run;
use kernel_data::zeroed;
use kernel_run::{F32, F64, agree, run_c, run_c_mutated, same_bytes};

const VALUES: [f64; 12] = [
    1.0e8,
    1.0,
    -1.0e8,
    0.5,
    2.0,
    f64::NAN,
    1.0,
    2.0,
    f64::INFINITY,
    -0.0,
    0.1,
    -3.0,
];

/// A buffer of `shape` whose element at logical row-major index `i` is `value(i)`, stored with
/// `strides` (so a transposed operand is the same logical data in another layout).
fn buffer(
    dtype: ElementKind,
    shape: &[i64],
    strides: &[i64],
    value: impl Fn(usize) -> f64,
) -> TypedBuffer {
    let size = shape.iter().product::<i64>() as usize;
    let extent = shape
        .iter()
        .zip(strides)
        .map(|(s, st)| (s - 1) * st)
        .sum::<i64>() as usize
        + 1;
    let width = if dtype == F64 { 8 } else { 4 };
    let mut bytes = vec![0u8; extent * width];
    for lin in 0..size {
        let (mut rem, mut off) = (lin as i64, 0i64);
        for (s, st) in shape.iter().zip(strides).rev() {
            off += (rem % s) * st;
            rem /= s;
        }
        let at = off as usize * width;
        let v = value(lin);
        if dtype == F64 {
            bytes[at..at + 8].copy_from_slice(&v.to_le_bytes());
        } else {
            bytes[at..at + 4].copy_from_slice(&(v as f32).to_le_bytes());
        }
    }
    TypedBuffer::new(dtype, shape.to_vec(), strides.to_vec(), bytes)
}

fn value(i: usize) -> f64 {
    VALUES[i % VALUES.len()]
}

fn ints(vs: &[i64]) -> impl Iterator<Item = KernelArg> + '_ {
    vs.iter().map(|&v| KernelArg::Int(i128::from(v)))
}

/// The name of the kernel's accumulator, from the emitted C: the first `double` local declared
/// (the fold's identity), which the positive control and the mutant below confirm.
fn accumulator(k: &KernelFn) -> String {
    let src = CpuC.lower_kernel(k).unwrap().source;
    let name = src
        .lines()
        .find_map(|l| l.trim_start().strip_prefix("double l"))
        .and_then(|l| l.split(" = ").next())
        .unwrap_or_else(|| {
            panic!(
                "no double local in:
{src}"
            )
        });
    format!("l{name}")
}

/// Run `k` on `args` with the accumulator declared `double` (as emitted) or `float` (the mutant)
/// and return the first output element.
fn first_out(tag: &str, k: &KernelFn, args: &[KernelArg], acc: &str, float: bool) -> f32 {
    let ran = run_c_mutated(tag, k, args, |s| {
        if float {
            s.replace(&format!("double {acc} = "), &format!("float {acc} = "))
        } else {
            s
        }
    })
    .expect("a host C compiler");
    f32::from_le_bytes(ran.outs[0][..4].try_into().unwrap())
}

fn matmul_kernel(batched: bool, dtype: ElementKind) -> KernelFn {
    let axes = if batched {
        ContractionAxes::batched_matmul()
    } else {
        ContractionAxes::matmul()
    };
    let op = OpDef::contraction("mm", &[dtype], axes, reduced(0));
    let k = lift_contraction_opdef(&op, dtype).expect("the contraction lifts");
    assert_eq!(check_kernel(&k), Ok(()));
    k
}

/// `[3,5] x [5,4]` with the given operand strides.
fn matmul_args(dtype: ElementKind, lhs: [i64; 2], rhs: [i64; 2]) -> Vec<KernelArg> {
    let mut a = vec![
        KernelArg::Buffer(buffer(dtype, &[3, 5], &lhs, value)),
        KernelArg::Buffer(buffer(dtype, &[5, 4], &rhs, |i| value(i + 3))),
        KernelArg::Buffer(zeroed(dtype, &[3, 4])),
    ];
    a.extend(ints(&[3, 4, 5]));
    a
}

#[test]
fn matmul_agrees_dense_and_transposed() {
    for dtype in [F32, F64] {
        let k = matmul_kernel(false, dtype);
        let a = matmul_args(dtype, [5, 1], [4, 1]);
        agree(&format!("mm_{dtype:?}"), &k, a);
    }
    // Column-major operands: a lift assuming dense row-major would pass the case above only.
    let k = matmul_kernel(false, F32);
    agree("mm_t", &k, matmul_args(F32, [1, 3], [1, 5]));
}

#[test]
fn batched_matmul_agrees() {
    let k = matmul_kernel(true, F32);
    let mut a = vec![
        KernelArg::Buffer(buffer(F32, &[2, 3, 5], &[15, 5, 1], value)),
        KernelArg::Buffer(buffer(F32, &[2, 5, 4], &[20, 4, 1], |i| value(i + 3))),
        KernelArg::Buffer(zeroed(F32, &[2, 3, 4])),
    ];
    a.extend(ints(&[2, 3, 4, 5]));
    agree("bmm", &k, a);
}

#[test]
fn the_contraction_accumulator_is_f64_in_the_emitted_c() {
    // lhs row (1e8, 1, -1e8, 0.5, 2) times a column of ones: the f64 sum is 3.5, an f32 sum 2.5.
    let k = matmul_kernel(false, F32);
    let mut a = vec![
        KernelArg::Buffer(buffer(F32, &[1, 5], &[5, 1], value)),
        KernelArg::Buffer(buffer(F32, &[5, 1], &[1, 1], |_| 1.0)),
        KernelArg::Buffer(zeroed(F32, &[1, 1])),
    ];
    a.extend(ints(&[1, 1, 5]));
    let acc = accumulator(&k);
    assert_eq!(first_out("mm_acc", &k, &a, &acc, false), 3.5);
    assert_eq!(first_out("mm_acc", &k, &a, &acc, true), 2.5);
}

fn window_kernel(wop: ReduceOp, dtype: ElementKind, geo: [u8; 4], include_pad: bool) -> KernelFn {
    let [size, stride, dilation, pad] = geo;
    let op = OpDef::window_simple(
        "win",
        &[dtype],
        wop,
        1,
        size,
        stride,
        dilation,
        pad,
        pad,
        include_pad,
    );
    let k = lift_window_opdef(&op, dtype, 2).expect("the window lifts");
    assert_eq!(check_kernel(&k), Ok(()));
    k
}

fn window_args(dtype: ElementKind, rows: i64, k_in: i64, k_out: i64) -> Vec<KernelArg> {
    let mut a = vec![
        KernelArg::Buffer(buffer(dtype, &[rows, k_in], &[k_in, 1], value)),
        KernelArg::Buffer(zeroed(dtype, &[rows, k_out])),
    ];
    a.extend(ints(&[rows, k_in, k_out]));
    a
}

#[test]
fn every_pool_agrees_with_both_mean_divisors() {
    // Padded and strided: k_out = (9 + 2 - 3) / 2 + 1 = 5, and the edge windows hold fewer valid
    // taps than `size`, the only place the two `Mean` divisors differ.
    let mut means = Vec::new();
    for (wop, include) in [
        (ReduceOp::Sum, false),
        (ReduceOp::Max, false),
        (ReduceOp::Min, false),
        (ReduceOp::Mean, false),
        (ReduceOp::Mean, true),
    ] {
        let k = window_kernel(wop, F32, [3, 2, 1, 1], include);
        let tag = format!("win_{wop:?}_{include}");
        agree(&tag, &k, window_args(F32, 3, 9, 5));
        if wop == ReduceOp::Mean {
            let ran = run_c(&tag, &k, &window_args(F32, 3, 9, 5)).expect("a host C compiler");
            means.push(ran.outs[0].clone());
        }
    }
    assert_ne!(means[0], means[1], "the two divisors give different bytes");
}

#[test]
fn dilation_and_f64_agree() {
    // size 3, stride 3, dilation 2, pad 2: k_out = (12 + 4 - 5) / 3 + 1 = 4.
    for dtype in [F32, F64] {
        let k = window_kernel(ReduceOp::Sum, dtype, [3, 3, 2, 2], false);
        let tag = format!("win_dil_{dtype:?}");
        agree(&tag, &k, window_args(dtype, 2, 12, 4));
    }
}

#[test]
fn the_window_accumulator_is_f64_in_the_emitted_c() {
    // The first window is (1e8, 1, -1e8): f64 keeps the 1, a float fold loses it.
    let k = window_kernel(ReduceOp::Sum, F32, [3, 3, 1, 0], false);
    let a = window_args(F32, 1, 9, 3);
    let acc = accumulator(&k);
    assert_eq!(first_out("win_acc", &k, &a, &acc, false), 1.0);
    assert_eq!(first_out("win_acc", &k, &a, &acc, true), 0.0);
}

fn f32s(vs: &[f32]) -> Vec<Vec<u8>> {
    vec![vs.iter().flat_map(|v| v.to_le_bytes()).collect()]
}

fn f64s(vs: &[f64]) -> Vec<Vec<u8>> {
    vec![vs.iter().flat_map(|v| v.to_le_bytes()).collect()]
}

fn agrees(got: &[Vec<u8>], want: &[Vec<u8>], dtype: ElementKind) -> bool {
    std::panic::catch_unwind(|| same_bytes("nan", got, want, &[dtype], 0)).is_ok()
}

#[test]
fn nan_is_equal_only_to_nan() {
    // The loosening: a NaN of either sign equals a NaN of either sign (positive control).
    let neg32 = f32::from_bits(0xffc0_0000);
    let neg64 = f64::from_bits(0xfff8_0000_0000_0000);
    assert!(agrees(&f32s(&[neg32]), &f32s(&[f32::NAN]), F32));
    assert!(agrees(&f64s(&[neg64]), &f64s(&[f64::NAN]), F64));
    // Everything else still fails, in both directions and both widths.
    for (a, b) in [
        (f32::NAN, 1.0),
        (1.0, f32::NAN),
        (f32::INFINITY, f32::NAN),
        (f32::NAN, f32::NEG_INFINITY),
        (0.0, f32::NAN),
        (-0.0, 0.0),
        (f32::INFINITY, f32::NEG_INFINITY),
    ] {
        assert!(!agrees(&f32s(&[a]), &f32s(&[b]), F32), "{a:?} vs {b:?}");
    }
    // An integer buffer is compared byte for byte whatever its bits look like as a float: these two
    // are both NaN as `f32` and different as `i32` (the dtype is the buffer's, not the kernel's).
    let (x, y) = (0x7fc0_0000_i32, 0xffc0_0000_u32 as i32);
    let ints = |v: i32| vec![v.to_le_bytes().to_vec()];
    assert!(!agrees(&ints(x), &ints(y), ElementKind::I32));
    assert!(agrees(&ints(x), &ints(x), ElementKind::I32));
    for (a, b) in [(f64::NAN, 1.0), (f64::INFINITY, f64::NAN), (-0.0, 0.0)] {
        assert!(!agrees(&f64s(&[a]), &f64s(&[b]), F64), "{a:?} vs {b:?}");
    }
}
