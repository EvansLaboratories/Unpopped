// SPDX-License-Identifier: MIT OR Apache-2.0
//! **The emitted C agrees with the kernel interpreter on im2col**
//! (`docs/e1-emit-from-the-kernel-layer.md`, stage E1b): a lifted 2-D `Access::Im2Col` is lowered
//! by `CpuC::lower_kernel`, compiled, run, and compared with `run_kernel` (bit-exact except NaN
//! sign/payload). Im2col is a gather with a typed zero where a tap falls outside the input, so
//! every input element is distinct (a NaN and a negative zero planted) and the geometries cover
//! padding, stride, dilation, a non-square window and a rectangular input: a wrong tap, not a
//! wrong value, is what this exposes. The output starts non-zero so an unwritten cell shows. The
//! serial C is a correctness reference; nothing is timed.

use unpopped::backend::Backend;
use unpopped::ir::OpDef;
use unpopped::kernel::{KernelFn, check_kernel};
use unpopped::kernel_lift::lift_im2col_opdef;
use unpopped::oracle::TypedBuffer;
use unpopped::oracle::kernel_eval::KernelArg;
use unpopped_cpu_c::CpuC;
use unpopped_vocab::ElementKind;

#[path = "support/cc.rs"]
mod cc;
#[path = "support/kernel_run.rs"]
mod kernel_run;
use kernel_run::{F32, F64, agree, run_c_mutated};

#[derive(Clone, Copy)]
struct Case {
    /// `[N, C, H, W]`.
    shape: [i64; 4],
    kernel: (u8, u8),
    stride: (u8, u8),
    pad: (u8, u8),
    dilation: (u8, u8),
}

const SQUARE: Case = Case {
    shape: [1, 1, 3, 3],
    kernel: (3, 3),
    stride: (1, 1),
    pad: (1, 1),
    dilation: (1, 1),
};

impl Case {
    fn out_extent(&self, extent: i64, axis: usize) -> i64 {
        let (k, s, p, d) = [
            (self.kernel.0, self.stride.0, self.pad.0, self.dilation.0),
            (self.kernel.1, self.stride.1, self.pad.1, self.dilation.1),
        ][axis];
        let (k, s, p, d) = (i64::from(k), i64::from(s), i64::from(p), i64::from(d));
        (extent + 2 * p - d * (k - 1) - 1) / s + 1
    }
}

fn strides(shape: &[i64]) -> Vec<i64> {
    (0..shape.len())
        .map(|i| shape[i + 1..].iter().product())
        .collect()
}

/// A dense buffer whose element `i` is `value(i)`.
fn buffer(dtype: ElementKind, shape: &[i64], value: impl Fn(usize) -> f64) -> TypedBuffer {
    let n: i64 = shape.iter().product();
    let bytes = (0..n as usize)
        .flat_map(|i| match dtype {
            F32 => (value(i) as f32).to_le_bytes().to_vec(),
            _ => value(i).to_le_bytes().to_vec(),
        })
        .collect();
    TypedBuffer::new(dtype, shape.to_vec(), strides(shape), bytes)
}

/// Distinct values, with a NaN and a negative zero planted.
fn value(i: usize) -> f64 {
    match i {
        2 => f64::NAN,
        5 => -0.0,
        _ => (i + 1) as f64,
    }
}

fn lifted(case: Case, dtype: ElementKind) -> (KernelFn, Vec<KernelArg>) {
    let [n, c, h, w] = case.shape;
    let (oh, ow) = (case.out_extent(h, 0), case.out_extent(w, 1));
    let rows = c * i64::from(case.kernel.0) * i64::from(case.kernel.1);
    let op = OpDef::im2col_2d(
        "im2col",
        dtype,
        case.kernel,
        case.stride,
        case.pad,
        case.dilation,
    );
    let k = lift_im2col_opdef(&op, dtype).expect("the im2col lifts");
    assert_eq!(check_kernel(&k), Ok(()));
    let out = buffer(dtype, &[n, rows, oh * ow], |_| 7.0);
    let mut args = vec![
        KernelArg::Buffer(buffer(dtype, &case.shape, value)),
        KernelArg::Buffer(out),
    ];
    args.extend([n, c, h, w, oh, ow].map(|e| KernelArg::Int(i128::from(e))));
    (k, args)
}

#[test]
fn every_geometry_agrees() {
    let cases = [
        ("padded", SQUARE),
        (
            "batch_channels",
            Case {
                shape: [2, 3, 4, 4],
                ..SQUARE
            },
        ),
        (
            "stride2",
            Case {
                shape: [1, 2, 5, 5],
                stride: (2, 2),
                ..SQUARE
            },
        ),
        (
            "dilated",
            Case {
                shape: [1, 1, 6, 6],
                dilation: (2, 2),
                pad: (2, 2),
                ..SQUARE
            },
        ),
        (
            // Different extents, kernels, strides, pads and dilations per axis: a lift that
            // swapped the axes or used one axis's parameter for both disagrees here.
            "rectangular",
            Case {
                shape: [1, 2, 5, 7],
                kernel: (2, 3),
                stride: (1, 2),
                pad: (0, 2),
                dilation: (2, 1),
            },
        ),
    ];
    for (tag, case) in cases {
        let (k, a) = lifted(case, F32);
        agree(&format!("im2col_{tag}"), &k, a);
    }
    let (k, a) = lifted(SQUARE, F64);
    agree("im2col_f64", &k, a);
}

/// What the C reports for `k` on `a` once `from` is replaced by `to` in its source; with an empty
/// `from`, the unmutated C.
fn reports(tag: &str, k: &KernelFn, a: &[KernelArg], from: &str, to: &str) -> (i32, Vec<Vec<u8>>) {
    let ran = run_c_mutated(tag, k, a, |s| {
        if from.is_empty() {
            return s;
        }
        assert!(
            s.contains(from),
            "{tag}: the anchor `{from}` is in the emitted C:\n{s}"
        );
        s.replace(from, to)
    })
    .expect("a host C compiler");
    (ran.status, ran.outs)
}

#[test]
fn the_corpus_catches_a_wrong_tap_bound_or_pad_value() {
    // Seeded mutants of the emitted C; each must change what the C reports (the unmutated C agrees
    // with the interpreter above). The anchors are read from the emitter's own output.
    let (k, a) = lifted(SQUARE, F32);
    let src = CpuC.lower_kernel(&k).unwrap().source;
    let base = reports("im2col_base", &k, &a, "", "");
    // `(a)` is the tap test: the coordinate clamped into [0, extent-1] differs from itself.
    let (up_h, up_w) = (
        "(unsigned int)(l2) - (unsigned int)(((int)(1LL)))",
        "(unsigned int)(l3) - (unsigned int)(((int)(1LL)))",
    );
    for (tag, from, to) in [
        // The clamp's upper limit one too high, for the row and for the column.
        (
            "im2col_row_hi",
            up_h,
            "(unsigned int)(l2) - (unsigned int)(((int)(0LL)))",
        ),
        (
            "im2col_col_hi",
            up_w,
            "(unsigned int)(l3) - (unsigned int)(((int)(0LL)))",
        ),
        // The tap test inverted, for the row and for the column.
        ("im2col_row_inv", "(l12)))) != 0", "(l12)))) == 0"),
        ("im2col_col_inv", "(l13)))) != 0", "(l13)))) == 0"),
        // The typed zero of a tap outside the input.
        (
            "im2col_pad_value",
            "float l14 = (float)(((float)0.0));",
            "float l14 = (float)(((float)1.0));",
        ),
    ] {
        assert!(
            src.contains(from),
            "{tag}: the anchor `{from}` is in the emitted C"
        );
        assert_ne!(
            reports(tag, &k, &a, from, to),
            base,
            "{tag}: went unnoticed"
        );
    }
}
