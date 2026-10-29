// SPDX-License-Identifier: MIT OR Apache-2.0
//! **The layers agree on im2col** (`docs/ir-kernel-layer.md` section 7): a 2-D `Access::Im2Col`
//! lifted into the kernel layer and run by the kernel interpreter produces byte for byte what
//! `oracle::evaluate` produces.
//!
//! Im2Col is a raw-bit gather with a typed zero where a tap falls outside the input, so the corpus
//! is built to expose a wrong tap, not a wrong value: every input element is distinct (its
//! linear index plus one), with a NaN and a negative zero planted, and the geometries cover
//! padding, stride, dilation, a non-square window and a rectangular input. A lift that swapped
//! the row and column of a tap, or tested `<` where it needed `<=`, agrees on none of them.

use unpopped::ir::OpDef;
use unpopped::kernel::check_kernel;
use unpopped::kernel_lift::{LiftDecline, lift_im2col_opdef};
use unpopped::oracle::kernel_eval::{KernelArg, run_kernel};
use unpopped::oracle::{TypedBuffer, evaluate};
use unpopped::plan::try_build_plan;
use unpopped_vocab::{ArchSku, ElementKind, OpCategory, OperandDesc, try_structure_key};

#[derive(Clone, Copy)]
struct Case {
    /// `[N, C, H, W]`.
    shape: [i64; 4],
    kernel: (u8, u8),
    stride: (u8, u8),
    pad: (u8, u8),
    dilation: (u8, u8),
}

impl Case {
    fn out_extent(&self, extent: i64, axis: usize) -> i64 {
        let (k, s, p, d) = [
            (self.kernel.0, self.stride.0, self.pad.0, self.dilation.0),
            (self.kernel.1, self.stride.1, self.pad.1, self.dilation.1),
        ][axis];
        let (k, s, p, d) = (i64::from(k), i64::from(s), i64::from(p), i64::from(d));
        (extent + 2 * p - d * (k - 1) - 1) / s + 1
    }

    fn op(&self, dtype: ElementKind) -> OpDef {
        OpDef::im2col_2d("im2col", dtype, self.kernel, self.stride, self.pad, self.dilation)
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
    let bytes: Vec<u8> = (0..n as usize)
        .flat_map(|i| match dtype {
            ElementKind::F32 => (value(i) as f32).to_le_bytes().to_vec(),
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

/// Run `case` through the oracle and through the lifted kernel.
fn both(case: Case, dtype: ElementKind) -> (Vec<u8>, Vec<u8>) {
    let [n, c, h, w] = case.shape;
    let (oh, ow) = (case.out_extent(h, 0), case.out_extent(w, 1));
    let rows = c * i64::from(case.kernel.0) * i64::from(case.kernel.1);
    let src = buffer(dtype, &case.shape, value);
    let out_shape = [n, rows, oh * ow];
    let op = case.op(dtype);

    let ops = [
        OperandDesc::new(4, &case.shape, &strides(&case.shape), dtype, 256),
        OperandDesc::new(3, &out_shape, &strides(&out_shape), dtype, 256),
    ];
    let key = try_structure_key(OpCategory::Convolution, &ops, ArchSku::Sm89).expect("a key");
    let plan = try_build_plan(&op, &key).expect("the im2col plans");
    let want = evaluate(&plan, &ops, &[src.clone()], &[]).remove(0).bytes;

    let kernel = lift_im2col_opdef(&op, dtype).expect("the im2col lifts");
    assert_eq!(check_kernel(&kernel), Ok(()), "a lifted kernel is well formed");
    let elem = want.len() / out_shape.iter().product::<i64>() as usize;
    let mut args = vec![
        KernelArg::Buffer(src),
        KernelArg::Buffer(TypedBuffer::new(
            dtype,
            out_shape.to_vec(),
            strides(&out_shape),
            vec![0xAB; want.len().max(elem)],
        )),
    ];
    for e in [n, c, h, w, oh, ow] {
        args.push(KernelArg::Int(i128::from(e)));
    }
    run_kernel(&kernel, &mut args).expect("the kernel runs");
    let KernelArg::Buffer(got) = &args[1] else {
        unreachable!()
    };
    (want.clone(), got.bytes[..want.len()].to_vec())
}

const SQUARE: Case = Case {
    shape: [1, 1, 3, 3],
    kernel: (3, 3),
    stride: (1, 1),
    pad: (1, 1),
    dilation: (1, 1),
};

#[test]
fn a_padded_3x3_agrees() {
    let (want, got) = both(SQUARE, ElementKind::F32);
    assert_eq!(got, want);
    // Control: the case does read outside the input (the corner tap of cell 0 is a pad zero), so
    // an always-in-bounds lift cannot pass it.
    assert!(want[..4].iter().all(|b| *b == 0), "top-left tap pads");
}

#[test]
fn batch_and_channels_agree() {
    let case = Case {
        shape: [2, 3, 4, 4],
        ..SQUARE
    };
    let (want, got) = both(case, ElementKind::F32);
    assert_eq!(got, want);
}

#[test]
fn stride_two_agrees() {
    let case = Case {
        shape: [1, 2, 5, 5],
        stride: (2, 2),
        ..SQUARE
    };
    let (want, got) = both(case, ElementKind::F32);
    assert_eq!(got, want);
}

#[test]
fn dilation_agrees() {
    let case = Case {
        shape: [1, 1, 6, 6],
        dilation: (2, 2),
        pad: (2, 2),
        ..SQUARE
    };
    let (want, got) = both(case, ElementKind::F32);
    assert_eq!(got, want);
}

#[test]
fn a_rectangular_input_and_window_agree() {
    // Different heights and widths, kernels, strides, pads and dilations on the two axes: a lift
    // that swapped the axes, or used one axis's parameter for both, disagrees here.
    let case = Case {
        shape: [1, 2, 5, 7],
        kernel: (2, 3),
        stride: (1, 2),
        pad: (0, 2),
        dilation: (2, 1),
    };
    let (want, got) = both(case, ElementKind::F32);
    assert_eq!(got, want);
}

#[test]
fn f64_agrees() {
    let (want, got) = both(SQUARE, ElementKind::F64);
    assert_eq!(got, want);
}

#[test]
fn forms_the_lift_does_not_cover_decline_with_a_reason() {
    assert!(lift_im2col_opdef(&SQUARE.op(ElementKind::F32), ElementKind::F32).is_ok());
    let ew = OpDef::elementwise(
        "e",
        1,
        &[ElementKind::F32],
        unpopped::ir::Expr(unpopped::ir::input(0).0),
    );
    assert_eq!(
        lift_im2col_opdef(&ew, ElementKind::F32),
        Err(LiftDecline::NotIm2Col)
    );
    assert_eq!(
        lift_im2col_opdef(&SQUARE.op(ElementKind::I32), ElementKind::I32),
        Err(LiftDecline::UnsupportedForm("a non-float im2col"))
    );
}
