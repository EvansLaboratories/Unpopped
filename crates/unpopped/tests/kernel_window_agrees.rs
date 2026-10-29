// SPDX-License-Identifier: MIT OR Apache-2.0
//! **The layers agree on windows** (`docs/ir-kernel-layer.md` section 7): a 1-D `Access::Window`
//! pool (`Sum`, `Mean`, `Max`, `Min`) over the innermost axis, lifted into the kernel layer and
//! run by the kernel interpreter, produces byte for byte what `oracle::evaluate` produces.
//!
//! The oracle folds the valid taps in `f64`, ascending, divides a `Mean` by `size`
//! (`count_include_pad`) or by the valid-tap count, keeps a NaN in `Max`/`Min`, and rounds once at
//! the store. The corpus has an f32-cancelling run (1e8, 1, -1e8), NaN, infinities and signed
//! zeros, and the geometries pad, stride and dilate so that edge windows hold fewer valid taps than
//! `size`: that is the only place the two `Mean` divisors differ.

use unpopped::ir::{OpDef, ReduceOp};
use unpopped::kernel::check_kernel;
use unpopped::kernel_lift::{LiftDecline, lift_window_opdef};
use unpopped::oracle::kernel_eval::{KernelArg, run_kernel};
use unpopped::oracle::{TypedBuffer, evaluate};
use unpopped::plan::try_build_plan;
use unpopped_vocab::{ArchSku, ElementKind, OpCategory, OperandDesc, try_structure_key};

#[derive(Clone, Copy)]
struct Case {
    rows: i64,
    k_in: i64,
    size: u8,
    stride: u8,
    dilation: u8,
    pad: u8,
    include_pad: bool,
}

impl Case {
    fn k_out(&self) -> i64 {
        let span = i64::from(self.dilation) * (i64::from(self.size) - 1) + 1;
        (self.k_in + 2 * i64::from(self.pad) - span) / i64::from(self.stride) + 1
    }

    fn op(&self, wop: ReduceOp, dtype: ElementKind) -> OpDef {
        OpDef::window_simple(
            "win",
            &[dtype],
            wop,
            1,
            self.size,
            self.stride,
            self.dilation,
            self.pad,
            self.pad,
            self.include_pad,
        )
    }
}

const BASE: Case = Case {
    rows: 3,
    k_in: 9,
    size: 3,
    stride: 2,
    dilation: 1,
    pad: 1,
    include_pad: false,
};

const VALUES: [f64; 11] = [
    1.0e8,
    1.0,
    -1.0e8,
    0.5,
    f64::NAN,
    2.0,
    -0.0,
    f64::INFINITY,
    -3.0,
    0.1,
    0.25,
];

fn buffer(dtype: ElementKind, shape: [i64; 2], values: &[f64]) -> TypedBuffer {
    let n = (shape[0] * shape[1]) as usize;
    let bytes: Vec<u8> = (0..n)
        .flat_map(|i| {
            let v = values[i % values.len()];
            match dtype {
                ElementKind::F32 => (v as f32).to_le_bytes().to_vec(),
                _ => v.to_le_bytes().to_vec(),
            }
        })
        .collect();
    TypedBuffer::new(dtype, shape.to_vec(), vec![shape[1], 1], bytes)
}

/// Run the case through the oracle and through the lifted kernel on `values`.
fn both(case: Case, wop: ReduceOp, dtype: ElementKind, values: &[f64]) -> (Vec<u8>, Vec<u8>) {
    let (rows, k_in, k_out) = (case.rows, case.k_in, case.k_out());
    let src = buffer(dtype, [rows, k_in], values);
    let op = case.op(wop, dtype);
    let ops = [
        OperandDesc::new(2, &[rows, k_in], &[k_in, 1], dtype, 256),
        OperandDesc::new(2, &[rows, k_out], &[k_out, 1], dtype, 256),
    ];
    let key = try_structure_key(OpCategory::Pooling, &ops, ArchSku::Sm89).expect("a key");
    let plan = try_build_plan(&op, &key).expect("the window plans");
    let want = evaluate(&plan, &ops, std::slice::from_ref(&src), &[])
        .remove(0)
        .bytes;

    let kernel = lift_window_opdef(&op, dtype, 2).expect("the window lifts");
    assert_eq!(
        check_kernel(&kernel),
        Ok(()),
        "a lifted kernel is well formed"
    );
    let mut args = vec![
        KernelArg::Buffer(src),
        KernelArg::Buffer(TypedBuffer::new(
            dtype,
            vec![rows, k_out],
            vec![k_out, 1],
            vec![0xAB; want.len()],
        )),
    ];
    for e in [rows, k_in, k_out] {
        args.push(KernelArg::Int(i128::from(e)));
    }
    run_kernel(&kernel, &mut args).expect("the kernel runs");
    let KernelArg::Buffer(got) = &args[1] else {
        unreachable!()
    };
    (want.clone(), got.bytes[..want.len()].to_vec())
}

#[test]
fn every_pool_agrees() {
    for wop in [ReduceOp::Sum, ReduceOp::Mean, ReduceOp::Max, ReduceOp::Min] {
        let (want, got) = both(BASE, wop, ElementKind::F32, &VALUES);
        assert_eq!(got, want, "{wop:?}");
    }
}

#[test]
fn the_two_mean_divisors_agree_and_differ() {
    let exclude = both(BASE, ReduceOp::Mean, ElementKind::F32, &VALUES);
    assert_eq!(exclude.1, exclude.0);
    let include_case = Case {
        include_pad: true,
        ..BASE
    };
    let include = both(include_case, ReduceOp::Mean, ElementKind::F32, &VALUES);
    assert_eq!(include.1, include.0);
    // Control: the edge windows hold fewer than `size` valid taps, so the divisors give different
    // bytes; a lift that used one divisor for both cannot pass both halves.
    assert_ne!(exclude.0, include.0);
}

#[test]
fn dilation_and_stride_agree() {
    let case = Case {
        rows: 2,
        k_in: 12,
        size: 3,
        stride: 3,
        dilation: 2,
        pad: 2,
        include_pad: false,
    };
    for wop in [ReduceOp::Sum, ReduceOp::Mean, ReduceOp::Max, ReduceOp::Min] {
        let (want, got) = both(case, wop, ElementKind::F32, &VALUES);
        assert_eq!(got, want, "{wop:?}");
    }
}

#[test]
fn no_padding_agrees() {
    let case = Case {
        pad: 0,
        stride: 1,
        ..BASE
    };
    for wop in [ReduceOp::Sum, ReduceOp::Max] {
        let (want, got) = both(case, wop, ElementKind::F32, &VALUES);
        assert_eq!(got, want, "{wop:?}");
    }
}

#[test]
fn a_nan_is_kept_by_max_and_min() {
    // One NaN per window position in the first row; the later finite values must not replace it.
    let values = [1.0, f64::NAN, 5.0, 4.0, 3.0, 2.0, 7.0, 6.0, 8.0];
    let case = Case {
        rows: 1,
        k_in: 9,
        size: 3,
        stride: 3,
        dilation: 1,
        pad: 0,
        include_pad: false,
    };
    for wop in [ReduceOp::Max, ReduceOp::Min] {
        let (want, got) = both(case, wop, ElementKind::F32, &values);
        assert_eq!(got, want, "{wop:?}");
        let first = f32::from_le_bytes(got[..4].try_into().unwrap());
        assert!(
            first.is_nan(),
            "{wop:?}: the window holding a NaN stays NaN"
        );
    }
}

#[test]
fn the_accumulator_is_f64_and_ascending() {
    // 1e16 + 1 - 1e16 + 1 + ...: order-sensitive in f64, and an f32 accumulator loses the 1s.
    let values = [1.0e16, 1.0, -1.0e16, 1.0, 0.1, 0.2, 0.3, 0.4, 0.5];
    let case = Case {
        rows: 1,
        k_in: 9,
        size: 9,
        stride: 1,
        dilation: 1,
        pad: 0,
        include_pad: false,
    };
    let (want, got) = both(case, ReduceOp::Sum, ElementKind::F64, &values);
    assert_eq!(got, want);
    let asc: f64 = values.iter().sum();
    let desc: f64 = values.iter().rev().sum();
    assert_ne!(
        asc.to_bits(),
        desc.to_bits(),
        "the case discriminates the order"
    );
    assert_eq!(
        f64::from_le_bytes(got[..8].try_into().unwrap()).to_bits(),
        asc.to_bits()
    );
    // f32 compute with f32-cancelling data agrees too.
    let (want32, got32) = both(case, ReduceOp::Sum, ElementKind::F32, &values);
    assert_eq!(got32, want32);
}

#[test]
fn forms_the_lift_does_not_cover_decline_with_a_reason() {
    assert!(
        lift_window_opdef(
            &BASE.op(ReduceOp::Sum, ElementKind::F32),
            ElementKind::F32,
            2
        )
        .is_ok()
    );
    let ew = OpDef::elementwise(
        "e",
        1,
        &[ElementKind::F32],
        unpopped::ir::Expr(unpopped::ir::input(0).0),
    );
    assert_eq!(
        lift_window_opdef(&ew, ElementKind::F32, 2),
        Err(LiftDecline::NotWindow)
    );
    assert_eq!(
        lift_window_opdef(
            &BASE.op(ReduceOp::Sum, ElementKind::I32),
            ElementKind::I32,
            2
        ),
        Err(LiftDecline::UnsupportedForm("a non-float window"))
    );
}
