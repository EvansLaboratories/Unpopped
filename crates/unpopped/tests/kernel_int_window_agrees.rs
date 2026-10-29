// SPDX-License-Identifier: MIT OR Apache-2.0
//! **The layers agree on integer windows** (`docs/ir-kernel-layer.md` section 7): a 1-D
//! `Access::Window` pool (`Sum`, `Max`, `Min`) at `I32` and `I64`, lifted into the kernel layer and
//! run by the kernel interpreter, produces byte for byte what `oracle::evaluate` produces.
//!
//! The oracle folds the valid taps exactly and wraps. The corpus holds wrap-around, negatives and
//! ties, and the geometries pad, stride and dilate so edge windows hold fewer valid taps than `size`.
//! (The plan refuses a window made only of padding, so the lift's no-valid-tap identity is unreachable
//! through it.)

use unpopped::ir::{OpDef, ReduceOp};
use unpopped::kernel::check_kernel;
use unpopped::kernel_lift::{LiftDecline, lift_window_opdef};
use unpopped::oracle::kernel_eval::{KernelArg, run_kernel};
use unpopped::oracle::{TypedBuffer, evaluate};
use unpopped::plan::try_build_plan;
use unpopped::types::int_range;
use unpopped_vocab::{ArchSku, ElementKind, OpCategory, OperandDesc, try_structure_key};

#[derive(Clone, Copy)]
struct Case {
    rows: i64,
    k_in: i64,
    size: u8,
    stride: u8,
    dilation: u8,
    pad: u8,
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
            false,
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
};
const DILATED: Case = Case {
    rows: 3,
    k_in: 11,
    size: 3,
    stride: 1,
    dilation: 2,
    pad: 2,
};
const INTS: [ElementKind; 2] = [ElementKind::I32, ElementKind::I64];
const OPS: [ReduceOp; 3] = [ReduceOp::Sum, ReduceOp::Max, ReduceOp::Min];

fn width(dtype: ElementKind) -> usize {
    if dtype == ElementKind::I64 { 8 } else { 4 }
}

fn values(dtype: ElementKind) -> Vec<i128> {
    let (lo, hi) = int_range(dtype).expect("an integer dtype");
    vec![hi, 1, -1, 2, hi, 3, lo, -5, -7, -7, 4]
}

fn buffer(dtype: ElementKind, shape: [i64; 2]) -> TypedBuffer {
    let vals = values(dtype);
    let n = (shape[0] * shape[1]) as usize;
    let bytes: Vec<u8> = (0..n)
        .flat_map(|i| vals[i % vals.len()].to_le_bytes()[..width(dtype)].to_vec())
        .collect();
    TypedBuffer::new(dtype, shape.to_vec(), vec![shape[1], 1], bytes)
}

fn both(case: Case, wop: ReduceOp, dtype: ElementKind) -> (Vec<u8>, Vec<u8>) {
    let (rows, k_in, k_out) = (case.rows, case.k_in, case.k_out());
    let src = buffer(dtype, [rows, k_in]);
    let op = case.op(wop, dtype);
    let ops = [
        OperandDesc::new(2, &[rows, k_in], &[k_in, 1], dtype, 256),
        OperandDesc::new(2, &[rows, k_out], &[k_out, 1], dtype, 256),
    ];
    let key = try_structure_key(OpCategory::Pooling, &ops, ArchSku::Sm89).expect("a key");
    let plan = try_build_plan(&op, &key).expect("the integer window plans");
    let want = evaluate(&plan, &ops, std::slice::from_ref(&src), &[])
        .remove(0)
        .bytes;
    let len = (rows * k_out) as usize * width(dtype);

    let kernel = lift_window_opdef(&op, dtype, 2).expect("the integer window lifts");
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
            vec![0xAB; len],
        )),
    ];
    for e in [rows, k_in, k_out] {
        args.push(KernelArg::Int(i128::from(e)));
    }
    run_kernel(&kernel, &mut args).expect("the kernel runs");
    let KernelArg::Buffer(got) = &args[1] else {
        unreachable!()
    };
    (want[..len].to_vec(), got.bytes[..len].to_vec())
}

#[test]
fn integer_pools_agree_in_every_geometry() {
    for dtype in INTS {
        for case in [BASE, DILATED] {
            for wop in OPS {
                let (want, got) = both(case, wop, dtype);
                assert_eq!(
                    got, want,
                    "{dtype:?} {wop:?} size={} stride={} dilation={} pad={}",
                    case.size, case.stride, case.dilation, case.pad
                );
            }
        }
    }
}

fn read(dtype: ElementKind, b: &[u8], i: usize) -> i128 {
    let w = width(dtype);
    let mut x = [0u8; 16];
    x[..w].copy_from_slice(&b[i * w..(i + 1) * w]);
    if x[w - 1] & 0x80 != 0 {
        x[w..].fill(0xff);
    }
    i128::from_le_bytes(x)
}

#[test]
fn a_fold_starts_from_its_first_tap_not_from_zero() {
    // Controls, read from the ORACLE (row 0 of `BASE`, windows of taps `2o-1..=2o+1`): window 4 holds
    // only negatives (-5, -7, -7), so its `Max` is -5, not 0; window 2 holds only positives
    // (2, hi, 3), so its `Min` is 2, not 0. An accumulator that started at 0 gets both wrong.
    for dtype in INTS {
        let (max, _) = both(BASE, ReduceOp::Max, dtype);
        let (min, _) = both(BASE, ReduceOp::Min, dtype);
        assert_eq!(read(dtype, &max, 4), -5, "{dtype:?} all-negative Max");
        assert_eq!(read(dtype, &min, 2), 2, "{dtype:?} all-positive Min");
    }
}

#[test]
fn a_sum_wraps_like_the_oracle() {
    // Control: the wrap actually happens in the corpus (two `hi`s within one window).
    for dtype in INTS {
        let (hi, wrapped) = (
            int_range(dtype).unwrap().1,
            both(DILATED, ReduceOp::Sum, dtype).0,
        );
        let any_wrapped = (0..wrapped.len() / width(dtype)).any(|i| read(dtype, &wrapped, i) < 0);
        assert!(
            hi > 0 && any_wrapped,
            "{dtype:?} a window sum wrapped negative"
        );
    }
}

#[test]
fn forms_the_lift_does_not_cover_decline_with_a_reason() {
    let w = |dtype, wop| BASE.op(wop, dtype);
    // Control: the plain integer pool lifts.
    assert!(lift_window_opdef(&w(ElementKind::I32, ReduceOp::Sum), ElementKind::I32, 2).is_ok());
    assert_eq!(
        lift_window_opdef(&w(ElementKind::I32, ReduceOp::Mean), ElementKind::I32, 2),
        Err(LiftDecline::UnsupportedForm("an integer Mean"))
    );
    assert_eq!(
        lift_window_opdef(&w(ElementKind::I16, ReduceOp::Sum), ElementKind::I16, 2),
        Err(LiftDecline::UnsupportedForm(
            "a window outside F32, F64, I32 and I64"
        ))
    );
}
