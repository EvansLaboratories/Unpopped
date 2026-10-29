// SPDX-License-Identifier: MIT OR Apache-2.0
//! **The layers agree on `bincount`** (`docs/ir-kernel-layer.md` section 7): the one scatter whose
//! output dtype is not its compute dtype. `OpDef::bincount` reads one integer tensor `x` that is
//! also its own index, adds the constant `1` to `out[x[i]]` for every in-range `x[i]`, and writes
//! an `I32` count. `lift_scatter_opdef` lifts it with a `ConstInt` increment at the count dtype,
//! and the lifted kernel produces byte for byte what `oracle::evaluate` produces.
//!
//! The data holds duplicates (so a count above 1 is observable), values below zero and at and past
//! the bin count (skipped), and a bin nobody hits (so a destination that was not left alone, or
//! one that counted a skipped value, shows).

use unpopped::ir::{OpDef, ScalarExpr, WriteCombine, WriteIndex};
use unpopped::kernel::check_kernel;
use unpopped::kernel_lift::{LiftDecline, lift_scatter_opdef};
use unpopped::oracle::kernel_eval::{KernelArg, run_kernel};
use unpopped::oracle::{TypedBuffer, evaluate};
use unpopped::plan::try_build_plan;
use unpopped_vocab::{ArchSku, ElementKind, OpCategory, OperandDesc, try_structure_key};

const BINS: i64 = 6;

fn width(dtype: ElementKind) -> usize {
    if dtype == ElementKind::I64 { 8 } else { 4 }
}

/// The data: bin 4 is never hit, 0 and 2 repeat, and the rest are out of range.
fn data(dtype: ElementKind) -> TypedBuffer {
    let vals: [i64; 14] = [0, 2, 2, 0, 5, 1, 0, -1, 6, 9, 2, 1, -7, 0];
    let w = width(dtype);
    let bytes: Vec<u8> = vals
        .iter()
        .flat_map(|v| v.to_le_bytes()[..w].to_vec())
        .collect();
    TypedBuffer::new(dtype, vec![vals.len() as i64], vec![1], bytes)
}

fn counts(b: &[u8]) -> Vec<i32> {
    b.chunks(4)
        .map(|c| i32::from_le_bytes(c.try_into().unwrap()))
        .collect()
}

fn both(dtype: ElementKind) -> (Vec<u8>, Vec<u8>) {
    let x = data(dtype);
    let ops = [
        OperandDesc::new(1, &x.shape, &[1], dtype, 256),
        OperandDesc::new(1, &[BINS], &[1], ElementKind::I32, 256),
    ];
    let op = OpDef::bincount("bc", dtype);
    let key = try_structure_key(OpCategory::UnaryElementwise, &ops, ArchSku::Sm89).expect("a key");
    let plan = try_build_plan(&op, &key).expect("the bincount plans");
    let len = BINS as usize * 4;
    let want = evaluate(&plan, &ops, std::slice::from_ref(&x), &[])
        .remove(0)
        .bytes[..len]
        .to_vec();

    let kernel = lift_scatter_opdef(&op, dtype, 1).expect("the bincount lifts");
    assert_eq!(
        check_kernel(&kernel),
        Ok(()),
        "a lifted kernel is well formed"
    );
    let n = x.shape[0];
    let mut args = vec![
        KernelArg::Buffer(x),
        KernelArg::Buffer(TypedBuffer::new(
            ElementKind::I32,
            vec![BINS],
            vec![1],
            vec![0; len],
        )),
        KernelArg::Int(i128::from(n)),
        KernelArg::Int(i128::from(BINS)),
    ];
    run_kernel(&kernel, &mut args).expect("the kernel runs");
    let KernelArg::Buffer(out) = &args[1] else {
        unreachable!()
    };
    (want, out.bytes[..len].to_vec())
}

#[test]
fn bincount_agrees_at_both_index_widths() {
    for dtype in [ElementKind::I32, ElementKind::I64] {
        let (want, got) = both(dtype);
        assert_eq!(got, want, "{dtype:?}");
    }
}

#[test]
fn the_corpus_exercises_what_the_lift_must_get_right() {
    // Controls, read from the ORACLE: three zeros, two ones, three twos, a bin nobody hits, and the
    // skipped values (-1, -7, 6, 9) counted nowhere, so the total is the 10 in-range values.
    let (want, _) = both(ElementKind::I32);
    assert_eq!(counts(&want), vec![4, 2, 3, 0, 0, 1]);
    assert_eq!(counts(&want).iter().sum::<i32>(), 10);
}

#[test]
fn only_a_counting_per_output_dtype_lifts() {
    // Control: bincount lifts.
    let bc = OpDef::bincount("bc", ElementKind::I32);
    assert!(lift_scatter_opdef(&bc, ElementKind::I32, 1).is_ok());
    // A count that is not an integer constant, a non-add combine and a float count all decline.
    let mut half = bc.clone();
    half.body = ScalarExpr::Const(0.5);
    let mut max = bc.clone();
    max.write_index = WriteIndex::ScatterIndexed {
        index_operand: 0,
        axis: 0,
        combine: WriteCombine::AtomicMax,
        oob: unpopped::ir::OobPolicy::Skip,
        index_dtype: ElementKind::I32,
    };
    let mut float_out = bc.clone();
    float_out.out_dtype = Some(ElementKind::F32);
    for (name, op) in [("0.5", half), ("max", max), ("float out", float_out)] {
        assert_eq!(
            lift_scatter_opdef(&op, ElementKind::I32, 1),
            Err(LiftDecline::UnsupportedForm(
                "a per-output dtype that is not a count"
            )),
            "{name}"
        );
    }
}
