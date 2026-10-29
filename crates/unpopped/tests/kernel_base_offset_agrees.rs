// SPDX-License-Identifier: MIT OR Apache-2.0
//! **The layers agree on runtime base offsets** (`docs/ir-kernel-layer.md` section 7): an
//! Elementwise op carrying `BaseOffset::Runtime` on some inputs and/or its output, lifted by
//! `lift_elementwise_opdef` and run by the kernel interpreter, produces what `oracle::evaluate`
//! produces.
//!
//! A runtime offset is a launch argument added to an operand's base before any address math. In the
//! kernel layer that is the argument buffer's `base_offset`, so the lifted kernel is the offset-free
//! one and the caller binds the offset. Each buffer here sits in storage padded with junk on both
//! sides, so an offset that is dropped or applied to the wrong operand reads junk and disagrees.

use unpopped::ir::{BaseOffset, Expr, OpDef, ScalarExpr, input};
use unpopped::kernel::check_kernel;
use unpopped::kernel_lift::{LiftDecline, lift_elementwise_opdef};
use unpopped::oracle::kernel_eval::{KernelArg, run_kernel};
use unpopped::oracle::{TypedBuffer, evaluate};
use unpopped::plan::try_build_plan;
use unpopped_vocab::{ArchSku, ElementKind, OpCategory, OperandDesc, try_structure_key};

const SHAPE: [i64; 2] = [3, 4];
const JUNK: u8 = 0x5A;

fn width(dtype: ElementKind) -> usize {
    if dtype == ElementKind::F64 { 8 } else { 4 }
}

fn value_bytes(dtype: ElementKind, seed: i64) -> Vec<u8> {
    (0..(SHAPE[0] * SHAPE[1]))
        .flat_map(|i| {
            let v = (i * 3 + seed) % 11 - 5;
            match dtype {
                ElementKind::F32 => (v as f32 * 1.5).to_le_bytes().to_vec(),
                ElementKind::F64 => (v as f64 * 1.5).to_le_bytes().to_vec(),
                _ => (v as i32 * 7).to_le_bytes().to_vec(),
            }
        })
        .collect()
}

/// `values` stored `offset` elements into a junk-padded buffer, bound with that `base_offset`.
fn at_offset(dtype: ElementKind, offset: i64, values: Vec<u8>) -> TypedBuffer {
    let w = width(dtype);
    let pad = vec![JUNK; offset as usize * w];
    let mut bytes = pad;
    bytes.extend(values);
    bytes.extend(vec![JUNK; 5 * w]);
    let mut b = TypedBuffer::new(dtype, SHAPE.to_vec(), vec![SHAPE[1], 1], bytes);
    b.base_offset = offset;
    b
}

struct Offsets {
    inputs: [BaseOffset; 2],
    out: BaseOffset,
}

/// Run `kernel` on `inputs` and an output bound `out_at` elements into junk padding; the whole
/// output storage comes back.
fn run_on(
    dtype: ElementKind,
    kernel: &unpopped::kernel::KernelFn,
    inputs: &[TypedBuffer; 2],
    out_at: i64,
) -> Vec<u8> {
    let len = (SHAPE[0] * SHAPE[1]) as usize * width(dtype);
    let out_bytes = vec![JUNK; (out_at as usize) * width(dtype) + len + 5 * width(dtype)];
    let mut out = TypedBuffer::new(dtype, SHAPE.to_vec(), vec![SHAPE[1], 1], out_bytes);
    out.base_offset = out_at;
    let mut args: Vec<KernelArg> = inputs.iter().cloned().map(KernelArg::Buffer).collect();
    args.push(KernelArg::Buffer(out));
    args.extend(SHAPE.iter().map(|&e| KernelArg::Int(i128::from(e))));
    run_kernel(kernel, &mut args).expect("the kernel runs");
    let KernelArg::Buffer(got) = &args[2] else {
        unreachable!()
    };
    got.bytes.clone()
}

/// Run `Input(0) * Input(1) + Input(0)` with the given runtime offsets and compare the output cells.
fn agree(dtype: ElementKind, off: &Offsets, in_at: [i64; 2], out_at: i64) {
    let body = ScalarExpr::Add(
        Box::new(ScalarExpr::Mul(Box::new(input(0).0), Box::new(input(1).0))),
        Box::new(input(0).0),
    );
    let op = OpDef::elementwise("ew", 2, &[dtype], Expr(body))
        .with_base_offsets(off.inputs.to_vec(), off.out);
    let inputs = [
        at_offset(dtype, in_at[0], value_bytes(dtype, 1)),
        at_offset(dtype, in_at[1], value_bytes(dtype, 4)),
    ];
    let d = OperandDesc::new(2, &SHAPE, &[SHAPE[1], 1], dtype, 256);
    let ops = [d, d, d];
    let key = try_structure_key(OpCategory::BinaryElementwise, &ops, ArchSku::Sm89).expect("a key");
    let plan = try_build_plan(&op, &key).expect("the offset op plans");
    let len = (SHAPE[0] * SHAPE[1]) as usize * width(dtype);
    let want = evaluate(&plan, &ops, &inputs, &[]).remove(0).bytes[..len].to_vec();

    let kernel = lift_elementwise_opdef(&op, dtype, 2).expect("the offset op lifts");
    assert_eq!(
        check_kernel(&kernel),
        Ok(()),
        "a lifted kernel is well formed"
    );
    let got = run_on(dtype, &kernel, &inputs, out_at);
    let start = out_at as usize * width(dtype);
    assert_eq!(
        &got[start..start + len],
        &want[..],
        "{dtype:?} in_at={in_at:?} out_at={out_at}"
    );
    // Control: with an input's offset dropped the same kernel reads junk, so the agreement above
    // does depend on the offset being honored.
    for k in (0..2).filter(|&k| in_at[k] != 0) {
        let mut dropped = inputs.clone();
        dropped[k].base_offset = 0;
        let g = run_on(dtype, &kernel, &dropped, out_at);
        assert_ne!(
            &g[start..start + len],
            &want[..],
            "input {k}'s offset matters"
        );
    }
    // The padding on both sides of the output cells is untouched.
    assert!(got[..start].iter().all(|&b| b == JUNK), "front pad");
    assert!(got[start + len..].iter().all(|&b| b == JUNK), "back pad");
}

use BaseOffset::{Runtime, Zero};

#[test]
fn offsets_on_every_operand_agree() {
    for dtype in [ElementKind::F32, ElementKind::F64, ElementKind::I32] {
        let all = Offsets {
            inputs: [Runtime, Runtime],
            out: Runtime,
        };
        agree(dtype, &all, [3, 5], 2);
    }
}

#[test]
fn an_offset_on_one_operand_only_agrees() {
    let dtype = ElementKind::F32;
    let first = Offsets {
        inputs: [Runtime, Zero],
        out: Zero,
    };
    agree(dtype, &first, [4, 0], 0);
    let second = Offsets {
        inputs: [Zero, Runtime],
        out: Zero,
    };
    agree(dtype, &second, [0, 7], 0);
    let out_only = Offsets {
        inputs: [Zero, Zero],
        out: Runtime,
    };
    agree(dtype, &out_only, [0, 0], 6);
}

#[test]
fn forms_the_lift_does_not_cover_decline_with_a_reason() {
    // Control: an offset-free and an offset Elementwise op both lift.
    let plain = OpDef::elementwise("p", 1, &[ElementKind::F32], Expr(input(0).0));
    assert!(lift_elementwise_opdef(&plain, ElementKind::F32, 2).is_ok());
    let off = plain.clone().with_base_offsets(vec![Runtime], Zero);
    assert!(lift_elementwise_opdef(&off, ElementKind::F32, 2).is_ok());
    // The plan admits an offset on Elementwise only, so any other access declines it.
    let scan = OpDef::scan_simple(
        "s",
        &[ElementKind::F32],
        unpopped::ir::ReduceOp::Sum,
        1,
        false,
        false,
    );
    let mut scan = scan;
    scan.base_offsets = vec![Runtime];
    assert_eq!(
        unpopped::kernel_lift::lift_scan_opdef(&scan, ElementKind::F32, 2),
        Err(LiftDecline::UnsupportedForm(
            "a runtime base offset outside Elementwise"
        ))
    );
    // A base_offsets that does not match the input count is malformed.
    let mut bad = plain;
    bad.base_offsets = vec![Runtime, Runtime];
    assert_eq!(
        lift_elementwise_opdef(&bad, ElementKind::F32, 2),
        Err(LiftDecline::UnsupportedForm("an ill-formed base_offsets"))
    );
}
