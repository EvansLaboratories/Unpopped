// SPDX-License-Identifier: MIT OR Apache-2.0
//! **Runtime base offsets on a gather and on a scatter** (`docs/ir-kernel-layer.md` section 7). The
//! offset is the argument buffer's `base_offset`, applied by the interpreter to every `Place` access,
//! so the lifted gather and scatter are the offset-free kernels and the caller binds the offset
//! (`kernel_base_offset_agrees.rs` covers the plain Elementwise lift). This file holds the other two
//! indexed lifts to that: every operand (data or updates, index, output or destination) sits at an
//! offset inside junk padding, and an offset that is dropped reads junk and disagrees.

use unpopped::ir::{BaseOffset, OobPolicy, OpDef};
use unpopped::kernel::KernelFn;
use unpopped::kernel::check_kernel;
use unpopped::kernel_lift::{lift_gather_opdef, lift_scatter_opdef};
use unpopped::oracle::kernel_eval::{KernelArg, run_kernel};
use unpopped::oracle::{TypedBuffer, evaluate};
use unpopped::plan::try_build_plan;
use unpopped_vocab::{ArchSku, ElementKind, OpCategory, OperandDesc, try_structure_key};

const JUNK: u8 = 0x5A;
const F32: ElementKind = ElementKind::F32;
const I32: ElementKind = ElementKind::I32;

/// `bytes` stored `offset` 4-byte elements into junk padding, bound with that `base_offset`.
fn at_offset(dtype: ElementKind, shape: [i64; 2], offset: i64, bytes: Vec<u8>) -> TypedBuffer {
    let mut all = vec![JUNK; offset as usize * 4];
    all.extend(bytes);
    all.extend([JUNK; 20]);
    let mut b = TypedBuffer::new(dtype, shape.to_vec(), vec![shape[1], 1], all);
    b.base_offset = offset;
    b
}

fn floats(n: usize) -> Vec<u8> {
    (0..n)
        .flat_map(|i| ((i as i64 * 5 % 13 - 6) as f32).to_le_bytes())
        .collect()
}

/// Targets with duplicates and values below zero and at and past `extent` (all skipped).
fn targets(n: usize, extent: i32) -> Vec<u8> {
    let pool = [0, extent - 1, 1, extent, -1, 2, extent + 9, 0, 1];
    (0..n)
        .flat_map(|i| pool[(i * 2 + i / 3) % pool.len()].to_le_bytes())
        .collect()
}

struct Bound {
    op: OpDef,
    inputs: [TypedBuffer; 2],
    out_shape: [i64; 2],
    ops: [OperandDesc; 3],
}

/// Offsets for the first input, the index and the output.
const AT: [i64; 3] = [3, 2, 4];

fn offsets(op: OpDef) -> OpDef {
    op.with_base_offsets(
        vec![BaseOffset::Runtime, BaseOffset::Runtime],
        BaseOffset::Runtime,
    )
}

fn desc(shape: [i64; 2], dtype: ElementKind) -> OperandDesc {
    OperandDesc::new(2, &shape, &[shape[1], 1], dtype, 256)
}

/// Gather: data `[4, 5]`, output and index `[6, 5]`, axis 0.
fn gather() -> Bound {
    let (data, out) = ([4, 5], [6, 5]);
    let op = offsets(OpDef::gather("g", &[F32], 0, OobPolicy::Skip, I32));
    Bound {
        op,
        inputs: [
            at_offset(F32, data, AT[0], floats(20)),
            at_offset(I32, out, AT[1], targets(30, 4)),
        ],
        out_shape: out,
        ops: [desc(data, F32), desc(out, I32), desc(out, F32)],
    }
}

/// Scatter-add: updates and index `[6, 4]`, destination `[5, 4]`, axis 0.
fn scatter() -> Bound {
    let (src, dst) = ([6, 4], [5, 4]);
    let op = offsets(OpDef::scatter_add("s", &[F32], 0, I32));
    Bound {
        op,
        inputs: [
            at_offset(F32, src, AT[0], floats(24)),
            at_offset(I32, src, AT[1], targets(24, 5)),
        ],
        out_shape: dst,
        ops: [desc(src, F32), desc(src, I32), desc(dst, F32)],
    }
}

fn oracle(b: &Bound) -> Vec<u8> {
    let key =
        try_structure_key(OpCategory::BinaryElementwise, &b.ops, ArchSku::Sm89).expect("a key");
    let plan = try_build_plan(&b.op, &key).expect("the offset op plans");
    let len = (b.out_shape[0] * b.out_shape[1]) as usize * 4;
    evaluate(&plan, &b.ops, &b.inputs, &[]).remove(0).bytes[..len].to_vec()
}

/// The kernel's output buffer (junk-padded, bound at `AT[2]`) after running on `inputs`.
fn run_on(b: &Bound, kernel: &KernelFn, inputs: &[TypedBuffer; 2], extra: &[i64]) -> Vec<u8> {
    let len = (b.out_shape[0] * b.out_shape[1]) as usize * 4;
    let mut out = at_offset(F32, b.out_shape, AT[2], vec![0; len]);
    out.bytes[..AT[2] as usize * 4].fill(JUNK);
    let mut args: Vec<KernelArg> = inputs.iter().cloned().map(KernelArg::Buffer).collect();
    args.push(KernelArg::Buffer(out));
    args.extend(extra.iter().map(|&e| KernelArg::Int(i128::from(e))));
    run_kernel(kernel, &mut args).expect("the kernel runs");
    let KernelArg::Buffer(got) = &args[2] else {
        unreachable!()
    };
    got.bytes.clone()
}

/// Agree with the oracle on the output cells, leave the padding alone, and (control) disagree once
/// any one operand's offset is dropped.
fn check(b: &Bound, kernel: &KernelFn, extra: &[i64]) {
    assert_eq!(
        check_kernel(kernel),
        Ok(()),
        "a lifted kernel is well formed"
    );
    let want = oracle(b);
    let (start, len) = (AT[2] as usize * 4, want.len());
    let got = run_on(b, kernel, &b.inputs, extra);
    assert_eq!(&got[start..start + len], &want[..]);
    assert!(got[..start].iter().all(|&x| x == JUNK), "front pad");
    assert!(got[start + len..].iter().all(|&x| x == JUNK), "back pad");
    for k in 0..2 {
        let mut dropped = b.inputs.clone();
        dropped[k].base_offset = 0;
        let g = run_on(b, kernel, &dropped, extra);
        assert_ne!(
            &g[start..start + len],
            &want[..],
            "input {k}'s offset matters"
        );
    }
}

#[test]
fn a_gather_with_offsets_on_every_operand_agrees() {
    let b = gather();
    let kernel = lift_gather_opdef(&b.op, F32, 2).expect("the offset gather lifts");
    // Output extents, then the data extent along the gathered axis.
    check(&b, &kernel, &[6, 5, 4]);
}

#[test]
fn a_scatter_with_offsets_on_every_operand_agrees() {
    let b = scatter();
    let kernel = lift_scatter_opdef(&b.op, F32, 2).expect("the offset scatter lifts");
    // Source extents, then the destination extent along the scattered axis.
    check(&b, &kernel, &[6, 4, 5]);
}
