// SPDX-License-Identifier: MIT OR Apache-2.0
//! **The layers agree on scatters** (`docs/ir-kernel-layer.md` section 7): an `Access::Elementwise`
//! op with a `WriteIndex::ScatterIndexed` output, lifted by `lift_scatter_opdef` and run by the
//! kernel interpreter, produces byte for byte what `oracle::evaluate` produces.
//!
//! `Assign`, `AtomicAdd`, `AtomicMax` and `AtomicMin`, a full-shape index (torch `scatter`) and a 1-D
//! index broadcast over the other axis (`index_add`), on axis 0 and axis 1. The index corpus holds
//! duplicate targets, negatives and targets at and past the destination's extent (skipped). The
//! oracle walks the source in row-major order and so does the kernel, so even a duplicate target
//! under `Assign` (last writer wins) agrees. Float adds are on integer-valued data, where the
//! oracle's `f64` sum rounded once and the kernel's `F32` add are the same number.

use unpopped::ir::{OobPolicy, OpDef, WriteCombine, WriteIndex};
use unpopped::kernel::check_kernel;
use unpopped::kernel_lift::{LiftDecline, lift_scatter_opdef};
use unpopped::oracle::kernel_eval::{KernelArg, run_kernel};
use unpopped::oracle::{TypedBuffer, evaluate};
use unpopped::plan::try_build_plan;
use unpopped_vocab::{ArchSku, ElementKind, OpCategory, OperandDesc, try_structure_key};

const INDEX_TYPES: [ElementKind; 3] = [ElementKind::I32, ElementKind::I64, ElementKind::U32];

fn width(dtype: ElementKind) -> usize {
    match dtype {
        ElementKind::F64 | ElementKind::I64 => 8,
        _ => 4,
    }
}

/// The source (iteration) shape `src`, the destination `dst`; the index is full-shape over `src`
/// or, when `ids_1d`, runs along `axis` only.
#[derive(Clone, Copy)]
struct Case {
    axis: u8,
    src: [i64; 2],
    dst: [i64; 2],
    ids_1d: bool,
}

const FULL_AXIS0: Case = Case {
    axis: 0,
    src: [6, 4],
    dst: [5, 4],
    ids_1d: false,
};
const FULL_AXIS1: Case = Case {
    axis: 1,
    src: [3, 6],
    dst: [3, 4],
    ids_1d: false,
};
const INDEX_ADD: Case = Case {
    axis: 0,
    src: [7, 3],
    dst: [4, 3],
    ids_1d: true,
};

/// Data the combine can be checked on: small integers (exact in any float), and for `I32` the type's
/// extremes so an add wraps.
fn data_bytes(dtype: ElementKind, n: usize) -> Vec<u8> {
    (0..n)
        .flat_map(|i| {
            let small = (i as i64 * 5 % 13) - 6;
            match dtype {
                ElementKind::F32 => (small as f32).to_le_bytes().to_vec(),
                ElementKind::F64 => (small as f64).to_le_bytes().to_vec(),
                ElementKind::I64 => (small * 1_000_000_007).to_le_bytes().to_vec(),
                _ => {
                    let v = if i % 4 == 0 {
                        i32::MAX - small.abs() as i32
                    } else {
                        small as i32
                    };
                    v.to_le_bytes().to_vec()
                }
            }
        })
        .collect()
}

fn indices(index_dtype: ElementKind, extent: i64) -> Vec<i128> {
    let e = i128::from(extent);
    let mut v: Vec<i128> = vec![0, e - 1, 1, e, e + 9, 0, 2, e - 1, 1];
    if index_dtype != ElementKind::U32 {
        v.extend([-1, -7]);
    }
    v.push(match index_dtype {
        ElementKind::I64 => i128::from(i64::MAX),
        ElementKind::I32 => i128::from(i32::MAX),
        _ => i128::from(u32::MAX),
    });
    v
}

fn index_buffer(case: Case, index_dtype: ElementKind) -> TypedBuffer {
    let pool = indices(index_dtype, case.dst[usize::from(case.axis)]);
    let w = width(index_dtype);
    let n = if case.ids_1d {
        case.src[usize::from(case.axis)]
    } else {
        case.src[0] * case.src[1]
    } as usize;
    let bytes: Vec<u8> = (0..n)
        .flat_map(|i| pool[(i * 3 + i / 4) % pool.len()].to_le_bytes()[..w].to_vec())
        .collect();
    let strides = match (case.ids_1d, case.axis) {
        (false, _) => vec![case.src[1], 1],
        (true, 0) => vec![1, 0],
        (true, _) => vec![0, 1],
    };
    TypedBuffer::new(index_dtype, case.src.to_vec(), strides, bytes)
}

fn make_op(
    case: Case,
    combine: WriteCombine,
    dtype: ElementKind,
    index_dtype: ElementKind,
) -> OpDef {
    match combine {
        WriteCombine::Assign => OpDef::scatter("s", &[dtype], case.axis, index_dtype),
        WriteCombine::AtomicAdd => OpDef::scatter_add("s", &[dtype], case.axis, index_dtype),
        _ => OpDef::scatter("s", &[dtype], case.axis, index_dtype).with_scatter(
            WriteIndex::ScatterIndexed {
                index_operand: 1,
                axis: case.axis,
                combine,
                oob: OobPolicy::Skip,
                index_dtype,
            },
        ),
    }
}

/// What the oracle writes for `op` over `updates` and `idx`.
fn oracle_out(case: Case, op: &OpDef, dtype: ElementKind, inputs: &[TypedBuffer; 2]) -> Vec<u8> {
    let ops = [
        OperandDesc::new(2, &case.src, &[case.src[1], 1], dtype, 256),
        OperandDesc::new(2, &case.src, &inputs[1].strides, inputs[1].dtype, 256),
        OperandDesc::new(2, &case.dst, &[case.dst[1], 1], dtype, 256),
    ];
    let key = try_structure_key(OpCategory::BinaryElementwise, &ops, ArchSku::Sm89).expect("a key");
    let plan = try_build_plan(op, &key).expect("the scatter plans");
    let len = (case.dst[0] * case.dst[1]) as usize * width(dtype);
    evaluate(&plan, &ops, inputs, &[]).remove(0).bytes[..len].to_vec()
}

/// What the lifted kernel writes, its destination starting as `fill` repeated per element.
fn kernel_out(
    case: Case,
    op: &OpDef,
    dtype: ElementKind,
    inputs: &[TypedBuffer; 2],
    fill: Option<&[u8]>,
) -> Vec<u8> {
    let kernel = lift_scatter_opdef(op, dtype, 2).expect("the scatter lifts");
    assert_eq!(
        check_kernel(&kernel),
        Ok(()),
        "a lifted kernel is well formed"
    );
    let len = (case.dst[0] * case.dst[1]) as usize * width(dtype);
    let init: Vec<u8> = match fill {
        Some(f) => f.iter().copied().cycle().take(len).collect(),
        None => vec![0; len],
    };
    let mut args = vec![
        KernelArg::Buffer(inputs[0].clone()),
        KernelArg::Buffer(inputs[1].clone()),
        KernelArg::Buffer(TypedBuffer::new(
            dtype,
            case.dst.to_vec(),
            vec![case.dst[1], 1],
            init,
        )),
        KernelArg::Int(i128::from(case.src[0])),
        KernelArg::Int(i128::from(case.src[1])),
        KernelArg::Int(i128::from(case.dst[usize::from(case.axis)])),
    ];
    run_kernel(&kernel, &mut args).expect("the kernel runs");
    let KernelArg::Buffer(out) = &args[2] else {
        unreachable!()
    };
    out.bytes[..len].to_vec()
}

/// The oracle's destination and the kernel's, the kernel's destination starting as `fill` repeated
/// per element (the oracle's starts zeroed).
fn run(
    case: Case,
    combine: WriteCombine,
    dtype: ElementKind,
    index_dtype: ElementKind,
    fill: Option<&[u8]>,
) -> (Vec<u8>, Vec<u8>) {
    let n_src = (case.src[0] * case.src[1]) as usize;
    let updates = TypedBuffer::new(
        dtype,
        case.src.to_vec(),
        vec![case.src[1], 1],
        data_bytes(dtype, n_src),
    );
    let inputs = [updates, index_buffer(case, index_dtype)];
    let op = make_op(case, combine, dtype, index_dtype);
    (
        oracle_out(case, &op, dtype, &inputs),
        kernel_out(case, &op, dtype, &inputs, fill),
    )
}

#[test]
fn assign_and_add_agree_for_every_shape_and_index_type() {
    for dtype in [ElementKind::F32, ElementKind::F64, ElementKind::I32] {
        for index_dtype in INDEX_TYPES {
            for combine in [WriteCombine::Assign, WriteCombine::AtomicAdd] {
                for case in [FULL_AXIS0, FULL_AXIS1, INDEX_ADD] {
                    let (want, got) = run(case, combine, dtype, index_dtype, None);
                    assert_eq!(
                        got, want,
                        "{dtype:?} idx={index_dtype:?} {combine:?} axis={} ids_1d={}",
                        case.axis, case.ids_1d
                    );
                }
            }
        }
    }
}

#[test]
fn atomic_max_and_min_agree_on_integers() {
    for dtype in [ElementKind::I32, ElementKind::I64] {
        for combine in [WriteCombine::AtomicMax, WriteCombine::AtomicMin] {
            for case in [FULL_AXIS0, FULL_AXIS1, INDEX_ADD] {
                let (want, got) = run(case, combine, dtype, ElementKind::I32, None);
                assert_eq!(got, want, "{dtype:?} {combine:?} axis={}", case.axis);
            }
        }
    }
}

fn le_i32s(b: &[u8]) -> Vec<i32> {
    b.chunks(4)
        .map(|c| i32::from_le_bytes(c.try_into().unwrap()))
        .collect()
}

#[test]
fn an_add_accumulates_into_the_destination_it_is_given() {
    // Not in the oracle (which starts zeroed): the kernel owns no initial value, so adding into a
    // destination that holds 100 everywhere gives the oracle's sums plus 100, and an out-of-range
    // target leaves its cell at 100.
    let fill = 100i32.to_le_bytes();
    let (want, got) = run(
        FULL_AXIS0,
        WriteCombine::AtomicAdd,
        ElementKind::I32,
        ElementKind::I32,
        Some(&fill),
    );
    let (want, got) = (le_i32s(&want), le_i32s(&got));
    assert!(
        want.iter().any(|&w| w != 0),
        "the corpus scatters something"
    );
    for (i, (g, w)) in got.iter().zip(&want).enumerate() {
        assert_eq!(*g, w.wrapping_add(100), "cell {i}");
    }
}

#[test]
fn the_corpus_exercises_what_the_lift_must_get_right() {
    // Controls, read from the ORACLE: a destination cell hit twice (so Assign's last-writer order and
    // the add are both observable), and a cell never hit (so the skipped targets are observable).
    let (assign, _) = run(
        FULL_AXIS0,
        WriteCombine::Assign,
        ElementKind::I32,
        ElementKind::I32,
        None,
    );
    let (add, _) = run(
        FULL_AXIS0,
        WriteCombine::AtomicAdd,
        ElementKind::I32,
        ElementKind::I32,
        None,
    );
    assert_ne!(
        le_i32s(&assign),
        le_i32s(&add),
        "a duplicate target differs by combine"
    );
    assert!(
        le_i32s(&assign).contains(&0),
        "some destination cell is never written"
    );
}

#[test]
fn forms_the_lift_does_not_cover_decline_with_a_reason() {
    let f32k = [ElementKind::F32];
    // Control: a scatter lifts.
    let s = OpDef::scatter("s", &f32k, 0, ElementKind::I32);
    assert!(lift_scatter_opdef(&s, ElementKind::F32, 2).is_ok());
    let plain = OpDef::elementwise("e", 1, &f32k, unpopped::ir::Expr(unpopped::ir::input(0).0));
    assert_eq!(
        lift_scatter_opdef(&plain, ElementKind::F32, 2),
        Err(LiftDecline::NotScatter)
    );
    // The elementwise lift and the gather lift still decline a scatter.
    assert!(matches!(
        unpopped::kernel_lift::lift_elementwise_opdef(&s, ElementKind::F32, 2),
        Err(LiftDecline::UnsupportedForm(_))
    ));
    // A float atomic max has no meaning.
    let fmax =
        OpDef::scatter("m", &f32k, 0, ElementKind::I32).with_scatter(WriteIndex::ScatterIndexed {
            index_operand: 1,
            axis: 0,
            combine: WriteCombine::AtomicMax,
            oob: OobPolicy::Skip,
            index_dtype: ElementKind::I32,
        });
    assert_eq!(
        lift_scatter_opdef(&fmax, ElementKind::F32, 2),
        Err(LiftDecline::UnsupportedForm("a float atomic max or min"))
    );
    // bincount, the scatter with a per-output (count) dtype, lifts: `kernel_bincount_agrees.rs`.
    let bc = OpDef::bincount("b", ElementKind::I32);
    assert!(lift_scatter_opdef(&bc, ElementKind::I32, 1).is_ok());
}
