// SPDX-License-Identifier: MIT OR Apache-2.0
//! **The layers agree on gathers** (`docs/ir-kernel-layer.md` section 7): an `Access::Elementwise`
//! op with one `ReadIndex::Indexed` input, lifted by `lift_gather_opdef` and run by the kernel
//! interpreter, produces byte for byte what `oracle::evaluate` produces.
//!
//! Both a full-shape index (torch `gather`) and a 1-D index broadcast over the other axis (an
//! embedding's ids), on axis 0 and axis 1, for every out-of-range policy and every index dtype. The
//! index corpus holds negatives, the last valid row, one past it, and far-out values (including the
//! index type's extremes). The oracle zeroes its output, so the agreement runs on a zeroed kernel
//! output; the policies differ only on a pre-filled one, which `policies_differ_on_a_prefilled_output`
//! checks against the policy's own definition.

use unpopped::ir::{Expr, OobPolicy, OpDef, ReadIndex, ScalarExpr, input};
use unpopped::kernel::check_kernel;
use unpopped::kernel_lift::{LiftDecline, lift_gather_opdef};
use unpopped::oracle::kernel_eval::{KernelArg, run_kernel};
use unpopped::oracle::{TypedBuffer, evaluate};
use unpopped::plan::try_build_plan;
use unpopped_vocab::{ArchSku, ElementKind, OpCategory, OperandDesc, try_structure_key};

const POLICIES: [OobPolicy; 3] = [OobPolicy::Skip, OobPolicy::Clamp, OobPolicy::ZeroFill];
const INDEX_TYPES: [ElementKind; 3] = [ElementKind::I32, ElementKind::I64, ElementKind::U32];

fn width(dtype: ElementKind) -> usize {
    match dtype {
        ElementKind::F64 | ElementKind::I64 => 8,
        _ => 4,
    }
}

/// One gather case: the data is `[d0, d1]`, the output `[o0, o1]`; the index is full-shape
/// (`[o0, o1]`) or, when `ids_1d`, runs along `axis` only (stride 0 on the other axis).
#[derive(Clone, Copy)]
struct Case {
    axis: u8,
    data: [i64; 2],
    out: [i64; 2],
    ids_1d: bool,
}

const FULL_AXIS0: Case = Case {
    axis: 0,
    data: [4, 5],
    out: [6, 5],
    ids_1d: false,
};
const FULL_AXIS1: Case = Case {
    axis: 1,
    data: [3, 6],
    out: [3, 4],
    ids_1d: false,
};
const IDS_AXIS0: Case = Case {
    axis: 0,
    data: [5, 3],
    out: [7, 3],
    ids_1d: true,
};

fn float_bytes(dtype: ElementKind, n: usize) -> Vec<u8> {
    (0..n)
        .flat_map(|i| {
            let v = i as f64 * 1.5 - 7.0;
            match dtype {
                ElementKind::F32 => (v as f32).to_le_bytes().to_vec(),
                _ => v.to_le_bytes().to_vec(),
            }
        })
        .collect()
}

/// The index corpus: in range, the last valid index, one past, negatives, and far values.
fn indices(index_dtype: ElementKind, extent: i64) -> Vec<i128> {
    let (lo, hi): (i128, i128) = match index_dtype {
        ElementKind::I32 => (i128::from(i32::MIN), i128::from(i32::MAX)),
        ElementKind::I64 => (i128::from(i64::MIN), i128::from(i64::MAX)),
        _ => (0, i128::from(u32::MAX)),
    };
    let e = i128::from(extent);
    let mut v = vec![0, e - 1, 1, e, e + 7, 2, hi, e - 2];
    if lo < 0 {
        v.extend([-1, lo, -3]);
    }
    v
}

fn index_buffer(case: Case, index_dtype: ElementKind) -> TypedBuffer {
    let extent = case.data[usize::from(case.axis)];
    let pool = indices(index_dtype, extent);
    let w = width(index_dtype);
    let n_ids = if case.ids_1d {
        case.out[usize::from(case.axis)]
    } else {
        case.out[0] * case.out[1]
    } as usize;
    let bytes: Vec<u8> = (0..n_ids)
        .flat_map(|i| pool[(i * 5 + i / 3) % pool.len()].to_le_bytes()[..w].to_vec())
        .collect();
    let strides = if case.ids_1d {
        if case.axis == 0 {
            vec![1, 0]
        } else {
            vec![0, 1]
        }
    } else {
        vec![case.out[1], 1]
    };
    TypedBuffer::new(index_dtype, case.out.to_vec(), strides, bytes)
}

fn desc(shape: [i64; 2], strides: &[i64], dtype: ElementKind) -> OperandDesc {
    OperandDesc::new(2, &shape, strides, dtype, 256)
}

/// The op: `Input(0)` gathered along `axis` by `Input(1)`; with `bias`, `Input(0) + Input(2)`.
fn make_op(
    case: Case,
    oob: OobPolicy,
    dtype: ElementKind,
    index_dtype: ElementKind,
    bias: bool,
) -> OpDef {
    if !bias {
        return OpDef::gather("g", &[dtype], case.axis, oob, index_dtype);
    }
    let body = ScalarExpr::Add(Box::new(input(0).0), Box::new(input(2).0));
    OpDef::elementwise("gb", 3, &[dtype], Expr(body)).with_indexed(vec![
        ReadIndex::Indexed {
            index_operand: 1,
            axis: case.axis,
            oob,
            index_dtype,
        },
        ReadIndex::Direct,
        ReadIndex::Direct,
    ])
}

struct Run {
    want: Vec<u8>,
    got: Vec<u8>,
    idx: TypedBuffer,
}

/// The op, its operand descriptions and the buffers it reads, built once for both layers.
struct Setup {
    op: OpDef,
    ops: Vec<OperandDesc>,
    inputs: Vec<TypedBuffer>,
    idx: TypedBuffer,
}

fn setup(
    case: Case,
    oob: OobPolicy,
    dtype: ElementKind,
    index_dtype: ElementKind,
    bias: bool,
) -> Setup {
    let n_data = (case.data[0] * case.data[1]) as usize;
    let n_out = (case.out[0] * case.out[1]) as usize;
    let out_strides = [case.out[1], 1];
    let data = TypedBuffer::new(
        dtype,
        case.data.to_vec(),
        vec![case.data[1], 1],
        float_bytes(dtype, n_data),
    );
    let idx = index_buffer(case, index_dtype);
    let mut ops = vec![
        desc(case.data, &[case.data[1], 1], dtype),
        desc(case.out, &idx.strides, index_dtype),
    ];
    let mut inputs = vec![data, idx.clone()];
    if bias {
        ops.push(desc(case.out, &out_strides, dtype));
        inputs.push(TypedBuffer::new(
            dtype,
            case.out.to_vec(),
            out_strides.to_vec(),
            float_bytes(dtype, n_out + 3)[width(dtype) * 3..].to_vec(),
        ));
    }
    ops.push(desc(case.out, &out_strides, dtype));
    Setup {
        op: make_op(case, oob, dtype, index_dtype, bias),
        ops,
        inputs,
        idx,
    }
}

/// What the oracle writes for `s`.
fn oracle_out(s: &Setup, len: usize) -> Vec<u8> {
    let key =
        try_structure_key(OpCategory::BinaryElementwise, &s.ops, ArchSku::Sm89).expect("a key");
    let plan = try_build_plan(&s.op, &key).expect("the gather plans");
    evaluate(&plan, &s.ops, &s.inputs, &[]).remove(0).bytes[..len].to_vec()
}

/// What the lifted kernel writes for `s`, its output starting as `fill` bytes.
fn kernel_out(s: &Setup, case: Case, dtype: ElementKind, len: usize, fill: u8) -> Vec<u8> {
    let kernel = lift_gather_opdef(&s.op, dtype, 2).expect("the gather lifts");
    assert_eq!(
        check_kernel(&kernel),
        Ok(()),
        "a lifted kernel is well formed"
    );
    let mut args: Vec<KernelArg> = s.inputs.iter().cloned().map(KernelArg::Buffer).collect();
    args.push(KernelArg::Buffer(TypedBuffer::new(
        dtype,
        case.out.to_vec(),
        vec![case.out[1], 1],
        vec![fill; len],
    )));
    args.extend(case.out.iter().map(|&e| KernelArg::Int(i128::from(e))));
    args.push(KernelArg::Int(i128::from(
        case.data[usize::from(case.axis)],
    )));
    run_kernel(&kernel, &mut args).expect("the kernel runs");
    let KernelArg::Buffer(out) = &args[args.len() - 4] else {
        unreachable!()
    };
    out.bytes[..len].to_vec()
}

fn run(
    case: Case,
    oob: OobPolicy,
    dtype: ElementKind,
    index_dtype: ElementKind,
    bias: bool,
    fill: u8,
) -> Run {
    let s = setup(case, oob, dtype, index_dtype, bias);
    let len = (case.out[0] * case.out[1]) as usize * width(dtype);
    Run {
        want: oracle_out(&s, len),
        got: kernel_out(&s, case, dtype, len, fill),
        idx: s.idx,
    }
}

#[test]
fn gathers_agree_for_every_policy_index_type_and_shape() {
    for dtype in [ElementKind::F32, ElementKind::F64] {
        for index_dtype in INDEX_TYPES {
            for oob in POLICIES {
                for case in [FULL_AXIS0, FULL_AXIS1, IDS_AXIS0] {
                    let r = run(case, oob, dtype, index_dtype, false, 0);
                    assert_eq!(
                        r.got, r.want,
                        "{dtype:?} idx={index_dtype:?} {oob:?} axis={} ids_1d={}",
                        case.axis, case.ids_1d
                    );
                }
            }
        }
    }
}

#[test]
fn a_gather_body_over_a_second_input_agrees() {
    for oob in POLICIES {
        for case in [FULL_AXIS0, IDS_AXIS0] {
            let r = run(case, oob, ElementKind::F32, ElementKind::I32, true, 0);
            assert_eq!(r.got, r.want, "{oob:?} axis={}", case.axis);
        }
    }
}

fn index_at(idx: &TypedBuffer, case: Case, row: i64, col: i64) -> i128 {
    let w = width(idx.dtype);
    let off = (row * idx.strides[0] + col * idx.strides[1]) as usize * w;
    let mut x = [0u8; 16];
    x[..w].copy_from_slice(&idx.bytes[off..off + w]);
    if idx.dtype != ElementKind::U32 && x[w - 1] & 0x80 != 0 {
        x[w..].fill(0xff);
    }
    let _ = case;
    i128::from_le_bytes(x)
}

#[test]
fn policies_differ_on_a_prefilled_output() {
    // The policies' own definitions, checked on a PRE-FILLED output (the oracle zeroes its output, so
    // Skip and ZeroFill agree with it there): Skip leaves the cell, ZeroFill writes 0, Clamp writes
    // the clamped read.
    let case = FULL_AXIS0;
    let extent = i128::from(case.data[0]);
    for oob in POLICIES {
        let r = run(case, oob, ElementKind::F32, ElementKind::I32, false, 0xAB);
        let clamped = run(
            case,
            OobPolicy::Clamp,
            ElementKind::F32,
            ElementKind::I32,
            false,
            0,
        );
        let mut saw_oob = false;
        for row in 0..case.out[0] {
            for col in 0..case.out[1] {
                let cell = (row * case.out[1] + col) as usize * 4;
                let got = &r.got[cell..cell + 4];
                let i = index_at(&r.idx, case, row, col);
                if (0..extent).contains(&i) {
                    assert_eq!(
                        got,
                        &clamped.got[cell..cell + 4],
                        "{oob:?} in range ({row},{col})"
                    );
                    continue;
                }
                saw_oob = true;
                match oob {
                    OobPolicy::Skip => assert_eq!(got, [0xAB; 4], "Skip leaves ({row},{col})"),
                    OobPolicy::ZeroFill => assert_eq!(got, [0; 4], "ZeroFill zeroes ({row},{col})"),
                    OobPolicy::Clamp => {
                        assert_eq!(
                            got,
                            &clamped.got[cell..cell + 4],
                            "Clamp stores ({row},{col})"
                        );
                    }
                }
            }
        }
        assert!(saw_oob, "the corpus has out-of-range indices");
    }
}

#[test]
fn forms_the_lift_does_not_cover_decline_with_a_reason() {
    let f32k = [ElementKind::F32];
    // Control: a gather lifts.
    let g = OpDef::gather("g", &f32k, 0, OobPolicy::Skip, ElementKind::I32);
    assert!(lift_gather_opdef(&g, ElementKind::F32, 2).is_ok());
    // No indexed input.
    let plain = OpDef::elementwise("e", 1, &f32k, Expr(input(0).0));
    assert_eq!(
        lift_gather_opdef(&plain, ElementKind::F32, 2),
        Err(LiftDecline::NotGather)
    );
    // The plain elementwise lift still declines a gather.
    assert!(matches!(
        unpopped::kernel_lift::lift_elementwise_opdef(&g, ElementKind::F32, 2),
        Err(LiftDecline::UnsupportedForm(_))
    ));
    // A float index is not a gather index.
    let bad = OpDef::gather("b", &f32k, 0, OobPolicy::Skip, ElementKind::F32);
    assert_eq!(
        lift_gather_opdef(&bad, ElementKind::F32, 2),
        Err(LiftDecline::UnsupportedForm("a non-integer gather index"))
    );
}
