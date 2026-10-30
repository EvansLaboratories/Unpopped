// SPDX-License-Identifier: MIT OR Apache-2.0
//! **The emitted C agrees with the kernel interpreter on integer gathers, scatters and
//! `bincount`** (`docs/e1-emit-from-the-kernel-layer.md`, stage E1b-int): an `I32`/`I64` compute
//! dtype is lowered by `CpuC::lower_kernel`, compiled, run, and its output bytes compared with
//! `run_kernel`.
//!
//! The data holds the type's extremes (so an add wraps), the destination starts **non-zero** (so an
//! untouched cell is visible), and the index corpus holds negatives, the last valid row, one past
//! it and far-out values. The serial C is a correctness reference; nothing here times anything.

use unpopped::backend::{Backend, LowerError};
use unpopped::ir::{OobPolicy, OpDef, WriteCombine, WriteIndex};
use unpopped::kernel::{KernelFn, check_kernel};
use unpopped::kernel_lift::{lift_gather_opdef, lift_scatter_opdef};
use unpopped::oracle::TypedBuffer;
use unpopped::oracle::kernel_eval::KernelArg;
use unpopped_cpu_c::CpuC;
use unpopped_vocab::ElementKind;

#[path = "support/cc.rs"]
mod cc;
#[path = "support/kernel_run.rs"]
mod kernel_run;
use kernel_run::{agree, run_c_mutated, width};

const I32: ElementKind = ElementKind::I32;
const I64: ElementKind = ElementKind::I64;

/// Iteration space `iter` (the source of a scatter, the output of a gather), table `table`.
#[derive(Clone, Copy)]
struct Case {
    axis: u8,
    table: [i64; 2],
    iter: [i64; 2],
    ids_1d: bool,
}

const FULL0: Case = Case {
    axis: 0,
    table: [4, 5],
    iter: [6, 5],
    ids_1d: false,
};
const FULL1: Case = Case {
    axis: 1,
    table: [3, 6],
    iter: [3, 4],
    ids_1d: false,
};
const IDS0: Case = Case {
    axis: 0,
    table: [5, 3],
    iter: [7, 3],
    ids_1d: true,
};

fn le(dtype: ElementKind, v: i64) -> Vec<u8> {
    v.to_le_bytes()[..width(dtype)].to_vec()
}

fn dense(dtype: ElementKind, shape: [i64; 2], f: impl Fn(usize) -> i64) -> TypedBuffer {
    let bytes = (0..(shape[0] * shape[1]) as usize)
        .flat_map(|i| le(dtype, f(i)))
        .collect();
    TypedBuffer::new(dtype, shape.to_vec(), vec![shape[1], 1], bytes)
}

/// Small values, with the type's maximum (less a little) every fourth element so an add wraps.
fn data(dtype: ElementKind, shape: [i64; 2]) -> TypedBuffer {
    let max = if dtype == I64 {
        i64::MAX
    } else {
        i64::from(i32::MAX)
    };
    dense(dtype, shape, |i| {
        let small = i as i64 * 5 % 13 - 6;
        if i % 4 == 0 { max - small.abs() } else { small }
    })
}

fn sevens(dtype: ElementKind, shape: [i64; 2]) -> TypedBuffer {
    dense(dtype, shape, |_| 7)
}

fn index_buffer(case: Case, idx: ElementKind) -> TypedBuffer {
    let e = case.table[usize::from(case.axis)];
    let top = if idx == I64 {
        i64::MAX
    } else {
        i64::from(i32::MAX)
    };
    let pool = [0, e - 1, 1, e, e + 7, 2, e - 2, -1, -3, top];
    let n = if case.ids_1d {
        case.iter[usize::from(case.axis)]
    } else {
        case.iter[0] * case.iter[1]
    } as usize;
    let bytes = (0..n)
        .flat_map(|i| le(idx, pool[(i * 5 + i / 3) % pool.len()]))
        .collect();
    let st = match (case.ids_1d, case.axis) {
        (false, _) => vec![case.iter[1], 1],
        (true, 0) => vec![1, 0],
        (true, _) => vec![0, 1],
    };
    TypedBuffer::new(idx, case.iter.to_vec(), st, bytes)
}

fn ints(vs: &[i64]) -> impl Iterator<Item = KernelArg> + '_ {
    vs.iter().map(|&v| KernelArg::Int(i128::from(v)))
}

fn gather(
    case: Case,
    oob: OobPolicy,
    dtype: ElementKind,
    idx: ElementKind,
) -> (KernelFn, Vec<KernelArg>) {
    let op = OpDef::gather("g", &[dtype], case.axis, oob, idx);
    let k = lift_gather_opdef(&op, dtype, 2).expect("the integer gather lifts");
    assert_eq!(check_kernel(&k), Ok(()));
    let mut a = vec![
        KernelArg::Buffer(data(dtype, case.table)),
        KernelArg::Buffer(index_buffer(case, idx)),
        KernelArg::Buffer(sevens(dtype, case.iter)),
    ];
    a.extend(ints(&case.iter));
    a.extend(ints(&[case.table[usize::from(case.axis)]]));
    (k, a)
}

fn scatter_op(case: Case, combine: WriteCombine, dtype: ElementKind, idx: ElementKind) -> OpDef {
    match combine {
        WriteCombine::Assign => OpDef::scatter("s", &[dtype], case.axis, idx),
        WriteCombine::AtomicAdd => OpDef::scatter_add("s", &[dtype], case.axis, idx),
        _ => {
            OpDef::scatter("s", &[dtype], case.axis, idx).with_scatter(WriteIndex::ScatterIndexed {
                index_operand: 1,
                axis: case.axis,
                combine,
                oob: OobPolicy::Skip,
                index_dtype: idx,
            })
        }
    }
}

fn scatter(
    case: Case,
    combine: WriteCombine,
    dtype: ElementKind,
    idx: ElementKind,
) -> (KernelFn, Vec<KernelArg>) {
    let op = scatter_op(case, combine, dtype, idx);
    let k = lift_scatter_opdef(&op, dtype, 2).expect("the integer scatter lifts");
    assert_eq!(check_kernel(&k), Ok(()));
    let mut a = vec![
        KernelArg::Buffer(data(dtype, case.iter)),
        KernelArg::Buffer(index_buffer(case, idx)),
        KernelArg::Buffer(sevens(dtype, case.table)),
    ];
    a.extend(ints(&case.iter));
    a.extend(ints(&[case.table[usize::from(case.axis)]]));
    (k, a)
}

#[test]
fn integer_gathers_agree_for_every_policy() {
    for (n, oob) in [OobPolicy::Skip, OobPolicy::Clamp, OobPolicy::ZeroFill]
        .into_iter()
        .enumerate()
    {
        let (k, a) = gather(FULL0, oob, I32, I32);
        agree(&format!("igather_{n}"), &k, a);
    }
    let (k, a) = gather(IDS0, OobPolicy::ZeroFill, I64, I64);
    agree("igather_i64", &k, a);
    let (k, a) = gather(FULL1, OobPolicy::Clamp, I32, I64);
    agree("igather_axis1", &k, a);
}

#[test]
fn integer_scatters_agree_for_every_combine() {
    let combines = [
        WriteCombine::Assign,
        WriteCombine::AtomicAdd,
        WriteCombine::AtomicMax,
        WriteCombine::AtomicMin,
    ];
    for (m, combine) in combines.into_iter().enumerate() {
        let (k, a) = scatter(FULL0, combine, I32, I32);
        agree(&format!("iscatter_i32_{m}"), &k, a);
        let (k, a) = scatter(FULL0, combine, I64, I32);
        agree(&format!("iscatter_i64_{m}"), &k, a);
    }
    for (n, case) in [FULL1, IDS0].into_iter().enumerate() {
        let (k, a) = scatter(case, WriteCombine::AtomicAdd, I32, I32);
        agree(&format!("iscatter_case_{n}"), &k, a);
    }
}

fn bincount(idx: ElementKind) -> (KernelFn, Vec<KernelArg>) {
    const BINS: i64 = 6;
    let vals: [i64; 14] = [0, 2, 2, 0, 5, 1, 0, -1, 6, 9, 2, 1, -7, 0];
    let bytes = vals.iter().flat_map(|&v| le(idx, v)).collect();
    let x = TypedBuffer::new(idx, vec![vals.len() as i64], vec![1], bytes);
    // The counts start non-zero, so a bin that was not left alone shows.
    let counts = TypedBuffer::new(
        I32,
        vec![BINS],
        vec![1],
        (0..BINS).flat_map(|i| le(I32, 10 + i)).collect(),
    );
    let op = OpDef::bincount("bc", idx);
    let k = lift_scatter_opdef(&op, idx, 1).expect("the bincount lifts");
    assert_eq!(check_kernel(&k), Ok(()));
    let a = vec![
        KernelArg::Buffer(x),
        KernelArg::Buffer(counts),
        KernelArg::Int(vals.len() as i128),
        KernelArg::Int(i128::from(BINS)),
    ];
    (k, a)
}

#[test]
fn bincount_agrees_at_both_index_widths() {
    for (n, idx) in [I32, I64].into_iter().enumerate() {
        let (k, a) = bincount(idx);
        agree(&format!("bincount_{n}"), &k, a);
    }
}

#[test]
fn unsupported_integer_widths_are_still_a_typed_decline() {
    // Control: the I32 scatter lowers. A 16-bit compute dtype has no wrapping form here and is
    // refused with the typed decline, not spelled by default and not a panic.
    let (k, _) = scatter(FULL0, WriteCombine::AtomicAdd, I32, I32);
    assert!(CpuC.lower_kernel(&k).is_ok());
    let i16k = ElementKind::I16;
    let op = OpDef::scatter_add("s", &[i16k], 0, I32);
    let ik = lift_scatter_opdef(&op, i16k, 2).expect("the 16-bit scatter lifts");
    assert!(matches!(
        CpuC.lower_kernel(&ik),
        Err(LowerError::UnsupportedKernel { .. })
    ));
}

const NEG: &str = "(l5) < (((int)(0LL)))";
const PAST: &str = "(l5) >= (l6)";

/// What the C leaves in each writable buffer once `from` is replaced by `to` in its source (the
/// anchor asserted present first); with an empty `from`, the unmutated C.
fn outs(tag: &str, k: &KernelFn, a: &[KernelArg], from: &str, to: &str) -> Vec<Vec<u8>> {
    run_c_mutated(tag, k, a, |s| {
        if from.is_empty() {
            return s;
        }
        assert!(s.contains(from), "{tag}: anchor `{from}` in the C:\n{s}");
        s.replacen(from, to, 1)
    })
    .expect("a host C compiler")
    .outs
}

#[test]
fn the_corpus_catches_a_wrong_combine_or_wrap() {
    // Seeded mutants of the emitted C; each must change what the C leaves behind (the unmutated C
    // agrees with the interpreter above), or the corpus could not tell a wrong combine from a
    // right one.
    let (k, a) = scatter(FULL0, WriteCombine::AtomicAdd, I32, I32);
    let src = CpuC.lower_kernel(&k).unwrap().source;
    let base = outs("is_add_base", &k, &a, "", "");
    // The add goes through the unsigned type, and the narrowed mutant loses the high half of the
    // wrapped cells.
    assert!(
        src.contains("(unsigned int)"),
        "an I32 add is wrapping:\n{src}"
    );
    assert_ne!(
        outs(
            "is_add_short",
            &k,
            &a,
            "(unsigned int)(",
            "(unsigned short)("
        ),
        base,
        "a 16-bit add went unnoticed"
    );
    for (tag, k, a, from, to) in [
        (
            "is_add_neg",
            k.clone(),
            a.clone(),
            NEG,
            "(l5) <= (((int)(0LL)))",
        ),
        ("is_add_past", k, a, PAST, "(l5) > (l6)"),
    ] {
        let base = outs(&format!("{tag}_base"), &k, &a, "", "");
        assert_ne!(outs(tag, &k, &a, from, to), base, "{tag}: went unnoticed");
    }
    let (k, a) = scatter(FULL0, WriteCombine::AtomicMax, I32, I32);
    let base = outs("is_max_base", &k, &a, "", "");
    let flipped = outs("is_max_flip", &k, &a, "((l7) > (l4) ?", "((l7) < (l4) ?");
    assert_ne!(flipped, base, "a Max that kept the smaller went unnoticed");
    let (k, a) = bincount(I32);
    let base = outs("is_bc_base", &k, &a, "", "");
    let twice = outs(
        "is_bc_two",
        &k,
        &a,
        "(unsigned int)(((int)(1LL)))",
        "(unsigned int)(((int)(2LL)))",
    );
    assert_ne!(twice, base, "a count that added 2 went unnoticed");
}
