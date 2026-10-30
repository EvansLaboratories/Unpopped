// SPDX-License-Identifier: MIT OR Apache-2.0
//! **The emitted C agrees with the kernel interpreter on gathers and scatters**
//! (`docs/e1-emit-from-the-kernel-layer.md`, stage E1b): lifted indexed-read and indexed-write
//! kernels, with `I32`, `I64` and `U32` index buffers, every out-of-range policy for gathers and
//! `Assign` / `AtomicAdd` for scatters, are lowered by `CpuC::lower_kernel`, compiled, run, and
//! compared with `run_kernel` (bit-exact except NaN sign/payload).
//!
//! The output starts **non-zero** (the oracle's zeroed output cannot tell `Skip` from `ZeroFill`),
//! and the index corpus holds negatives, the last valid row, one past it, far-out values and the
//! index type's extremes. Float kernels only: an integer compute dtype is not emitted yet
//! (`UnsupportedKernel`), which `integer_compute_is_a_typed_decline` pins. The serial C is a
//! correctness reference; nothing is timed.

use unpopped::backend::{Backend, LowerError};
use unpopped::ir::{OobPolicy, OpDef};
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
use kernel_run::{F32, F64, agree, run_c_mutated, width};

const I32: ElementKind = ElementKind::I32;
const I64: ElementKind = ElementKind::I64;
const U32: ElementKind = ElementKind::U32;

/// `data`/`out` shapes (gather) or `src`/`dst` shapes (scatter); the index is full-shape over the
/// iteration space or, when `ids_1d`, runs along `axis` only (stride 0 on the other axis).
#[derive(Clone, Copy)]
struct Case {
    axis: u8,
    /// Gather: the data. Scatter: the destination.
    table: [i64; 2],
    /// Gather: the output. Scatter: the source.
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

fn strides(shape: &[i64]) -> Vec<i64> {
    (0..shape.len())
        .map(|d| shape[d + 1..].iter().product())
        .collect()
}

fn floats(dtype: ElementKind, shape: [i64; 2], f: impl Fn(usize) -> f64) -> TypedBuffer {
    let n = (shape[0] * shape[1]) as usize;
    let bytes = (0..n)
        .flat_map(|i| match dtype {
            F32 => (f(i) as f32).to_le_bytes().to_vec(),
            _ => f(i).to_le_bytes().to_vec(),
        })
        .collect();
    TypedBuffer::new(dtype, shape.to_vec(), strides(&shape), bytes)
}

/// Data a combine can be checked on: small integers, exact in any float.
fn data(dtype: ElementKind, shape: [i64; 2]) -> TypedBuffer {
    floats(dtype, shape, |i| (i as i64 * 5 % 13 - 6) as f64 * 1.5)
}

/// A destination that is not zero: `7.0` everywhere, so an untouched cell is visible.
fn sevens(dtype: ElementKind, shape: [i64; 2]) -> TypedBuffer {
    floats(dtype, shape, |_| 7.0)
}

fn index_pool(index_dtype: ElementKind, extent: i64) -> Vec<i128> {
    let e = i128::from(extent);
    let mut v = vec![0, e - 1, 1, e, e + 7, 2, e - 2];
    if index_dtype != U32 {
        v.extend([-1, -3]);
    }
    v.push(match index_dtype {
        I64 => i128::from(i64::MAX),
        I32 => i128::from(i32::MAX),
        _ => i128::from(u32::MAX),
    });
    if index_dtype == I64 {
        v.push(i128::from(i64::MIN));
    }
    v
}

fn index_buffer(case: Case, index_dtype: ElementKind) -> TypedBuffer {
    let pool = index_pool(index_dtype, case.table[usize::from(case.axis)]);
    let w = width(index_dtype);
    let n = if case.ids_1d {
        case.iter[usize::from(case.axis)]
    } else {
        case.iter[0] * case.iter[1]
    } as usize;
    let bytes = (0..n)
        .flat_map(|i| pool[(i * 5 + i / 3) % pool.len()].to_le_bytes()[..w].to_vec())
        .collect();
    let st = match (case.ids_1d, case.axis) {
        (false, _) => vec![case.iter[1], 1],
        (true, 0) => vec![1, 0],
        (true, _) => vec![0, 1],
    };
    TypedBuffer::new(index_dtype, case.iter.to_vec(), st, bytes)
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
    let k = lift_gather_opdef(&op, dtype, 2).expect("the gather lifts");
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

fn scatter_op(case: Case, add: bool, dtype: ElementKind, idx: ElementKind) -> OpDef {
    if add {
        OpDef::scatter_add("s", &[dtype], case.axis, idx)
    } else {
        OpDef::scatter("s", &[dtype], case.axis, idx)
    }
}

fn scatter(
    case: Case,
    add: bool,
    dtype: ElementKind,
    idx: ElementKind,
) -> (KernelFn, Vec<KernelArg>) {
    let op = scatter_op(case, add, dtype, idx);
    let k = lift_scatter_opdef(&op, dtype, 2).expect("the scatter lifts");
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
fn gathers_agree_for_every_policy_and_index_type() {
    for (n, idx) in [I32, I64, U32].into_iter().enumerate() {
        for (m, oob) in [OobPolicy::Skip, OobPolicy::Clamp, OobPolicy::ZeroFill]
            .into_iter()
            .enumerate()
        {
            let (k, a) = gather(FULL0, oob, F32, idx);
            agree(&format!("gather_{n}_{m}"), &k, a);
        }
    }
}

#[test]
fn gathers_agree_on_the_other_axis_broadcast_ids_and_f64() {
    let (k, a) = gather(FULL1, OobPolicy::Clamp, F32, I32);
    agree("gather_axis1", &k, a);
    let (k, a) = gather(IDS0, OobPolicy::ZeroFill, F32, I64);
    agree("gather_ids", &k, a);
    let (k, a) = gather(FULL0, OobPolicy::Skip, F64, I32);
    agree("gather_f64", &k, a);
}

#[test]
fn scatters_agree_for_assign_and_add() {
    for (n, idx) in [I32, I64, U32].into_iter().enumerate() {
        for (m, add) in [false, true].into_iter().enumerate() {
            let (k, a) = scatter(FULL0, add, F32, idx);
            agree(&format!("scatter_{n}_{m}"), &k, a);
        }
    }
    for (n, case) in [FULL1, IDS0].into_iter().enumerate() {
        let (k, a) = scatter(case, true, F32, I32);
        agree(&format!("scatter_case_{n}"), &k, a);
    }
    let (k, a) = scatter(FULL0, true, F64, I32);
    agree("scatter_f64", &k, a);
}

#[test]
fn integer_compute_is_a_typed_decline() {
    // Control: the float scatter lowers. The integer one is refused with the typed decline, not
    // spelled by default and not a panic.
    let (k, _) = scatter(FULL0, true, F32, I32);
    assert!(CpuC.lower_kernel(&k).is_ok());
    let op = OpDef::scatter_add("s", &[I32], 0, I32);
    let ik = lift_scatter_opdef(&op, I32, 2).expect("the integer scatter lifts");
    assert!(matches!(
        CpuC.lower_kernel(&ik),
        Err(LowerError::UnsupportedKernel { .. })
    ));
}

/// What the C reports for `k` on `a` once `from` is replaced by `to` in its source (the anchor
/// asserted present first); with an empty `from`, the unmutated C.
fn reports(tag: &str, k: &KernelFn, a: &[KernelArg], from: &str, to: &str) -> (i32, Vec<Vec<u8>>) {
    let ran = run_c_mutated(tag, k, a, |s| {
        if from.is_empty() {
            return s;
        }
        assert!(
            s.contains(from),
            "{tag}: the anchor `{from}` is in the emitted C:
{s}"
        );
        s.replacen(from, to, 1)
    })
    .expect("a host C compiler");
    (ran.status, ran.outs)
}

const NEG: &str = "(l5) < (((int)(0LL)))";
const PAST: &str = "(l5) >= (l6)";
const CLAMP_LO: &str = "(l5) > (((int)(0LL))) ? (l5)";

#[test]
fn the_corpus_catches_a_wrong_bounds_test_policy_or_combine() {
    // Seeded mutants of the emitted C; each must change what the C reports (the unmutated C agrees
    // with the interpreter above), or the corpus could not tell a wrong policy from a right one.
    let (k, a) = gather(FULL0, OobPolicy::Skip, F32, I32);
    let base = reports("gs_skip_base", &k, &a, "", "");
    for (tag, from, to) in [
        ("gs_skip_neg", NEG, "(l5) <= (((int)(0LL)))"),
        ("gs_skip_past", PAST, "(l5) > (l6)"),
    ] {
        assert_ne!(
            reports(tag, &k, &a, from, to),
            base,
            "{tag}: went unnoticed"
        );
    }
    let (k, a) = gather(FULL0, OobPolicy::Clamp, F32, I32);
    let base = reports("gs_clamp_base", &k, &a, "", "");
    let to = "(l5) < (((int)(0LL))) ? (l5)";
    assert_ne!(reports("gs_clamp_lo", &k, &a, CLAMP_LO, to), base, "clamp");

    let (k, a) = scatter(FULL0, true, F32, I32);
    let base = reports("gs_add_base", &k, &a, "", "");
    for (tag, from, to) in [
        ("gs_add_neg", NEG, "(l5) <= (((int)(0LL)))"),
        ("gs_add_past", PAST, "(l5) > (l6)"),
        ("gs_add_assign", "(l7 + l4)", "(l4)"),
    ] {
        assert_ne!(
            reports(tag, &k, &a, from, to),
            base,
            "{tag}: went unnoticed"
        );
    }
}
