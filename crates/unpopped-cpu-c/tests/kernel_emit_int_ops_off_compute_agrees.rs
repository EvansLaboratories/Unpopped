// SPDX-License-Identifier: MIT OR Apache-2.0
//! **The emitted C agrees with the kernel interpreter on integer `Div`, shifts and bitwise operators off the compute type**
//! (U19 finding, pm-kernels option A, 2026-11-22). The kernel layer now admits these at any integer type in a kernel whose
//! compute type is a float (`crates/unpopped/tests/kernel_int_ops_any_width.rs` checks the interpreter against the ruling). Here
//! the same kernels are lowered by `CpuC::lower_kernel`, compiled, and run on the ruled vectors: where cpu-c lowers an
//! (operator, width) pair its output must equal the interpreter's byte for byte; where it does not, the refusal must be a typed
//! `LowerError` naming the construct, never wrong C.

use unpopped::backend::Backend;
use unpopped::ir::{BinaryOp, ScalarExpr};
use unpopped::kernel::{Block, BufferRole, KernelFn, LocalDecl, Param, Place, Stmt, Unroll};
use unpopped::oracle::TypedBuffer;
use unpopped::oracle::kernel_eval::KernelArg;
use unpopped::types::{int_range, typed_bits};
use unpopped_cpu_c::CpuC;
use unpopped_vocab::ElementKind as E;

#[path = "support/cc.rs"]
mod cc;
#[path = "support/kernel_run.rs"]
mod kernel_run;
use kernel_run::{agree, run_c_mutated};
use std::panic::{AssertUnwindSafe, catch_unwind};
use unpopped::oracle::kernel_eval::run_kernel;

const INTS: [E; 8] = [E::I8, E::U8, E::I16, E::U16, E::I32, E::U32, E::I64, E::U64];

#[derive(Copy, Clone, Debug, PartialEq)]
enum Op {
    Div,
    Shl,
    Shr,
    And,
    Or,
    Xor,
}

const OPS: [Op; 6] = [Op::Div, Op::Shl, Op::Shr, Op::And, Op::Or, Op::Xor];

fn local(i: u32) -> Box<ScalarExpr> {
    Box::new(ScalarExpr::Local(i))
}

fn at_i(buffer: u32) -> Place {
    Place::Logical {
        buffer,
        coords: vec![ScalarExpr::Local(0)],
    }
}

/// `out_k[i] = a[i] <ops[k]> b[i]` for `i` in `0..n`, operands and results of type `t`, kernel compute type `F32`: one
/// output buffer per operation, so one compile covers every operation a type lowers.
fn kernel(t: E, ops: &[Op], n: usize) -> KernelFn {
    let value = |op| match op {
        Op::Div => ScalarExpr::Div(local(1), local(2)),
        Op::Shl => ScalarExpr::Binary(BinaryOp::Shl, local(1), local(2)),
        Op::Shr => ScalarExpr::Binary(BinaryOp::Shr, local(1), local(2)),
        Op::And => ScalarExpr::Binary(BinaryOp::BitAnd, local(1), local(2)),
        Op::Or => ScalarExpr::Binary(BinaryOp::BitOr, local(1), local(2)),
        Op::Xor => ScalarExpr::Binary(BinaryOp::BitXor, local(1), local(2)),
    };
    let mut body = vec![
        Stmt::Load {
            local: 1,
            place: at_i(0),
        },
        Stmt::Load {
            local: 2,
            place: at_i(1),
        },
    ];
    let mut params = vec![
        Param::buffer("a", t, BufferRole::Read, Some(1)),
        Param::buffer("b", t, BufferRole::Read, Some(1)),
    ];
    for (k, &op) in ops.iter().enumerate() {
        body.push(Stmt::Store {
            place: at_i(2 + k as u32),
            value: value(op),
        });
        params.push(Param::buffer(
            format!("out{k}"),
            t,
            BufferRole::Write,
            Some(1),
        ));
    }
    KernelFn::new(
        "int_op",
        E::F32,
        params,
        vec![
            LocalDecl::scalar(E::I32),
            LocalDecl::scalar(t),
            LocalDecl::scalar(t),
        ],
        Block(vec![Stmt::For {
            local: 0,
            start: ScalarExpr::ConstInt(0, E::I32),
            end: ScalarExpr::ConstInt(n as i64, E::I32),
            step: ScalarExpr::ConstInt(1, E::I32),
            unroll: Unroll::None,
            body: Block(body),
        }]),
    )
}

fn width(t: E) -> usize {
    typed_bits(t).expect("an integer type") as usize / 8
}

/// Dividends and bit patterns: both signs, the extremes, a lone high bit, an alternating pattern.
fn lhs(t: E) -> Vec<i128> {
    let (lo, hi) = int_range(t).expect("an integer type");
    let bits = i128::from(typed_bits(t).expect("an integer type"));
    let mut v = vec![0, 1, lo, hi, hi - 1, 7, 85, 1 << (bits - 2), -1, -7, -85];
    v.retain(|x| (lo..=hi).contains(x));
    v.sort_unstable();
    v.dedup();
    v
}

/// The right operands, one set for every operation (so one input drives them all): divisors (zero, one, `-1`, the extremes),
/// counts (the width, past it, negative) and bit patterns.
fn rhs(t: E) -> Vec<i128> {
    let (lo, hi) = int_range(t).expect("an integer type");
    let bits = i128::from(typed_bits(t).expect("an integer type"));
    let mut v = vec![
        0,
        1,
        -1,
        lo,
        hi,
        2,
        -3,
        7,
        bits - 1,
        bits,
        bits + 1,
        31,
        32,
        33,
        63,
        64,
        85,
        -86,
    ];
    v.retain(|x| (lo..=hi).contains(x));
    v.sort_unstable();
    v.dedup();
    v
}

fn buffer(t: E, vs: &[i128]) -> TypedBuffer {
    let bytes = vs
        .iter()
        .flat_map(|v| v.to_le_bytes()[..width(t)].to_vec())
        .collect();
    TypedBuffer::new(t, vec![vs.len() as i64], vec![1], bytes)
}

fn args(t: E, outs: usize) -> (usize, Vec<KernelArg>) {
    let (l, r) = (lhs(t), rhs(t));
    let a: Vec<i128> = l.iter().flat_map(|&x| r.iter().map(move |_| x)).collect();
    let b: Vec<i128> = l.iter().flat_map(|_| r.clone()).collect();
    let n = a.len();
    let mut v = vec![
        KernelArg::Buffer(buffer(t, &a)),
        KernelArg::Buffer(buffer(t, &b)),
    ];
    v.extend((0..outs).map(|_| KernelArg::Buffer(buffer(t, &vec![0; n]))));
    (n, v)
}

/// The operations cpu-c lowers at `t` in a float kernel, and the typed refusals of the rest.
fn split(t: E) -> (Vec<Op>, Vec<String>) {
    let (mut lowered, mut declined) = (Vec::new(), Vec::new());
    for op in OPS {
        match CpuC.lower_kernel(&kernel(t, &[op], 4)) {
            Ok(_) => lowered.push(op),
            Err(e) => declined.push(format!("{op:?}: {e}")),
        }
    }
    (lowered, declined)
}

#[test]
fn the_emitted_c_agrees_with_the_interpreter_wherever_cpu_c_lowers_and_declines_by_name_elsewhere()
{
    for t in INTS {
        let (lowered, declined) = split(t);
        // Every operation that lowers, in ONE kernel: one compile, outputs compared byte for byte.
        let (n, a) = args(t, lowered.len());
        agree(&format!("off_{t:?}"), &kernel(t, &lowered, n), a);
        // Exactly the narrow `Div`, `Shl`, `Shr` refuse (cpu-c has total helpers for 32- and 64-bit types only), and each refusal
        // is a typed `LowerError` that names the operation and the type.
        let narrow = typed_bits(t).is_some_and(|b| b < 32);
        let want_declined = if narrow {
            vec![Op::Div, Op::Shl, Op::Shr]
        } else {
            vec![]
        };
        assert_eq!(lowered.len() + declined.len(), OPS.len());
        assert_eq!(declined.len(), want_declined.len(), "{t:?}: {declined:?}");
        for (d, op) in declined.iter().zip(&want_declined) {
            assert!(
                d.contains("cpu_c has no integer") && d.contains(&format!("{t:?}")),
                "{t:?} {op:?}: {d}"
            );
        }
    }
}

/// `true` if the mutated C and the interpreter disagree, or the mutated program dies.
fn seen(tag: &str, t: E, op: Op, mutate: impl FnOnce(String) -> String) -> bool {
    let (n, a) = args(t, 1);
    let k = kernel(t, &[op], n);
    let mut interp = args(t, 1).1;
    run_kernel(&k, &mut interp).expect("the interpreter runs");
    let KernelArg::Buffer(want) = &interp[2] else {
        unreachable!()
    };
    match catch_unwind(AssertUnwindSafe(|| run_c_mutated(tag, &k, &a, mutate))) {
        Err(_) => true,
        Ok(None) => panic!("{tag}: no host C compiler; the mutant cannot be judged"),
        Ok(Some(r)) => r.outs[0] != want.bytes,
    }
}

/// Replace `from` with `to` in the emitted source, failing if `from` is not there exactly once.
fn swap(from: &'static str, to: &'static str) -> impl FnOnce(String) -> String {
    move |src| {
        assert_eq!(src.matches(from).count(), 1, "the mutation anchor `{from}`");
        src.replace(from, to)
    }
}

#[test]
fn a_wrong_helper_or_a_wrong_operator_in_the_emitted_c_is_seen() {
    // the control first: the same judge, no mutation, says "fine"
    assert!(
        !seen("off_ctl_shr", E::U64, Op::Shr, |s| s),
        "the unmutated source must not be seen as broken"
    );
    // `>>` spelled as `<<`, a division spelled with the wrong helper, `&` spelled `|`
    assert!(seen(
        "off_mut_shr",
        E::U64,
        Op::Shr,
        swap("unp_shr_u64(l1, l2)", "unp_shl_u64(l1, l2)")
    ));
    assert!(seen(
        "off_mut_div",
        E::I32,
        Op::Div,
        swap("unp_div_i32(l1, l2)", "unp_div_i32(l2, l1)")
    ));
    assert!(seen(
        "off_mut_and",
        E::I8,
        Op::And,
        swap(
            "((signed char)((l1) & (l2)))",
            "((signed char)((l1) | (l2)))"
        )
    ));
}
