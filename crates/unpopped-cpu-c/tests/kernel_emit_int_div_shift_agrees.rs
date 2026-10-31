// SPDX-License-Identifier: MIT OR Apache-2.0
//! **The emitted C agrees with the kernel interpreter on integer division and shifts**
//! (`docs/int-div-shift-semantics.md`, PM ruling 2026-10-31): for `I32`, `I64`, `U32` and `U64`,
//! `CpuC::lower_kernel` spells `/`, `<<` and `>>` through total helpers, and the compiled program
//! produces the interpreter's bytes on the ruling's vectors (divisors and dividends over `0, 1, -1,
//! MIN, MAX`; counts `0, 1, N-1, N, N+1, 255, -1`).
//!
//! The interpreter is itself checked against host-native integer types in
//! `crates/unpopped/tests/kernel_int_div_shift_semantics.rs`, so the chain is: rule, native
//! reference, interpreter, compiled C. Three seeded mutants of the emitted source (drop the
//! zero-divisor guard, drop the `MIN / -1` guard, narrow the shift mask) must each be seen.
//!
//! On a platform whose integer divide traps (x86), an unguarded `b == 0` kills the program; that
//! counts as seen, as does any output that differs from the interpreter's.

use std::panic::{AssertUnwindSafe, catch_unwind};

use unpopped::backend::Backend;
use unpopped::ir::{BinaryOp, ScalarExpr};
use unpopped::kernel::{Block, BufferRole, KernelFn, LocalDecl, Param, Place, Stmt};
use unpopped::oracle::TypedBuffer;
use unpopped::oracle::kernel_eval::{KernelArg, run_kernel};
use unpopped::types::{int_range, typed_bits};
use unpopped_cpu_c::CpuC;
use unpopped_vocab::ElementKind;

#[path = "support/cc.rs"]
mod cc;
#[path = "support/kernel_run.rs"]
mod kernel_run;
use kernel_run::{agree, run_c_mutated};

const I32: ElementKind = ElementKind::I32;
const INTS: [ElementKind; 4] = [I32, ElementKind::I64, ElementKind::U32, ElementKind::U64];

#[derive(Copy, Clone, Debug, PartialEq)]
enum Op {
    Div,
    Shl,
    Shr,
}

fn local(i: u32) -> Box<ScalarExpr> {
    Box::new(ScalarExpr::Local(i))
}

fn at_i(buffer: u32) -> Place {
    Place::Logical {
        buffer,
        coords: vec![ScalarExpr::Local(1)],
    }
}

/// `out[i] = a[i] <op> b[i]` for `i` in `0..n`, at compute dtype `t`.
fn kernel(t: ElementKind, op: Op) -> KernelFn {
    let value = match op {
        Op::Div => ScalarExpr::Div(local(2), local(3)),
        Op::Shl => ScalarExpr::Binary(BinaryOp::Shl, local(2), local(3)),
        Op::Shr => ScalarExpr::Binary(BinaryOp::Shr, local(2), local(3)),
    };
    let body = Block(vec![
        Stmt::Load {
            local: 2,
            place: at_i(0),
        },
        Stmt::Load {
            local: 3,
            place: at_i(1),
        },
        Stmt::Store {
            place: at_i(2),
            value,
        },
    ]);
    let looped = Stmt::For {
        local: 1,
        start: ScalarExpr::ConstInt(0, I32),
        end: ScalarExpr::Local(0),
        step: ScalarExpr::ConstInt(1, I32),
        body,
    };
    KernelFn::new(
        "int_op",
        t,
        vec![
            Param::buffer("a", t, BufferRole::Read, Some(1)),
            Param::buffer("b", t, BufferRole::Read, Some(1)),
            Param::buffer("out", t, BufferRole::Write, Some(1)),
            Param::scalar("n", I32, 0),
        ],
        vec![
            LocalDecl::scalar(I32),
            LocalDecl::scalar(I32),
            LocalDecl::scalar(t),
            LocalDecl::scalar(t),
        ],
        Block(vec![looped]),
    )
}

fn width(t: ElementKind) -> usize {
    typed_bits(t).expect("an integer type") as usize / 8
}

/// `-1` on an unsigned type means all ones.
fn raw(t: ElementKind, v: i128) -> i128 {
    let (lo, hi) = int_range(t).expect("an integer type");
    if v < 0 && lo == 0 { hi } else { v }
}

/// The ruling's vectors: every dividend against every divisor or count.
fn args(t: ElementKind, op: Op) -> Vec<KernelArg> {
    let (lo, hi) = int_range(t).expect("an integer type");
    let n = i128::from(typed_bits(t).expect("an integer type"));
    let mut lhs = vec![0, 1, -1, lo, hi, hi - 1, 7, -7, 1 << (n - 2)];
    let mut rhs = match op {
        Op::Div => vec![0, 1, -1, lo, hi, 2, -3, 7],
        _ => vec![0, 1, n - 1, n, n + 1, 255, -1, 32, 31, 63, 64],
    };
    for v in lhs.iter_mut().chain(rhs.iter_mut()) {
        *v = raw(t, *v);
    }
    lhs.sort_unstable();
    lhs.dedup();
    rhs.sort_unstable();
    rhs.dedup();
    let a: Vec<i128> = lhs
        .iter()
        .flat_map(|&d| rhs.iter().map(move |_| d))
        .collect();
    let b: Vec<i128> = lhs.iter().flat_map(|_| rhs.clone()).collect();
    let len = a.len();
    let buf = |vs: &[i128]| {
        let bytes = vs
            .iter()
            .flat_map(|v| v.to_le_bytes()[..width(t)].to_vec())
            .collect();
        KernelArg::Buffer(TypedBuffer::new(t, vec![len as i64], vec![1], bytes))
    };
    vec![
        buf(&a),
        buf(&b),
        KernelArg::Buffer(TypedBuffer::new(
            t,
            vec![len as i64],
            vec![1],
            vec![0u8; len * width(t)],
        )),
        KernelArg::Int(len as i128),
    ]
}

#[test]
fn the_emitted_c_agrees_with_the_interpreter_on_the_ruled_vectors() {
    for t in INTS {
        for op in [Op::Div, Op::Shl, Op::Shr] {
            agree(&format!("idiv_{t:?}_{op:?}"), &kernel(t, op), args(t, op));
        }
    }
}

/// Run the mutated C and the interpreter on the same inputs: `true` if the program dies or its
/// output differs.
fn seen(tag: &str, t: ElementKind, op: Op, mutate: impl FnOnce(String) -> String) -> bool {
    let k = kernel(t, op);
    let mut interp = args(t, op);
    run_kernel(&k, &mut interp).expect("the interpreter runs");
    let KernelArg::Buffer(want) = &interp[2] else {
        unreachable!("the output is the third argument")
    };
    let ran = catch_unwind(AssertUnwindSafe(|| {
        run_c_mutated(tag, &k, &args(t, op), mutate)
    }));
    match ran {
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
fn a_dropped_zero_divisor_guard_is_seen() {
    let m = swap(
        "if (b == 0 || (a == -2147483647 - 1 && b == -1)) return a;",
        "if ((a == -2147483647 - 1 && b == -1)) return a;",
    );
    assert!(seen("idiv_mut_zero", I32, Op::Div, m));
}

#[test]
fn a_dropped_min_over_minus_one_guard_is_seen() {
    let m = swap(
        "if (b == 0 || (a == -2147483647 - 1 && b == -1)) return a;",
        "if (b == 0) return a;",
    );
    assert!(seen("idiv_mut_min", I32, Op::Div, m));
}

#[test]
fn a_narrowed_shift_mask_is_seen() {
    let m = swap(
        "return (int)((unsigned int)a << ((unsigned int)c & 31u));",
        "return (int)((unsigned int)a << ((unsigned int)c & 15u));",
    );
    assert!(seen("idiv_mut_mask", I32, Op::Shl, m));
}

#[test]
fn the_unmutated_source_is_not_seen_as_broken() {
    // The control for the three above: the same judge, no mutation, says "fine".
    assert!(!seen("idiv_ctl", I32, Op::Div, |s| s));
    assert!(!seen("idiv_ctl_shl", I32, Op::Shl, |s| s));
}

/// The emitter spells each operation through the total helper for its type, not a bare C operator.
#[test]
fn the_emitter_calls_the_total_helpers_not_bare_operators() {
    for (t, ty) in [(I32, "i32"), (ElementKind::U64, "u64")] {
        for (op, name) in [(Op::Div, "div"), (Op::Shl, "shl"), (Op::Shr, "shr")] {
            let g = CpuC
                .lower_kernel(&kernel(t, op))
                .unwrap_or_else(|e| panic!("{t:?} {op:?} lowers: {e}"));
            let call = format!("unp_{name}_{ty}(l2, l3)");
            assert!(
                g.source.contains(&call),
                "{t:?} {op:?}: no `{call}` in the source"
            );
        }
    }
}
