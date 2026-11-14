// SPDX-License-Identifier: MIT OR Apache-2.0
//! **A narrow compute dtype in C agrees with the oracle** (ROADMAP U26): a kernel whose compute dtype is `F16` or `Bf16`
//! (constants are that narrow float, the arithmetic computes in `float`, one encode at the store), compiled with a host C
//! compiler, run, and compared byte for byte with the oracle over finite patterns.

use unpopped::backend::Backend;
use unpopped::ir::{BinaryOp, ScalarExpr};
use unpopped::kernel::{Block, BufferRole, KernelFn, LocalDecl, Param, Place, Stmt};
use unpopped::oracle::TypedBuffer;
use unpopped::oracle::kernel_eval::KernelArg;
use unpopped_cpu_c::CpuC;
use unpopped_vocab::ElementKind;

#[path = "support/cc.rs"]
mod cc;
#[allow(dead_code)]
#[path = "support/kernel_run.rs"]
mod kernel_run;
#[allow(dead_code)]
use kernel_run::agree;

const F32: ElementKind = ElementKind::F32;
const I32: ElementKind = ElementKind::I32;

fn ci(v: i64) -> ScalarExpr {
    ScalarExpr::ConstInt(v, I32)
}

fn at(buffer: u32, i: ScalarExpr) -> Place {
    Place::Logical {
        buffer,
        coords: vec![i],
    }
}

fn narrow(t: ElementKind, patterns: &[u16]) -> KernelArg {
    let bytes = patterns.iter().flat_map(|p| p.to_le_bytes()).collect();
    KernelArg::Buffer(TypedBuffer::new(
        t,
        vec![patterns.len() as i64],
        vec![1],
        bytes,
    ))
}

/// `for i in 0..n { body(i) }` over locals `0` (the index, `I32`) and the given extras.
fn looped(
    name: &str,
    params: Vec<Param>,
    extra: Vec<ElementKind>,
    n: usize,
    body: Vec<Stmt>,
) -> KernelFn {
    let mut locals = vec![LocalDecl::scalar(I32)];
    locals.extend(extra.into_iter().map(LocalDecl::scalar));
    KernelFn::new(
        name,
        F32,
        params,
        locals,
        Block(vec![Stmt::For {
            local: 0,
            start: ci(0),
            end: ci(n as i64),
            step: ci(1),
            body: Block(body),
        }]),
    )
}

fn load(local: u32, buffer: u32) -> Stmt {
    Stmt::Load {
        local,
        place: at(buffer, ScalarExpr::Local(0)),
    }
}

fn store(buffer: u32, value: ScalarExpr) -> Stmt {
    Stmt::Store {
        place: at(buffer, ScalarExpr::Local(0)),
        value,
    }
}

fn scatter16(n: usize) -> Vec<u16> {
    let mut s = 0x9E37_79B9_7F4A_7C15u64;
    (0..n)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            (s >> 24) as u16
        })
        .collect()
}

fn finite_only(t: ElementKind, v: Vec<u16>) -> Vec<u16> {
    // NaN payloads and signs are the move's concern (U25), not the arithmetic's
    let exp_mask = if t == ElementKind::F16 {
        0x7C00
    } else {
        0x7F80
    };
    v.into_iter().filter(|p| p & exp_mask != exp_mask).collect()
}

fn unary_kernel(
    tag: &str,
    t: ElementKind,
    n: usize,
    e: impl Fn(ScalarExpr) -> ScalarExpr,
) -> KernelFn {
    let mut k = looped(
        tag,
        vec![
            Param::buffer("a", t, BufferRole::Read, Some(1)),
            Param::buffer("out", t, BufferRole::Write, Some(1)),
        ],
        vec![t],
        n,
        vec![load(1, 0), store(1, e(ScalarExpr::Local(1)))],
    );
    k.compute = t;
    k
}

#[test]
fn a_narrow_compute_dtype_with_constants_and_unary_ops_agrees() {
    let b = |x: ScalarExpr| Box::new(x);
    let ops: Vec<(&str, Box<dyn Fn(ScalarExpr) -> ScalarExpr>)> = vec![
        (
            "add_const",
            Box::new(move |x| ScalarExpr::Add(b(x), b(ScalarExpr::Const(0.1)))),
        ),
        (
            "mul_const",
            Box::new(move |x| ScalarExpr::Mul(b(x), b(ScalarExpr::Const(3.0)))),
        ),
        (
            "lt_const",
            Box::new(move |x| ScalarExpr::Binary(BinaryOp::CmpLt, b(x), b(ScalarExpr::Const(0.5)))),
        ),
        (
            "lt_const_inexact",
            Box::new(move |x| ScalarExpr::Binary(BinaryOp::CmpLt, b(x), b(ScalarExpr::Const(0.1)))),
        ),
        // no narrow leaf at all: the expression is narrow only because the compute dtype is
        ("const_only", Box::new(move |_| ScalarExpr::Const(0.1))),
        (
            "const_sum",
            Box::new(move |_| {
                ScalarExpr::Add(b(ScalarExpr::Const(0.1)), b(ScalarExpr::Const(0.2)))
            }),
        ),
        (
            "sqr",
            Box::new(move |x| ScalarExpr::Mul(b(x.clone()), b(x))),
        ),
    ];
    for t in [ElementKind::F16, ElementKind::Bf16] {
        let mut a = finite_only(t, scatter16(6000));
        // the patterns at the narrow float nearest 0.1, where a comparison against it is decided by rounding the constant
        a.extend(if t == ElementKind::F16 {
            [0x2E65, 0x2E66, 0x2E67]
        } else {
            [0x3DCC, 0x3DCD, 0x3DCE]
        });
        let n = a.len();
        for (name, f) in &ops {
            let tag = format!("{name}_{t:?}");
            let kernel = unary_kernel(&tag, t, n, f);
            // a narrow compute dtype is spelled, not refused (it was a typed refusal before ROADMAP U26)
            let lowered = CpuC.lower_kernel(&kernel);
            assert!(lowered.is_ok(), "{tag}: {lowered:?}");
            agree(&tag, &kernel, vec![narrow(t, &a), narrow(t, &vec![0; n])]);
        }
    }
}
