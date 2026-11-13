// SPDX-License-Identifier: MIT OR Apache-2.0
//! **Arithmetic on `F16` and `Bf16` in C agrees with the oracle** (ROADMAP U26): a narrow operand decodes to `float`, the
//! node computes in `float`, and the result is encoded once at the store. Compiled with a host C compiler, run, and
//! compared byte for byte with the oracle (which evaluates at `f64` and rounds once), over finite patterns.

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

fn bin(
    tag: &str,
    t: ElementKind,
    n: usize,
    e: impl Fn(ScalarExpr, ScalarExpr) -> ScalarExpr,
) -> KernelFn {
    looped(
        tag,
        vec![
            Param::buffer("a", t, BufferRole::Read, Some(1)),
            Param::buffer("b", t, BufferRole::Read, Some(1)),
            Param::buffer("out", t, BufferRole::Write, Some(1)),
        ],
        vec![t, t],
        n,
        vec![
            load(1, 0),
            Stmt::Load {
                local: 2,
                place: at(1, ScalarExpr::Local(0)),
            },
            store(2, e(ScalarExpr::Local(1), ScalarExpr::Local(2))),
        ],
    )
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

#[test]
fn add_sub_mul_and_compare_on_narrow_floats_agree() {
    let ops: Vec<(&str, Box<dyn Fn(ScalarExpr, ScalarExpr) -> ScalarExpr>)> = vec![
        (
            "add",
            Box::new(|a, b| ScalarExpr::Add(Box::new(a), Box::new(b))),
        ),
        (
            "sub",
            Box::new(|a, b| ScalarExpr::Sub(Box::new(a), Box::new(b))),
        ),
        (
            "mul",
            Box::new(|a, b| ScalarExpr::Mul(Box::new(a), Box::new(b))),
        ),
        (
            "lt",
            Box::new(|a, b| ScalarExpr::Binary(BinaryOp::CmpLt, Box::new(a), Box::new(b))),
        ),
    ];
    for t in [ElementKind::F16, ElementKind::Bf16] {
        let a = finite_only(t, scatter16(6000));
        let b = finite_only(t, scatter16(7000).into_iter().rev().collect());
        let n = a.len().min(b.len());
        for (name, f) in &ops {
            let tag = format!("{name}_{t:?}");
            let kernel = bin(&tag, t, n, f);
            // the arithmetic is spelled, not refused (it was a typed refusal before ROADMAP U26)
            let lowered = CpuC.lower_kernel(&kernel);
            assert!(lowered.is_ok(), "{tag}: {lowered:?}");
            agree(
                &tag,
                &kernel,
                vec![
                    narrow(t, &a[..n]),
                    narrow(t, &b[..n]),
                    narrow(t, &vec![0; n]),
                ],
            );
        }
    }
}
