// SPDX-License-Identifier: MIT OR Apache-2.0
//! **`Div`, the transcendental unary ops and a compound expression on `F16`/`Bf16` agree with the oracle** (ROADMAP U34).
//! The C evaluates in `float` and encodes once at the store; the oracle evaluates at `f64` and rounds once. Measured over EVERY
//! finite pattern (2026-11-14, a host libm): `Div`, `Sqrt`, `Rsqrt` and `Tanh` agree byte for byte on both dtypes; `Exp`, `Log`,
//! `Sin` and `Cos` on `F16` differ in 1 to 4 of 63,488 patterns, each by exactly one narrow ulp (a `float` libm result is not
//! correctly rounded, and the rounding to 11 bits lands on the other side of a tie); `Bf16` agrees everywhere. The decision
//! (ROADMAP U34): those four transcendentals are held to **one narrow ulp**, as the oracle's tolerance band already promises
//! (`unit_roundoff`), and everything else, a compound expression included, to the same bound; `Div`, `Sqrt`, `Rsqrt` and `Tanh`
//! stay exact, so a regression to a double rounding shows. The test samples every 16th pattern to stay fast; the measurement
//! above was the exhaustive one.

use unpopped::backend::Backend;
use unpopped::ir::{ScalarExpr, UnaryOp};
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

/// One kernel computing every op of `ops` on the same input, one output buffer each (one C compile per dtype).
fn un_all(tag: &str, t: ElementKind, n: usize, ops: &[UnaryOp]) -> KernelFn {
    let mut params = vec![Param::buffer("a", t, BufferRole::Read, Some(1))];
    let mut body = vec![load(1, 0)];
    for (k, op) in ops.iter().enumerate() {
        params.push(Param::buffer(
            format!("out{k}"),
            t,
            BufferRole::Write,
            Some(1),
        ));
        body.push(store(
            1 + k as u32,
            ScalarExpr::Unary(*op, Box::new(ScalarExpr::Local(1))),
        ));
    }
    looped(tag, params, vec![t], n, body)
}

/// The position of a finite 16-bit pattern on the number line, so `-0` and `+0` coincide and neighbours differ by one.
fn ordered(p: u16) -> i64 {
    let m = i64::from(p & 0x7FFF);
    if p & 0x8000 != 0 { -m } else { m }
}

/// Run `k` in C and in the oracle; assert every element of writable buffer `j` is within `tols[j]` narrow ulps.
fn within(tag: &str, k: &KernelFn, args: Vec<KernelArg>, tols: &[i64]) {
    let Some(c) = kernel_run::run_c(tag, k, &args) else {
        return;
    };
    assert_eq!(c.status, 0, "{tag}: the C failed");
    let mut interp = args;
    unpopped::oracle::kernel_eval::run_kernel(k, &mut interp).expect("the oracle runs it");
    let u16s = |b: &[u8]| -> Vec<u16> {
        b.chunks(2)
            .map(|c| u16::from_le_bytes([c[0], c[1]]))
            .collect()
    };
    let outs: Vec<Vec<u16>> = k
        .params
        .iter()
        .zip(&interp)
        .filter_map(|(p, a)| match (&p.kind, a) {
            (unpopped::kernel::ParamKind::Buffer { role, .. }, KernelArg::Buffer(b))
                if *role != BufferRole::Read =>
            {
                Some(u16s(&b.bytes))
            }
            _ => None,
        })
        .collect();
    assert_eq!(outs.len(), tols.len(), "{tag}: output count");
    for (j, (want, tol)) in outs.iter().zip(tols).enumerate() {
        let got = u16s(&c.outs[j]);
        assert_eq!(want.len(), got.len(), "{tag}: element count");
        for (i, (g, w)) in got.iter().zip(want).enumerate() {
            let d = (ordered(*g) - ordered(*w)).abs();
            assert!(
                d <= *tol,
                "{tag}: output {j} element {i}: C {g:#06x} vs oracle {w:#06x} ({d} ulps > {tol})"
            );
        }
    }
}

fn sampled(t: ElementKind) -> Vec<u16> {
    finite_only(t, (0..=u16::MAX).step_by(16).collect())
}

#[test]
fn unary_ops_on_narrow_floats_stay_within_their_bound() {
    const OPS: [(UnaryOp, i64); 7] = [
        (UnaryOp::Sqrt, 0),
        (UnaryOp::Rsqrt, 0),
        (UnaryOp::Tanh, 0),
        (UnaryOp::Exp, 1),
        (UnaryOp::Log, 1),
        (UnaryOp::Sin, 1),
        (UnaryOp::Cos, 1),
    ];
    let ops: Vec<UnaryOp> = OPS.iter().map(|o| o.0).collect();
    let tols: Vec<i64> = OPS.iter().map(|o| o.1).collect();
    for t in [ElementKind::F16, ElementKind::Bf16] {
        let all = sampled(t);
        let tag = format!("unary_{t:?}");
        let kernel = un_all(&tag, t, all.len(), &ops);
        assert!(CpuC.lower_kernel(&kernel).is_ok(), "{tag}: refused");
        let mut args = vec![narrow(t, &all)];
        args.extend(ops.iter().map(|_| narrow(t, &vec![0; all.len()])));
        within(&tag, &kernel, args, &tols);
    }
}

fn pair(t: ElementKind) -> (Vec<u16>, Vec<u16>) {
    let a = finite_only(t, scatter16(6000));
    let b = finite_only(t, scatter16(7000).into_iter().rev().collect());
    let n = a.len().min(b.len());
    (a[..n].to_vec(), b[..n].to_vec())
}

#[test]
fn division_on_narrow_floats_is_exact() {
    for t in [ElementKind::F16, ElementKind::Bf16] {
        let (a, b) = pair(t);
        let tag = format!("div_{t:?}");
        let kernel = bin(&tag, t, a.len(), |x, y| {
            ScalarExpr::Div(Box::new(x), Box::new(y))
        });
        assert!(CpuC.lower_kernel(&kernel).is_ok(), "{tag}: refused");
        within(
            &tag,
            &kernel,
            vec![narrow(t, &a), narrow(t, &b), narrow(t, &vec![0; a.len()])],
            &[0],
        );
    }
}

/// `(a * b) + a`: two operations in the `float` domain before one encode; the oracle holds the intermediate at `f64`.
#[test]
fn a_two_deep_expression_on_narrow_floats_stays_within_one_ulp() {
    for t in [ElementKind::F16, ElementKind::Bf16] {
        let (a, b) = pair(t);
        let tag = format!("fma_shape_{t:?}");
        let kernel = bin(&tag, t, a.len(), |x, y| {
            ScalarExpr::Add(
                Box::new(ScalarExpr::Mul(Box::new(x.clone()), Box::new(y))),
                Box::new(x),
            )
        });
        assert!(CpuC.lower_kernel(&kernel).is_ok(), "{tag}: refused");
        within(
            &tag,
            &kernel,
            vec![narrow(t, &a), narrow(t, &b), narrow(t, &vec![0; a.len()])],
            &[1],
        );
    }
}
