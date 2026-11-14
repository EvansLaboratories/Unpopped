// SPDX-License-Identifier: MIT OR Apache-2.0
//! `F16` and `Bf16` in a kernel (ROADMAP U14): what the serial C reference spells, and the guard that keeps it from
//! spelling a wrong number.
//!
//! A narrow float lives in C as its 16-bit pattern (an `unsigned short`): a buffer element, a parameter, a local. The
//! patterns move unchanged through `Load`, `Store`, `Select` and `CopyAsync`; the only arithmetic C sees on them is
//! the codec of [`unpopped::cfamily::half_helpers`], reached through a `Cast`. Arithmetic, comparison or any other
//! operator applied to the pattern itself would compute on integers, so [`guard`] refuses it with a typed error
//! (`Cast` to `F32` first) instead of emitting it.

use unpopped::ir::{CastSpec, RoundMode, Saturation, ScalarExpr};
use unpopped::kernel::{KernelFn, ParamKind};
use unpopped_vocab::ElementKind;

/// `F16` or `Bf16`: a value C carries as a bare 16-bit pattern.
pub(crate) const fn is_narrow(t: ElementKind) -> bool {
    matches!(t, ElementKind::F16 | ElementKind::Bf16)
}

/// The decode helper (`float` from the pattern) of a narrow float, or `None` for any other element.
pub(crate) const fn load_fn(t: ElementKind) -> Option<&'static str> {
    match t {
        ElementKind::F16 => Some("unpopped_f16_load"),
        ElementKind::Bf16 => Some("unpopped_bf16_load"),
        _ => None,
    }
}

/// The encode helper (the pattern from a `float`, rounded to nearest even) of a narrow float: the codec's `store` with
/// a NaN's sign kept (the codec writes the canonical positive NaN; the oracle's cast keeps the sign bit).
pub(crate) const fn store_fn(t: ElementKind) -> Option<&'static str> {
    match t {
        ElementKind::F16 => Some("unp_f16_enc"),
        ElementKind::Bf16 => Some("unp_bf16_enc"),
        _ => None,
    }
}

/// The encode helper that rounds a `double` to a narrow float once (nearest even), for a source `float` cannot hold
/// without a first rounding: an `F64`, or a 32/64-bit integer (ROADMAP U26). A NaN keeps its sign.
pub(crate) const fn exact_store_fn(t: ElementKind) -> Option<&'static str> {
    match t {
        ElementKind::F16 => Some("unp_d_to_f16"),
        ElementKind::Bf16 => Some("unp_d_to_bf16"),
        _ => None,
    }
}

/// The C of [`exact_store_fn`]: the codec's algorithm in `double`, where every scaled significand is exact.
const fn exact_encoder(t: ElementKind) -> Option<&'static str> {
    match t {
        ElementKind::F16 => Some(
            "static unsigned short unp_d_to_f16(double x) {
    unsigned short sign = signbit(x) ? 0x8000u : 0u;
    double a = fabs(x), q, f;
    int e = 0, sub;
    if (x != x) { return (unsigned short)(sign | 0x7E00u); }
    if (a >= 65520.0) { return (unsigned short)(sign | 0x7C00u); }
    sub = a < 6.103515625e-05;
    if (sub) { q = ldexp(a, 24); } else { frexp(a, &e); q = ldexp(a, 11 - e); }
    f = floor(q);
    if (q - f > 0.5 || (q - f == 0.5 && ((long)f & 1))) { f += 1.0; }
    if (sub) { return (unsigned short)(sign | (unsigned short)f); }
    return (unsigned short)(sign | (((e + 14) << 10) + ((int)f - 1024)));
}
",
        ),
        ElementKind::Bf16 => Some(
            "static unsigned short unp_d_to_bf16(double x) {
    unsigned short sign = signbit(x) ? 0x8000u : 0u;
    double a = fabs(x), q, f;
    int e = 0, sub;
    if (x != x) { return (unsigned short)(sign | 0x7FC0u); }
    if (a >= ldexp(511.0, 119)) { return (unsigned short)(sign | 0x7F80u); }
    sub = a < ldexp(1.0, -126);
    if (sub) { q = ldexp(a, 133); } else { frexp(a, &e); q = ldexp(a, 8 - e); }
    f = floor(q);
    if (q - f > 0.5 || (q - f == 0.5 && ((long)f & 1))) { f += 1.0; }
    if (sub) { return (unsigned short)(sign | (unsigned short)f); }
    return (unsigned short)(sign | (((e + 126) << 7) + ((int)f - 128)));
}
",
        ),
        _ => None,
    }
}

/// The wrapper [`store_fn`] names for a narrow float: the codec's `store`, then the sign of a NaN.
const fn encoder(t: ElementKind) -> Option<&'static str> {
    match t {
        ElementKind::F16 => Some(
            "static unsigned short unp_f16_enc(float x) {
    return (unsigned short)(unpopped_f16_store(x) | ((x != x && signbit(x)) ? 0x8000u : 0u));
}
",
        ),
        ElementKind::Bf16 => Some(
            "static unsigned short unp_bf16_enc(float x) {
    return (unsigned short)(unpopped_bf16_store(x) | ((x != x && signbit(x)) ? 0x8000u : 0u));
}
",
        ),
        _ => None,
    }
}

/// The codec source for each helper `body` calls, in the order `F16`, `Bf16` (empty when it calls neither).
pub(crate) fn helpers_for(body: &str) -> String {
    let mut out = String::new();
    for t in [ElementKind::F16, ElementKind::Bf16] {
        let encodes = store_fn(t).is_some_and(|f| body.contains(f));
        let decodes = load_fn(t).is_some_and(|f| body.contains(f));
        let Some(codec) = unpopped::cfamily::half_helpers(t).filter(|_| encodes || decodes) else {
            continue;
        };
        out.push_str(codec);
        if let (true, Some(wrapper)) = (encodes, encoder(t)) {
            out.push_str(wrapper);
        }
    }
    // the exact encoders need only <math.h>, not the codec
    for t in [ElementKind::F16, ElementKind::Bf16] {
        if let (Some(name), Some(text)) = (exact_store_fn(t), exact_encoder(t)) {
            if body.contains(name) {
                out.push_str(text);
            }
        }
    }
    out
}

fn children(e: &ScalarExpr) -> Vec<&ScalarExpr> {
    match e {
        ScalarExpr::Input(_)
        | ScalarExpr::Const(_)
        | ScalarExpr::Param(_)
        | ScalarExpr::Reduced(_)
        | ScalarExpr::Coord(_)
        | ScalarExpr::CoordInt(..)
        | ScalarExpr::ConstInt(..)
        | ScalarExpr::Local(_)
        | ScalarExpr::Exec(_) => vec![],
        ScalarExpr::Add(a, b)
        | ScalarExpr::Sub(a, b)
        | ScalarExpr::Mul(a, b)
        | ScalarExpr::Div(a, b)
        | ScalarExpr::Binary(_, a, b) => vec![a, b],
        ScalarExpr::Unary(_, x)
        | ScalarExpr::Cast(x, _)
        | ScalarExpr::Bitcast(x, _)
        | ScalarExpr::Qualified(x, _)
        | ScalarExpr::Extract(x, _)
        | ScalarExpr::BitcastPacked(x, ..) => vec![x],
        ScalarExpr::Fma(a, b, c) | ScalarExpr::Select(a, b, c) => vec![a, b, c],
        ScalarExpr::Pack(xs) => xs.iter().collect(),
    }
}

fn spec(to: ElementKind) -> CastSpec {
    CastSpec {
        to,
        round: RoundMode::NearestEven,
        sat: Saturation::Wrap,
    }
}

/// `e` with its children replaced, in the order [`children`] lists them.
fn rebuild(e: &ScalarExpr, mut kids: Vec<ScalarExpr>) -> ScalarExpr {
    let mut next = || Box::new(kids.remove(0));
    match e {
        ScalarExpr::Add(..) => ScalarExpr::Add(next(), next()),
        ScalarExpr::Sub(..) => ScalarExpr::Sub(next(), next()),
        ScalarExpr::Mul(..) => ScalarExpr::Mul(next(), next()),
        ScalarExpr::Div(..) => ScalarExpr::Div(next(), next()),
        ScalarExpr::Binary(op, ..) => ScalarExpr::Binary(*op, next(), next()),
        ScalarExpr::Unary(op, _) => ScalarExpr::Unary(*op, next()),
        ScalarExpr::Cast(_, s) => ScalarExpr::Cast(next(), *s),
        ScalarExpr::Bitcast(_, t) => ScalarExpr::Bitcast(next(), *t),
        ScalarExpr::Qualified(_, q) => ScalarExpr::Qualified(next(), *q),
        ScalarExpr::Extract(_, i) => ScalarExpr::Extract(next(), *i),
        ScalarExpr::BitcastPacked(_, t, n) => ScalarExpr::BitcastPacked(next(), *t, *n),
        ScalarExpr::Fma(..) => ScalarExpr::Fma(next(), next(), next()),
        ScalarExpr::Select(..) => ScalarExpr::Select(next(), next(), next()),
        ScalarExpr::Pack(_) => ScalarExpr::Pack(kids),
        leaf => leaf.clone(),
    }
}

/// The narrow float `e` is typed as, or `None`: a narrow local or parameter, a `Cast` to one, and any arithmetic,
/// comparison or `Select` over one (a comparison yields 1.0 or 0.0 in its operands' dtype).
fn narrow_ty(k: &KernelFn, e: &ScalarExpr) -> Option<ElementKind> {
    match e {
        // a constant has the kernel's compute dtype
        ScalarExpr::Const(_) => Some(k.compute).filter(|t| is_narrow(*t)),
        ScalarExpr::Local(i) => k
            .locals
            .get(*i as usize)
            .map(|d| d.elem)
            .filter(|t| is_narrow(*t)),
        ScalarExpr::Param(i) => match k.params.get(*i as usize).map(|p| &p.kind) {
            Some(ParamKind::Scalar { ty, .. }) if is_narrow(*ty) => Some(*ty),
            _ => None,
        },
        ScalarExpr::Cast(_, s) => Some(s.to).filter(|t| is_narrow(*t)),
        ScalarExpr::Select(_, a, b) => narrow_ty(k, a).or_else(|| narrow_ty(k, b)),
        ScalarExpr::Bitcast(..) | ScalarExpr::Extract(..) | ScalarExpr::BitcastPacked(..) => None,
        _ => children(e).into_iter().find_map(|c| narrow_ty(k, c)),
    }
}

/// A leaf or cast whose C value already is the narrow float's bit pattern.
fn is_pattern(e: &ScalarExpr) -> bool {
    matches!(
        e,
        ScalarExpr::Local(_) | ScalarExpr::Param(_) | ScalarExpr::Cast(..)
    )
}

/// `e` as a value C can compute on: every narrow operand is decoded to `float`, so arithmetic runs in `float` with no
/// rounding between nodes (the oracle's `f64` and the device's `f32` differ from this by the tolerance band, rounded once
/// at the store).
fn wide(k: &KernelFn, e: &ScalarExpr) -> ScalarExpr {
    if let ScalarExpr::Cast(x, sp) = e {
        let inner = ScalarExpr::Cast(Box::new(moved(k, x)), *sp);
        return if is_narrow(sp.to) {
            ScalarExpr::Cast(Box::new(inner), spec(ElementKind::F32))
        } else {
            inner
        };
    }
    if narrow_ty(k, e).is_some() && is_pattern(e) {
        return ScalarExpr::Cast(Box::new(e.clone()), spec(ElementKind::F32));
    }
    if let ScalarExpr::Binary(op, ..) = e {
        if op.is_cmp() && is_narrow(k.compute) {
            // the oracle decides a comparison on the compute dtype's value of each operand: a constant is rounded to it
            // (arithmetic keeps the exact constant and rounds once at the store)
            let kids = children(e).into_iter().map(|c| match c {
                ScalarExpr::Const(_) => ScalarExpr::Cast(
                    Box::new(ScalarExpr::Cast(Box::new(c.clone()), spec(k.compute))),
                    spec(ElementKind::F32),
                ),
                _ => wide(k, c),
            });
            return rebuild(e, kids.collect());
        }
    }
    rebuild(e, children(e).into_iter().map(|c| wide(k, c)).collect())
}

/// `e` as a value that is moved, bound or stored: a bare pattern stays one; narrow arithmetic is computed wide and encoded
/// once; a `Cast` reads its operand wide.
pub(crate) fn moved(k: &KernelFn, e: &ScalarExpr) -> ScalarExpr {
    match (narrow_ty(k, e), e) {
        (_, ScalarExpr::Cast(x, sp)) => ScalarExpr::Cast(Box::new(moved(k, x)), *sp),
        (Some(_), e) if is_pattern(e) => e.clone(),
        (Some(_), ScalarExpr::Select(c, a, b)) => ScalarExpr::Select(
            Box::new(wide(k, c)),
            Box::new(moved(k, a)),
            Box::new(moved(k, b)),
        ),
        (Some(t), e) => ScalarExpr::Cast(Box::new(wide(k, e)), spec(t)),
        (None, e) => wide(k, e),
    }
}

/// `e` as an operand (a condition, an index): decoded wide.
pub(crate) fn operand(k: &KernelFn, e: &ScalarExpr) -> ScalarExpr {
    wide(k, e)
}
