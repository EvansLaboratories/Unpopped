// SPDX-License-Identifier: MIT OR Apache-2.0
//! `F16` and `Bf16` in a kernel (ROADMAP U14): what the serial C reference spells, and the guard that keeps it from
//! spelling a wrong number.
//!
//! A narrow float lives in C as its 16-bit pattern (an `unsigned short`): a buffer element, a parameter, a local. The
//! patterns move unchanged through `Load`, `Store`, `Select` and `CopyAsync`; the only arithmetic C sees on them is
//! the codec of [`unpopped::cfamily::half_helpers`], reached through a `Cast`. Arithmetic, comparison or any other
//! operator applied to the pattern itself would compute on integers, so [`guard`] refuses it with a typed error
//! (`Cast` to `F32` first) instead of emitting it.

use unpopped::backend::LowerError;
use unpopped::ir::ScalarExpr;
use unpopped::kernel::{KernelFn, ParamKind};
use unpopped_vocab::ElementKind;

use crate::kernel::refuse;

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

/// Refuse `e` when a narrow float's bit pattern is the operand of anything but a `Cast`, a `Select` arm or (when `moved`
/// holds, the expression being a stored, bound or assigned value) the whole expression. `moved` is true exactly where
/// the pattern is copied rather than computed on.
pub(crate) fn guard(k: &KernelFn, e: &ScalarExpr, moved: bool) -> Result<(), LowerError> {
    let narrow = match e {
        ScalarExpr::Local(i) => k
            .locals
            .get(*i as usize)
            .map(|d| d.elem)
            .filter(|t| is_narrow(*t)),
        ScalarExpr::Param(i) => match k.params.get(*i as usize).map(|p| &p.kind) {
            Some(ParamKind::Scalar { ty, .. }) if is_narrow(*ty) => Some(*ty),
            _ => None,
        },
        ScalarExpr::Cast(_, spec) if is_narrow(spec.to) => Some(spec.to),
        _ => None,
    };
    if let (Some(t), false) = (narrow, moved) {
        return Err(refuse(format!(
            "arithmetic or a comparison on a {t:?} value; Cast it to F32 first (the narrow float is a bit pattern in C)"
        )));
    }
    match e {
        // a Cast reads its operand as a pattern and decodes it: the one place a pattern is consumed
        ScalarExpr::Cast(x, _) => guard(k, x, true),
        ScalarExpr::Select(c, a, b) => {
            guard(k, c, false)?;
            guard(k, a, moved)?;
            guard(k, b, moved)
        }
        _ => children(e).into_iter().try_for_each(|c| guard(k, c, false)),
    }
}
