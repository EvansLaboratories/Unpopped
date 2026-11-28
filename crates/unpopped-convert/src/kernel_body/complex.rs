// SPDX-License-Identifier: MIT OR Apache-2.0
//! `cuComplex.h` (ROADMAP U19 census slice; 12 real instantiations stopped at the type `cuFloatComplex`/`cuDoubleComplex`). The
//! header is NOT in the census corpus (`#include <cuComplex.h>` is a system include), so the names are builtins here, **pinned to
//! the text of CUDA 13.3's `include/cuComplex.h`**:
//!
//! - `typedef float2 cuFloatComplex;` (line 77), `typedef double2 cuDoubleComplex;` (186), `typedef cuFloatComplex cuComplex;` (299):
//!   type aliases of the vector types (`Walker::type_aliases`), a source typedef of the same name wins;
//! - `cuCrealf(x)` / `cuCreal(x)` is `x.x` (79, 188), `cuCimagf` / `cuCimag` is `x.y` (84, 192);
//! - `make_cuFloatComplex(r, i)`, `make_cuDoubleComplex`, `make_cuComplex` (89, 196, 301): the pair `(r, i)`;
//! - `cuConjf(x)` = `(x.x, -x.y)` (98-100); `cuCaddf(x, y)` = `(x.x + y.x, x.y + y.y)` (102-106); `cuCsubf` the same with `-`
//!   (109-113);
//! - `cuCmulf(x, y)` = `((x.x * y.x) - (x.y * y.y), (x.x * y.y) + (x.y * y.x))` (121-128): the header's own plain operations, **no
//!   fused multiply-add**: the lowered IR carries plain `Mul`/`Sub`/`Add` nodes and never a `Fma` (`tests/kernel_body_complex.rs` pins
//!   that shape). A backend must spell it with no contraction (nvcc's default `-fmad=true` would fuse). The ORACLE cannot tell fused from
//!   unfused by value: it evaluates a statement's expression at f64 and rounds once (`oracle/kernel_eval.rs`), see
//!   `docs/ir-numerics-qualifiers.md` 6b.
//!
//! `NaN` and infinity follow the formula, not a complex-arithmetic convention (`(inf, 0) * (0, inf)` is `(NaN, inf)` here).
//! `cuCdivf` (scaling inside), `cuCabsf` (a `hypot` inside), `cuCfmaf` and their double forms decline by name.

use super::*;

/// The lane type of a complex name's precision: the `f` forms are `float`, the plain ones `double`.
fn precision(name: &str, single: &[&str], double: &[&str]) -> Option<E> {
    if single.contains(&name) {
        Some(E::F32)
    } else if double.contains(&name) {
        Some(E::F64)
    } else {
        None
    }
}

impl<'s, 't> Walker<'s, 't> {
    /// Seeds the `cuComplex.h` type names as aliases of the vector types, unless the source defines its own.
    pub(super) fn seed_complex_aliases(&mut self) {
        for (name, to) in [
            ("cuFloatComplex", "float2"),
            ("cuDoubleComplex", "double2"),
            ("cuComplex", "float2"),
        ] {
            self.type_aliases
                .entry(name.to_string())
                .or_insert_with(|| to.to_string());
        }
    }

    /// `cuCrealf(z)`, `cuCimagf(z)`, `cuCreal(z)`, `cuCimag(z)`; a name that is not one declines (`cuCdivf`, `cuCabsf`, `cuCfmaf`...) or
    /// is `None`.
    pub(super) fn complex_scalar(
        &mut self,
        call: Node<'_>,
        name: &str,
    ) -> Option<R<(ScalarExpr, E)>> {
        let (elem, lane) = match name {
            "cuCrealf" => (E::F32, 0),
            "cuCimagf" => (E::F32, 1),
            "cuCreal" => (E::F64, 0),
            "cuCimag" => (E::F64, 1),
            "cuCdivf" | "cuCdiv" | "cuCabsf" | "cuCabs" | "cuCfmaf" | "cuCfma" => {
                return Some(self.not_yet(
                    call,
                    format!(
                        "`{name}` (cuComplex.h scales, takes a hypot or fuses: not lifted; only add, sub, mul, conj, real, imag and make_ are)"
                    ),
                ));
            }
            _ => return None,
        };
        let args = self.call_args(call);
        let [z] = args.as_slice() else {
            return Some(self.not_yet(call, format!("`{name}` with {} argument(s)", args.len())));
        };
        Some((|| {
            let lanes = self.vector_value(*z, (elem, 2))?;
            Ok((lanes[lane].clone(), elem))
        })())
    }

    /// The two lanes of a complex-valued call: `make_cuFloatComplex(r, i)` (also `make_cuDoubleComplex`, `make_cuComplex`),
    /// `cuCaddf`, `cuCsubf`, `cuCmulf`, `cuConjf` and the `double` forms without the `f`. `None` when `n` is not one, or the lanes
    /// asked for are not that precision's pair.
    pub(super) fn complex_lanes(
        &mut self,
        n: Node<'_>,
        (elem, lanes): (E, u8),
    ) -> R<Option<Vec<ScalarExpr>>> {
        if n.kind() != "call_expression" || lanes != 2 {
            return Ok(None);
        }
        let Some((name, is_make)) = self.complex_callee(n, elem)? else {
            return Ok(None);
        };
        let name = name.as_str();
        let args = self.call_args(n);
        if is_make {
            let [r, i] = args.as_slice() else {
                return self.not_yet(n, format!("`{name}` with {} argument(s)", args.len()));
            };
            return Ok(Some(self.complex_make(elem, [*r, *i])?));
        }
        let want = if name.starts_with("cuCconj") || name.starts_with("cuConj") {
            1
        } else {
            2
        };
        if args.len() != want {
            return self.not_yet(n, format!("`{name}` with {} argument(s)", args.len()));
        }
        let x = self.vector_value(args[0], (elem, 2))?;
        let y = if want == 1 {
            None
        } else {
            Some(self.vector_value(args[1], (elem, 2))?)
        };
        Ok(Some(complex_arith(name, &x, y.as_deref())))
    }

    /// The `cuComplex.h` callee of `n` and whether it is a `make_` constructor; `None` when it is not one of the lifted family.
    fn complex_callee(&mut self, n: Node<'_>, elem: E) -> R<Option<(String, bool)>> {
        let Some(name) = n
            .child_by_field_name("function")
            .map(|f| self.short_callee(text(f, self.src)).to_string())
        else {
            return Ok(None);
        };
        let name = name.as_str();
        if matches!(name, "cuCdivf" | "cuCdiv" | "cuCfmaf" | "cuCfma") {
            return self.not_yet(
                n,
                format!(
                    "`{name}` (cuComplex.h scales or fuses: not lifted; only add, sub, mul, conj, real, imag and make_ are)"
                ),
            );
        }
        let make = precision(
            name,
            &["make_cuFloatComplex", "make_cuComplex"],
            &["make_cuDoubleComplex"],
        );
        let op = precision(
            name,
            &["cuCaddf", "cuCsubf", "cuCmulf", "cuConjf"],
            &["cuCadd", "cuCsub", "cuCmul", "cuConj"],
        );
        let Some(prec) = make.or(op) else {
            return Ok(None);
        };
        if prec != elem {
            return self.not_yet(
                n,
                format!("`{name}` gives {prec:?} lanes where {elem:?} lanes are wanted"),
            );
        }
        Ok(Some((name.to_string(), make.is_some())))
    }

    /// `make_cuFloatComplex(r, i)`: the two lanes converted to the element type.
    fn complex_make(&mut self, elem: E, parts: [Node<'_>; 2]) -> R<Vec<ScalarExpr>> {
        let mut out = Vec::new();
        for a in parts {
            let (v, t) = self.expr(a)?;
            out.push(self.convert(v, t, elem, a)?);
        }
        Ok(out)
    }
}

/// The lanes of a `cuComplex.h` arithmetic call on lane vectors `x` (and `y` for the binary ones): plain operations, no fma.
fn complex_arith(name: &str, x: &[ScalarExpr], y: Option<&[ScalarExpr]>) -> Vec<ScalarExpr> {
    let b = |e: &ScalarExpr| Box::new(e.clone());
    let Some(y) = y else {
        // cuConjf (cuComplex.h 98-100)
        return vec![x[0].clone(), ScalarExpr::Unary(UnaryOp::Neg, b(&x[1]))];
    };
    match name.trim_end_matches('f') {
        // cuCaddf / cuCsubf (102-113)
        "cuCadd" => vec![
            ScalarExpr::Add(b(&x[0]), b(&y[0])),
            ScalarExpr::Add(b(&x[1]), b(&y[1])),
        ],
        "cuCsub" => vec![
            ScalarExpr::Sub(b(&x[0]), b(&y[0])),
            ScalarExpr::Sub(b(&x[1]), b(&y[1])),
        ],
        // cuCmulf (121-128): plain operations, no fma
        _ => vec![
            ScalarExpr::Sub(
                Box::new(ScalarExpr::Mul(b(&x[0]), b(&y[0]))),
                Box::new(ScalarExpr::Mul(b(&x[1]), b(&y[1]))),
            ),
            ScalarExpr::Add(
                Box::new(ScalarExpr::Mul(b(&x[0]), b(&y[1]))),
                Box::new(ScalarExpr::Mul(b(&x[1]), b(&y[0]))),
            ),
        ],
    }
}
