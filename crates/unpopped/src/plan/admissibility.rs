// SPDX-License-Identifier: MIT OR Apache-2.0
//! Section of the plan gate; split from `plan.rs` (ROADMAP U30). Paths unchanged.

use super::*;

/// Whether `e` contains a [`ScalarExpr::Coord`] leaf anywhere — drives the
/// increment-0d Strided schedule routing in [`build_plan`] (a Coord body must
/// reach the one emitter that materializes per-axis coordinates) and mirrors
/// `contract::expr_contains_cmp` in shape.
pub(crate) fn expr_contains_coord(e: &ScalarExpr) -> bool {
    match e {
        ScalarExpr::Coord(_) | ScalarExpr::CoordInt(..) => true,
        ScalarExpr::Input(_)
        | ScalarExpr::Const(_)
        | ScalarExpr::ConstInt(..)
        | ScalarExpr::Local(_)
        | ScalarExpr::Exec(_)
        | ScalarExpr::Param(_)
        | ScalarExpr::Reduced(_) => false,
        ScalarExpr::Unary(_, x) => expr_contains_coord(x),
        ScalarExpr::Cast(x, _)
        | ScalarExpr::Bitcast(x, _)
        | ScalarExpr::Qualified(x, _)
        | ScalarExpr::Extract(x, _)
        | ScalarExpr::BitcastPacked(x, ..) => expr_contains_coord(x),
        ScalarExpr::Pack(xs) => xs.iter().any(expr_contains_coord),
        ScalarExpr::Add(a, b)
        | ScalarExpr::Sub(a, b)
        | ScalarExpr::Mul(a, b)
        | ScalarExpr::Div(a, b)
        | ScalarExpr::Binary(_, a, b) => expr_contains_coord(a) || expr_contains_coord(b),
        ScalarExpr::Select(c, a, b) | ScalarExpr::Fma(c, a, b) => {
            expr_contains_coord(c) || expr_contains_coord(a) || expr_contains_coord(b)
        }
    }
}

/// Increment-0d [`ScalarExpr::Coord`] admissibility gate — runs at the TOP of
/// [`build_plan`] and walks the expressions of EVERY Access arm (the 0a
/// lesson: emitter backstops alone are bypassed by the reduction-class
/// lowering paths), with independent emitter backstops in `cuda`. Three
/// validate-reject rules, all honest misses:
///
/// 1. **Access**: Coord is legal ONLY in an [`Access::Elementwise`] body (v1).
///    A coordinate along a reduced/folded axis is ambiguous (which fold
///    iteration produced the output element?), and the RowReduce/Contraction
///    epilogues iterate their own coordinate spaces ((row, j) and (m, n)) —
///    lifting Coord into them needs explicit per-arm semantics, deferred.
/// 2. **Dtype**: `F32`/`F32Strict`/`F64` ONLY. f16/bf16 reject — the max
///    exactly-representable integer is 2048 (bf16: 256), which real axis
///    extents exceed, so a half coordinate would silently round. Int dtypes
///    reject — the coordinate lowers as a float cast (`(float)c{d}`), the
///    same double-math hazard as `Const`/`Param` at int dtypes; the
///    int-literal coordinate spelling is the queued follow-up.
/// 3. **Axis**: `axis < key.rank` — an out-of-range axis has no `c{d}` to
///    read (the emitter would spell an undefined identifier).
///
/// The exactness bound (f32 coordinates exact to 2²⁴, f64 to 2⁵³) is a CALLER
/// precondition — the key abstracts extents away, the same trust level as the
/// RowReduce column-weight extent precondition (see [`ScalarExpr::Coord`]).
pub(crate) fn check_coord_admissibility(op: &OpDef, key: &StructureKey) -> Result<(), PlanError> {
    let elementwise = matches!(op.access, Access::Elementwise);
    fn walk(
        e: &ScalarExpr,
        op_name: &str,
        dtype: ElementKind,
        rank: u8,
        elementwise: bool,
    ) -> Result<(), PlanError> {
        match e {
            ScalarExpr::Input(_)
            | ScalarExpr::Const(_)
            | ScalarExpr::Param(_)
            | ScalarExpr::Reduced(_) => Ok(()),
            ScalarExpr::ConstInt(..) | ScalarExpr::Local(_) | ScalarExpr::Exec(_) => Ok(()),
            ScalarExpr::CoordInt(d, _) => {
                if !elementwise {
                    return Err(PlanError::InadmissibleOpInCell {
                        detail: format!(
                            "op '{op_name}': CoordInt({d}) is Elementwise-only, like Coord (a                              coordinate along a reduced/folded axis is ambiguous); miss honestly"
                        ),
                    });
                }
                if *d >= rank {
                    return Err(PlanError::InadmissibleOpInCell {
                        detail: format!(
                            "op '{op_name}': CoordInt({d}) axis out of range for rank {rank} —                              the iteration space has no such coordinate"
                        ),
                    });
                }
                Ok(())
            }
            ScalarExpr::Coord(d) => {
                // Message text is preserved BYTE-FOR-BYTE from the `assert!`s
                // this replaced. `build_plan` panics with `Display`, which is
                // this `detail`, so every existing `#[should_panic(expected =
                // ...)]` against the panicking path keeps matching — including
                // an adopter's 34, which this crate cannot run.
                if !elementwise {
                    return Err(PlanError::InadmissibleOpInCell {
                        detail: format!(
                            "op '{op_name}': Coord({d}) is Elementwise-only in 0d — a coordinate \
                             along a reduced/folded axis is ambiguous (which fold iteration?), and \
                             the RowReduce/Contraction stages/epilogues iterate their own \
                             coordinate spaces; miss honestly"
                        ),
                    });
                }
                if !matches!(
                    dtype,
                    ElementKind::F32 | ElementKind::F32Strict | ElementKind::F64
                ) {
                    return Err(PlanError::InadmissibleOpAtDtype {
                        dtype,
                        detail: format!(
                            "op '{op_name}': Coord({d}) requires an f32/f64 compute dtype, got \
                             {dtype:?} — f16/bf16 coordinates round past extent 2048 (bf16: 256) \
                             and int dtypes would inject the float-cast coordinate into integer \
                             math (int-literal coordinate spelling is a follow-up); miss honestly"
                        ),
                    });
                }
                if *d >= rank {
                    return Err(PlanError::InadmissibleOpInCell {
                        detail: format!(
                            "op '{op_name}': Coord({d}) axis out of range for rank {rank} — the \
                             iteration space has no such coordinate"
                        ),
                    });
                }
                Ok(())
            }
            ScalarExpr::Unary(_, x) => walk(x, op_name, dtype, rank, elementwise),
            ScalarExpr::Cast(x, _)
            | ScalarExpr::Bitcast(x, _)
            | ScalarExpr::Qualified(x, _)
            | ScalarExpr::Extract(x, _)
            | ScalarExpr::BitcastPacked(x, ..) => walk(x, op_name, dtype, rank, elementwise),
            ScalarExpr::Pack(xs) => xs
                .iter()
                .try_for_each(|x| walk(x, op_name, dtype, rank, elementwise)),
            ScalarExpr::Add(a, b)
            | ScalarExpr::Sub(a, b)
            | ScalarExpr::Mul(a, b)
            | ScalarExpr::Div(a, b)
            | ScalarExpr::Binary(_, a, b) => {
                walk(a, op_name, dtype, rank, elementwise)?;
                walk(b, op_name, dtype, rank, elementwise)
            }
            ScalarExpr::Select(c, a, b) | ScalarExpr::Fma(c, a, b) => {
                walk(c, op_name, dtype, rank, elementwise)?;
                walk(a, op_name, dtype, rank, elementwise)?;
                walk(b, op_name, dtype, rank, elementwise)
            }
        }
    }
    let mut exprs: Vec<&ScalarExpr> = vec![&op.body];
    match &op.access {
        Access::RowReduce { stages, epilogue } => {
            exprs.extend(stages.iter().map(|s| &s.pre));
            exprs.push(epilogue);
        }
        Access::Contraction { epilogue, .. } => exprs.push(epilogue),
        // A Coord in a reduction post-expr is doubly rejected (here, non-
        // elementwise → the Coord arm fires; and by assert_valid_reduction_post).
        Access::Reduction { post, .. } => exprs.push(post),
        // Increment 6 SCAN: a Coord in `pre`/`post` is doubly rejected (here, the
        // scan is non-elementwise → the Coord arm fires; and the emitter's
        // panicking `coord` closure). The scan iterates the (row, j) space, not an
        // elementwise output coordinate space.
        Access::Scan { pre, post, .. } => {
            exprs.push(pre);
            exprs.push(post);
        }
        // Increment 7 WINDOW: a Coord in `pre`/`post` is doubly rejected (here, the
        // window is non-elementwise → the Coord arm fires; and the emitter's
        // panicking `coord` closure). The window iterates the (row, o) space, not an
        // elementwise output coordinate space.
        Access::Window { pre, post, .. } => {
            exprs.push(pre);
            exprs.push(post);
        }
        // Increment 8 SORT_PERM: `body` is pinned `Input(0)` (no Coord), and
        // RowSort is non-elementwise (`elementwise == false`), so any Coord that
        // ever appeared would be rejected by the Coord arm; nothing extra to walk.
        Access::RowSort { .. } => {}
        // Increment 11 IM2COL: `body` is pinned `Input(0)` (no Coord), and im2col is
        // non-elementwise (`elementwise == false`), so any Coord would be rejected by
        // the Coord arm; im2col has no pre/post — nothing extra to walk.
        Access::Im2Col { .. } => {}
        Access::Elementwise => {}
    }
    for e in exprs {
        walk(e, &op.name, key.dtype, key.rank, elementwise)?;
    }
    Ok(())
}

/// Increment-I1 typed-expression admissibility gate: [`ScalarExpr::Cast`] and
/// [`ScalarExpr::Bitcast`], the first nodes whose type is not the compute dtype
/// (`docs/ir-typed-expressions.md`). Runs at the top of [`try_build_plan`].
///
/// An op with neither node returns at once, so every pre-existing op plans exactly
/// as before. For one with either, v1 admits only:
///
/// 1. **an [`Access::Elementwise`] op with a single output** (a typed reduction,
///    contraction, scan or window epilogue needs per-arm semantics, deferred);
/// 2. **`f32`/`f32s`/`f64` compute** (the half, fp8, sub-byte and complex endpoints
///    are not in the conversion table yet);
/// 3. **a well-typed body** ([`crate::types::infer`]) whose type equals the store
///    type, `out_dtype` or the compute dtype;
/// 4. **typed values only at the root or directly under a `Cast`/`Bitcast`**:
///    arithmetic, `Select` and the math ops stay compute-typed, because their
///    lowering, hoisting and oracle semantics are per compute dtype today.
///
/// The typed-expression gate, with `total_ints` (`Backend::total_integer_ops`): at an integer compute
/// dtype a `ConstInt` of exactly that dtype is a compute-typed leaf, not a typed value, so it
/// does not send the body down the Cast/Bitcast rules.
pub(crate) fn check_typed_admissibility_for(
    op: &OpDef,
    dtype: ElementKind,
    total_ints: bool,
) -> Result<(), PlanError> {
    let beyond = (total_ints && is_int_dtype(dtype)).then_some(dtype);
    let typed = |e: &ScalarExpr| crate::types::is_typed_beyond(e, beyond);
    let any_typed = op.output_bodies().iter().any(|e| typed(e))
        || match &op.access {
            Access::RowReduce { stages, epilogue } => {
                stages.iter().any(|s| typed(&s.pre)) || typed(epilogue)
            }
            Access::Contraction { epilogue, .. } => typed(epilogue),
            Access::Reduction { post, .. } => typed(post),
            Access::Scan { pre, post, .. } | Access::Window { pre, post, .. } => {
                typed(pre) || typed(post)
            }
            _ => false,
        };
    if !any_typed {
        return Ok(());
    }
    let in_cell = |detail: String| Err(PlanError::InadmissibleOpInCell { detail });
    // Packed values are typed (section 9) but no emitter has a seam for them yet, so the
    // gate declines the op honestly rather than let a lowering decline mid-kernel.
    let mut all_exprs: Vec<&ScalarExpr> = op.output_bodies();
    match &op.access {
        Access::RowReduce { stages, epilogue } => {
            all_exprs.extend(stages.iter().map(|s| &s.pre));
            all_exprs.push(epilogue);
        }
        Access::Contraction { epilogue, .. } => all_exprs.push(epilogue),
        Access::Reduction { post, .. } => all_exprs.push(post),
        Access::Scan { pre, post, .. } | Access::Window { pre, post, .. } => {
            all_exprs.push(pre);
            all_exprs.push(post);
        }
        _ => {}
    }
    if all_exprs.iter().any(|e| crate::types::has_local(e)) {
        return in_cell(format!(
            "op '{}': Local and Exec are kernel-layer leaves (docs/ir-kernel-layer.md); an OpDef body must not contain one",
            op.name
        ));
    }
    if all_exprs.iter().any(|e| crate::types::has_packed(e)) {
        return in_cell(format!(
            "op '{}': Pack/Extract (packed values) have no lowering yet; miss honestly",
            op.name
        ));
    }
    if !matches!(op.access, Access::Elementwise) {
        return in_cell(format!(
            "op '{}': Cast/Bitcast are Elementwise-only in v1 (a typed reduction,              contraction, scan or window epilogue needs its own semantics); miss honestly",
            op.name
        ));
    }
    if op.output_bodies().len() != 1 {
        return in_cell(format!(
            "op '{}': Cast/Bitcast are single-output in v1; miss honestly",
            op.name
        ));
    }
    if !crate::types::is_cast_float(dtype) {
        return Err(PlanError::InadmissibleOpAtDtype {
            dtype,
            detail: format!(
                "op '{}': Cast/Bitcast need f32/f32s/f64 compute in v1 (the half, fp8,                  sub-byte and complex endpoints are not in the conversion table yet)",
                op.name
            ),
        });
    }
    let ill = |detail: String| Err(PlanError::InadmissibleOpAtDtype { dtype, detail });
    let root = match crate::types::infer(&op.body, dtype) {
        Ok(t) => t,
        Err(e) => return ill(format!("op '{}': ill-typed expression: {e:?}", op.name)),
    };
    let store = op.out_dtype.unwrap_or(dtype);
    if crate::types::is_mx_element(store) || crate::types::is_mx_element(root) {
        return ill(format!(
            "op '{}': an MX element ({root:?}) has no buffer layout yet (sub-byte float packing is a later slice), so it may be an intermediate under a Cast but not the stored result; miss honestly",
            op.name
        ));
    }
    if root != store {
        return ill(format!(
            "op '{}': the body has type {root:?} but the store type is {store:?};              state the conversion with a Cast, and set out_dtype to the result type",
            op.name
        ));
    }
    // Typed (non-compute) values only at the root or directly under a Cast/Bitcast,
    // except an INTEGER value, which `infer` already confines to the operators that
    // have an integer form (`Add`/`Sub`/`Mul`, `Select`, `Min`/`Max`, comparisons).
    fn walk(e: &ScalarExpr, compute: ElementKind, typed_ok: bool) -> Result<(), String> {
        let ty = crate::types::infer(e, compute).map_err(|x| format!("{x:?}"))?;
        if ty != compute && !typed_ok && !crate::types::is_cast_int(ty) {
            return Err(format!(
                "a value of type {ty:?} feeds an operator; v1 admits a non-compute-typed                  value only at the root or directly under a Cast/Bitcast"
            ));
        }
        match e {
            ScalarExpr::Cast(x, _)
            | ScalarExpr::Bitcast(x, _)
            | ScalarExpr::Qualified(x, _)
            | ScalarExpr::Extract(x, _)
            | ScalarExpr::BitcastPacked(x, ..) => walk(x, compute, true),
            ScalarExpr::Pack(xs) => xs.iter().try_for_each(|x| walk(x, compute, true)),
            ScalarExpr::Unary(_, x) => walk(x, compute, false),
            ScalarExpr::Add(a, b)
            | ScalarExpr::Sub(a, b)
            | ScalarExpr::Mul(a, b)
            | ScalarExpr::Div(a, b)
            | ScalarExpr::Binary(_, a, b) => {
                walk(a, compute, false)?;
                walk(b, compute, false)
            }
            ScalarExpr::Select(c, a, b) | ScalarExpr::Fma(c, a, b) => {
                walk(c, compute, false)?;
                walk(a, compute, false)?;
                walk(b, compute, false)
            }
            ScalarExpr::Input(_)
            | ScalarExpr::Const(_)
            | ScalarExpr::Param(_)
            | ScalarExpr::Reduced(_)
            | ScalarExpr::Coord(_)
            | ScalarExpr::CoordInt(..)
            | ScalarExpr::ConstInt(..)
            | ScalarExpr::Local(_)
            | ScalarExpr::Exec(_) => Ok(()),
        }
    }
    walk(&op.body, dtype, true).map_err(|why| PlanError::InadmissibleOpAtDtype {
        dtype,
        detail: format!("op '{}': {why}", op.name),
    })
}

/// Increment-I1 numerics-qualifier gate: [`ScalarExpr::Qualified`]
/// (`docs/ir-numerics-qualifiers.md`). Runs at the top of [`try_build_plan`].
///
/// An op with no `Qualified` node returns at once. For one with any, v1 admits:
///
/// 1. **an [`Access::Elementwise`] op with a single output**;
/// 2. **`f32`/`f32s` compute**;
/// 3. **no `Cast`/`Bitcast` in the same op** (the combination is not specified yet);
/// 4. per node: the wrapped expression is one of [`QualOp`]'s operations, a rounding
///    mode other than nearest-even only on an operation that takes one, `approx`
///    only on an operation that takes it, and never `approx` with a directed mode.
pub(crate) fn check_qualified_admissibility(
    op: &OpDef,
    dtype: ElementKind,
) -> Result<(), PlanError> {
    let mut exprs: Vec<&ScalarExpr> = op.output_bodies();
    match &op.access {
        Access::RowReduce { stages, epilogue } => {
            exprs.extend(stages.iter().map(|s| &s.pre));
            exprs.push(epilogue);
        }
        Access::Contraction { epilogue, .. } => exprs.push(epilogue),
        Access::Reduction { post, .. } => exprs.push(post),
        Access::Scan { pre, post, .. } | Access::Window { pre, post, .. } => {
            exprs.push(pre);
            exprs.push(post);
        }
        Access::RowSort { .. } | Access::Im2Col { .. } | Access::Elementwise => {}
    }
    if !exprs.iter().any(|e| crate::types::has_qualified(e)) {
        return Ok(());
    }
    let in_cell = |detail: String| Err(PlanError::InadmissibleOpInCell { detail });
    if !matches!(op.access, Access::Elementwise) {
        return in_cell(format!(
            "op '{}': rounding/ftz/approx qualifiers are Elementwise-only in v1; miss honestly",
            op.name
        ));
    }
    if op.output_bodies().len() != 1 {
        return in_cell(format!(
            "op '{}': rounding/ftz/approx qualifiers are single-output in v1; miss honestly",
            op.name
        ));
    }
    let ill = |detail: String| Err(PlanError::InadmissibleOpAtDtype { dtype, detail });
    if !matches!(dtype, ElementKind::F32 | ElementKind::F32Strict) {
        return ill(format!(
            "op '{}': rounding/ftz/approx qualifiers need f32/f32s compute in v1 (f64 has no              ftz form and no exact reference for directed rounding yet)",
            op.name
        ));
    }
    if crate::types::is_typed(&op.body) {
        return ill(format!(
            "op '{}': a Qualified operation combined with Cast/Bitcast is not specified yet",
            op.name
        ));
    }
    let dag = crate::ir::ExprDag::from_expr(&op.body);
    for id in 0..dag.len() as u32 {
        let DagNode::Qualified(child, n) = dag.node(id) else {
            continue;
        };
        let Some((qop, _)) = QualOp::of_node(dag.node(*child)) else {
            return ill(format!(
                "op '{}': a Qualified node wraps {:?}, which is not one of Add, Sub, Mul, Div,                  Fma, Sqrt, Rsqrt, Recip, Exp2, Log2",
                op.name,
                dag.node(*child)
            ));
        };
        let directed = n.round != RoundMode::NearestEven;
        if directed && !qop.takes_directed_round() {
            return ill(format!(
                "op '{}': {qop:?} takes no rounding mode (got {:?})",
                op.name, n.round
            ));
        }
        if n.approx && !qop.takes_approx() {
            return ill(format!("op '{}': {qop:?} has no approximate form", op.name));
        }
        if n.approx && directed {
            return ill(format!(
                "op '{}': {qop:?} cannot combine approx with RoundMode::{:?} (an approximation                  has no rounding direction to honour)",
                op.name, n.round
            ));
        }
    }
    Ok(())
}

/// Increment-I1 [`ScalarExpr::Fma`] admissibility gate: runs at the top of
/// [`try_build_plan`] and walks every expression of the op (output bodies and
/// the `Access` arm's own expressions).
///
/// `Fma` is admissible at `F32`/`F32Strict`/`F64` ONLY. `F16`/`Bf16` miss
/// honestly: the only way to spell a half FMA is to widen to `f32`, fuse there,
/// and round the `f32` result to half, which is a second rounding, so the result
/// is not the fused one. Integer, complex and bool dtypes are rejected by their
/// own walks (an FMA has no meaning there).
pub(crate) fn check_fma_admissibility(op: &OpDef, dtype: ElementKind) -> Result<(), PlanError> {
    fn has_fma(e: &ScalarExpr) -> bool {
        match e {
            ScalarExpr::Fma(..) => true,
            ScalarExpr::Input(_)
            | ScalarExpr::Const(_)
            | ScalarExpr::Param(_)
            | ScalarExpr::Reduced(_)
            | ScalarExpr::Coord(_)
            | ScalarExpr::CoordInt(..)
            | ScalarExpr::ConstInt(..)
            | ScalarExpr::Local(_)
            | ScalarExpr::Exec(_) => false,
            ScalarExpr::Unary(_, x) => has_fma(x),
            ScalarExpr::Cast(x, _)
            | ScalarExpr::Bitcast(x, _)
            | ScalarExpr::Qualified(x, _)
            | ScalarExpr::Extract(x, _)
            | ScalarExpr::BitcastPacked(x, ..) => has_fma(x),
            ScalarExpr::Pack(xs) => xs.iter().any(has_fma),
            ScalarExpr::Add(a, b)
            | ScalarExpr::Sub(a, b)
            | ScalarExpr::Mul(a, b)
            | ScalarExpr::Div(a, b)
            | ScalarExpr::Binary(_, a, b) => has_fma(a) || has_fma(b),
            ScalarExpr::Select(c, a, b) => has_fma(c) || has_fma(a) || has_fma(b),
        }
    }
    if !matches!(dtype, ElementKind::F16 | ElementKind::Bf16) {
        return Ok(());
    }
    let mut exprs: Vec<&ScalarExpr> = op.output_bodies();
    match &op.access {
        Access::RowReduce { stages, epilogue } => {
            exprs.extend(stages.iter().map(|s| &s.pre));
            exprs.push(epilogue);
        }
        Access::Contraction { epilogue, .. } => exprs.push(epilogue),
        Access::Reduction { post, .. } => exprs.push(post),
        Access::Scan { pre, post, .. } | Access::Window { pre, post, .. } => {
            exprs.push(pre);
            exprs.push(post);
        }
        Access::RowSort { .. } | Access::Im2Col { .. } | Access::Elementwise => {}
    }
    if exprs.into_iter().any(has_fma) {
        return Err(PlanError::InadmissibleOpAtDtype {
            dtype,
            detail: format!(
                "op '{}': Fma at {dtype:?} is rejected: a half fused multiply-add can only \
                 be spelled by widening to f32, fusing, and rounding to half, which rounds \
                 twice and is not the fused result; miss honestly",
                op.name
            ),
        });
    }
    Ok(())
}
