// SPDX-License-Identifier: MIT OR Apache-2.0
//! Section of the plan gate; split from `plan.rs` (ROADMAP U30). Paths unchanged.

use super::*;

/// `true` if a contraction epilogue references only admissible leaves: `Reduced(0)`
/// (the K-sum), constants, and — when `n_inputs > 2` — the fused bias `Input(i)`
/// for `2 <= i < n_inputs` (a per-column `[N]` bias broadcast over the M rows).
/// Never `Input(0)`/`Input(1)` (lhs/rhs are consumed by the contraction, not the
/// epilogue), `Param`, another reduced stage, or `Coord`. For `n_inputs == 2` no
/// `Input` is admitted, so this is byte-identical to the plain-contraction
/// "Reduced(0) only" rule.
pub(super) fn contraction_epilogue_admissible(e: &crate::ir::ScalarExpr, n_inputs: u8) -> bool {
    use crate::ir::ScalarExpr as E;
    match e {
        E::Reduced(0) | E::Const(_) => true,
        // A fused bias leaf: Input(i), 2 <= i < n_inputs.
        E::Input(i) => *i >= 2 && *i < n_inputs,
        // Coord rejects here too: a contraction epilogue iterates the (m, n)
        // output space, not an elementwise cell's — Coord's v1 semantics are
        // Elementwise-only (`check_coord_admissibility` fires first with the
        // targeted message; this arm keeps the predicate honest regardless).
        E::Param(_)
        | E::Reduced(_)
        | E::Coord(_)
        | E::CoordInt(..)
        | E::ConstInt(..)
        | E::Local(_)
        | E::Exec(_) => false,
        E::Unary(_, x) => contraction_epilogue_admissible(x, n_inputs),
        E::Cast(x, _)
        | E::Bitcast(x, _)
        | E::Qualified(x, _)
        | E::Extract(x, _)
        | E::BitcastPacked(x, ..) => contraction_epilogue_admissible(x, n_inputs),
        E::Pack(xs) => xs
            .iter()
            .all(|x| contraction_epilogue_admissible(x, n_inputs)),
        E::Add(a, b) | E::Sub(a, b) | E::Mul(a, b) | E::Div(a, b) | E::Binary(_, a, b) => {
            contraction_epilogue_admissible(a, n_inputs)
                && contraction_epilogue_admissible(b, n_inputs)
        }
        E::Select(c, a, b) | E::Fma(c, a, b) => {
            contraction_epilogue_admissible(c, n_inputs)
                && contraction_epilogue_admissible(a, n_inputs)
                && contraction_epilogue_admissible(b, n_inputs)
        }
    }
}

/// Validate that a [`crate::ir::ContractionAxes`] role assignment is legal for
/// operands of the given ranks: the role vector must match the operand's rank,
/// each side must carry exactly the free role it's allowed to produce (`lhs` ⇒
/// one `FreeM`, no `FreeN`; `rhs` ⇒ one `FreeN`, no `FreeM`), exactly one
/// `ContractedK` per operand (multi-K-group contraction is deferred past v1),
/// and `lhs`/`rhs` must agree on how many `Batch` axes they carry. A pure
/// predicate over roles only — it does not look at concrete extents/strides.
/// Consulted by [`build_plan`]'s `Access::Contraction` arm.
pub(crate) fn validate_contraction_roles(
    axes: &crate::ir::ContractionAxes,
    lhs_rank: usize,
    rhs_rank: usize,
) -> Result<(), String> {
    use crate::ir::AxisRole::*;
    if axes.lhs.len() != lhs_rank || axes.rhs.len() != rhs_rank {
        return Err("role vector length must equal operand rank".into());
    }
    let count =
        |v: &[crate::ir::AxisRole], r: crate::ir::AxisRole| v.iter().filter(|&&x| x == r).count();
    if count(&axes.lhs, FreeM) != 1 {
        return Err("lhs must have exactly one FreeM".into());
    }
    if count(&axes.lhs, FreeN) != 0 {
        return Err("lhs must not carry FreeN".into());
    }
    if count(&axes.rhs, FreeN) != 1 {
        return Err("rhs must have exactly one FreeN".into());
    }
    if count(&axes.rhs, FreeM) != 0 {
        return Err("rhs must not carry FreeM".into());
    }
    // single K-group in v1
    if count(&axes.lhs, ContractedK) != 1 || count(&axes.rhs, ContractedK) != 1 {
        return Err("v1: exactly one ContractedK per operand (multi-group deferred)".into());
    }
    // Batch structure — v1 lowers a SINGLE batch coordinate (`blockIdx.z`), so a
    // contraction operand is rank-2 (no Batch) or rank-3 (exactly one Batch); 2+
    // Batch axes (rank >= 4) cannot be emitted (the emitter's `ext(Batch)` has no
    // second batch coordinate). Each operand's Batch count must equal `rank - 2`,
    // and lhs/rhs must agree — a batched matmul needs BOTH operands batched (a
    // rank-3 · rank-2 pair is illegal).
    if !matches!(lhs_rank, 2 | 3) || !matches!(rhs_rank, 2 | 3) {
        return Err("v1: contraction operands must be rank-2 or rank-3".into());
    }
    let lhs_batch = count(&axes.lhs, Batch);
    let rhs_batch = count(&axes.rhs, Batch);
    if lhs_batch != lhs_rank - 2 || rhs_batch != rhs_rank - 2 {
        return Err(
            "v1: rank-3 carries exactly one Batch and rank-2 none (single batch coord)".into(),
        );
    }
    if lhs_batch != rhs_batch {
        return Err("lhs and rhs must share the same batch-axis count".into());
    }
    Ok(())
}

/// The contraction cell's admissibility, as a typed refusal: everything the
/// `Access::Contraction` arm of `build_plan_core` used to assert, plus the two facts a
/// key derived from declared roles can now get wrong against the op (`BatchDisagrees`)
/// and against every emitter (`BatchNotOutermost`). Not a contraction: `Ok`.
///
/// Expected role-vector length is the key's rank, NOT `axes.*.len()` (which would make
/// the length check vacuous): a malformed OpDef whose roles disagree with the keyed
/// rank is refused here before it can panic downstream.
pub(super) fn check_contraction(op: &OpDef, key: &StructureKey) -> Result<(), PlanError> {
    use crate::ir::AxisRole;
    let Access::Contraction {
        ref axes,
        ref epilogue,
        ref accum,
    } = op.access
    else {
        return Ok(());
    };
    let decline = |d| Err(PlanError::UnservableContraction(d));
    if let Some((want, k_step)) = accum_policy_fit(accum) {
        if !want.contains(&key.dtype) || k_step == 0 {
            return decline(ContractionDecline::AccumPolicyMismatch {
                dtype: key.dtype,
                k_step,
            });
        }
    }
    let rank = key.rank as usize;
    if let Err(detail) = validate_contraction_roles(axes, rank, rank) {
        return decline(ContractionDecline::RolesInvalid { detail });
    }
    let Some(c) = key.contraction else {
        return decline(ContractionDecline::KeyHasNoContractionFacts);
    };
    if let Some(step) = accum_integer_step(accum) {
        let int8 = |t| matches!(t, ElementKind::I8 | ElementKind::U8);
        let why = if !int8(key.dtype) || !int8(c.wdt) {
            Some(format!(
                "operands must be I8 or U8, got {:?} and {:?}",
                key.dtype, c.wdt
            ))
        } else if c.acc != ElementKind::I32 || c.out != ElementKind::I32 {
            Some(format!(
                "the key must carry an I32 accumulator and output (CellFacts::accumulator), got {:?} and {:?}",
                c.acc, c.out
            ))
        } else if op.out_dtype != Some(ElementKind::I32) {
            Some(format!(
                "the op's out_dtype must be Some(I32), got {:?}",
                op.out_dtype
            ))
        } else if step == Some(0) {
            Some("k_step must be at least 1".to_string())
        } else if !matches!(epilogue, crate::ir::ScalarExpr::Reduced(0)) {
            Some("the epilogue must be the identity Reduced(0)".to_string())
        } else {
            None
        };
        if let Some(detail) = why {
            return decline(ContractionDecline::IntegerPolicyMismatch { detail });
        }
    }
    if !contraction_epilogue_admissible(epilogue, op.n_inputs) {
        return decline(ContractionDecline::EpilogueInadmissible);
    }
    let op_batched = axes.lhs.contains(&AxisRole::Batch);
    let key_batched = c.batch.is_some();
    if op_batched != key_batched {
        return decline(ContractionDecline::BatchDisagrees {
            op_batched,
            key_batched,
        });
    }
    if key_batched {
        // A broadcast (stride-0) batch axis occupies no storage, so it is exempt.
        for (operand, order) in [(0, c.lhs_order), (1, c.rhs_order)] {
            if !key.operands[operand].bcast.is_set(0) && order.perm()[0] != 0 {
                return decline(ContractionDecline::BatchNotOutermost { operand });
            }
        }
    }
    Ok(())
}

/// For an integer accumulation policy: `Some(None)` for [`AccumSpec::Int32`], `Some(Some(k_step))` for
/// [`AccumSpec::Int32Sat`]; `None` for every other policy.
pub(crate) fn accum_integer_step(accum: &AccumSpec) -> Option<Option<u32>> {
    match *accum {
        AccumSpec::Int32 => Some(None),
        AccumSpec::Int32Sat { k_step } => Some(Some(k_step)),
        _ => None,
    }
}

/// The operand dtypes a reduced-precision accumulation policy admits and its `k_step`; `None` for [`AccumSpec::WideFloat`].
pub(super) fn accum_policy_fit(accum: &AccumSpec) -> Option<(&'static [ElementKind], u32)> {
    match *accum {
        AccumSpec::Tf32 { k_step }
        | AccumSpec::Tf32x3 { k_step }
        | AccumSpec::Bf16x3 { k_step } => Some((&[ElementKind::F32], k_step)),
        AccumSpec::F16Accum { k_step } => Some((&[ElementKind::F16], k_step)),
        AccumSpec::Fp8FastAccum {
            k_step,
            promote_every,
            mantissa_bits,
        } => {
            // a zero promotion interval or a mantissa width outside 1..=23 is reported as `k_step` 0, the decline's one number
            let valid = promote_every >= 1 && (1..=23).contains(&mantissa_bits);
            Some((
                &[ElementKind::Fp8E4M3FN, ElementKind::Fp8E5M2],
                if valid { k_step } else { 0 },
            ))
        }
        _ => None,
    }
}
