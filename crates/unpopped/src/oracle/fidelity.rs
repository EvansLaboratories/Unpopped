// SPDX-License-Identifier: MIT OR Apache-2.0
//! [`compare`]: the bit-exact-versus-tolerance dichotomy and the fidelity a plan is entitled to.
//!
//! A section of the oracle (ROADMAP U30): `oracle.rs` glob-imports it, so every path it had is unchanged.

use super::*;

// ===========================================================================
// compare — the bit-exact-vs-tolerance dichotomy.
// ===========================================================================

/// How two output buffers are compared: bit-exact (raw storage memcmp) or f64-
/// accurate within a rel+abs tolerance band.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Fidelity {
    /// Raw-byte equality (identity bodies, Im2Col, permutation/movement, Select
    /// arm moves, integer arithmetic/bitwise/shift — exact-wrapping).
    BitExact,
    /// Float arithmetic within `abs + rel·max(|a|,|b|)`; both-NaN payload-agnostic
    /// equal; ±0 equal.
    Tolerant {
        /// Relative tolerance.
        rel: f64,
        /// Absolute tolerance.
        abs: f64,
    },
}

/// What comparison band a cell is **entitled to** — derived from the plan and
/// its operands, not chosen by hand at the call site.
///
/// # The gap this closes
///
/// Every caller of [`compare`] picked a [`Fidelity`] by hand. Nothing said what
/// a given cell was owed, so the choice was a judgement re-made per test, with
/// no way to tell a tolerance that reflects the kernel's real error from one
/// that was widened until the test passed. Those look identical in a diff.
///
/// # Derived from the SAME number the contract declares
///
/// The approximate-op term comes from [`crate::contract::ulp_bound`] — the
/// function that feeds the emitted contract's `max_ulp`. Two independently
/// chosen accuracy figures would drift, silently and in both directions: a
/// validator looser than the contract passes kernels the contract promises are
/// tighter, and a tighter one fails kernels that honour it.
///
/// # What the band is made of
///
/// `rel = (steps + 2·ulp_bound) · u`, where `u` is the compute dtype's unit
/// roundoff (`2⁻²⁴` for `f32`, `2⁻⁵³` for `f64`).
///
/// * **`steps`** — the roundings the DEVICE performs that the oracle does not.
///   The oracle accumulates in `f64` on purpose ("a tolerance referee, not a
///   bit-for-bit device mirror"), so each arithmetic node in the body, plus the
///   reduction length where the cell reduces, is one rounding the reference
///   skipped. Each contributes at most `u` relative error; summing them is the
///   standard forward bound `n·u/(1−n·u) ≈ n·u`.
/// * **`2·ulp_bound`** — the vendor-approximate ops' declared error. One ULP is
///   `2u` relative at the top of a binade, so a `k`-ULP op contributes `2k·u`.
///
/// # `abs` is zero, and that is a stated limit rather than an omission
///
/// A relative band collapses at zero, so a body whose result **cancels** to near
/// zero from larger intermediates needs an absolute floor — and the size of that
/// floor depends on the intermediate magnitudes, which a plan does not know and
/// cannot know without the input values. Returning a fabricated `abs` would be a
/// number with no derivation behind it, which is worse than none: it would look
/// like a bound. Callers whose cells cancel must widen `abs` themselves and say
/// why.
///
/// # Returns `None` when no bound is derivable
///
/// A body containing floored `Rem` has no finite result-ULP bound — its
/// quotient-boundary flip is a discontinuity, not an error term, and
/// [`crate::contract::precision_of`] correctly declares `approximate` with **no**
/// `max_ulp` rather than an understated number. `None` says the same thing here:
/// the caller must decide, and cannot be handed a bound that does not exist.
#[must_use]
pub fn required_fidelity(plan: &KernelPlan<'_>, operands: &[OperandDesc]) -> Option<Fidelity> {
    required_fidelity_with_exactness(plan, operands, None)
}

/// [`required_fidelity`], with what a device measured: a `Flushes` record for the plan's body on
/// `measured`'s stamp yields `None` (no claim), as an unknown target does. A stamp for another
/// target than the plan's is ignored. See [`crate::contract::MeasuredOn`].
#[must_use]
pub fn required_fidelity_with_exactness(
    plan: &KernelPlan<'_>,
    operands: &[OperandDesc],
    measured: Option<crate::exactness::Measured<'_>>,
) -> Option<Fidelity> {
    // Integer, bool and sub-byte cells are EXACT on both sides: the emitter's
    // arithmetic wraps at the C width and the oracle models that wrapping in
    // `i128`, so there is no rounding for a tolerance to absorb. Bit-exact is
    // not a tight choice here, it is the only correct one — a tolerant compare
    // would accept a genuinely wrong integer.
    if is_int(plan.dtype) && is_int(plan.out_dtype) {
        return Some(Fidelity::BitExact);
    }

    // KNOWN LIMIT, stated at the use site because it is invisible from the
    // signature: `ulp_bound` sums a **CUDA** per-op ULP table, and neither it nor
    // this function takes a target. So the band computed here is CUDA's accuracy
    // for every backend. On a target whose guarantees are looser — Vulkan's `exp`
    // is 3 ULP against CUDA `expf`'s 2 — this band is too tight to be met, and a
    // conforming kernel fails a comparison it should pass. See `ulp_bound`'s
    // "KNOWN LIMIT" section; closing it needs a per-target accuracy seam.
    let ulp = crate::contract::ulp_bound_measured(
        plan.body,
        &crate::contract::AccuracyKey::for_target(plan.key.target),
        crate::contract::MeasuredOn::for_target(measured, plan.key.target, plan.dtype).as_ref(),
    );
    if !ulp.is_finite() {
        return None;
    }

    // A body that only MOVES values rounds nothing: an identity passthrough, a
    // Select picking between arms, a gather. The `Fidelity` doc already names
    // this class as bit-exact; this derives it instead of trusting the caller to
    // recognise it.
    let steps = arith_steps(plan.body) + reduction_len(plan, operands);
    if steps == 0 && ulp == 0.0 {
        return Some(Fidelity::BitExact);
    }

    #[allow(
        clippy::cast_precision_loss,
        reason = "`steps` is a small step count, exact in f64"
    )]
    let rel = (steps as f64 + 2.0 * ulp) * unit_roundoff(plan.dtype);
    Some(Fidelity::Tolerant { rel, abs: 0.0 })
}

/// The compute dtype's unit roundoff — the largest relative error a single
/// correctly-rounded operation can introduce.
///
/// Keyed to the precision the DEVICE computes at, which for every narrow float
/// is `f32`: the emitters promote an `f16`/`bf16`/FP8 load to `float` and
/// compute there (`cfamily`'s narrow-float seam). Using the storage precision
/// would produce a band far looser than the kernel's real error.
pub(super) fn unit_roundoff(dt: ElementKind) -> f64 {
    match dt {
        ElementKind::F64 | ElementKind::Complex128 => f64::EPSILON / 2.0,
        _ => f64::from(f32::EPSILON) / 2.0,
    }
}

/// Count the arithmetic nodes whose rounding the device performs and the
/// `f64` oracle does not.
///
/// Leaves and `Select` contribute nothing: a leaf is a load and a `Select`
/// picks an arm without arithmetic (the same call `contract::ulp_bound` makes,
/// for the same reason).
pub(super) fn arith_steps(e: &ScalarExpr) -> u32 {
    match e {
        ScalarExpr::Input(_)
        | ScalarExpr::Const(_)
        | ScalarExpr::Param(_)
        | ScalarExpr::Reduced(_)
        | ScalarExpr::Coord(_)
        | ScalarExpr::CoordInt(..)
        | ScalarExpr::ConstInt(..)
        | ScalarExpr::Local(_)
        | ScalarExpr::Exec(_) => 0,
        ScalarExpr::Add(a, b)
        | ScalarExpr::Sub(a, b)
        | ScalarExpr::Mul(a, b)
        | ScalarExpr::Div(a, b) => 1 + arith_steps(a) + arith_steps(b),
        ScalarExpr::Binary(_, a, b) => 1 + arith_steps(a) + arith_steps(b),
        ScalarExpr::Unary(_, x) => 1 + arith_steps(x),
        // A Select rounds nothing itself; its subexpressions still count.
        ScalarExpr::Select(c, a, b) => arith_steps(c) + arith_steps(a) + arith_steps(b),
        // A fused multiply-add rounds exactly once on the device.
        ScalarExpr::Fma(a, b, c) => 1 + arith_steps(a) + arith_steps(b) + arith_steps(c),
        // A conversion is computed exactly by the oracle (the same single rounding the
        // device performs), so it adds no gap of its own.
        ScalarExpr::Cast(x, _)
        | ScalarExpr::Bitcast(x, _)
        | ScalarExpr::Extract(x, _)
        | ScalarExpr::BitcastPacked(x, ..) => arith_steps(x),
        // Every lane is computed on the device, whichever one is extracted.
        ScalarExpr::Pack(xs) => xs.iter().map(arith_steps).sum(),
        // The qualified operation is computed exactly in its stated mode (the
        // oracle's directed rounding is built from the exact residual), so the
        // wrapper adds no gap beyond the wrapped operation's own.
        ScalarExpr::Qualified(x, _) => arith_steps(x),
    }
}

/// How many accumulation steps a reducing cell performs, or `0` if it does not
/// reduce.
///
/// A K-long sequential sum accumulates one rounding per element, and the oracle
/// sums the same series in `f64`, so the whole length is error the band must
/// cover. This is the term that makes a large reduction's band legitimately wide
/// — and hiding it would be exactly the hand-widening this function replaces.
pub(super) fn reduction_len(plan: &KernelPlan<'_>, operands: &[OperandDesc]) -> u32 {
    let reduced = matches!(
        plan.schedule,
        Schedule::Reduction { .. } | Schedule::RowReduce { .. } | Schedule::Contraction
    );
    if !reduced {
        return 0;
    }
    // The reduced extent is the largest input extent — an upper bound that does
    // not require knowing which axis the schedule folds. Over-stating widens the
    // band, which is safe; under-stating would reject a correct kernel.
    let n = operands
        .iter()
        .flat_map(|o| o.shape[..usize::from(o.rank)].iter().copied())
        .max()
        .unwrap_or(1);
    u32::try_from(n.max(1)).unwrap_or(u32::MAX)
}

/// [`compare`] under the band the cell is **entitled to**: [`required_fidelity`] derives it from `plan`
/// and `operands`, so the caller does not choose one (ROADMAP C5). A band widened until a test passed
/// cannot arise here, because there is no band argument.
///
/// # Errors
/// The first mismatch, as [`compare`]; or `"no entitlement"` when [`required_fidelity`] declines (a
/// target with no accuracy table, an unrateable body): there is then nothing to certify against, and
/// the caller must pick a band and call [`compare`] knowingly.
pub fn compare_for_plan(
    expected: &TypedBuffer,
    actual: &TypedBuffer,
    plan: &KernelPlan<'_>,
    operands: &[OperandDesc],
) -> Result<(), String> {
    compare_for_plan_with_exactness(expected, actual, plan, operands, None)
}

/// [`compare_for_plan`] with what a device measured, as [`required_fidelity_with_exactness`].
///
/// # Errors
/// As [`compare_for_plan`].
pub fn compare_for_plan_with_exactness(
    expected: &TypedBuffer,
    actual: &TypedBuffer,
    plan: &KernelPlan<'_>,
    operands: &[OperandDesc],
    measured: Option<crate::exactness::Measured<'_>>,
) -> Result<(), String> {
    let entitled =
        required_fidelity_with_exactness(plan, operands, measured).ok_or_else(no_entitlement)?;
    compare(expected, actual, entitled)
}

/// [`compare`] at `band`, but only if `band` is **no looser** than the cell's entitlement: a band tighter
/// than [`required_fidelity`] is harmless and is the one applied; a looser one is refused, because it is
/// the defect (a tolerance wider than the kernel's real error). For a caller that has a tighter band of its
/// own (a backend that is more accurate than the table says).
///
/// # Errors
/// `"... looser than the entitled ..."` for a band looser than the entitlement; `"no entitlement"` as
/// [`compare_for_plan`]; otherwise the first mismatch, as [`compare`].
pub fn compare_for_plan_within(
    expected: &TypedBuffer,
    actual: &TypedBuffer,
    plan: &KernelPlan<'_>,
    operands: &[OperandDesc],
    band: Fidelity,
) -> Result<(), String> {
    let entitled = required_fidelity(plan, operands).ok_or_else(no_entitlement)?;
    if looser_than(band, entitled) {
        return Err(format!(
            "the band {band:?} is looser than the entitled {entitled:?}: a tolerance wider than the cell's              derived error hides a wrong answer; use a band no looser than the entitlement"
        ));
    }
    compare(expected, actual, band)
}

pub(super) fn no_entitlement() -> String {
    "no entitlement: required_fidelity declines for this cell (no accuracy table for the target, or an      unrateable body), so there is nothing to certify a band against; choose one and call `compare`"
        .to_string()
}

/// Whether `band` accepts more than `entitled` does. `BitExact` is never looser; a tolerant band is looser
/// than `BitExact`, and than another tolerant band when either of its two tolerances is larger.
pub(super) fn looser_than(band: Fidelity, entitled: Fidelity) -> bool {
    match (band, entitled) {
        (Fidelity::BitExact, _) => false,
        (Fidelity::Tolerant { .. }, Fidelity::BitExact) => true,
        (Fidelity::Tolerant { rel, abs }, Fidelity::Tolerant { rel: er, abs: ea }) => {
            rel > er || abs > ea
        }
    }
}

/// Compare `expected` against `actual` under `fidelity`. `Ok(())` if they match;
/// `Err(msg)` naming the first mismatch otherwise.
///
/// # Errors
/// Returns a description of the first mismatching element (or a shape/length
/// mismatch).
pub fn compare(
    expected: &TypedBuffer,
    actual: &TypedBuffer,
    fidelity: Fidelity,
) -> Result<(), String> {
    if expected.shape != actual.shape {
        return Err(format!(
            "shape mismatch: {:?} vs {:?}",
            expected.shape, actual.shape
        ));
    }
    match fidelity {
        Fidelity::BitExact => {
            if expected.bytes != actual.bytes {
                let sz = elem_size(expected.dtype);
                for i in 0..(expected.bytes.len() / sz) {
                    let a = expected.bits_at(i);
                    let b = actual.bits_at(i);
                    if a != b {
                        return Err(format!(
                            "bit-exact mismatch at element {i}: 0x{a:x} vs 0x{b:x}"
                        ));
                    }
                }
                return Err("byte-length mismatch".to_string());
            }
            Ok(())
        }
        // An INTEGER dtype is compared exactly, in `i128`, whatever tolerance was
        // requested. Not a special case for tidiness: routing integers through
        // the `f64` projection means `Tolerant { rel: 0.0, abs: 0.0 }` reports
        // `2^53` and `2^53 + 1` as EQUAL, because both round to the same `f64`.
        // A comparator that cannot distinguish two distinct values of the dtype
        // it was handed is not a comparator, and it fails in the silent
        // direction — agreeing rather than complaining.
        //
        // The tolerance is still honoured, in integer units: integer arithmetic
        // has no rounding, so `abs` here means a genuine allowed distance rather
        // than an accumulated-error band, and `rel` scales against the larger
        // magnitude exactly as it does for floats.
        // Complex compares COMPONENT-WISE. A single magnitude tolerance would
        // pass `(0, 5)` against `(5, 0)` — equal magnitude, orthogonal values —
        // and complex has no ordering to fall back on, so magnitude is the only
        // scalar available and it is not enough.
        Fidelity::Tolerant { rel, abs }
            if is_complex(expected.dtype) && is_complex(actual.dtype) =>
        {
            let ea = expected.to_complex_vec();
            let ac = actual.to_complex_vec();
            if ea.len() != ac.len() {
                return Err(format!("length mismatch: {} vs {}", ea.len(), ac.len()));
            }
            for (i, (&(er, ei), &(ar, ai))) in ea.iter().zip(ac.iter()).enumerate() {
                for (part, e, a) in [("re", er, ar), ("im", ei, ai)] {
                    if e == a || (e.is_nan() && a.is_nan()) {
                        continue;
                    }
                    if e.is_infinite() || a.is_infinite() {
                        return Err(format!(
                            "infinity mismatch at element {i} ({part}): expected {e}, got {a}"
                        ));
                    }
                    let diff = (e - a).abs();
                    if diff <= abs + rel * e.abs().max(a.abs()) {
                        continue;
                    }
                    return Err(format!(
                        "tolerance mismatch at element {i} ({part}):                          expected {e}, got {a} (|Δ|={diff})"
                    ));
                }
            }
            Ok(())
        }
        Fidelity::Tolerant { rel, abs } if is_int(expected.dtype) && is_int(actual.dtype) => {
            let ea = expected.to_i128_vec();
            let ac = actual.to_i128_vec();
            if ea.len() != ac.len() {
                return Err(format!("length mismatch: {} vs {}", ea.len(), ac.len()));
            }
            for (i, (&e, &a)) in ea.iter().zip(ac.iter()).enumerate() {
                if e == a {
                    continue;
                }
                let diff = (e - a).unsigned_abs();
                let band = abs + rel * (e.unsigned_abs().max(a.unsigned_abs()) as f64);
                // `band` is computed in f64 only as a WIDTH, never as a value —
                // a lossy width still rejects a mismatch far outside it, whereas
                // a lossy value silently equates distinct ones.
                if band.is_finite() && (diff as f64) <= band {
                    continue;
                }
                return Err(format!(
                    "tolerance mismatch at element {i}: expected {e}, got {a} (|Δ|={diff})"
                ));
            }
            Ok(())
        }
        Fidelity::Tolerant { rel, abs } => {
            let ea = expected.to_f64_vec();
            let ac = actual.to_f64_vec();
            if ea.len() != ac.len() {
                return Err(format!("length mismatch: {} vs {}", ea.len(), ac.len()));
            }
            for (i, (&e, &a)) in ea.iter().zip(ac.iter()).enumerate() {
                if e.is_nan() && a.is_nan() {
                    continue;
                }
                if e == a {
                    continue; // covers ±0 and equal infinities
                }
                // A non-equal infinite expected/actual makes the relative band
                // `rel * max(|e|,|a|)` infinite, which would swallow ANY mismatch
                // (+inf vs -inf, inf vs finite, finite vs inf). An infinity only
                // matches an exactly-equal one (handled just above) — otherwise
                // reject, so an output overflowing to inf can't silently pass.
                if e.is_infinite() || a.is_infinite() {
                    return Err(format!(
                        "infinity mismatch at element {i}: expected {e}, got {a}"
                    ));
                }
                let diff = (e - a).abs();
                if diff <= abs + rel * e.abs().max(a.abs()) {
                    continue;
                }
                return Err(format!(
                    "tolerance mismatch at element {i}: expected {e}, got {a} (|Δ|={diff})"
                ));
            }
            Ok(())
        }
    }
}
