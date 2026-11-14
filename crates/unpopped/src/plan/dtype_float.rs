// SPDX-License-Identifier: MIT OR Apache-2.0
//! Section of the plan gate; split from `plan.rs` (ROADMAP U30). Paths unchanged.

use super::*;

/// A `structure_key` dtype position must name a dtype a kernel can COMPUTE IN.
///
/// Seven §6.1 rows are not that, and the vocabulary says so in its own doc:
///
/// - **`Fp8E4M3FNUZ` / `Fp8E5M2FNUZ`** — RESERVED. *"Every use in a
///   `structure_key` dtype position is a **typed decline**."* They are recognized
///   and distinguished from unknown precisely so they can be refused rather than
///   mistaken for their `FN` near-twins, which they are **byte-incompatible**
///   with.
/// - **`F8E8M0` / `F8E6M2`** — MX shared block **scales** (KISS-CLASSIFY
///   §6.1-0013): a sibling operand that scales a block, **never an element value
///   dtype a kernel computes in**. A kernel does not compute *in* a scale.
/// - **`F4E2M1` / `F6E2M3` / `F6E3M2`** — OCP MX **element** formats
///   (§6.1-0014): storage and key identity only until a codec lands (no
///   `oracle::elem_size`, no emitter carrier). Added with the rows themselves, so
///   the oracle never meets a dtype it cannot size.
///
/// # Why this is a gate rather than a comment
///
/// The emitters already refuse all four, so the emission path was safe and
/// `the_reserved_dtypes_never_lower` proved it. **But `unpopped::oracle` is a
/// second consumer of a plan and it is `pub`** — `evaluate` takes a
/// `KernelPlan` and never sees a backend. Measured 2026-09-02: the gate ADMITTED
/// all four and `oracle::elem_size` then **panicked** on each, which is a
/// reachable panic on valid input in a published crate.
///
/// ⚠️ **The op-level coverage was complete and the DTYPE axis was not, and
/// nothing was measuring that axis.** `Coverage` in `oracle.rs` enumerates
/// `Access` exhaustively — schedules — so it went green while four dtypes had no
/// evaluation at all. **An exhaustive check over the wrong axis reads exactly
/// like an exhaustive check.**
pub(super) fn check_dtype_is_a_compute_dtype(dtype: ElementKind) -> Result<(), PlanError> {
    let why = match dtype {
        ElementKind::Fp8E4M3FNUZ | ElementKind::Fp8E5M2FNUZ => {
            "a RESERVED dtype with no computation semantics at this schema              version; it is recognized so it can be declined rather than confused              with its byte-incompatible `fn` twin"
        }
        ElementKind::F8E8M0 | ElementKind::F8E6M2 => {
            "an MX shared block SCALE (KISS-CLASSIFY 6.1-0013) — a sibling              operand that scales a block, never an element value dtype a kernel              computes in"
        }
        ElementKind::F4E2M1 | ElementKind::F6E2M3 | ElementKind::F6E3M2 => {
            "an OCP MX ELEMENT format (KISS-CLASSIFY 6.1-0014) — sub-byte storage              and key identity only; no oracle sizing or emitter codec exists yet"
        }
        _ => return Ok(()),
    };
    Err(PlanError::InadmissibleOpInCell {
        detail: format!("{dtype:?} is not a compute dtype: {why}"),
    })
}

pub(crate) fn check_no_half_nextafter(op: &OpDef, dtype: ElementKind) -> Result<(), PlanError> {
    use crate::ir::BinaryOp;
    if !matches!(dtype, ElementKind::F16 | ElementKind::Bf16) {
        return Ok(());
    }
    fn walk(e: &ScalarExpr) -> bool {
        match e {
            ScalarExpr::Input(_)
            | ScalarExpr::Const(_)
            | ScalarExpr::Param(_)
            | ScalarExpr::Reduced(_)
            | ScalarExpr::Coord(_)
            | ScalarExpr::CoordInt(..)
            | ScalarExpr::ConstInt(..)
            | ScalarExpr::Local(_)
            | ScalarExpr::Exec(_) => false,
            ScalarExpr::Add(a, b)
            | ScalarExpr::Sub(a, b)
            | ScalarExpr::Mul(a, b)
            | ScalarExpr::Div(a, b) => walk(a) || walk(b),
            ScalarExpr::Unary(_, a) => walk(a),
            ScalarExpr::Cast(a, _)
            | ScalarExpr::Bitcast(a, _)
            | ScalarExpr::Qualified(a, _)
            | ScalarExpr::Extract(a, _)
            | ScalarExpr::BitcastPacked(a, ..) => walk(a),
            ScalarExpr::Pack(xs) => xs.iter().any(walk),
            ScalarExpr::Binary(bop, a, b) => {
                matches!(bop, BinaryOp::Nextafter) || walk(a) || walk(b)
            }
            ScalarExpr::Select(c, a, b) | ScalarExpr::Fma(c, a, b) => walk(c) || walk(a) || walk(b),
        }
    }
    let mut exprs: Vec<&ScalarExpr> = vec![&op.body];
    match &op.access {
        Access::RowReduce { stages, epilogue } => {
            exprs.extend(stages.iter().map(|s| &s.pre));
            exprs.push(epilogue);
        }
        Access::Contraction { epilogue, .. } => exprs.push(epilogue),
        // The reduction post-expr (0e) lowers through the accumulator-width
        // spellers too, so the honest-miss walk must cover it (body is already in).
        Access::Reduction { post, .. } => exprs.push(post),
        // Increment 6 SCAN: the `pre` (per-element pre-map) and `post` (per-element
        // epilogue) both lower through the accumulator-width spellers (body == post
        // is already in), so a half `Nextafter` hidden in `pre` must miss honestly.
        Access::Scan { pre, post, .. } => {
            exprs.push(pre);
            exprs.push(post);
        }
        // Increment 7 WINDOW: `pre` (per-tap pre-map) and `post` (per-output
        // epilogue) both lower through the accumulator-width spellers (body ==
        // post is already in), so a half `Nextafter` hidden in `pre` must miss
        // honestly.
        Access::Window { pre, post, .. } => {
            exprs.push(pre);
            exprs.push(post);
        }
        // Increment 8 SORT_PERM: `body` is pinned `Input(0)` (already in `exprs`)
        // and RowSort has no pre/post, so there is nothing extra to walk — the arm
        // is forced only to keep the honest-miss walk exhaustive/total.
        Access::RowSort { .. } => {}
        // Increment 11 IM2COL: `body` is pinned `Input(0)` (already in `exprs`) and
        // im2col has no pre/post — a pure raw-bit gather has no arithmetic to hide a
        // half `Nextafter` in, so there is nothing extra to walk; the arm is forced
        // only to keep the honest-miss walk exhaustive/total.
        Access::Im2Col { .. } => {}
        // Review-caught gate asymmetry (increment 1): a multi-output op's EXTRA
        // output bodies must be walked too — else a half `Nextafter` hidden in an
        // extra body bypasses this honest-miss gate. Non-elementwise multi-output
        // is already rejected by `assert_valid_multi_output` (runs first), so this
        // arm is the ONLY place `extra_out_bodies` is legal; empty for every
        // single-output op (byte-identical).
        Access::Elementwise => exprs.extend(op.extra_out_bodies.iter()),
    }
    for e in exprs {
        if walk(e) {
            return Err(PlanError::InadmissibleOpAtDtype {
                dtype,
                detail: format!(
                    "Nextafter has no half-precision lowering (IR contract: f32/f64 only; \
                     the promote-to-f32 path silently no-ops after the demote) — op '{}' \
                     at {dtype:?} must miss honestly",
                    op.name
                ),
            });
        }
    }
    Ok(())
}

/// The integer compute dtypes of increment 0c: `I32`/`I64` (already lowering
/// pre-0c) plus the newly-promoted `S8` (FKC `I8`) and `U8`.
pub fn is_int_dtype(dt: ElementKind) -> bool {
    matches!(
        dt,
        ElementKind::I32
            | ElementKind::I64
            | ElementKind::I8
            | ElementKind::U8
            | ElementKind::I16
            | ElementKind::U16
            | ElementKind::U32
            | ElementKind::U64
            | ElementKind::I4
            | ElementKind::U4
            | ElementKind::B1
    )
}
