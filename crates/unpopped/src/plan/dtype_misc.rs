// SPDX-License-Identifier: MIT OR Apache-2.0
//! Section of the plan gate; split from `plan.rs` (ROADMAP U30). Paths unchanged.

use super::*;

/// Panicking form of [`check_int_op_admissibility`] — the AOT gate.
///
/// The admissibility rules live once, in the checking form. This is the shape
/// Which ops may appear in a body at a **complex** dtype.
///
/// Complex is a third compute domain, and most of the IR's vocabulary is simply
/// not defined on it:
///
/// * **Ordered ops** — `Max`/`Min`/`Cmp*` — have no meaning at all. The complex
///   numbers are not an ordered field; there is no total order compatible with
///   the arithmetic. This is not "unimplemented", it is undefined, and a future
///   version should not quietly add it.
/// * **Transcendentals** (`Unary`) and `Div` **are** mathematically defined for
///   complex, and are rejected here as *unimplemented* — the oracle carries
///   complex rules for `Add`/`Sub`/`Mul` only. Different reason, same verdict,
///   and the message says which so the next reader does not have to guess.
/// * **Int-only ops** never reach here: they are already refused by
///   [`check_int_op_admissibility`], because a complex dtype is not an int dtype.
///
/// # Why this gate has to exist rather than letting the oracle refuse
///
/// Without it a `Max` at a complex dtype builds a plan, reaches the evaluator,
/// and **panics** — `Val::f64()` on a complex is deliberately fatal, since
/// silently taking the real part would drop the imaginary half and return a
/// wrong answer that looks right. A panic is the correct behaviour *there* and
/// the wrong behaviour *here*: this is the trust boundary, and an unserveable
/// request must come back as a typed decline rather than unwind into the caller.
/// Measured before writing this: `Max` and `CmpLt` at `Complex128` both panicked.
pub(crate) fn check_complex_op_admissibility(
    op: &OpDef,
    dtype: ElementKind,
) -> Result<(), PlanError> {
    if !is_complex_dtype(dtype) {
        return Ok(());
    }
    fn walk(e: &ScalarExpr, op_name: &str, dtype: ElementKind) -> Result<(), PlanError> {
        let reject = |what: &str, why: &str| {
            Err(PlanError::InadmissibleOpAtDtype {
                dtype,
                detail: format!("op '{op_name}': {what} at {dtype:?} — {why}"),
            })
        };
        match e {
            ScalarExpr::Input(_) | ScalarExpr::Const(_) => Ok(()),
            ScalarExpr::CoordInt(..) | ScalarExpr::ConstInt(..) => reject(
                "an integer-typed leaf",
                "integer index expressions are float-compute only in v1",
            ),
            // All four field operations. `Div` joined them once both sides
            // carried a rule: the oracle evaluates Smith's algorithm and
            // `cfamily::complex_helpers` emits the same recurrence. It was
            // refused here purely because it was unimplemented, which is a
            // reason that expires rather than a reason that holds.
            ScalarExpr::Add(a, b)
            | ScalarExpr::Sub(a, b)
            | ScalarExpr::Mul(a, b)
            | ScalarExpr::Div(a, b) => {
                walk(a, op_name, dtype)?;
                walk(b, op_name, dtype)
            }
            // `Max`/`Min` belong with `Cmp*`, not with the unimplemented rest:
            // all four are *ordering* ops, and ordering is what complex does not
            // have. Lumping them together would tell an author their `Max` might
            // arrive later, which is the one thing that must not happen here.
            ScalarExpr::Binary(bop, _, _)
                if bop.is_cmp()
                    || matches!(bop, crate::ir::BinaryOp::Max | crate::ir::BinaryOp::Min) =>
            {
                reject(
                    &format!("{bop:?}"),
                    "the complex numbers are NOT ORDERED — no total order is compatible \
                     with the arithmetic, so this is undefined rather than unimplemented, \
                     and no future version should add it",
                )
            }
            ScalarExpr::Binary(bop, _, _) => reject(
                &format!("{bop:?}"),
                "no complex lowering — defined for complex but unimplemented",
            ),
            ScalarExpr::Unary(uop, _) => reject(
                &format!("{uop:?}"),
                "complex transcendentals are defined but not implemented",
            ),
            ScalarExpr::Select(_, _, _) => reject(
                "Select",
                "its predicate would have to be an ordered comparison, which complex \
                 does not have",
            ),
            ScalarExpr::Fma(_, _, _) => {
                reject("Fma", "no complex fused multiply-add lowering exists")
            }
            ScalarExpr::Cast(..) | ScalarExpr::Bitcast(..) => reject(
                "Cast/Bitcast",
                "typed expressions are f32/f32s/f64-compute only in v1",
            ),
            ScalarExpr::Pack(_)
            | ScalarExpr::Extract(..)
            | ScalarExpr::BitcastPacked(..)
            | ScalarExpr::Local(_)
            | ScalarExpr::Exec(_) => reject("Pack/Extract", "packed values have no lowering yet"),
            ScalarExpr::Qualified(..) => reject(
                "Qualified",
                "rounding/ftz/approx qualifiers are f32-compute only in v1",
            ),
            ScalarExpr::Param(_) => reject("Param", "scalar params are f32-only"),
            ScalarExpr::Reduced(_) | ScalarExpr::Coord(_) => reject(
                "Reduced/Coord",
                "no complex reduction or coordinate path exists",
            ),
        }
    }
    walk(&op.body, &op.name, dtype)
}

/// Which ops may appear in a body at the **`bool`** dtype.
///
/// KISS-CLASSIFY §6.1 defines `bool` as a 1-byte truth value where `0` is false,
/// any non-zero byte is true, and **ops normalize to 0/1**. §6.2-0001 makes
/// `bool` its own numeric *kind*, not a spelling of `u8` — same storage width,
/// different semantics.
///
/// So the admissible surface is the **logical** ops, whose results are already
/// normalized (`(a != 0 && b != 0) ? 1 : 0`). Arithmetic is refused: `true +
/// true` is `2`, which is not a value of the dtype, and silently normalizing it
/// would make `+` mean `or` — a coincidence at 1 that stops being one the moment
/// anyone adds three.
///
/// # This gate replaces an inversion
///
/// Measured before writing it: `Add` at `Bool` **built a plan**, while
/// `LogicalAnd` was **refused** with the message "LogicalAnd is int-only
/// (I32/I64/S8/U8) — float dtype Bool must miss honestly". Exactly backwards on
/// both counts, and the message called `bool` a float because the int gate
/// treats every non-int dtype as one. The bespoke logical surface was pinned to
/// `U8` — the representation — while `Bool`, the dtype that surface exists for,
/// fell through to the wrong branch.
pub(crate) fn check_bool_op_admissibility(op: &OpDef, dtype: ElementKind) -> Result<(), PlanError> {
    if dtype != ElementKind::Bool {
        return Ok(());
    }
    fn walk(e: &ScalarExpr, op_name: &str) -> Result<(), PlanError> {
        let reject = |what: &str, why: &str| {
            Err(PlanError::InadmissibleOpAtDtype {
                dtype: ElementKind::Bool,
                detail: format!("op '{op_name}': {what} at Bool — {why}"),
            })
        };
        match e {
            ScalarExpr::Input(_) | ScalarExpr::Const(_) => Ok(()),
            ScalarExpr::Binary(bop, a, b) if bop.is_logical() => {
                walk(a, op_name)?;
                walk(b, op_name)
            }
            ScalarExpr::Add(_, _)
            | ScalarExpr::Sub(_, _)
            | ScalarExpr::Mul(_, _)
            | ScalarExpr::Div(_, _) => reject(
                "arithmetic",
                "`bool` is a truth value and its ops normalize to 0/1 (§6.1);                  `true + true` is 2, which is not a value of the dtype. Use the                  logical ops, whose results are normalized by construction",
            ),
            ScalarExpr::Binary(bop, _, _) => reject(
                &format!("{bop:?}"),
                "only the logical ops (LogicalAnd/LogicalOr/LogicalXor) are the                  bespoke `bool` surface",
            ),
            ScalarExpr::Unary(uop, _) => {
                reject(&format!("{uop:?}"), "no unary math on a truth value")
            }
            ScalarExpr::Select(_, _, _) => reject("Select", "no bool select lowering"),
            ScalarExpr::Fma(_, _, _) => reject("Fma", "no fused multiply-add on a truth value"),
            ScalarExpr::Cast(..) | ScalarExpr::Bitcast(..) => reject(
                "Cast/Bitcast",
                "typed expressions are f32/f32s/f64-compute only in v1",
            ),
            ScalarExpr::Pack(_)
            | ScalarExpr::Extract(..)
            | ScalarExpr::BitcastPacked(..)
            | ScalarExpr::Local(_)
            | ScalarExpr::Exec(_) => reject("Pack/Extract", "packed values have no lowering yet"),
            ScalarExpr::Qualified(..) => reject(
                "Qualified",
                "rounding/ftz/approx qualifiers are f32-compute only in v1",
            ),
            ScalarExpr::Param(_) => reject("Param", "scalar params are f32-only"),
            ScalarExpr::CoordInt(..) | ScalarExpr::ConstInt(..) => reject(
                "CoordInt/ConstInt",
                "integer index expressions are f32/f32s/f64-compute only in v1",
            ),
            ScalarExpr::Reduced(_) | ScalarExpr::Coord(_) => {
                reject("Reduced/Coord", "no bool reduction or coordinate path")
            }
        }
    }
    walk(&op.body, &op.name)
}

/// `true` for the complex dtypes — the third compute domain alongside int and
/// float.
pub fn is_complex_dtype(dt: ElementKind) -> bool {
    matches!(dt, ElementKind::Complex64 | ElementKind::Complex128)
}

/// Increment-0c op × dtype admissibility gate — the plan-level enforcement of
/// the table in [`crate::ir::BinaryOp`]'s docs. Runs at the TOP of
/// [`build_plan`] and walks the body of EVERY Access arm (the 0a lesson: the
/// emitter backstops alone were bypassed by the reduction-class lowering
/// paths), so no lowering path — present or future backend — can bypass it.
///
/// Two directions, both validate-reject (honest miss, never silent):
///
/// 1. **Int-only ops** (`BitAnd`/`BitOr`/`BitXor`/`Shl`/`Shr`/`Logical*`) may
///    appear ONLY in an [`Access::Elementwise`] body at an int dtype (the
///    reduction/RowReduce/contraction paths lower through the FLOAT
///    accumulator spellers, which have no int arms), and the logical ops
///    narrow further to `U8` — the bespoke Bool surface (`uint8_t` only).
/// 2. **At an int dtype**, only the audited op set lowers: infix
///    `Add`/`Sub`/`Mul` (wrapping) and the int-only ops. Everything else —
///    every `UnaryOp`, infix `Div` (no bespoke int div; `/0` is device-UB),
///    the float binary fns, and the `Cmp*` predicates (bespoke cmp is
///    `_fp`-only) — rejects. `Const` rejects too: it is spelled as an f64 C
///    literal, so an int body would silently run double math (and f64 cannot
///    even represent all i64); an int-literal speller is a follow-up. `Param`
///    rejects at int for the same f32-only reason the emitter asserts.
/// 3. **8-bit composition pin (v1):** at `U8`/`S8`, EVERY operand of an
///    int-only op must be a leaf [`ScalarExpr::Input`]. Why: `Add`/`Sub`/
///    `Mul`/`BitAnd`/`BitOr`/`BitXor` compositions are congruent under
///    deferred truncation (the wrapping ring ops and the bit-local ops
///    commute with the final 8-bit store truncate), but `Shr`, shift
///    AMOUNTS, and the logical `!= 0` tests OBSERVE the un-truncated
///    promoted-`int` value — and the DAG emitter truncates a composed
///    interior only when sharing hoists it to an 8-bit tmp, so one body
///    could compute two different results depending on DAG sharing
///    (`(in0+in1)>>in2` at u8 with `(200,100,1)`: inlined `300>>1 = 150`,
///    hoisted `44>>1 = 22`). Rather than a per-position observer analysis,
///    v1 pins ALL int-op operands at 8-bit to leaves — the bespoke surface
///    has no 8-bit bitwise at all, so zero parity is lost; the dtype-aware
///    truncating speller is the follow-up that lifts this. At `I32`/`I64`
///    compositions stay legal: integer promotion never widens past the
///    compute width there, so no un-truncated wider value exists to observe.
/// 4. **int8 any/all/count reduction-predicate lift (this increment):** rule 2
///    rejects `Cmp*`/`Const` at every int dtype UNCONDITIONALLY — but
///    `any`/`all`/`count` are exactly a `Sum`/`Max`/`Min` fold over a fused
///    `Cmp*` predicate (`count = Sum(in != 0)`, `I64` out; `any`/`all` = a
///    `Max`/`Min` fold whose `post` casts back through `Cmp*(Reduced(0),
///    Const)`, `U8` out — see `assert_valid_out_dtype`'s `U8`/`I64` branches,
///    the two authored shapes). So rule 2's `Cmp*`/`Const` rejection is lifted
///    **only** while walking an [`Access::Reduction`]'s `body` or `post` at an
///    int dtype (`in_reduction` below) — the ELEMENTWISE arm is untouched and
///    keeps rejecting outright. The lift is further pinned to keep the
///    double-math hazard closed: every admitted `Cmp*`'s operands must be a
///    leaf `Input`/`Reduced` or a `Const` of EXACTLY `0.0`/`1.0` (the only
///    values `any`/`all`/`count` ever compare against, and the only values
///    that round-trip losslessly through `Const`'s dtype-oblivious f64
///    spelling — see `backend::const_lit`); anything else still declines.
pub(crate) fn check_int_op_admissibility(op: &OpDef, dtype: ElementKind) -> Result<(), PlanError> {
    check_int_op_admissibility_for(op, dtype, false)
}

/// The 32/64-bit integer dtypes whose `Div`, `Select` and comparisons a backend declaring
/// `Backend::total_integer_ops` is handed (`docs/int-div-shift-semantics.md`).
pub(super) fn is_total_int_dtype(dtype: ElementKind) -> bool {
    matches!(
        dtype,
        ElementKind::I32 | ElementKind::I64 | ElementKind::U32 | ElementKind::U64
    )
}

/// The reduce ops added in 0.42.0 (PM ruling 2026-11-02, `docs/options-u5-collectives-and-richer-reductions.md`):
/// `And`/`Or`/`Xor` fold integers, and `MaxNum`/`MinNum` fold ignoring NaN (0.42.1), in an [`Access::Reduction`]
/// only. Anything else declines here, typed, before a plan the oracle and the emitters would refuse is built.
pub(super) fn check_reduce_op_admissibility(
    op: &OpDef,
    dtype: ElementKind,
) -> Result<(), PlanError> {
    let ops: Vec<ReduceOp> = match &op.access {
        Access::Reduction { op, .. } | Access::Scan { op, .. } | Access::Window { op, .. } => {
            vec![*op]
        }
        Access::RowReduce { stages, .. } => stages.iter().map(|s| s.op).collect(),
        _ => Vec::new(),
    };
    let in_reduction = matches!(op.access, Access::Reduction { .. });
    let identity_post = matches!(
        &op.access,
        Access::Reduction { post, .. } if matches!(post, ScalarExpr::Reduced(0))
    );
    for rop in ops {
        let detail = match rop {
            ReduceOp::And
            | ReduceOp::Or
            | ReduceOp::Xor
            | ReduceOp::MaxNum
            | ReduceOp::MinNum
            | ReduceOp::ArgMax
            | ReduceOp::ArgMin
                if !in_reduction =>
            {
                format!("{rop:?} folds in a Reduction only, not in a scan, window or row reduction")
            }
            ReduceOp::ArgMax | ReduceOp::ArgMin if op.out_dtype != Some(ElementKind::I64) => {
                format!("{rop:?} gives an index: the op needs out_dtype Some(I64)")
            }
            ReduceOp::ArgMax | ReduceOp::ArgMin if !identity_post => {
                format!("{rop:?} gives an index: its post must be the identity Reduced(0)")
            }
            ReduceOp::And | ReduceOp::Or | ReduceOp::Xor if !is_int_dtype(dtype) => {
                format!("{rop:?} is a bitwise fold and needs an integer dtype")
            }
            _ => continue,
        };
        return Err(PlanError::InadmissibleOpAtDtype {
            dtype,
            detail: format!("op '{}': {detail}", op.name),
        });
    }
    Ok(())
}
