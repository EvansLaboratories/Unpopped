// SPDX-License-Identifier: MIT OR Apache-2.0
//! The cpu_c spelling of [`ScalarExpr::Qualified`](unpopped::ir::ScalarExpr::Qualified)
//! (`docs/ir-numerics-qualifiers.md`).
//!
//! Spelled only where plain C99 honours the qualifier exactly:
//!
//! - **`ftz`**: an explicit flush in C ([`FTZ_HELPER`]) of each operand and of the
//!   result, never the host's FPU control register, so the answer does not depend
//!   on how the process was started.
//! - **`approx`**: spelled as the exact function. That is inside the `approx`
//!   bound by construction (zero error), which is the licence's whole content for
//!   a backend with no faster form.
//! - **a rounding mode other than nearest-even**: DECLINED. C99 reaches a directed
//!   mode only through `fesetround`, which is global state a kernel must not touch,
//!   and the alternative (an emulation with error-free transforms in C) is not
//!   worth shipping unmeasured. The decline names the mode; the bare operation is
//!   never substituted.

use unpopped::backend::{Decline, DeclinedOp, LowerError, Spelling};
use unpopped::ir::{Numerics, QualOp, RoundMode, UnaryOp};
use unpopped_vocab::ElementKind;

/// The flush helper a kernel with an `ftz` qualifier carries (emitted only then, so
/// every other kernel is byte-identical to before). `0x1p-126f` is the smallest
/// normal float; a smaller magnitude is subnormal or zero, and `copysignf(0, x)`
/// keeps the sign. NaN fails the comparison and passes through.
pub(crate) const FTZ_HELPER: &str = "\
static float unp_ftz(float x) {
    return fabsf(x) < 0x1p-126f ? copysignf(0.0f, x) : x;
}
";

fn decline(why: String) -> Result<Spelling, LowerError> {
    Ok(Spelling::Declined(Decline::UnsupportedOp {
        op: DeclinedOp::Qualified,
        why,
    }))
}

/// Spell `Qualified(op(operands))` per `n` at compute dtype `dtype`, or decline.
pub(crate) fn cpu_qualified(
    op: QualOp,
    operands: Vec<String>,
    n: Numerics,
    dtype: ElementKind,
) -> Result<Spelling, LowerError> {
    if !matches!(dtype, ElementKind::F32 | ElementKind::F32Strict) {
        return decline(format!(
            "cpu_c spells a Qualified {op:?} only at f32/f32s compute in v1, got {dtype:?}"
        ));
    }
    if n.round != RoundMode::NearestEven {
        return decline(format!(
            "cpu_c does not spell {op:?} under RoundMode::{:?}: C99 reaches a directed \
             mode only through fesetround, which is global state a kernel must not touch, \
             and substituting the round-to-nearest operation would drop the qualifier",
            n.round
        ));
    }
    if operands.len() != op.arity() {
        return decline(format!(
            "Qualified {op:?} expects {} operands, got {}",
            op.arity(),
            operands.len()
        ));
    }
    let flush = |x: &String| {
        if n.ftz {
            format!("unp_ftz({x})")
        } else {
            x.clone()
        }
    };
    let a: Vec<String> = operands.iter().map(flush).collect();
    let body = match op {
        QualOp::Add => format!("({} + {})", a[0], a[1]),
        QualOp::Sub => format!("({} - {})", a[0], a[1]),
        QualOp::Mul => format!("({} * {})", a[0], a[1]),
        QualOp::Div => format!("({} / {})", a[0], a[1]),
        QualOp::Fma => format!("fmaf({}, {}, {})", a[0], a[1], a[2]),
        QualOp::Sqrt | QualOp::Rsqrt | QualOp::Recip | QualOp::Exp2 | QualOp::Log2 => {
            let u = match op {
                QualOp::Sqrt => UnaryOp::Sqrt,
                QualOp::Rsqrt => UnaryOp::Rsqrt,
                QualOp::Recip => UnaryOp::Recip,
                QualOp::Exp2 => UnaryOp::Exp2,
                _ => UnaryOp::Log2,
            };
            match crate::cpu_unary(u, a[0].clone(), dtype)? {
                Spelling::Spelled(s) => s,
                declined => return Ok(declined),
            }
        }
    };
    Ok(Spelling::Spelled(if n.ftz {
        format!("unp_ftz({body})")
    } else {
        body
    }))
}
