// SPDX-License-Identifier: MIT OR Apache-2.0
//! The checker for `Stmt::Collective` (ROADMAP U5 P5, `docs/options-u5-collectives-and-richer-reductions.md`).
//!
//! A collective is a rendezvous of the threads of a `Subgroup` or `Block` in a launch. Its `value` is a numeric
//! scalar (an integer, `F32` or `F64`); its `result` local is declared by the statement and has the type of `value`,
//! except a vote, whose result is a `U32` (`1`/`0`, or the ballot mask). The ops a reduction or scan folds with are
//! `Sum`, `Max`, `Min` and, on integers, `And`, `Or`, `Xor`; an exclusive scan needs an identity, so it takes only
//! `Sum`, `And`, `Or` and `Xor`. A ballot is a subgroup operation.

use unpopped_vocab::ElementKind;

use super::{Scope, check_expr, declare};
use crate::exec::Scope as ExecScope;
use crate::ir::{ReduceOp, ScalarExpr};
use crate::kernel::{CollectiveKind, KernelError, KernelFn, ParamKind, Stmt, VoteKind};
use crate::types::{self, Ty};

/// Check a `Stmt::Collective` and declare its result local.
pub(super) fn check_collective(
    k: &KernelFn,
    st: &Stmt,
    sc: &mut Scope,
    declared_here: &mut Vec<usize>,
) -> Result<(), KernelError> {
    let Stmt::Collective {
        kind,
        value,
        result,
        scope,
    } = st
    else {
        unreachable!("check_collective is called for a Collective");
    };
    if k.launch.is_none() {
        return Err(KernelError::ExecWithoutLaunch);
    }
    if !matches!(scope, ExecScope::Subgroup | ExecScope::Block) {
        return Err(KernelError::BadBarrierScope);
    }
    let Ty::Scalar(t) = check_expr(k, value, sc)? else {
        return Err(refuse("a packed collective value"));
    };
    if types::int_range(t).is_none() && !matches!(t, ElementKind::F32 | ElementKind::F64) {
        return Err(refuse("a collective value outside integers, F32 and F64"));
    }
    let result_ty = match kind {
        CollectiveKind::Shuffle { operand, .. } => lane_operand(k, operand, sc).map(|()| t)?,
        CollectiveKind::Broadcast { lane } => lane_operand(k, lane, sc).map(|()| t)?,
        CollectiveKind::Vote(v) => {
            if matches!(v, VoteKind::Ballot) && matches!(scope, ExecScope::Block) {
                return Err(refuse("a ballot over a Block (it is a Subgroup operation)"));
            }
            ElementKind::U32
        }
        CollectiveKind::Reduce(op) => fold_op(*op, t, false).map(|()| t)?,
        CollectiveKind::Scan { op, exclusive } => fold_op(*op, t, *exclusive).map(|()| t)?,
        CollectiveKind::ReduceWith { combine } | CollectiveKind::ScanWith { combine } => {
            check_combine(sc, *combine, t).map(|()| t)?
        }
    };
    declare(k, *result, Ty::Scalar(result_ty), sc, declared_here)
}

/// The function a combining collective names is a plain function `(a: T, b: T) -> T` of the module, with no barrier.
/// Alone (`check_kernel`) there is no module to resolve it in.
fn check_combine(sc: &Scope, callee: u32, t: ElementKind) -> Result<(), KernelError> {
    if sc.callees.is_empty() {
        return Err(KernelError::CallUnsupported {
            what: "a combining collective outside a module (check_module resolves its function)",
        });
    }
    let Some(sig) = sc.callees.get(callee as usize) else {
        return Err(KernelError::BadCallee { callee });
    };
    let scalar_t = |p: &ParamKind| matches!(p, ParamKind::Scalar { ty, .. } if *ty == t);
    if sig.params.len() != 2
        || !sig.params.iter().all(scalar_t)
        || sig.result != Some(Ty::Scalar(t))
    {
        return Err(refuse(
            "a combine function that is not (a: T, b: T) -> T over the collective's value type",
        ));
    }
    if sig.barrier {
        return Err(refuse("a combine function that reaches a barrier"));
    }
    Ok(())
}

fn refuse(what: &'static str) -> KernelError {
    KernelError::CollectiveUnsupported { what }
}

/// A lane index, distance or mask is an integer.
fn lane_operand(k: &KernelFn, e: &ScalarExpr, sc: &Scope) -> Result<(), KernelError> {
    match check_expr(k, e, sc)? {
        Ty::Scalar(t) if types::int_range(t).is_some() => Ok(()),
        _ => Err(KernelError::TypeMismatch {
            what: "a collective's lane operand",
        }),
    }
}

/// The fold of a reduction or scan over values of type `t`.
fn fold_op(op: ReduceOp, t: ElementKind, exclusive: bool) -> Result<(), KernelError> {
    let int = types::int_range(t).is_some();
    let ok = match op {
        ReduceOp::Sum | ReduceOp::Max | ReduceOp::Min => true,
        ReduceOp::And | ReduceOp::Or | ReduceOp::Xor => int,
        _ => false,
    };
    if !ok {
        return Err(refuse(
            "a collective fold outside Sum, Max, Min and the integer bit ops",
        ));
    }
    if exclusive && matches!(op, ReduceOp::Max | ReduceOp::Min) {
        return Err(refuse("an exclusive Max/Min scan (no identity for lane 0)"));
    }
    if exclusive && !int && !matches!(op, ReduceOp::Sum) {
        return Err(refuse("an exclusive scan of floats other than Sum"));
    }
    Ok(())
}
