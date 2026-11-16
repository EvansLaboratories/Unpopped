// SPDX-License-Identifier: MIT OR Apache-2.0
//! The checker for the `mbarrier` statements (`Stmt::MbarInit`, `MbarArrive`, `MbarCompleteTx`, `MbarWait`; ROADMAP U8c,
//! `docs/options-u8-copy-atoms.md` S3): the barrier is a place of a `Shared` allocation of 64-bit elements, and every
//! operand (arrival count, transaction bytes, parity) is an integer expression. Whether an arrival is legal (the barrier
//! initialised, no more arrivals than pending) depends on run-time control flow, so the oracle traps on those.

use super::{Scope, check_expr, check_place};
use crate::exec::MemorySpace;
use crate::kernel::{KernelError, KernelFn, Place, Stmt};
use crate::types::{self, Ty, elem_bits};
use crate::ir::ScalarExpr;

fn refuse(what: &'static str) -> KernelError {
    KernelError::MbarrierUnsupported { what }
}

/// Check an `mbarrier` statement.
pub(super) fn check_mbar(k: &KernelFn, st: &Stmt, sc: &Scope) -> Result<(), KernelError> {
    let (bar, operands): (&Place, Vec<&ScalarExpr>) = match st {
        Stmt::MbarInit { bar, count } => (bar, vec![count]),
        Stmt::MbarArrive { bar, tx } => (bar, tx.iter().collect()),
        Stmt::MbarCompleteTx { bar, bytes } => (bar, vec![bytes]),
        Stmt::MbarWait { bar, parity } => (bar, vec![parity]),
        _ => unreachable!("check_mbar is called for an mbarrier statement"),
    };
    let Place::Alloc { alloc, .. } = bar else {
        return Err(refuse("a barrier that is not a Shared allocation place"));
    };
    if k.allocs
        .get(*alloc as usize)
        .is_none_or(|a| a.space != MemorySpace::Shared)
    {
        return Err(refuse("a barrier that is not a Shared allocation place"));
    }
    let (elem, _) = check_place(k, bar, sc)?;
    if elem_bits(elem) != 64 {
        return Err(refuse("a barrier whose elements are not 64 bits"));
    }
    for e in operands {
        let int = matches!(check_expr(k, e, sc)?, Ty::Scalar(t) if types::int_range(t).is_some());
        if !int {
            return Err(refuse("an mbarrier operand that is not an integer"));
        }
    }
    Ok(())
}
