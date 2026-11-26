// SPDX-License-Identifier: MIT OR Apache-2.0
//! The checker for the `mbarrier` statements (`Stmt::MbarInit`, `MbarArrive`, `MbarCompleteTx`, `MbarWait`; ROADMAP U8c,
//! `docs/options-u8-copy-atoms.md` S3): the barrier is a place of a `Shared` allocation of 64-bit elements, and every
//! operand (arrival count, transaction bytes, parity) is an integer expression. Whether an arrival is legal (the barrier
//! initialised, no more arrivals than pending) depends on run-time control flow, so the oracle traps on those.

use super::{Scope, check_expr, check_place};
use crate::exec::MemorySpace;
use crate::ir::ScalarExpr;
use crate::kernel::{KernelError, KernelFn, Place, Stmt};
use crate::types::{self, Ty, elem_bits};
use unpopped_vocab::ElementKind;

fn refuse(what: &'static str) -> KernelError {
    KernelError::MbarrierUnsupported { what }
}

fn refuse_bulk(what: &'static str) -> KernelError {
    KernelError::BulkCopyUnsupported { what }
}

/// Check a `Stmt::CopyBulk`: a `Shared` destination, a readable global source of the same element type, a size that is a
/// multiple of 16 bytes, and a `Shared` 64-bit barrier.
pub(super) fn check_copy_bulk(k: &KernelFn, st: &Stmt, sc: &Scope) -> Result<(), KernelError> {
    let Stmt::CopyBulk {
        dst,
        src,
        elems,
        bar,
    } = st
    else {
        unreachable!("check_copy_bulk is called for a CopyBulk");
    };
    let shared = |p: &Place| {
        matches!(p, Place::Alloc { alloc, .. }
            if k.allocs.get(*alloc as usize).is_some_and(|a| a.space == MemorySpace::Shared))
    };
    if !shared(dst) {
        return Err(refuse_bulk(
            "a destination that is not a Shared allocation place",
        ));
    }
    if !shared(bar) {
        return Err(refuse_bulk(
            "a barrier that is not a Shared allocation place",
        ));
    }
    let Place::Logical { buffer, .. } = src else {
        return Err(refuse_bulk("a source that is not a global buffer place"));
    };
    let (de, _) = check_place(k, dst, sc)?;
    let (se, role) = check_place(k, src, sc)?;
    if !role.reads() {
        return Err(KernelError::ReadFromWriteOnly { buffer: *buffer });
    }
    if de != se {
        return Err(KernelError::TypeMismatch {
            what: "a CopyBulk destination and source element type",
        });
    }
    if (elem_bits(de) * *elems as usize) % 128 != 0 || *elems == 0 {
        return Err(refuse_bulk("a size that is not a multiple of 16 bytes"));
    }
    let (be, _) = check_place(k, bar, sc)?;
    if elem_bits(be) != 64 {
        return Err(refuse_bulk("a barrier whose elements are not 64 bits"));
    }
    Ok(())
}

/// Check a `Stmt::CopyUnpack`: a `Shared` `U8` destination, a readable global `F4E2M1` source, at least one group, and a
/// `Shared` 64-bit barrier.
pub(super) fn check_copy_unpack(k: &KernelFn, st: &Stmt, sc: &Scope) -> Result<(), KernelError> {
    let Stmt::CopyUnpack {
        dst,
        src,
        groups,
        bar,
    } = st
    else {
        unreachable!("check_copy_unpack is called for a CopyUnpack");
    };
    let shared = |p: &Place| {
        matches!(p, Place::Alloc { alloc, .. }
            if k.allocs.get(*alloc as usize).is_some_and(|a| a.space == MemorySpace::Shared))
    };
    if !shared(dst) || !shared(bar) {
        return Err(refuse_bulk(
            "a destination or barrier that is not a Shared allocation place",
        ));
    }
    let Place::Logical { buffer, .. } = src else {
        return Err(refuse_bulk("a source that is not a global buffer place"));
    };
    let (de, _) = check_place(k, dst, sc)?;
    let (se, role) = check_place(k, src, sc)?;
    if !role.reads() {
        return Err(KernelError::ReadFromWriteOnly { buffer: *buffer });
    }
    if de != ElementKind::U8 {
        return Err(refuse_bulk(
            "a destination that is not a byte (U8) allocation",
        ));
    }
    if se != ElementKind::F4E2M1 {
        return Err(refuse_bulk(
            "a source that is not F4E2M1 (FP6 buffers have no KISS packing pin yet)",
        ));
    }
    if *groups == 0 {
        return Err(refuse_bulk("a copy of no groups"));
    }
    let (be, _) = check_place(k, bar, sc)?;
    if elem_bits(be) != 64 {
        return Err(refuse_bulk("a barrier whose elements are not 64 bits"));
    }
    Ok(())
}

/// Check an `mbarrier` statement.
pub(super) fn check_mbar(k: &KernelFn, st: &Stmt, sc: &Scope) -> Result<(), KernelError> {
    let (bar, operands): (&Place, Vec<&ScalarExpr>) = match st {
        Stmt::MbarInit { bar, count } => (bar, vec![count]),
        Stmt::MbarArrive { bar, tx } => (bar, tx.iter().collect()),
        Stmt::Tcgen05Commit { bar } => (bar, vec![]),
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
