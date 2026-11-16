// SPDX-License-Identifier: MIT OR Apache-2.0
//! The checker for `Stmt::MatrixLoad` and `Stmt::MatrixStore` (ROADMAP U8b, `docs/options-u8-copy-atoms.md` S2). A matrix load is a rendezvous of
//! one 32-lane warp, so the launch's subgroup size is exactly 32; `count` is 1, 2 or 4; the source is a place of a
//! `Shared` allocation of 16-bit elements; the destination is a `Private`, statically sized allocation of `2 * count`
//! elements of the source's type.

use super::{Scope, check_place};
use crate::exec::{Extent, MemorySpace};
use crate::kernel::{KernelError, KernelFn, Place, Stmt};
use crate::types::elem_bits;

fn refuse_load(what: &'static str) -> KernelError {
    KernelError::MatrixLoadUnsupported { what }
}

fn refuse_store(what: &'static str) -> KernelError {
    KernelError::MatrixStoreUnsupported { what }
}

/// Check a `Stmt::MatrixLoad`.
pub(super) fn check_matrix_load(k: &KernelFn, st: &Stmt, sc: &Scope) -> Result<(), KernelError> {
    let Stmt::MatrixLoad {
        count,
        src,
        dst,
        trans: _,
    } = st
    else {
        unreachable!("check_matrix_load is called for a MatrixLoad");
    };
    check_matrix(k, (*count, src, *dst), sc, refuse_load)
}

/// Check a `Stmt::MatrixStore`: the same shape as a load with the roles of the two operands exchanged.
pub(super) fn check_matrix_store(k: &KernelFn, st: &Stmt, sc: &Scope) -> Result<(), KernelError> {
    let Stmt::MatrixStore {
        count,
        dst,
        src,
        trans: _,
    } = st
    else {
        unreachable!("check_matrix_store is called for a MatrixStore");
    };
    check_matrix(k, (*count, dst, *src), sc, refuse_store)
}

/// What a matrix load or store needs: a 32-lane launch, a tile count, a `Shared` 16-bit place and a `Private` fragment.
fn check_matrix(
    k: &KernelFn,
    (count, place, frag): (u32, &Place, u32),
    sc: &Scope,
    refuse: fn(&'static str) -> KernelError,
) -> Result<(), KernelError> {
    let Some(launch) = k.launch.as_ref() else {
        return Err(KernelError::ExecWithoutLaunch);
    };
    if !matches!(launch.subgroup, Some(s) if s.min == 32 && s.max == 32) {
        return Err(refuse("a launch whose subgroup size is not exactly 32"));
    }
    if !matches!(count, 1 | 2 | 4) {
        return Err(refuse("a tile count other than 1, 2 or 4"));
    }
    let Place::Alloc { alloc, .. } = place else {
        return Err(refuse("a matrix place that is not a Shared allocation place"));
    };
    if k.allocs
        .get(*alloc as usize)
        .is_none_or(|a| a.space != MemorySpace::Shared)
    {
        return Err(refuse("a matrix place that is not a Shared allocation place"));
    }
    let (elem, _) = check_place(k, place, sc)?;
    if elem_bits(elem) != 16 {
        return Err(refuse("a matrix place whose elements are not 16 bits"));
    }
    let Some(d) = k.allocs.get(frag as usize) else {
        return Err(refuse("a fragment that is not a declared allocation"));
    };
    if d.space != MemorySpace::Private {
        return Err(refuse("a fragment outside Private memory"));
    }
    if d.elem != elem {
        return Err(refuse(
            "a fragment whose element type is not the place's",
        ));
    }
    if d.extent != Extent::Static(2 * count) {
        return Err(refuse("a fragment whose length is not 2 * count"));
    }
    Ok(())
}
