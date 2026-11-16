// SPDX-License-Identifier: MIT OR Apache-2.0
//! The checker for `Stmt::MatrixLoad` (ROADMAP U8b, `docs/options-u8-copy-atoms.md` S2). A matrix load is a rendezvous of
//! one 32-lane warp, so the launch's subgroup size is exactly 32; `count` is 1, 2 or 4; the source is a place of a
//! `Shared` allocation of 16-bit elements; the destination is a `Private`, statically sized allocation of `2 * count`
//! elements of the source's type.

use super::{Scope, check_place};
use crate::exec::{Extent, MemorySpace};
use crate::kernel::{KernelError, KernelFn, Place, Stmt};
use crate::types::elem_bits;

fn refuse(what: &'static str) -> KernelError {
    KernelError::MatrixLoadUnsupported { what }
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
    let Some(launch) = k.launch.as_ref() else {
        return Err(KernelError::ExecWithoutLaunch);
    };
    if !matches!(launch.subgroup, Some(s) if s.min == 32 && s.max == 32) {
        return Err(refuse("a launch whose subgroup size is not exactly 32"));
    }
    if !matches!(count, 1 | 2 | 4) {
        return Err(refuse("a tile count other than 1, 2 or 4"));
    }
    let Place::Alloc { alloc, .. } = src else {
        return Err(refuse("a source that is not a Shared allocation place"));
    };
    if k.allocs
        .get(*alloc as usize)
        .is_none_or(|a| a.space != MemorySpace::Shared)
    {
        return Err(refuse("a source that is not a Shared allocation place"));
    }
    let (elem, _) = check_place(k, src, sc)?;
    if elem_bits(elem) != 16 {
        return Err(refuse("a source whose elements are not 16 bits"));
    }
    let Some(d) = k.allocs.get(*dst as usize) else {
        return Err(refuse("a destination that is not a declared allocation"));
    };
    if d.space != MemorySpace::Private {
        return Err(refuse("a destination outside Private memory"));
    }
    if d.elem != elem {
        return Err(refuse(
            "a destination whose element type is not the source's",
        ));
    }
    if d.extent != Extent::Static(2 * *count) {
        return Err(refuse("a destination whose length is not 2 * count"));
    }
    Ok(())
}
