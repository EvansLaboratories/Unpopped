// SPDX-License-Identifier: MIT OR Apache-2.0
//! The checker for `Stmt::Wgmma` (ROADMAP U7 S4, option A): the launch is whole warpgroups of 32-lane subgroups (a block
//! `[x, 1, 1]` with `x` a multiple of 128), `n` is a multiple of 8 up to 256, the inputs are 16-bit floats and the
//! accumulator `F32`, both tiles are places of `Shared` allocations of the input type, the fragment is a `Private`
//! allocation of exactly `n / 2` accumulator elements, and `scale_d` is an integer.

use super::{Scope, check_expr, check_place};
use crate::exec::{Extent, MemorySpace};
use crate::kernel::{KernelError, KernelFn, Place, Stmt};
use crate::types::{self, Ty};
use unpopped_vocab::ElementKind;

fn refuse(what: &'static str) -> KernelError {
    KernelError::WgmmaUnsupported { what }
}

fn check_launch(k: &KernelFn) -> Result<(), KernelError> {
    let Some(launch) = k.launch.as_ref() else {
        return Err(KernelError::ExecWithoutLaunch);
    };
    if !matches!(launch.subgroup, Some(s) if s.min == 32 && s.max == 32) {
        return Err(refuse("a launch whose subgroup size is not exactly 32"));
    }
    let [x, y, z] = launch.block;
    if y != 1 || z != 1 || x % 128 != 0 {
        return Err(refuse(
            "a launch block that is not [x, 1, 1] with x a multiple of 128",
        ));
    }
    Ok(())
}

pub(super) fn check_tile(
    k: &KernelFn,
    place: &Place,
    ab: ElementKind,
    sc: &Scope,
) -> Result<(), KernelError> {
    let Place::Alloc { alloc, .. } = place else {
        return Err(refuse("a tile that is not a Shared allocation place"));
    };
    if k.allocs
        .get(*alloc as usize)
        .is_none_or(|a| a.space != MemorySpace::Shared)
    {
        return Err(refuse("a tile that is not a Shared allocation place"));
    }
    let (elem, _) = check_place(k, place, sc)?;
    if elem != ab {
        return Err(refuse("a tile whose element type is not the input type"));
    }
    Ok(())
}

/// Check a `Stmt::Wgmma`.
pub(super) fn check_wgmma(k: &KernelFn, st: &Stmt, sc: &Scope) -> Result<(), KernelError> {
    let Stmt::Wgmma {
        n,
        ab,
        acc,
        a,
        b,
        d,
        scale_d,
    } = st
    else {
        unreachable!("check_wgmma is called for a Wgmma");
    };
    check_launch(k)?;
    if *n == 0 || *n > 256 || *n % 8 != 0 {
        return Err(refuse("an n that is not a multiple of 8 from 8 to 256"));
    }
    if !matches!(ab, ElementKind::F16 | ElementKind::Bf16) {
        return Err(refuse("an input type other than f16 or bf16"));
    }
    if *acc != ElementKind::F32 {
        return Err(refuse("an accumulator type other than f32"));
    }
    check_tile(k, a, *ab, sc)?;
    check_tile(k, b, *ab, sc)?;
    let Some(frag) = k.allocs.get(*d as usize) else {
        return Err(refuse("a fragment that is not a declared allocation"));
    };
    if frag.space != MemorySpace::Private || frag.elem != *acc {
        return Err(refuse(
            "a fragment that is not Private of the accumulator type",
        ));
    }
    if frag.extent != Extent::Static(n / 2) {
        return Err(refuse("a fragment whose length is not n / 2"));
    }
    let int = matches!(check_expr(k, scale_d, sc)?, Ty::Scalar(t) if types::int_range(t).is_some());
    if !int {
        return Err(refuse("a scale_d that is not an integer"));
    }
    Ok(())
}
