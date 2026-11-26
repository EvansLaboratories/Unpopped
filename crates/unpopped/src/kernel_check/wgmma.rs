// SPDX-License-Identifier: MIT OR Apache-2.0
//! The checker for `Stmt::Wgmma` (ROADMAP U7 S4, option A): the launch is whole warpgroups of 32-lane subgroups (a block
//! `[x, 1, 1]` with `x` a multiple of 128), `n` is a multiple of 8 up to 256, the inputs are 16-bit floats and the
//! accumulator `F32`, both tiles are places of `Shared` allocations of the input type, the fragment is a `Private`
//! allocation of exactly `n / 2` accumulator elements, and `scale_d` is an integer.

use super::{Scope, check_expr, check_place};
use crate::exec::{Extent, MemorySpace};
use crate::ir::ScalarExpr;
use crate::kernel::{KernelError, KernelFn, Major, OperandLayout, Place, Stmt, Swizzle};
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

/// The largest byte offset a matrix descriptor's leading or stride field can carry: 14 bits of `offset >> 4` (PTX ISA 9.5
/// 9.7.17.5.1.2.2, `matrix-descriptor-encode(x) = (x & 0x3FFFF) >> 4`).
pub(super) const MAX_DESCRIPTOR_OFFSET: u32 = (1 << 18) - 16;

/// Check how a `Wgmma` tile of `rows` rows is laid out. A `Canonical` layout is what an emitter lowers to a descriptor,
/// so it must be one: no swizzle (the others are not built), byte offsets the descriptor can carry (multiples of 16,
/// at most `MAX_DESCRIPTOR_OFFSET`), a start the descriptor can name (a constant, 16-byte aligned, so the descriptor's
/// base offset is 0) and offsets that put every element of the tile at its own place inside the allocation.
pub(super) fn check_layout(
    k: &KernelFn,
    place: &Place,
    layout: &OperandLayout,
    rows: usize,
    max_lbo: u32,
) -> Result<(), &'static str> {
    let OperandLayout::Canonical {
        major,
        swizzle,
        lbo,
        sbo,
    } = *layout
    else {
        return Ok(());
    };
    // the leading offset is "not used, assumed to be 1" for a swizzled K-major operand (so it must be absent); every
    // other layout needs it
    let swizzled = swizzle != Swizzle::None;
    let offsets_to_check: Vec<(u32, u32)> = match (major == Major::K && swizzled, lbo) {
        (true, None) => vec![(sbo, MAX_DESCRIPTOR_OFFSET)],
        (true, Some(_)) => {
            return Err(
                "a swizzled K-major operand layout with a leading offset (the ISA does not use it)",
            );
        }
        (false, Some(l)) => vec![(l, max_lbo), (sbo, MAX_DESCRIPTOR_OFFSET)],
        (false, None) => {
            return Err("an operand layout without the leading dimension byte offset it needs");
        }
    };
    for (off, max) in offsets_to_check {
        if off == 0 || off % 16 != 0 || off > max {
            return Err(
                "a leading or stride byte offset that is zero, not a multiple of 16, or beyond the descriptor's 14-bit field",
            );
        }
    }
    let Place::Alloc { alloc, index } = place else {
        return Err("a tile that is not a Shared allocation place");
    };
    let ScalarExpr::ConstInt(start, _) = index else {
        return Err(
            "a Canonical operand layout on a tile whose start is not a constant (the descriptor needs a known, aligned address)",
        );
    };
    // 16-bit elements: 2 bytes each; the start is a multiple of the swizzle pattern (16 bytes with none)
    let Ok(start) = usize::try_from(*start) else {
        return Err("a negative tile start");
    };
    if start % (swizzle.pattern_bytes() as usize / 2) != 0 {
        return Err(
            "a tile start that is not a multiple of the swizzle pattern (1024 / 512 / 256 bytes; 16 with none)",
        );
    }
    let Some(Extent::Static(len)) = k.allocs.get(*alloc as usize).map(|a| a.extent) else {
        return Err("a Canonical operand layout on an allocation without a static extent");
    };
    let mut seen = std::collections::HashSet::new();
    for mn in 0..rows {
        for kk in 0..16 {
            let Some(off) = layout.element_offset(mn, kk) else {
                return Err("an operand layout with no element offset");
            };
            if start + off >= len as usize {
                return Err("an operand layout that leaves the allocation");
            }
            if !seen.insert(off) {
                return Err("an operand layout that puts two elements of the tile at one place");
            }
        }
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
        a_layout,
        b_layout,
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
    check_layout(k, a, a_layout, 64, MAX_DESCRIPTOR_OFFSET).map_err(refuse)?;
    check_layout(k, b, b_layout, *n as usize, MAX_DESCRIPTOR_OFFSET).map_err(refuse)?;
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
