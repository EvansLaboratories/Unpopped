// SPDX-License-Identifier: MIT OR Apache-2.0
//! The checker for `Stmt::Tcgen05Mma` and `Stmt::Tcgen05Ld` (ROADMAP U7 S5): the tensor-memory allocation is `Tensor`,
//! 32-bit elements, `128 * columns` of them (at most 512 columns); the product's tiles are `Shared` places of a 16-bit
//! float type, `n` is a multiple of 8 up to 256, the accumulator is `F32`; a load needs 32-lane subgroups in a
//! one-dimensional block and a `Private` destination of the tensor element type.

use super::wgmma::{MAX_DESCRIPTOR_OFFSET, check_layout, check_tile};
use super::{Scope, check_expr};
use crate::exec::{Extent, MemorySpace};
use crate::kernel::{KernelError, KernelFn, OperandLayout, Stmt};
use crate::types::{self, Ty};
use unpopped_vocab::ElementKind;

fn refuse(what: &'static str) -> KernelError {
    KernelError::Tcgen05Unsupported { what }
}

/// The element type of the tensor-memory allocation `alloc`, once it is shown to be one.
fn check_tmem(k: &KernelFn, alloc: u32) -> Result<ElementKind, KernelError> {
    let Some(a) = k.allocs.get(alloc as usize) else {
        return Err(refuse("a tensor allocation that is not declared"));
    };
    if a.space != MemorySpace::Tensor {
        return Err(refuse(
            "a tensor-memory operand that is not a Tensor allocation",
        ));
    }
    if !matches!(
        a.elem,
        ElementKind::F32 | ElementKind::I32 | ElementKind::U32
    ) {
        return Err(refuse("a tensor-memory element that is not 32 bits"));
    }
    match a.extent {
        Extent::Static(n) if n != 0 && n % 128 == 0 && n / 128 <= 512 => Ok(a.elem),
        _ => Err(refuse(
            "a tensor allocation that is not 128 lanes of 1 to 512 columns",
        )),
    }
}

fn check_int(k: &KernelFn, e: &crate::ir::ScalarExpr, sc: &Scope) -> Result<(), KernelError> {
    match check_expr(k, e, sc)? {
        Ty::Scalar(t) if types::int_range(t).is_some() => Ok(()),
        _ => Err(refuse("a tensor-memory operand that is not an integer")),
    }
}

fn check_mma(k: &KernelFn, st: &Stmt, sc: &Scope) -> Result<(), KernelError> {
    let Stmt::Tcgen05Mma {
        n,
        ab,
        acc,
        a,
        b,
        d,
        d_col,
        scale_d,
        a_layout,
        b_layout,
        m,
        d_lane,
        b_ty,
    } = st
    else {
        unreachable!("check_mma is called for a Tcgen05Mma");
    };
    if k.launch.is_none() {
        return Err(KernelError::ExecWithoutLaunch);
    }
    if *n == 0 || *n > 256 || *n % 8 != 0 {
        return Err(refuse("an n that is not a multiple of 8 from 8 to 256"));
    }
    let bt = b_ty.unwrap_or(*ab);
    let fp8 = |t: ElementKind| matches!(t, ElementKind::Fp8E4M3FN | ElementKind::Fp8E5M2);
    let half = |t: ElementKind| matches!(t, ElementKind::F16 | ElementKind::Bf16);
    if !((half(*ab) && bt == *ab) || (fp8(*ab) && fp8(bt))) {
        return Err(refuse(
            "input types other than f16, bf16 (A and B alike) or two 8-bit floats (e4m3, e5m2)",
        ));
    }
    if *acc != ElementKind::F32 || check_tmem(k, *d)? != ElementKind::F32 {
        return Err(refuse("an accumulator that is not f32 in tensor memory"));
    }
    check_tile(k, a, *ab, sc)
        .map_err(|_| refuse("an A tile that is not a Shared place of the input type"))?;
    check_tile(k, b, bt, sc)
        .map_err(|_| refuse("a B tile that is not a Shared place of the input type"))?;
    // the tcgen05 descriptor's leading field is 15 bits (PTX ISA 9.5 9.7.18.4.1): twice the wgmma range
    let max_lbo = 2 * MAX_DESCRIPTOR_OFFSET + 16;
    // PTX ISA 9.5 9.7.18.10.5: M = 64 (non-.ws, cta_group::1: Layout F, lane alignment 0 or 16) or M = 128 (Layout D)
    if !matches!((*m, *d_lane), (128, 0) | (64, 0 | 16)) {
        return Err(refuse(
            "an m other than 128 (d_lane 0) or 64 (d_lane 0 or 16)",
        ));
    }
    check_layout(
        k,
        a,
        a_layout,
        *m as usize,
        max_lbo,
        if fp8(*ab) { 1 } else { 2 },
    )
    .map_err(refuse)?;
    check_layout(
        k,
        b,
        b_layout,
        *n as usize,
        max_lbo,
        if fp8(bt) { 1 } else { 2 },
    )
    .map_err(refuse)?;
    check_int(k, d_col, sc)?;
    check_int(k, scale_d, sc)
}

fn check_ld(k: &KernelFn, st: &Stmt, sc: &Scope) -> Result<(), KernelError> {
    let Stmt::Tcgen05Ld {
        src,
        col,
        count,
        dst,
    } = st
    else {
        unreachable!("check_ld is called for a Tcgen05Ld");
    };
    let Some(launch) = k.launch.as_ref() else {
        return Err(KernelError::ExecWithoutLaunch);
    };
    if !matches!(launch.subgroup, Some(s) if s.min == 32 && s.max == 32)
        || launch.block[1] != 1
        || launch.block[2] != 1
    {
        return Err(refuse(
            "a tensor-memory load outside a one-dimensional launch of 32-lane subgroups",
        ));
    }
    let elem = check_tmem(k, *src)?;
    if !count.is_power_of_two() || *count > 128 {
        return Err(refuse(
            "a column count that is not a power of two up to 128",
        ));
    }
    let ok = k.allocs.get(*dst as usize).is_some_and(|a| {
        a.space == MemorySpace::Private
            && a.elem == elem
            && matches!(a.extent, Extent::Static(n) if n >= *count)
    });
    if !ok {
        return Err(refuse(
            "a destination that is not Private of the tensor element type and at least count long",
        ));
    }
    check_int(k, col, sc)
}

/// Check a `Stmt::Tcgen05Mma` or `Stmt::Tcgen05Ld`.
pub(super) fn check_tcgen05(k: &KernelFn, st: &Stmt, sc: &Scope) -> Result<(), KernelError> {
    match st {
        Stmt::Tcgen05Mma { .. } => check_mma(k, st, sc),
        Stmt::Tcgen05Alloc { d, ncols } | Stmt::Tcgen05Dealloc { d, ncols } => {
            if k.launch.is_none() {
                return Err(KernelError::ExecWithoutLaunch);
            }
            check_tmem(k, *d)?;
            // PTX ISA 9.5 9.7.18.7.1: "the operand nCols must be a power of 2, in the range [32, 512]"
            if !ncols.is_power_of_two() || !(32..=512).contains(ncols) {
                return Err(refuse("an nCols that is not a power of two from 32 to 512"));
            }
            match k.allocs[*d as usize].extent {
                Extent::Static(n) if n == 128 * ncols => Ok(()),
                _ => Err(refuse(
                    "an nCols that is not the declared column count of the tensor allocation",
                )),
            }
        }
        Stmt::Tcgen05RelinquishPermit => {
            if k.launch.is_none() {
                return Err(KernelError::ExecWithoutLaunch);
            }
            Ok(())
        }
        _ => check_ld(k, st, sc),
    }
}
