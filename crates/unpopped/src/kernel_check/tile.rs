// SPDX-License-Identifier: MIT OR Apache-2.0
//! The checker for the opaque-tile statements (ROADMAP T2, `docs/design-opaque-tile-mma.md`): `TileLoad`, `TileMma`,
//! `TileFill`, `TileStore`, and the rule that a [`MemorySpace::Tile`] allocation carries its shape and that no ordinary access
//! reaches one. Uniform control is the uniformity pass's rule, not this file's.

use super::{Scope, check_expr, check_place};
use crate::exec::{AllocDecl, Extent, MemorySpace};
use crate::ir::ScalarExpr;
use crate::kernel::{BufferRole, KernelError, KernelFn, Place, Stmt};
use crate::types::Ty;
use unpopped_vocab::ElementKind::{self, Bf16, F16, F32, F64};

fn refuse(alloc: u32, why: &'static str) -> KernelError {
    KernelError::Tile { alloc, why }
}

/// One allocation's tile rules: a `Tile` space has a tile shape, a static extent of `rows * cols` and no layout, and no other
/// space has a tile shape.
pub(super) fn check_tile_alloc(a: &AllocDecl, alloc: u32) -> Result<(), KernelError> {
    match (a.space == MemorySpace::Tile, a.tile) {
        (false, None) => Ok(()),
        (false, Some(_)) => Err(refuse(
            alloc,
            "a tile shape on an allocation that is not a Tile",
        )),
        (true, None) => Err(refuse(alloc, "a Tile allocation without a tile shape")),
        (true, Some(t)) => {
            if t.rows == 0 || t.cols == 0 {
                return Err(refuse(alloc, "a tile with no rows or no columns"));
            }
            if a.extent != Extent::Static(t.rows.saturating_mul(t.cols)) {
                return Err(refuse(alloc, "a Tile whose extent is not rows * cols"));
            }
            if a.layout.is_some() || a.align_bytes.is_some() {
                return Err(refuse(
                    alloc,
                    "a layout or alignment on a Tile (no element is addressable)",
                ));
            }
            if !matches!(a.elem, F16 | Bf16 | F32 | F64) {
                return Err(refuse(
                    alloc,
                    "a tile of an element type other than F16, BF16, F32 or F64",
                ));
            }
            Ok(())
        }
    }
}

/// The tile `alloc` of a statement: it must exist and be a `Tile` (its shape and element come back).
fn tile_of(k: &KernelFn, alloc: u32) -> Result<(u32, u32, ElementKind), KernelError> {
    match k.allocs.get(alloc as usize) {
        Some(a) if a.space == MemorySpace::Tile => {
            let t = a
                .tile
                .ok_or_else(|| refuse(alloc, "a Tile allocation without a tile shape"))?;
            Ok((t.rows, t.cols, a.elem))
        }
        _ => Err(refuse(
            alloc,
            "a tile statement over an allocation that is not a Tile",
        )),
    }
}

/// Check one tile statement.
pub(super) fn check_tile(k: &KernelFn, st: &Stmt, sc: &Scope) -> Result<(), KernelError> {
    if k.launch.is_none() {
        return Err(KernelError::ExecWithoutLaunch);
    }
    match st {
        Stmt::TileLoad {
            d,
            src,
            elements_per_row,
            origin,
        } => tile_move(
            k,
            sc,
            &Move {
                tile: *d,
                place: src,
                per_row: elements_per_row,
                origin,
            },
            true,
        ),
        Stmt::TileStore {
            a,
            dst,
            elements_per_row,
            origin,
        } => tile_move(
            k,
            sc,
            &Move {
                tile: *a,
                place: dst,
                per_row: elements_per_row,
                origin,
            },
            false,
        ),
        Stmt::TileFill { d, value } => check_tile_fill(k, sc, *d, value),
        Stmt::TileMma { d, a, b, c } => check_tile_mma(k, [*d, *a, *b, *c]),
        _ => unreachable!("check_tile is called for a tile statement"),
    }
}

/// A `TileFill`: the value has the tile's element type.
fn check_tile_fill(
    k: &KernelFn,
    sc: &Scope,
    d: u32,
    value: &ScalarExpr,
) -> Result<(), KernelError> {
    let (_, _, elem) = tile_of(k, d)?;
    if check_expr(k, value, sc)? != Ty::Scalar(elem) {
        return Err(refuse(
            d,
            "a TileFill value whose type is not the tile's element type",
        ));
    }
    Ok(())
}

/// A tile's place: a `Shared` allocation place or a one-index buffer place.
fn check_tile_place(k: &KernelFn, tile: u32, place: &Place) -> Result<(), KernelError> {
    let ok = match place {
        Place::Alloc { alloc, .. } => k
            .allocs
            .get(*alloc as usize)
            .is_some_and(|a| a.space == MemorySpace::Shared),
        Place::Logical { coords, .. } => coords.len() == 1,
        _ => false,
    };
    if ok {
        Ok(())
    } else {
        Err(refuse(
            tile,
            "a tile place that is not a Shared allocation place or a one-index buffer place",
        ))
    }
}

/// What a `TileLoad` or a `TileStore` names: the tile, the place, the row stride and the origin.
struct Move<'a> {
    tile: u32,
    place: &'a Place,
    per_row: &'a ScalarExpr,
    origin: &'a [ScalarExpr; 2],
}

/// A `TileLoad` (`load`) or a `TileStore`: the tile, the place, the strides and origins.
fn tile_move(k: &KernelFn, sc: &Scope, mv: &Move<'_>, load: bool) -> Result<(), KernelError> {
    let Move {
        tile,
        place,
        per_row,
        origin,
    } = *mv;
    let (_, _, elem) = tile_of(k, tile)?;
    check_tile_place(k, tile, place)?;
    let (pe, role) = check_place(k, place, sc)?;
    if pe != elem {
        return Err(refuse(
            tile,
            "a tile place whose element type is not the tile's",
        ));
    }
    let wrong_way = if load {
        BufferRole::Write
    } else {
        BufferRole::Read
    };
    if role == wrong_way {
        return Err(refuse(
            tile,
            "a tile place whose buffer is not readable (load) or writable (store)",
        ));
    }
    for e in [per_row, &origin[0], &origin[1]] {
        if !matches!(check_expr(k, e, sc)?, Ty::Scalar(t) if crate::types::int_range(t).is_some()) {
            return Err(refuse(
                tile,
                "a tile stride or origin that is not an integer",
            ));
        }
    }
    Ok(())
}

/// A `TileMma` over `[d, a, b, c]`: `a` is `m x k`, `b` is `k x n`, `c` and `d` are `m x n`; `a` and `b` share an element type, `c`
/// and `d` share the accumulator type (`F32` over any, `F64` over `F64`).
fn check_tile_mma(k: &KernelFn, [d, a, b, c]: [u32; 4]) -> Result<(), KernelError> {
    let (da, db, dc, dd) = (
        tile_of(k, a)?,
        tile_of(k, b)?,
        tile_of(k, c)?,
        tile_of(k, d)?,
    );
    if da.2 != db.2 {
        return Err(refuse(
            a,
            "a TileMma whose A and B have different element types",
        ));
    }
    if dc.2 != dd.2 {
        return Err(refuse(
            c,
            "a TileMma whose C and D have different element types",
        ));
    }
    let (m, kk, n) = (da.0, da.1, db.1);
    if db.0 != kk || (dc.0, dc.1) != (m, n) || (dd.0, dd.1) != (m, n) {
        return Err(refuse(
            d,
            "a TileMma whose shapes are not (m x k) (k x n) + (m x n) -> (m x n)",
        ));
    }
    let acc_ok = dc.2 == F32 || (dc.2 == F64 && da.2 == F64);
    if !acc_ok {
        return Err(refuse(
            c,
            "a TileMma accumulator other than F32, or F64 over F64",
        ));
    }
    Ok(())
}
