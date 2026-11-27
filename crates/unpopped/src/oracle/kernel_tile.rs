// SPDX-License-Identifier: MIT OR Apache-2.0
//! **What the opaque-tile statements compute** (ROADMAP T2, `docs/design-opaque-tile-mma.md`). A [`MemorySpace::Tile`]
//! allocation is stored like a `Private` array: the checker (and the uniformity pass) make every lane of the subgroup reach
//! a tile statement together with the same operands, so every lane's copy of a tile holds the same elements and the tile
//! behaves as one value held by the subgroup. `TileLoad` is an ordinary logged load on every lane (a read is no race);
//! `TileStore` is performed by the subgroup's first lane only, so the race log sees one writer; `TileMma` is the contract of
//! `Stmt::Mma`: exact `f64` products, the sum of `C` and the products taken in ascending `k`, rounded once to the accumulator.

use super::kernel_eval::{KernelTrap, Machine};
use super::{Val, store_val};
use crate::exec::{ExecId, MemorySpace};
use crate::ir::ScalarExpr;
use crate::kernel::{Place, Stmt};
use crate::types::{self, Ty};
use unpopped_vocab::ElementKind;

impl Machine<'_> {
    /// Run one tile statement.
    pub(super) fn exec_tile(&mut self, st: &Stmt) -> Result<(), KernelTrap> {
        match st {
            Stmt::TileLoad {
                d,
                src,
                elements_per_row,
                origin,
            } => {
                let (rows, cols) = self.tile_shape(*d)?;
                let at = self.tile_origin(elements_per_row, origin)?;
                for r in 0..rows {
                    for c in 0..cols {
                        let p = self.tile_place(src, at.element(r, c))?;
                        let v = self.load(&p)?;
                        self.tile_write(*d, (r * cols + c) as usize, v)?;
                    }
                }
            }
            Stmt::TileStore {
                a,
                dst,
                elements_per_row,
                origin,
            } => {
                let (rows, cols) = self.tile_shape(*a)?;
                let at = self.tile_origin(elements_per_row, origin)?;
                let vals = self.private_all(*a)?;
                // one writer for the whole subgroup: its first lane
                if self.exec_val(ExecId::LaneId)? == 0 {
                    for r in 0..rows {
                        for c in 0..cols {
                            let p = self.tile_place(dst, at.element(r, c))?;
                            self.store(&p, vals[(r * cols + c) as usize])?;
                        }
                    }
                }
            }
            Stmt::TileFill { d, value } => {
                let v = self.eval_expr(value)?;
                let (rows, cols) = self.tile_shape(*d)?;
                for i in 0..(rows * cols) as usize {
                    self.tile_write(*d, i, v)?;
                }
            }
            Stmt::TileMma { d, a, b, c } => self.tile_mma([*d, *a, *b, *c])?,
            _ => unreachable!("exec_tile is called for a tile statement"),
        }
        Ok(())
    }

    fn tile_shape(&self, tile: u32) -> Result<(u32, u32), KernelTrap> {
        self.alloc_k
            .allocs
            .get(tile as usize)
            .and_then(|a| a.tile)
            .map(|t| (t.rows, t.cols))
            .ok_or(KernelTrap::Unsupported("a tile statement over a non-Tile"))
    }

    /// The row stride and the origin, evaluated.
    fn tile_origin(
        &self,
        per_row: &ScalarExpr,
        origin: &[ScalarExpr; 2],
    ) -> Result<Origin, KernelTrap> {
        Ok(Origin {
            per_row: self.tile_int(per_row)?,
            y: self.tile_int(&origin[0])?,
            x: self.tile_int(&origin[1])?,
        })
    }

    fn tile_int(&self, e: &ScalarExpr) -> Result<i128, KernelTrap> {
        let Ty::Scalar(t) = types::infer_ty_in(e, self.k.compute, &self.local_tys)
            .map_err(|_| KernelTrap::Unsupported("a tile offset type"))?
        else {
            return Err(KernelTrap::Unsupported("a packed tile offset"));
        };
        self.eval_int(e, t)
    }

    /// The place `offset` elements after the one `place` names, as a place with a constant index.
    fn tile_place(&self, place: &Place, offset: i128) -> Result<Place, KernelTrap> {
        let (base, t) = match place {
            Place::Alloc { index, .. } => (index, self.index_ty(index)?),
            Place::Logical { coords, .. } if coords.len() == 1 => {
                (&coords[0], self.index_ty(&coords[0])?)
            }
            _ => {
                return Err(KernelTrap::Unsupported(
                    "a tile place that is not an allocation or a one-index buffer place",
                ));
            }
        };
        let at = ScalarExpr::ConstInt((self.eval_int(base, t)? + offset) as i64, t);
        Ok(match place {
            Place::Alloc { alloc, .. } => Place::Alloc {
                alloc: *alloc,
                index: at,
            },
            Place::Logical { buffer, .. } => Place::Logical {
                buffer: *buffer,
                coords: vec![at],
            },
            _ => unreachable!("matched above"),
        })
    }

    fn index_ty(&self, e: &ScalarExpr) -> Result<ElementKind, KernelTrap> {
        match types::infer_ty_in(e, self.k.compute, &self.local_tys) {
            Ok(Ty::Scalar(t)) => Ok(t),
            _ => Err(KernelTrap::Unsupported("a tile place index type")),
        }
    }

    /// Set element `i` of the tile `tile`.
    fn tile_write(&mut self, tile: u32, i: usize, v: Val) -> Result<(), KernelTrap> {
        let c = self
            .ctx
            .as_mut()
            .ok_or(KernelTrap::Unsupported("a tile outside run_launch"))?;
        let m = c
            .mem_mut(tile, MemorySpace::Tile)
            .ok_or(KernelTrap::Unsupported("a tile with no storage"))?;
        store_val(v, m.dtype, &mut m.bytes, i);
        m.init[i] = true;
        Ok(())
    }

    /// `D = A * B + C` over the tiles `[d, a, b, c]`.
    fn tile_mma(&mut self, [d, a, b, c]: [u32; 4]) -> Result<(), KernelTrap> {
        let (m, k) = self.tile_shape(a)?;
        let (_, n) = self.tile_shape(b)?;
        let (av, bv, cv) = (
            self.private_all(a)?,
            self.private_all(b)?,
            self.private_all(c)?,
        );
        let mut out = Vec::with_capacity((m * n) as usize);
        for i in 0..m as usize {
            for j in 0..n as usize {
                let sum = (0..k as usize).fold(cv[i * n as usize + j].f64(), |s, l| {
                    s + av[i * k as usize + l].f64() * bv[l * n as usize + j].f64()
                });
                // the one rounding to the accumulator type is the store's (`F32`) or none (`F64`)
                out.push(Val::Float(sum));
            }
        }
        for (i, v) in out.into_iter().enumerate() {
            self.tile_write(d, i, v)?;
        }
        Ok(())
    }
}

/// A tile's evaluated row stride and origin.
struct Origin {
    per_row: i128,
    y: i128,
    x: i128,
}

impl Origin {
    /// The element offset of tile element `(r, c)`.
    fn element(&self, r: u32, c: u32) -> i128 {
        (self.y + i128::from(r)) * self.per_row + self.x + i128::from(c)
    }
}
