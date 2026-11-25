// SPDX-License-Identifier: MIT OR Apache-2.0
//! **What `Stmt::Tcgen05Mma` and `Stmt::Tcgen05Ld` do** (ROADMAP U7 slice S5; no descriptors, no swizzle, no `alloc`). A
//! tensor-memory allocation is `128 * columns` 32-bit cells in the block's array (`MemorySpace::Tensor`, stored beside
//! `Shared`), cell `(lane, column)` at `lane * columns + column`. The issuing thread of a `Tcgen05Mma` reads the `A` tile
//! (128 rows) and the `B` tile (`n` rows) as ordinary logged `Shared` loads, forms `D = A.B + scale_d * D` with the
//! numeric contract of `Stmt::Mma` (exact `f64` products, ascending `k`, one `F32` rounding) and writes `D` into tensor
//! memory as logged writes by that thread; row `m` is lane `m`. A thread that later reads it must have acquired an
//! `mbarrier` the issuer released after the write (`Tcgen05Commit`), or the read is a `DataRace`. `Tcgen05Ld` is the
//! `32x32b` shape: thread `l` of warp `w` reads lane `32 (w % 4) + l`.

use super::Val;
use super::kernel_eval::{KernelTrap, Machine};
use crate::exec::{ExecId, MemorySpace};
use crate::ir::ScalarExpr;
use crate::kernel::{Place, Stmt};
use unpopped_vocab::ElementKind;

/// Lanes of tensor memory, rows of `A` and of `D`.
const M: usize = 128;

fn cell(i: usize) -> ScalarExpr {
    ScalarExpr::ConstInt(i as i64, ElementKind::I64)
}

impl Machine<'_> {
    /// Columns of the tensor allocation `alloc`.
    fn tmem_columns(&self, alloc: u32) -> usize {
        self.ctx
            .as_ref()
            .and_then(|c| c.mem(alloc, MemorySpace::Tensor))
            .map_or(0, |m| m.init.len() / M)
    }

    /// The `rows` rows of `K` elements of the tile at `place`.
    fn tcgen05_tile(&mut self, place: &Place, rows: usize) -> Result<Vec<Vec<Val>>, KernelTrap> {
        (0..rows)
            .map(|r| self.wgmma_row(place, r, &crate::kernel::OperandLayout::RowMajor))
            .collect()
    }

    /// Run a `Tcgen05Mma`: `D` of `n` columns from `d_col` of the tensor allocation `d`.
    pub(super) fn tcgen05_mma(&mut self, st: &Stmt) -> Result<(), KernelTrap> {
        let Stmt::Tcgen05Mma {
            n,
            a,
            b,
            d,
            d_col,
            scale_d,
            ..
        } = st
        else {
            unreachable!("tcgen05_mma is called for a Tcgen05Mma");
        };
        let (n, d) = (*n as usize, *d);
        let cols = self.tmem_columns(d);
        let col0 = usize::try_from(self.wgmma_int(d_col)?)
            .ok()
            .filter(|c| c + n <= cols)
            .ok_or(KernelTrap::AllocOutOfBounds { alloc: d })?;
        let accumulate = self.wgmma_int(scale_d)? != 0;
        let (a, b) = (self.tcgen05_tile(a, M)?, self.tcgen05_tile(b, n)?);
        for (m, arow) in a.iter().enumerate() {
            for (j, brow) in b.iter().enumerate() {
                let at = cell(m * cols + col0 + j);
                let c = if accumulate {
                    self.alloc_load(d, &at)?.f64()
                } else {
                    0.0
                };
                let sum = arow
                    .iter()
                    .zip(brow)
                    .fold(c, |s, (x, y)| s + x.f64() * y.f64());
                self.alloc_store(d, &at, Val::Float(f64::from(sum as f32)))?;
            }
        }
        Ok(())
    }

    /// Run a `Tcgen05Ld`: this thread's lane, `count` columns from `col`, into `dst`.
    pub(super) fn tcgen05_ld(
        &mut self,
        src: u32,
        col: &ScalarExpr,
        count: u32,
        dst: u32,
    ) -> Result<(), KernelTrap> {
        let t = self.exec_val(ExecId::ThreadIdx(0))? as usize;
        let lane = 32 * ((t / 32) % 4) + t % 32;
        let cols = self.tmem_columns(src);
        let col0 = usize::try_from(self.wgmma_int(col)?)
            .ok()
            .filter(|c| c + count as usize <= cols)
            .ok_or(KernelTrap::AllocOutOfBounds { alloc: src })?;
        for j in 0..count as usize {
            let v = self.alloc_load(src, &cell(lane * cols + col0 + j))?;
            self.alloc_store(dst, &cell(j), v)?;
        }
        Ok(())
    }
}
