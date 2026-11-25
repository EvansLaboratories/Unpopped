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
use super::accum_policy::to_f16;
use super::float_codec::{f16_to_f64, f32_to_f16_bits};
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

    /// `tcgen05.alloc` (`alloc = true`) or `tcgen05.dealloc` for the `Tensor` allocation `d` with `ncols` columns. The
    /// instruction is issued by a whole warp (`.sync.aligned`): the warp's first lane acts, the others do nothing.
    pub(super) fn tcgen05_alloc(
        &mut self,
        d: u32,
        ncols: u32,
        alloc: bool,
    ) -> Result<(), KernelTrap> {
        let t = self.exec_val(ExecId::ThreadIdx(0))? as usize;
        if t % 32 != 0 {
            return Ok(());
        }
        let declared = self.tmem_columns(d) as u32;
        let Some(c) = self.ctx.as_mut() else {
            return Err(KernelTrap::Unsupported("tensor memory outside run_launch"));
        };
        let misuse = |what| Err(KernelTrap::TmemMisuse { what });
        let tmem = &mut c.block.tmem;
        if alloc {
            if tmem.relinquished {
                return misuse("a tcgen05.alloc after tcgen05.relinquish_alloc_permit");
            }
            if tmem.live.contains_key(&d) {
                return misuse("a tcgen05.alloc of tensor memory that is already allocated");
            }
            if ncols != declared {
                return misuse("a tcgen05.alloc whose nCols is not the declared column count");
            }
            if tmem.last_cols.is_some_and(|last| ncols > last) {
                return misuse("the number of columns allocated increased between two allocations");
            }
            tmem.live.insert(d, ncols);
            tmem.last_cols = Some(ncols);
        } else {
            match tmem.live.get(&d) {
                None => return misuse("a tcgen05.dealloc of tensor memory that is not allocated"),
                Some(&n) if n != ncols => {
                    return misuse("a tcgen05.dealloc whose nCols is not the allocated count");
                }
                Some(_) => {
                    tmem.live.remove(&d);
                }
            }
        }
        Ok(())
    }

    /// `tcgen05.relinquish_alloc_permit` (the warp's first lane acts).
    pub(super) fn tcgen05_relinquish(&mut self) -> Result<(), KernelTrap> {
        let t = self.exec_val(ExecId::ThreadIdx(0))? as usize;
        if t % 32 == 0 {
            if let Some(c) = self.ctx.as_mut() {
                c.block.tmem.relinquished = true;
            }
        }
        Ok(())
    }

    /// The `Tensor` allocation `d` must be allocated (PTX ISA 9.5 9.7.18.1.2).
    fn require_tmem(&self, d: u32) -> Result<(), KernelTrap> {
        match &self.ctx {
            Some(c) if !c.block.tmem.live.contains_key(&d) => Err(KernelTrap::TmemMisuse {
                what: "tensor memory used before tcgen05.alloc or after tcgen05.dealloc",
            }),
            _ => Ok(()),
        }
    }

    /// The `rows` rows of `K` elements of the tile at `place`.
    fn tcgen05_tile(
        &mut self,
        place: &Place,
        rows: usize,
        layout: &crate::kernel::OperandLayout,
        k: usize,
    ) -> Result<Vec<Vec<Val>>, KernelTrap> {
        (0..rows)
            .map(|r| self.wgmma_row(place, r, layout, k))
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
            a_layout,
            b_layout,
            m: rows,
            d_lane,
            ab,
            acc,
            negate_a,
            negate_b,
            ..
        } = st
        else {
            unreachable!("tcgen05_mma is called for a Tcgen05Mma");
        };
        let (n, d) = (*n as usize, *d);
        self.require_tmem(d)?;
        let cols = self.tmem_columns(d);
        let col0 = usize::try_from(self.wgmma_int(d_col)?)
            .ok()
            .filter(|c| c + n <= cols)
            .ok_or(KernelTrap::AllocOutOfBounds { alloc: d })?;
        let accumulate = self.wgmma_int(scale_d)? != 0;
        let half_acc = *acc == unpopped_vocab::ElementKind::F16;
        // negating A's or B's elements flips every product's sign; both cancel
        let sign = if negate_a ^ negate_b { -1.0 } else { 1.0 };
        // `.kind::f8f6f4` takes K = 32 per instruction, the 16-bit kinds 16 (PTX ISA 9.5 Table 50)
        let k = if matches!(
            ab,
            unpopped_vocab::ElementKind::Fp8E4M3FN | unpopped_vocab::ElementKind::Fp8E5M2
        ) {
            32
        } else {
            16
        };
        let (a, b) = (
            self.tcgen05_tile(a, *rows as usize, a_layout, k)?,
            self.tcgen05_tile(b, n, b_layout, k)?,
        );
        for (m, arow) in a.iter().enumerate() {
            for (j, brow) in b.iter().enumerate() {
                // PTX ISA 9.5 Layout D (m = 128): row r is lane r. Layout F (m = 64): rows 16w..16w+15 are in warp w's
                // quadrant, in the half `d_lane` picks (Figures 221, 222)
                let lane = if *rows == 64 {
                    32 * (m / 16) + *d_lane as usize + m % 16
                } else {
                    m
                };
                let at = cell(lane * cols + col0 + j);
                let c = if !accumulate {
                    0.0
                } else if half_acc {
                    // the old D is a word whose low 16 bits are a binary16
                    let w = self.alloc_load(d, &at)?.f64() as u64;
                    f16_to_f64((w & 0xFFFF) as u16)
                } else {
                    self.alloc_load(d, &at)?.f64()
                };
                let sum = arow
                    .iter()
                    .zip(brow)
                    .fold(c, |s, (x, y)| s + sign * (x.f64() * y.f64()));
                let out = if half_acc {
                    let h = to_f16(sum) as f32;
                    Val::Int(i128::from(f32_to_f16_bits(h)))
                } else {
                    Val::Float(f64::from(sum as f32))
                };
                self.alloc_store(d, &at, out)?;
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
        self.require_tmem(src)?;
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
