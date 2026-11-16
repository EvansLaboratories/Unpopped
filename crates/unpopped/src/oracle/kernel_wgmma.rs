// SPDX-License-Identifier: MIT OR Apache-2.0
//! **What a `Stmt::Wgmma` computes** (ROADMAP U7 slice S4, option A: no swizzle, no descriptor bits). The 128 threads of
//! a warpgroup stop at the statement. Thread `t` reads, as ordinary logged `Shared` loads, row `t` of `A` (when `t < 64`)
//! and rows `t` and `t + 128` of `B` (when below `n`), 16 elements each (both tiles are K-major, row-major), and hands
//! them over with its current accumulator fragment (only when `scale_d` is nonzero). When all 128 have arrived,
//! [`compute`] forms `D = A.B + scale_d * C`: every product is exact in `f64`, the sum starts from `C` and takes the 16
//! products in ascending `k`, and it is rounded once to the accumulator type, the same contract as `Stmt::Mma`. Each
//! thread's share of `D` is queued on the thread and written to its `Private` fragment at the `WgmmaWait` that covers
//! its committed group.
//!
//! **Fragment layout** (`wgmma.m64nNk16`, from the PTX ISA description; no ISA text was read in this repository): warp
//! `w` holds rows `16 w ..`, and value `v` of lane `l` is n8 block `v / 4`, row `l / 4 + 8 * ((v % 4) / 2)`, column
//! `2 * (l % 4) + v % 2` within the block.

use super::Val;
use super::kernel_eval::{KernelTrap, Machine};
use crate::exec::ExecId;
use crate::ir::ScalarExpr;
use crate::kernel::{Place, Stmt};
use crate::types::{self, Ty};
use unpopped_vocab::ElementKind;

/// Threads in a warpgroup.
pub(super) const WARPGROUP: usize = 128;
/// Rows of `A` and of `D`.
const M: usize = 64;
/// Elements along `k` for the 16-bit input types (one instruction).
const K: usize = 16;

/// One thread's hand-over.
#[derive(Clone, Debug)]
pub(super) struct WgmmaContribution {
    pub(super) n: usize,
    pub(super) acc: ElementKind,
    /// Row `t` of `A`, for `t < 64`.
    pub(super) a_row: Option<Vec<Val>>,
    /// Rows of `B` this thread read: `(row, elements)`.
    pub(super) b_rows: Vec<(usize, Vec<Val>)>,
    pub(super) scale_d: bool,
    /// The thread's current accumulator fragment (empty when `scale_d` is 0).
    pub(super) c: Vec<Val>,
    /// The `Private` allocation `D` lands in.
    pub(super) d: u32,
}

/// The `(row, column)` of `D` that value `v` of thread `t` (in the warpgroup) holds.
fn position(t: usize, v: usize) -> (usize, usize) {
    let (w, l) = (t / 32, t % 32);
    (
        16 * w + l / 4 + 8 * ((v % 4) / 2),
        8 * (v / 4) + 2 * (l % 4) + v % 2,
    )
}

/// Every thread's `D` fragment, in thread order.
pub(super) fn compute(lanes: &[WgmmaContribution]) -> Result<Vec<Vec<Val>>, KernelTrap> {
    if lanes.len() != WARPGROUP {
        return Err(KernelTrap::Unsupported(
            "a wgmma over a group that is not 128 threads",
        ));
    }
    let (n, acc, scale) = (lanes[0].n, lanes[0].acc, lanes[0].scale_d);
    let missing = KernelTrap::Unsupported("a wgmma tile row no thread read");
    let a: Vec<&Vec<Val>> = (0..M)
        .map(|r| {
            lanes[r]
                .a_row
                .as_ref()
                .ok_or(KernelTrap::Unsupported("a wgmma A row no thread read"))
        })
        .collect::<Result<_, _>>()?;
    let mut b: Vec<Option<&Vec<Val>>> = vec![None; n];
    for l in lanes {
        for (r, row) in &l.b_rows {
            b[*r] = Some(row);
        }
    }
    let b: Vec<&Vec<Val>> = b.into_iter().collect::<Option<_>>().ok_or(missing)?;
    let mut c = vec![0.0f64; M * n];
    if scale {
        for (t, l) in lanes.iter().enumerate() {
            for (v, x) in l.c.iter().enumerate() {
                let (row, col) = position(t, v);
                c[row * n + col] = x.f64();
            }
        }
    }
    let dot = |row: usize, col: usize| -> Val {
        let sum = (0..K).fold(c[row * n + col], |s, k| {
            s + a[row][k].f64() * b[col][k].f64()
        });
        Val::Float(if acc == ElementKind::F32 {
            f64::from(sum as f32)
        } else {
            sum
        })
    };
    Ok((0..WARPGROUP)
        .map(|t| {
            (0..n / 2)
                .map(|v| {
                    let (row, col) = position(t, v);
                    dot(row, col)
                })
                .collect()
        })
        .collect())
}

impl Machine<'_> {
    /// This thread's hand-over for the `Stmt::Wgmma` `st`.
    pub(super) fn wgmma_contribution(
        &mut self,
        st: &Stmt,
    ) -> Result<WgmmaContribution, KernelTrap> {
        let Stmt::Wgmma {
            n,
            acc,
            a,
            b,
            d,
            scale_d,
            ..
        } = st
        else {
            unreachable!("wgmma_contribution is called for a Wgmma");
        };
        let t = self.exec_val(ExecId::ThreadIdx(0))? as usize % WARPGROUP;
        let n = *n as usize;
        let a_row = if t < M {
            Some(self.wgmma_row(a, t)?)
        } else {
            None
        };
        let mut b_rows = Vec::new();
        for row in [t, t + WARPGROUP] {
            if row < n {
                b_rows.push((row, self.wgmma_row(b, row)?));
            }
        }
        let scale_d = self.wgmma_int(scale_d)? != 0;
        let c = if scale_d {
            self.private_all(*d)?
        } else {
            Vec::new()
        };
        Ok(WgmmaContribution {
            n,
            acc: *acc,
            a_row,
            b_rows,
            scale_d,
            c,
            d: *d,
        })
    }

    fn wgmma_int(&mut self, e: &ScalarExpr) -> Result<i128, KernelTrap> {
        let Ty::Scalar(t) = types::infer_ty_in(e, self.k.compute, &self.local_tys)
            .map_err(|_| KernelTrap::Unsupported("a wgmma operand type"))?
        else {
            return Err(KernelTrap::Unsupported("a packed wgmma operand"));
        };
        self.eval_int(e, t)
    }

    /// Row `row` (16 elements) of the tile whose first element is `place`, as logged `Shared` loads.
    fn wgmma_row(&mut self, place: &Place, row: usize) -> Result<Vec<Val>, KernelTrap> {
        let Place::Alloc { alloc, index } = place else {
            return Err(KernelTrap::Unsupported(
                "a wgmma tile that is not an allocation place",
            ));
        };
        let first = self.wgmma_int(index)? + (row * K) as i128;
        (0..K as i128)
            .map(|j| {
                let at = ScalarExpr::ConstInt((first + j) as i64, ElementKind::I64);
                self.alloc_load(*alloc, &at)
            })
            .collect()
    }
}
