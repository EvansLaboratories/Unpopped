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
use crate::kernel::{OperandLayout, Place, Stmt};
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
    /// Row `t` of `A`, for `t < 64` (from `Shared`).
    pub(super) a_row: Option<Vec<Val>>,
    /// This thread's `A` fragment (8 values) when `A` comes from registers.
    pub(super) a_frag: Option<Vec<Val>>,
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

/// The `(row, column)` of `A` that value `i` of thread `t` holds when `A` is in registers (PTX ISA 9.5 Figure 151, a page
/// image; cross-checked against the PTX text for `mma.m16n8k16` A and CUTLASS's `GMMA::ALayout_64x16`: see
/// `tests/kernel_wgmma.rs`, `fill_a_fragment`): warp
/// `w` holds rows `16 w ..`; lane `l` holds rows `l / 4` and `l / 4 + 8`, columns `2 * (l % 4)` and `+ 1`, and the same
/// again 8 columns on.
fn a_position(t: usize, i: usize) -> (usize, usize) {
    let (w, l) = (t / 32, t % 32);
    (
        16 * w + l / 4 + 8 * ((i / 2) % 2),
        2 * (l % 4) + i % 2 + 8 * (i / 4),
    )
}

/// The rows of `A` (64) and of `B` (`n`) the threads read, in row order.
#[allow(
    clippy::type_complexity,
    reason = "the pair of row-reference vectors is local to this one helper; a named type would add nothing"
)]
fn tile_rows(lanes: &[WgmmaContribution]) -> Result<(Vec<Vec<Val>>, Vec<&Vec<Val>>), KernelTrap> {
    let a: Vec<Vec<Val>> = if lanes[0].a_frag.is_some() {
        // `A` in registers: scatter every thread's fragment to its (row, column)
        let mut rows: Vec<Vec<Option<Val>>> = vec![vec![None; K]; M];
        for (t, l) in lanes.iter().enumerate() {
            let frag = l.a_frag.as_ref().ok_or(KernelTrap::Unsupported(
                "a wgmma whose threads disagree on where A is",
            ))?;
            for (i, v) in frag.iter().enumerate() {
                let (row, col) = a_position(t, i);
                rows[row][col] = Some(*v);
            }
        }
        rows.into_iter()
            .map(|r| r.into_iter().collect::<Option<Vec<Val>>>())
            .collect::<Option<_>>()
            .ok_or(KernelTrap::Unsupported(
                "a wgmma A element no fragment held",
            ))?
    } else {
        (0..M)
            .map(|r| {
                lanes[r]
                    .a_row
                    .clone()
                    .ok_or(KernelTrap::Unsupported("a wgmma A row no thread read"))
            })
            .collect::<Result<_, _>>()?
    };
    let mut b: Vec<Option<&Vec<Val>>> = vec![None; lanes[0].n];
    for l in lanes {
        for (r, row) in &l.b_rows {
            b[*r] = Some(row);
        }
    }
    let b = b
        .into_iter()
        .collect::<Option<_>>()
        .ok_or(KernelTrap::Unsupported("a wgmma B row no thread read"))?;
    Ok((a, b))
}

/// The accumulator `C` as a row-major `64 x n` matrix: the threads' fragments when `scale_d` is set, otherwise zero.
fn accumulator(lanes: &[WgmmaContribution]) -> Vec<f64> {
    let n = lanes[0].n;
    let mut c = vec![0.0f64; M * n];
    if lanes[0].scale_d {
        for (t, l) in lanes.iter().enumerate() {
            for (v, x) in l.c.iter().enumerate() {
                let (row, col) = position(t, v);
                c[row * n + col] = x.f64();
            }
        }
    }
    c
}

/// Every thread's `D` fragment, in thread order.
pub(super) fn compute(lanes: &[WgmmaContribution]) -> Result<Vec<Vec<Val>>, KernelTrap> {
    if lanes.len() != WARPGROUP {
        return Err(KernelTrap::Unsupported(
            "a wgmma over a group that is not 128 threads",
        ));
    }
    let (n, acc) = (lanes[0].n, lanes[0].acc);
    let (a, b) = tile_rows(lanes)?;
    let c = accumulator(lanes);
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
            a_layout,
            b_layout,
            a_regs,
            ..
        } = st
        else {
            unreachable!("wgmma_contribution is called for a Wgmma");
        };
        let t = self.exec_val(ExecId::ThreadIdx(0))? as usize % WARPGROUP;
        let n = *n as usize;
        let a_frag = match a_regs {
            Some(f) => Some(self.private_all(*f)?),
            None => None,
        };
        let a_row = if a_frag.is_none() && t < M {
            Some(self.wgmma_row(a, t, a_layout)?)
        } else {
            None
        };
        let mut b_rows = Vec::new();
        for row in [t, t + WARPGROUP] {
            if row < n {
                b_rows.push((row, self.wgmma_row(b, row, b_layout)?));
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
            a_frag,
            b_rows,
            scale_d,
            c,
            d: *d,
        })
    }

    pub(super) fn wgmma_int(&mut self, e: &ScalarExpr) -> Result<i128, KernelTrap> {
        let Ty::Scalar(t) = types::infer_ty_in(e, self.k.compute, &self.local_tys)
            .map_err(|_| KernelTrap::Unsupported("a wgmma operand type"))?
        else {
            return Err(KernelTrap::Unsupported("a packed wgmma operand"));
        };
        self.eval_int(e, t)
    }

    /// Row `row` (16 elements) of the tile whose first element is `place`, as logged `Shared` loads, read through
    /// `layout` (element `k` of the row is at `layout.element_offset(row, k)` from the tile's first element).
    pub(super) fn wgmma_row(
        &mut self,
        place: &Place,
        row: usize,
        layout: &OperandLayout,
    ) -> Result<Vec<Val>, KernelTrap> {
        let Place::Alloc { alloc, index } = place else {
            return Err(KernelTrap::Unsupported(
                "a wgmma tile that is not an allocation place",
            ));
        };
        let first = self.wgmma_int(index)?;
        (0..K)
            .map(|j| {
                let off = layout.element_offset(row, j).ok_or(KernelTrap::Unsupported(
                    "a wgmma operand layout the oracle does not read (a swizzle, or no leading offset)",
                ))?;
                let at_i = first + off as i128;
                let at = ScalarExpr::ConstInt(at_i as i64, ElementKind::I64);
                let v = self.alloc_load(*alloc, &at)?;
                // the async proxy reads this tile: a generic write to it needs this writer's `ProxyFence` first
                self.require_proxy_fenced(*alloc, at_i as usize)?;
                Ok(v)
            })
            .collect()
    }
}
