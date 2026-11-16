// SPDX-License-Identifier: MIT OR Apache-2.0
//! **What a `Stmt::MatrixLoad` computes** (ROADMAP U8b, `docs/options-u8-copy-atoms.md` S2). Each lane that supplies a row
//! (lane `8 * t + r` supplies row `r` of tile `t`) reads its 8 contiguous `Shared` elements at the statement, as ordinary
//! logged loads (so a race with another lane's store is the usual `DataRace`), and hands them over. When all 32 lanes have
//! arrived, [`distribute`] gives lane `l` its `2 * count` elements: for tile `t`, without `trans`, row `l / 4`, columns
//! `2 * (l % 4)` and `+ 1`; with `trans`, rows `2 * (l % 4)` and `+ 1`, column `l / 4`.

use super::Val;
use super::kernel_eval::{KernelTrap, Machine};
use crate::exec::ExecId;
use crate::ir::ScalarExpr;
use crate::kernel::{Place, Stmt};
use crate::types::{self, Ty};

/// Lanes in the warp a matrix load is a rendezvous of.
pub(super) const WARP: usize = 32;

/// One lane's hand-over: its row (when it supplies one) and where its fragment goes.
#[derive(Clone, Debug)]
pub(super) struct MatrixContribution {
    pub(super) count: u32,
    pub(super) trans: bool,
    /// The 8 elements of this lane's row, `None` for a lane that supplies none.
    pub(super) row: Option<[Val; 8]>,
    pub(super) dst: u32,
}

/// Every lane's fragment (`2 * count` elements), in lane order.
pub(super) fn distribute(lanes: &[MatrixContribution]) -> Result<Vec<Vec<Val>>, KernelTrap> {
    if lanes.len() != WARP {
        return Err(KernelTrap::Unsupported(
            "a matrix load over a subgroup that is not 32 lanes",
        ));
    }
    let count = lanes[0].count as usize;
    let trans = lanes[0].trans;
    let element = |t: usize, row: usize, col: usize| -> Result<Val, KernelTrap> {
        lanes[8 * t + row]
            .row
            .as_ref()
            .map(|r| r[col])
            .ok_or(KernelTrap::Unsupported("a matrix row no lane supplied"))
    };
    (0..WARP)
        .map(|l| {
            let mut frag = Vec::with_capacity(2 * count);
            for t in 0..count {
                for e in 0..2 {
                    let (row, col) = if trans {
                        (2 * (l % 4) + e, l / 4)
                    } else {
                        (l / 4, 2 * (l % 4) + e)
                    };
                    frag.push(element(t, row, col)?);
                }
            }
            Ok(frag)
        })
        .collect()
}

impl Machine<'_> {
    /// This lane's hand-over for the `Stmt::MatrixLoad` `st`: a lane below `8 * count` reads its row now.
    pub(super) fn matrix_contribution(
        &mut self,
        st: &Stmt,
    ) -> Result<MatrixContribution, KernelTrap> {
        let Stmt::MatrixLoad {
            count,
            trans,
            src,
            dst,
        } = st
        else {
            unreachable!("matrix_contribution is called for a MatrixLoad");
        };
        let lane = self.exec_val(ExecId::LaneId)?;
        let row = if lane < 8 * i128::from(*count) {
            let Place::Alloc { alloc, index } = src else {
                return Err(KernelTrap::Unsupported(
                    "a matrix load source that is not an allocation place",
                ));
            };
            let Ty::Scalar(t) = types::infer_ty_in(index, self.k.compute, &self.local_tys)
                .map_err(|_| KernelTrap::Unsupported("matrix load index type"))?
            else {
                return Err(KernelTrap::Unsupported("a packed matrix load index"));
            };
            let first = self.eval_int(index, t)?;
            let mut vals = Vec::with_capacity(8);
            for j in 0..8i128 {
                let at = ScalarExpr::ConstInt((first + j) as i64, t);
                vals.push(self.alloc_load(*alloc, &at)?);
            }
            Some(vals.try_into().expect("eight elements"))
        } else {
            None
        };
        Ok(MatrixContribution {
            count: *count,
            trans: *trans,
            row,
            dst: *dst,
        })
    }
}
