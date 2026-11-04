// SPDX-License-Identifier: MIT OR Apache-2.0
//! **What a `Stmt::Mma` computes** (ROADMAP U7 S2). The driver stops every lane of the atom's subgroup at the
//! statement, each having handed over its A, B and C fragments, and calls [`compute`] once all have arrived.
//!
//! The lanes' fragments are gathered into the logical column-major matrices through the atom's thread-value
//! layouts, `D = A.B + C` is formed, and each lane's D fragment is read back through the C/D layout. Floats: every
//! product is exact in `f64`, the sum of `C` and the `K` products is taken in `f64` in ascending `k` order and
//! rounded **once** to the accumulator type, so a device that sums in another order or at another width is compared
//! within a tolerance, never by bytes. Integers: exact, wrapping at the accumulator type.

use super::kernel_eval::{KernelTrap, Machine, wrap};
use super::{Val, int_val};
use crate::kernel::Stmt;
use crate::mma::MmaAtom;
use crate::types;
use unpopped_vocab::ElementKind;

/// One lane's hand-over: its A, B and C fragments (index = value slot) and where its D fragment goes.
#[derive(Clone, Debug)]
pub(super) struct MmaContribution {
    pub(super) atom: MmaAtom,
    pub(super) frags: [Vec<Val>; 3],
    pub(super) d: u32,
}

const SLOT: KernelTrap = KernelTrap::Unsupported("an MMA fragment slot outside the atom");

/// The logical column-major matrix `which` (0 = A, 1 = B, 2 = C), gathered from the lanes' fragments.
fn gather(atom: &MmaAtom, which: usize, lanes: &[MmaContribution]) -> Result<Vec<Val>, KernelTrap> {
    let (m, n, k) = atom.mnk();
    let len = [m * k, k * n, m * n][which] as usize;
    let mut out = vec![None; len];
    for (lane, c) in lanes.iter().enumerate() {
        for (v, val) in c.frags[which].iter().enumerate() {
            let id = atom.id(which, lane as u32, v as u32).ok_or(SLOT)?;
            out[id as usize] = Some(*val);
        }
    }
    out.into_iter()
        .collect::<Option<Vec<_>>>()
        .ok_or(KernelTrap::Unsupported(
            "an MMA matrix element no lane holds",
        ))
}

/// `D[row, col]` of the logical matrices `a` (`m x k`), `b` (`k x n`), `c` (`m x n`), all column-major.
fn element(
    (a, b, c): (&[Val], &[Val], &[Val]),
    (m, k): (usize, usize),
    (row, col): (usize, usize),
    acc: ElementKind,
) -> Val {
    if types::int_range(acc).is_some() {
        let sum: i128 = (0..k)
            .map(|i| a[row + m * i].i128() * b[i + k * col].i128())
            .sum::<i128>()
            + c[row + m * col].i128();
        return int_val(wrap(acc, sum), acc);
    }
    let sum = (0..k)
        .map(|i| a[row + m * i].f64() * b[i + k * col].f64())
        .fold(c[row + m * col].f64(), |s, p| s + p);
    Val::Float(if acc == ElementKind::F32 {
        f64::from(sum as f32)
    } else {
        sum
    })
}

/// Every lane's D fragment, in lane order.
pub(super) fn compute(lanes: &[MmaContribution]) -> Result<Vec<Vec<Val>>, KernelTrap> {
    let atom = &lanes[0].atom;
    if lanes.len() != atom.lanes() as usize {
        return Err(KernelTrap::Unsupported(
            "an MMA over a subgroup that is not the atom's lane count",
        ));
    }
    let (m, n, k) = atom.mnk();
    let (m, n, k) = (m as usize, n as usize, k as usize);
    let a = gather(atom, 0, lanes)?;
    let b = gather(atom, 1, lanes)?;
    let c = gather(atom, 2, lanes)?;
    let acc = atom.dtypes().2;
    let mut d = Vec::with_capacity(m * n);
    for col in 0..n {
        for row in 0..m {
            d.push(element((&a, &b, &c), (m, k), (row, col), acc));
        }
    }
    (0..lanes.len())
        .map(|lane| {
            (0..atom.values(2))
                .map(|v| {
                    atom.id(2, lane as u32, v)
                        .map(|id| d[id as usize])
                        .ok_or(SLOT)
                })
                .collect()
        })
        .collect()
}

impl Machine<'_> {
    /// This lane's hand-over for the `Stmt::Mma` `st`: the A, B and C fragments read out of its `Private` arrays
    /// (every element must have been stored).
    pub(super) fn mma_contribution(&self, st: &Stmt) -> Result<MmaContribution, KernelTrap> {
        let Stmt::Mma { atom, a, b, c, d } = st else {
            unreachable!("mma_contribution is called for an Mma");
        };
        Ok(MmaContribution {
            atom: atom.clone(),
            frags: [
                self.private_all(*a)?,
                self.private_all(*b)?,
                self.private_all(*c)?,
            ],
            d: *d,
        })
    }
}
