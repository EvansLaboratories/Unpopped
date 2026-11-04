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
use crate::mma::{ExtraRole, MmaAtom};
use crate::types;
use unpopped_vocab::ElementKind;

/// One lane's hand-over: its A, B and C fragments (index = value slot) and where its D fragment goes.
#[derive(Clone, Debug)]
pub(super) struct MmaContribution {
    pub(super) atom: MmaAtom,
    /// A, B, C, then the atom's extra operands.
    pub(super) frags: Vec<Vec<Val>>,
    pub(super) d: u32,
}

const SLOT: KernelTrap = KernelTrap::Unsupported("an MMA fragment slot outside the atom");

/// Whether two fragment values are the same bits (lanes that hold one element of a broadcast operand must agree).
fn same(a: Val, b: Val) -> bool {
    match (a, b) {
        (Val::Float(x), Val::Float(y)) => x.to_bits() == y.to_bits(),
        (Val::Int(x), Val::Int(y)) => x == y,
        (Val::Raw(x, tx), Val::Raw(y, ty)) => x == y && tx == ty,
        _ => false,
    }
}

/// The logical column-major matrix `which` (0 = A, 1 = B, 2 = C, then the extras), gathered from the lanes'
/// fragments. Lanes that hold the same element (a broadcast scale) must hand over the same value.
fn gather(atom: &MmaAtom, which: usize, lanes: &[MmaContribution]) -> Result<Vec<Val>, KernelTrap> {
    let (rows, cols) = atom.dims(which);
    let mut out: Vec<Option<Val>> = vec![None; (rows * cols) as usize];
    for (lane, c) in lanes.iter().enumerate() {
        for (v, val) in c.frags[which].iter().enumerate() {
            let id = atom.id(which, lane as u32, v as u32).ok_or(SLOT)? as usize;
            if out[id].is_some_and(|prev| !same(prev, *val)) {
                return Err(KernelTrap::Unsupported(
                    "lanes that hold one MMA element disagree",
                ));
            }
            out[id] = Some(*val);
        }
    }
    out.into_iter()
        .collect::<Option<Vec<_>>>()
        .ok_or(KernelTrap::Unsupported(
            "an MMA matrix element no lane holds",
        ))
}

/// The scale operands of a block-scaled atom: `sfa` is `m x k/block`, `sfb` is `k/block x n`.
struct Scales<'a> {
    sfa: &'a [Val],
    sfb: &'a [Val],
    block: usize,
}

/// The gathered logical matrices: `a` is `m x k` (expanded, when the atom is sparse), `b` is `k x n`, `c` is
/// `m x n`, all column-major.
struct Matrices<'a> {
    a: &'a [Val],
    b: &'a [Val],
    c: &'a [Val],
    scales: Option<Scales<'a>>,
    m: usize,
    k: usize,
}

/// The dense `m x k` `A` of a 2:4 sparse atom: group `g` of row `r` keeps its two compressed values at the
/// positions the metadata names, and the other two are zero. A nibble that is not `i0 < i1` traps.
fn expand_sparse(
    (m, k): (usize, usize),
    compressed: &[Val],
    meta: &[Val],
) -> Result<Vec<Val>, KernelTrap> {
    let zero = match compressed.first() {
        Some(Val::Float(_)) => Val::Float(0.0),
        _ => Val::Int(0),
    };
    let mut full = vec![zero; m * k];
    for g in 0..k / 4 {
        for r in 0..m {
            let nib = meta[r + m * g].i128() & 0xF;
            let (i0, i1) = ((nib & 3) as usize, (nib >> 2) as usize);
            if i0 >= i1 {
                return Err(KernelTrap::Unsupported(
                    "a 2:4 metadata entry whose positions are not ascending",
                ));
            }
            full[r + m * (4 * g + i0)] = compressed[r + m * (2 * g)];
            full[r + m * (4 * g + i1)] = compressed[r + m * (2 * g + 1)];
        }
    }
    Ok(full)
}

impl Matrices<'_> {
    /// `D[row, col]` in the accumulator type `acc`.
    fn element(&self, (row, col): (usize, usize), acc: ElementKind) -> Val {
        let Self { a, b, c, m, k, .. } = *self;
        if types::int_range(acc).is_some() {
            let sum: i128 = (0..k)
                .map(|i| a[row + m * i].i128() * b[i + k * col].i128())
                .sum::<i128>()
                + c[row + m * col].i128();
            return int_val(wrap(acc, sum), acc);
        }
        let product = |i: usize| {
            let (x, y) = (a[row + m * i].f64(), b[i + k * col].f64());
            match &self.scales {
                Some(s) => {
                    let (sa, sb) = (
                        s.sfa[row + m * (i / s.block)].f64(),
                        s.sfb[i / s.block + (k / s.block) * col].f64(),
                    );
                    (x * sa) * (y * sb)
                }
                None => x * y,
            }
        };
        let sum = (0..k)
            .map(product)
            .fold(c[row + m * col].f64(), |s, p| s + p);
        Val::Float(if acc == ElementKind::F32 {
            f64::from(sum as f32)
        } else {
            sum
        })
    }
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
    let mut a = gather(atom, 0, lanes)?;
    let b = gather(atom, 1, lanes)?;
    let c = gather(atom, 2, lanes)?;
    if let Some(w) = atom.extra_which(ExtraRole::Metadata) {
        a = expand_sparse((m, k), &a, &gather(atom, w, lanes)?)?;
    }
    let scale_mats = match (
        atom.block(),
        atom.extra_which(ExtraRole::ScaleA),
        atom.extra_which(ExtraRole::ScaleB),
    ) {
        (Some(block), Some(wa), Some(wb)) => Some((
            gather(atom, wa, lanes)?,
            gather(atom, wb, lanes)?,
            block as usize,
        )),
        _ => None,
    };
    let acc = atom.dtypes().2;
    let logical = Matrices {
        a: &a,
        b: &b,
        c: &c,
        scales: scale_mats.as_ref().map(|(sfa, sfb, block)| Scales {
            sfa,
            sfb,
            block: *block,
        }),
        m,
        k,
    };
    let mut d = Vec::with_capacity(m * n);
    for col in 0..n {
        for row in 0..m {
            d.push(logical.element((row, col), acc));
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
        let Stmt::Mma {
            atom,
            a,
            b,
            c,
            d,
            extra,
        } = st
        else {
            unreachable!("mma_contribution is called for an Mma");
        };
        let frags = [a, b, c]
            .into_iter()
            .chain(extra)
            .map(|alloc| self.private_all(*alloc))
            .collect::<Result<Vec<_>, _>>()?;
        Ok(MmaContribution {
            atom: atom.clone(),
            frags,
            d: *d,
        })
    }
}
