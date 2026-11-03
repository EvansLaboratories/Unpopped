// SPDX-License-Identifier: MIT OR Apache-2.0
//! **What a collective computes** (ROADMAP U5 P5, `docs/options-u5-collectives-and-richer-reductions.md`): the
//! per-lane results of one rendezvous. The driver (`kernel_launch`) stops every thread of the scope at the
//! statement, each having contributed its value, and calls [`results`] once all have arrived.
//!
//! The rules, for `n` lanes in ascending order (a lane's index is its position in the scope):
//!
//! - **Shuffle** `Idx`/`Broadcast`: lane `i` gets the value of lane `operand mod n`. `Up`: lane `i - operand` if it
//!   exists, else its own value. `Down`: lane `i + operand` if it exists, else its own. `Xor`: lane `i ^ operand` if
//!   it exists, else its own. (CUDA's out-of-range rule: a lane with no source keeps its value.)
//! - **Vote** `All`/`Any`: `1` or `0` (a `U32`) for "every / some lane's value is non-zero" (a NaN is non-zero).
//!   `Ballot`: a `U32` whose bit `i` is lane `i`'s predicate; more than 32 lanes is a typed trap.
//! - **Reduce** `op`: every lane gets the left fold of the values in lane order. **Scan**: lane `i` gets the fold of
//!   lanes `0..=i` (`0..i` when exclusive; lane 0 then gets the identity of `Sum`/`And`/`Or`/`Xor`). Float `Sum` is
//!   the left fold in lane order, which a device may reassociate (compare within a tolerance, not bytes).
//!
//! A collective does **not** order memory: it is a rendezvous that exchanges registers, not a fence, so accesses on
//! either side of it still race unless a barrier orders them.

use super::kernel_eval::{KernelTrap, Machine, wrap};
use super::{Val, int_val};
use crate::ir::ReduceOp;
use crate::ir::ScalarExpr;
use crate::kernel::{CollectiveKind, ShuffleMode, VoteKind};
use crate::types;
use crate::types::Ty;
use unpopped_vocab::ElementKind;

/// A collective with its expressions evaluated away: what the driver needs once every lane has arrived.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(super) enum Plan {
    Shuffle(ShuffleMode),
    Vote(VoteKind),
    Reduce(ReduceOp),
    Scan(ReduceOp, bool),
    Broadcast,
}

/// One lane's contribution: its value, the operand of a shuffle or broadcast, and where the result goes.
#[derive(Clone, Copy, Debug)]
pub(super) struct Contribution {
    pub(super) plan: Plan,
    /// The type of `value`.
    pub(super) ty: ElementKind,
    pub(super) value: Val,
    pub(super) operand: i128,
    /// The local that receives the lane's result.
    pub(super) result: u32,
}

/// Every lane's result, in lane order.
pub(super) fn results(lanes: &[Contribution]) -> Result<Vec<Val>, KernelTrap> {
    let n = lanes.len() as i128;
    let (plan, ty) = (lanes[0].plan, lanes[0].ty);
    Ok(match plan {
        Plan::Shuffle(mode) => (0..n)
            .map(|i| lanes[source(mode, i, lanes[i as usize].operand, n) as usize].value)
            .collect(),
        Plan::Broadcast => (0..n)
            .map(|i| lanes[lanes[i as usize].operand.rem_euclid(n) as usize].value)
            .collect(),
        Plan::Vote(kind) => vote(kind, lanes)?,
        Plan::Reduce(op) => {
            let all = fold(op, ty, lanes.iter().map(|l| l.value))?;
            vec![all; lanes.len()]
        }
        Plan::Scan(op, exclusive) => scan(op, ty, exclusive, lanes)?,
    })
}

/// The lane a shuffle of `mode` reads for lane `i` (itself when there is no such lane).
fn source(mode: ShuffleMode, i: i128, operand: i128, n: i128) -> i128 {
    let want = match mode {
        ShuffleMode::Idx => return operand.rem_euclid(n),
        ShuffleMode::Up => i - operand,
        ShuffleMode::Down => i + operand,
        _ => i ^ operand,
    };
    if (0..n).contains(&want) { want } else { i }
}

fn truthy(v: Val, ty: ElementKind) -> bool {
    if types::int_range(ty).is_some() {
        v.i128() != 0
    } else {
        v.f64() != 0.0
    }
}

fn vote(kind: VoteKind, lanes: &[Contribution]) -> Result<Vec<Val>, KernelTrap> {
    let flags: Vec<bool> = lanes.iter().map(|l| truthy(l.value, l.ty)).collect();
    let u32v = |v: i128| int_val(v, ElementKind::U32);
    let one = match kind {
        VoteKind::All => u32v(i128::from(flags.iter().all(|f| *f))),
        VoteKind::Any => u32v(i128::from(flags.iter().any(|f| *f))),
        _ => {
            if flags.len() > 32 {
                return Err(KernelTrap::Unsupported("a ballot over more than 32 lanes"));
            }
            u32v(
                flags
                    .iter()
                    .enumerate()
                    .fold(0, |m, (i, f)| m | (i128::from(*f) << i)),
            )
        }
    };
    Ok(vec![one; lanes.len()])
}

fn scan(
    op: ReduceOp,
    ty: ElementKind,
    exclusive: bool,
    lanes: &[Contribution],
) -> Result<Vec<Val>, KernelTrap> {
    let mut out = Vec::with_capacity(lanes.len());
    let mut acc: Option<Val> = None;
    for l in lanes {
        let before = acc;
        acc = Some(match acc {
            None => l.value,
            Some(a) => combine(op, ty, a, l.value)?,
        });
        out.push(if exclusive {
            before.map_or_else(|| identity(op, ty), Ok)?
        } else {
            acc.expect("just set")
        });
    }
    Ok(out)
}

fn fold(op: ReduceOp, ty: ElementKind, vals: impl Iterator<Item = Val>) -> Result<Val, KernelTrap> {
    let mut vals = vals;
    let first = vals.next().expect("a scope has at least one lane");
    vals.try_fold(first, |a, b| combine(op, ty, a, b))
}

/// The identity an exclusive scan hands lane 0: only `Sum`, `And`, `Or` and `Xor` have one.
fn identity(op: ReduceOp, ty: ElementKind) -> Result<Val, KernelTrap> {
    let int = types::int_range(ty).is_some();
    Ok(match (op, int) {
        (ReduceOp::Sum | ReduceOp::Or | ReduceOp::Xor, true) => int_val(0, ty),
        (ReduceOp::Sum, false) => Val::Float(0.0),
        (ReduceOp::And, true) => int_val(wrap(ty, -1), ty),
        _ => {
            return Err(KernelTrap::Unsupported(
                "an exclusive scan with no identity",
            ));
        }
    })
}

/// `a op b` at type `ty`: an integer wraps; a float rounds once to `ty`; `Max`/`Min` keep the first of equal values
/// and propagate NaN, as the reductions do.
fn combine(op: ReduceOp, ty: ElementKind, a: Val, b: Val) -> Result<Val, KernelTrap> {
    if types::int_range(ty).is_some() {
        let (x, y) = (a.i128(), b.i128());
        let r = match op {
            ReduceOp::Sum => wrap(ty, x + y),
            ReduceOp::Max => x.max(y),
            ReduceOp::Min => x.min(y),
            ReduceOp::And => x & y,
            ReduceOp::Or => x | y,
            ReduceOp::Xor => x ^ y,
            _ => {
                return Err(KernelTrap::Unsupported(
                    "a collective combine outside the v1 set",
                ));
            }
        };
        return Ok(int_val(r, ty));
    }
    let (x, y) = (a.f64(), b.f64());
    let r = match op {
        ReduceOp::Sum => x + y,
        ReduceOp::Max => nan_first(x, y, y > x),
        ReduceOp::Min => nan_first(x, y, y < x),
        _ => {
            return Err(KernelTrap::Unsupported(
                "a collective combine outside the v1 set",
            ));
        }
    };
    Ok(Val::Float(if ty == ElementKind::F32 {
        f64::from(r as f32)
    } else {
        r
    }))
}

/// `Max`/`Min` on floats: a NaN is kept once seen; otherwise the strictly better one replaces.
fn nan_first(x: f64, y: f64, y_better: bool) -> f64 {
    if x.is_nan() {
        x
    } else if y.is_nan() || y_better {
        y
    } else {
        x
    }
}

impl Machine<'_> {
    /// This thread's contribution to the collective `kind`: `value` and the operand evaluated, the type of `value`
    /// resolved. (The checker has already typed everything, so an odd type is a bug and a typed trap.)
    pub(super) fn contribution(
        &self,
        kind: &CollectiveKind,
        value: &ScalarExpr,
        result: u32,
    ) -> Result<Contribution, KernelTrap> {
        let Ty::Scalar(ty) = types::infer_ty_in(value, self.k.compute, &self.local_tys)
            .map_err(|_| KernelTrap::Unsupported("ill-typed collective value"))?
        else {
            return Err(KernelTrap::Unsupported("a packed collective value"));
        };
        let (plan, operand) = match kind {
            CollectiveKind::Shuffle { mode, operand } => {
                (Plan::Shuffle(*mode), self.eval_expr(operand)?.i128())
            }
            CollectiveKind::Broadcast { lane } => (Plan::Broadcast, self.eval_expr(lane)?.i128()),
            CollectiveKind::Vote(v) => (Plan::Vote(*v), 0),
            CollectiveKind::Reduce(op) => (Plan::Reduce(*op), 0),
            CollectiveKind::Scan { op, exclusive } => (Plan::Scan(*op, *exclusive), 0),
        };
        Ok(Contribution {
            plan,
            ty,
            value: self.eval_expr(value)?,
            operand,
            result,
        })
    }
}
