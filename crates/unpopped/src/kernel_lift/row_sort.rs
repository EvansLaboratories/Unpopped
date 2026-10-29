// SPDX-License-Identifier: MIT OR Apache-2.0
//! The `Access::RowSort` lift: a per-row stable sort of the innermost axis, NaN greatest.
//!
//! The kernel is a **rank sort**, not the oracle's `sort_by`: for each element it counts the
//! elements that come before it under the oracle's total order, and that count is the element's
//! output position. The order is `(key, original index)` with NaN greatest, so ranks are distinct
//! and the result is the same unique permutation whatever the algorithm. A descending sort swaps
//! the two keys of the comparison but not the index tiebreak (ties keep ascending original index
//! in both directions). Values move by `Load`/`Store`, which carry raw bits, so NaN payloads and
//! the sign of zero survive exactly as the oracle's raw-bit copy keeps them.
//!
//! The `TopK` cap needs "store only if `rank < k_out`", and `If` conditions are compute-dtype
//! comparisons, so the guard is a loop that runs once or not at all: `for t in rank..min(rank + 1,
//! k_out)`. `I32` keys use the same comparisons; `x != x` is never true for them,
//! so the NaN arms are dead code there.

use unpopped_vocab::ElementKind;

use super::im2col::{add, int, local};
use super::{LiftDecline, check_plain};
use crate::ir::{Access, BinaryOp, OpDef, ScalarExpr, SortLimit, SortOrder, SortOut};
use crate::kernel::{Block, BufferRole, KernelFn, LocalDecl, Param, Place, Stmt};

/// Lift an `Access::RowSort` over `rank` axes (the sorted axis is the innermost), computed at
/// `dtype` (a float or an `I32`). Byte-exact against the oracle for every `SortOut` and `SortLimit`.
///
/// Parameters: the input (`Read`), then the value output (`Write`, input dtype) unless the sort
/// writes indices only, then the `I32` index output (`Write`) unless it writes values only, then
/// one `I32` extent per input axis, then (for `TopK`) the `I32` output width `k_out`. The output
/// has the input's shape except that the last extent is `k_out` for `TopK`.
///
/// # Errors
///
/// A [`LiftDecline`].
pub fn lift_row_sort_opdef(
    op: &OpDef,
    dtype: ElementKind,
    rank: u8,
) -> Result<KernelFn, LiftDecline> {
    let (order, out, limit) = check_row_sort(op, dtype, rank)?;
    let s = Sort::new(dtype, rank, order, out, limit);
    let mut row = s.row_body();
    for d in (0..s.r - 1).rev() {
        row = s.for_lead(d, row);
    }
    let (params, locals) = s.signature();
    Ok(KernelFn::new(op.name.clone(), dtype, params, locals, row))
}

fn check_row_sort(
    op: &OpDef,
    dtype: ElementKind,
    rank: u8,
) -> Result<(SortOrder, SortOut, SortLimit), LiftDecline> {
    let Access::RowSort {
        order,
        stable,
        out,
        limit,
    } = op.access
    else {
        return Err(LiftDecline::NotRowSort);
    };
    // The plan gate admits `I32` among the integers. `I64` is declined although it plans: the oracle
    // orders keys through `f64`, which collapses distinct 64-bit values, so an exact kernel would
    // disagree with it on large keys.
    if !matches!(
        dtype,
        ElementKind::F32 | ElementKind::F64 | ElementKind::I32
    ) {
        return Err(LiftDecline::UnsupportedForm(
            "a row sort outside F32, F64 and I32",
        ));
    }
    if !stable {
        return Err(LiftDecline::UnsupportedForm("an unstable row sort"));
    }
    let want = matches!(out, SortOut::Indices).then_some(ElementKind::I32);
    if op.out_dtype != want {
        return Err(LiftDecline::UnsupportedForm("a row sort's output dtype"));
    }
    let mut plain = op.clone();
    plain.out_dtype = None;
    check_plain(&plain, dtype, rank)?;
    Ok((order, out, limit))
}

/// Local ids. Extents `0..r`; counters `r..2r` (the last is the position `i` along the row);
/// `j_lo`, `j_hi`, `rank`, `t`; `k_out` when capped; then the keys `ki` and `kj`.
struct Sort {
    dtype: ElementKind,
    r: u32,
    order: SortOrder,
    out: SortOut,
    capped: bool,
}

impl Sort {
    fn new(dtype: ElementKind, rank: u8, order: SortOrder, out: SortOut, limit: SortLimit) -> Self {
        Self {
            dtype,
            r: u32::from(rank),
            order,
            out,
            capped: matches!(limit, SortLimit::TopK),
        }
    }

    fn counter(&self, d: u32) -> u32 {
        self.r + d
    }

    fn i(&self) -> u32 {
        self.counter(self.r - 1)
    }

    fn j_lo(&self) -> u32 {
        2 * self.r
    }

    fn j_hi(&self) -> u32 {
        2 * self.r + 1
    }

    fn rank(&self) -> u32 {
        2 * self.r + 2
    }

    fn t(&self) -> u32 {
        2 * self.r + 3
    }

    /// The output width: the capped parameter, else the row length.
    fn k_out(&self) -> u32 {
        if self.capped {
            2 * self.r + 4
        } else {
            self.r - 1
        }
    }

    fn ki(&self) -> u32 {
        2 * self.r + 4 + u32::from(self.capped)
    }

    fn kj(&self) -> u32 {
        self.ki() + 1
    }

    fn writes_values(&self) -> bool {
        !matches!(self.out, SortOut::Indices)
    }

    fn writes_indices(&self) -> bool {
        !matches!(self.out, SortOut::Values)
    }

    /// `for counter(d) in 0..extent(d) { inner }`.
    fn for_lead(&self, d: u32, inner: Block) -> Block {
        Block(vec![self.counted(self.counter(d), int(0), local(d), inner)])
    }

    fn counted(&self, var: u32, start: ScalarExpr, end: ScalarExpr, body: Block) -> Stmt {
        Stmt::For {
            local: var,
            start,
            end,
            step: int(1),
            body,
        }
    }

    /// The place at the leading counters and `last` along the row.
    fn place(&self, buffer: u32, last: u32) -> Place {
        let mut coords: Vec<ScalarExpr> = (0..self.r - 1).map(|d| local(self.counter(d))).collect();
        coords.push(local(last));
        Place::Logical { buffer, coords }
    }

    /// One row: for each position `i`, rank it against the whole row and place it.
    fn row_body(&self) -> Block {
        let last = self.r - 1;
        let body = vec![
            Stmt::Load {
                local: self.ki(),
                place: self.place(0, self.i()),
            },
            Stmt::Let {
                local: self.rank(),
                init: int(0),
            },
            // Elements before `i` win a tie, elements after it do not.
            self.counted(
                self.j_lo(),
                int(0),
                local(self.i()),
                self.compare(self.j_lo(), true),
            ),
            self.counted(
                self.j_hi(),
                add(local(self.i()), int(1)),
                local(last),
                self.compare(self.j_hi(), false),
            ),
            self.place_element(),
        ];
        self.for_lead(last, Block(body))
    }

    /// Load `key[j]` and count it if it precedes `ki`; a tie counts when `tie_counts`.
    fn compare(&self, j: u32, tie_counts: bool) -> Block {
        let (ki, kj) = (local(self.ki()), local(self.kj()));
        // The pair `(a, b)` such that `a` precedes `b` iff `a < b` in this sort's direction.
        let (a, b) = match self.order {
            SortOrder::Asc => (kj, ki),
            SortOrder::Desc => (ki, kj),
        };
        let nan = |x: &ScalarExpr| {
            ScalarExpr::Binary(BinaryOp::CmpNe, Box::new(x.clone()), Box::new(x.clone()))
        };
        let inc = || {
            Block(vec![Stmt::Assign {
                local: self.rank(),
                value: add(local(self.rank()), int(1)),
            }])
        };
        let when =
            |cond: ScalarExpr, then: Block, els: Block| Block(vec![Stmt::If { cond, then, els }]);
        let cmp = if tie_counts {
            BinaryOp::CmpLe
        } else {
            BinaryOp::CmpLt
        };
        let ordered = ScalarExpr::Binary(cmp, Box::new(a.clone()), Box::new(b.clone()));
        // NaN is greatest: a NaN `a` never precedes a number; a NaN `b` is preceded by any number.
        let a_is_nan = if tie_counts {
            when(nan(&b), inc(), Block(vec![]))
        } else {
            Block(vec![])
        };
        let a_is_number = when(nan(&b), inc(), when(ordered, inc(), Block(vec![])));
        let mut body = vec![Stmt::Load {
            local: self.kj(),
            place: self.place(0, j),
        }];
        body.extend(when(nan(&a), a_is_nan, a_is_number).0);
        Block(body)
    }

    /// Store the element at its rank, if the rank is below `k_out`. The guard is a loop that runs
    /// once (`t = rank`) when `rank < k_out` and not at all otherwise.
    fn place_element(&self) -> Stmt {
        let mut stores: Vec<Stmt> = Vec::new();
        let mut buffer = 1;
        if self.writes_values() {
            stores.push(Stmt::Store {
                place: self.place(buffer, self.t()),
                value: local(self.ki()),
            });
            buffer += 1;
        }
        if self.writes_indices() {
            stores.push(Stmt::Store {
                place: self.place(buffer, self.t()),
                value: local(self.i()),
            });
        }
        let end = ScalarExpr::Binary(
            BinaryOp::Min,
            Box::new(add(local(self.rank()), int(1))),
            Box::new(local(self.k_out())),
        );
        self.counted(self.t(), local(self.rank()), end, Block(stores))
    }

    fn signature(&self) -> (Vec<Param>, Vec<LocalDecl>) {
        let rank = u8::try_from(self.r).expect("rank is at most 8");
        let buf = |name: &str, dt, role| Param::buffer(name, dt, role, Some(rank));
        let mut params = vec![buf("in", self.dtype, BufferRole::Read)];
        if self.writes_values() {
            params.push(buf("out_val", self.dtype, BufferRole::Write));
        }
        if self.writes_indices() {
            params.push(buf("out_idx", ElementKind::I32, BufferRole::Write));
        }
        params.extend((0..self.r).map(|d| Param::scalar(format!("n{d}"), ElementKind::I32, d)));
        if self.capped {
            params.push(Param::scalar("k_out", ElementKind::I32, self.k_out()));
        }
        let ints = 2 * self.r + 4 + u32::from(self.capped);
        let mut locals = vec![LocalDecl::scalar(ElementKind::I32); ints as usize];
        locals.push(LocalDecl::scalar(self.dtype)); // ki
        locals.push(LocalDecl::scalar(self.dtype)); // kj
        (params, locals)
    }
}
