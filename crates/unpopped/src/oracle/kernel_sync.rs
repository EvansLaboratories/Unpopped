// SPDX-License-Identifier: MIT OR Apache-2.0
//! **Release/acquire edges for the launch interpreter** (ROADMAP U4 S4; `docs/question-u4-memory-model.md`, option A,
//! ruled by CireSnave 2026-11-03): the race rule of `kernel_mem` classifies a program; this adds the one thing it
//! could not say, that an atomic with a memory order can order the plain accesses around it.
//!
//! Each thread keeps a *knowledge* clock: for every other thread, the latest access of that thread it has acquired
//! from. A release-class atomic (`Release`, `AcqRel`, `SeqCst`) on an element leaves a record on that element: the
//! thread's knowledge plus its own latest access, and the scope it was issued at. An acquire-class atomic
//! (`Acquire`, `AcqRel`, `SeqCst`) on the element joins the records into the reader's knowledge. A plain access `p`
//! of thread `u` is then ordered before a later access of thread `v` when `v` knows `u` up to `p`. A relaxed atomic
//! leaves the record alone (it continues the release sequence, as a read-modify-write does in C++). `SeqCst` is treated as `AcqRel`: the oracle does not model the single total
//! order (the CHANGELOG says so).
//!
//! **Scope.** An edge exists only when both atomics' scopes, the narrower of the two, contain both threads
//! (`Subgroup`: one subgroup of one block; `Block`: one block; `Device`: everything). Two atomics on one element whose
//! narrower scope does not contain both threads are a `DataRace` (the atomicity does not hold across that gap).

use std::collections::HashMap;

use super::kernel_mem::Stamp;
use crate::exec::Scope;
use crate::kernel::MemoryOrder;

/// One thread's identity: block, thread in block, subgroup in block.
type Tid = (u32, u32, u32);

/// For each thread, the latest access (by sequence number) of it that the owner has acquired from.
type Clock = HashMap<Tid, u64>;

/// What a release left on an element: the releaser, the scope it was issued at, and what it knew.
#[derive(Clone)]
struct Record {
    from: Tid,
    scope: Scope,
    clock: Clock,
}

/// An element: kind (0 global buffer, 1 `Shared`), block (shared only), buffer or allocation id, element offset.
pub(super) type ElemKey = (u8, u32, u32, usize);

/// The launch-wide synchronization state.
#[derive(Default)]
pub(super) struct Edges {
    seq: u64,
    know: HashMap<Tid, Clock>,
    records: HashMap<ElemKey, Vec<Record>>,
    /// Whether any acquire-class atomic has run: a block that then runs away may be waiting on a later block.
    pub(super) acquired: bool,
}

pub(super) fn tid(s: Stamp) -> Tid {
    (s.block, s.thread, s.sub)
}

/// How wide a scope is: a thread, a subgroup, a block, a cluster, the device.
pub(super) fn width(s: Scope) -> u8 {
    match s {
        Scope::Thread => 0,
        Scope::Subgroup => 1,
        Scope::Block => 2,
        Scope::Cluster => 3,
        _ => 4,
    }
}

fn narrower(a: Scope, b: Scope) -> Scope {
    if width(a) <= width(b) { a } else { b }
}

/// Whether `scope` contains both threads.
pub(super) fn contains(scope: Scope, a: Tid, b: Tid) -> bool {
    match scope {
        Scope::Thread => a == b,
        Scope::Subgroup => a.0 == b.0 && a.2 == b.2,
        Scope::Block => a.0 == b.0,
        _ => true,
    }
}

fn join(into: &mut Clock, from: &Clock) {
    for (t, s) in from {
        let e = into.entry(*t).or_insert(0);
        *e = (*e).max(*s);
    }
}

fn releases(o: MemoryOrder) -> bool {
    matches!(
        o,
        MemoryOrder::Release | MemoryOrder::AcqRel | MemoryOrder::SeqCst
    )
}

fn acquires(o: MemoryOrder) -> bool {
    matches!(
        o,
        MemoryOrder::Acquire | MemoryOrder::AcqRel | MemoryOrder::SeqCst
    )
}

impl Edges {
    /// The next access sequence number (never zero).
    pub(super) fn next(&mut self) -> u64 {
        self.seq += 1;
        self.seq
    }

    /// Whether `now` has acquired from `prev`'s thread up to `prev`'s access.
    pub(super) fn knows(&self, now: Tid, prev: Stamp) -> bool {
        self.know
            .get(&now)
            .and_then(|c| c.get(&tid(prev)))
            .is_some_and(|s| *s >= prev.seq)
    }

    /// An atomic read-modify-write by `now` on `key`: acquire from the records first, then (if it releases) leave one.
    pub(super) fn atomic(
        &mut self,
        now: Stamp,
        key: ElemKey,
        (order, scope): (MemoryOrder, Scope),
    ) {
        let me = tid(now);
        if acquires(order) {
            self.acquired = true;
            let mine = self.know.entry(me).or_default();
            for r in self.records.get(&key).into_iter().flatten() {
                if contains(narrower(r.scope, scope), r.from, me) {
                    join(mine, &r.clock);
                }
            }
        }
        if releases(order) {
            let mut clock = self.know.get(&me).cloned().unwrap_or_default();
            clock.insert(me, now.seq);
            // a later release by one thread knows at least what its earlier one did, so it replaces it
            let recs = self.records.entry(key).or_default();
            recs.retain(|r| r.from != me);
            recs.push(Record {
                from: me,
                scope,
                clock,
            });
        }
    }
}
