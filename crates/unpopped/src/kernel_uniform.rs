// SPDX-License-Identifier: MIT OR Apache-2.0
//! **The barrier-uniformity rule** (decision D-R4-5, `docs/u3-row4-scopes-and-memory-spaces.md`
//! section 4): a `Barrier` is defined only if every thread of its scope reaches it, so the checker
//! rejects one that sits under control that may differ between threads.
//!
//! A conservative taint analysis. A value is *divergent* if it reads an `Exec` leaf that differs
//! between the threads of a barrier's scope (everything but `BlockIdx`, `BlockDim` and `GridDim`,
//! which every thread of a block, and so of a subgroup, reads alike) or a `Load`
//! from global, shared or private memory, or flows from one. The analysis is flow-insensitive per
//! local (a local assigned a divergent value anywhere is divergent everywhere) and iterated to a
//! fixed point, so a value carried round a loop is still seen. A `Barrier` is rejected under an
//! `If` or `While` with a divergent condition, a `For` with a divergent bound, or after a `Break`
//! or `Continue` that sat under such control **in the same loop body** (the poison ends with its loop). It rejects some correct programs and never accepts
//! a divergent barrier; the four known over-rejections are pinned in `tests/kernel_barrier.rs`.
//!
//! **Warp-uniform control.** A second taint, *lane-divergent*, says a value may differ between the lanes of ONE subgroup. A
//! thread index is lane-divergent, but `tid >> k` and `tid / 2^k` with `2^k` equal to the subgroup size the launch STATES
//! (`with_subgroup(N, N)`, one size, not a range) are not, when the block's x extent is a multiple of that size (so a subgroup
//! lies in one row of x): every lane of a subgroup has the same warp number. A **Subgroup-scope** `Collective` is accepted
//! under control that is warp-uniform (`if (warp == 0) { shuffle }`), because only its own subgroup must reach it; a `Block`
//! barrier or collective still needs block-uniform control, and so does anything else that is a rendezvous. `lane == c` is
//! lane-divergent. A target that cannot pin the subgroup size to the launch's stated one declines the launch; the checker takes
//! the stated size as the contract.

use crate::exec::{ExecId, MemorySpace, Scope};
use crate::ir::{BinaryOp, ScalarExpr};
use crate::kernel::{BarrierKind, Block, CallArg, KernelError, KernelFn, Place, Stmt};
use crate::kernel_extern::ExternSig;
use crate::types::operands;
use unpopped_vocab::ElementKind;

/// Reject a `Barrier` that may not be reached by every thread of its scope.
///
/// `barrier_callees[i]` is whether function `i` of the module reaches a barrier: a `Call` of it is a barrier
/// site (it must sit in uniform control and pass uniform scalar arguments, since the callee was checked
/// with its scalar parameters uniform).
///
/// `entry`: nothing calls `k`, so its buffer parameters are distinct storage the launcher bound (see `read_only`).
pub(crate) fn check_barriers(
    k: &KernelFn,
    barrier_callees: &[bool],
    externs: &[ExternSig],
    entry: bool,
) -> Result<(), KernelError> {
    let mut w = Walk {
        k,
        barrier_callees,
        externs,
        entry,
        device: holds_device_barrier(&k.body),
        divergent: vec![false; k.locals.len()],
        lane: vec![false; k.locals.len()],
        tid_like: vec![false; k.locals.len()],
        warp: stated_warp(k),
        lane_ctx: false,
        returned: false,
        changed: false,
        check: false,
    };
    w.find_tid_like();
    loop {
        w.changed = false;
        w.returned = false;
        w.block(&k.body, false, &mut false)?;
        if !w.changed {
            break;
        }
    }
    w.check = true;
    w.returned = false;
    w.block(&k.body, false, &mut false)
}

/// Whether `b` holds a `Barrier` of `Device` or `Cluster` scope: its threads span blocks, so a value read from `BlockIdx`
/// differs between them (the conservative rule for a `Cluster` barrier, whose blocks differ within the cluster).
fn holds_device_barrier(b: &Block) -> bool {
    b.0.iter().any(|st| match st {
        Stmt::Barrier { scope, .. } => {
            matches!(scope, Scope::Device | Scope::Cluster | Scope::System)
        }
        Stmt::If { then, els, .. } => holds_device_barrier(then) || holds_device_barrier(els),
        Stmt::For { body, .. } | Stmt::While { body, .. } => holds_device_barrier(body),
        _ => false,
    })
}

/// The subgroup size the launch states (`min == max`) when a subgroup lies in one x-row of the block (`block.x % N == 0`).
fn stated_warp(k: &KernelFn) -> Option<u32> {
    let l = k.launch.as_ref()?;
    let s = l.subgroup?;
    (s.min == s.max && s.min >= 1 && l.block[0] % s.min == 0).then_some(s.min)
}

struct Walk<'a> {
    k: &'a KernelFn,
    barrier_callees: &'a [bool],
    externs: &'a [ExternSig],
    /// Nothing calls this function: its buffer parameters are distinct storage (`read_only` relies on it).
    entry: bool,
    /// The kernel holds a `Device` barrier, so `BlockIdx` is divergent.
    device: bool,
    divergent: Vec<bool>,
    /// Per local: may differ between the lanes of one subgroup (a subset of `divergent`).
    lane: Vec<bool>,
    /// Per local: always a copy of the thread index along x (defined once, by a `Let` of `ThreadIdx(0)` or a cast of one).
    tid_like: Vec<bool>,
    /// The subgroup size the launch states, when it states exactly one and the block's x extent is a multiple of it: only then
    /// is `tid >> log2(N)` the same for every lane of a subgroup.
    warp: Option<u32>,
    /// Control reaching the statements being walked may differ between the lanes of one subgroup.
    lane_ctx: bool,
    /// A `Return` under lane-divergent control has been walked (in program order, so far): some lanes of a subgroup may have
    /// left, and a `Subgroup` collective after it would see fewer lanes than the oracle runs it over (`CollectiveAfterReturn`).
    returned: bool,
    changed: bool,
    check: bool,
}

impl Walk<'_> {
    /// The local may differ between threads of the block and between lanes of a subgroup.
    fn mark(&mut self, local: u32) {
        self.mark_block(local);
        let l = &mut self.lane[local as usize];
        if !*l {
            *l = true;
            self.changed = true;
        }
    }

    /// The local may differ between threads of the block (not necessarily between lanes of one subgroup).
    fn mark_block(&mut self, local: u32) {
        let d = &mut self.divergent[local as usize];
        if !*d {
            *d = true;
            self.changed = true;
        }
    }

    /// Mark `local` as block-divergent when `block`, and also lane-divergent when `lane`.
    fn mark_by(&mut self, local: u32, block: bool, lane: bool) {
        if lane {
            self.mark(local);
        } else if block {
            self.mark_block(local);
        }
    }

    /// May `e` differ between threads of the block?
    fn expr(&self, e: &ScalarExpr) -> bool {
        self.expr_at(e, false)
    }

    /// May `e` differ between the lanes of one subgroup?
    fn expr_lane(&self, e: &ScalarExpr) -> bool {
        self.expr_at(e, true)
    }

    fn expr_at(&self, e: &ScalarExpr, lane: bool) -> bool {
        match e {
            // the block's index in its cluster (rank or per-axis id) is, like `BlockIdx`, the same for every thread of a block
            ScalarExpr::Exec(
                ExecId::BlockIdx(_) | ExecId::ClusterCtaRank | ExecId::ClusterCtaId(_),
            ) => self.device,
            // every lane of a subgroup is in the same subgroup, and the warpgroup is the subgroup index divided by four: a
            // function of the subgroup, so the same for every lane of one
            ScalarExpr::Exec(ExecId::SubgroupId | ExecId::WarpgroupId) if lane => false,
            // When the launch states one subgroup size and the block's x extent is a multiple of it, a subgroup lies in one
            // (y, z) row of the block, so `threadIdx.y` and `.z` are the same for every lane of a subgroup.
            ScalarExpr::Exec(ExecId::ThreadIdx(1 | 2)) if lane && self.warp.is_some() => false,
            ScalarExpr::Exec(id) => !matches!(
                id,
                ExecId::BlockDim(_)
                    | ExecId::GridDim(_)
                    | ExecId::DeviceCount
                    | ExecId::ClusterDim(_)
            ),
            ScalarExpr::Local(i) => {
                if lane {
                    self.lane[*i as usize]
                } else {
                    self.divergent[*i as usize]
                }
            }
            _ if lane && self.warp_number(e) => false,
            _ => operands(e).into_iter().any(|x| self.expr_at(x, lane)),
        }
    }

    /// Is `e` the thread index along x, or a copy of it (an integer cast, or a local that only ever holds it)?
    fn is_tid(&self, e: &ScalarExpr) -> bool {
        match e {
            ScalarExpr::Exec(ExecId::ThreadIdx(0)) => true,
            ScalarExpr::Cast(inner, spec) => {
                matches!(
                    spec.to,
                    ElementKind::I32 | ElementKind::U32 | ElementKind::I64 | ElementKind::U64
                ) && self.is_tid(inner)
            }
            ScalarExpr::Local(i) => self.tid_like[*i as usize],
            _ => false,
        }
    }

    /// Is `e` the number of the thread's subgroup: `tid >> k` or `tid / n` with `n == 2^k` the subgroup size the launch states
    /// (so every lane of a subgroup computes the same value)? A shift or divisor of any other size, and a launch that states
    /// no one size, are not.
    fn warp_number(&self, e: &ScalarExpr) -> bool {
        let Some(n) = self.warp else { return false };
        match e {
            ScalarExpr::Binary(BinaryOp::Shr, a, b) => {
                matches!(const_int(b), Some(k) if (0..32).contains(&k) && 1u64 << k == u64::from(n))
                    && self.is_tid(a)
            }
            ScalarExpr::Div(a, b) => const_int(b) == Some(i64::from(n)) && self.is_tid(a),
            _ => false,
        }
    }

    /// The locals that only ever hold the thread index along x: defined by exactly one `Let` whose initialiser is
    /// [`Walk::is_tid`], and by nothing else. The least fixed point, so a copy of a copy counts.
    fn find_tid_like(&mut self) {
        if self.warp.is_none() {
            return;
        }
        let n = self.k.locals.len();
        let mut defs = vec![0u32; n];
        let mut lets: Vec<Option<&ScalarExpr>> = vec![None; n];
        collect_defs(&self.k.body, &mut defs, &mut lets);
        loop {
            let mut grew = false;
            for l in 0..n {
                if !self.tid_like[l] && defs[l] == 1 && lets[l].is_some_and(|e| self.is_tid(e)) {
                    self.tid_like[l] = true;
                    grew = true;
                }
            }
            if !grew {
                break;
            }
        }
    }

    /// Whether buffer parameter `buffer` is declared `Read` (the checker refuses a `Store` to it) **and** is distinct storage
    /// from every parameter this function writes, so it never changes during the launch. The distinctness is the launcher's
    /// contract for an entry function (the oracle passes distinct buffers; `BufferGuarantee::restrict` is an emitted kernel's
    /// statement of it): a function reached through a `Call` may receive one buffer under two parameters, so it gets `false`.
    fn read_only(&self, buffer: u32) -> bool {
        self.entry
            && self.k.params.get(buffer as usize).is_some_and(|p| {
                matches!(
                    p.kind,
                    crate::kernel::ParamKind::Buffer {
                        role: crate::kernel::BufferRole::Read,
                        ..
                    }
                )
            })
    }

    /// Whether a value loaded from `place` may differ between threads (`lane`: between the lanes of one subgroup).
    fn load_divergent(&self, place: &Place, lane: bool) -> bool {
        match place {
            // another block's array differs from this block's, so the value may differ between threads
            Place::ClusterPeer { .. }
            | Place::Peer { .. }
            | Place::AllocRaw { .. }
            | Place::Multicast { .. } => true,
            // A buffer the kernel never writes holds one value per element for the whole launch, so every thread reading the
            // same element (an index no thread-dependent value reaches) reads the same value.
            Place::Logical { buffer, coords } => {
                !self.read_only(*buffer) || coords.iter().any(|c| self.expr_at(c, lane))
            }
            Place::Raw {
                buffer,
                byte_offset,
                ..
            } => !self.read_only(*buffer) || self.expr_at(byte_offset, lane),
            Place::Alloc { alloc, index } => {
                let constant = self
                    .k
                    .allocs
                    .get(*alloc as usize)
                    .is_some_and(|a| a.space == MemorySpace::Constant);
                !constant || self.expr_at(index, lane)
            }
        }
    }

    /// Walk `b`. `div`: control reaching `b` may differ between threads. `poisoned`: a `Break` or
    /// `Continue` under such control has been seen, so what follows may be skipped by some threads.
    fn block(&mut self, b: &Block, div: bool, poisoned: &mut bool) -> Result<(), KernelError> {
        for st in &b.0 {
            self.stmt(st, div, poisoned)?;
        }
        Ok(())
    }

    /// A loop body. A `Break` or `Continue` under divergent control late in the body also skips
    /// the start of the next pass, so a body that poisons is walked again as poisoned. The poison belongs to **its loop**: a
    /// `Break` ends only the innermost loop and a `Continue` only its pass, so every thread that left or skipped meets the
    /// statements after the loop, and at loop exit the state is what it was BEFORE the loop (an outer poison stays). A `Return`
    /// is not poison here (the checker has always accepted one before a barrier and the oracle traps a stranded peer).
    fn loop_body(&mut self, b: &Block, div: bool, poisoned: &mut bool) -> Result<(), KernelError> {
        let before = *poisoned;
        let returned_before = self.returned;
        self.block(b, div, poisoned)?;
        // a lane that left (`Break`, `Continue`) or returned late in one pass is also gone at the start of the next
        if (!before && *poisoned) || (!returned_before && self.returned) {
            self.block(b, div, poisoned)?;
        }
        *poisoned = before;
        Ok(())
    }

    /// An `ExternCall`. A pure one's results differ between threads only if an argument does; any other effect may
    /// (a clock, a lane id), so its results are marked. A `sync` call is a barrier site: `div` is whether control
    /// reaching it may differ between threads, and its arguments must be uniform like a call's.
    fn extern_call(
        &mut self,
        decl: u32,
        args: &[ScalarExpr],
        results: &[u32],
        div: bool,
    ) -> Result<(), KernelError> {
        let sig = self.externs.get(decl as usize);
        let divergent_arg = args.iter().any(|a| self.expr(a));
        if sig.is_none_or(|s| !s.pure || divergent_arg) {
            for r in results {
                self.mark(*r);
            }
        }
        if sig.is_some_and(|s| s.sync.is_some()) && self.check && (div || divergent_arg) {
            return Err(KernelError::BarrierDivergent);
        }
        Ok(())
    }

    /// A `Call`. Its result may differ between threads (the conservative mark). A call that reaches a barrier
    /// is a barrier site: `div` is whether control reaching it may differ between threads.
    fn call(
        &mut self,
        callee: u32,
        args: &[CallArg],
        results: &[u32],
        div: bool,
    ) -> Result<(), KernelError> {
        for local in results {
            self.mark(*local);
        }
        let reaches = self
            .barrier_callees
            .get(callee as usize)
            .copied()
            .unwrap_or(false);
        let divergent_arg = args.iter().any(|a| match a {
            CallArg::Scalar(e) => self.expr(e),
            CallArg::Struct(vals) => vals.iter().any(|e| self.expr(e)),
            _ => false,
        });
        if reaches && self.check && (div || divergent_arg) {
            return Err(KernelError::BarrierDivergent);
        }
        Ok(())
    }

    fn stmt(&mut self, st: &Stmt, div: bool, poisoned: &mut bool) -> Result<(), KernelError> {
        match st {
            Stmt::Let { .. }
            | Stmt::Load { .. }
            | Stmt::MaskedLoad { .. }
            | Stmt::Assign { .. }
            | Stmt::AtomicRmw { .. }
            | Stmt::AtomicCas { .. }
            | Stmt::MulticastLoadReduce { .. }
            | Stmt::Collective { .. }
            | Stmt::CollectiveTuple { .. }
            | Stmt::Call { .. }
            | Stmt::CallTuple { .. }
            | Stmt::ExternCall { .. } => self.defines(st, div, *poisoned)?,
            Stmt::If { .. } | Stmt::For { .. } | Stmt::While { .. } => {
                self.control(st, div, poisoned)?;
            }
            Stmt::Store { .. }
            | Stmt::MaskedStore { .. }
            | Stmt::Assume { .. }
            | Stmt::Fence { .. }
            | Stmt::ProxyFence { .. }
            | Stmt::CopyAsync { .. }
            | Stmt::CopyBulk { .. }
            | Stmt::CopyUnpack { .. }
            | Stmt::WgmmaCommit
            | Stmt::WgmmaWait { .. }
            | Stmt::GridDepWait
            | Stmt::GridDepLaunchDependents
            | Stmt::Tcgen05Mma { .. }
            | Stmt::Tcgen05Commit { .. }
            | Stmt::Tcgen05Ld { .. }
            | Stmt::Tcgen05Alloc { .. }
            | Stmt::Tcgen05Dealloc { .. }
            | Stmt::Tcgen05RelinquishPermit
            | Stmt::AsyncCommit
            | Stmt::AsyncWait { .. }
            | Stmt::MbarInit { .. }
            | Stmt::MbarArrive { .. }
            | Stmt::MbarCompleteTx { .. }
            | Stmt::MbarWait { .. } => {}
            // A `Return` under control that may differ between the lanes of a subgroup takes some lanes out. It is no poison
            // for a barrier (the checker has always accepted one before a `Barrier`; the oracle traps a stranded peer), but
            // a `Subgroup` collective after it is refused (`CollectiveAfterReturn`).
            Stmt::Return => self.returned |= self.lane_ctx,
            // an MMA is a rendezvous of its subgroup, like a barrier; it defines no local
            Stmt::Mma { .. }
            | Stmt::Wgmma { .. }
            | Stmt::MatrixLoad { .. }
            | Stmt::MatrixStore { .. }
            | Stmt::TileLoad { .. }
            | Stmt::TileMma { .. }
            | Stmt::TileFill { .. }
            | Stmt::TileStore { .. } => {
                if self.check && (div || *poisoned) {
                    return Err(KernelError::BarrierDivergent);
                }
            }
            Stmt::Break | Stmt::Continue => *poisoned |= div,
            // a named barrier names its participants by arrival, so it may sit under thread-dependent control
            Stmt::Barrier {
                kind: BarrierKind::Named { .. } | BarrierKind::NamedArrive { .. },
                ..
            } => {}
            Stmt::Barrier { .. } => {
                if self.check && (div || *poisoned) {
                    return Err(KernelError::BarrierDivergent);
                }
            }
        }
        Ok(())
    }

    /// The statements that define a local. `div`: control reaching it may differ between threads of the block; `poisoned`: a
    /// `Break` or `Continue` under such control precedes it (so some threads may have skipped it). The same control may or may
    /// not differ between the lanes of one subgroup (`self.lane_ctx`).
    fn defines(&mut self, st: &Stmt, div: bool, poisoned: bool) -> Result<(), KernelError> {
        let skipped = div || poisoned;
        let skipped_lane = self.lane_ctx || poisoned;
        match st {
            Stmt::Let { local, init } => {
                self.mark_by(*local, self.expr(init), self.expr_lane(init));
            }
            Stmt::Load { local, place } => {
                self.mark_by(
                    *local,
                    self.load_divergent(place, false),
                    self.load_divergent(place, true),
                );
            }
            Stmt::MaskedLoad {
                local,
                place,
                mask,
                fill,
            } => {
                let block = self.load_divergent(place, false) || self.expr(mask) || self.expr(fill);
                let lane = self.load_divergent(place, true)
                    || self.expr_lane(mask)
                    || self.expr_lane(fill);
                self.mark_by(*local, block, lane);
            }
            Stmt::Assign { local, value } => {
                self.mark_by(
                    *local,
                    skipped || self.expr(value),
                    skipped_lane || self.expr_lane(value),
                );
            }
            // read at different moments by different threads, so the value may differ between them
            Stmt::MulticastLoadReduce { local, .. } => self.mark(*local),
            Stmt::Call {
                callee,
                args,
                result,
            } => self.call(*callee, args, result.as_slice(), skipped)?,
            Stmt::CallTuple {
                callee,
                args,
                results,
            } => self.call(*callee, args, results, skipped)?,
            Stmt::ExternCall {
                decl,
                args,
                results,
            } => self.extern_call(*decl, args, results, skipped)?,
            // A collective is a rendezvous like a barrier: every lane of its scope must reach it, and a lane's
            // result differs from another's.
            // Only the lanes of a `Subgroup` collective's own subgroup must reach it, so control that is warp-uniform does not
            // skip any of them; every other scope needs block-uniform control.
            Stmt::Collective { result, scope, .. } => {
                self.mark(*result);
                self.collective_site(*scope, skipped, skipped_lane)?;
            }
            Stmt::CollectiveTuple { results, scope, .. } => {
                for r in results {
                    self.mark(*r);
                }
                self.collective_site(*scope, skipped, skipped_lane)?;
            }
            // The old value depends on the order the threads reach the element, so it differs
            // between threads whatever the operands are: a barrier under `if old == 0` is divergent.
            Stmt::AtomicRmw { old, .. } | Stmt::AtomicCas { old, .. } => {
                if let Some(local) = old {
                    self.mark(*local);
                }
            }
            _ => unreachable!("defines is called for local-defining statements only"),
        }
        Ok(())
    }

    /// The rendezvous rule of a `Collective` or `CollectiveTuple` of `scope`: a `Subgroup` one is refused under lane-divergent
    /// control (`skipped_lane`) and after a lane-divergent `Return`; any other scope needs block-uniform control (`skipped`).
    fn collective_site(
        &self,
        scope: Scope,
        skipped: bool,
        skipped_lane: bool,
    ) -> Result<(), KernelError> {
        if !self.check {
            return Ok(());
        }
        if scope == Scope::Subgroup {
            if skipped_lane {
                return Err(KernelError::BarrierDivergent);
            }
            if self.returned {
                return Err(KernelError::CollectiveAfterReturn);
            }
        } else if skipped {
            return Err(KernelError::BarrierDivergent);
        }
        Ok(())
    }

    /// `If`, `For` and `While`: the body runs under control that may differ between threads when the
    /// condition or a bound does.
    fn control(&mut self, st: &Stmt, div: bool, poisoned: &mut bool) -> Result<(), KernelError> {
        match st {
            Stmt::If { cond, then, els } => {
                let d = div || self.expr(cond);
                let saved = self.lane_ctx;
                self.lane_ctx = saved || self.expr_lane(cond);
                let r = self
                    .block(then, d, poisoned)
                    .and_then(|()| self.block(els, d, poisoned));
                self.lane_ctx = saved;
                r?;
            }
            Stmt::For {
                local,
                start,
                end,
                step,
                body,
                ..
            } => {
                let bound = [start, end, step].into_iter().any(|e| self.expr(e));
                let bound_lane = [start, end, step].into_iter().any(|e| self.expr_lane(e));
                self.mark_by(*local, bound, bound_lane);
                let saved = self.lane_ctx;
                self.lane_ctx = saved || bound_lane;
                let r = self.loop_body(body, div || bound, poisoned);
                self.lane_ctx = saved;
                r?;
            }
            Stmt::While { cond, body } => {
                let d = div || self.expr(cond);
                let saved = self.lane_ctx;
                self.lane_ctx = saved || self.expr_lane(cond);
                let r = self.loop_body(body, d, poisoned);
                self.lane_ctx = saved;
                r?;
            }
            _ => unreachable!("control is called for If, For and While only"),
        }
        Ok(())
    }
}

/// The value of an integer constant, possibly under integer casts (`Cast(ConstInt(32, I32), U32)`, how a front end writes
/// `threadIdx.x / 32` with an `unsigned` thread index).
fn const_int(e: &ScalarExpr) -> Option<i64> {
    match e {
        ScalarExpr::ConstInt(v, _) => Some(*v),
        ScalarExpr::Cast(inner, spec)
            if matches!(
                spec.to,
                ElementKind::I32 | ElementKind::U32 | ElementKind::I64 | ElementKind::U64
            ) =>
        {
            const_int(inner)
        }
        _ => None,
    }
}

/// Count the statements that define each local, and keep the initialiser of the `Let` that defines one.
fn collect_defs<'a>(b: &'a Block, defs: &mut [u32], lets: &mut [Option<&'a ScalarExpr>]) {
    let one = |l: u32, defs: &mut [u32]| defs[l as usize] += 1;
    for st in &b.0 {
        match st {
            Stmt::Let { local, init } => {
                one(*local, defs);
                lets[*local as usize] = Some(init);
            }
            Stmt::Load { local, .. }
            | Stmt::MaskedLoad { local, .. }
            | Stmt::Assign { local, .. }
            | Stmt::MulticastLoadReduce { local, .. } => one(*local, defs),
            Stmt::Collective { result, .. } => one(*result, defs),
            Stmt::CollectiveTuple { results, .. }
            | Stmt::CallTuple { results, .. }
            | Stmt::ExternCall { results, .. } => results.iter().for_each(|r| one(*r, defs)),
            Stmt::Call { result, .. } => result.as_slice().iter().for_each(|r| one(*r, defs)),
            Stmt::AtomicRmw { old, .. } | Stmt::AtomicCas { old, .. } => {
                if let Some(l) = old {
                    one(*l, defs);
                }
            }
            Stmt::If { then, els, .. } => {
                collect_defs(then, defs, lets);
                collect_defs(els, defs, lets);
            }
            Stmt::For { local, body, .. } => {
                one(*local, defs);
                collect_defs(body, defs, lets);
            }
            Stmt::While { body, .. } => collect_defs(body, defs, lets),
            _ => {}
        }
    }
}
