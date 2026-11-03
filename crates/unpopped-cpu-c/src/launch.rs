// SPDX-License-Identifier: MIT OR Apache-2.0
//! The serialized thread loop: a launched [`KernelFn`] spelled as serial C (ROADMAP U4 / the I3 oracle
//! promise of `docs/ir-union-design.md`; PM-approved 2026-11-02). A reference, never a performance claim.
//!
//! # Shape of the C
//!
//! ```text
//! int NAME(args) {
//!     T l0[NT]; T l1[NT]; ...                    // every local, one element per thread of a block
//!     for (bz) for (by) for (bx) {               // the blocks, one after another
//!         T sh0[N] = {0};                        // each Shared allocation, per block
//!         for (int t = 0; t < NT; ++t) { ... }   // a run of barrier-free statements, thread by thread
//!         // Barrier: the thread loop ends here, the next run starts a new one
//!         for (int t = 0; t < NT; ++t) { ... }
//!     }
//!     return 0;
//! }
//! ```
//!
//! A statement that holds a `Barrier` is **block level**: an `If` or a counted `For` whose condition or
//! bounds `check_kernel` proved uniform across the block, so they are read at thread 0 and the construct is
//! spelled once, around the thread loops of its body. The counter of such a `For` is copied into every
//! thread's element each pass.
//!
//! # What is refused (typed, [`LowerError::UnsupportedKernel`])
//!
//! A `Barrier` of a scope other than `Block`, except a `Device` barrier at the top level of the body (below); a `While` that holds a barrier; a `Break` or `Continue` that
//! leaves a loop holding a barrier; a `Return` in a kernel that has a barrier (ending one thread would
//! have to skip the barriers after it); execution ids other than `ThreadIdx`, `BlockIdx`, `BlockDim` and
//! `GridDim`; a `Private` or `Constant` allocation or a dynamic extent; a block over 1024 threads;
//! `Fence`; and a `Call` of a function that reaches a barrier (a callee runs inside one thread's run, so the
//! thread loop cannot be split in it: `lower_module` refuses it).
//!
//! # `Device` barriers (0.42.14)
//!
//! A top-level `Device` barrier cuts the body into phases of the whole grid. Every block is resident, so each local
//! and each `Shared` array becomes an array of the grid (`l0_all`, `sh0_all`), and each phase is one pass over the
//! blocks (`for blk2 { for blk1 { for blk0 { l0 = &l0_all[bl*NT]; ... } } }`) running that phase's block-level code.
//! Blocks ascend within a phase, as in the oracle. A grid of more than 65536 resident threads is refused.
//!
//! # Order
//!
//! Blocks ascend (`x` fastest) and threads ascend (`x` fastest), each thread running to the next
//! barrier before the next thread starts. That is one valid order for atomics; the IR does not promise
//! it (`docs/ir-kernel-layer.md` 5a), so a test of an old value asserts properties, not this order.

use unpopped::backend::{LowerError, Spelling};
use unpopped::exec::{AllocDecl, ExecId, Extent, MemorySpace, Scope};
use unpopped::ir::ScalarExpr;
use unpopped::kernel::{AtomicOp, KernelFn, ParamKind, Place, Stmt};
use unpopped_vocab::ElementKind;

use crate::kernel::{Emitter, ctype, refuse};

/// A block with more threads than this is refused: its per-thread local arrays live on the C stack.
const MAX_THREADS: u32 = 1024;

/// Blocks times threads that may be resident at once across a `Device` barrier.
const MAX_RESIDENT: u64 = 1 << 16;

/// The launch geometry an [`Emitter`] spells ids from.
#[derive(Clone, Copy)]
pub(crate) struct LaunchInfo {
    pub(crate) block: [u32; 3],
    pub(crate) grid: [u32; 3],
    /// The subgroup size a `Subgroup` collective groups by: the launch's required minimum (the oracle's default),
    /// or `1` when the launch states none.
    pub(crate) subgroup: u32,
}

impl LaunchInfo {
    pub(crate) fn of(k: &KernelFn) -> Option<Self> {
        k.launch.as_ref().map(|l| Self {
            block: l.block,
            grid: l.grid,
            subgroup: l.subgroup.map_or(1, |s| s.min.max(1)),
        })
    }

    pub(crate) fn threads(self) -> u32 {
        self.block.iter().product()
    }
}

/// Everything about a launched `k` this emitter cannot spell, refused before any text is written.
pub(crate) fn admit_launch(k: &KernelFn) -> Result<(), LowerError> {
    let info = LaunchInfo::of(k).expect("admit_launch is called for a launched kernel");
    if info.threads() == 0 || info.threads() > MAX_THREADS {
        return Err(refuse(format!(
            "a block of {} threads (1 to {MAX_THREADS} are spelled)",
            info.threads()
        )));
    }
    for a in &k.allocs {
        admit_alloc(a)?;
    }
    let barriers = has_barrier_block(&k.body.0);
    let segments = device_segments(&k.body.0);
    if segments.len() > 1 {
        // blocks stay resident across a Device barrier: their locals and Shared memory are per block
        let resident = u64::from(info.grid.iter().product::<u32>()) * u64::from(info.threads());
        if resident > MAX_RESIDENT {
            return Err(refuse(format!(
                "{resident} resident threads across a Device barrier (at most {MAX_RESIDENT} are spelled)"
            )));
        }
    }
    segments.iter().try_for_each(|s| admit_stmts(s, barriers))
}

/// The kernel body cut at its top-level `Device` and `Cluster` barriers (a `Cluster` barrier is spelled as a barrier of the
/// whole grid, a valid and stronger schedule since clusters are independent): the runs of statements between them, each a
/// phase of the whole grid. A `Device` barrier anywhere else stays in its run, where `admit_stmts` refuses it.
fn device_segments(body: &[Stmt]) -> Vec<&[Stmt]> {
    let mut out = Vec::new();
    let mut start = 0;
    for (i, s) in body.iter().enumerate() {
        if matches!(
            s,
            Stmt::Barrier {
                scope: Scope::Device | Scope::Cluster,
                ..
            }
        ) {
            out.push(&body[start..i]);
            start = i + 1;
        }
    }
    out.push(&body[start..]);
    out
}

fn admit_alloc(a: &AllocDecl) -> Result<(), LowerError> {
    if a.space != MemorySpace::Shared {
        return Err(refuse(format!(
            "allocation `{}` is {:?}; only Shared is spelled",
            a.name, a.space
        )));
    }
    if !matches!(a.extent, Extent::Static(_)) {
        return Err(refuse(format!(
            "allocation `{}` has a dynamic extent",
            a.name
        )));
    }
    ctype(a.elem, &format!("allocation `{}`", a.name)).map(|_| ())
}

fn has_barrier_block(b: &[Stmt]) -> bool {
    b.iter().any(contains_barrier)
}

fn has_return(b: &[Stmt]) -> bool {
    b.iter().any(|s| match s {
        Stmt::Return => true,
        Stmt::If { then, els, .. } => has_return(&then.0) || has_return(&els.0),
        Stmt::For { body, .. } | Stmt::While { body, .. } => has_return(&body.0),
        _ => false,
    })
}

/// Whether `s` is a barrier or holds one.
fn contains_barrier(s: &Stmt) -> bool {
    match s {
        // a collective is a rendezvous like a barrier: the thread loop ends before it and a new one starts after
        Stmt::Barrier { .. } | Stmt::Collective { .. } => true,
        Stmt::If { then, els, .. } => has_barrier_block(&then.0) || has_barrier_block(&els.0),
        Stmt::For { body, .. } | Stmt::While { body, .. } => has_barrier_block(&body.0),
        _ => false,
    }
}

/// The statements a launched kernel may hold, given whether it has any barrier.
fn admit_stmts(b: &[Stmt], barriers: bool) -> Result<(), LowerError> {
    for s in b {
        match s {
            Stmt::Barrier { scope, .. } if *scope != Scope::Block => {
                return Err(refuse(format!(
                    "a {scope:?} Barrier; only Block is spelled"
                )));
            }
            Stmt::Return if barriers => {
                return Err(refuse("a Return in a kernel that has a Barrier"));
            }
            Stmt::While { body, .. } if has_barrier_block(&body.0) => {
                return Err(refuse("a While loop that holds a Barrier"));
            }
            Stmt::If { then, els, .. } => {
                admit_stmts(&then.0, barriers)?;
                admit_stmts(&els.0, barriers)?;
            }
            Stmt::For { body, .. } | Stmt::While { body, .. } => admit_stmts(&body.0, barriers)?,
            _ => {}
        }
    }
    Ok(())
}

/// The `local` seam: a plain variable in a serial kernel, this thread's array element in a launch.
pub(crate) fn local_c(launch: Option<LaunchInfo>, i: u32) -> Result<Spelling, LowerError> {
    Ok(Spelling::Spelled(if launch.is_some() {
        format!("l{i}[t]")
    } else {
        format!("l{i}")
    }))
}

/// The `exec` seam: an execution id has a meaning only in a launch. `launch` is the kernel's own, the
/// thread loop's variables; `ids` is set in a callee under a launch, whose ids are its arguments.
pub(crate) fn exec_spelling(
    (launch, ids): (Option<LaunchInfo>, Option<LaunchInfo>),
    id: ExecId,
) -> Result<Spelling, LowerError> {
    match (ids, launch) {
        (Some(info), _) => exec_param(info, id),
        (None, Some(info)) => exec_c(info, id).map(Spelling::Spelled),
        (None, None) => Err(refuse("an execution id without a launch shape")),
    }
}

/// The C spelling of an execution id inside a callee, whose thread and block ids are its arguments
/// `ti0..2` and `bi0..2` (the sizes are the entry's launch, constants).
pub(crate) fn exec_param(info: LaunchInfo, id: ExecId) -> Result<Spelling, LowerError> {
    let s = match id {
        ExecId::ThreadIdx(a @ 0..=2) => format!("ti{a}"),
        ExecId::BlockIdx(a @ 0..=2) => format!("bi{a}"),
        ExecId::BlockDim(a @ 0..=2) => format!("{}u", info.block[a as usize]),
        ExecId::GridDim(a @ 0..=2) => format!("{}u", info.grid[a as usize]),
        other => return Err(refuse(format!("the execution id {other:?}"))),
    };
    Ok(Spelling::Spelled(s))
}

/// The C spelling of an execution id inside a thread run (`t` is the thread's linear index in its block).
pub(crate) fn exec_c(info: LaunchInfo, id: ExecId) -> Result<String, LowerError> {
    let [bx, by, _] = info.block;
    Ok(match id {
        ExecId::ThreadIdx(0) => format!("((unsigned int)(t % {bx}))"),
        ExecId::ThreadIdx(1) => format!("((unsigned int)((t / {bx}) % {by}))"),
        ExecId::ThreadIdx(2) => format!("((unsigned int)(t / {}))", bx * by),
        ExecId::BlockIdx(a @ 0..=2) => format!("((unsigned int)blk{a})"),
        ExecId::BlockDim(a @ 0..=2) => format!("{}u", info.block[a as usize]),
        ExecId::GridDim(a @ 0..=2) => format!("{}u", info.grid[a as usize]),
        other => return Err(refuse(format!("the execution id {other:?}"))),
    })
}

impl Emitter<'_> {
    /// The launched function body: the local arrays, the scalar parameters, then the blocks.
    pub(crate) fn launched_body(&mut self, info: LaunchInfo) -> Result<(), LowerError> {
        let body = &self.k.body;
        let segments = device_segments(&body.0);
        if segments.len() > 1 {
            return self.device_body(info, &segments);
        }
        let nt = info.threads();
        for (i, d) in self.k.locals.iter().enumerate() {
            let t = ctype(d.elem, &format!("local {i}"))?;
            self.line(&format!("{t} l{i}[{nt}];"));
        }
        for (i, p) in self.k.params.iter().enumerate() {
            if let ParamKind::Scalar { ty, local } = &p.kind {
                let t = ctype(*ty, "a scalar")?;
                self.line(&format!(
                    "for (int t = 0; t < {nt}; ++t) l{local}[t] = ({t})p{i};"
                ));
            }
        }
        for d in ["z", "y", "x"] {
            let a = match d {
                "x" => 0,
                "y" => 1,
                _ => 2,
            };
            self.line(&format!(
                "for (long long blk{a} = 0; blk{a} < {}; ++blk{a}) {{",
                info.grid[a]
            ));
            self.depth += 1;
        }
        self.declare_shared()?;
        let body = &self.k.body;
        self.phases(&body.0)?;
        for _ in 0..3 {
            self.depth -= 1;
            self.line("}");
        }
        Ok(())
    }

    /// A kernel with `Device` barriers: one pass over all the blocks per phase. Every block's locals and `Shared`
    /// memory live in arrays of the whole grid (`l<i>_all`, `sh<i>_all`) and each block loop points `l<i>` and `sh<i>`
    /// at its own slice, so the thread code is the one a single block uses. Blocks ascend within a phase, as in the
    /// oracle, which runs each block to the barrier before the next starts.
    fn device_body(&mut self, info: LaunchInfo, segments: &[&[Stmt]]) -> Result<(), LowerError> {
        self.declare_grid_storage(info)?;
        for (si, segment) in segments.iter().enumerate() {
            self.line(&format!(
                "// phase {si}: every block, up to the next Device barrier"
            ));
            self.open_block_view(info, si == 0)?;
            self.phases(segment)?;
            for _ in 0..3 {
                self.depth -= 1;
                self.line("}");
            }
        }
        Ok(())
    }

    /// The arrays of the whole grid: every block's locals and `Shared` memory.
    fn declare_grid_storage(&mut self, info: LaunchInfo) -> Result<(), LowerError> {
        let nt = info.threads();
        let nb: u32 = info.grid.iter().product();
        for (i, d) in self.k.locals.iter().enumerate() {
            let t = ctype(d.elem, &format!("local {i}"))?;
            self.line(&format!("{t} l{i}_all[{}];", nb * nt));
        }
        for (i, a) in self.k.allocs.iter().enumerate() {
            let Extent::Static(n) = a.extent else {
                return Err(refuse("a dynamic extent"));
            };
            let t = ctype(a.elem, "an allocation")?;
            self.line(&format!(
                "{t} sh{i}_all[{}] = {{0}};",
                u64::from(nb) * u64::from(n)
            ));
        }
        Ok(())
    }

    /// Open the three block loops of a phase and point `l<i>` and `sh<i>` at the block's slices; the first phase also
    /// copies the scalar parameters into the locals.
    fn open_block_view(&mut self, info: LaunchInfo, first: bool) -> Result<(), LowerError> {
        let nt = info.threads();
        for a in [2usize, 1, 0] {
            self.line(&format!(
                "for (long long blk{a} = 0; blk{a} < {}; ++blk{a}) {{",
                info.grid[a]
            ));
            self.depth += 1;
        }
        let [gx, gy, _] = info.grid;
        self.line(&format!(
            "const long long bl = (blk2 * {gy} + blk1) * {gx} + blk0; (void)bl;"
        ));
        for (i, d) in self.k.locals.iter().enumerate() {
            let t = ctype(d.elem, &format!("local {i}"))?;
            self.line(&format!("{t} *l{i} = &l{i}_all[bl * {nt}];"));
        }
        self.view_shared_of_block()?;
        if first {
            for (i, p) in self.k.params.iter().enumerate() {
                if let ParamKind::Scalar { ty, local } = &p.kind {
                    let t = ctype(*ty, "a scalar")?;
                    self.line(&format!(
                        "for (int t = 0; t < {nt}; ++t) l{local}[t] = ({t})p{i};"
                    ));
                }
            }
        }
        Ok(())
    }

    /// Point `sh<i>` at this block's slice of `sh<i>_all`.
    fn view_shared_of_block(&mut self) -> Result<(), LowerError> {
        for (i, a) in self.k.allocs.iter().enumerate() {
            let Extent::Static(n) = a.extent else {
                return Err(refuse("a dynamic extent"));
            };
            let t = ctype(a.elem, "an allocation")?;
            self.line(&format!("{t} *sh{i} = &sh{i}_all[bl * {n}];"));
            if self.module.is_some() {
                self.line(&format!(
                    "const long long ash{i}[1] = {{{n}}}; const long long ast{i}[1] = {{1}};"
                ));
            }
        }
        Ok(())
    }

    fn declare_shared(&mut self) -> Result<(), LowerError> {
        for (i, a) in self.k.allocs.iter().enumerate() {
            let Extent::Static(n) = a.extent else {
                return Err(refuse("a dynamic extent"));
            };
            let t = ctype(a.elem, "an allocation")?;
            self.line(&format!("{t} sh{i}[{n}] = {{0}};"));
            if self.module.is_some() {
                // the shape and stride a callee's buffer parameter is bound to when the array is passed
                self.line(&format!(
                    "const long long ash{i}[1] = {{{n}}}; const long long ast{i}[1] = {{1}};"
                ));
            }
        }
        Ok(())
    }

    /// Block-level statements: barrier-free runs become thread loops, a barrier ends one, and a
    /// construct holding a barrier is spelled once around its own phases.
    fn phases(&mut self, stmts: &[Stmt]) -> Result<(), LowerError> {
        let mut run: Vec<&Stmt> = Vec::new();
        for s in stmts {
            if !contains_barrier(s) {
                run.push(s);
                continue;
            }
            self.flush_run(&mut run)?;
            match s {
                Stmt::Barrier { .. } => self.line("// barrier: the thread loop ends here"),
                Stmt::Collective { .. } => self.collective(s)?,
                Stmt::If { cond, then, els } => self.block_if(cond, then, els)?,
                Stmt::For {
                    local,
                    start,
                    end,
                    step,
                    body,
                } => self.block_for(*local, [start, end, step], body)?,
                _ => {
                    return Err(refuse(
                        "a statement holding a Barrier this emitter cannot split",
                    ));
                }
            }
        }
        self.flush_run(&mut run)
    }

    /// Spell the pending barrier-free statements as one loop over the block's threads.
    fn flush_run(&mut self, run: &mut Vec<&Stmt>) -> Result<(), LowerError> {
        if run.is_empty() {
            return Ok(());
        }
        let nt = self.launch.expect("a launched emitter").threads();
        self.line(&format!("for (int t = 0; t < {nt}; ++t) {{"));
        self.depth += 1;
        self.thread_loops = 0;
        for s in run.drain(..) {
            self.stmt(s)?;
        }
        self.depth -= 1;
        if has_return(&self.k.body.0) {
            // Only a kernel without a barrier may hold a Return, so there is exactly one run and one label.
            self.line("thread_end: ;");
        }
        self.line("}");
        Ok(())
    }

    /// An `If` whose condition `check_kernel` proved uniform: read at thread 0.
    fn block_if(
        &mut self,
        cond: &ScalarExpr,
        then: &unpopped::kernel::Block,
        els: &unpopped::kernel::Block,
    ) -> Result<(), LowerError> {
        let c = self.expr(cond)?;
        self.line("{");
        self.depth += 1;
        self.line("const int t = 0;");
        self.line(&format!("if (({c}) != 0) {{"));
        self.depth += 1;
        self.phases(&then.0)?;
        self.depth -= 1;
        self.line("} else {");
        self.depth += 1;
        self.phases(&els.0)?;
        self.depth -= 1;
        self.line("}");
        self.depth -= 1;
        self.line("}");
        Ok(())
    }

    /// A counted `For` holding a barrier: bounds at thread 0, the counter copied to every thread.
    fn block_for(
        &mut self,
        local: u32,
        [start, end, step]: [&ScalarExpr; 3],
        body: &unpopped::kernel::Block,
    ) -> Result<(), LowerError> {
        let nt = self.launch.expect("a launched emitter").threads();
        let t = self.local_ctype(local)?;
        let n = self.fresh;
        self.fresh += 1;
        let (s, e, st) = (self.expr(start)?, self.expr(end)?, self.expr(step)?);
        self.line("{");
        self.depth += 1;
        self.line("const int t = 0;");
        self.line(&format!(
            "long long s{n} = (long long)({s}), e{n} = (long long)({e}), t{n} = (long long)({st});"
        ));
        self.line(&format!("if (t{n} == 0) return 2;"));
        self.line(&format!(
            "for (long long k{n} = s{n}; t{n} > 0 ? k{n} < e{n} : k{n} > e{n}; k{n} += t{n}) {{"
        ));
        self.depth += 1;
        self.line(&format!(
            "for (int tt = 0; tt < {nt}; ++tt) l{local}[tt] = ({t})k{n};"
        ));
        self.phases(&body.0)?;
        self.depth -= 1;
        self.line("}");
        self.depth -= 1;
        self.line("}");
        Ok(())
    }

    /// The address of a `Shared` allocation element, bounds checked like a buffer access.
    pub(crate) fn alloc_address(&mut self, place: &Place) -> Result<String, LowerError> {
        let Place::Alloc { alloc, index } = place else {
            return Err(refuse("an address this emitter does not spell"));
        };
        if self.launch.is_none() {
            return Err(refuse("a Place::Alloc without a launch shape"));
        }
        let extent = match self.k.allocs[*alloc as usize].extent {
            Extent::Static(n) => n,
            _ => return Err(refuse("a dynamic extent")),
        };
        let n = self.fresh;
        self.fresh += 1;
        let v = self.expr(index)?;
        self.line(&format!(
            "long long o{n} = (long long)({v}); if (o{n} < 0 || o{n} >= {extent}) return 1;"
        ));
        Ok(format!("sh{alloc}[o{n}]"))
    }

    /// An `AtomicRmw` as a plain read-modify-write: serial, hence atomic. `old` receives the value read.
    pub(crate) fn atomic_rmw(&mut self, s: &Stmt) -> Result<(), LowerError> {
        let Stmt::AtomicRmw {
            op,
            place,
            value,
            old,
            ..
        } = s
        else {
            unreachable!("atomic_rmw is called for an AtomicRmw");
        };
        if self.launch.is_none() {
            return Err(refuse("an AtomicRmw without a launch shape"));
        }
        let elem = self.place_elem(place)?;
        let t = ctype(elem, "an atomic element")?;
        let (at, v) = (self.address(place)?, self.expr(value)?);
        let n = self.fresh;
        self.fresh += 1;
        self.line("{");
        self.depth += 1;
        self.line(&format!("{t} a{n} = ({t})({v}); {t} o{n}v = {at};"));
        let new = atomic_new(*op, elem, &format!("o{n}v"), &format!("a{n}"))?;
        self.line(&format!("{at} = ({t})({new});"));
        self.bind_old(*old, n)?;
        self.depth -= 1;
        self.line("}");
        Ok(())
    }

    /// An `AtomicCas` as a plain compare and store: serial, hence atomic. `old` receives the value found.
    pub(crate) fn atomic_cas(&mut self, s: &Stmt) -> Result<(), LowerError> {
        let Stmt::AtomicCas {
            place,
            expected,
            desired,
            old,
            ..
        } = s
        else {
            unreachable!("atomic_cas is called for an AtomicCas");
        };
        if self.launch.is_none() {
            return Err(refuse("an AtomicCas without a launch shape"));
        }
        let elem = self.place_elem(place)?;
        let t = ctype(elem, "an atomic element")?;
        let (at, e, d) = (
            self.address(place)?,
            self.expr(expected)?,
            self.expr(desired)?,
        );
        let n = self.fresh;
        self.fresh += 1;
        self.line("{");
        self.depth += 1;
        self.line(&format!(
            "{t} e{n} = ({t})({e}); {t} d{n} = ({t})({d}); {t} o{n}v = {at};"
        ));
        self.line(&format!("if (o{n}v == e{n}) {at} = d{n};"));
        self.bind_old(*old, n)?;
        self.depth -= 1;
        self.line("}");
        Ok(())
    }

    /// `local = o{n}v;` when the statement asked for the value it found.
    fn bind_old(&mut self, old: Option<u32>, n: u32) -> Result<(), LowerError> {
        if let Some(local) = old {
            let r = self.local_ref(local);
            let lt = self.local_ctype(local)?;
            self.line(&format!("{r} = ({lt})o{n}v;"));
        }
        Ok(())
    }

    fn place_elem(&self, place: &Place) -> Result<ElementKind, LowerError> {
        match place {
            Place::Logical { buffer, .. } => match &self.k.params[*buffer as usize].kind {
                ParamKind::Buffer { elem, .. } => Ok(*elem),
                _ => Err(refuse("an atomic on a parameter that is not a buffer")),
            },
            Place::Alloc { alloc, .. } => Ok(self.k.allocs[*alloc as usize].elem),
            _ => Err(refuse("a place this emitter does not spell")),
        }
    }
}

/// The new element value `old (op) v` in C, both operands already of the element type.
fn atomic_new(op: AtomicOp, t: ElementKind, old: &str, v: &str) -> Result<String, LowerError> {
    let float = matches!(t, ElementKind::F32 | ElementKind::F64);
    let u = match t {
        ElementKind::I32 | ElementKind::U32 => "unsigned int",
        _ => "unsigned long long",
    };
    Ok(match (op, float) {
        (AtomicOp::Add, true) => format!("{old} + {v}"),
        (AtomicOp::Add, false) => format!("({u}){old} + ({u}){v}"),
        (AtomicOp::Max, false) => format!("{old} > {v} ? {old} : {v}"),
        (AtomicOp::Min, false) => format!("{old} < {v} ? {old} : {v}"),
        (AtomicOp::And, false) => format!("{old} & {v}"),
        (AtomicOp::Or, false) => format!("{old} | {v}"),
        (AtomicOp::Xor, false) => format!("{old} ^ {v}"),
        (AtomicOp::Exchange, _) => v.to_string(),
        (other, _) => return Err(refuse(format!("the atomic combine {other:?} on {t:?}"))),
    })
}
