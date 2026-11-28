// SPDX-License-Identifier: MIT OR Apache-2.0
//! **The kernel-layer interpreter** (`docs/ir-kernel-layer.md` section 5): runs a
//! [`KernelFn`] on one logical thread, in plain Rust, with no emitter and no device.
//!
//! It exists so that "the layers agree" (`lift_opdef` of an `OpDef` computes what
//! `oracle::evaluate` computes) is a test that runs anywhere. It reuses the oracle's own
//! evaluator for floating-point expressions, so a lifted body is evaluated by the *same code*
//! that evaluates the original body: the comparison is bit-for-bit by construction, not by
//! approximation.
//!
//! # How expressions are evaluated
//!
//! * An **integer** expression (loop bounds, coordinates, integer conditions) goes to a small
//!   evaluator here: `ConstInt`, `Local`, `Add`, `Sub`, `Mul`, `Min`, `Max`, wrapped to the
//!   type's width exactly as `docs/ir-typed-expressions.md` section 8.
//! * A **compute-dtype** expression is rewritten (`Local(i)` becomes `Input(slot)`) and handed to
//!   the oracle's evaluator, with the locals as the leaf values. This is what the oracle does to
//!   an `OpDef` body, with `Input(k)` read from a buffer; here it is read from a local.
//! * Anything else (a packed value, an integer operation outside the subset above, a float
//!   expression that reads a non-compute local) is [`KernelTrap::Unsupported`], never a guess.
//!
//! # Rounding: once per STATEMENT, at f64, not once per operation
//!
//! A float expression is evaluated at `f64` and rounded to the compute dtype **once, where it is stored** (`settle` for a local, the
//! element store for a buffer): `a*b - c*d` computes both products exactly. This is a higher-precision REFERENCE that a device is
//! compared against inside a fidelity band (`oracle/fidelity.rs`), not an interpreter of C semantics: it is neither C's unfused
//! per-operation `float` arithmetic nor a literal fused multiply-add. A kernel whose result depends on per-operation rounding
//! (a cancellation such as `cuCmulf((x, 1 + 2^-11), (x, 1)).re` with `x = 1 + 2^-12`: `2^-24` here and in a fused kernel, `0` in
//! unfused `f32`) is therefore NOT discriminated by value, only by the shape of the IR (`tests/kernel_body_complex.rs` pins
//! the shape). A per-operation mode for bit-exact device tests is ROADMAP U45.

use super::kernel_f128::scalar_float;
use super::kernel_mem::{AccessKind, ThreadCtx};
pub use super::kernel_trap::KernelTrap;
use super::wide_ops::{
    as_wide, wide_add, wide_binary, wide_cast, wide_compare, wide_div, wide_mul, wide_sub,
};
use super::{
    Eval, TypedBuffer, Val, eval, int_val, load_val, panic_reduced, raw_to_i128, read_elem,
    store_val,
};
use crate::ir::{BinaryOp, CastSpec, Saturation, ScalarExpr, UnaryOp};
use crate::kernel::{AtomicOp, Block, KernelFn, ParamKind, Place, Stmt, check_kernel};
use crate::types::{self, Ty, map_children};
use crate::wide_int::W256;
use unpopped_vocab::ElementKind;

/// A buffer map entry at or above this is not a root argument but the entry's allocation `entry - ALLOC_BIND`:
/// a callee's buffer parameter bound to a `Shared` or `Private` array passed as a `CallArg::Alloc`.
pub(super) const ALLOC_BIND: usize = 1 << 40;

/// Statements the interpreter will execute before declaring a run runaway. It bounds the
/// *interpreter* (so a test of a bad program terminates); it is not part of the language.
pub const STEP_LIMIT: u64 = 10_000_000;

/// One argument, aligned with [`KernelFn::params`].
#[derive(Clone, Debug)]
#[non_exhaustive]
pub enum KernelArg {
    /// A buffer parameter's storage. Outputs must be passed zeroed or pre-filled by the caller.
    Buffer(TypedBuffer),
    /// An integer scalar parameter.
    Int(i128),
    /// A floating-point scalar parameter.
    Float(f64),
    /// A struct parameter's field values, in field order (`Int` or `Float` each).
    Struct(Vec<KernelArg>),
}

pub(super) enum Flow {
    Next,
    Break,
    Continue,
    Return,
}

pub(super) struct Machine<'a> {
    pub(super) k: &'a KernelFn,
    pub(super) local_tys: Vec<Ty>,
    pub(super) locals: Vec<Option<Val>>,
    pub(super) args: &'a mut [KernelArg],
    /// The module a `Call` resolves in (`run_module` only).
    pub(super) module: Option<&'a crate::kernel::Module>,
    /// Inside a callee: the root argument each buffer parameter is bound to. Empty in the entry frame.
    pub(super) bufmap: Vec<usize>,
    /// The index of the running function in `module` (the entry's, outside a call).
    pub(super) fn_idx: u32,
    /// The function whose `allocs` an allocation index names: always the entry, since only it declares any.
    pub(super) alloc_k: &'a KernelFn,
    pub(super) steps: u64,
    /// The statements a thread may execute before [`KernelTrap::StepLimit`] ([`STEP_LIMIT`] unless a launch lowers it).
    pub(super) limit: u64,
    /// Present only under `run_launch`: which thread this is, and the memory it shares.
    pub(super) ctx: Option<ThreadCtx<'a>>,
}

/// Run `k` on `args` (one per parameter, in order). Buffers are updated in place.
///
/// # Errors
///
/// A [`KernelTrap`].
pub fn run_kernel(k: &KernelFn, args: &mut [KernelArg]) -> Result<(), KernelTrap> {
    check_kernel(k).map_err(KernelTrap::Malformed)?;
    if k.launch.is_some() || !k.allocs.is_empty() {
        return Err(KernelTrap::Unsupported(
            "a launched or allocating kernel runs under run_launch, not run_kernel",
        ));
    }
    if args.len() != k.params.len() {
        return Err(KernelTrap::ArgMismatch("one argument per parameter"));
    }
    Machine::new(k, args, None).run_body()
}

impl<'a> Machine<'a> {
    pub(super) fn new(
        k: &'a KernelFn,
        args: &'a mut [KernelArg],
        ctx: Option<ThreadCtx<'a>>,
    ) -> Self {
        Self {
            k,
            local_tys: k.locals.iter().map(|d| d.ty()).collect(),
            locals: vec![None; k.locals.len()],
            args,
            module: None,
            bufmap: Vec::new(),
            fn_idx: 0,
            alloc_k: k,
            steps: 0,
            limit: STEP_LIMIT,
            ctx,
        }
    }

    /// The same machine, resolving calls in `m`, running function `entry` of it.
    pub(super) fn with_module(mut self, m: &'a crate::kernel::Module, entry: u32) -> Self {
        self.module = Some(m);
        self.fn_idx = entry;
        self
    }

    /// The entry's allocation a buffer parameter of the running function is bound to, with the index
    /// expression of an access through it (`Place::Logical` of rank 1 on an allocation-bound parameter).
    pub(super) fn alloc_bound<'p>(&self, place: &'p Place) -> Option<(u32, &'p ScalarExpr)> {
        let Place::Logical { buffer, coords } = place else {
            return None;
        };
        let root = self.root(*buffer);
        match (root.checked_sub(ALLOC_BIND), coords.as_slice()) {
            (Some(a), [index]) => Some((u32::try_from(a).ok()?, index)),
            _ => None,
        }
    }

    /// The root argument a buffer parameter of the running function names (an allocation binding is
    /// `ALLOC_BIND + index`).
    pub(super) fn root(&self, buffer: u32) -> usize {
        self.bufmap
            .get(buffer as usize)
            .copied()
            .unwrap_or(buffer as usize + self.dev_base())
    }

    /// Where this device's arguments start in `args`: a multi-device launch lays the devices' arguments end to end.
    pub(super) fn dev_base(&self) -> usize {
        self.ctx
            .as_ref()
            .map_or(0, |c| c.device as usize * c.stride)
    }

    /// The root argument of the copy that device `device` (an integer expression) holds of the buffer whose root on
    /// this device is `root` (a `Place::Peer`).
    fn peer_root(&mut self, root: usize, device: &ScalarExpr) -> Result<usize, KernelTrap> {
        let Ty::Scalar(t) = types::infer_ty_in(device, self.k.compute, &self.local_tys)
            .map_err(|_| KernelTrap::Unsupported("device rank type"))?
        else {
            return Err(KernelTrap::Unsupported("packed device rank"));
        };
        let rank = self.eval_int(device, t)?;
        let (base, stride, devices) = match &self.ctx {
            Some(c) => (self.dev_base(), c.stride, c.launch.devices),
            None => return Err(KernelTrap::Unsupported("a peer access outside run_launch")),
        };
        if rank < 0 || rank >= i128::from(devices) {
            return Err(KernelTrap::DeviceRankOutOfRange { rank });
        }
        if root < base || root >= base + stride {
            return Err(KernelTrap::Unsupported(
                "a peer place on a buffer bound to an allocation",
            ));
        }
        Ok(root - base + usize::try_from(rank).unwrap_or(0) * stride)
    }

    /// Every device's copy of the element a `Place::Multicast` names: `(root argument, element offset)` per device, in
    /// rank order, each bounds-checked against that device's own copy (ROADMAP U12 slice 3).
    fn multicast_targets(&mut self, place: &Place) -> Result<Vec<(usize, usize)>, KernelTrap> {
        let Place::Multicast { buffer, coords } = place else {
            unreachable!("multicast_targets is called for a Place::Multicast");
        };
        let mut cs = Vec::with_capacity(coords.len());
        for c in coords {
            let Ty::Scalar(t) = types::infer_ty_in(c, self.k.compute, &self.local_tys)
                .map_err(|_| KernelTrap::Unsupported("coordinate type"))?
            else {
                return Err(KernelTrap::Unsupported("packed coordinate"));
            };
            cs.push(self.eval_int(c, t)?);
        }
        let (stride, devices) = match &self.ctx {
            Some(c) => (c.stride, c.launch.devices),
            None => {
                return Err(KernelTrap::Unsupported(
                    "a multicast access outside run_launch",
                ));
            }
        };
        let root = self.root(*buffer);
        let base = self.dev_base();
        if root < base || root >= base + stride {
            return Err(KernelTrap::Unsupported(
                "a multicast place on a buffer bound to an allocation",
            ));
        }
        let first = root - base;
        let mut out = Vec::new();
        for d in 0..usize::try_from(devices).unwrap_or(0) {
            let at = first + d * stride;
            let KernelArg::Buffer(tb) = &self.args[at] else {
                return Err(KernelTrap::ArgMismatch(
                    "a Place names a non-buffer argument",
                ));
            };
            let mut off = i128::from(tb.base_offset);
            for (a, &c) in cs.iter().enumerate() {
                if c < 0 || c >= i128::from(tb.shape[a]) {
                    return Err(KernelTrap::OutOfBounds { buffer: *buffer });
                }
                off += c * i128::from(tb.strides[a]);
            }
            let off =
                usize::try_from(off).map_err(|_| KernelTrap::OutOfBounds { buffer: *buffer })?;
            out.push((at, off));
        }
        Ok(out)
    }

    /// Bind the arguments and run the body to completion (one logical thread).
    pub(super) fn run_body(&mut self) -> Result<(), KernelTrap> {
        self.bind_args()?;
        let k = self.k;
        self.exec_block(&k.body).map(|_| ())
    }

    pub(super) fn bind_args(&mut self) -> Result<(), KernelTrap> {
        let base = self.dev_base();
        for (p, a) in self.k.params.iter().zip(self.args[base..].iter()) {
            match (&p.kind, a) {
                (ParamKind::Struct { fields }, KernelArg::Struct(vals))
                    if fields.len() == vals.len() =>
                {
                    for (f, v) in fields.iter().zip(vals) {
                        match v {
                            KernelArg::Int(x) if types::int_range(f.ty).is_some() => {
                                self.locals[f.local as usize] = Some(int_val(*x, f.ty));
                            }
                            KernelArg::Float(x) if types::int_range(f.ty).is_none() => {
                                self.locals[f.local as usize] = Some(scalar_float(f.ty, *x));
                            }
                            _ => {
                                return Err(KernelTrap::ArgMismatch(
                                    "a struct field argument differs from its field type",
                                ));
                            }
                        }
                    }
                }
                (ParamKind::Buffer { elem, .. }, KernelArg::Buffer(b)) if b.dtype == *elem => {}
                (ParamKind::Scalar { ty, local }, KernelArg::Int(v))
                    if types::int_range(*ty).is_some() =>
                {
                    self.locals[*local as usize] = Some(int_val(*v, *ty));
                }
                (ParamKind::Scalar { ty, local }, KernelArg::Float(v))
                    if types::int_range(*ty).is_none() =>
                {
                    self.locals[*local as usize] = Some(scalar_float(*ty, *v));
                }
                _ => {
                    return Err(KernelTrap::ArgMismatch(
                        "argument kind or dtype differs from its parameter",
                    ));
                }
            }
        }
        Ok(())
    }

    pub(super) fn tick(&mut self) -> Result<(), KernelTrap> {
        self.steps += 1;
        if self.steps > self.limit {
            Err(KernelTrap::StepLimit)
        } else {
            Ok(())
        }
    }

    pub(super) fn exec_block(&mut self, b: &Block) -> Result<Flow, KernelTrap> {
        for st in &b.0 {
            self.tick()?;
            match self.exec_stmt(st)? {
                Flow::Next => {}
                other => return Ok(other),
            }
        }
        Ok(Flow::Next)
    }

    fn exec_stmt(&mut self, st: &Stmt) -> Result<Flow, KernelTrap> {
        match st {
            Stmt::Let { .. }
            | Stmt::Load { .. }
            | Stmt::MaskedLoad { .. }
            | Stmt::Assign { .. }
            | Stmt::Store { .. }
            | Stmt::MaskedStore { .. }
            | Stmt::AtomicRmw { .. }
            | Stmt::AtomicCas { .. }
            | Stmt::MulticastLoadReduce { .. }
            | Stmt::GridDepWait
            | Stmt::GridDepLaunchDependents
            | Stmt::ProxyFence { .. }
            | Stmt::Assume { .. }
            | Stmt::Fence { .. } => {
                self.exec_simple(st)?;
            }
            Stmt::Collective { .. } | Stmt::CollectiveTuple { .. } => {
                return Err(KernelTrap::Unsupported("Collective"));
            }
            Stmt::Mma { .. } => return Err(KernelTrap::Unsupported("Mma")),
            Stmt::TileLoad { .. }
            | Stmt::TileMma { .. }
            | Stmt::TileFill { .. }
            | Stmt::TileStore { .. } => self.exec_tile(st)?,
            Stmt::Tcgen05Mma { .. }
            | Stmt::Tcgen05Commit { .. }
            | Stmt::Tcgen05Ld { .. }
            | Stmt::Tcgen05Alloc { .. }
            | Stmt::Tcgen05Dealloc { .. }
            | Stmt::Tcgen05RelinquishPermit => {
                return Err(KernelTrap::Unsupported("a tcgen05 statement"));
            }
            Stmt::Wgmma { .. } | Stmt::WgmmaCommit | Stmt::WgmmaWait { .. } => {
                return Err(KernelTrap::Unsupported("a wgmma statement"));
            }
            Stmt::MatrixLoad { .. } => return Err(KernelTrap::Unsupported("MatrixLoad")),
            Stmt::MbarInit { .. }
            | Stmt::MbarArrive { .. }
            | Stmt::MbarCompleteTx { .. }
            | Stmt::MbarWait { .. } => {
                return Err(KernelTrap::Unsupported("an mbarrier statement"));
            }
            Stmt::MatrixStore { .. } => return Err(KernelTrap::Unsupported("MatrixStore")),
            Stmt::CopyBulk { .. } => return Err(KernelTrap::Unsupported("CopyBulk")),
            Stmt::CopyUnpack { .. } => return Err(KernelTrap::Unsupported("CopyUnpack")),
            Stmt::ExternCall {
                decl,
                args,
                results,
            } => self.exec_extern(*decl, args, results)?,
            Stmt::Call {
                callee,
                args,
                result,
            } => self.exec_call(*callee, args, result.as_slice())?,
            Stmt::CallTuple {
                callee,
                args,
                results,
            } => self.exec_call(*callee, args, results)?,
            Stmt::If { cond, then, els } => {
                let taken = self.truthy(cond)?;
                return self.exec_block(if taken { then } else { els });
            }
            Stmt::For {
                local,
                start,
                end,
                step,
                body,
                ..
            } => return self.exec_for(*local, [start, end, step], body),
            Stmt::While { cond, body } => {
                while self.truthy(cond)? {
                    self.tick()?;
                    match self.exec_block(body)? {
                        Flow::Break => break,
                        Flow::Return => return Ok(Flow::Return),
                        Flow::Next | Flow::Continue => {}
                    }
                }
            }
            Stmt::Break => return Ok(Flow::Break),
            Stmt::Continue => return Ok(Flow::Continue),
            Stmt::Return => return Ok(Flow::Return),
            Stmt::Barrier { .. } => {
                return Err(KernelTrap::Unsupported("Barrier (run_launch runs it)"));
            }
            Stmt::CopyAsync { .. } | Stmt::AsyncCommit | Stmt::AsyncWait { .. } => {
                return Err(KernelTrap::Unsupported(
                    "an async copy (run_launch runs it)",
                ));
            }
        }
        Ok(Flow::Next)
    }

    /// A statement with no control flow: shared by the recursive evaluator and the resumable one.
    pub(super) fn exec_simple(&mut self, st: &Stmt) -> Result<(), KernelTrap> {
        match st {
            Stmt::Let { local, init } => {
                let v = self.eval_expr(init)?;
                self.locals[*local as usize] = Some(self.settle(*local, v));
            }
            Stmt::Load { local, place } => {
                let v = self.load(place)?;
                self.locals[*local as usize] = Some(v);
            }
            Stmt::MulticastLoadReduce {
                local,
                place,
                op,
                order,
            } => {
                self.set_site(*order, crate::exec::Scope::System);
                let v = self.multicast_load_reduce(*op, place)?;
                self.locals[*local as usize] = Some(v);
            }
            Stmt::MaskedLoad {
                local,
                place,
                mask,
                fill,
            } => {
                // a false mask never reaches the memory: no bounds check, no race record
                let v = if self.truthy(mask)? {
                    self.load(place)?
                } else {
                    let f = self.eval_expr(fill)?;
                    self.settle(*local, f)
                };
                self.locals[*local as usize] = Some(v);
            }
            Stmt::Assign { local, value } => {
                let v = self.eval_expr(value)?;
                self.locals[*local as usize] = Some(self.settle(*local, v));
            }
            Stmt::Store { place, value } => {
                let v = self.eval_expr(value)?;
                self.store(place, v)?;
            }
            Stmt::MaskedStore { place, value, mask } => {
                if self.truthy(mask)? {
                    let v = self.eval_expr(value)?;
                    self.store(place, v)?;
                }
            }
            Stmt::AtomicRmw { .. } | Stmt::AtomicCas { .. } => self.exec_atomic(st)?,
            Stmt::Assume { cond } => {
                if !self.truthy(cond)? {
                    return Err(KernelTrap::AssumptionViolated);
                }
            }
            Stmt::Fence { scope, order } => self.fence(*order, *scope),
            Stmt::ProxyFence { .. } => self.proxy_fence(),
            Stmt::GridDepWait | Stmt::GridDepLaunchDependents => {}
            Stmt::CopyAsync {
                dst,
                src,
                elems,
                valid,
                ..
            } => self.copy_async(dst, src, *elems, valid.as_ref())?,
            Stmt::CopyBulk {
                dst,
                src,
                elems,
                bar,
            } => self.copy_bulk(dst, src, *elems, bar)?,
            Stmt::CopyUnpack {
                dst,
                src,
                groups,
                bar,
            } => self.copy_unpack(dst, src, *groups, bar)?,
            Stmt::Tcgen05Mma { .. } => self.tcgen05_mma(st)?,
            Stmt::Tcgen05Alloc { d, ncols } => self.tcgen05_alloc(*d, *ncols, true)?,
            Stmt::Tcgen05Dealloc { d, ncols } => self.tcgen05_alloc(*d, *ncols, false)?,
            Stmt::Tcgen05RelinquishPermit => self.tcgen05_relinquish()?,
            Stmt::Tcgen05Ld {
                src,
                col,
                count,
                dst,
            } => self.tcgen05_ld(*src, col, *count, *dst)?,
            Stmt::AsyncCommit => self.async_commit()?,
            Stmt::AsyncWait { pending } => self.async_wait(*pending)?,
            _ => unreachable!("exec_simple is called for straight-line statements only"),
        }
        Ok(())
    }

    /// An atomic statement: its operands, then the step at the statement's order and scope; the old value, if wanted.
    fn exec_atomic(&mut self, st: &Stmt) -> Result<(), KernelTrap> {
        let (prev, old) = match st {
            Stmt::AtomicRmw {
                op,
                place,
                value,
                old,
                order,
                scope,
            } => {
                let v = self.eval_expr(value)?;
                self.set_site(*order, *scope);
                (self.atomic_rmw(*op, place, v)?, old)
            }
            Stmt::AtomicCas {
                place,
                expected,
                desired,
                old,
                order,
                scope,
            } => {
                let (e, d) = (self.eval_expr(expected)?, self.eval_expr(desired)?);
                self.set_site(*order, *scope);
                (self.atomic_cas(place, e, d)?, old)
            }
            _ => unreachable!("exec_atomic is called for atomic statements only"),
        };
        if let Some(local) = old {
            self.locals[*local as usize] = Some(prev);
        }
        Ok(())
    }

    /// The three bounds are read once, before the first pass; the counter is a local the body
    /// can read but not assign (the checker forbids it). A counter that would leave its type's
    /// range ends the loop.
    fn exec_for(
        &mut self,
        local: u32,
        bounds: [&ScalarExpr; 3],
        body: &Block,
    ) -> Result<Flow, KernelTrap> {
        let Ty::Scalar(t) = self.local_tys[local as usize] else {
            return Err(KernelTrap::Unsupported("packed loop counter"));
        };
        let (lo, hi) = types::int_range(t).ok_or(KernelTrap::Unsupported("float loop counter"))?;
        let start = self.eval_int(bounds[0], t)?;
        let end = self.eval_int(bounds[1], t)?;
        let step = self.eval_int(bounds[2], t)?;
        if step == 0 {
            return Err(KernelTrap::ZeroStep);
        }
        let mut i = start;
        while if step > 0 { i < end } else { i > end } {
            self.tick()?;
            self.locals[local as usize] = Some(int_val(i, t));
            match self.exec_block(body)? {
                Flow::Break => break,
                Flow::Return => return Ok(Flow::Return),
                Flow::Next | Flow::Continue => {}
            }
            i += step;
            if i < lo || i > hi {
                break;
            }
        }
        Ok(Flow::Next)
    }

    pub(super) fn truthy(&mut self, cond: &ScalarExpr) -> Result<bool, KernelTrap> {
        if self.is_f128(cond) {
            return self.f128_taken(cond);
        }
        Ok(match self.eval_expr(cond)? {
            Val::Raw(b, dt) if types::int_range(dt).is_some() => raw_to_i128(b, dt) != 0,
            v => super::round_to_compute(v.f64(), self.k.compute) != 0.0,
        })
    }

    // ---- places -------------------------------------------------------------------------

    /// The element offset of `place` through the buffer's view, bounds-checked against its shape.
    pub(super) fn offset(&mut self, place: &Place) -> Result<(usize, usize), KernelTrap> {
        let (buffer, coords, device) = match place {
            Place::Logical { buffer, coords } => (buffer, coords, None),
            Place::Peer {
                buffer,
                device,
                coords,
            } => (buffer, coords, Some(device)),
            _ => return Err(KernelTrap::Unsupported("Place::Alloc (U3 row 4 PR 2)")),
        };
        let mut cs = Vec::with_capacity(coords.len());
        for c in coords {
            let Ty::Scalar(t) = types::infer_ty_in(c, self.k.compute, &self.local_tys)
                .map_err(|_| KernelTrap::Unsupported("coordinate type"))?
            else {
                return Err(KernelTrap::Unsupported("packed coordinate"));
            };
            cs.push(self.eval_int(c, t)?);
        }
        let mut root = self.root(*buffer);
        if let Some(d) = device {
            root = self.peer_root(root, d)?;
        }
        let KernelArg::Buffer(tb) = &self.args[root] else {
            return Err(KernelTrap::ArgMismatch(
                "a Place names a non-buffer argument",
            ));
        };
        let mut off = i128::from(tb.base_offset);
        for (d, &c) in cs.iter().enumerate() {
            if c < 0 || c >= i128::from(tb.shape[d]) {
                return Err(KernelTrap::OutOfBounds { buffer: *buffer });
            }
            off += c * i128::from(tb.strides[d]);
        }
        let off = usize::try_from(off).map_err(|_| KernelTrap::OutOfBounds { buffer: *buffer })?;
        Ok((root, off))
    }

    pub(super) fn load(&mut self, place: &Place) -> Result<Val, KernelTrap> {
        if matches!(place, Place::Raw { .. }) {
            return self.raw_load(place);
        }
        if matches!(place, Place::AllocRaw { .. }) {
            return self.alloc_raw_load(place);
        }
        if let Place::Alloc { alloc, index } = place {
            return self.alloc_load(*alloc, index);
        }
        if let Place::ClusterPeer {
            alloc,
            block,
            index,
        } = place
        {
            return self.on_peer(block, |m| m.alloc_load(*alloc, index));
        }
        if let Some((alloc, index)) = self.alloc_bound(place) {
            return self.alloc_load(alloc, index);
        }
        let (b, off) = self.offset(place)?;
        self.note_global(b, off, AccessKind::Read)?;
        let KernelArg::Buffer(tb) = &self.args[b] else {
            unreachable!("offset() returned a buffer");
        };
        Ok(load_val(&tb.bytes, off, tb.dtype))
    }

    /// An atomic read-modify-write of a global integer element: logged as an atomic access (it
    /// races only with a plain access), the combine computed at the element type and wrapped.
    /// Returns the value the element held before.
    fn atomic_rmw(&mut self, op: AtomicOp, place: &Place, v: Val) -> Result<Val, KernelTrap> {
        let Val::Raw(bits, vdt) = v else {
            return Err(KernelTrap::Unsupported("a non-integer atomic operand"));
        };
        self.atomic_step(place, &|before, dt| atomic_apply(op, dt, before, bits, vdt))
    }

    /// Compare-and-exchange: the element becomes `desired` if it equals `expected` (both read at the
    /// element's integer type), and the value found comes back either way.
    fn atomic_cas(
        &mut self,
        place: &Place,
        expected: Val,
        desired: Val,
    ) -> Result<Val, KernelTrap> {
        let (Val::Raw(e, edt), Val::Raw(d, ddt)) = (expected, desired) else {
            return Err(KernelTrap::Unsupported("a non-integer AtomicCas operand"));
        };
        self.atomic_step(place, &|before, dt| {
            let same = raw_to_i128(before, dt) == raw_to_i128(e, edt);
            Ok(if same {
                Val::Raw(d, ddt)
            } else {
                Val::Raw(before, dt)
            })
        })
    }

    /// One indivisible step on an element of a global buffer or a `Shared` allocation: logged as an
    /// atomic access, `step` maps the raw value before (and the element type) to the value after.
    /// Returns the value before.
    fn atomic_step(
        &mut self,
        place: &Place,
        step: &dyn Fn(u128, ElementKind) -> Result<Val, KernelTrap>,
    ) -> Result<Val, KernelTrap> {
        if let Place::Alloc { alloc, index } = place {
            return self.alloc_atomic(*alloc, index, step);
        }
        if let Place::ClusterPeer {
            alloc,
            block,
            index,
        } = place
        {
            return self.on_peer(block, |m| m.alloc_atomic(*alloc, index, step));
        }
        if matches!(place, Place::Multicast { .. }) {
            return self.multicast_atomic(place, step);
        }
        if let Some((alloc, index)) = self.alloc_bound(place) {
            return self.alloc_atomic(alloc, index, step);
        }
        let (b, off) = self.offset(place)?;
        self.note_global(b, off, AccessKind::Atomic)?;
        let KernelArg::Buffer(tb) = &mut self.args[b] else {
            unreachable!("offset() returned a buffer");
        };
        let dt = tb.dtype;
        let before = read_elem(&tb.bytes, off, dt);
        let r = step(before, dt)?;
        store_val(r, dt, &mut tb.bytes, off);
        Ok(Val::Raw(before, dt))
    }

    /// `multimem.red`: the step applied to every device's copy, each logged as an atomic access. The value before on
    /// the first device comes back (the checker admits no `old`, so nothing reads it).
    fn multicast_atomic(
        &mut self,
        place: &Place,
        step: &dyn Fn(u128, ElementKind) -> Result<Val, KernelTrap>,
    ) -> Result<Val, KernelTrap> {
        let mut first = None;
        for (b, off) in self.multicast_targets(place)? {
            self.note_global(b, off, AccessKind::Atomic)?;
            let KernelArg::Buffer(tb) = &mut self.args[b] else {
                unreachable!("multicast_targets returned a buffer");
            };
            let dt = tb.dtype;
            let before = read_elem(&tb.bytes, off, dt);
            let r = step(before, dt)?;
            store_val(r, dt, &mut tb.bytes, off);
            first.get_or_insert(Val::Raw(before, dt));
        }
        first.ok_or(KernelTrap::Unsupported("a multicast place on no device"))
    }

    /// `multimem.ld_reduce`: the reduction by `op` of every device's copy of the element, each read logged as an atomic
    /// access (it races with a plain write only, and an `Acquire` site takes in a `System` release).
    fn multicast_load_reduce(&mut self, op: AtomicOp, place: &Place) -> Result<Val, KernelTrap> {
        let mut seen = Vec::new();
        for (b, off) in self.multicast_targets(place)? {
            self.note_global(b, off, AccessKind::Atomic)?;
            let KernelArg::Buffer(tb) = &self.args[b] else {
                unreachable!("multicast_targets returned a buffer");
            };
            seen.push(read_elem(&tb.bytes, off, tb.dtype));
        }
        let KernelArg::Buffer(tb) = &self.args[self.root(match place {
            Place::Multicast { buffer, .. } => *buffer,
            _ => unreachable!("multicast_targets accepted a multicast place"),
        })] else {
            unreachable!("multicast_targets returned a buffer");
        };
        reduce_devices(op, tb.dtype, &seen)
    }

    pub(super) fn store(&mut self, place: &Place, v: Val) -> Result<(), KernelTrap> {
        if matches!(place, Place::Raw { .. }) {
            return self.raw_store(place, v);
        }
        if matches!(place, Place::AllocRaw { .. }) {
            return self.alloc_raw_store(place, v);
        }
        if let Place::Alloc { alloc, index } = place {
            return self.alloc_store(*alloc, index, v);
        }
        if matches!(place, Place::Multicast { .. }) {
            // `multimem.st`: the value lands in every device's copy, each a write for the race detector
            for (b, off) in self.multicast_targets(place)? {
                self.note_global(b, off, AccessKind::Write)?;
                let KernelArg::Buffer(tb) = &mut self.args[b] else {
                    unreachable!("multicast_targets returned a buffer");
                };
                let dt = tb.dtype;
                store_val(v, dt, &mut tb.bytes, off);
            }
            return Ok(());
        }
        if let Place::ClusterPeer {
            alloc,
            block,
            index,
        } = place
        {
            return self.on_peer(block, |m| m.alloc_store(*alloc, index, v));
        }
        if let Some((alloc, index)) = self.alloc_bound(place) {
            return self.alloc_store(alloc, index, v);
        }
        let (b, off) = self.offset(place)?;
        // whether this store writes the bytes the element already holds (a benign race when the launch allows it)
        let same = match &self.args[b] {
            KernelArg::Buffer(tb) if types::elem_bits(tb.dtype) % 8 == 0 => {
                let esz = types::elem_size(tb.dtype);
                let mut tmp = [0u8; 32];
                store_val(v, tb.dtype, &mut tmp, 0);
                tb.bytes.get(off * esz..(off + 1) * esz) == Some(&tmp[..esz])
            }
            _ => false,
        };
        self.note_global_write(b, off, AccessKind::Write, same)?;
        let KernelArg::Buffer(tb) = &mut self.args[b] else {
            unreachable!("offset() returned a buffer");
        };
        let dt = tb.dtype;
        store_val(v, dt, &mut tb.bytes, off);
        Ok(())
    }

    // ---- expressions --------------------------------------------------------------------

    /// `v` as local `local` holds it: a floating local of a narrower type than `f64` stores the value rounded to that type, as the
    /// serial C does for a `float` variable (an expression is evaluated at `f64` and rounded once per statement, not once per kernel).
    fn settle(&self, local: u32, v: Val) -> Val {
        match (v, self.local_tys[local as usize]) {
            (Val::Float(f), Ty::Scalar(t)) => Val::Float(super::round_to_compute(f, t)),
            _ => v,
        }
    }

    pub(super) fn local_val(&self, i: u32) -> Val {
        self.locals[i as usize].expect("check_kernel guarantees a local is set before it is read")
    }

    pub(super) fn eval_expr(&self, e: &ScalarExpr) -> Result<Val, KernelTrap> {
        match types::infer_ty_in(e, self.k.compute, &self.local_tys)
            .map_err(|_| KernelTrap::Unsupported("ill-typed expression"))?
        {
            Ty::Packed(..) => Err(KernelTrap::Unsupported("packed value")),
            // a binary128 value (`F128`, ROADMAP E3) is computed by the soft binary128, never in `f64`; a `Cast` out of one is too
            Ty::Scalar(ElementKind::F128) => self.eval_f128(e),
            Ty::Scalar(_) if self.is_f128_source_cast(e) => self.f128_cast_out(e),
            Ty::Scalar(t) if types::is_subbyte_int(t) => {
                Ok(Val::Raw((self.eval_subbyte(e, t)? as u128) & 0xf, t))
            }
            // a wide integer (`I128`, `U128`, `I256`, `U256`, ROADMAP E2) is computed on its 256-bit pattern, never on `i128`
            Ty::Scalar(t) if crate::wide_int::wide_int_bits(t).is_some() => self.eval_wide(e, t),
            // a float-to-integer `Cast` (round, saturate, NaN to 0) is a float node producing an integer: the node-by-node
            // evaluation reads the operand at its own type and the oracle's cast does the conversion
            Ty::Scalar(t) if types::int_range(t).is_some() && self.is_float_to_int_cast(e) => {
                self.eval_mixed(e)
            }
            // an integer `Bitcast` of a float (`__float_as_int`): the bits of the operand, read as the integer type
            Ty::Scalar(t) if types::int_range(t).is_some() && self.is_float_bitcast(e) => {
                match self.eval_mixed(e)? {
                    Val::Raw(bits, _) => Ok(int_val(wrap(t, bits as i128), t)),
                    other => Ok(other),
                }
            }
            Ty::Scalar(t) if types::int_range(t).is_some() => Ok(int_val(self.eval_int(e, t)?, t)),
            Ty::Scalar(t) => self.eval_float(e, t),
        }
    }

    /// Whether `e` is a `Bitcast` whose operand is a float (the integer evaluator cannot read one).
    fn is_float_bitcast(&self, e: &ScalarExpr) -> bool {
        let ScalarExpr::Bitcast(x, _) = e else {
            return false;
        };
        matches!(
            types::infer_ty_in(x, self.k.compute, &self.local_tys),
            Ok(Ty::Scalar(from)) if types::is_cast_float(from)
        )
    }

    /// Whether `e` is a `Cast` whose operand is a float (a value the integer evaluator cannot read).
    fn is_float_to_int_cast(&self, e: &ScalarExpr) -> bool {
        let ScalarExpr::Cast(x, _) = e else {
            return false;
        };
        matches!(
            types::infer_ty_in(x, self.k.compute, &self.local_tys),
            Ok(Ty::Scalar(from)) if types::is_cast_float(from)
        )
    }

    /// An integer expression at type `t`. `Div` is the total definition of
    /// `docs/int-div-shift-semantics.md` (PM ruling 2026-10-31): a zero divisor gives the dividend, and
    /// so does `MIN / -1`; shifts read the count as unsigned bits, modulo the promoted width; a
    /// comparison yields the compute type's 1 or 0; a cast is an integer-to-integer wrap.
    pub(super) fn eval_int(&self, e: &ScalarExpr, t: ElementKind) -> Result<i128, KernelTrap> {
        let two = |a: &ScalarExpr, b: &ScalarExpr| -> Result<(i128, i128), KernelTrap> {
            Ok((self.eval_int(a, t)?, self.eval_int(b, t)?))
        };
        Ok(match e {
            ScalarExpr::ConstInt(v, _) => wrap(t, i128::from(*v)),
            ScalarExpr::Local(i) => self.int_local(*i)?,
            ScalarExpr::Exec(id) => wrap(t, self.exec_val(*id)?),
            // an F128 source: rounded under the mode, saturated into the integer range, a NaN to 0 (`kernel_f128`)
            ScalarExpr::Cast(x, spec) if self.is_f128(x) => self.f128_cast_to_int(x, spec)?,
            // a wide source: the conversion is `wide_ops::wide_cast` (wrap or clamp), the result a narrow integer
            ScalarExpr::Cast(x, spec) if self.wide_ty_of(x).is_some() => {
                let from = self.wide_ty_of(x).expect("guarded");
                let w = as_wide(self.eval_expr(x)?, from);
                match wide_cast(w, from, spec.to, spec.sat) {
                    Val::Raw(b, dt) => raw_to_i128(b, dt),
                    _ => Err(KernelTrap::Unsupported("a wide cast to a non-integer"))?,
                }
            }
            ScalarExpr::Cast(x, spec) if spec.to == t && spec.sat == Saturation::Wrap => {
                wrap(t, self.eval_operand_int(x)?)
            }
            ScalarExpr::Add(a, b) => {
                let (x, y) = two(a, b)?;
                wrap(t, x + y)
            }
            ScalarExpr::Sub(a, b) => {
                let (x, y) = two(a, b)?;
                wrap(t, x - y)
            }
            ScalarExpr::Mul(a, b) => {
                let (x, y) = two(a, b)?;
                // the product of two 64-bit values can exceed i128; the low bits are all `wrap` keeps
                wrap(t, x.wrapping_mul(y))
            }
            ScalarExpr::Div(a, b) => {
                let (x, y) = two(a, b)?;
                int_div(t, x, y)
            }
            ScalarExpr::Binary(op, a, b) if is_int_binary(*op) => {
                let (x, y) = two(a, b)?;
                int_binary(t, *op, x, y)
            }
            ScalarExpr::Binary(op, a, b) if op.is_cmp() => self.eval_cmp(*op, a, b)?,
            ScalarExpr::Select(c, a, b) => self.eval_int_select(c, a, b, t)?,
            // an integer bit operation reads its operand at the result's own type (the typing rule: same type)
            ScalarExpr::Unary(op, x) if op.is_bit_op() => int_bit_op(*op, t, self.eval_int(x, t)?),
            _ => Err(KernelTrap::Unsupported(
                "integer expression outside the v0 subset",
            ))?,
        })
    }

    /// An integer `Select`: the integer condition tested `!= 0`, then ONE arm evaluated at `t` (the
    /// other is not, as in the float path). This is what an integer `Rem` composes from.
    fn eval_int_select(
        &self,
        c: &ScalarExpr,
        a: &ScalarExpr,
        b: &ScalarExpr,
        t: ElementKind,
    ) -> Result<i128, KernelTrap> {
        // A comparison is the condition most `Select`s have, and its inferred type is the compute type (`1.0`/`0.0` in an `f32`
        // kernel), not an integer: evaluate it as the comparison it is, over its own operands' integer types.
        let taken = match c {
            ScalarExpr::Binary(op, x, y) if op.is_cmp() => self.eval_cmp(*op, x, y)? != 0,
            _ if self.wide_ty_of(c).is_some() => !self.wide_operand(c)?.0.is_zero(),
            _ => self.eval_operand_int(c)? != 0,
        };
        let arm = if taken { a } else { b };
        self.eval_int(arm, t)
    }

    /// An integer local read inside an integer expression.
    fn int_local(&self, i: u32) -> Result<i128, KernelTrap> {
        match self.local_val(i) {
            Val::Raw(b, dt) => Ok(raw_to_i128(b, dt)),
            _ => Err(KernelTrap::Unsupported(
                "non-integer local in an integer expression",
            )),
        }
    }

    /// A comparison of two integer operands: `1` or `0`.
    fn eval_cmp(&self, op: BinaryOp, a: &ScalarExpr, b: &ScalarExpr) -> Result<i128, KernelTrap> {
        if self.is_f128(a) {
            return Ok(i128::from(self.f128_cmp(op, a, b)?));
        }
        if self.wide_ty_of(a).is_some() {
            let ((x, t), (y, _)) = (self.wide_operand(a)?, self.wide_operand(b)?);
            return Ok(i128::from(wide_compare(op, x, y, t).expect("a comparison")));
        }
        let (x, y) = (self.eval_operand_int(a)?, self.eval_operand_int(b)?);
        Ok(i128::from(match op {
            BinaryOp::CmpEq => x == y,
            BinaryOp::CmpNe => x != y,
            BinaryOp::CmpLt => x < y,
            BinaryOp::CmpLe => x <= y,
            BinaryOp::CmpGt => x > y,
            _ => x >= y,
        }))
    }

    /// The wide integer dtype of `e`, or `None` when `e` is any other type.
    fn wide_ty_of(&self, e: &ScalarExpr) -> Option<ElementKind> {
        match types::infer_ty_in(e, self.k.compute, &self.local_tys) {
            Ok(Ty::Scalar(t)) if crate::wide_int::wide_int_bits(t).is_some() => Some(t),
            _ => None,
        }
    }

    /// `e`, a wide integer expression, as its 256-bit pattern and dtype.
    fn wide_operand(&self, e: &ScalarExpr) -> Result<(W256, ElementKind), KernelTrap> {
        let t = self.wide_ty_of(e).ok_or(KernelTrap::Unsupported(
            "comparison of mixed wide and other operands",
        ))?;
        Ok((as_wide(self.eval_expr(e)?, t), t))
    }

    /// An operand of a wide operation, brought to the wide type `t` (a narrow integer operand is sign- or zero-extended;
    /// the type checker admits only operands of one type, so this is the identity for every checked kernel).
    fn wide_at(&self, e: &ScalarExpr, t: ElementKind) -> Result<W256, KernelTrap> {
        Ok(as_wide(self.eval_expr(e)?, t))
    }

    /// Whether the condition `c` of a `Select` over wide arms is taken (`!= 0`): a comparison by its operands' type, a wide value by
    /// its pattern, any other integer by value.
    fn wide_taken(&self, c: &ScalarExpr) -> Result<bool, KernelTrap> {
        Ok(match c {
            ScalarExpr::Binary(op, x, y) if op.is_cmp() => self.eval_cmp(*op, x, y)? != 0,
            _ if self.wide_ty_of(c).is_some() => !self.wide_operand(c)?.0.is_zero(),
            other => self.eval_operand_int(other)? != 0,
        })
    }

    /// A `Cast` whose result is a wide integer: the operand is a wide pattern or a narrow integer extended from `i128`.
    fn wide_cast_node(&self, x: &ScalarExpr, spec: &CastSpec) -> Result<Val, KernelTrap> {
        let (src, from) = match self.wide_ty_of(x) {
            Some(from) => (as_wide(self.eval_expr(x)?, from), from),
            None => match types::infer_ty_in(x, self.k.compute, &self.local_tys) {
                Ok(Ty::Scalar(from)) if types::int_range(from).is_some() => {
                    (W256::from_i128(self.eval_int(x, from)?), from)
                }
                _ => Err(KernelTrap::Unsupported(
                    "a cast into a wide integer from a non-integer",
                ))?,
            },
        };
        Ok(wide_cast(src, from, spec.to, spec.sat))
    }

    /// A wide integer expression at type `t`: wrapping arithmetic, the total `Div`, shifts modulo the width, ordering and
    /// bitwise operators, `Select`, and a `Cast` into the wide type (`wide_ops`). The value is the canonical pattern of `t`.
    fn eval_wide(&self, e: &ScalarExpr, t: ElementKind) -> Result<Val, KernelTrap> {
        let two = |a: &ScalarExpr, b: &ScalarExpr| -> Result<(W256, W256), KernelTrap> {
            Ok((self.wide_at(a, t)?, self.wide_at(b, t)?))
        };
        let w = match e {
            ScalarExpr::Local(i) => return Ok(self.local_val(*i)),
            ScalarExpr::ConstInt(v, _) => W256::from_i128(i128::from(*v)).wrap_to(t),
            ScalarExpr::Cast(x, spec) => return self.wide_cast_node(x, spec),
            ScalarExpr::Add(a, b) => {
                let (x, y) = two(a, b)?;
                wide_add(x, y, t)
            }
            ScalarExpr::Sub(a, b) => {
                let (x, y) = two(a, b)?;
                wide_sub(x, y, t)
            }
            ScalarExpr::Mul(a, b) => {
                let (x, y) = two(a, b)?;
                wide_mul(x, y, t)
            }
            ScalarExpr::Div(a, b) => {
                let (x, y) = two(a, b)?;
                wide_div(x, y, t)
            }
            ScalarExpr::Binary(op, a, b) if is_int_binary(*op) => {
                let (x, y) = two(a, b)?;
                wide_binary(*op, x, y, t)
            }
            ScalarExpr::Select(c, a, b) => {
                self.wide_at(if self.wide_taken(c)? { a } else { b }, t)?
            }
            _ => Err(KernelTrap::Unsupported(
                "wide integer expression outside the v0 subset",
            ))?,
        };
        Ok(Val::Wide(w, t))
    }

    /// A sub-byte integer expression (`I4`, `U4`): only a local, a constant or a narrowing `Cast` of an integer exists
    /// (a sub-byte value is read, widened and narrowed, never computed on). The value is in the type's range.
    fn eval_subbyte(&self, e: &ScalarExpr, t: ElementKind) -> Result<i128, KernelTrap> {
        let (lo, hi) = types::subbyte_range(t).expect("a sub-byte type");
        let wrap = |v: i128| (v - lo).rem_euclid(hi - lo + 1) + lo;
        match e {
            ScalarExpr::Local(i) => match self.local_val(*i) {
                Val::Raw(b, dt) => Ok(raw_to_i128(b, dt)),
                _ => Err(KernelTrap::Unsupported(
                    "a non-raw value in a sub-byte local",
                )),
            },
            ScalarExpr::ConstInt(v, _) => Ok(wrap(i128::from(*v))),
            ScalarExpr::Cast(x, spec) if spec.to == t && spec.sat == Saturation::Wrap => {
                Ok(wrap(self.eval_operand_int(x)?))
            }
            _ => Err(KernelTrap::Unsupported(
                "arithmetic on a sub-byte value (widen it with a Cast first)",
            )),
        }
    }

    /// An integer operand evaluated at its own inferred type.
    pub(super) fn eval_operand_int(&self, e: &ScalarExpr) -> Result<i128, KernelTrap> {
        match types::infer_ty_in(e, self.k.compute, &self.local_tys) {
            Ok(Ty::Scalar(t)) if types::is_subbyte_int(t) => self.eval_subbyte(e, t),
            Ok(Ty::Scalar(t)) if types::int_range(t).is_some() => self.eval_int(e, t),
            _ => Err(KernelTrap::Unsupported(
                "comparison of a non-integer operand",
            )),
        }
    }

    /// A floating-point expression of type `t`.
    ///
    /// When `t` and every local it reads are the compute dtype, this is exactly what the oracle
    /// does to an `OpDef` body: rewrite `Local(i)` to `Input(slot)` and evaluate. Otherwise (an
    /// `f64` accumulator inside an `f32` kernel, a cast of an integer local) the oracle's
    /// compute-dtype machinery would round at the wrong width, so the expression is evaluated
    /// node by node, each node at the type of its operands ([`Self::eval_mixed`]).
    fn eval_float(&self, e: &ScalarExpr, t: ElementKind) -> Result<Val, KernelTrap> {
        if t == self.k.compute && self.reads_only_compute_locals(e) {
            let mut slots: Vec<u32> = Vec::new();
            let rewritten = self.rewrite(e, &mut slots)?;
            let leaf = |s: u8| self.local_val(slots[usize::from(s)]);
            return Ok(self.run_oracle(&rewritten, self.k.compute, &leaf));
        }
        self.eval_mixed(e)
    }

    fn reads_only_compute_locals(&self, e: &ScalarExpr) -> bool {
        match e {
            ScalarExpr::Local(i) => self.local_tys[*i as usize] == Ty::Scalar(self.k.compute),
            // An execution id is a U32, never a compute-dtype local; its parent runs node by node.
            ScalarExpr::Exec(_) => false,
            _ => types::operands(e)
                .into_iter()
                .all(|c| self.reads_only_compute_locals(c)),
        }
    }

    fn run_oracle(&self, e: &ScalarExpr, dtype: ElementKind, leaf: &dyn Fn(u8) -> Val) -> Val {
        let coord = |_d: u8| -> Val { unreachable!("check_kernel rejects Coord inside a kernel") };
        let ev = Eval {
            dtype,
            params: &[],
            leaf,
            reduced: &panic_reduced,
            coord: &coord,
        };
        eval(e, &ev)
    }

    /// Evaluate `e` one node at a time: children first (each by [`Self::eval_expr`], so a pure
    /// compute-dtype subtree still takes the fast path), then the node itself under the oracle,
    /// with the children's values as its leaves and the node's operand type as the evaluation
    /// dtype. So `Add` of two `f64` locals is rounded at `f64`, and a `Cast` of an `i32` local
    /// reads its operand as `i32`.
    fn eval_mixed(&self, e: &ScalarExpr) -> Result<Val, KernelTrap> {
        match e {
            ScalarExpr::Local(i) => return Ok(self.local_val(*i)),
            ScalarExpr::Const(v) => return Ok(Val::Float(*v)),
            ScalarExpr::Qualified(..) => {
                return Err(KernelTrap::Unsupported(
                    "a Qualified node over a mixed-type operand",
                ));
            }
            _ => {}
        }
        let kids = types::operands(e);
        if kids.is_empty() {
            return Err(KernelTrap::Unsupported(
                "a leaf with no mixed-type evaluation",
            ));
        }
        let mut vals = Vec::with_capacity(kids.len());
        let mut tys = Vec::with_capacity(kids.len());
        for k in &kids {
            let ty = types::infer_ty_in(k, self.k.compute, &self.local_tys)
                .map_err(|_| KernelTrap::Unsupported("ill-typed operand"))?;
            let Ty::Scalar(t) = ty else {
                return Err(KernelTrap::Unsupported("packed operand"));
            };
            tys.push(t);
            vals.push(self.eval_expr(k)?);
        }
        // The dtype the node runs at: its operands' (a `Select`'s arms are operands 1 and 2).
        let dtype = if matches!(e, ScalarExpr::Select(..)) {
            tys[1]
        } else {
            tys[0]
        };
        let mut next = 0u8;
        let node = map_children(e, &mut |_| {
            let slot = next;
            next += 1;
            Ok::<_, KernelTrap>(ScalarExpr::Input(slot))
        })?;
        let leaf = |s: u8| vals[usize::from(s)];
        Ok(self.run_oracle(&node, dtype, &leaf))
    }

    fn rewrite(&self, e: &ScalarExpr, slots: &mut Vec<u32>) -> Result<ScalarExpr, KernelTrap> {
        if let ScalarExpr::Local(i) = e {
            if self.local_tys[*i as usize] != Ty::Scalar(self.k.compute) {
                return Err(KernelTrap::Unsupported(
                    "a float expression reads a local that is not a compute-dtype scalar",
                ));
            }
            let slot = slots.iter().position(|s| s == i).unwrap_or_else(|| {
                slots.push(*i);
                slots.len() - 1
            });
            let slot = u8::try_from(slot)
                .map_err(|_| KernelTrap::Unsupported("more than 256 locals in one expression"))?;
            return Ok(ScalarExpr::Input(slot));
        }
        map_children(e, &mut |c| self.rewrite(c, slots))
    }
}

/// `x / y` at integer type `t`, total (`docs/int-div-shift-semantics.md`, rules 1 and 2): `y == 0` gives
/// `x`; so does `MIN / -1` (the wrapping result); otherwise the quotient truncated toward zero.
/// `old (op) new` at the element type `t`, from the raw bits of both. Integers: `Add` wraps,
/// `Max`/`Min` select, `And`/`Or`/`Xor` are the bitwise operations, `Exchange` takes the new value.
/// `F16`/`Bf16` `Add`: the exact sum, rounded once to the element (ROADMAP U12b); `Exchange` moves the bits.
/// `F32`/`F64` `Add`: one correctly rounded add in the element type (an `f32` sum of two `f32` is
/// exact in an `f64`, so adding there and rounding once is the same number); `F32`/`F64` `Exchange`
/// moves the bits unchanged.
pub(super) fn atomic_apply(
    op: AtomicOp,
    t: ElementKind,
    old: u128,
    new: u128,
    new_ty: ElementKind,
) -> Result<Val, KernelTrap> {
    match (t, op) {
        (
            ElementKind::F32 | ElementKind::F64 | ElementKind::F16 | ElementKind::Bf16,
            AtomicOp::Exchange,
        ) => Ok(Val::Raw(new, t)),
        (ElementKind::F16 | ElementKind::Bf16, AtomicOp::Add) => {
            // the sum of two 16-bit floats is exact in an `f64` for `F16` (and rounds like the exact sum for `Bf16`,
            // whose far-apart operands differ by far more than a `Bf16` half-ulp), so one rounding to `t` is the result
            let of = |b: u128| super::raw_to_f64(b, t);
            Ok(Val::Raw(super::encode_float(of(old) + of(new), t), t))
        }
        (ElementKind::F32, AtomicOp::Add) => {
            let sum = f32::from_bits(old as u32) + f32::from_bits(new as u32);
            Ok(Val::Raw(u128::from(sum.to_bits()), t))
        }
        (ElementKind::F64, AtomicOp::Add) => {
            let sum = f64::from_bits(old as u64) + f64::from_bits(new as u64);
            Ok(Val::Raw(u128::from(sum.to_bits()), t))
        }
        _ if types::int_range(t).is_some() => {
            let (a, b) = (raw_to_i128(old, t), raw_to_i128(new, new_ty));
            Ok(int_val(
                match op {
                    AtomicOp::Add => wrap(t, a + b),
                    AtomicOp::Max => a.max(b),
                    AtomicOp::Min => a.min(b),
                    AtomicOp::And => a & b,
                    AtomicOp::Or => a | b,
                    AtomicOp::Xor => a ^ b,
                    AtomicOp::Exchange => b,
                    AtomicOp::Inc => {
                        if a >= b {
                            0
                        } else {
                            a + 1
                        }
                    }
                    AtomicOp::Dec => {
                        if a == 0 || a > b {
                            b
                        } else {
                            a - 1
                        }
                    }
                },
                t,
            ))
        }
        _ => Err(KernelTrap::Unsupported("an atomic outside the v1 form")),
    }
}

/// The reduction by `op` of the raw values `seen` (one per device, in device order) of an element of type `dt`
/// (`multimem.ld_reduce`). Integers and `F32` fold through [`atomic_apply`] (an integer `Add` wraps; `F32` `Add` is one
/// rounded add per device, in order: the order is unspecified in a band). `F16` and `Bf16` accumulate in `f32` and round
/// to the element once (`Add`, `Min`, `Max`; a NaN operand of `Min`/`Max` is unspecified).
fn reduce_devices(op: AtomicOp, dt: ElementKind, seen: &[u128]) -> Result<Val, KernelTrap> {
    let (&first, rest) = seen
        .split_first()
        .ok_or(KernelTrap::Unsupported("a multicast place on no device"))?;
    match dt {
        ElementKind::F16 | ElementKind::Bf16 => {
            let of = |b: u128| super::raw_to_f64(b, dt) as f32;
            let mut acc = of(first);
            for &b in rest {
                let x = of(b);
                acc = match op {
                    AtomicOp::Add => acc + x,
                    AtomicOp::Min => acc.min(x),
                    AtomicOp::Max => acc.max(x),
                    _ => return Err(KernelTrap::Unsupported("a multicast reduce op")),
                };
            }
            Ok(Val::Raw(super::encode_float(f64::from(acc), dt), dt))
        }
        _ => {
            let mut acc = Val::Raw(first, dt);
            for &b in rest {
                let Val::Raw(a, _) = acc else {
                    unreachable!("atomic_apply returns a raw value");
                };
                acc = atomic_apply(op, dt, a, b, dt)?;
            }
            Ok(acc)
        }
    }
}

fn int_div(t: ElementKind, x: i128, y: i128) -> i128 {
    let (lo, _) = types::int_range(t).expect("int_div is called with an integer type");
    if y == 0 || (lo < 0 && x == lo && y == -1) {
        return x;
    }
    x / y
}

/// The integer binary ops `eval_int` folds with both operands already at the type `t`.
fn is_int_binary(op: BinaryOp) -> bool {
    matches!(
        op,
        BinaryOp::Shl
            | BinaryOp::Shr
            | BinaryOp::Min
            | BinaryOp::Max
            | BinaryOp::BitAnd
            | BinaryOp::BitOr
            | BinaryOp::BitXor
    )
}

/// One of [`is_int_binary`]'s ops on `x` and `y`. The bitwise ops act on the type's values (sign- or
/// zero-extended), so their results are values of the type too.
fn int_binary(t: ElementKind, op: BinaryOp, x: i128, y: i128) -> i128 {
    match op {
        BinaryOp::Min => x.min(y),
        BinaryOp::Max => x.max(y),
        BinaryOp::BitAnd => x & y,
        BinaryOp::BitOr => x | y,
        BinaryOp::BitXor => x ^ y,
        _ => int_shift(t, matches!(op, BinaryOp::Shl), x, y),
    }
}

/// `x << y` or `x >> y` at integer type `t` (rule 3): the count's bits read as unsigned, taken modulo
/// the promoted width, which is 64 for a 64-bit type and 32 for every narrower one (an 8- or 16-bit
/// operand promotes to `int`, PM ruling 2026-10-31). `<<` discards the bits shifted out and is defined
/// for a negative or overflowing `x`; `>>` is arithmetic on a signed type and logical on an unsigned one.
fn int_shift(t: ElementKind, left: bool, x: i128, y: i128) -> i128 {
    let width: u128 = if types::typed_bits(t) == Some(64) {
        64
    } else {
        32
    };
    // `y as u128` sign-extends a negative count, whose low bits are its two's-complement bits.
    let k = (y as u128 & (width - 1)) as u32;
    if left { wrap(t, x << k) } else { x >> k }
}

/// `v` wrapped to `t`'s width (two's complement for signed, modulo for unsigned).
pub(super) fn wrap(t: ElementKind, v: i128) -> i128 {
    let (lo, hi) = types::int_range(t).expect("wrap is called with an integer type");
    if (lo..=hi).contains(&v) {
        return v;
    }
    let m = hi - lo + 1;
    (v - lo).rem_euclid(m) + lo
}

/// `Popc`/`Clz`/`Brev` of `v` at the integer type `t` (`UnaryOp::is_bit_op`): the operand's `w`-bit two's-complement pattern
/// (`w` is `t`'s width), the result wrapped back into `t`. `Popc` counts set bits, `Clz` counts leading zero bits (`w` for a
/// zero pattern), `Brev` moves bit `i` to bit `w - 1 - i`.
pub(super) fn int_bit_op(op: UnaryOp, t: ElementKind, v: i128) -> i128 {
    let (lo, hi) = types::int_range(t).expect("an integer type");
    let w = (hi + 1).trailing_zeros() + u32::from(lo < 0);
    let mask = if w == 128 {
        u128::MAX
    } else {
        (1u128 << w) - 1
    };
    let pat = (v as u128) & mask;
    let out = match op {
        UnaryOp::Popc => i128::from(pat.count_ones()),
        UnaryOp::Clz => i128::from(pat.leading_zeros() - (128 - w)),
        UnaryOp::Brev => (pat.reverse_bits() >> (128 - w)) as i128,
        other => panic!("oracle: {other:?} is not an integer bit operation"),
    };
    wrap(t, out)
}
