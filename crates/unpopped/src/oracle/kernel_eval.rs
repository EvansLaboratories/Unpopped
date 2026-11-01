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

use super::kernel_mem::{AccessKind, ThreadCtx};
use super::{
    Eval, TypedBuffer, Val, eval, int_val, panic_reduced, raw_to_i128, read_elem, store_val,
};
use crate::ir::{BinaryOp, Saturation, ScalarExpr};
use crate::kernel::{AtomicOp, Block, KernelError, KernelFn, ParamKind, Place, Stmt, check_kernel};
use crate::types::{self, Ty, map_children};
use unpopped_vocab::ElementKind;

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
}

/// Why a run stopped without finishing.
#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum KernelTrap {
    /// [`check_kernel`] refuses the kernel; the interpreter never runs a malformed one.
    Malformed(KernelError),
    /// The arguments do not match the parameters.
    ArgMismatch(&'static str),
    /// A `For` with `step == 0`.
    ZeroStep,
    /// More than [`STEP_LIMIT`] statements.
    StepLimit,
    /// A coordinate outside its buffer's logical shape.
    OutOfBounds {
        /// The buffer parameter index.
        buffer: u32,
    },
    /// A construct this interpreter does not evaluate (named), by design not guessed at.
    Unsupported(&'static str),
    /// An index outside a launched kernel's allocation (never wrapped, never silent).
    AllocOutOfBounds {
        /// The allocation index.
        alloc: u32,
    },
    /// A read of a `Shared` element no thread has stored. Shared memory is not zero-filled: a
    /// zero would hide the missing-barrier bug the memory space exists to catch.
    UninitializedShared {
        /// The allocation index.
        alloc: u32,
    },
    /// A read of a `Private` element this thread has not stored.
    UninitializedPrivate {
        /// The allocation index.
        alloc: u32,
    },
    /// Threads of one barrier's scope did not all reach the same barrier: one finished while
    /// peers waited, or they waited at different barriers and none could be released.
    BarrierMismatch,
    /// Two threads touched one element with no ordering between them and at least one wrote it:
    /// a store racing a store, or a read of another thread's store.
    DataRace,
}

enum Flow {
    Next,
    Break,
    Continue,
    Return,
}

pub(super) struct Machine<'a> {
    pub(super) k: &'a KernelFn,
    pub(super) local_tys: Vec<Ty>,
    pub(super) locals: Vec<Option<Val>>,
    args: &'a mut [KernelArg],
    pub(super) steps: u64,
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
            steps: 0,
            ctx,
        }
    }

    /// Bind the arguments and run the body to completion (one logical thread).
    pub(super) fn run_body(&mut self) -> Result<(), KernelTrap> {
        self.bind_args()?;
        let k = self.k;
        self.exec_block(&k.body).map(|_| ())
    }

    pub(super) fn bind_args(&mut self) -> Result<(), KernelTrap> {
        for (p, a) in self.k.params.iter().zip(self.args.iter()) {
            match (&p.kind, a) {
                (ParamKind::Buffer { elem, .. }, KernelArg::Buffer(b)) if b.dtype == *elem => {}
                (ParamKind::Scalar { ty, local }, KernelArg::Int(v))
                    if types::int_range(*ty).is_some() =>
                {
                    self.locals[*local as usize] = Some(int_val(*v, *ty));
                }
                (ParamKind::Scalar { ty, local }, KernelArg::Float(v))
                    if types::int_range(*ty).is_none() =>
                {
                    self.locals[*local as usize] = Some(Val::Float(*v));
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
        if self.steps > STEP_LIMIT {
            Err(KernelTrap::StepLimit)
        } else {
            Ok(())
        }
    }

    fn exec_block(&mut self, b: &Block) -> Result<Flow, KernelTrap> {
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
            | Stmt::Assign { .. }
            | Stmt::Store { .. }
            | Stmt::AtomicRmw { .. } => {
                self.exec_simple(st)?;
            }
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
        }
        Ok(Flow::Next)
    }

    /// A statement with no control flow: shared by the recursive evaluator and the resumable one.
    pub(super) fn exec_simple(&mut self, st: &Stmt) -> Result<(), KernelTrap> {
        match st {
            Stmt::Let { local, init } => {
                let v = self.eval_expr(init)?;
                self.locals[*local as usize] = Some(v);
            }
            Stmt::Load { local, place } => {
                let v = self.load(place)?;
                self.locals[*local as usize] = Some(v);
            }
            Stmt::Assign { local, value } => {
                let v = self.eval_expr(value)?;
                self.locals[*local as usize] = Some(v);
            }
            Stmt::Store { place, value } => {
                let v = self.eval_expr(value)?;
                self.store(place, v)?;
            }
            Stmt::AtomicRmw { op, place, value } => {
                let v = self.eval_expr(value)?;
                self.atomic_rmw(*op, place, v)?;
            }
            _ => unreachable!("exec_simple is called for straight-line statements only"),
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
        Ok(match self.eval_expr(cond)? {
            Val::Raw(b, dt) if types::int_range(dt).is_some() => raw_to_i128(b, dt) != 0,
            v => super::round_to_compute(v.f64(), self.k.compute) != 0.0,
        })
    }

    // ---- places -------------------------------------------------------------------------

    /// The element offset of `place` through the buffer's view, bounds-checked against its shape.
    fn offset(&mut self, place: &Place) -> Result<(usize, usize), KernelTrap> {
        let Place::Logical { buffer, coords } = place else {
            return Err(KernelTrap::Unsupported("Place::Alloc (U3 row 4 PR 2)"));
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
        let KernelArg::Buffer(tb) = &self.args[*buffer as usize] else {
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
        Ok((*buffer as usize, off))
    }

    fn load(&mut self, place: &Place) -> Result<Val, KernelTrap> {
        if let Place::Alloc { alloc, index } = place {
            return self.alloc_load(*alloc, index);
        }
        let (b, off) = self.offset(place)?;
        self.note_global(b, off, AccessKind::Read)?;
        let KernelArg::Buffer(tb) = &self.args[b] else {
            unreachable!("offset() returned a buffer");
        };
        Ok(Val::Raw(read_elem(&tb.bytes, off, tb.dtype), tb.dtype))
    }

    /// An atomic read-modify-write of a global integer element: logged as an atomic access (it
    /// races only with a plain access), the combine computed at the element type and wrapped.
    fn atomic_rmw(&mut self, op: AtomicOp, place: &Place, v: Val) -> Result<(), KernelTrap> {
        if let Place::Alloc { alloc, index } = place {
            return self.alloc_atomic(op, *alloc, index, v);
        }
        let (b, off) = self.offset(place)?;
        self.note_global(b, off, AccessKind::Atomic)?;
        let KernelArg::Buffer(tb) = &mut self.args[b] else {
            unreachable!("offset() returned a buffer");
        };
        let dt = tb.dtype;
        let Val::Raw(bits, vdt) = v else {
            return Err(KernelTrap::Unsupported("a non-integer atomic operand"));
        };
        let r = atomic_apply(op, dt, read_elem(&tb.bytes, off, dt), bits, vdt)?;
        store_val(r, dt, &mut tb.bytes, off);
        Ok(())
    }

    fn store(&mut self, place: &Place, v: Val) -> Result<(), KernelTrap> {
        if let Place::Alloc { alloc, index } = place {
            return self.alloc_store(*alloc, index, v);
        }
        let (b, off) = self.offset(place)?;
        self.note_global(b, off, AccessKind::Write)?;
        let KernelArg::Buffer(tb) = &mut self.args[b] else {
            unreachable!("offset() returned a buffer");
        };
        let dt = tb.dtype;
        store_val(v, dt, &mut tb.bytes, off);
        Ok(())
    }

    // ---- expressions --------------------------------------------------------------------

    fn local_val(&self, i: u32) -> Val {
        self.locals[i as usize].expect("check_kernel guarantees a local is set before it is read")
    }

    fn eval_expr(&self, e: &ScalarExpr) -> Result<Val, KernelTrap> {
        match types::infer_ty_in(e, self.k.compute, &self.local_tys)
            .map_err(|_| KernelTrap::Unsupported("ill-typed expression"))?
        {
            Ty::Packed(..) => Err(KernelTrap::Unsupported("packed value")),
            Ty::Scalar(t) if types::int_range(t).is_some() => Ok(int_val(self.eval_int(e, t)?, t)),
            Ty::Scalar(t) => self.eval_float(e, t),
        }
    }

    pub(super) fn eval_int(&self, e: &ScalarExpr, t: ElementKind) -> Result<i128, KernelTrap> {
        let two = |a: &ScalarExpr, b: &ScalarExpr| -> Result<(i128, i128), KernelTrap> {
            Ok((self.eval_int(a, t)?, self.eval_int(b, t)?))
        };
        Ok(match e {
            ScalarExpr::ConstInt(v, _) => wrap(t, i128::from(*v)),
            ScalarExpr::Local(i) => match self.local_val(*i) {
                Val::Raw(b, dt) => raw_to_i128(b, dt),
                _ => {
                    return Err(KernelTrap::Unsupported(
                        "non-integer local in an integer expression",
                    ));
                }
            },
            ScalarExpr::Exec(id) => wrap(t, self.exec_val(*id)?),
            // An integer-to-integer cast that wraps: the operand at its own type, then into `t`.
            // Saturating casts stay outside the subset.
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
                wrap(t, x * y)
            }
            // Integer division, remainder-free: the total definition of `docs/int-div-shift-semantics.md`
            // (PM ruling 2026-10-31): a zero divisor gives the dividend, and so does `MIN / -1`.
            ScalarExpr::Div(a, b) => {
                let (x, y) = two(a, b)?;
                int_div(t, x, y)
            }
            // Shifts read the count as unsigned bits, modulo the promoted width.
            ScalarExpr::Binary(op @ (BinaryOp::Shl | BinaryOp::Shr), a, b) => {
                let (x, y) = two(a, b)?;
                int_shift(t, matches!(op, BinaryOp::Shl), x, y)
            }
            ScalarExpr::Binary(op @ (BinaryOp::Min | BinaryOp::Max), a, b) => {
                let (x, y) = two(a, b)?;
                if matches!(op, BinaryOp::Min) {
                    x.min(y)
                } else {
                    x.max(y)
                }
            }
            // A comparison yields the compute type's 1 or 0, as the oracle does; its operands are
            // integers of their own type, not necessarily `t`.
            ScalarExpr::Binary(op, a, b) if op.is_cmp() => self.eval_cmp(*op, a, b)?,
            // An integer `Select`: the integer condition tested `!= 0`, then ONE arm evaluated at `t`
            // (the other is not, as in the float path). This is what an integer `Rem` composes from.
            ScalarExpr::Select(c, a, b) => {
                if self.eval_operand_int(c)? != 0 {
                    self.eval_int(a, t)?
                } else {
                    self.eval_int(b, t)?
                }
            }
            _ => {
                return Err(KernelTrap::Unsupported(
                    "integer expression outside the v0 subset",
                ));
            }
        })
    }

    /// A comparison of two integer operands: `1` or `0`.
    fn eval_cmp(&self, op: BinaryOp, a: &ScalarExpr, b: &ScalarExpr) -> Result<i128, KernelTrap> {
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

    /// An integer operand evaluated at its own inferred type.
    fn eval_operand_int(&self, e: &ScalarExpr) -> Result<i128, KernelTrap> {
        match types::infer_ty_in(e, self.k.compute, &self.local_tys) {
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
/// `Max`/`Min` select. `F32`/`F64` `Add`: one correctly rounded add in the element type (an `f32`
/// sum of two `f32` is exact in an `f64`, so adding there and rounding once is the same number).
pub(super) fn atomic_apply(
    op: AtomicOp,
    t: ElementKind,
    old: u128,
    new: u128,
    new_ty: ElementKind,
) -> Result<Val, KernelTrap> {
    match (t, op) {
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
                },
                t,
            ))
        }
        _ => Err(KernelTrap::Unsupported("an atomic outside the v1 form")),
    }
}

fn int_div(t: ElementKind, x: i128, y: i128) -> i128 {
    let (lo, _) = types::int_range(t).expect("int_div is called with an integer type");
    if y == 0 || (lo < 0 && x == lo && y == -1) {
        return x;
    }
    x / y
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
fn wrap(t: ElementKind, v: i128) -> i128 {
    let (lo, hi) = types::int_range(t).expect("wrap is called with an integer type");
    if (lo..=hi).contains(&v) {
        return v;
    }
    let m = hi - lo + 1;
    (v - lo).rem_euclid(m) + lo
}
