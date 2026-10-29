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

use super::{
    Eval, TypedBuffer, Val, eval, int_val, panic_reduced, raw_to_i128, read_elem, store_val,
};
use crate::ir::{BinaryOp, ScalarExpr};
use crate::kernel::{Block, KernelError, KernelFn, ParamKind, Place, Stmt, check_kernel};
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
}

enum Flow {
    Next,
    Break,
    Continue,
    Return,
}

struct Machine<'a> {
    k: &'a KernelFn,
    local_tys: Vec<Ty>,
    locals: Vec<Option<Val>>,
    args: &'a mut [KernelArg],
    steps: u64,
}

/// Run `k` on `args` (one per parameter, in order). Buffers are updated in place.
///
/// # Errors
///
/// A [`KernelTrap`].
pub fn run_kernel(k: &KernelFn, args: &mut [KernelArg]) -> Result<(), KernelTrap> {
    check_kernel(k).map_err(KernelTrap::Malformed)?;
    if args.len() != k.params.len() {
        return Err(KernelTrap::ArgMismatch("one argument per parameter"));
    }
    let mut m = Machine {
        k,
        local_tys: k.locals.iter().map(|d| d.ty()).collect(),
        locals: vec![None; k.locals.len()],
        args,
        steps: 0,
    };
    m.bind_args()?;
    m.exec_block(&k.body).map(|_| ())
}

impl Machine<'_> {
    fn bind_args(&mut self) -> Result<(), KernelTrap> {
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

    fn tick(&mut self) -> Result<(), KernelTrap> {
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
        }
        Ok(Flow::Next)
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

    fn truthy(&mut self, cond: &ScalarExpr) -> Result<bool, KernelTrap> {
        Ok(match self.eval_expr(cond)? {
            Val::Raw(b, dt) if types::int_range(dt).is_some() => raw_to_i128(b, dt) != 0,
            v => super::round_to_compute(v.f64(), self.k.compute) != 0.0,
        })
    }

    // ---- places -------------------------------------------------------------------------

    /// The element offset of `place` through the buffer's view, bounds-checked against its shape.
    fn offset(&mut self, place: &Place) -> Result<(usize, usize), KernelTrap> {
        let Place::Logical { buffer, coords } = place;
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
        let (b, off) = self.offset(place)?;
        let KernelArg::Buffer(tb) = &self.args[b] else {
            unreachable!("offset() returned a buffer");
        };
        Ok(Val::Raw(read_elem(&tb.bytes, off, tb.dtype), tb.dtype))
    }

    fn store(&mut self, place: &Place, v: Val) -> Result<(), KernelTrap> {
        let (b, off) = self.offset(place)?;
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

    fn eval_int(&self, e: &ScalarExpr, t: ElementKind) -> Result<i128, KernelTrap> {
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
            ScalarExpr::Binary(op @ (BinaryOp::Min | BinaryOp::Max), a, b) => {
                let (x, y) = two(a, b)?;
                if matches!(op, BinaryOp::Min) {
                    x.min(y)
                } else {
                    x.max(y)
                }
            }
            _ => {
                return Err(KernelTrap::Unsupported(
                    "integer expression outside the v0 subset",
                ));
            }
        })
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

/// `v` wrapped to `t`'s width (two's complement for signed, modulo for unsigned).
fn wrap(t: ElementKind, v: i128) -> i128 {
    let (lo, hi) = types::int_range(t).expect("wrap is called with an integer type");
    if (lo..=hi).contains(&v) {
        return v;
    }
    let m = hi - lo + 1;
    (v - lo).rem_euclid(m) + lo
}
