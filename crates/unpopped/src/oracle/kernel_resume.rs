// SPDX-License-Identifier: MIT OR Apache-2.0
//! **The resumable evaluator** (`docs/u3-row4-scopes-and-memory-spaces.md` section 5, step 2):
//! one thread's program as an explicit frame stack instead of Rust recursion, because a `Barrier`
//! can sit inside a loop and the thread must stop there and later continue.
//!
//! It is used by [`super::kernel_launch::run_launch`] only. `run_kernel` and the lifting
//! differential keep the recursive evaluator; `tests/kernel_launch.rs` runs the same programs
//! through both (`the_two_evaluators_agree`) so a drift in loop semantics shows.

use super::int_val;
use super::kernel_collective::Contribution;
use super::kernel_eval::{KernelTrap, Machine};
use super::kernel_mma::MmaContribution;
use crate::exec::Scope;
use crate::ir::ScalarExpr;
use crate::kernel::{Block, CallArg, KernelFn, Module, Stmt};
use crate::types::{self, Ty};

/// One level of the control stack.
pub(super) enum Frame<'a> {
    /// The statements of a block, `pc` of them done.
    Seq { stmts: &'a [Stmt], pc: usize },
    /// A counted loop between passes. `started` is false before the first pass.
    For {
        local: u32,
        ty: unpopped_vocab::ElementKind,
        range: (i128, i128),
        i: i128,
        end: i128,
        step: i128,
        started: bool,
        body: &'a Block,
    },
    /// A `while` loop between passes.
    While {
        cond: &'a ScalarExpr,
        body: &'a Block,
    },
    /// A call in progress: the caller's frame, put back when the callee's body ends (or it `Return`s).
    Call {
        caller: u32,
        locals: Vec<Option<super::Val>>,
        bufmap: Vec<usize>,
        result: Option<u32>,
    },
}

impl<'a> Frame<'a> {
    /// The frame that runs `block` from its first statement.
    fn seq(block: &'a Block) -> Self {
        Frame::Seq {
            stmts: &block.0,
            pc: 0,
        }
    }
}

/// Why a thread stopped running.
pub(super) enum Stop {
    /// The program finished (fell off the end, or `Return`).
    Done,
    /// At a barrier statement (`at` identifies it) waiting for the rest of `scope`.
    Barrier { scope: Scope, at: usize },
    /// At a collective statement, having contributed `c`, waiting for the rest of `scope`.
    Collective {
        scope: Scope,
        at: usize,
        c: Contribution,
    },
    /// At an MMA statement, having handed over its fragments, waiting for the rest of the subgroup.
    Mma { at: usize, c: Box<MmaContribution> },
}

/// The frame stack of a thread that has not started.
pub(super) fn initial_stack(k: &KernelFn) -> Vec<Frame<'_>> {
    vec![Frame::Seq {
        stmts: &k.body.0,
        pc: 0,
    }]
}

/// The frames borrow only the kernel (`'k`), not the machine, so a stack outlives one phase.
impl<'a> Machine<'a> {
    /// Run until the thread finishes or reaches a barrier. On a barrier the stack is positioned
    /// just past it, so the next call continues there. `module` resolves a `Call` (a launched module).
    pub(super) fn resume<'k: 'a>(
        &mut self,
        stack: &mut Vec<Frame<'k>>,
        module: Option<&'k Module>,
    ) -> Result<Stop, KernelTrap> {
        while let Some(top) = stack.last_mut() {
            match top {
                Frame::Seq { stmts, pc } => {
                    let stmts: &'k [Stmt] = stmts;
                    let Some(st) = stmts.get(*pc) else {
                        stack.pop();
                        continue;
                    };
                    *pc += 1;
                    self.tick()?;
                    if let Some(stop) = self.step(st, stack, module)? {
                        return Ok(stop);
                    }
                }
                Frame::Call { .. } => self.leave_call(stack, module)?,
                Frame::For { .. } => self.advance_for(stack)?,
                Frame::While { cond, body } => {
                    let (cond, body) = (*cond, *body);
                    if self.truthy(cond)? {
                        self.tick()?;
                        stack.push(Frame::Seq {
                            stmts: &body.0,
                            pc: 0,
                        });
                    } else {
                        stack.pop();
                    }
                }
            }
        }
        Ok(Stop::Done)
    }

    /// Execute one statement against the stack. `Some` when the thread must stop.
    fn step<'k: 'a>(
        &mut self,
        st: &'k Stmt,
        stack: &mut Vec<Frame<'k>>,
        module: Option<&'k Module>,
    ) -> Result<Option<Stop>, KernelTrap> {
        match st {
            Stmt::Let { .. }
            | Stmt::Load { .. }
            | Stmt::Assign { .. }
            | Stmt::Store { .. }
            | Stmt::AtomicRmw { .. }
            | Stmt::AtomicCas { .. }
            | Stmt::Fence { .. } => {
                self.exec_simple(st)?;
            }
            Stmt::Collective { .. } => return self.collective_stop(st).map(Some),
            Stmt::Mma { .. } => {
                return Ok(Some(Stop::Mma {
                    at: std::ptr::from_ref(st) as usize,
                    c: Box::new(self.mma_contribution(st)?),
                }));
            }
            Stmt::Call {
                callee,
                args,
                result,
            } => self.call_stmt(stack, module, (*callee, args, *result))?,
            Stmt::If { cond, then, els } => {
                let block = if self.truthy(cond)? { then } else { els };
                stack.push(Frame::seq(block));
            }
            Stmt::For {
                local,
                start,
                end,
                step,
                body,
            } => {
                let frame = self.enter_for(*local, [start, end, step], body)?;
                stack.push(frame);
            }
            Stmt::While { cond, body } => stack.push(Frame::While { cond, body }),
            Stmt::Break => break_loop(stack),
            Stmt::Continue => unwind_to_loop(stack),
            Stmt::Return => unwind_to_call(stack),
            Stmt::Barrier { scope, .. } => {
                return Ok(Some(Stop::Barrier {
                    scope: *scope,
                    at: std::ptr::from_ref(st) as usize,
                }));
            }
        }
        Ok(None)
    }

    /// The thread reached a collective: evaluate its contribution and stop to wait for the rest of the scope.
    fn collective_stop(&mut self, st: &Stmt) -> Result<Stop, KernelTrap> {
        let Stmt::Collective {
            kind,
            value,
            result,
            scope,
        } = st
        else {
            unreachable!("collective_stop is called for a Collective");
        };
        Ok(Stop::Collective {
            scope: *scope,
            at: std::ptr::from_ref(st) as usize,
            c: self.contribution(kind, value, *result)?,
        })
    }

    /// A `Call` against the module: push the callee's frame.
    fn call_stmt<'k: 'a>(
        &mut self,
        stack: &mut Vec<Frame<'k>>,
        module: Option<&'k Module>,
        (callee, args, result): (u32, &[CallArg], Option<u32>),
    ) -> Result<(), KernelTrap> {
        let Some(m) = module else {
            return Err(KernelTrap::Unsupported(
                "a Call outside run_launch_module (check_kernel refuses it)",
            ));
        };
        self.enter_call(stack, (callee, &m.fns[callee as usize]), (args, result))
    }

    /// Evaluate a `For`'s bounds once; the counter has not taken its first value yet.
    fn enter_for<'k>(
        &mut self,
        local: u32,
        bounds: [&ScalarExpr; 3],
        body: &'k Block,
    ) -> Result<Frame<'k>, KernelTrap> {
        let Ty::Scalar(ty) = self.local_tys[local as usize] else {
            return Err(KernelTrap::Unsupported("packed loop counter"));
        };
        let range = types::int_range(ty).ok_or(KernelTrap::Unsupported("float loop counter"))?;
        let start = self.eval_int(bounds[0], ty)?;
        let end = self.eval_int(bounds[1], ty)?;
        let step = self.eval_int(bounds[2], ty)?;
        if step == 0 {
            return Err(KernelTrap::ZeroStep);
        }
        Ok(Frame::For {
            local,
            ty,
            range,
            i: start,
            end,
            step,
            started: false,
            body,
        })
    }

    /// The `For` on top of the stack takes its next pass, or ends. A counter that would leave its
    /// type's range ends the loop.
    fn advance_for<'k>(&mut self, stack: &mut Vec<Frame<'k>>) -> Result<(), KernelTrap> {
        let Some(Frame::For {
            local,
            ty,
            range: (lo, hi),
            i,
            end,
            step,
            started,
            body,
        }) = stack.last_mut()
        else {
            unreachable!("advance_for is called with a For on top");
        };
        if *started {
            *i += *step;
            if *i < *lo || *i > *hi {
                stack.pop();
                return Ok(());
            }
        }
        *started = true;
        if !(if *step > 0 { *i < *end } else { *i > *end }) {
            stack.pop();
            return Ok(());
        }
        let (local, value, ty, body) = (*local, *i, *ty, *body);
        self.tick()?;
        self.locals[local as usize] = Some(int_val(value, ty));
        stack.push(Frame::Seq {
            stmts: &body.0,
            pc: 0,
        });
        Ok(())
    }
}

/// `Break`: leave the innermost loop.
fn break_loop(stack: &mut Vec<Frame<'_>>) {
    unwind_to_loop(stack);
    stack.pop();
}

/// A `Return` ends the running function: pop down to its call frame, or empty the stack (the thread ends).
fn unwind_to_call(stack: &mut Vec<Frame<'_>>) {
    while !matches!(stack.last(), Some(Frame::Call { .. }) | None) {
        stack.pop();
    }
}

/// Pop frames until a loop frame is on top (the checker guarantees there is one).
fn unwind_to_loop(stack: &mut Vec<Frame<'_>>) {
    while !matches!(
        stack.last(),
        Some(Frame::For { .. } | Frame::While { .. }) | None
    ) {
        stack.pop();
    }
}
