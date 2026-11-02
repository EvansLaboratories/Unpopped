// SPDX-License-Identifier: MIT OR Apache-2.0
//! **Running a call** (`docs/options-u3-calls.md`, Q3 and Q4, slice M3): [`run_module`] and the recursive
//! evaluator's `Stmt::Call`. A call is a frame swap on the one machine: the callee's function, locals and
//! buffer map replace the caller's for the length of its body and are put back after, so every access in
//! the callee goes through the same argument storage as the caller's (a buffer passed by alias *is* the
//! caller's buffer). Unlaunched entries only; a launched module is the resumable evaluator's (slice M4).

use super::kernel_eval::{ALLOC_BIND, KernelArg, KernelTrap, Machine};
use super::kernel_resume::Frame;
use crate::kernel::{CallArg, KernelFn, Module, ParamKind, check_module};
use crate::types::Ty;

/// Run function `entry` of `m` on `args` (one per parameter, in order). Buffers are updated in place.
///
/// # Errors
///
/// A [`KernelTrap`]: `Malformed` if [`check_module`] refuses the module.
pub fn run_module(m: &Module, entry: u32, args: &mut [KernelArg]) -> Result<(), KernelTrap> {
    check_module(m, entry).map_err(KernelTrap::Malformed)?;
    let k = &m.fns[entry as usize];
    if k.launch.is_some() || !k.allocs.is_empty() {
        return Err(KernelTrap::Unsupported(
            "a launched or allocating entry is not run by run_module yet",
        ));
    }
    if args.len() != k.params.len() {
        return Err(KernelTrap::ArgMismatch("one argument per parameter"));
    }
    Machine::new(k, args, None).with_module(m, entry).run_body()
}

/// The caller's frame while a callee runs.
struct Saved<'a> {
    k: &'a KernelFn,
    local_tys: Vec<Ty>,
    locals: Vec<Option<super::Val>>,
    bufmap: Vec<usize>,
    fn_idx: u32,
}

/// The caller's side of a call, evaluated before the frame swap: the scalar values by callee local, and
/// the root buffer each callee parameter names.
pub(super) struct Bound {
    pub(super) scalars: Vec<(u32, super::Val)>,
    pub(super) bufmap: Vec<usize>,
}

impl<'a> Machine<'a> {
    /// `result = callee(args...)` on the calling thread.
    pub(super) fn exec_call(
        &mut self,
        callee: u32,
        args: &[CallArg],
        result: Option<u32>,
    ) -> Result<(), KernelTrap> {
        let Some(m) = self.module else {
            return Err(KernelTrap::Unsupported(
                "a Call outside run_module (check_kernel refuses it)",
            ));
        };
        let f: &'a KernelFn = &m.fns[callee as usize];
        if !f.allocs.is_empty() {
            return Err(KernelTrap::Unsupported("a callee with allocations"));
        }
        let bound = self.bind_call(f, args)?;
        let value = self.run_frame((callee, f), bound)?;
        if let (Some(dst), Some(v)) = (result, value) {
            self.locals[dst as usize] = Some(v);
        }
        Ok(())
    }

    /// Evaluate the arguments in the caller's frame.
    pub(super) fn bind_call(
        &mut self,
        f: &KernelFn,
        args: &[CallArg],
    ) -> Result<Bound, KernelTrap> {
        let mut bound = Bound {
            scalars: Vec::new(),
            bufmap: Vec::with_capacity(f.params.len()),
        };
        for (a, p) in args.iter().zip(&f.params) {
            match (a, &p.kind) {
                (CallArg::Scalar(e), ParamKind::Scalar { local, .. }) => {
                    bound.scalars.push((*local, self.eval_expr(e)?));
                    bound.bufmap.push(usize::MAX);
                }
                (CallArg::Buffer(b), ParamKind::Buffer { .. }) => {
                    bound.bufmap.push(self.root(*b));
                }
                (CallArg::Alloc(a), ParamKind::Buffer { .. }) => {
                    bound.bufmap.push(ALLOC_BIND + *a as usize);
                }
                _ => return Err(KernelTrap::ArgMismatch("a call argument and its parameter")),
            }
        }
        Ok(bound)
    }

    /// The resumable evaluator's call: push the caller's frame and the callee's body, so a thread can stop
    /// inside the callee (at a barrier) and be resumed there.
    pub(super) fn enter_call<'k: 'a>(
        &mut self,
        stack: &mut Vec<Frame<'k>>,
        (callee, f): (u32, &'k KernelFn),
        (args, result): (&[CallArg], Option<u32>),
    ) -> Result<(), KernelTrap> {
        if !f.allocs.is_empty() {
            return Err(KernelTrap::Unsupported("a callee with allocations"));
        }
        let bound = self.bind_call(f, args)?;
        let locals = std::mem::replace(&mut self.locals, vec![None; f.locals.len()]);
        stack.push(Frame::Call {
            caller: self.fn_idx,
            locals,
            bufmap: std::mem::replace(&mut self.bufmap, bound.bufmap),
            result,
        });
        self.k = f;
        self.fn_idx = callee;
        self.local_tys = f.locals.iter().map(|d| d.ty()).collect();
        for (local, v) in bound.scalars {
            self.locals[local as usize] = Some(v);
        }
        stack.push(Frame::Seq {
            stmts: &f.body.0,
            pc: 0,
        });
        Ok(())
    }

    /// The callee's body has ended: put the caller's frame back and hand over the result.
    pub(super) fn leave_call<'k: 'a>(
        &mut self,
        stack: &mut Vec<Frame<'k>>,
        module: Option<&'k Module>,
    ) -> Result<(), KernelTrap> {
        let (
            Some(Frame::Call {
                caller,
                locals,
                bufmap,
                result,
            }),
            Some(module),
        ) = (stack.pop(), module)
        else {
            unreachable!("leave_call is called with a Call frame on top and a module");
        };
        let value = self.k.result.map(|l| self.locals[l as usize]);
        let k = &module.fns[caller as usize];
        self.k = k;
        self.local_tys = k.locals.iter().map(|d| d.ty()).collect();
        self.locals = locals;
        self.bufmap = bufmap;
        self.fn_idx = caller;
        match (result, value) {
            (_, Some(None)) => Err(KernelTrap::Unsupported(
                "a callee returned without setting its result local",
            )),
            (Some(dst), Some(Some(v))) => {
                self.locals[dst as usize] = Some(v);
                Ok(())
            }
            _ => Ok(()),
        }
    }

    /// Install `f`'s frame (fresh locals, `bufmap` for its buffer parameters); return the caller's.
    fn swap_frame(&mut self, (callee, f): (u32, &'a KernelFn), bufmap: Vec<usize>) -> Saved<'a> {
        Saved {
            k: std::mem::replace(&mut self.k, f),
            local_tys: std::mem::replace(
                &mut self.local_tys,
                f.locals.iter().map(|d| d.ty()).collect(),
            ),
            locals: std::mem::replace(&mut self.locals, vec![None; f.locals.len()]),
            bufmap: std::mem::replace(&mut self.bufmap, bufmap),
            fn_idx: std::mem::replace(&mut self.fn_idx, callee),
        }
    }

    fn restore_frame(&mut self, s: Saved<'a>) {
        self.k = s.k;
        self.local_tys = s.local_tys;
        self.locals = s.locals;
        self.bufmap = s.bufmap;
        self.fn_idx = s.fn_idx;
    }

    /// Run `f`'s body in a fresh frame and return the value of its result local. The caller's frame is
    /// restored whatever happens.
    fn run_frame(
        &mut self,
        (callee, f): (u32, &'a KernelFn),
        bound: Bound,
    ) -> Result<Option<super::Val>, KernelTrap> {
        let saved = self.swap_frame((callee, f), bound.bufmap);
        for (local, v) in bound.scalars {
            self.locals[local as usize] = Some(v);
        }
        let ran = self.exec_block(&f.body);
        let value = f.result.map(|l| self.locals[l as usize]);
        self.restore_frame(saved);
        ran?;
        match (f.result, value) {
            (Some(_), Some(None)) => Err(KernelTrap::Unsupported(
                "a callee returned without setting its result local",
            )),
            (_, v) => Ok(v.flatten()),
        }
    }
}
