// SPDX-License-Identifier: MIT OR Apache-2.0
//! **Running a call** (`docs/options-u3-calls.md`, Q3 and Q4, slice M3): [`run_module`] and the recursive
//! evaluator's `Stmt::Call`. A call is a frame swap on the one machine: the callee's function, locals and
//! buffer map replace the caller's for the length of its body and are put back after, so every access in
//! the callee goes through the same argument storage as the caller's (a buffer passed by alias *is* the
//! caller's buffer). Unlaunched entries only; a launched module is the resumable evaluator's (slice M4).

use super::kernel_eval::{KernelArg, KernelTrap, Machine};
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
    Machine::new(k, args, None).with_module(m).run_body()
}

/// The caller's frame while a callee runs.
struct Saved<'a> {
    k: &'a KernelFn,
    local_tys: Vec<Ty>,
    locals: Vec<Option<super::Val>>,
    bufmap: Vec<usize>,
}

/// The caller's side of a call, evaluated before the frame swap: the scalar values by callee local, and
/// the root buffer each callee parameter names.
struct Bound {
    scalars: Vec<(u32, super::Val)>,
    bufmap: Vec<usize>,
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
        let value = self.run_frame(f, bound)?;
        if let (Some(dst), Some(v)) = (result, value) {
            self.locals[dst as usize] = Some(v);
        }
        Ok(())
    }

    /// Evaluate the arguments in the caller's frame.
    fn bind_call(&mut self, f: &KernelFn, args: &[CallArg]) -> Result<Bound, KernelTrap> {
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
                (CallArg::Alloc(_), _) => {
                    return Err(KernelTrap::Unsupported("an allocation passed to a callee"));
                }
                _ => return Err(KernelTrap::ArgMismatch("a call argument and its parameter")),
            }
        }
        Ok(bound)
    }

    /// Install `f`'s frame (fresh locals, `bufmap` for its buffer parameters); return the caller's.
    fn swap_frame(&mut self, f: &'a KernelFn, bufmap: Vec<usize>) -> Saved<'a> {
        Saved {
            k: std::mem::replace(&mut self.k, f),
            local_tys: std::mem::replace(
                &mut self.local_tys,
                f.locals.iter().map(|d| d.ty()).collect(),
            ),
            locals: std::mem::replace(&mut self.locals, vec![None; f.locals.len()]),
            bufmap: std::mem::replace(&mut self.bufmap, bufmap),
        }
    }

    fn restore_frame(&mut self, s: Saved<'a>) {
        self.k = s.k;
        self.local_tys = s.local_tys;
        self.locals = s.locals;
        self.bufmap = s.bufmap;
    }

    /// Run `f`'s body in a fresh frame and return the value of its result local. The caller's frame is
    /// restored whatever happens.
    fn run_frame(
        &mut self,
        f: &'a KernelFn,
        bound: Bound,
    ) -> Result<Option<super::Val>, KernelTrap> {
        let saved = self.swap_frame(f, bound.bufmap);
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
