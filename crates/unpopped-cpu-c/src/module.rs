// SPDX-License-Identifier: MIT OR Apache-2.0
//! A [`Module`] of kernels as serial C (`docs/options-u3-calls.md`, slice M5): [`lower_module`].
//!
//! # Shape of the C
//!
//! One translation unit. Each function the entry reaches through calls is a `static int
//! unpopped_cpu_fn_<index>_<name>(...)` in dependency order (callees first), taking the same arguments as
//! a lone kernel (a buffer is four, a scalar one) plus `T* res` when it has a result; the entry is
//! `int unpopped_cpu_kernel_<name>(...)` exactly as [`crate::CpuC`] spells a lone kernel. Every function
//! returns the same status (`0`, `1` out of bounds, `2` zero step) and a caller returns a callee's
//! nonzero status at once. A buffer passed by alias is the caller's four arguments, forwarded; a scalar is
//! passed by value; the result local is declared up front and stored through `res` on every exit.
//!
//! # What is refused (typed, [`LowerError::UnsupportedKernel`])
//!
//! A function other than the entry that holds a barrier under a launch (the serialized thread loop cannot be
//! split inside a C function), a `Private` array (a launched entry's allocations are `Shared` only), and anything
//! `check_module` refuses.
//!
//! # Under a launch
//!
//! The entry is spelled by the serialized thread loop (`launch.rs`); a callee runs inside one thread's run, so
//! it takes the thread's ids as six extra arguments `ti0..2`, `bi0..2` (before `res`) and reads
//! `BlockDim`/`GridDim` as constants of the entry's launch. A `Shared` array passed to a callee
//! (`CallArg::Alloc`) is the block's C array `shN` with a declared shape `{N}` and stride `{1}` (`ashN`,
//! `astN`), so the callee's buffer parameter indexes it exactly as it indexes a buffer.

use unpopped::backend::{GeneratedKernel, LowerError};
use unpopped::kernel::{Block, ExternDecl, Module, Stmt, check_module};

use crate::base::{ModuleCtx, ident, refuse};
use crate::kernel::{function_text, with_prelude};

/// Lower function `entry` of `m`, and every function it calls, to one C source whose entry point is
/// [`GeneratedKernel::name`].
///
/// # Errors
///
/// [`LowerError::UnsupportedKernel`]: the module does not check, a callee holds a barrier under a launch, or a
/// function uses something this emitter has not been taught.
pub fn lower_module(m: &Module, entry: u32) -> Result<GeneratedKernel, LowerError> {
    check_module(m, entry).map_err(|e| refuse(format!("the module is ill-formed: {e:?}")))?;
    let k = &m.fns[entry as usize];
    if k.launch.is_some() {
        refuse_callee_barriers(m, entry)?;
    }
    let names: Vec<String> = m
        .fns
        .iter()
        .enumerate()
        .map(|(i, f)| {
            if i == entry as usize {
                format!("unpopped_cpu_kernel_{}", ident(&f.name))
            } else {
                format!("unpopped_cpu_fn_{i}_{}", ident(&f.name))
            }
        })
        .collect();
    let ctx = ModuleCtx {
        fns: &m.fns,
        externs: &m.externs,
        names,
        launch: crate::base::LaunchInfo::of(k),
    };
    let mut body = String::new();
    for i in postorder(m, entry) {
        let callee = i != entry as usize;
        let text = function_text(&m.fns[i], (Some(&ctx), &ctx.names[i], callee))?;
        if callee {
            body.push_str("static ");
        }
        body.push_str(&text);
        body.push('\n');
    }
    Ok(GeneratedKernel::new(
        ctx.names[entry as usize].clone(),
        with_prelude(&k.name, &body),
    ))
}

/// A callee runs inside one thread's run of the thread loop, so it cannot hold a barrier (the loop would
/// have to be split in the middle of a C function).
fn refuse_callee_barriers(m: &Module, entry: u32) -> Result<(), LowerError> {
    for (i, f) in m.fns.iter().enumerate() {
        if i != entry as usize && holds_barrier(&f.body) {
            return Err(refuse(format!(
                "function `{}` holds a Barrier and is called under a launch (the serialized thread loop cannot split a callee)",
                f.name
            )));
        }
    }
    Ok(())
}

fn holds_barrier(b: &Block) -> bool {
    b.0.iter().any(|st| match st {
        Stmt::Barrier { .. }
        | Stmt::Collective { .. }
        | Stmt::CollectiveTuple { .. }
        | Stmt::Mma { .. } => true,
        Stmt::If { then, els, .. } => holds_barrier(then) || holds_barrier(els),
        Stmt::For { body, .. } | Stmt::While { body, .. } => holds_barrier(body),
        _ => false,
    })
}

/// The functions `entry` reaches, callees before callers, the entry last.
fn postorder(m: &Module, entry: u32) -> Vec<usize> {
    fn go(f: usize, m: &Module, seen: &mut [bool], out: &mut Vec<usize>) {
        if std::mem::replace(&mut seen[f], true) {
            return;
        }
        let mut calls = Vec::new();
        collect_calls(&m.fns[f].body, &m.externs, &mut calls);
        for c in calls {
            go(c, m, seen, out);
        }
        out.push(f);
    }
    let mut out = Vec::new();
    go(entry as usize, m, &mut vec![false; m.fns.len()], &mut out);
    out
}

/// The functions `b` calls: its `Call`s, and the reference of each extern it calls (spelled as a call).
fn collect_calls(b: &Block, externs: &[ExternDecl], out: &mut Vec<usize>) {
    for st in &b.0 {
        match st {
            Stmt::Call { callee, .. } | Stmt::CallTuple { callee, .. } => {
                out.push(*callee as usize)
            }
            // a combining collective calls its combine function
            Stmt::Collective {
                kind:
                    unpopped::kernel::CollectiveKind::ReduceWith { combine }
                    | unpopped::kernel::CollectiveKind::ScanWith { combine },
                ..
            }
            | Stmt::CollectiveTuple {
                kind:
                    unpopped::kernel::CollectiveKind::ReduceWith { combine }
                    | unpopped::kernel::CollectiveKind::ScanWith { combine },
                ..
            } => out.push(*combine as usize),
            Stmt::ExternCall { decl, .. } => {
                if let Some(r) = externs.get(*decl as usize).and_then(|d| d.reference) {
                    out.push(r as usize);
                }
            }
            Stmt::If { then, els, .. } => {
                collect_calls(then, externs, out);
                collect_calls(els, externs, out);
            }
            Stmt::For { body, .. } | Stmt::While { body, .. } => {
                collect_calls(body, externs, out);
            }
            _ => {}
        }
    }
}
