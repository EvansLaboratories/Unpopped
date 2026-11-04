// SPDX-License-Identifier: MIT OR Apache-2.0
//! The checker for `Stmt::ExternCall` (ROADMAP U11, `docs/options-u11-extern-and-inline-asm.md`): the declaration
//! exists in the module, the arguments are the declared scalar types in order, the result locals are declared by the
//! statement with the declared result types, and a `sync` effect names a scope the launch has.

use super::{ExternSig, Scope, check_expr, declare, scope_ok};
use crate::ir::ScalarExpr;
use crate::kernel::{KernelError, KernelFn};
use crate::types::Ty;

fn refuse(what: &'static str) -> KernelError {
    KernelError::ExternUnsupported { what }
}

/// Check a `Stmt::ExternCall` and declare its result locals.
pub(super) fn check_extern_call(
    k: &KernelFn,
    (decl, args, results): (u32, &[ScalarExpr], &[u32]),
    sc: &mut Scope,
    declared_here: &mut Vec<usize>,
) -> Result<(), KernelError> {
    if sc.callees.is_empty() {
        return Err(refuse(
            "an ExternCall outside a module: check it with check_module",
        ));
    }
    let Some(sig) = sc.externs.get(decl as usize).cloned() else {
        return Err(KernelError::BadExtern { decl });
    };
    check_signature(k, (args, results), &sig, sc)?;
    if sig.sync.is_some_and(|scope| !scope_ok(k, scope)) {
        return Err(KernelError::SyncUnsupported {
            what: "an extern's sync scope the launch does not have",
        });
    }
    for (local, ty) in results.iter().zip(&sig.results) {
        declare(k, *local, Ty::Scalar(*ty), sc, declared_here)?;
    }
    Ok(())
}

fn check_signature(
    k: &KernelFn,
    (args, results): (&[ScalarExpr], &[u32]),
    sig: &ExternSig,
    sc: &Scope,
) -> Result<(), KernelError> {
    if args.len() != sig.params.len() {
        return Err(refuse(
            "an ExternCall with a different number of arguments than the declaration has parameters",
        ));
    }
    for (a, want) in args.iter().zip(&sig.params) {
        if check_expr(k, a, sc)? != Ty::Scalar(*want) {
            return Err(KernelError::TypeMismatch {
                what: "an ExternCall argument and its declared parameter",
            });
        }
    }
    if results.len() != sig.results.len() {
        return Err(refuse(
            "an ExternCall with a different number of result locals than the declaration has results",
        ));
    }
    Ok(())
}
