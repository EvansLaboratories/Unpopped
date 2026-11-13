// SPDX-License-Identifier: MIT OR Apache-2.0
//! The checker for the [`ExternDecl`]s of a [`Module`] (ROADMAP U11, `docs/options-u11-extern-and-inline-asm.md`).
//!
//! Unpopped cannot look inside an inline-asm body, so what it checks is the declaration **against its own words**:
//! `pure` excludes every other effect, a pure call must have a result (it could only be dead otherwise), v1 has no
//! pointer operands so no memory effect can be declared, a body is of a sane shape (one per target, operand
//! placeholders in range), and a `reference` is a plain scalar kernel with exactly the declared signature. Whether
//! an asm body *means* what its reference means is not checked here; only a device leg can show that.

use std::collections::BTreeSet;

use crate::exec::Scope;
use crate::kernel::{Block, BodyText, ExternDecl, KernelError, KernelFn, Module, ParamKind, Stmt};
use crate::types::Ty;
use unpopped_vocab::ElementKind;

/// What an `ExternCall` is checked against: a declaration's parameter and result types and the effects the checker
/// and the uniformity rule read.
#[derive(Clone)]
pub(crate) struct ExternSig {
    pub(crate) params: Vec<ElementKind>,
    pub(crate) results: Vec<ElementKind>,
    /// No effect: the results are as uniform as the arguments. Otherwise they may differ between threads.
    pub(crate) pure: bool,
    /// The scope of a rendezvous, making a call a barrier site.
    pub(crate) sync: Option<Scope>,
}

fn refuse(what: &'static str) -> KernelError {
    KernelError::ExternUnsupported { what }
}

/// Check every declaration of `m`.
///
/// # Errors
///
/// The first [`KernelError`], declarations in index order.
pub(crate) fn check_extern_decls(m: &Module) -> Result<(), KernelError> {
    m.externs.iter().try_for_each(|d| check_decl(m, d))
}

fn check_decl(m: &Module, d: &ExternDecl) -> Result<(), KernelError> {
    let e = &d.effects;
    if e.pure && (e.volatile || e.sync.is_some()) {
        return Err(refuse("a pure declaration that is also volatile or sync"));
    }
    if e.pure && d.results.is_empty() {
        return Err(refuse(
            "a pure declaration with no result (it could only be dead)",
        ));
    }
    if !e.reads.is_empty() || !e.writes.is_empty() {
        return Err(refuse(
            "a declared memory effect (reads and writes need pointer operands, which v1 has not)",
        ));
    }
    if e.sync == Some(Scope::Thread) {
        return Err(refuse(
            "a sync at Thread scope (it rendezvouses with no one)",
        ));
    }
    check_bodies(d)?;
    d.reference.map_or(Ok(()), |r| check_reference(m, d, r))
}

fn check_bodies(d: &ExternDecl) -> Result<(), KernelError> {
    let operands = d.params.len() + d.results.len();
    let mut seen = BTreeSet::new();
    for b in &d.bodies {
        if b.target.is_empty() {
            return Err(refuse("a body with an empty target name"));
        }
        if !seen.insert(b.target.as_str()) {
            return Err(refuse("two bodies for one target"));
        }
        match &b.body {
            BodyText::Asm { template, .. } => {
                if template.trim().is_empty() {
                    return Err(refuse("an asm body with an empty template"));
                }
                if placeholders(template).any(|i| i >= operands) {
                    return Err(refuse(
                        "an asm placeholder past the declaration's parameters and results",
                    ));
                }
            }
            BodyText::Symbol { library, symbol } => {
                if library.is_empty() || symbol.is_empty() {
                    return Err(refuse("a symbol body with an empty library or symbol"));
                }
            }
            #[allow(unreachable_patterns, reason = "BodyText is non_exhaustive")]
            _ => return Err(refuse("a body of a kind this checker does not know")),
        }
    }
    Ok(())
}

/// The operand numbers of the `{N}` placeholders of an asm template.
fn placeholders(template: &str) -> impl Iterator<Item = usize> + '_ {
    template.split('{').skip(1).filter_map(|rest| {
        let (digits, tail) = rest.split_at(rest.find(|c: char| !c.is_ascii_digit())?);
        if digits.is_empty() || !tail.starts_with('}') {
            return None;
        }
        digits.parse().ok()
    })
}

/// The reference is a plain scalar kernel with the declared signature and no extern call of its own.
fn check_reference(m: &Module, d: &ExternDecl, index: u32) -> Result<(), KernelError> {
    let Some(f) = m.fns.get(index as usize) else {
        return Err(refuse("a reference that is not a function of the module"));
    };
    if f.launch.is_some() || !f.allocs.is_empty() {
        return Err(refuse(
            "a reference with a launch shape or allocations (it runs on the calling thread)",
        ));
    }
    if holds_extern_call(&f.body) {
        return Err(refuse("a reference that itself calls an extern"));
    }
    let scalar_types: Option<Vec<_>> = f
        .params
        .iter()
        .map(|p| match &p.kind {
            ParamKind::Scalar { ty, .. } => Some(*ty),
            _ => None,
        })
        .collect();
    if scalar_types.as_deref() != Some(d.params.as_slice()) {
        return Err(refuse(
            "a reference whose parameters are not the declared scalar parameters",
        ));
    }
    match (d.results.as_slice(), f.result) {
        ([], None) => Ok(()),
        ([want], Some(local)) => {
            if reference_result(f, local) == Some(Ty::Scalar(*want)) {
                Ok(())
            } else {
                Err(refuse(
                    "a reference whose result type is not the declared result",
                ))
            }
        }
        ([_, _, ..], _) => Err(refuse(
            "a reference for a tuple result (a kernel has one result; not run yet)",
        )),
        _ => Err(refuse(
            "a reference whose result presence differs from the declaration",
        )),
    }
}

fn reference_result(f: &KernelFn, local: u32) -> Option<Ty> {
    f.locals.get(local as usize).map(|d| d.ty())
}

fn holds_extern_call(b: &Block) -> bool {
    b.0.iter().any(|st| match st {
        Stmt::ExternCall { .. } => true,
        Stmt::If { then, els, .. } => holds_extern_call(then) || holds_extern_call(els),
        Stmt::For { body, .. } | Stmt::While { body, .. } => holds_extern_call(body),
        _ => false,
    })
}
