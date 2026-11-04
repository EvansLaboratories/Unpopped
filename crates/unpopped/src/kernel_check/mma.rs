// SPDX-License-Identifier: MIT OR Apache-2.0
//! The checker for `Stmt::Mma` (ROADMAP U7 S2, `docs/options-u7-mma-atoms.md`).
//!
//! An MMA is a rendezvous of one subgroup, so it needs a launch whose subgroup size is exactly the atom's lane
//! count. Its four fragments are `Private` arrays of exactly the atom's fragment length and element type
//! (A and B in the atom's operand types, C and D in its accumulator type). It declares no local.

use crate::exec::{Extent, MemorySpace};
use crate::kernel::{KernelError, KernelFn, Stmt};
use crate::mma::MmaAtom;

fn refuse(what: &'static str) -> KernelError {
    KernelError::MmaUnsupported { what }
}

/// Check a `Stmt::Mma`.
pub(super) fn check_mma(k: &KernelFn, st: &Stmt) -> Result<(), KernelError> {
    let Stmt::Mma {
        atom,
        a,
        b,
        c,
        d,
        extra,
    } = st
    else {
        unreachable!("check_mma is called for an Mma");
    };
    let Some(launch) = k.launch.as_ref() else {
        return Err(KernelError::ExecWithoutLaunch);
    };
    if !matches!(launch.subgroup, Some(s) if s.min == atom.lanes() && s.max == atom.lanes()) {
        return Err(refuse(
            "a launch whose subgroup size is not exactly the atom's lane count",
        ));
    }
    let (ta, tb, tacc) = atom.dtypes();
    for (which, alloc, ty) in [(0, *a, ta), (1, *b, tb), (2, *c, tacc), (2, *d, tacc)] {
        fragment(k, atom, which, alloc, ty)?;
    }
    if extra.len() != atom.extras().count() {
        return Err(refuse(
            "a statement whose extra operands are not the atom's (2:4 metadata, scales)",
        ));
    }
    for (i, ((_, ty), alloc)) in atom.extras().zip(extra).enumerate() {
        fragment(k, atom, 3 + i, *alloc, ty)?;
    }
    Ok(())
}

/// One fragment: a `Private`, statically sized allocation of `atom.values(which)` elements of `ty`.
fn fragment(
    k: &KernelFn,
    atom: &MmaAtom,
    which: usize,
    alloc: u32,
    ty: unpopped_vocab::ElementKind,
) -> Result<(), KernelError> {
    let Some(decl) = k.allocs.get(alloc as usize) else {
        return Err(refuse("a fragment that is not a declared allocation"));
    };
    if decl.space != MemorySpace::Private {
        return Err(refuse("a fragment outside Private memory"));
    }
    if decl.elem != ty {
        return Err(refuse("a fragment whose element type is not the atom's"));
    }
    if decl.extent != Extent::Static(atom.values(which)) {
        return Err(refuse(
            "a fragment whose length is not the atom's values per lane",
        ));
    }
    Ok(())
}
