// SPDX-License-Identifier: MIT OR Apache-2.0
//! The checker for `Stmt::CopyAsync`, `AsyncCommit` and `AsyncWait` (ROADMAP U8 S1,
//! `docs/options-u8-copy-atoms.md`): the destination is a `Shared` allocation place, the source a readable global
//! buffer place of the same element type, and the copy is 4, 8 or 16 bytes. Whether a wait covers a copy, and whether
//! the destination is read too early, depend on run-time control flow, so the oracle traps on those
//! (`AsyncRace`, `AsyncUnwaited`); the checker does not guess.

use super::{Scope, check_place};
use crate::exec::MemorySpace;
use crate::kernel::{KernelError, KernelFn, Place, Stmt};
use crate::oracle::elem_bits;

fn refuse(what: &'static str) -> KernelError {
    KernelError::AsyncUnsupported { what }
}

/// Check an async-copy statement.
pub(super) fn check_async(k: &KernelFn, st: &Stmt, sc: &Scope) -> Result<(), KernelError> {
    let Stmt::CopyAsync { dst, src, elems } = st else {
        return Ok(());
    };
    let Place::Alloc { alloc, .. } = dst else {
        return Err(refuse(
            "a CopyAsync destination that is not a Shared allocation place",
        ));
    };
    if k.allocs
        .get(*alloc as usize)
        .is_none_or(|a| a.space != MemorySpace::Shared)
    {
        return Err(refuse(
            "a CopyAsync destination that is not a Shared allocation place",
        ));
    }
    let Place::Logical { buffer, .. } = src else {
        return Err(refuse(
            "a CopyAsync source that is not a global buffer place",
        ));
    };
    let (de, _) = check_place(k, dst, sc)?;
    let (se, role) = check_place(k, src, sc)?;
    if !role.reads() {
        return Err(KernelError::ReadFromWriteOnly { buffer: *buffer });
    }
    if de != se {
        return Err(KernelError::TypeMismatch {
            what: "a CopyAsync destination and source element type",
        });
    }
    let bits = elem_bits(de) * *elems as usize;
    if bits % 8 != 0 || !matches!(bits / 8, 4 | 8 | 16) {
        return Err(refuse("a CopyAsync of other than 4, 8 or 16 bytes"));
    }
    Ok(())
}
