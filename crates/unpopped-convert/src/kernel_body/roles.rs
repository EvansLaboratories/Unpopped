// SPDX-License-Identifier: MIT OR Apache-2.0
//! Buffer roles from use (ROADMAP U19 slice 5). C marks a read-only buffer `const`, but real code often does not (`half* q`
//! that the kernel only reads), and a role of `ReadWrite` everywhere hides what the kernel does from the checker: its
//! barrier-uniformity rule treats a load from a buffer no statement writes as uniform when the index is, which the decode
//! kernel's `block_valid_mask[bx]` early exit needs. After the body is built, a non-`const` buffer **no statement stores to**
//! becomes `Read`. The scan knows only the statements the walker emits; any other kind keeps every role as declared.

use std::collections::BTreeSet;
use unpopped::kernel::{Block, BufferRole, KernelFn, ParamKind, Place, Stmt};

/// Collect the buffers written by `b`. `false` if `b` holds a statement the scan does not understand.
fn writes(b: &Block, out: &mut BTreeSet<u32>) -> bool {
    let mut known = true;
    for s in &b.0 {
        match s {
            Stmt::Store { place, .. }
            | Stmt::AtomicRmw { place, .. }
            | Stmt::AtomicCas { place, .. } => {
                if let Place::Logical { buffer, .. } | Place::Raw { buffer, .. } = place {
                    out.insert(*buffer);
                } else if !matches!(place, Place::Alloc { .. } | Place::AllocRaw { .. }) {
                    known = false;
                }
            }
            Stmt::If { then, els, .. } => {
                known &= writes(then, out);
                known &= writes(els, out);
            }
            Stmt::For { body, .. } | Stmt::While { body, .. } => known &= writes(body, out),
            Stmt::Let { .. }
            | Stmt::Assign { .. }
            | Stmt::Load { .. }
            | Stmt::Barrier { .. }
            | Stmt::CopyAsync { .. }
            | Stmt::AsyncCommit
            | Stmt::AsyncWait { .. }
            | Stmt::Collective { .. }
            | Stmt::GridDepWait
            | Stmt::GridDepLaunchDependents
            | Stmt::Return
            | Stmt::Break
            | Stmt::Continue => {}
            _ => known = false,
        }
    }
    known
}

/// Mark every buffer parameter that no statement writes as `Read`.
pub(super) fn refine_roles(k: &mut KernelFn) {
    let mut written = BTreeSet::new();
    if !writes(&k.body, &mut written) {
        return;
    }
    for (i, p) in k.params.iter_mut().enumerate() {
        if let ParamKind::Buffer { role, .. } = &mut p.kind
            && *role == BufferRole::ReadWrite
            && !written.contains(&u32::try_from(i).unwrap_or(u32::MAX))
        {
            *role = BufferRole::Read;
        }
    }
}
