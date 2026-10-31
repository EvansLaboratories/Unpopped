// SPDX-License-Identifier: MIT OR Apache-2.0
//! **`Backend::lower_kernel`'s default is the typed kernel decline, for every backend that does not
//! override it** (PM ruling 2026-10-30: the method is additive, so Baracuda and Fuel compile
//! unchanged; the default must not panic, must not return `Ok`, and must not be
//! `UnsupportedOp`, which means "no spelling for an expression node").
//!
//! Every in-tree backend is checked: `CpuC` (overridden, so a kernel outside its coverage reaches
//! the variant by its own path) and a bare backend that implements only the required methods, which
//! is what a downstream backend looks like and which exercises the default itself. The Slang
//! emitter's half of this check lives with it, in Vulkane's `vulkane-slang-emit`.

use std::panic::{AssertUnwindSafe, catch_unwind};

use unpopped::backend::{Backend, DtypeSupport, GeneratedKernel, LowerError};
use unpopped::exec::{LaunchShape, Scope};
use unpopped::kernel::{Block, KernelFn, Stmt};
use unpopped::plan::KernelPlan;
use unpopped_cpu_c::CpuC;
use unpopped_vocab::{ElementKind, TargetId};

/// A downstream-shaped backend: only what the trait requires.
struct Bare;

impl Backend for Bare {
    fn name(&self) -> &str {
        "bare"
    }
    fn provider(&self) -> &str {
        "test"
    }
    fn supports_dtype(&self, _: ElementKind, _: TargetId) -> DtypeSupport {
        DtypeSupport::Supported
    }
    fn lower(&self, _: &KernelPlan<'_>) -> Result<GeneratedKernel, LowerError> {
        Err(LowerError::UnsupportedSchedule {
            detail: "n/a".into(),
        })
    }
}

fn empty() -> KernelFn {
    KernelFn::new("k", ElementKind::F32, vec![], vec![], Block(vec![]))
}

/// A kernel `CpuC` must refuse: it has a launch and a barrier.
fn barrier() -> KernelFn {
    let mut k = empty().with_launch(LaunchShape::new([1, 1, 1], [1, 1, 1]));
    k.body.0.push(Stmt::Barrier {
        scope: Scope::Block,
    });
    k
}

fn typed_decline(who: &str, backend: &dyn Backend, k: &KernelFn) {
    let r = catch_unwind(AssertUnwindSafe(|| backend.lower_kernel(k)))
        .unwrap_or_else(|_| panic!("{who}: lower_kernel panicked"));
    match r {
        Err(LowerError::UnsupportedKernel { detail }) => {
            assert!(!detail.is_empty(), "{who}: the decline carries no reason");
        }
        Err(other) => panic!("{who}: expected UnsupportedKernel, got {other:?}"),
        Ok(g) => panic!("{who}: expected a decline, got a kernel `{}`", g.name),
    }
}

#[test]
fn a_backend_without_a_kernel_path_declines_with_the_typed_variant() {
    typed_decline("bare (the default)", &Bare, &empty());
}

#[test]
fn the_default_names_the_backend_so_the_decline_is_attributable() {
    let Err(LowerError::UnsupportedKernel { detail }) = Bare.lower_kernel(&empty()) else {
        panic!("not the typed decline");
    };
    assert!(detail.contains("bare"), "{detail}");
}

#[test]
fn an_overriding_backend_refuses_what_it_does_not_cover_with_the_same_variant() {
    typed_decline("cpu_c (outside its coverage)", &CpuC, &barrier());
}
