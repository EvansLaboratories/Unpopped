// SPDX-License-Identifier: MIT OR Apache-2.0
//! `Backend::total_integer_ops` (0.40.7): the plan gate admits integer `Div` and an integer
//! `ConstInt` at an integer compute dtype only for a backend that declares it spells them by
//! the total rule, `docs/int-div-shift-semantics.md` (the single source for that rule).
//!
//! The unchanged-behaviour proof is the first half: a backend that does NOT override the method
//! is refused exactly as before, by `try_generate` and by `try_build_plan` alike.

use unpopped::backend::{Backend, DtypeSupport, GeneratedKernel, LowerError};
use unpopped::ir::{Expr, OpDef, ScalarExpr, input};
use unpopped::plan::{KernelPlan, PlanError, try_build_plan, try_build_plan_for};
use unpopped::try_generate;
use unpopped_vocab::{
    ArchSku, ElementKind, OpCategory, OperandDesc, StructureKey, TargetId, structure_key,
};

/// Spells everything and leaves `total_integer_ops` at its default (`false`).
struct Default_;
/// The same, but declares the total integer rule.
struct Total;

macro_rules! backend {
    ($t:ty, $total:expr) => {
        impl Backend for $t {
            fn name(&self) -> &str {
                "gate-test"
            }
            fn provider(&self) -> &str {
                "unpopped-tests"
            }
            fn supports_dtype(&self, _d: ElementKind, _t: TargetId) -> DtypeSupport {
                DtypeSupport::Supported
            }
            fn total_integer_ops(&self) -> bool {
                $total
            }
            fn lower(&self, _p: &KernelPlan<'_>) -> Result<GeneratedKernel, LowerError> {
                Ok(GeneratedKernel::new("k".into(), "/* spelled */".into()))
            }
        }
    };
}
backend!(Total, true);

// `Default_` must NOT override the method: written out by hand so the default is what runs.
impl Backend for Default_ {
    fn name(&self) -> &str {
        "gate-test"
    }
    fn provider(&self) -> &str {
        "unpopped-tests"
    }
    fn supports_dtype(&self, _d: ElementKind, _t: TargetId) -> DtypeSupport {
        DtypeSupport::Supported
    }
    fn lower(&self, _p: &KernelPlan<'_>) -> Result<GeneratedKernel, LowerError> {
        Ok(GeneratedKernel::new("k".into(), "/* spelled */".into()))
    }
}

fn key(dtype: ElementKind) -> StructureKey {
    let d = OperandDesc::new(1, &[7], &[1], dtype, 256);
    structure_key(OpCategory::BinaryElementwise, &[d, d, d], ArchSku::Sm89)
}

fn idiv(dtype: ElementKind) -> OpDef {
    OpDef::elementwise("idiv", 2, &[dtype], input(0) / input(1))
}

fn with_const(dtype: ElementKind) -> OpDef {
    let c = Expr(ScalarExpr::ConstInt(3, dtype));
    OpDef::elementwise("addc", 1, &[dtype], input(0) + c)
}

#[test]
fn a_backend_that_does_not_opt_in_is_refused_exactly_as_before() {
    for dt in [
        ElementKind::I32,
        ElementKind::I64,
        ElementKind::U32,
        ElementKind::U64,
    ] {
        for op in [idiv(dt), with_const(dt)] {
            let k = key(dt);
            let plain = try_build_plan(&op, &k).map(|_| ());
            assert!(
                matches!(plain, Err(PlanError::InadmissibleOpAtDtype { .. })),
                "{dt:?} {}: try_build_plan must refuse, got {plain:?}",
                op.name
            );
            let via_default = try_build_plan_for(&op, &k, &Default_).map(|_| ());
            assert_eq!(
                via_default, plain,
                "{dt:?} {}: default backend differs",
                op.name
            );
            let generated = try_generate(&op, &k, &Default_);
            assert!(
                matches!(generated, Err(LowerError::InadmissiblePlan { .. })),
                "{dt:?} {}: try_generate must refuse a non-opted-in backend",
                op.name
            );
        }
    }
}

#[test]
fn an_opted_in_backend_gets_int_div_and_const_int_at_the_defined_dtypes() {
    for dt in [
        ElementKind::I32,
        ElementKind::I64,
        ElementKind::U32,
        ElementKind::U64,
    ] {
        for op in [idiv(dt), with_const(dt)] {
            let k = key(dt);
            try_build_plan_for(&op, &k, &Total).expect("admitted for an opted-in backend");
            try_generate(&op, &k, &Total).expect("lowers through try_generate");
        }
    }
}

#[test]
fn opting_in_widens_nothing_else() {
    // Div at an 8-bit or 16-bit dtype stays refused (the ruling defines I32/I64/U32/U64), a
    // reduction-class Div stays refused, and a float Div is unaffected.
    for dt in [
        ElementKind::I8,
        ElementKind::U8,
        ElementKind::I16,
        ElementKind::U16,
    ] {
        let r = try_build_plan_for(&idiv(dt), &key(dt), &Total).map(|_| ());
        assert!(r.is_err(), "{dt:?}: Div must stay refused, got {r:?}");
    }
    try_build_plan_for(&idiv(ElementKind::F32), &key(ElementKind::F32), &Total)
        .expect("float Div is admitted regardless");
}
