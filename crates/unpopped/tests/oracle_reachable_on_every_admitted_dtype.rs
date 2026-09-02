//! Every dtype the plan gate ADMITS, the oracle can evaluate.
//!
//! # The gap this closes, and how it hid
//!
//! `oracle.rs`'s `Coverage` classifier enumerates `Access` **exhaustively** — the
//! compiler enforces it — and every variant now reads `Evaluated`. So the oracle
//! looked completely covered.
//!
//! **It was covered over the SCHEDULE axis. Nothing was measuring the DTYPE
//! axis.** `elem_size` handled 21 of the 25 `ElementKind` rows and panicked on
//! the other four, and the plan gate admitted all four — so
//! `unpopped::oracle::evaluate`, a `pub` function taking a `KernelPlan` and no
//! backend, **panicked on valid input in a published crate**.
//!
//! ⚠️ **An exhaustive check over the wrong axis reads exactly like an exhaustive
//! check.** The emitters refused all four, so the emission path was safe and
//! `the_reserved_dtypes_never_lower` proved it — but the oracle is a *second*
//! consumer of a plan, and no test crossed the plan gate with the oracle.
//!
//! Found 2026-09-02 by the portfolio PM asking whether the oracle's coverage was
//! complete over the op set or merely over a region that happened to be covered.

use unpopped::ir::{OpDef, input};
use unpopped::oracle::{TypedBuffer, evaluate};
use unpopped::plan::try_build_plan;
use unpopped_vocab::{ArchSku, ElementKind, OpCategory, OperandDesc, structure_key};

/// Every §6.1 row, so a new dtype cannot be added without meeting this test.
const ALL: &[ElementKind] = &[
    ElementKind::F32,
    ElementKind::F32Strict,
    ElementKind::F64,
    ElementKind::F16,
    ElementKind::Bf16,
    ElementKind::I8,
    ElementKind::U8,
    ElementKind::I16,
    ElementKind::U16,
    ElementKind::I32,
    ElementKind::U32,
    ElementKind::I64,
    ElementKind::U64,
    ElementKind::Bool,
    ElementKind::I4,
    ElementKind::U4,
    ElementKind::B1,
    ElementKind::Fp8E4M3FN,
    ElementKind::Fp8E5M2,
    ElementKind::Complex64,
    ElementKind::Complex128,
    // The four that had no evaluation and were admitted anyway.
    ElementKind::Fp8E4M3FNUZ,
    ElementKind::Fp8E5M2FNUZ,
    ElementKind::F8E8M0,
    ElementKind::F8E6M2,
];

#[test]
fn no_dtype_the_gate_admits_can_panic_the_oracle() {
    let mut admitted = 0usize;
    let mut declined = 0usize;
    for &dt in ALL {
        let op = OpDef::elementwise("probe", 2, &[dt], input(0) + input(1));
        let d = OperandDesc::new(1, &[7], &[1], dt, 4);
        let key = structure_key(OpCategory::BinaryElementwise, &[d, d, d], ArchSku::Sm89);

        let Ok(plan) = try_build_plan(&op, &key) else {
            declined += 1;
            continue;
        };
        admitted += 1;
        // Sized for the WIDEST element (Complex128 = 16 bytes), not for the
        // common case. An undersized buffer makes the oracle index out of range,
        // which is a caller error rather than the defect under test — and it
        // failed here first, which is the probe catching its own bug.
        let buf = TypedBuffer::new(dt, vec![7], vec![1], vec![0u8; 7 * 16]);
        let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _ = evaluate(&plan, &[d, d, d], &[buf.clone(), buf], &[]);
        }));
        assert!(
            outcome.is_ok(),
            "{dt:?}: the plan gate ADMITS it and `oracle::evaluate` PANICS on it. \
             `oracle` is a pub module and `evaluate` takes a plan with no backend, \
             so this is a reachable panic on valid input. Either the oracle must \
             evaluate this dtype or the gate must decline it — and the gate is the \
             right side when the dtype has no computation semantics."
        );
    }

    // VACUITY CONTROLS. A gate that declined everything, or a probe that built no
    // plan, would satisfy the loop above without evaluating anything at all.
    assert!(
        admitted >= 15,
        "only {admitted} dtypes were admitted — the probe is not reaching the \
         oracle and a green here would certify nothing"
    );
    assert!(
        declined >= 4,
        "only {declined} dtypes declined; the four non-compute rows \
         (Fp8E4M3FNUZ/Fp8E5M2FNUZ/F8E8M0/F8E6M2) must be refused BY THE GATE, \
         which is what makes the oracle's panic unreachable"
    );
}
