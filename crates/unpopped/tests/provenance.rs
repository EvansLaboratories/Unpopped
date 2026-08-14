//! A generated kernel names what it was baked against.
//!
//! Every test here carries a **negative control** — that a kernel baked against a
//! *different* cell gets a *different* stamp. Asserting only that a stamp is
//! present, or only that two matching things match, certifies nothing: a stamp
//! that returned a constant would pass every positive assertion. Fuel hit exactly
//! this while closing their KV-allocation staleness bug — their first tests
//! compared graph-local `NodeId`s that were re-minted on rebuild, so the check
//! could not distinguish reuse from rebuild in *either* direction, and one test
//! was passing vacuously. A cross-project conformance test with that defect
//! certifies nothing while looking green.

mod common;
use common::{OtherStub, StubBackend};
use unpopped::backend::Backend;
use unpopped::ir::{OpDef, input};
use unpopped::{generate, generate_variants};
use unpopped_vocab::{ArchSku, ElementKind, OpCategory, OperandDesc, structure_key};

fn add_op() -> OpDef {
    OpDef::elementwise("add", 2, &[ElementKind::F32], input(0) + input(1))
}

fn key_for(n: i64, dtype: ElementKind) -> unpopped_vocab::StructureKey {
    let d = OperandDesc::new(1, &[n], &[1], dtype, 256);
    structure_key(OpCategory::BinaryElementwise, &[d, d, d], ArchSku::Sm89)
}

#[test]
fn generate_stamps_the_cell_it_was_asked_for() {
    let key = key_for(7, ElementKind::F32);
    let k = generate(&add_op(), &key, &StubBackend);

    let p = k
        .provenance()
        .expect("a kernel from `generate` must carry its validity key");
    assert_eq!(p.structure_key, key.to_token());
    assert!(
        p.generator.starts_with("unpopped "),
        "generator must name the producer and its version, got {:?}",
        p.generator
    );

    // The stamp names *whichever* backend produced it. This asserted the literal
    // `"cpu_c"` while the emitter lived in core — which made a property of
    // `generate` read as a fact about C, and would have been satisfied by a
    // stamper that returned a constant. Two backends with different names is the
    // control that distinguishes "reports the backend" from "reports a string".
    assert_eq!(p.backend, StubBackend.name());
    assert_eq!(p.provider, StubBackend.provider());

    let other = generate(&add_op(), &key, &OtherStub);
    let q = other.provenance().expect("stamped");
    assert_eq!(q.backend, OtherStub.name());
    assert_ne!(
        p.backend, q.backend,
        "two different backends must not stamp the same name"
    );
    // Same cell, different backend: the CELL half of the stamp must be identical
    // and the BACKEND half different. A stamp that mixed them would fail here.
    assert_eq!(p.structure_key, q.structure_key);
}

/// The negative control for the cell identity.
///
/// Without this, `generate_stamps_the_cell_it_was_asked_for` is satisfied by a
/// stamp that always returns the same token.
#[test]
fn a_different_cell_gets_a_different_stamp() {
    let op = add_op();
    let f32_key = key_for(7, ElementKind::F32);
    let i32_key = key_for(7, ElementKind::I32);
    assert_ne!(
        f32_key.to_token(),
        i32_key.to_token(),
        "harness precondition: the two cells must actually differ"
    );

    let a = generate(&op, &f32_key, &StubBackend);
    let b = generate(&op, &i32_key, &StubBackend);

    assert_ne!(
        a.provenance().unwrap().structure_key,
        b.provenance().unwrap().structure_key,
        "two different cells must not share a validity key — a stamp that cannot \
         tell them apart is worse than no stamp, because a cache would trust it"
    );
}

/// A raw backend fragment is unstamped, and that distinction is the point.
///
/// `GeneratedKernel::new` is what an out-of-crate backend calls; it cannot stamp,
/// because the private field and `pub(crate)` setter make it unrepresentable
/// rather than merely discouraged.
#[test]
fn a_backend_fragment_is_unstamped() {
    let raw = unpopped::backend::GeneratedKernel::new("k".into(), "void k(){}".into());
    assert!(
        raw.provenance().is_none(),
        "a fragment a backend built directly has no cell identity to claim"
    );
}

/// Every kernel of every variant is stamped — not just the first one.
///
/// A split-K variant ships two kernels under one cell. Both are cacheable, so a
/// guarantee that covered only `kernels[0]` would hold for whichever one the
/// caller happened to inspect.
#[test]
fn every_variant_kernel_is_stamped() {
    let key = key_for(7, ElementKind::F32);
    let vs = generate_variants(&add_op(), &key, &StubBackend);
    assert!(
        !vs.is_empty(),
        "harness precondition: at least a base variant"
    );

    let mut seen = 0;
    for v in &vs {
        assert!(
            !v.kernels.is_empty(),
            "variant {:?} shipped no kernels",
            v.tag
        );
        for k in &v.kernels {
            let p = k
                .provenance()
                .unwrap_or_else(|| panic!("variant {:?} kernel {:?} unstamped", v.tag, k.name));
            assert_eq!(p.structure_key, key.to_token());
            seen += 1;
        }
    }
    assert!(seen > 0, "no kernels were checked — vacuous pass");
}
