//! Complex as a first-class oracle domain — `c64` and `c128`.
//!
//! # Why first-class rather than a pair of reals
//!
//! Modelling complex as two reals threaded through the float path would have put
//! a "which half am I" parameter on every node, and the operations that make
//! complex *complex* — `Mul` mixing both components, the absence of an ordering —
//! would have lived as special cases inside code whose type says it handles one
//! real number. `Val::Complex(re, im)` puts them where they can be seen.
//!
//! # Precision
//!
//! `f64` components are exact for **both** dtypes: `c64` is a pair of `f32`
//! (every value of which is an exact `f64`), `c128` a pair of `f64`. So unlike
//! the wide integers, complex needed a new *domain* rather than a wider one.

use unpopped::oracle::{Fidelity, TypedBuffer, compare};

#[test]
fn complex_buffers_round_trip_exactly() {
    let c64 = TypedBuffer::from_complex64(&[3], &[(1.5, -2.25), (0.0, 0.0), (-3.75, 4.5)]);
    assert_eq!(
        c64.to_complex_vec(),
        vec![(1.5, -2.25), (0.0, 0.0), (-3.75, 4.5)]
    );

    // c128 carries values no f32 can hold, which is the point of the wider dtype.
    let big = 1.234_567_890_123_456_7e300;
    let c128 = TypedBuffer::from_complex128(&[2], &[(big, -big), (f64::MIN_POSITIVE, 1.0)]);
    assert_eq!(
        c128.to_complex_vec(),
        vec![(big, -big), (f64::MIN_POSITIVE, 1.0)]
    );
}

/// The two dtypes are 8 and 16 bytes — a pair each, named by TOTAL width.
///
/// `c128` is why the oracle's raw storage carrier had to widen from `u64` to
/// `u128`: a 16-byte element does not fit the old one, and `read_le` was
/// `[0u8; 8]`.
#[test]
fn the_element_sizes_are_pairs_named_by_total_width() {
    let c64 = TypedBuffer::from_complex64(&[4], &[(1.0, 2.0); 4]);
    let c128 = TypedBuffer::from_complex128(&[4], &[(1.0, 2.0); 4]);
    assert_eq!(c64.to_complex_vec().len(), 4);
    assert_eq!(c128.to_complex_vec().len(), 4);
    // Distinct storage patterns per element prove the stride is right rather
    // than every element aliasing the first.
    let mixed = TypedBuffer::from_complex128(&[2], &[(1.0, 2.0), (3.0, 4.0)]);
    assert_eq!(mixed.to_complex_vec(), vec![(1.0, 2.0), (3.0, 4.0)]);
}

/// Comparison is **component-wise**, not by magnitude.
///
/// `(0, 5)` and `(5, 0)` have equal magnitude and are completely different
/// values. Complex has no ordering, so magnitude is the only scalar a naive
/// comparator would reach for — and it would pass this pair.
#[test]
fn comparison_is_component_wise_not_by_magnitude() {
    let a = TypedBuffer::from_complex128(&[1], &[(0.0, 5.0)]);
    let b = TypedBuffer::from_complex128(&[1], &[(5.0, 0.0)]);

    assert!(
        compare(&a, &b, Fidelity::Tolerant { rel: 0.0, abs: 0.0 }).is_err(),
        "equal magnitude must not mean equal value"
    );
    assert!(compare(&a, &a, Fidelity::Tolerant { rel: 0.0, abs: 0.0 }).is_ok());

    // A deviation in the IMAGINARY part alone must be caught — the half a
    // real-projecting comparator would silently drop.
    let c = TypedBuffer::from_complex128(&[1], &[(1.0, 1.0)]);
    let d = TypedBuffer::from_complex128(&[1], &[(1.0, 1.5)]);
    assert!(
        compare(&c, &d, Fidelity::Tolerant { rel: 0.0, abs: 0.1 }).is_err(),
        "an imaginary-only deviation outside tolerance must be rejected"
    );
    assert!(
        compare(&c, &d, Fidelity::Tolerant { rel: 0.0, abs: 0.6 }).is_ok(),
        "control: the same deviation inside tolerance must pass"
    );
}

/// Bit-exact comparison works on 16-byte elements.
///
/// The widened carrier has to reach `bits_at`, which `BitExact` walks.
#[test]
fn bit_exact_comparison_handles_16_byte_elements() {
    let a = TypedBuffer::from_complex128(&[2], &[(1.0, 2.0), (3.0, 4.0)]);
    let b = TypedBuffer::from_complex128(&[2], &[(1.0, 2.0), (3.0, 4.0)]);
    let c = TypedBuffer::from_complex128(&[2], &[(1.0, 2.0), (3.0, 4.5)]);
    assert!(compare(&a, &b, Fidelity::BitExact).is_ok());
    assert!(
        compare(&a, &c, Fidelity::BitExact).is_err(),
        "a differing high-half component must be visible to a bit comparison"
    );
}

/// Signed zero survives, which a magnitude- or sum-based model would lose.
#[test]
fn signed_zero_is_preserved_per_component() {
    let pos = TypedBuffer::from_complex128(&[1], &[(0.0, 0.0)]);
    let neg = TypedBuffer::from_complex128(&[1], &[(-0.0, -0.0)]);
    assert!(
        compare(&pos, &neg, Fidelity::BitExact).is_err(),
        "+0 and -0 differ in bits"
    );
    // But they compare equal numerically, as IEEE says they must.
    assert!(compare(&pos, &neg, Fidelity::Tolerant { rel: 0.0, abs: 0.0 }).is_ok());
}

// ---------------------------------------------------------------------------
// Plan side: what a complex cell may compute, and what it must refuse.
// ---------------------------------------------------------------------------

use unpopped::ir::{BinaryOp, OpDef, input};
use unpopped::oracle::evaluate;
use unpopped::plan::try_build_plan;
use unpopped_vocab::{ArchSku, ElementKind, OpCategory, OperandDesc, structure_key};

fn complex_cell(op: &OpDef) -> Result<Vec<(f64, f64)>, String> {
    let d = OperandDesc::new(1, &[2], &[1], ElementKind::Complex128, 8);
    let operands = vec![d; 3];
    let key = structure_key(OpCategory::BinaryElementwise, &operands, ArchSku::Sm89);
    let plan = try_build_plan(op, &key).map_err(|e| e.to_string())?;
    let bufs = vec![
        TypedBuffer::from_complex128(&[2], &[(1.0, 2.0), (0.0, 1.0)]),
        TypedBuffer::from_complex128(&[2], &[(3.0, 4.0), (0.0, 1.0)]),
    ];
    Ok(evaluate(&plan, &operands, &bufs, &[])
        .into_iter()
        .next()
        .expect("one output")
        .to_complex_vec())
}

/// `Add`/`Sub`/`Mul` compute end-to-end through the real plan and oracle.
///
/// This is the test that makes the complex domain *live* rather than a value
/// type nothing reaches. `i × i = -1` is the second lane, because a domain that
/// cannot produce it is not the complex numbers.
#[test]
fn complex_arithmetic_computes_through_the_plan() {
    let c = ElementKind::Complex128;
    assert_eq!(
        complex_cell(&OpDef::elementwise("a", 2, &[c], input(0) + input(1))).unwrap(),
        vec![(4.0, 6.0), (0.0, 2.0)]
    );
    assert_eq!(
        complex_cell(&OpDef::elementwise("m", 2, &[c], input(0) * input(1))).unwrap(),
        vec![(-5.0, 10.0), (-1.0, 0.0)],
        "(1+2i)(3+4i) = -5+10i, and i*i = -1"
    );
}

/// Ordered ops are refused as **undefined**, not as unimplemented — and refused
/// as a typed decline rather than a panic.
///
/// Both halves matter. Before the gate existed, `Max` and `CmpLt` at a complex
/// dtype built a plan, reached the evaluator, and **panicked**: `Val::f64()` on a
/// complex is deliberately fatal, because silently taking the real part would
/// drop the imaginary half and return a wrong answer that looks right. That
/// panic is correct inside the evaluator and wrong at the trust boundary, where
/// an unserveable request must come back as a decline.
///
/// The wording is load-bearing too. `Max`/`Min` sit with `Cmp*` rather than with
/// the unimplemented ops, because the complex numbers have no total order
/// compatible with their arithmetic — telling an author their `Max` might arrive
/// in a later version would be actively wrong.
#[test]
fn ordered_ops_are_undefined_on_complex_and_decline_rather_than_panic() {
    let c = ElementKind::Complex128;
    for (name, bop) in [
        ("Max", BinaryOp::Max),
        ("Min", BinaryOp::Min),
        ("CmpLt", BinaryOp::CmpLt),
    ] {
        let op = OpDef::elementwise("x", 2, &[c], input(0).binary(bop, input(1)));
        let err =
            complex_cell(&op).expect_err(&format!("{name} at complex must decline, not compute"));
        assert!(
            err.contains("NOT ORDERED"),
            "{name} must be refused as undefined, not as unimplemented: {err}"
        );
    }
}

/// Ops that ARE defined for complex but unimplemented decline with a different
/// reason — the distinction a future implementer needs.
#[test]
fn unimplemented_complex_ops_say_so_distinctly() {
    let c = ElementKind::Complex128;
    let div = OpDef::elementwise("d", 2, &[c], input(0) / input(1));
    let err = complex_cell(&div).expect_err("complex Div is not implemented");
    assert!(
        err.contains("DEFINED but not implemented"),
        "Div must be refused as unimplemented, not as undefined: {err}"
    );
    assert!(
        !err.contains("NOT ORDERED"),
        "Div has nothing to do with ordering: {err}"
    );
}

/// A complex cell still declines at the BACKEND — no emitter has a complex ABI.
///
/// Plan-admissibility and lowering are separate gates, and this pins that
/// admitting complex to the plan did not accidentally claim an emitter can
/// produce a kernel for it. The oracle can evaluate one; nothing can compile one.
#[test]
fn a_complex_cell_still_has_no_backend() {
    use unpopped::cpu_c::CpuC;
    use unpopped::try_generate;
    let c = ElementKind::Complex128;
    let op = OpDef::elementwise("a", 2, &[c], input(0) + input(1));
    let d = OperandDesc::new(1, &[2], &[1], c, 8);
    let key = structure_key(OpCategory::BinaryElementwise, &[d, d, d], ArchSku::Sm89);
    assert!(
        try_generate(&op, &key, &CpuC).is_err(),
        "no backend has a complex ABI yet — admitting the plan must not imply one"
    );
}
