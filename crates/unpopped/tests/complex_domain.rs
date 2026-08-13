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
