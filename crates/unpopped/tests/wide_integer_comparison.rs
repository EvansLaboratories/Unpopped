//! Integer comparison is exact above 2⁵³ — the range where the `f64` projection
//! stops being able to tell two values apart.
//!
//! # The failure this rules out
//!
//! `f64` has a 53-bit significand. `9_007_199_254_740_992` (2⁵³) and
//! `9_007_199_254_740_993` both project to the same `f64`. Compared through that
//! projection, two genuinely different `i64` results are **equal** — and equal in
//! the silent direction, agreeing rather than complaining. A comparator that
//! cannot distinguish two values of the dtype it was handed is not a comparator.
//!
//! Bit-exact comparison never had this problem, and the one consumer-facing
//! comparison in the e2e suite uses it. This covers the *tolerant* path, which
//! is the one a caller reaches for when they want "close enough" and would not
//! expect to lose exactness as a side effect.

use unpopped::oracle::{Fidelity, TypedBuffer, compare};

/// 2⁵³ and its successor: the smallest pair of integers `f64` cannot separate.
const TWO_53: i64 = 9_007_199_254_740_992;

#[test]
fn tolerant_comparison_separates_i64_values_above_2_53() {
    let a = TypedBuffer::from_i64(&[1], &[TWO_53]);
    let b = TypedBuffer::from_i64(&[1], &[TWO_53 + 1]);

    // The projection that used to back this comparison genuinely cannot tell
    // them apart — asserted, so the test states the hazard rather than alluding
    // to it. If this ever stops being true, the rest of the file is moot.
    assert_eq!(
        a.to_f64_vec(),
        b.to_f64_vec(),
        "precondition: the f64 projection must collapse these, or this test \
         is not testing what it claims"
    );
    assert_ne!(
        a.to_i128_vec(),
        b.to_i128_vec(),
        "the i128 projection must separate them"
    );

    assert!(
        compare(&a, &b, Fidelity::Tolerant { rel: 0.0, abs: 0.0 }).is_err(),
        "zero tolerance must reject two distinct i64 values"
    );
    assert!(
        compare(&a, &b, Fidelity::BitExact).is_err(),
        "bit-exact must reject them too"
    );
}

/// The tolerance is still honoured — in integer units.
///
/// Integer arithmetic has no rounding, so `abs` here is a genuine allowed
/// distance rather than an accumulated-error band. The positive control matters:
/// without it, "rejects everything" would pass the test above.
#[test]
fn integer_tolerance_is_honoured_in_integer_units() {
    let a = TypedBuffer::from_i64(&[1], &[TWO_53]);
    let b = TypedBuffer::from_i64(&[1], &[TWO_53 + 1]);

    assert!(
        compare(&a, &b, Fidelity::Tolerant { rel: 0.0, abs: 1.0 }).is_ok(),
        "a distance of 1 must fall inside abs=1"
    );
    let c = TypedBuffer::from_i64(&[1], &[TWO_53 + 2]);
    assert!(
        compare(&a, &c, Fidelity::Tolerant { rel: 0.0, abs: 1.0 }).is_err(),
        "a distance of 2 must fall outside abs=1 — otherwise the band is not \
         being applied at all"
    );
}

/// Equality still works, and across every integer width.
///
/// The exact path is new; this is the check that it did not break the ordinary
/// case it now handles for every integer dtype.
#[test]
fn equal_integer_buffers_still_compare_equal() {
    let i64s = TypedBuffer::from_i64(&[3], &[i64::MIN, 0, i64::MAX]);
    let i32s = TypedBuffer::from_i32(&[3], &[i32::MIN, 0, i32::MAX]);
    let u32s = TypedBuffer::from_u32(&[3], &[0, 1, u32::MAX]);
    let u8s = TypedBuffer::from_u8(&[3], &[0, 127, 255]);

    for (name, buf) in [("i64", &i64s), ("i32", &i32s), ("u32", &u32s), ("u8", &u8s)] {
        assert!(
            compare(buf, buf, Fidelity::Tolerant { rel: 0.0, abs: 0.0 }).is_ok(),
            "{name}: a buffer must equal itself under zero tolerance"
        );
        assert!(
            compare(buf, buf, Fidelity::BitExact).is_ok(),
            "{name}: and bit-exactly"
        );
    }
}

/// `i64::MIN`/`i64::MAX` survive the projection, which `f64` cannot promise.
///
/// The extremes are where a width bug shows first, and `i64::MAX` is exactly the
/// value an `f64` round-trip rounds *up* past — to 2⁶³, which is not even a
/// representable `i64`.
#[test]
fn the_i64_extremes_project_exactly() {
    let buf = TypedBuffer::from_i64(&[2], &[i64::MIN, i64::MAX]);
    assert_eq!(
        buf.to_i128_vec(),
        vec![i128::from(i64::MIN), i128::from(i64::MAX)]
    );

    // And the f64 projection does not — stated so the contrast is on the record.
    let via_f64 = buf.to_f64_vec();
    assert_ne!(
        via_f64[1] as i128,
        i128::from(i64::MAX),
        "f64 cannot represent i64::MAX; if it could, this whole file is unnecessary"
    );
}
