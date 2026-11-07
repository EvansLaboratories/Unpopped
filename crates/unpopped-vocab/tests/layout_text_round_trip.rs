// SPDX-License-Identifier: MIT OR Apache-2.0
//! The canonical text of `Layout`, `Swizzle` and `ComposedLayout` (`swz(B,M,S)(shape:stride)`), for KISS
//! Contract 6.5 `output_layouts` (KISS board 378): `parse(print(x)) == x`, the two entries of KISS's
//! `output_layout_vectors` (`nested`, `nested_swizzled`) parse and print back byte for byte, and every
//! non-canonical or invalid form is declined with the reason a reader needs.

use unpopped_vocab::layout_algebra::{ComposedLayout, IntTuple, Layout, LayoutError, Swizzle};

/// KISS `conformance/corpus/contract_vectors.json`, `output_layout_vectors`, element of `nested`.
const KISS_NESTED: &str = "(2,2):(1,2)";
/// The same file, element of `nested_swizzled`.
const KISS_NESTED_SWIZZLED: &str = "swz(2,3,-3)((8,8):(8,1))";

fn layout(t: &str) -> Layout {
    t.parse().unwrap_or_else(|e| panic!("{t}: {e}"))
}

#[test]
fn the_two_kiss_vectors_round_trip_byte_for_byte() {
    assert_eq!(layout(KISS_NESTED).to_string(), KISS_NESTED);
    let c: ComposedLayout = KISS_NESTED_SWIZZLED.parse().unwrap();
    assert_eq!(c.to_string(), KISS_NESTED_SWIZZLED);
    // and they are the values they say
    assert_eq!(c.swizzle(), Swizzle::new(2, 3, -3).unwrap());
    assert_eq!(c.layout(), &layout("(8,8):(8,1)"));
    assert_eq!(
        layout(KISS_NESTED),
        Layout::new(IntTuple::ints([2, 2]), IntTuple::ints([1, 2])).unwrap()
    );
}

#[test]
fn layouts_of_every_shape_category_round_trip() {
    for t in [
        "1:0",                      // scalar
        "8:1",                      // flat
        "(2,2):(1,2)",              // rank 2
        "((2,3),4):((1,2),6)",      // nested
        "(4,3):(0,1)",              // broadcast, stride 0
        "(2,(3,4)):(1,(2,6))",      // second mode nested
        "(1):(0)",                  // one-element tuple is not an integer
        "((2)):((1))",              // and nests
        "(9223372036854775807):(0)", // the largest leaf
    ] {
        let l = layout(t);
        assert_eq!(l.to_string(), t, "print(parse(t)) == t");
        assert_eq!(layout(&l.to_string()), l, "parse(print(x)) == x");
    }
    // a one-element tuple and a bare integer are different values with different text
    assert_ne!(layout("(1):(0)"), layout("1:0"));
}

#[test]
fn swizzles_round_trip_including_negative_shift() {
    for (b, m, s) in [(2, 3, -3), (3, 4, 3), (0, 0, 0), (1, 0, 1), (3, 0, -4)] {
        let z = Swizzle::new(b, m, s).unwrap();
        let t = format!("swz({b},{m},{s})");
        assert_eq!(z.to_string(), t);
        assert_eq!(t.parse::<Swizzle>().unwrap(), z);
        let c = ComposedLayout::new(z, layout("(8,8):(8,1)"));
        assert_eq!(c.to_string(), format!("{t}((8,8):(8,1))"));
        let back: ComposedLayout = c.to_string().parse().unwrap();
        assert_eq!(back, c);
        // the parsed value computes the same offsets, not merely the same fields
        for i in 0..64 {
            assert_eq!(back.call(i), c.call(i));
        }
    }
}

#[test]
fn every_non_canonical_text_is_declined() {
    for (t, why) in [
        ("(2, 2):(1,2)", "a space"),
        (" (2,2):(1,2)", "a leading space"),
        ("(2,2):(1,2) ", "a trailing space"),
        ("(+2,2):(1,2)", "a leading plus"),
        ("(02,2):(1,2)", "a leading zero"),
        ("(2,2):(1,-0)", "minus zero"),
        ("(2,2):(1,2)x", "trailing text"),
        ("(2,2)(1,2)", "no colon"),
        ("(2,2):", "no stride"),
        ("2,2:1,2", "an unparenthesised tuple"),
        ("(2,2:(1,2)", "an unclosed tuple"),
        ("(2,):(1,)", "a trailing comma"),
        ("99999999999999999999:1", "an integer past i64"),
    ] {
        assert!(t.parse::<Layout>().is_err(), "{why}: {t}");
    }
    assert!("swz(2,3,-3) ((8,8):(8,1))".parse::<ComposedLayout>().is_err());
    assert!("swz(2,3,-3)(8,8):(8,1)".parse::<ComposedLayout>().is_err());
    assert!("swz(2,3,-3)((8,8):(8,1)".parse::<ComposedLayout>().is_err());
    assert!("swz(2,3,+3)((8,8):(8,1))".parse::<ComposedLayout>().is_err());
    assert!("swz(2,3,-0)((8,8):(8,1))".parse::<ComposedLayout>().is_err());
    assert!("swz(2,03,3)((8,8):(8,1))".parse::<ComposedLayout>().is_err());
    // `swz(0,0,0)` is a valid identity swizzle, so only the `-0` rule can decline its negative-zero spelling
    assert!("swz(0,0,0)".parse::<Swizzle>().is_ok());
    assert!("swz(0,0,-0)".parse::<Swizzle>().is_err());
    assert!("swz(-2,3,3)((8,8):(8,1))".parse::<ComposedLayout>().is_err());
    assert!("swz(2,3)((8,8):(8,1))".parse::<ComposedLayout>().is_err());
    assert!("(8,8):(8,1)".parse::<ComposedLayout>().is_err(), "a plain layout is not a swizzled one");
    assert!("swz(2,3,-3)".parse::<Layout>().is_err());
}

#[test]
fn invalid_layouts_are_declined_with_the_layout_algebras_own_reason() {
    assert_eq!("(2,2):(1)".parse::<Layout>(), Err(LayoutError::NotCongruent));
    assert_eq!("2:(1,2)".parse::<Layout>(), Err(LayoutError::NotCongruent));
    assert_eq!("(2,2):(1,-2)".parse::<Layout>(), Err(LayoutError::Parse("expected an integer")));
    assert_eq!("(0,2):(1,2)".parse::<Layout>(), Err(LayoutError::NonPositiveShape(0)));
    assert_eq!("():()".parse::<Layout>(), Err(LayoutError::EmptyTuple));
    let deep = |n: usize| format!("{0}1{1}:{0}0{1}", "(".repeat(n), ")".repeat(n));
    assert!(deep(16).parse::<Layout>().is_ok(), "depth 16 is the limit");
    assert_eq!(deep(17).parse::<Layout>(), Err(LayoutError::TooDeep));
    // a hostile depth is declined before it can overflow the stack
    assert_eq!(deep(100_000).parse::<Layout>(), Err(LayoutError::TooDeep));
}

#[test]
fn an_invalid_swizzle_is_declined_as_a_bad_swizzle() {
    // |S| < B: the two bit groups overlap, so it is not an involution
    assert_eq!("swz(3,0,2)".parse::<Swizzle>(), Err(LayoutError::BadSwizzle));
    // too wide for i64
    assert_eq!("swz(10,40,20)".parse::<Swizzle>(), Err(LayoutError::BadSwizzle));
    // a parameter past u32 / i32
    assert_eq!("swz(4294967296,0,5)".parse::<Swizzle>(), Err(LayoutError::BadSwizzle));
    assert_eq!("swz(1,0,3000000000)".parse::<Swizzle>(), Err(LayoutError::BadSwizzle));
}
