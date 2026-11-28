// SPDX-License-Identifier: MIT OR Apache-2.0
//! `DynLayout` (`docs/design-dynamic-layout.md`): a layout whose leaves may be scalar parameters, bound to a static `Layout` at
//! launch. The static `Layout` still refuses a symbolic mode, by design; a `DynLayout` parses the same text and binds it.

use unpopped_vocab::layout_algebra::{Dim, DynLayout, DynTuple, Layout, LayoutError};

const NAMES: &[(&str, u32)] = &[("m", 0), ("n", 1), ("lda", 2)];

fn dynl(t: &str) -> DynLayout {
    DynLayout::parse_with(t, NAMES).unwrap_or_else(|e| panic!("{t}: {e}"))
}

#[test]
fn the_text_a_static_layout_refuses_parses_binds_and_gives_the_static_layout() {
    // both halves of conformance's `declines_a_dynamic_mode_in_the_layout_algebra_by_name`
    assert!("(m,n):(lda,1)".parse::<Layout>().is_err());
    assert!("(5,6):(8,1)".parse::<Layout>().is_ok());
    let d = dynl("(m,n):(lda,1)");
    assert_eq!(d.params(), vec![0, 1, 2]);
    // (m, n, lda) = (5, 6, 8)
    let l = d.bind(&[(0, 5), (1, 6), (2, 8)]).unwrap();
    assert_eq!(l, "(5,6):(8,1)".parse::<Layout>().unwrap());
    // a different launch binds a different member of the family
    let l = d.bind(&[(0, 2), (1, 3), (2, 4)]).unwrap();
    assert_eq!(l, "(2,3):(4,1)".parse::<Layout>().unwrap());
}

#[test]
fn a_bind_is_refused_by_name_when_a_value_is_missing_or_a_bound_leaf_is_invalid() {
    let d = dynl("(m,n):(lda,1)");
    assert_eq!(d.bind(&[(0, 5), (1, 6)]), Err(LayoutError::UnboundParam(2)));
    assert_eq!(
        d.bind(&[(0, 0), (1, 6), (2, 8)]),
        Err(LayoutError::NonPositiveShape(0))
    );
    assert_eq!(
        d.bind(&[(0, 5), (1, 6), (2, -1)]),
        Err(LayoutError::NegativeStride(-1))
    );
}

#[test]
fn construction_checks_the_static_parts_and_the_profile() {
    let leaf = |v| DynTuple::Leaf(Dim::Static(v));
    assert_eq!(
        DynLayout::new(DynTuple::Tuple(vec![leaf(2), leaf(3)]), leaf(1)),
        Err(LayoutError::NotCongruent)
    );
    assert_eq!(
        DynLayout::new(leaf(0), leaf(1)),
        Err(LayoutError::NonPositiveShape(0))
    );
    assert_eq!(
        DynLayout::new(leaf(4), leaf(-1)),
        Err(LayoutError::NegativeStride(-1))
    );
    assert!(DynLayout::new(leaf(4), DynTuple::Leaf(Dim::Param(0))).is_ok());
}

#[test]
fn the_canonical_text_round_trips_and_names_resolve_to_it() {
    let d = dynl("((m,2),n):((1,lda),8)");
    let text = d.to_string();
    assert_eq!(text, "((?0,2),?1):((1,?2),8)");
    assert_eq!(text.parse::<DynLayout>().unwrap(), d);
    // one way to say one thing, and an unknown name is refused
    assert!("(?01):(1)".parse::<DynLayout>().is_err());
    assert!("(k):(1)".parse::<DynLayout>().is_err());
    assert!(DynLayout::parse_with("(m,n)", NAMES).is_err());
}
