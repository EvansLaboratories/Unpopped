// SPDX-License-Identifier: MIT OR Apache-2.0
//! `Layout::{make_ordered_layout, recast_layout, max_common_vector, max_common_layout}`.
//!
//! Hand-written examples, then brute-force properties over a fixed set of layouts; each
//! checker also runs against a deliberately broken variant that it must reject.

use unpopped_vocab::layout_algebra::{IntTuple, Layout, LayoutError};

fn lay(t: &str) -> Layout {
    t.parse().unwrap_or_else(|e| panic!("{t}: {e}"))
}
fn shp(t: &str) -> IntTuple {
    lay(&format!(
        "{t}:{}",
        t.replace(|c: char| c.is_ascii_digit(), "1")
    ))
    .shape()
    .clone()
}

#[test]
fn make_ordered_layout_gives_stride_one_to_the_lowest_order() {
    let o = |s: &str, ord: &[i64]| Layout::make_ordered_layout(&shp(s), ord);
    assert_eq!(o("(2,3,4)", &[1, 0, 2]).unwrap(), lay("(2,3,4):(3,1,6)"));
    assert_eq!(o("(2,3,4)", &[0, 1, 2]).unwrap(), lay("(2,3,4):(1,2,6)"));
    assert_eq!(o("(2,3,4)", &[2, 1, 0]).unwrap(), lay("(2,3,4):(12,4,1)"));
    // only the order's ranking matters, not its values
    assert_eq!(o("(2,3,4)", &[10, -5, 7]).unwrap(), lay("(2,3,4):(12,1,3)"));
    assert_eq!(o("((2,2),3)", &[0, 1]).unwrap(), lay("((2,2),3):((1,2),4)"));
    assert_eq!(o("((2,2),3)", &[1, 0]).unwrap(), lay("((2,2),3):((3,6),1)"));
    assert_eq!(o("5", &[0]).unwrap(), lay("5:1"));
    // refusals, each beside the accepted control above
    assert_eq!(o("(2,3,4)", &[0, 1]), Err(LayoutError::ProfileMismatch));
    assert_eq!(o("(2,3,4)", &[0, 1, 1]), Err(LayoutError::ProfileMismatch));
}

#[test]
fn recast_layout_widens_and_narrows_a_stride_one_layout() {
    assert_eq!(lay("8:1").recast_layout(16, 32).unwrap(), lay("4:1"));
    assert_eq!(
        lay("(4,2):(1,4)").recast_layout(16, 32).unwrap(),
        lay("(2,2):(1,2)")
    );
    assert_eq!(
        lay("(4,2):(1,4)").recast_layout(32, 16).unwrap(),
        lay("(8,2):(1,8)")
    );
    assert_eq!(
        lay("(4,2):(1,0)").recast_layout(16, 32).unwrap(),
        lay("(2,2):(1,0)")
    );
    assert_eq!(
        lay("(4,2):(1,4)").recast_layout(16, 16).unwrap(),
        lay("(4,2):(1,4)")
    );
    // refusals; the controls are the lines above
    assert_eq!(
        lay("(3,2):(1,3)").recast_layout(16, 32),
        Err(LayoutError::NotDivisible)
    );
    assert_eq!(
        lay("(4,2):(1,3)").recast_layout(16, 32),
        Err(LayoutError::NotDivisible)
    );
    assert_eq!(
        lay("4:2").recast_layout(16, 32),
        Err(LayoutError::NotDivisible)
    );
    assert_eq!(
        lay("8:1").recast_layout(16, 24),
        Err(LayoutError::NotDivisible)
    );
    assert_eq!(
        lay("8:1").recast_layout(0, 8),
        Err(LayoutError::NonPositiveCotarget(0))
    );
}

#[test]
fn the_common_vector_of_hand_picked_pairs() {
    assert_eq!(lay("8:1").max_common_vector(&lay("8:1")).unwrap(), 8);
    assert_eq!(
        lay("(4,2):(1,4)")
            .max_common_vector(&lay("(4,2):(1,8)"))
            .unwrap(),
        4
    );
    assert_eq!(
        lay("(2,4):(1,2)")
            .max_common_vector(&lay("(2,4):(4,1)"))
            .unwrap(),
        1
    );
    assert_eq!(
        lay("(4,2):(1,4)")
            .max_common_layout(&lay("(4,2):(1,8)"))
            .unwrap(),
        lay("4:1")
    );
    assert_eq!(
        lay("(2,4):(1,2)")
            .max_common_layout(&lay("(2,4):(4,1)"))
            .unwrap(),
        lay("1:0")
    );
}

const PAIRS: &[(&str, &str)] = &[
    ("8:1", "8:1"),
    ("(4,2):(1,4)", "(4,2):(1,8)"),
    ("(2,4):(1,2)", "(2,4):(4,1)"),
    ("(4,2):(1,4)", "(4,2):(1,4)"),
    ("(2,2,2):(1,2,4)", "(2,2,2):(1,2,4)"),
    ("(2,2,2):(1,2,4)", "(2,2,2):(1,4,2)"),
    ("(2,2,2):(1,2,4)", "(2,2,2):(2,1,4)"),
    ("(4,4):(1,4)", "(4,4):(4,1)"),
    ("(4,4):(1,4)", "(4,4):(1,8)"),
    ("16:1", "(4,4):(1,4)"),
    ("(2,3):(1,2)", "(2,3):(3,1)"),
];

type Pair = dyn Fn(&Layout, &Layout) -> Result<(i64, Layout), LayoutError>;

fn check(op: &Pair) -> Result<usize, String> {
    let mut n = 0;
    for (ta, tb) in PAIRS {
        let (a, b) = (lay(ta), lay(tb));
        let inv = b.right_inverse();
        // brute force: the longest prefix where a(b^-1(i)) == i
        let mut want = 1;
        while want < inv.size() && a.call(inv.call(want).unwrap()).unwrap() == want {
            want += 1;
        }
        let (v, r) = op(&a, &b).map_err(|e| format!("{ta} / {tb}: {e:?}"))?;
        if v != want {
            return Err(format!("{ta} / {tb}: vector {v}, brute force {want}"));
        }
        if r.size() != want {
            return Err(format!("{ta} / {tb}: layout size {} != {want}", r.size()));
        }
        for i in 0..r.size() {
            let o = r.call(i).unwrap();
            if a.call(o).unwrap() != i || b.call(o).unwrap() != i {
                return Err(format!("{ta} / {tb}: R({i}) = {o} is not common"));
            }
        }
        n += 1;
    }
    Ok(n)
}

#[test]
fn the_common_layout_and_vector_agree_with_a_brute_force_reading() {
    let good: &Pair = &|a, b| Ok((a.max_common_vector(b)?, a.max_common_layout(b)?));
    assert!(check(good).unwrap() >= 11);
    // mutants: a vector one too long; a layout that is the whole inverse
    assert!(check(&|a, b| Ok((a.max_common_vector(b)? + 1, a.max_common_layout(b)?))).is_err());
    assert!(check(&|a, b| Ok((a.max_common_vector(b)?, b.right_inverse()))).is_err());
    // a mutant that ignores its second argument
    assert!(check(&|a, _b| Ok((a.max_common_vector(a)?, a.max_common_layout(a)?))).is_err());
}

#[test]
fn a_recast_there_and_back_is_the_identity() {
    for t in [
        "8:1",
        "(8,2):(1,8)",
        "(8,2,2):(1,8,0)",
        "((8,2),3):((1,8),16)",
    ] {
        let a = lay(t);
        let wide = a.recast_layout(16, 64).unwrap();
        assert_eq!(wide.recast_layout(64, 16).unwrap(), a, "{t}");
        assert_eq!(wide.size() * 4, a.size(), "{t}");
    }
    // mutant: narrowing by the wrong ratio does not return to the start
    let a = lay("(4,2):(1,4)");
    assert_ne!(
        a.recast_layout(16, 32)
            .unwrap()
            .recast_layout(32, 8)
            .unwrap(),
        a
    );
}
