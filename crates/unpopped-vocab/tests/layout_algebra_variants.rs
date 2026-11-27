// SPDX-License-Identifier: MIT OR Apache-2.0
//! `Layout::{coalesce_profile, logical_divide_by_mode, zipped_divide, tiled_divide, flat_divide, blocked_product,
//! raked_product, zipped_product, tiled_product, flat_product}`.
//!
//! The hand-computed examples below were worked by hand from the definitions of `complement` and `composition`
//! (not read from CuTe's headers: the groupings are UNVERIFIED against `cute/layout.hpp`). The properties read
//! every coordinate of every result through the per-mode definition, so a regrouping that permutes modes or
//! pairs the wrong halves fails. Each property checker is also run against a deliberately broken grouping.

use unpopped_vocab::layout_algebra::{IntTuple, Layout, LayoutError};

fn lay(t: &str) -> Layout {
    t.parse().unwrap_or_else(|e| panic!("{t}: {e}"))
}

fn ints(v: &[i64]) -> IntTuple {
    IntTuple::ints(v.iter().copied())
}

fn tup(v: Vec<IntTuple>) -> IntTuple {
    IntTuple::tuple(v)
}

fn i(x: i64) -> IntTuple {
    IntTuple::Int(x)
}

// ---- hand-computed examples ----

#[test]
fn the_divide_variants_regroup_one_per_mode_divide() {
    // mode 0: 4:1 / 2:1 = (2,2):(1,2)   mode 1: 6:4 / 3:1 = (3,2):(4,12)
    let a = lay("(4,6):(1,4)");
    let t = [lay("2:1"), lay("3:1")];
    assert_eq!(
        a.logical_divide_by_mode(&t).unwrap(),
        lay("((2,2),(3,2)):((1,2),(4,12))")
    );
    assert_eq!(
        a.zipped_divide(&t).unwrap(),
        lay("((2,3),(2,2)):((1,4),(2,12))")
    );
    assert_eq!(a.tiled_divide(&t).unwrap(), lay("((2,3),2,2):((1,4),2,12)"));
    assert_eq!(a.flat_divide(&t).unwrap(), lay("(2,3,2,2):(1,4,2,12)"));
}

#[test]
fn the_product_variants_regroup_one_per_mode_product() {
    // mode 0: 2:1 x 2:1 = (2,2):(1,2)   mode 1: 3:2 x 2:1 = (3,2):(2,1)
    let a = lay("(2,3):(1,2)");
    let b = Layout::from_modes(&[lay("2:1"), lay("2:1")]).unwrap();
    assert_eq!(
        a.blocked_product(&b).unwrap(),
        lay("((2,2),(3,2)):((1,2),(2,1))")
    );
    assert_eq!(
        a.raked_product(&b).unwrap(),
        lay("((2,2),(2,3)):((2,1),(1,2))")
    );
    assert_eq!(
        a.zipped_product(&b).unwrap(),
        lay("((2,3),(2,2)):((1,2),(2,1))")
    );
    assert_eq!(a.tiled_product(&b).unwrap(), lay("((2,3),2,2):((1,2),2,1)"));
    assert_eq!(a.flat_product(&b).unwrap(), lay("(2,3,2,2):(1,2,2,1)"));
}

#[test]
fn coalesce_by_profile_merges_inside_a_mode_only() {
    let a = lay("((2,3),4,(2,2)):((1,2),6,(24,48))");
    // each mode alone: (2,3):(1,2) -> 6:1 ; 4:6 stays ; (2,2):(24,48) -> 4:24
    assert_eq!(
        a.coalesce_profile(&ints(&[1, 1, 1])).unwrap(),
        lay("(6,4,4):(1,6,24)")
    );
    // a leaf profile is plain coalesce of the whole layout
    assert_eq!(a.coalesce_profile(&i(1)).unwrap(), a.coalesce());
    // a nested profile recurses: only the last mode is coalesced
    let p = tup(vec![tup(vec![i(1), i(1)]), i(1), tup(vec![i(1), i(1)])]);
    assert_eq!(a.coalesce_profile(&p).unwrap(), a);
}

// ---- refusals, each beside its accepted control ----

#[test]
fn a_tiler_count_or_rank_that_does_not_match_is_refused() {
    let a = lay("(4,6):(1,4)");
    assert!(
        a.zipped_divide(&[lay("2:1"), lay("3:1")]).is_ok(),
        "control"
    );
    assert_eq!(
        a.zipped_divide(&[lay("2:1")]),
        Err(LayoutError::ProfileMismatch)
    );
    let blk = lay("(2,3):(1,2)");
    let two = Layout::from_modes(&[lay("2:1"), lay("2:1")]).unwrap();
    assert!(blk.blocked_product(&two).is_ok(), "control");
    assert_eq!(
        blk.blocked_product(&lay("2:1")),
        Err(LayoutError::ProfileMismatch)
    );
}

#[test]
fn a_profile_that_does_not_match_the_modes_is_refused() {
    let a = lay("(2,3,4):(1,2,6)");
    assert!(a.coalesce_profile(&ints(&[1, 1, 1])).is_ok(), "control");
    assert_eq!(
        a.coalesce_profile(&ints(&[1, 1])),
        Err(LayoutError::ProfileMismatch)
    );
}

// ---- brute-force reading through the per-mode definition ----

const SHAPES: &[(&str, &[&str])] = &[
    ("(4,6):(1,4)", &["2:1", "3:1"]),
    ("(8,4):(1,8)", &["4:1", "2:1"]),
    ("(6,8):(8,1)", &["3:1", "4:2"]),
    ("(12,4):(1,12)", &["3:4", "4:1"]),
    ("(4,2,3):(2,1,8)", &["2:1", "1:0", "3:1"]),
];

fn divide_offset(a: &Layout, tilers: &[Layout], crd: &[(i64, i64)]) -> i64 {
    // sum over modes of  mode_i( tiler_i(t) + complement(tiler_i, size(mode_i))(r) )
    a.modes()
        .iter()
        .zip(tilers)
        .zip(crd)
        .map(|((m, t), &(tc, rc))| {
            let comp = t.complement(m.size()).unwrap();
            m.call(t.call(tc).unwrap() + comp.call(rc).unwrap())
                .unwrap()
        })
        .sum()
}

fn all_crds(tilers: &[Layout], a: &Layout) -> Vec<Vec<(i64, i64)>> {
    let dims: Vec<(i64, i64)> = a
        .modes()
        .iter()
        .zip(tilers)
        .map(|(m, t)| (t.size(), t.complement(m.size()).unwrap().size()))
        .collect();
    let mut out: Vec<Vec<(i64, i64)>> = vec![vec![]];
    for (ts, rs) in dims {
        let mut next = Vec::new();
        for pre in &out {
            for t in 0..ts {
                for r in 0..rs {
                    let mut v = pre.clone();
                    v.push((t, r));
                    next.push(v);
                }
            }
        }
        out = next;
    }
    out
}

/// The coordinate of each regrouping for the per-mode coordinates `crd`.
fn coords(kind: &str, crd: &[(i64, i64)]) -> IntTuple {
    let ts: Vec<IntTuple> = crd.iter().map(|c| i(c.0)).collect();
    let rs: Vec<IntTuple> = crd.iter().map(|c| i(c.1)).collect();
    match kind {
        "by_mode" => tup(crd.iter().map(|c| tup(vec![i(c.0), i(c.1)])).collect()),
        "zipped" => tup(vec![tup(ts), tup(rs)]),
        "tiled" => {
            let mut v = vec![tup(ts)];
            v.extend(rs);
            tup(v)
        }
        _ => {
            let mut v = ts;
            v.extend(rs);
            tup(v)
        }
    }
}

type Check = Result<(), String>;

fn check_divides(
    a: &Layout,
    tilers: &[Layout],
    by: impl Fn(&Layout, &[Layout]) -> [Layout; 4],
) -> Check {
    let [bm, z, t, f] = by(a, tilers);
    for crd in all_crds(tilers, a) {
        let want = divide_offset(a, tilers, &crd);
        for (kind, l) in [
            ("by_mode", &bm),
            ("zipped", &z),
            ("tiled", &t),
            ("flat", &f),
        ] {
            let got = l
                .crd2idx(&coords(kind, &crd))
                .map_err(|e| format!("{e:?}"))?;
            if got != want {
                return Err(format!("{kind} {a} {crd:?}: {got} != {want}"));
            }
        }
    }
    Ok(())
}

fn real(a: &Layout, t: &[Layout]) -> [Layout; 4] {
    [
        a.logical_divide_by_mode(t).unwrap(),
        a.zipped_divide(t).unwrap(),
        a.tiled_divide(t).unwrap(),
        a.flat_divide(t).unwrap(),
    ]
}

/// Broken: `zipped` and `tiled` swapped, so each is read through the other's coordinate shape.
fn broken(a: &Layout, t: &[Layout]) -> [Layout; 4] {
    let [bm, z, tl, f] = real(a, t);
    [bm, tl, z, f]
}

#[test]
fn every_divide_variant_agrees_with_the_per_mode_definition() {
    for (a, ts) in SHAPES {
        let (a, ts): (Layout, Vec<Layout>) = (lay(a), ts.iter().map(|t| lay(t)).collect());
        assert_eq!(check_divides(&a, &ts, real), Ok(()), "{a}");
    }
}

#[test]
fn the_divide_checker_rejects_a_swapped_grouping() {
    let (a, ts) = SHAPES[0];
    let (a, ts): (Layout, Vec<Layout>) = (lay(a), ts.iter().map(|t| lay(t)).collect());
    assert!(check_divides(&a, &ts, broken).is_err(), "checker can fail");
}

const PRODUCTS: &[(&str, &str)] = &[
    ("(2,3):(1,2)", "(2,2):(1,2)"),
    ("(2,2):(1,2)", "(2,3):(1,2)"),
    ("(4,2):(1,4)", "(2,2):(1,2)"),
    ("(4,3):(1,4)", "(2,2):(1,2)"),
];

fn product_offset(a: &Layout, b: &Layout, crd: &[(i64, i64)]) -> i64 {
    // sum over modes of  A_i(x) + complement(A_i, size(A_i)*cosize(B_i))( B_i(y) )
    a.modes()
        .iter()
        .zip(b.modes())
        .zip(crd)
        .map(|((am, bm), &(x, y))| {
            let target = am.size() * bm.cosize();
            let comp = am.complement(target).unwrap();
            am.call(x).unwrap() + comp.call(bm.call(y).unwrap()).unwrap()
        })
        .sum()
}

fn product_crds(a: &Layout, b: &Layout) -> Vec<Vec<(i64, i64)>> {
    let mut out: Vec<Vec<(i64, i64)>> = vec![vec![]];
    for (am, bm) in a.modes().iter().zip(b.modes()) {
        let mut next = Vec::new();
        for pre in &out {
            for x in 0..am.size() {
                for y in 0..bm.size() {
                    let mut v = pre.clone();
                    v.push((x, y));
                    next.push(v);
                }
            }
        }
        out = next;
    }
    out
}

fn check_products(a: &Layout, b: &Layout, by: impl Fn(&Layout, &Layout) -> [Layout; 5]) -> Check {
    let [bl, rk, z, t, f] = by(a, b);
    for crd in product_crds(a, b) {
        let want = product_offset(a, b, &crd);
        let swapped = tup(crd.iter().map(|c| tup(vec![i(c.1), i(c.0)])).collect());
        let cases = [
            ("blocked", &bl, coords("by_mode", &crd)),
            ("raked", &rk, swapped),
            ("zipped", &z, coords("zipped", &crd)),
            ("tiled", &t, coords("tiled", &crd)),
            ("flat", &f, coords("flat", &crd)),
        ];
        for (kind, l, c) in cases {
            let got = l.crd2idx(&c).map_err(|e| format!("{e:?}"))?;
            if got != want {
                return Err(format!("{kind} {a} x {b} {crd:?}: {got} != {want}"));
            }
        }
    }
    Ok(())
}

fn real_products(a: &Layout, b: &Layout) -> [Layout; 5] {
    [
        a.blocked_product(b).unwrap(),
        a.raked_product(b).unwrap(),
        a.zipped_product(b).unwrap(),
        a.tiled_product(b).unwrap(),
        a.flat_product(b).unwrap(),
    ]
}

/// Broken: `blocked` and `raked` swapped (the copies and the block exchange places).
fn broken_products(a: &Layout, b: &Layout) -> [Layout; 5] {
    let [bl, rk, z, t, f] = real_products(a, b);
    [rk, bl, z, t, f]
}

#[test]
fn every_product_variant_agrees_with_the_per_mode_definition() {
    for (a, b) in PRODUCTS {
        let (a, b) = (lay(a), lay(b));
        assert_eq!(check_products(&a, &b, real_products), Ok(()), "{a} x {b}");
    }
}

#[test]
fn the_product_checker_rejects_blocked_and_raked_swapped() {
    let (a, b) = (lay(PRODUCTS[0].0), lay(PRODUCTS[0].1));
    assert!(check_products(&a, &b, broken_products).is_err());
}

#[test]
fn by_profile_coalesce_keeps_the_function_and_the_rank() {
    for src in [
        "((2,3),4,(2,2)):((1,2),6,(24,48))",
        "(2,3,4,5):(1,2,6,24)",
        "(2,(3,4)):(1,(2,6))",
    ] {
        let a = lay(src);
        let n = a.rank();
        let c = a.coalesce_profile(&ints(&vec![1; n])).unwrap();
        assert_eq!(c.rank(), n, "{src}: rank kept");
        assert_eq!(c.size(), a.size(), "{src}");
        for k in 0..a.size() {
            assert_eq!(c.call(k).unwrap(), a.call(k).unwrap(), "{src} at {k}");
        }
        assert!(
            c.modes().iter().all(|m| m.depth() <= 1),
            "{src}: modes flat"
        );
    }
}

#[test]
fn a_mode_whose_copies_do_not_divide_is_declined_not_guessed() {
    // mode 1: 2:2 x 2:3 composes 2:3 into (2,2):(1,4): stride 3 does not divide the first extent 2
    let (a, ok) = (lay("(2,2):(1,2)"), lay("(2,2):(1,2)"));
    assert!(a.blocked_product(&ok).is_ok(), "control");
    assert_eq!(
        a.blocked_product(&lay("(3,2):(1,3)")),
        Err(LayoutError::NotDivisible)
    );
}
