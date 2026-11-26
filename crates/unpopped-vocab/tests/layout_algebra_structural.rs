// SPDX-License-Identifier: MIT OR Apache-2.0
//! `Layout::{group, append, prepend, replace, dice, filter_zeros, filter}`.
//!
//! Method as the other `layout_algebra_*` tests: hand-written examples (from the CuTe
//! layout documentation, `01_layout.md`, not from the code), then brute-force properties
//! over a fixed set of layouts. Each property checker is also run against a deliberately
//! broken operation (`broken_*`) that it must reject, so a checker that cannot fail
//! cannot pass here.

use std::collections::BTreeSet;

use unpopped_vocab::layout_algebra::{Layout, LayoutError, SliceCrd};

fn lay(t: &str) -> Layout {
    t.parse().unwrap_or_else(|e| panic!("{t}: {e}"))
}

/// Offsets of every index in `0..size`.
fn table(x: &Layout) -> Vec<i64> {
    (0..x.size()).map(|i| x.call(i).unwrap()).collect()
}

const CORPUS: &[&str] = &[
    "(2,3,4):(1,2,6)",
    "(2,3,4,5):(1,2,6,24)",
    "(2,(3,4)):(1,(2,6))",
    "((2,2),3):((1,4),2)",
    "(4,2):(2,1)",
    "(2,3):(0,2)",
    "(2,2,4):(1,0,2)",
    "(3,4):(4,1)",
];
const MODES: &[&str] = &["5:7", "(2,2):(3,1)", "3:0", "1:0", "4:1"];

type Check = Result<(), String>;

fn need(ok: bool, what: impl FnOnce() -> String) -> Check {
    if ok { Ok(()) } else { Err(what()) }
}

// ---- hand-written examples ----

#[test]
fn group_nests_a_range_of_modes() {
    assert_eq!(
        lay("(2,3,4,5):(1,2,6,24)").group(1, 3).unwrap(),
        lay("(2,(3,4),5):(1,(2,6),24)")
    );
    assert_eq!(
        lay("(2,3,4):(1,2,6)").group(0, 2).unwrap(),
        lay("((2,3),4):((1,2),6)")
    );
    // a range of one mode is already a group; the layout is unchanged
    assert_eq!(
        lay("(2,3,4):(1,2,6)").group(1, 2).unwrap(),
        lay("(2,3,4):(1,2,6)")
    );
    // the control for the refusals below
    assert!(lay("(2,3,4):(1,2,6)").group(0, 3).is_ok());
    assert_eq!(
        lay("(2,3,4):(1,2,6)").group(1, 1),
        Err(LayoutError::OutOfRange)
    );
    assert_eq!(
        lay("(2,3,4):(1,2,6)").group(2, 4),
        Err(LayoutError::OutOfRange)
    );
    assert_eq!(
        lay("(2,3,4):(1,2,6)").group(2, 1),
        Err(LayoutError::OutOfRange)
    );
}

#[test]
fn append_and_prepend_add_a_mode_at_an_end() {
    assert_eq!(
        lay("(2,3):(1,2)").append(&lay("4:6")).unwrap(),
        lay("(2,3,4):(1,2,6)")
    );
    assert_eq!(
        lay("(3,4):(1,3)").prepend(&lay("2:12")).unwrap(),
        lay("(2,3,4):(12,1,3)")
    );
    // a single mode is read as a one-mode tuple
    assert_eq!(lay("4:1").append(&lay("3:4")).unwrap(), lay("(4,3):(1,4)"));
    assert_eq!(lay("4:1").prepend(&lay("3:4")).unwrap(), lay("(3,4):(4,1)"));
    // a nested mode is added as one mode
    assert_eq!(
        lay("(2,3):(1,2)").append(&lay("(2,2):(6,12)")).unwrap(),
        lay("(2,3,(2,2)):(1,2,(6,12))")
    );
}

#[test]
fn replace_swaps_one_mode() {
    assert_eq!(
        lay("(2,3,4):(1,2,6)").replace(1, &lay("5:7")).unwrap(),
        lay("(2,5,4):(1,7,6)")
    );
    assert_eq!(
        lay("(2,3,4):(1,2,6)")
            .replace(0, &lay("(2,2):(1,2)"))
            .unwrap(),
        lay("((2,2),3,4):((1,2),2,6)")
    );
    assert_eq!(lay("4:1").replace(0, &lay("5:2")).unwrap(), lay("5:2"));
    assert!(lay("(2,3,4):(1,2,6)").replace(2, &lay("5:7")).is_ok());
    assert_eq!(
        lay("(2,3,4):(1,2,6)").replace(3, &lay("5:7")),
        Err(LayoutError::OutOfRange)
    );
}

#[test]
fn dice_keeps_what_slice_drops() {
    use SliceCrd::{All, At, Tuple};
    let a = lay("(2,3,4):(1,2,6)");
    assert_eq!(a.dice(&Tuple(vec![All, At(0), All])).unwrap(), lay("3:2"));
    assert_eq!(
        a.dice(&Tuple(vec![At(0), All, At(0)])).unwrap(),
        lay("(2,4):(1,6)")
    );
    assert_eq!(a.dice(&Tuple(vec![All, All, All])).unwrap(), lay("1:0"));
    assert_eq!(a.dice(&Tuple(vec![At(0), At(0), At(0)])).unwrap(), a);
    let n = lay("(2,(3,4)):(1,(2,6))");
    assert_eq!(
        n.dice(&Tuple(vec![At(0), Tuple(vec![All, At(0)])]))
            .unwrap(),
        lay("(2,4):(1,6)")
    );
    assert_eq!(
        a.dice(&Tuple(vec![All, At(0)])),
        Err(LayoutError::ProfileMismatch)
    );
}

#[test]
fn filter_removes_stride_zero_and_unit_modes() {
    assert_eq!(
        lay("(2,3,4):(0,2,0)").filter_zeros(),
        lay("(1,3,1):(0,2,0)")
    );
    assert_eq!(lay("(2,3,4):(0,2,0)").filter(), lay("3:2"));
    assert_eq!(lay("(2,2,4):(1,0,2)").filter(), lay("8:1"));
    assert_eq!(
        lay("(2,(3,4)):(0,(1,3))").filter_zeros(),
        lay("(1,(3,4)):(0,(1,3))")
    );
    // control: a layout with no stride-0 mode is untouched by filter_zeros
    assert_eq!(
        lay("(2,3,4):(1,2,6)").filter_zeros(),
        lay("(2,3,4):(1,2,6)")
    );
}

// ---- properties ----

fn check_group(
    op: &dyn Fn(&Layout, usize, usize) -> Result<Layout, LayoutError>,
) -> Result<usize, String> {
    let mut n = 0;
    for t in CORPUS {
        let a = lay(t);
        let r = a.modes().len();
        for b in 0..r {
            for e in b + 1..=r {
                let g = op(&a, b, e).map_err(|x| format!("{t} group({b},{e}): {x:?}"))?;
                need(table(&g) == table(&a), || {
                    format!("{t} group({b},{e}) changed the map")
                })?;
                need(g.flat_modes() == a.flat_modes(), || {
                    format!("{t} group({b},{e}) reordered leaves")
                })?;
                if e - b >= 2 && r > e - b {
                    need(g.modes().len() == r - (e - b) + 1, || {
                        format!("{t} group({b},{e}) rank {}", g.modes().len())
                    })?;
                    let inner = a.take(b, e).map_err(|x| format!("{x:?}"))?;
                    need(g.modes()[b] == inner, || {
                        format!("{t} group({b},{e}) mode {b} is not the range")
                    })?;
                }
                n += 1;
            }
        }
    }
    Ok(n)
}

fn check_append_prepend(
    app: &dyn Fn(&Layout, &Layout) -> Result<Layout, LayoutError>,
    pre: &dyn Fn(&Layout, &Layout) -> Result<Layout, LayoutError>,
) -> Result<usize, String> {
    let mut n = 0;
    for t in CORPUS {
        let a = lay(t);
        for m in MODES {
            let m = lay(m);
            let x = app(&a, &m).map_err(|e| format!("{t} append {m}: {e:?}"))?;
            need(
                x.modes().len() == a.modes().len() + 1 && x.modes().last() == Some(&m),
                || format!("{t} append {m}: shape {x}"),
            )?;
            need(x.size() == a.size() * m.size(), || {
                format!("{t} append {m}: size")
            })?;
            for j in 0..m.size() {
                for i in 0..a.size() {
                    need(
                        x.call(i + a.size() * j).unwrap()
                            == a.call(i).unwrap() + m.call(j).unwrap(),
                        || format!("{t} append {m}: ({i},{j})"),
                    )?;
                }
            }
            let y = pre(&a, &m).map_err(|e| format!("{t} prepend {m}: {e:?}"))?;
            need(
                y.modes().first() == Some(&m) && y.modes().len() == a.modes().len() + 1,
                || format!("{t} prepend {m}: shape {y}"),
            )?;
            for i in 0..a.size() {
                for j in 0..m.size() {
                    need(
                        y.call(j + m.size() * i).unwrap()
                            == m.call(j).unwrap() + a.call(i).unwrap(),
                        || format!("{t} prepend {m}: ({j},{i})"),
                    )?;
                }
            }
            n += 1;
        }
    }
    Ok(n)
}

fn check_replace(
    op: &dyn Fn(&Layout, usize, &Layout) -> Result<Layout, LayoutError>,
) -> Result<usize, String> {
    let mut n = 0;
    for t in CORPUS {
        let a = lay(t);
        let modes = a.modes();
        for k in 0..modes.len() {
            for m in MODES {
                let m = lay(m);
                let x = op(&a, k, &m).map_err(|e| format!("{t} replace({k},{m}): {e:?}"))?;
                let got = x.modes();
                need(got.len() == modes.len(), || {
                    format!("{t} replace({k},{m}): rank")
                })?;
                for (i, g) in got.iter().enumerate() {
                    let want = if i == k { &m } else { &modes[i] };
                    need(g == want, || {
                        format!("{t} replace({k},{m}): mode {i} is {g}")
                    })?;
                }
                n += 1;
            }
        }
    }
    Ok(n)
}

fn image(x: &Layout) -> BTreeSet<i64> {
    table(x).into_iter().collect()
}

fn check_filter(
    filter: &dyn Fn(&Layout) -> Layout,
    zeros: &dyn Fn(&Layout) -> Layout,
) -> Result<usize, String> {
    let mut n = 0;
    for t in CORPUS {
        let a = lay(t);
        let f = filter(&a);
        need(image(&f) == image(&a), || format!("{t}: filter image {f}"))?;
        need(
            f.flat_modes().iter().all(|&(s, d)| s > 1 && d > 0) || f == lay("1:0"),
            || format!("{t}: filter left a unit or zero mode: {f}"),
        )?;
        let z = zeros(&a);
        need(z.shape().leaves().len() == a.shape().leaves().len(), || {
            format!("{t}: filter_zeros changed the profile")
        })?;
        need(image(&z) == image(&a), || {
            format!("{t}: filter_zeros image")
        })?;
        for (&(zs, zd), &(s, d)) in z.flat_modes().iter().zip(&a.flat_modes()) {
            need(zd == d && zs == if d == 0 { 1 } else { s }, || {
                format!("{t}: filter_zeros mode ({s}:{d}) -> ({zs}:{zd})")
            })?;
        }
        n += 1;
    }
    Ok(n)
}

fn ok_group() -> impl Fn(&Layout, usize, usize) -> Result<Layout, LayoutError> {
    |a, b, e| a.group(b, e)
}

#[test]
fn group_is_a_pure_regrouping() {
    assert!(check_group(&ok_group()).unwrap() >= 25);
    // mutant: the range is nested but a mode is dropped
    let broken = |a: &Layout, b: usize, e: usize| a.group(b, e).map(|g| g.coalesce());
    assert!(
        check_group(&broken).is_err(),
        "a checker that accepts a coalesced result"
    );
}

#[test]
fn append_and_prepend_index_the_new_mode_slowest_or_fastest() {
    let app = |a: &Layout, m: &Layout| a.append(m);
    let pre = |a: &Layout, m: &Layout| a.prepend(m);
    assert!(check_append_prepend(&app, &pre).unwrap() >= 30);
    // mutants: the two swapped; append that forgets the mode
    assert!(check_append_prepend(&pre, &pre).is_err());
    assert!(check_append_prepend(&app, &app).is_err());
    let lossy = |a: &Layout, _m: &Layout| Ok(a.clone());
    assert!(check_append_prepend(&lossy, &pre).is_err());
}

#[test]
fn replace_changes_exactly_one_mode() {
    assert!(check_replace(&|a, k, m| a.replace(k, m)).unwrap() >= 30);
    // mutant: replaces its neighbour
    assert!(check_replace(&|a, k, m| a.replace((k + 1) % a.modes().len(), m)).is_err());
}

#[test]
fn filter_keeps_the_image_and_drops_dead_modes() {
    let f = |a: &Layout| a.filter();
    let z = |a: &Layout| a.filter_zeros();
    assert!(check_filter(&f, &z).unwrap() >= 8);
    // mutants: filter that is only coalesce (keeps stride-0 modes); zeros that is identity
    assert!(check_filter(&|a| a.coalesce(), &z).is_err());
    assert!(check_filter(&f, &|a| a.clone()).is_err());
}

#[test]
fn dice_and_slice_partition_the_modes() {
    use SliceCrd::{All, At, Tuple};
    let a = lay("(2,3,4,5):(1,2,6,24)");
    for mask in 0u32..16 {
        let pat: Vec<SliceCrd> = (0..4)
            .map(|i| if mask >> i & 1 == 1 { At(0) } else { All })
            .collect();
        let kept = a.slice(&Tuple(pat.clone())).map(|(l, _)| l).unwrap();
        let diced = a.dice(&Tuple(pat)).unwrap();
        // the two sides together hold every mode exactly once
        let both = kept
            .flat_modes()
            .into_iter()
            .filter(|&m| m != (1, 0))
            .count()
            + diced
                .flat_modes()
                .into_iter()
                .filter(|&m| m != (1, 0))
                .count();
        assert_eq!(both, 4, "mask {mask:04b}");
    }
}
