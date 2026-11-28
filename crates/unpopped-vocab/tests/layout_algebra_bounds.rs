// SPDX-License-Identifier: MIT OR Apache-2.0
//! Bounds of an accepted tiling (`zipped_divide`, `local_tile`): where the result reaches, against the dividend's own extent.
//!
//! `layout_algebra_properties` checks the exact division (`size(tile) | size(A)`). This file covers the rest: a tile that does
//! NOT divide the extent. The vocab accepts it (the rest mode rounds up), so the result addresses `ceil(e / t) * t` elements of
//! an `e`-element dividend. The property is the precise one a consumer needs: **every coordinate whose linear index is below
//! `e` reads the right element in bounds, and the coordinates that leave the extent are exactly those a predicate
//! (`i + t*j < e`) excludes.** Exhaustive over a small grid, no generator, so a failure names its `(e, t, d)`.
//!
//! The checker runs against the real operation (must pass) and a broken one (must FAIL), and reports how many accepted
//! residue tilings it saw, so a refusal everywhere cannot turn it vacuous.

use unpopped_vocab::layout_algebra::{IntTuple, Layout, LayoutError};

type Divide = dyn Fn(&Layout, &[Layout]) -> Result<Layout, LayoutError>;
type Tile = dyn Fn(&Layout, &[Layout], &[i64]) -> Result<(Layout, i64), LayoutError>;

/// What the check saw.
#[derive(Debug, Default)]
struct Census {
    exact: usize,
    residue: usize,
    refused: usize,
    /// Coordinates past the extent that the accepted residue tilings address beyond the dividend's image.
    over_reach: usize,
}

/// One more than the largest offset the dividend `a` addresses.
fn hi(a: &Layout) -> i64 {
    (0..a.size()).map(|i| a.call(i).unwrap()).max().unwrap() + 1
}

fn need(ok: bool, what: impl FnOnce() -> String) -> Result<(), String> {
    if ok { Ok(()) } else { Err(what()) }
}

/// Every coordinate `(i, j)` of the tiling `z` of `a` by `t`: a real element (`i + t*j < e`) is the right one and in bounds; a
/// spare coordinate exists only for an inexact tiling. Returns how many spare coordinates address past `a`'s image.
fn check_coordinates(a: &Layout, z: &Layout, t: i64, rest: i64) -> Result<usize, String> {
    let e = a.size();
    let mut over = 0;
    for i in 0..t {
        for j in 0..rest {
            let got = z
                .crd2idx(&IntTuple::ints([i, j]))
                .map_err(|x| format!("{a} / {t}: {x}"))?;
            let lin = i + t * j;
            if lin < e {
                let want = a.call(lin).unwrap();
                need(got == want && got < hi(a), || {
                    format!("{a} / {t}: ({i},{j}) got {got} want {want}")
                })?;
            } else {
                need(e % t != 0, || {
                    format!("{a} / {t}: exact tiling has a spare coordinate ({i},{j})")
                })?;
                over += usize::from(got >= hi(a));
            }
        }
    }
    Ok(over)
}

fn check_zipped(f: &Divide) -> Result<Census, String> {
    let mut c = Census::default();
    for e in 1..=24i64 {
        for t in 1..=8i64 {
            for d in [1i64, 2, 3, 5] {
                let a = Layout::leaf(e, d).unwrap();
                let tile = Layout::leaf(t, 1).unwrap();
                let z = match f(&a, std::slice::from_ref(&tile)) {
                    Ok(z) => z,
                    Err(_) => {
                        c.refused += 1;
                        continue;
                    }
                };
                let m = z.modes();
                need(m.len() == 2 && m[0].size() == t, || {
                    format!("{a} / {t}: not (tile, rest) of tile size {t}: {z}")
                })?;
                let rest = m[1].size();
                need(rest * t >= e, || {
                    format!("{a} / {t}: {rest} tiles do not cover {e}")
                })?;
                if e % t == 0 {
                    c.exact += 1;
                } else {
                    c.residue += 1;
                }
                c.over_reach += check_coordinates(&a, &z, t, rest)?;
            }
        }
    }
    need(c.exact >= 40 && c.residue >= 40, || {
        format!("generator drift: {c:?}")
    })?;
    Ok(c)
}

fn check_local_tile(f: &Tile) -> Result<Census, String> {
    let mut c = Census::default();
    for e in 1..=24i64 {
        for t in 1..=8i64 {
            for d in [1i64, 2, 3] {
                let a = Layout::leaf(e, d).unwrap();
                let tile = Layout::leaf(t, 1).unwrap();
                let rest = (e + t - 1) / t;
                for j in 0..rest {
                    let Ok((l, off)) = f(&a, std::slice::from_ref(&tile), &[j]) else {
                        // a residue tiling is refused by name (NotDivisible): counted as a residue case
                        if e % t == 0 {
                            c.refused += 1;
                        } else {
                            c.residue += 1;
                        }
                        continue;
                    };
                    need(l.size() == t, || {
                        format!("{a} tile {t} #{j}: size {}", l.size())
                    })?;
                    for i in 0..t {
                        let at = off + l.call(i).unwrap();
                        if j * t + i < e {
                            let want = a.call(j * t + i).unwrap();
                            need(at == want && at < hi(&a), || {
                                format!("{a} tile {t} #{j}[{i}]: got {at} want {want}")
                            })?;
                        } else if at >= hi(&a) {
                            c.over_reach += 1;
                        }
                    }
                    if e % t == 0 {
                        c.exact += 1
                    } else {
                        c.residue += 1
                    }
                }
            }
        }
    }
    need(c.exact >= 40 && c.residue >= 40, || {
        format!("generator drift: {c:?}")
    })?;
    Ok(c)
}

fn real(a: &Layout, tilers: &[Layout]) -> Result<Layout, LayoutError> {
    a.zipped_divide(tilers)
}

/// Broken: tiles one element short, so the tile mode has the wrong size and the elements land on the wrong tiles.
fn one_short(a: &Layout, tilers: &[Layout]) -> Result<Layout, LayoutError> {
    let t = tilers[0].size();
    a.zipped_divide(&[Layout::leaf((t - 1).max(1), 1).unwrap()])
}

#[test]
fn an_accepted_tiling_reads_every_real_element_in_bounds_and_only_the_padding_leaves() {
    let c = check_zipped(&real).unwrap();
    assert!(c.exact + c.residue > 0, "{c:?}");
}

#[test]
fn a_tile_that_is_one_short_is_caught() {
    let e = check_zipped(&one_short).unwrap_err();
    assert!(e.contains("tile size") || e.contains("got"), "{e}");
}

/// Broken: the tile number is read one too high, so every tile starts one tile late.
fn tile_off_by_one(a: &Layout, t: &[Layout], c: &[i64]) -> Result<(Layout, i64), LayoutError> {
    a.local_tile(t, &[c[0] + 1])
}

#[test]
fn local_tile_reads_every_real_element_in_bounds_and_only_the_padding_leaves() {
    let c = check_local_tile(&|a, t, c| a.local_tile(t, c)).unwrap();
    // since the NotDivisible fix a residue tiling is refused, so nothing addresses past the extent
    assert!(c.exact > 0 && c.residue > 0 && c.over_reach == 0, "{c:?}");
}

#[test]
fn a_local_tile_one_tile_late_is_caught() {
    let e = check_local_tile(&tile_off_by_one).unwrap_err();
    assert!(e.contains("got"), "{e}");
}

/// The finding, pinned: a residue tiling IS accepted and DOES address past the dividend's image (the padding coordinates), so a
/// consumer must predicate on `i + t*j < e`. This test must flip, deliberately, the day the vocab refuses a residue tiling
/// (or clamps it); until then it is the only place the over-reach is asserted as a fact.
#[test]
fn a_residue_tiling_is_accepted_and_its_padding_leaves_the_extent() {
    let a = Layout::leaf(6, 1).unwrap();
    let z = a
        .zipped_divide(&[Layout::leaf(4, 1).unwrap()])
        .expect("accepted today");
    let m = z.modes();
    assert_eq!((m[0].size(), m[1].size()), (4, 2));
    let last = z.crd2idx(&IntTuple::ints([3, 1])).unwrap();
    assert_eq!(last, 7, "tile 1 of 4 over 6 ends at offset 7");
    assert!(last >= hi(&a), "offset {last} is past the 6-element extent");
    let c = check_zipped(&real).unwrap();
    assert!(
        c.over_reach > 0,
        "the grid must hit the residue class: {c:?}"
    );
}
