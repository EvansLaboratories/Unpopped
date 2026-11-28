// SPDX-License-Identifier: MIT OR Apache-2.0
//! `Layout::{inner_partition, outer_partition, local_tile, local_partition, retile}` (docs/design-tile-and-schedule-plan.md
//! T1b). Every result is read back against the ORIGINAL layout at the coordinates the definition names, never against another
//! algebra function, so a swapped half, a wrong pairing of coordinates or a wrong offset fails: the tile at `(a, b)` holds
//! the elements `(a*T0 + p, b*T1 + q)`, and thread `t` holds `(t0 + G0*r0, t1 + G1*r1)`. Source of the groupings: worked from
//! the definitions of `zipped_divide` and `slice` (UNVERIFIED against `cute/tensor_impl.hpp`).

use unpopped_vocab::layout_algebra::{IntTuple, Layout, LayoutError};

fn lay(t: &str) -> Layout {
    t.parse().unwrap_or_else(|e| panic!("{t}: {e}"))
}

/// `l` at the column-major linear index of `(c0, c1)` in an `(n0, n1)` frame.
fn at(l: &Layout, n0: i64, c0: i64, c1: i64) -> i64 {
    l.call(c0 + n0 * c1).expect("in range")
}

#[test]
fn a_hand_worked_tile() {
    // (4,6):(1,4) tiled by 2x3 blocks: tile (1,1) starts at element (2,3), offset 2 + 3*4 = 14, and is (2,3):(1,4)
    let a = lay("(4,6):(1,4)");
    let t = [lay("2:1"), lay("3:1")];
    let (piece, off) = a.local_tile(&t, &[1, 1]).unwrap();
    assert_eq!((piece, off), (lay("(2,3):(1,4)"), 14));
}

#[test]
fn every_tile_holds_the_elements_the_definition_names_and_the_tiles_cover_once() {
    // a strided, non-contiguous layout so a wrong offset or a swapped coordinate cannot pass by accident
    let a = lay("(8,6):(1,10)");
    let t = [lay("4:1"), lay("3:1")];
    let mut seen = Vec::new();
    for tb in 0..2 {
        for ta in 0..2 {
            let (piece, off) = a.local_tile(&t, &[ta, tb]).unwrap();
            assert_eq!(piece.size(), 12);
            for q in 0..3 {
                for p in 0..4 {
                    let got = at(&piece, 4, p, q) + off;
                    let want = at(&a, 8, 4 * ta + p, 3 * tb + q);
                    assert_eq!(got, want, "tile ({ta},{tb}) element ({p},{q})");
                    seen.push(want);
                }
            }
        }
    }
    seen.sort_unstable();
    let mut all: Vec<i64> = (0..48).map(|i| a.call(i).unwrap()).collect();
    all.sort_unstable();
    assert_eq!(seen, all, "the tiles cover the layout's image exactly once");
}

#[test]
fn every_thread_holds_its_strided_share() {
    let a = lay("(8,6):(1,10)");
    let g = [lay("2:1"), lay("2:1")];
    let mut seen = Vec::new();
    for t in 0..4 {
        let (piece, off) = a.local_partition(&g, t).unwrap();
        let (t0, t1) = (t % 2, t / 2);
        assert_eq!(piece.size(), 12);
        for r1 in 0..3 {
            for r0 in 0..4 {
                let got = at(&piece, 4, r0, r1) + off;
                let want = at(&a, 8, t0 + 2 * r0, t1 + 2 * r1);
                assert_eq!(got, want, "thread {t} element ({r0},{r1})");
                seen.push(want);
            }
        }
    }
    seen.sort_unstable();
    let mut all: Vec<i64> = (0..48).map(|i| a.call(i).unwrap()).collect();
    all.sort_unstable();
    assert_eq!(
        seen, all,
        "the threads partition the layout's image exactly once"
    );
    // the inner and the outer partition are different halves: a swap would make these equal
    let t = [lay("2:1"), lay("2:1")];
    assert_ne!(
        a.inner_partition(&t, &[1, 0]).unwrap(),
        a.outer_partition(&t, &[1, 0]).unwrap()
    );
}

#[test]
fn a_rank_one_tiling_and_the_refusals() {
    let a = lay("12:3");
    let (piece, off) = a.local_tile(&[lay("4:1")], &[2]).unwrap();
    assert_eq!((piece, off), (lay("(4):(3)"), 24));
    let t = [lay("4:1"), lay("3:1")];
    let b = lay("(8,6):(1,10)");
    // one coordinate per tiler
    assert_eq!(b.local_tile(&t, &[0]), Err(LayoutError::ProfileMismatch));
    // past the number of tiles, and past the thread grid
    assert_eq!(b.local_tile(&t, &[2, 0]), Err(LayoutError::OutOfRange));
    assert_eq!(
        b.local_partition(&[lay("2:1"), lay("2:1")], 4),
        Err(LayoutError::OutOfRange)
    );
}

#[test]
fn retile_re_expresses_the_same_elements_over_another_shape() {
    let a = lay("(4,6):(1,5)");
    for target in [
        IntTuple::ints([2, 12]),
        IntTuple::ints([24]),
        IntTuple::ints([2, 2, 6]),
    ] {
        let r = a.retile(&target).unwrap();
        assert_eq!(r.size(), 24);
        // the linear order is unchanged: element k of the retiled layout is element k of the original
        for k in 0..24 {
            assert_eq!(r.call(k).unwrap(), a.call(k).unwrap(), "{target:?} at {k}");
        }
        assert_eq!(r.shape().rank(), target.rank());
    }
    // a reshape whose modes do not align with the layout's (3 does not divide the first mode, 4) is not a layout: refused
    assert_eq!(
        a.retile(&IntTuple::ints([3, 2, 4])),
        Err(LayoutError::NotDivisible)
    );
    // control: a size that differs is refused, not silently truncated
    assert_eq!(
        a.retile(&IntTuple::ints([5, 5])),
        Err(LayoutError::ProfileMismatch)
    );
}

#[test]
fn a_tile_that_does_not_divide_the_shape_is_refused_not_rounded_up() {
    // 6 elements tiled by 4: rounding up would make a second tile reach past the layout (offset 49 of 36 in a (6,6) frame)
    let a = lay("(6,6):(1,6)");
    let t = [lay("4:1"), lay("2:1")];
    for coord in [[0, 0], [1, 0], [1, 1]] {
        assert_eq!(
            a.local_tile(&t, &coord),
            Err(LayoutError::NotDivisible),
            "{coord:?}"
        );
        assert_eq!(
            a.inner_partition(&t, &coord),
            Err(LayoutError::NotDivisible)
        );
        assert_eq!(
            a.outer_partition(&t, &coord),
            Err(LayoutError::NotDivisible)
        );
    }
    // a residue of one is a residue too (5 by 4)
    assert_eq!(
        lay("(5,6):(1,5)").local_tile(&t, &[0, 0]),
        Err(LayoutError::NotDivisible)
    );
    // a thread grid that does not divide the extent is refused too (4 threads over 6 elements)
    assert_eq!(
        a.local_partition(&[lay("4:1"), lay("1:1")], 0),
        Err(LayoutError::NotDivisible)
    );
    // controls: dividing tiles still work, including one exactly the size of the shape (a single tile)
    assert!(a.local_tile(&[lay("3:1"), lay("2:1")], &[1, 2]).is_ok());
    assert!(a.local_tile(&[lay("6:1"), lay("6:1")], &[0, 0]).is_ok());
    assert!(a.local_partition(&[lay("2:1"), lay("3:1")], 5).is_ok());
}
