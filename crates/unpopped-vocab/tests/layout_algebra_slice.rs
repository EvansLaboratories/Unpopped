// SPDX-License-Identifier: MIT OR Apache-2.0
//! Property tests for `Layout::select`, `take`, `slice` and `tile_to_shape`.
//!
//! Same method as the other `layout_algebra_*` tests: the oracle evaluates the
//! layouts at every index, each checker also runs against a deliberately broken
//! variant that must fail, and a floor on exercised cases stops a drifting
//! generator from making a property vacuous.

use unpopped_vocab::layout_algebra::{IntTuple, Layout, LayoutError, SliceCrd};

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    fn range(&mut self, lo: i64, hi: i64) -> i64 {
        lo + (self.next() % (hi - lo + 1) as u64) as i64
    }
}

type R = Result<usize, String>;

fn need(ok: bool, what: impl FnOnce() -> String) -> Result<(), String> {
    if ok { Ok(()) } else { Err(what()) }
}

/// A rank 2..=3 layout whose modes are an integer or a nested pair, strides 0..=5.
fn rand_tuple_layout(r: &mut Rng) -> Layout {
    let mode = |r: &mut Rng| {
        if r.range(0, 2) == 0 {
            let (a, b) = (r.range(1, 3), r.range(1, 3));
            (
                IntTuple::ints([a, b]),
                IntTuple::ints([r.range(0, 5), r.range(0, 5)]),
            )
        } else {
            (IntTuple::Int(r.range(1, 4)), IntTuple::Int(r.range(0, 5)))
        }
    };
    let (s, d): (Vec<_>, Vec<_>) = (0..r.range(2, 3)).map(|_| mode(r)).unzip();
    Layout::new(IntTuple::Tuple(s), IntTuple::Tuple(d)).unwrap()
}

/// Keep or fix (at a random valid coordinate) each top-level mode.
fn rand_pattern(r: &mut Rng, modes: &[Layout]) -> Vec<SliceCrd> {
    let pick = |r: &mut Rng, m: &Layout| {
        if r.range(0, 1) == 0 {
            SliceCrd::All
        } else {
            SliceCrd::At(r.range(0, m.size() - 1))
        }
    };
    modes.iter().map(|m| pick(r, m)).collect()
}

fn check_slice(f: &dyn Fn(&Layout, &SliceCrd) -> Result<(Layout, i64), LayoutError>) -> R {
    let mut r = Rng(21);
    let (mut partial, mut total) = (0, 0);
    for _ in 0..1500 {
        let l = rand_tuple_layout(&mut r);
        let modes = l.modes();
        let pats = rand_pattern(&mut r, &modes);
        let (sub, off) = f(&l, &SliceCrd::Tuple(pats.clone()))
            .map_err(|e| format!("{l} sliced by {pats:?}: {e}"))?;
        let kept: Vec<usize> = (0..modes.len())
            .filter(|&k| pats[k] == SliceCrd::All)
            .collect();
        let want_size: i64 = kept.iter().map(|&k| modes[k].size()).product();
        need(sub.size() == want_size, || {
            format!("{l} sliced by {pats:?} = {sub}: size")
        })?;
        for j in 0..sub.size() {
            // Digits of j over the kept modes, first kept mode fastest.
            let mut rest = j;
            let mut crd = Vec::new();
            for (k, p) in pats.iter().enumerate() {
                crd.push(IntTuple::Int(match p {
                    SliceCrd::At(c) => *c,
                    _ => {
                        let sz = modes[k].size();
                        let digit = rest % sz;
                        rest /= sz;
                        digit
                    }
                }));
            }
            let want = l.crd2idx(&IntTuple::Tuple(crd)).unwrap();
            let got = sub.call(j).unwrap() + off;
            need(got == want, || {
                format!("{l} sliced by {pats:?} = {sub} + {off}: at {j} got {got}, want {want}")
            })?;
        }
        total += 1;
        partial += usize::from(kept.len() > 1 && kept.len() < modes.len());
    }
    need(partial >= 150, || {
        format!("generator drift: {partial} partial slices of {total}")
    })?;
    Ok(total)
}

#[test]
fn slice_holds_and_each_mutant_is_caught() {
    assert_eq!(check_slice(&|l, p| l.slice(p)), Ok(1500));
    // Forgets the offset of the fixed coordinates.
    let e = check_slice(&|l, p| l.slice(p).map(|(s, _)| (s, 0))).unwrap_err();
    assert!(e.contains("got"), "{e}");
    // Keeps the modes in reverse order.
    let e = check_slice(&|l, p| {
        let (s, o) = l.slice(p)?;
        let rev: Vec<_> = s.modes().into_iter().rev().collect();
        Ok((Layout::from_modes(&rev)?, o))
    })
    .unwrap_err();
    assert!(e.contains("got") || e.contains("size"), "{e}");
}

#[test]
fn slice_declines_and_integer_layouts() {
    let l = Layout::new(IntTuple::ints([3, 4]), IntTuple::ints([1, 3])).unwrap();
    let all = SliceCrd::All;
    let tup = |v: Vec<SliceCrd>| SliceCrd::Tuple(v);
    assert_eq!(
        l.slice(&tup(vec![all.clone(), SliceCrd::At(4)])),
        Err(LayoutError::OutOfRange)
    );
    assert_eq!(
        l.slice(&tup(vec![SliceCrd::At(-1), all.clone()])),
        Err(LayoutError::OutOfRange)
    );
    assert_eq!(
        l.slice(&tup(vec![all.clone()])),
        Err(LayoutError::ProfileMismatch)
    );
    // Everything fixed: 1:0 and the full offset, 2*1 + 3*3 = 11.
    let (s, o) = l
        .slice(&tup(vec![SliceCrd::At(2), SliceCrd::At(3)]))
        .unwrap();
    assert_eq!((s.to_string(), o), ("1:0".to_string(), 11));
    // One kept mode comes back bare.
    let (s, o) = l.slice(&tup(vec![SliceCrd::At(1), all])).unwrap();
    assert_eq!((s.to_string(), o), ("4:3".to_string(), 1));
    // An integer layout: All keeps it, At fixes it.
    let leaf = Layout::leaf(5, 2).unwrap();
    assert_eq!(leaf.slice(&SliceCrd::All).unwrap(), (leaf.clone(), 0));
    assert_eq!(leaf.slice(&SliceCrd::At(3)).unwrap().1, 6);
}

#[test]
fn select_and_take() {
    let mut r = Rng(22);
    for _ in 0..300 {
        let l = rand_tuple_layout(&mut r);
        let n = l.modes().len();
        assert_eq!(l.select(&(0..n).collect::<Vec<_>>()).as_ref(), Ok(&l));
        assert_eq!(l.take(0, n).as_ref(), Ok(&l));
        assert_eq!(l.take(1, n), l.select(&(1..n).collect::<Vec<_>>()));
        let swapped = l.select(&[n - 1, 0]).unwrap().modes();
        assert_eq!(
            swapped,
            vec![l.modes()[n - 1].clone(), l.modes()[0].clone()]
        );
        assert_eq!(l.select(&[n]), Err(LayoutError::OutOfRange));
        assert_eq!(l.take(n, n), Err(LayoutError::OutOfRange));
        assert_eq!(l.take(0, n + 1), Err(LayoutError::OutOfRange));
    }
    assert_eq!(
        Layout::leaf(2, 1).unwrap().select(&[]),
        Err(LayoutError::EmptyTuple)
    );
}

fn check_tile(f: &dyn Fn(&Layout, &[i64]) -> Result<Layout, LayoutError>) -> R {
    let mut r = Rng(23);
    let mut tiled = 0;
    for _ in 0..800 {
        let dims = r.range(1, 2) as usize;
        let ext: Vec<i64> = (0..dims).map(|_| r.range(1, 3)).collect();
        let block = if dims == 1 {
            Layout::col_major(IntTuple::Int(ext[0])).unwrap()
        } else {
            Layout::col_major(IntTuple::ints(ext.clone())).unwrap()
        };
        let counts: Vec<i64> = (0..dims).map(|_| r.range(1, 3)).collect();
        let target: Vec<i64> = ext.iter().zip(&counts).map(|(e, c)| e * c).collect();
        let t = f(&block, &target).map_err(|e| format!("{block} to {target:?}: {e}"))?;
        need(t.rank() == dims, || {
            format!("{block} to {target:?} = {t}: rank")
        })?;
        for (i, m) in t.modes().iter().enumerate() {
            need(m.size() == target[i], || {
                format!("{block} to {target:?} = {t}: mode {i} size")
            })?;
        }
        // A compact block tiled over a grid covers [0, size) exactly once.
        let mut img: Vec<i64> = (0..t.size()).map(|i| t.call(i).unwrap()).collect();
        img.sort_unstable();
        need(img == (0..t.size()).collect::<Vec<_>>(), || {
            format!(
                "{block} to {target:?} = {t}: not a bijection onto [0, {})",
                t.size()
            )
        })?;
        // The first tile is the block itself.
        let pair = || SliceCrd::Tuple(vec![SliceCrd::All, SliceCrd::At(0)]);
        let first = SliceCrd::Tuple((0..dims).map(|_| pair()).collect());
        let (sub, off) = t.slice(&first).map_err(|e| format!("{t}: {e}"))?;
        need(off == 0 && sub.size() == block.size(), || {
            format!("{t}: first tile {sub}+{off}")
        })?;
        for j in 0..block.size() {
            need(sub.call(j) == block.call(j), || {
                format!("{t}: first tile is not the block")
            })?;
        }
        tiled += usize::from(counts.iter().any(|&c| c > 1));
    }
    need(tiled >= 400, || {
        format!("generator drift: {tiled} multi-tile cases")
    })?;
    Ok(tiled)
}

#[test]
fn tile_to_shape_holds_and_each_mutant_is_caught() {
    assert!(check_tile(&|b, t| b.tile_to_shape(t)).unwrap() >= 400);
    // Tiles stacked on stride 0: every tile lands on the first.
    let e = check_tile(&|b, t| {
        let per_dim: Vec<Layout> = b
            .modes()
            .iter()
            .zip(t)
            .map(|(m, &ext)| {
                Layout::from_modes(&[m.clone(), Layout::leaf(ext / m.size(), 0).unwrap()])
            })
            .collect::<Result<_, _>>()?;
        Layout::from_modes(&per_dim)
    })
    .unwrap_err();
    assert!(e.contains("bijection"), "{e}");
    // Block and tile modes swapped within each dimension.
    let e = check_tile(&|b, t| {
        let good = b.tile_to_shape(t)?;
        let swapped: Vec<Layout> = good
            .modes()
            .iter()
            .map(|m| {
                let p = m.modes();
                Layout::from_modes(&[p[1].clone(), p[0].clone()])
            })
            .collect::<Result<_, _>>()?;
        Layout::from_modes(&swapped)
    })
    .unwrap_err();
    assert!(e.contains("first tile") || e.contains("bijection"), "{e}");
}

#[test]
fn tile_to_shape_declines() {
    let b = Layout::col_major(IntTuple::ints([2, 3])).unwrap();
    assert_eq!(b.tile_to_shape(&[4, 7]), Err(LayoutError::NotDivisible));
    assert_eq!(b.tile_to_shape(&[4]), Err(LayoutError::ProfileMismatch));
    assert_eq!(b.tile_to_shape(&[0, 3]), Err(LayoutError::NotDivisible));
    // (2,3) col-major tiled to (4,3): the second row-tile sits after the 6-element block.
    assert_eq!(
        b.tile_to_shape(&[4, 3]).unwrap().to_string(),
        "((2,2),(3,1)):((1,6),(2,0))"
    );
}
