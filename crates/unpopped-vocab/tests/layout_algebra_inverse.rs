// SPDX-License-Identifier: MIT OR Apache-2.0
//! Property tests for `Layout::right_inverse` and `Layout::left_inverse`.
//!
//! Same method as `layout_algebra_properties.rs`: each checker runs against the
//! real operation (must pass) and a deliberately broken one (must fail), the
//! oracle is evaluating the layouts at every index, and a floor on exercised
//! cases keeps a drifting generator from making a property vacuous.

use unpopped_vocab::layout_algebra::{IntTuple, Layout, LayoutError};

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

fn flat(modes: &[(i64, i64)]) -> Layout {
    let l: Vec<_> = modes
        .iter()
        .map(|&(s, d)| Layout::leaf(s, d).unwrap())
        .collect();
    if l.len() == 1 {
        l[0].clone()
    } else {
        Layout::from_modes(&l).unwrap()
    }
}

/// A bijection onto `[0, size)`: compact strides over a random order of the modes.
fn rand_bijection(r: &mut Rng) -> Layout {
    let shapes: Vec<i64> = (0..r.range(1, 4)).map(|_| r.range(1, 4)).collect();
    let mut order: Vec<usize> = (0..shapes.len()).collect();
    for k in (1..order.len()).rev() {
        order.swap(k, r.range(0, k as i64) as usize);
    }
    let mut strides = vec![0; shapes.len()];
    let mut run = 1;
    for &k in &order {
        strides[k] = run;
        run *= shapes[k];
    }
    flat(&shapes.iter().copied().zip(strides).collect::<Vec<_>>())
}

/// Injective with gaps: each stride a multiple of the extent covered below it.
fn rand_gappy(r: &mut Rng) -> Layout {
    let mut cur = 1;
    let mut modes = Vec::new();
    for _ in 0..r.range(1, 3) {
        let (s, d) = (r.range(1, 4), cur * r.range(1, 3));
        modes.push((s, d));
        if s > 1 {
            cur = s * d;
        }
    }
    flat(&modes)
}

/// Any shape and strides 0..=4 (zero and overlapping strides included).
fn rand_any(r: &mut Rng) -> Layout {
    let modes: Vec<_> = (0..r.range(1, 4))
        .map(|_| (r.range(1, 4), r.range(0, 4)))
        .collect();
    flat(&modes)
}

type R = Result<usize, String>;

fn need(ok: bool, what: impl FnOnce() -> String) -> Result<(), String> {
    if ok { Ok(()) } else { Err(what()) }
}

fn check_right(f: &dyn Fn(&Layout) -> Layout) -> R {
    let mut r = Rng(11);
    let mut nontrivial = 0;
    for _ in 0..1500 {
        let l = if r.range(0, 1) == 0 {
            rand_any(&mut r)
        } else {
            rand_bijection(&mut r)
        };
        let inv = f(&l);
        for i in 0..inv.size() {
            let got = l.call(inv.call(i).unwrap()).unwrap();
            need(got == i, || {
                format!("{l}: right inverse {inv}: L(R({i})) = {got}")
            })?;
        }
        nontrivial += usize::from(inv.size() > 1);
    }
    need(nontrivial >= 500, || {
        format!("generator drift: {nontrivial} nontrivial inverses")
    })?;
    // A bijection onto [0, size) is inverted on both sides.
    for _ in 0..500 {
        let l = rand_bijection(&mut r);
        let inv = f(&l);
        need(inv.size() == l.size(), || {
            format!("{l}: inverse {inv} does not cover the layout")
        })?;
        for j in 0..l.size() {
            let got = inv.call(l.call(j).unwrap()).unwrap();
            need(got == j, || {
                format!("{l}: R(L({j})) = {got} with R = {inv}")
            })?;
        }
    }
    Ok(nontrivial)
}

fn check_left(f: &dyn Fn(&Layout) -> Result<Layout, LayoutError>) -> R {
    let mut r = Rng(12);
    let mut gappy = 0;
    for _ in 0..1000 {
        let l = if r.range(0, 1) == 0 {
            rand_gappy(&mut r)
        } else {
            rand_bijection(&mut r)
        };
        let inv = f(&l).map_err(|e| format!("{l}: {e}"))?;
        need(inv.size() >= l.cosize(), || {
            format!("{l}: left inverse {inv} smaller than cosize")
        })?;
        for i in 0..l.size() {
            let got = inv.call(l.call(i).unwrap()).unwrap();
            need(got == i, || {
                format!("{l}: left inverse {inv}: L'(L({i})) = {got}")
            })?;
        }
        gappy += usize::from(l.cosize() > l.size());
    }
    need(gappy >= 200, || {
        format!("generator drift: {gappy} layouts with gaps")
    })?;
    Ok(gappy)
}

#[test]
fn right_inverse_holds_and_each_mutant_is_caught() {
    assert!(check_right(&|l| l.right_inverse()).unwrap() >= 500);
    // The compact column-major layout of the same shape is not the inverse of a
    // permuted layout.
    let e = check_right(&|l| Layout::col_major(l.shape().clone()).unwrap()).unwrap_err();
    assert!(e.contains("L(R(") || e.contains("R(L("), "{e}");
    // The layout itself.
    let e = check_right(&|l| l.clone()).unwrap_err();
    assert!(e.contains("L(R(") || e.contains("R(L("), "{e}");
}

#[test]
fn left_inverse_holds_and_each_mutant_is_caught() {
    assert!(check_left(&|l| l.left_inverse()).unwrap() >= 200);
    // Skipping the complement: stops at the first gap.
    let e = check_left(&|l| Ok(l.right_inverse())).unwrap_err();
    assert!(
        e.contains("L'(L(") || e.contains("smaller than cosize"),
        "{e}"
    );
}

#[test]
fn worked_inverses() {
    let l = |s: &str, d: &str| {
        let p = |x: &str| IntTuple::ints(x.split(',').map(|n| n.parse().unwrap()));
        Layout::new(p(s), p(d)).unwrap()
    };
    // (2,3):(3,1) is row-major 2x3: its inverse sends offset o to the index with that offset.
    assert_eq!(l("2,3", "3,1").right_inverse().to_string(), "(3,2):(2,1)");
    assert_eq!(l("2,3", "1,2").right_inverse().to_string(), "6:1");
    // No stride-1 mode: nothing to invert.
    assert_eq!(l("4", "2").right_inverse().to_string(), "1:0");
    // Left inverse of a strided layout: 4:2 maps i -> 2i, so L'(o) = o/2 on its image.
    let inv = Layout::leaf(4, 2).unwrap().left_inverse().unwrap();
    for i in 0..4 {
        assert_eq!(inv.call(2 * i), Ok(i));
    }
    assert_eq!(
        Layout::leaf(2, 0).unwrap().left_inverse(),
        Err(LayoutError::ZeroStrideMode)
    );
}
