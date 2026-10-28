// SPDX-License-Identifier: MIT OR Apache-2.0
//! Property tests for `unpopped_vocab::layout_algebra`.
//!
//! Every property is a function of the operation under test, so the same checker
//! runs twice: against the real operation (must pass) and against a deliberately
//! broken one (must FAIL). A checker that cannot fail would pass the first run
//! and the second would expose it; each also reports how many generated cases it
//! actually exercised, and the tests assert a floor, so a generator that stops
//! producing valid inputs cannot turn a property vacuous.
//!
//! The oracle for every property is a brute-force reading of the definition
//! (evaluate the layout at every index), not another call of the code under
//! test. Cases come from a fixed-seed generator, so a failure reproduces.

use unpopped_vocab::layout_algebra::{ComposedLayout, IntTuple, Layout, LayoutError, Swizzle};

// ---------------------------------------------------------------- generator

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        // splitmix64
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    fn range(&mut self, lo: i64, hi: i64) -> i64 {
        lo + (self.next() % (hi - lo + 1) as u64) as i64
    }
    fn coin(&mut self) -> bool {
        self.next() & 1 == 1
    }
}

fn rand_shape(r: &mut Rng, depth: u32) -> IntTuple {
    if depth == 0 || r.range(0, 2) == 0 {
        IntTuple::Int(r.range(1, 4))
    } else {
        IntTuple::tuple((0..r.range(2, 3)).map(|_| rand_shape(r, depth - 1)))
    }
}

fn map_leaves(t: &IntTuple, f: &mut dyn FnMut() -> i64) -> IntTuple {
    match t {
        IntTuple::Int(_) => IntTuple::Int(f()),
        IntTuple::Tuple(v) => IntTuple::Tuple(v.iter().map(|c| map_leaves(c, f)).collect()),
    }
}

/// Any shape up to size 200, strides 0..=6 (zero, repeated and overlapping included).
fn rand_layout(r: &mut Rng) -> Layout {
    loop {
        let shape = rand_shape(r, 2);
        if shape.product().unwrap() > 200 {
            continue;
        }
        let stride = map_leaves(&shape, &mut || r.range(0, 6));
        return Layout::new(shape, stride).unwrap();
    }
}

/// An injective layout whose strides are successive multiples, in shuffled mode
/// order: the class `complement` accepts. Returns the layout and its span (the
/// extent its modes cover, `max(s*d)` over modes with `s > 1`).
fn rand_admissible(r: &mut Rng) -> (Layout, i64) {
    let n = r.range(1, 3);
    let mut cur = 1i64;
    let mut modes: Vec<(i64, i64)> = Vec::new();
    for _ in 0..n {
        let s = r.range(1, 4);
        let d = cur * r.range(1, 3);
        modes.push((s, d));
        if s > 1 {
            cur = d * s;
        }
    }
    for k in (1..modes.len()).rev() {
        let j = r.range(0, k as i64) as usize;
        modes.swap(k, j);
    }
    let nested = modes.len() >= 3 && r.coin();
    let leaf = |&(s, d): &(i64, i64)| (IntTuple::Int(s), IntTuple::Int(d));
    let (shape, stride) = if modes.len() == 1 {
        leaf(&modes[0])
    } else if nested {
        let head = (
            IntTuple::tuple([IntTuple::Int(modes[0].0), IntTuple::Int(modes[1].0)]),
            IntTuple::tuple([IntTuple::Int(modes[0].1), IntTuple::Int(modes[1].1)]),
        );
        let rest: Vec<_> = modes[2..].iter().map(leaf).collect();
        let mut s = vec![head.0];
        let mut d = vec![head.1];
        for (a, b) in rest {
            s.push(a);
            d.push(b);
        }
        (IntTuple::Tuple(s), IntTuple::Tuple(d))
    } else {
        let (s, d): (Vec<_>, Vec<_>) = modes.iter().map(leaf).unzip();
        (IntTuple::Tuple(s), IntTuple::Tuple(d))
    };
    (Layout::new(shape, stride).unwrap(), cur)
}

// ------------------------------------------------------------------ oracles

fn image(l: &Layout) -> Vec<i64> {
    (0..l.size()).map(|i| l.call(i).unwrap()).collect()
}

fn is_injective(v: &[i64]) -> bool {
    let mut s = v.to_vec();
    s.sort_unstable();
    s.windows(2).all(|w| w[0] != w[1])
}

/// `c` is a coarsening-compatible result for `b`: every leaf of `b` corresponds
/// to a (sub)tree of `c` with the same product.
fn compatible(b: &IntTuple, c: &IntTuple) -> bool {
    match (b, c) {
        (IntTuple::Int(s), c) => c.product().ok() == Some(*s),
        (IntTuple::Tuple(x), IntTuple::Tuple(y)) => {
            x.len() == y.len() && x.iter().zip(y).all(|(p, q)| compatible(p, q))
        }
        _ => false,
    }
}

type R = Result<usize, String>;

fn need(ok: bool, what: impl FnOnce() -> String) -> Result<(), String> {
    if ok { Ok(()) } else { Err(what()) }
}

// -------------------------------------------------------------- the checkers

fn check_coalesce(f: &dyn Fn(&Layout) -> Layout) -> R {
    let mut r = Rng(1);
    for _ in 0..600 {
        let l = rand_layout(&mut r);
        let c = f(&l);
        need(c.size() == l.size(), || {
            format!("{l}: size {} -> {}", l.size(), c.size())
        })?;
        need(image(&c) == image(&l), || {
            format!("{l} coalesced to {c}: different function")
        })?;
        need(c.depth() <= 1, || {
            format!("{l} coalesced to {c}: depth > 1")
        })?;
        let m = c.flat_modes();
        need(m.len() == 1 || m.iter().all(|&(s, _)| s > 1), || {
            format!("{l} coalesced to {c}: extent-1 mode kept")
        })?;
        need(m.windows(2).all(|w| w[1].1 != w[0].0 * w[0].1), || {
            format!("{l} coalesced to {c}: adjacent modes still mergeable")
        })?;
        need(f(&c) == c, || format!("{c}: coalesce not idempotent"))?;
    }
    Ok(600)
}

fn check_composition(f: &dyn Fn(&Layout, &Layout) -> Result<Layout, LayoutError>) -> R {
    let mut r = Rng(2);
    let (mut ok, mut declined) = (0usize, 0usize);
    for _ in 0..3000 {
        let a = rand_layout(&mut r);
        let b = rand_layout(&mut r);
        match f(&a, &b) {
            Ok(c) => {
                ok += 1;
                need(c.size() == b.size(), || format!("{a} o {b} = {c}: size"))?;
                need(compatible(b.shape(), c.shape()), || {
                    format!("{a} o {b} = {c}: profile not compatible with the right operand")
                })?;
                // Composition is defined per mode of the right operand
                // (A o (B1,B2) = (A o B1, A o B2)); it equals A(B(i)) as a
                // function only for a single mode, since B's modes may overlap.
                let leaves = b.flat_modes();
                for i in 0..b.size() {
                    let got = c.call(i).unwrap();
                    let want = if leaves.len() == 1 {
                        a.call(b.call(i).unwrap()).unwrap()
                    } else {
                        let crd = b.shape().idx2crd(i).unwrap().leaves();
                        let mut sum = 0;
                        for (&(s, d), &ck) in leaves.iter().zip(&crd) {
                            let leaf = f(&a, &Layout::leaf(s, d).unwrap())
                                .map_err(|e| format!("{a} o {s}:{d}: {e}"))?;
                            sum += leaf.call(ck).unwrap();
                        }
                        sum
                    };
                    need(got == want, || {
                        format!("{a} o {b} = {c}: at {i} got {got}, want {want}")
                    })?;
                }
            }
            Err(LayoutError::NotDivisible) => declined += 1,
            Err(e) => return Err(format!("{a} o {b}: unexpected {e}")),
        }
    }
    need(ok >= 600 && declined >= 100, || {
        format!("generator drift: {ok} compositions, {declined} declines")
    })?;
    Ok(ok)
}

fn check_complement(f: &dyn Fn(&Layout, i64) -> Result<Layout, LayoutError>) -> R {
    let mut r = Rng(3);
    let mut exercised = 0;
    for _ in 0..800 {
        let (l, span) = rand_admissible(&mut r);
        let cotarget = r.range(1, 2 * span + 3);
        let c = f(&l, cotarget).map_err(|e| format!("{l} / {cotarget}: {e}"))?;
        let modes = c.flat_modes();
        need(modes.windows(2).all(|w| w[0].1 < w[1].1), || {
            format!("complement({l}, {cotarget}) = {c}: strides not increasing")
        })?;
        let n = l.size() * c.size();
        let mut all: Vec<i64> = Vec::new();
        for i in 0..l.size() {
            for j in 0..c.size() {
                all.push(l.call(i).unwrap() + c.call(j).unwrap());
            }
        }
        all.sort_unstable();
        need(all == (0..n).collect::<Vec<_>>(), || {
            format!("complement({l}, {cotarget}) = {c}: (L, R) is not a bijection onto [0, {n})")
        })?;
        let want = if cotarget >= span {
            n >= cotarget
        } else {
            n == span
        };
        need(want, || {
            format!("complement({l}, {cotarget}) = {c}: covers {n}, span {span}")
        })?;
        exercised += 1;
    }
    Ok(exercised)
}

/// A layout of size `span * q` (random `q`), so a tiler spanning `span` divides it exactly.
fn rand_dividend(r: &mut Rng, span: i64) -> Layout {
    let q = r.range(1, 3);
    let (d1, d2) = (r.range(1, 5), r.range(1, 5));
    if r.coin() {
        Layout::new(IntTuple::ints([span, q]), IntTuple::ints([d1, d2])).unwrap()
    } else {
        Layout::leaf(span * q, d1).unwrap()
    }
}

fn check_divide(f: &dyn Fn(&Layout, &Layout) -> Result<Layout, LayoutError>) -> R {
    let mut r = Rng(4);
    let (mut exercised, mut declined) = (0usize, 0usize);
    for _ in 0..1500 {
        let (b, span) = rand_admissible(&mut r);
        let a = rand_dividend(&mut r, span);
        let d = match f(&a, &b) {
            Ok(d) => d,
            Err(LayoutError::NotDivisible) => {
                declined += 1;
                continue;
            }
            Err(e) => return Err(format!("{a} / {b}: unexpected {e}")),
        };
        need(d.rank() == 2, || {
            format!("{a} / {b} = {d}: rank {}", d.rank())
        })?;
        let modes = d.modes();
        need(modes[0].size() == b.size(), || {
            format!("{a} / {b} = {d}: tile size")
        })?;
        let rest = b.complement(a.size()).unwrap();
        for i in 0..modes[0].size() {
            for j in 0..modes[1].size() {
                let got = d
                    .crd2idx(&IntTuple::ints([i, j]))
                    .map_err(|e| format!("{a} / {b} = {d}: {e}"))?;
                let want = a.call(b.call(i).unwrap() + rest.call(j).unwrap()).unwrap();
                need(got == want, || {
                    format!("{a} / {b} = {d}: ({i},{j}) got {got} want {want}")
                })?;
            }
        }
        // span*q == size(a) here, so the division is exact: a pure re-indexing.
        let (mut x, mut y) = (image(&d), image(&a));
        x.sort_unstable();
        y.sort_unstable();
        need(x == y, || {
            format!("{a} / {b} = {d}: not a permutation of {a}'s elements")
        })?;
        exercised += 1;
    }
    need(exercised >= 600, || {
        format!("generator drift: {exercised} divisions, {declined} declines")
    })?;
    Ok(exercised)
}

fn check_product(f: &dyn Fn(&Layout, &Layout) -> Result<Layout, LayoutError>) -> R {
    let mut r = Rng(5);
    let (mut exercised, mut declined) = (0usize, 0usize);
    for _ in 0..1500 {
        let (a, _) = rand_admissible(&mut r);
        let (b, _) = rand_admissible(&mut r);
        let p = match f(&a, &b) {
            Ok(p) => p,
            Err(LayoutError::NotDivisible) => {
                declined += 1;
                continue;
            }
            Err(e) => return Err(format!("{a} x {b}: unexpected {e}")),
        };
        need(p.rank() == 2, || format!("{a} x {b} = {p}: rank"))?;
        let modes = p.modes();
        need(modes[0] == a, || {
            format!("{a} x {b} = {p}: mode 0 is not the left operand")
        })?;
        need(p.size() == a.size() * b.size(), || {
            format!("{a} x {b} = {p}: size")
        })?;
        need(is_injective(&image(&p)), || {
            format!("{a} x {b} = {p}: copies overlap although both operands are injective")
        })?;
        exercised += 1;
    }
    need(exercised >= 600, || {
        format!("generator drift: {exercised} products, {declined} declines")
    })?;
    Ok(exercised)
}

fn check_swizzle(f: &dyn Fn(Swizzle, i64) -> i64) -> R {
    let mut r = Rng(6);
    let mut exercised = 0;
    for _ in 0..200 {
        let bits = r.range(0, 3) as u32;
        let base = r.range(0, 4) as u32;
        let mag = r.range(i64::from(bits), 5) as i32;
        let shift = if r.coin() { mag } else { -mag };
        let sw = Swizzle::new(bits, base, shift).unwrap();
        let k = base + shift.unsigned_abs() + bits;
        let n = 1i64 << k;
        let mut seen = vec![false; n as usize];
        for x in 0..n {
            let y = f(sw, x);
            need((0..n).contains(&y), || {
                format!("{sw:?}: {x} -> {y} leaves [0, 2^{k})")
            })?;
            need(!seen[y as usize], || format!("{sw:?}: {x} -> {y} collides"))?;
            seen[y as usize] = true;
            need(f(sw, y) == x, || {
                format!("{sw:?}: not an involution at {x}")
            })?;
            need(y & ((1 << base) - 1) == x & ((1 << base) - 1), || {
                format!("{sw:?}: {x} -> {y} changed the low {base} bits")
            })?;
        }
        exercised += 1;
    }
    Ok(exercised)
}

// ------------------------------------------------------------------- mutants

fn coalesce_real(l: &Layout) -> Layout {
    l.coalesce()
}
/// Drops extent-1 modes but never merges.
fn coalesce_no_merge(l: &Layout) -> Layout {
    let m: Vec<_> = l.flat_modes().into_iter().filter(|&(s, _)| s > 1).collect();
    let m = if m.is_empty() { vec![(1, 0)] } else { m };
    Layout::from_modes(
        &m.iter()
            .map(|&(s, d)| Layout::leaf(s, d).unwrap())
            .collect::<Vec<_>>(),
    )
    .unwrap()
}
/// Merges when the next stride is `d0 + s0` instead of `d0 * s0`.
fn coalesce_wrong_rule(l: &Layout) -> Layout {
    let mut out: Vec<(i64, i64)> = Vec::new();
    for (s, d) in l.flat_modes() {
        if s == 1 {
            continue;
        }
        match out.last_mut() {
            Some((ps, pd)) if *pd + *ps == d => *ps *= s,
            _ => out.push((s, d)),
        }
    }
    if out.is_empty() {
        out.push((1, 0));
    }
    Layout::from_modes(
        &out.iter()
            .map(|&(s, d)| Layout::leaf(s, d).unwrap())
            .collect::<Vec<_>>(),
    )
    .unwrap()
}

#[test]
fn coalesce_holds_and_each_mutant_is_caught() {
    assert_eq!(check_coalesce(&coalesce_real), Ok(600));
    let e = check_coalesce(&coalesce_no_merge).unwrap_err();
    assert!(e.contains("mergeable"), "{e}");
    let e = check_coalesce(&coalesce_wrong_rule).unwrap_err();
    assert!(
        e.contains("different function") || e.contains("mergeable"),
        "{e}"
    );
}

#[test]
fn composition_holds_and_each_mutant_is_caught() {
    let n = check_composition(&|a, b| a.composition(b)).unwrap();
    assert!(n >= 600);
    // Wrong order: B o A.
    let e = check_composition(&|a, b| b.composition(a)).unwrap_err();
    assert!(e.contains(" o "), "{e}");
    // Ignores the left operand.
    let e = check_composition(&|_, b| Ok(b.clone())).unwrap_err();
    assert!(e.contains("got"), "{e}");
}

#[test]
fn complement_holds_and_each_mutant_is_caught() {
    assert_eq!(check_complement(&|l, m| l.complement(m)), Ok(800));
    // Strides in decreasing order.
    let e = check_complement(&|l, m| {
        let c = l.complement(m)?;
        let rev: Vec<_> = c.flat_modes().into_iter().rev().collect();
        Layout::from_modes(
            &rev.iter()
                .map(|&(s, d)| Layout::leaf(s, d).unwrap())
                .collect::<Vec<_>>(),
        )
    })
    .unwrap_err();
    assert!(
        e.contains("not increasing") || e.contains("bijection"),
        "{e}"
    );
    // Never fills the gaps.
    let e = check_complement(&|_, _| Layout::leaf(1, 0)).unwrap_err();
    assert!(e.contains("bijection"), "{e}");
    // Rounds the tail down instead of up: covers less than the cotarget.
    let e = check_complement(&|l, m| {
        let span = l
            .flat_modes()
            .iter()
            .filter(|&&(s, _)| s > 1)
            .map(|&(s, d)| s * d)
            .max()
            .unwrap_or(1);
        l.complement((m / span).max(1) * span)
    })
    .unwrap_err();
    assert!(e.contains("covers"), "{e}");
}

#[test]
fn divide_holds_and_each_mutant_is_caught() {
    let n = check_divide(&|a, b| a.logical_divide(b)).unwrap();
    assert!(n >= 600);
    // Only the tile, no rest mode.
    let e = check_divide(&|a, b| a.composition(b)).unwrap_err();
    assert!(e.contains("rank") || e.contains("tile size"), "{e}");
    // Rest mode taken from the wrong cotarget (cosize of the tiler, not size of A).
    let e = check_divide(&|a, b| {
        let rest = b.complement(b.cosize() * 2)?;
        a.composition(&Layout::from_modes(&[b.clone(), rest])?)
    })
    .unwrap_err();
    assert!(e.contains("/"), "{e}");
}

#[test]
fn product_holds_and_each_mutant_is_caught() {
    let n = check_product(&|a, b| a.logical_product(b)).unwrap();
    assert!(n >= 600);
    // No complement: copies land on top of the original.
    let e = check_product(&|a, b| Layout::from_modes(&[a.clone(), b.clone()])).unwrap_err();
    assert!(e.contains("overlap"), "{e}");
    // Complement of the wrong size: copies are placed too close.
    let e = check_product(&|a, b| {
        let rest = a.complement(a.size())?.composition(b)?;
        Layout::from_modes(&[a.clone(), rest])
    })
    .unwrap_err();
    assert!(e.contains("overlap") || e.contains("size"), "{e}");
}

#[test]
fn swizzle_holds_and_each_mutant_is_caught() {
    let apply = |s: Swizzle, x: i64| s.apply(x).unwrap();
    assert_eq!(check_swizzle(&apply), Ok(200));
    // Unmasked xor-shift: still a bijection on a power-of-two range for a right
    // shift, but not an involution and it disturbs the low bits.
    let e = check_swizzle(&|s, x| {
        if s.shift() > 0 {
            x ^ (x >> s.shift())
        } else {
            x
        }
    })
    .unwrap_err();
    assert!(
        e.contains("involution") || e.contains("low") || e.contains("leaves"),
        "{e}"
    );
    // Identity is a bijection and an involution but a swizzle with bits > 0 must move something...
    // caught instead by the explicit vector below.
}

#[test]
fn swizzle_known_vectors() {
    // Swizzle<3,0,3>: bits 3..5 are xored into bits 0..2.
    let s = Swizzle::new(3, 0, 3).unwrap();
    assert_eq!(s.apply(0b101_000).unwrap(), 0b101_101);
    assert_eq!(s.apply(0b000_111).unwrap(), 0b000_111);
    // Swizzle<2,1,-2>: bits 1..2 are xored into bits 3..4 (shift left by 2).
    let s = Swizzle::new(2, 1, -2).unwrap();
    assert_eq!(s.apply(0b0_0110).unwrap(), 0b1_1110);
    // Overlapping source and target groups are declined.
    assert_eq!(Swizzle::new(3, 0, 2), Err(LayoutError::BadSwizzle));
    assert_eq!(Swizzle::new(1, 40, 30), Err(LayoutError::BadSwizzle));
    assert_eq!(s.apply(-1), Err(LayoutError::OutOfRange));
}

#[test]
fn a_swizzled_bijective_layout_stays_bijective() {
    // 8 rows x 8 columns of 1-element, row stride 8: the classic bank-conflict case.
    let l = Layout::new(IntTuple::ints([8, 8]), IntTuple::ints([1, 8])).unwrap();
    let c = ComposedLayout::new(Swizzle::new(3, 0, 3).unwrap(), l);
    let mut v: Vec<i64> = (0..64).map(|i| c.call(i).unwrap()).collect();
    assert_ne!(
        v,
        (0..64).collect::<Vec<_>>(),
        "a swizzle that moves nothing"
    );
    v.sort_unstable();
    assert_eq!(v, (0..64).collect::<Vec<_>>());
    // Each column's 8 rows now hit 8 distinct low-3-bit residues.
    for col in 0..8 {
        let mut banks: Vec<i64> = (0..8)
            .map(|row| c.call(col + 8 * row).unwrap() % 8)
            .collect();
        banks.sort_unstable();
        assert_eq!(banks, (0..8).collect::<Vec<_>>(), "column {col}");
    }
}

#[test]
fn crd2idx_and_call_agree_everywhere() {
    let mut r = Rng(7);
    for _ in 0..400 {
        let l = rand_layout(&mut r);
        for i in 0..l.size() {
            let crd = l.shape().idx2crd(i).unwrap();
            assert_eq!(l.crd2idx(&crd), l.call(i), "{l} at {i}");
        }
    }
}
