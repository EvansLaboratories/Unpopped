// SPDX-License-Identifier: MIT OR Apache-2.0
//! Bounds of an accepted composition `A o B` (`Layout::composition`), against the left operand's own extent.
//!
//! `layout_algebra_properties` checks `C(i) == A(B(i))` on random layouts. This file asks the question a consumer indexing a
//! fixed allocation needs: **where does `C` reach, relative to `A`'s image?** Exhaustive over `A = e:d` (extent 1..=12, stride
//! 1..=3) and `B = t:s` (extent 1..=14, stride 1..=14), so a failure names its `(e, d, t, s)`.
//!
//! * `B`'s image inside `[0, size(A))`: `C` is accepted and every element of `C` is an element of `A`'s image, in bounds.
//! * `B` reaching past `size(A)`: `C` is accepted too (it is never refused), and `A` is read **extended**, its last mode
//!   continuing at its last stride, so `C` addresses offsets `A` does not hold. That is the CuTe reading of `A` as a function of
//!   its linear index; it is pinned here as behaviour, not endorsed as safe, and a consumer must bound `B` itself.
//!
//! The checker runs against the real operation (must pass) and a broken one (must FAIL), and counts both classes so a
//! generator that stops reaching either cannot make it vacuous.

use unpopped_vocab::layout_algebra::{Layout, LayoutError};

type Compose = dyn Fn(&Layout, &Layout) -> Result<Layout, LayoutError>;

#[derive(Debug, Default)]
struct Census {
    inside: usize,
    past: usize,
    past_leaves_image: usize,
}

/// One more than the largest offset `a` addresses.
fn hi(a: &Layout) -> i64 {
    (0..a.size()).map(|i| a.call(i).unwrap()).max().unwrap() + 1
}

fn need(ok: bool, what: impl FnOnce() -> String) -> Result<(), String> {
    if ok { Ok(()) } else { Err(what()) }
}

/// `a(x)` with the last mode extended past `size(a)`: for a rank-1 `e:d`, `x * d`.
fn extended(a: &Layout, x: i64) -> i64 {
    let (_, d) = a.flat_modes()[0];
    x * d
}

fn check(f: &Compose) -> Result<Census, String> {
    let mut c = Census::default();
    for e in 1..=12i64 {
        for d in 1..=3i64 {
            for t in 1..=14i64 {
                for s in 1..=14i64 {
                    let a = Layout::leaf(e, d).unwrap();
                    let b = Layout::leaf(t, s).unwrap();
                    let Ok(r) = f(&a, &b) else {
                        return Err(format!(
                            "{a} o {b}: refused; composition of two rank-1 leaves is always accepted"
                        ));
                    };
                    need(r.size() == t, || {
                        format!("{a} o {b} = {r}: size {}", r.size())
                    })?;
                    let inside = s * (t - 1) < e;
                    let mut leaves = false;
                    for i in 0..t {
                        let got = r.call(i).unwrap();
                        let want = extended(&a, b.call(i).unwrap());
                        need(got == want, || {
                            format!("{a} o {b} = {r}: C({i}) got {got} want {want}")
                        })?;
                        leaves |= got >= hi(&a);
                    }
                    if inside {
                        need(!leaves, || {
                            format!("{a} o {b} = {r}: B stays in A's extent yet C leaves A's image")
                        })?;
                        c.inside += 1;
                    } else {
                        c.past += 1;
                        c.past_leaves_image += usize::from(leaves);
                    }
                }
            }
        }
    }
    need(c.inside >= 100 && c.past >= 1000, || {
        format!("generator drift: {c:?}")
    })?;
    Ok(c)
}

fn real(a: &Layout, b: &Layout) -> Result<Layout, LayoutError> {
    a.composition(b)
}

/// Broken: ignores `A`'s stride, so every offset is off by a factor of `d`.
fn ignores_stride(a: &Layout, b: &Layout) -> Result<Layout, LayoutError> {
    Layout::leaf(a.size(), 1).unwrap().composition(b)
}

#[test]
fn a_composition_inside_the_extent_stays_in_the_image_and_reads_the_right_element() {
    let c = check(&real).unwrap();
    assert!(c.inside > 0, "{c:?}");
}

#[test]
fn a_composition_that_ignores_the_left_stride_is_caught() {
    let e = check(&ignores_stride).unwrap_err();
    assert!(e.contains("got"), "{e}");
}

/// Pinned behaviour: a right operand that reaches past the left's size is accepted and addresses past the left's image (the
/// left's last mode extended). `1:1 o 2:1` is two offsets over a one-element left operand. Flip this deliberately if the vocab
/// ever refuses such a composition.
#[test]
fn a_composition_reaching_past_the_left_extent_is_accepted_and_leaves_its_image() {
    let a = Layout::leaf(1, 1).unwrap();
    let r = a
        .composition(&Layout::leaf(2, 1).unwrap())
        .expect("accepted today");
    assert_eq!((r.call(0).unwrap(), r.call(1).unwrap()), (0, 1));
    assert!(
        r.call(1).unwrap() >= hi(&a),
        "offset 1 of a 1-element left operand"
    );
    let c = check(&real).unwrap();
    assert!(
        c.past_leaves_image > 0,
        "the grid must reach the class: {c:?}"
    );
}
