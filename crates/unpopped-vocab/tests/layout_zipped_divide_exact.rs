// SPDX-License-Identifier: MIT OR Apache-2.0
//! `Layout::zipped_divide_exact` against `Layout::zipped_divide`: the primitive accepts a tile that does not divide its mode (and
//! its last tile leaves the layout); the exact sibling agrees with it when every tile divides and refuses by name otherwise.

use unpopped_vocab::layout_algebra::{IntTuple, Layout, LayoutError};

fn lay(t: &str) -> Layout {
    t.parse().unwrap_or_else(|e| panic!("{t}: {e}"))
}

#[test]
fn exact_agrees_with_the_primitive_when_every_tile_divides_and_refuses_otherwise() {
    let (mut same, mut refused) = (0, 0);
    for e in 1..=24i64 {
        for t in 1..=8i64 {
            for d in [1i64, 2, 3] {
                let a = Layout::leaf(e, d).unwrap();
                let tile = [Layout::leaf(t, 1).unwrap()];
                let exact = a.zipped_divide_exact(&tile);
                if e % t == 0 {
                    assert_eq!(exact, a.zipped_divide(&tile), "{a} by {t}");
                    same += 1;
                } else {
                    assert_eq!(exact, Err(LayoutError::NotDivisible), "{a} by {t}");
                    // the primitive's residue behaviour is unchanged (the control for the refusal)
                    assert!(a.zipped_divide(&tile).is_ok(), "{a} by {t}");
                    refused += 1;
                }
            }
        }
    }
    assert!(same > 40 && refused > 40, "{same} {refused}");
}

#[test]
fn the_primitive_still_reaches_past_the_layout_and_exact_names_why_it_refuses() {
    let a = lay("6:1");
    let z = a.zipped_divide(&[lay("4:1")]).unwrap();
    assert_eq!(
        z.crd2idx(&IntTuple::ints([3, 1])).unwrap(),
        7,
        "offset 7 of 6"
    );
    assert_eq!(
        a.zipped_divide_exact(&[lay("4:1")]),
        Err(LayoutError::NotDivisible)
    );
    assert_eq!(
        a.zipped_divide_exact(&[lay("2:1"), lay("3:1")]),
        Err(LayoutError::ProfileMismatch)
    );
    assert!(a.zipped_divide_exact(&[lay("3:1")]).is_ok());
}
