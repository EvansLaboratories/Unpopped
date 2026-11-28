// SPDX-License-Identifier: MIT OR Apache-2.0
//! `Layout::composition_within` against `Layout::composition` (the extension rule: the last mode of the left layout is read
//! past its end at its last stride, CuTe's reading). Over a sweep of small layouts the two agree whenever the right layout
//! stays inside `size(lhs)`, and the checked one refuses by name exactly when it does not.

use unpopped_vocab::layout_algebra::{Layout, LayoutError};

fn lay(t: &str) -> Layout {
    t.parse().unwrap_or_else(|e| panic!("{t}: {e}"))
}

const LHS: &[&str] = &["8:1", "8:2", "(4,2):(1,4)", "(2,3):(1,2)", "(4,4):(1,8)", "6:3"];
const RHS: &[&str] = &["2:1", "4:1", "4:2", "2:4", "(2,2):(1,2)", "3:1", "8:1", "4:3", "2:8", "(2,2):(1,4)"];

#[test]
fn within_agrees_inside_the_extent_and_refuses_past_it() {
    let (mut agreed, mut refused, mut over_reach_accepted) = (0, 0, 0);
    for l in LHS.iter().map(|t| lay(t)) {
        for r in RHS.iter().map(|t| lay(t)) {
            let plain = l.composition(&r);
            let checked = l.composition_within(&r);
            if r.cosize() <= l.size() {
                assert_eq!(plain, checked, "{l} o {r}");
                if checked.is_ok() {
                    agreed += 1;
                }
            } else {
                assert_eq!(checked, Err(LayoutError::ExceedsExtent), "{l} o {r}");
                refused += 1;
                if let Ok(c) = &plain {
                    // the unchecked result really does read past size(lhs): the last element of rhs maps there
                    let last = r.call(r.size() - 1).unwrap();
                    assert!(last >= l.size(), "{l} o {r}");
                    assert_eq!(c.call(c.size() - 1).unwrap(), l.call(last).unwrap());
                    over_reach_accepted += 1;
                }
            }
        }
    }
    // positive controls: the sweep reaches each arm
    assert!(agreed > 0 && refused > 0 && over_reach_accepted > 0, "{agreed} {refused} {over_reach_accepted}");
}

#[test]
fn a_hand_worked_over_reach() {
    // 4:1 o 8:1 = 8:1 (reads 4:1 extended to index 7); within refuses it, and accepts 4:1 o 4:1
    assert_eq!(lay("4:1").composition(&lay("8:1")), Ok(lay("8:1")));
    assert_eq!(lay("4:1").composition_within(&lay("8:1")), Err(LayoutError::ExceedsExtent));
    assert_eq!(lay("4:1").composition_within(&lay("4:1")), Ok(lay("4:1")));
}
