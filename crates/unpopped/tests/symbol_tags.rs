//! **`dtype_tag` feeds generated symbol names, and its safety rests on a prose
//! precondition enforced nowhere.** This makes the precondition executable.
//!
//! # Why this file exists
//!
//! Found by sweeping for the class that produced the `0.2.0` f16 double-demote:
//! *a neutral code path whose in-tree callers cannot reach the interesting
//! dtypes.* `dtype_tag` was the one public `cfamily` function branching on
//! f16/bf16/FP8 with **no in-tree test at all**.
//!
//! It ends in a wildcard:
//!
//! ```text
//! _ => "x",
//! ```
//!
//! and **four dtypes fall into it** — `F8E8M0`, `F8E6M2`, and the reserved
//! `Fp8E4M3FNUZ`/`Fp8E5M2FNUZ` pair. All four would produce the *same* tag, so a
//! symbol built from one could collide with a symbol built from another.
//!
//! That is not live today, because the function's doc says *"Only called for
//! dtypes that pass [`scalar_ctype`]"* and `scalar_ctype` returns `None` for
//! exactly those four. **But that is a precondition stated in a comment and
//! checked by nothing** — which is the same shape as the bug this sweep came
//! from: correct for a reason the code never states.
//!
//! So the test below asserts the precondition rather than trusting it. If a
//! future dtype gains a `scalar_ctype` spelling without a `dtype_tag` arm, it
//! silently becomes `"x"` and starts colliding, and this goes red instead.

use unpopped::cfamily::{dtype_tag, scalar_ctype};
use unpopped_vocab::ElementKind;

/// **Every dtype that can reach `dtype_tag` has a real tag** — the documented
/// precondition, made checkable.
///
/// This is the assertion with teeth: it links the two tables rather than pinning
/// either alone, so the failure mode it guards (a dtype gaining a C spelling but
/// not a symbol tag) cannot slip between them.
#[test]
fn every_dtype_with_a_c_spelling_has_a_real_symbol_tag() {
    let mut reachable = 0usize;
    for dt in ElementKind::ALL {
        if scalar_ctype(dt).is_some() {
            reachable += 1;
            assert_ne!(
                dtype_tag(dt),
                "x",
                "{dt:?} passes `scalar_ctype`, so it CAN reach `dtype_tag` — but it \
                 falls into the `_ => \"x\"` wildcard. Every dtype landing there gets \
                 the same tag, so two different dtypes can produce the same generated \
                 symbol name. Add an explicit arm."
            );
        }
    }
    assert!(
        reachable > 0,
        "no dtype passed `scalar_ctype` — the assertion never ran"
    );
}

/// The wildcard's current occupants, pinned so the set is a decision rather than
/// an accident.
///
/// These four are **unreachable through the documented path** (none has a
/// `scalar_ctype` spelling). Pinned because if one later gains a C spelling, the
/// test above fires — and this one tells the next reader which dtypes were
/// deliberately left in the wildcard and why.
#[test]
fn the_wildcard_holds_exactly_the_dtypes_with_no_c_spelling() {
    let wildcarded: Vec<ElementKind> = ElementKind::ALL
        .iter()
        .copied()
        .filter(|&d| dtype_tag(d) == "x")
        .collect();

    assert_eq!(
        wildcarded,
        vec![
            // In `ElementKind::ALL` order, which interleaves the reserved fnuz
            // pair with the MX scales -- pinned as the iteration produces it.
            ElementKind::Fp8E4M3FNUZ,
            ElementKind::F8E8M0,
            ElementKind::F8E6M2,
            ElementKind::Fp8E5M2FNUZ,
        ],
        "the wildcard set moved. The MX scales (F8E8M0/F8E6M2) are sibling operands \
         rather than element compute dtypes, and the fnuz pair is reserved — none \
         lowers, so none has a C spelling. Anything else here is a dtype that lost \
         its tag."
    );

    for d in &wildcarded {
        assert!(
            scalar_ctype(*d).is_none(),
            "{d:?} is in the wildcard AND has a C spelling — those two facts together \
             are the collision this file exists to prevent"
        );
    }
}

/// `F32Strict` is the one place `dtype_tag` deliberately disagrees with
/// `unpopped_vocab::dtype_token`, and that difference is load-bearing.
///
/// The wire token is `f32` for both `F32` and `F32Strict` — the strict axis rides
/// the `<mp>` coordinate, not the dtype field. But a *symbol name* must
/// distinguish them, or two kernels with different rounding contracts collide on
/// one entry point. Pinned so nobody "fixes" the tables into agreement.
#[test]
fn the_strict_tag_diverges_from_the_wire_token_on_purpose() {
    assert_eq!(dtype_tag(ElementKind::F32), "f32");
    assert_eq!(dtype_tag(ElementKind::F32Strict), "f32s");
    assert_eq!(
        unpopped_vocab::dtype_token(ElementKind::F32),
        unpopped_vocab::dtype_token(ElementKind::F32Strict),
        "the WIRE token is deliberately the same for both — if this ever differs, \
         the reason for `f32s` has changed and this whole test wants rereading"
    );
}
