//! `docs/normative-seams.md` may not discuss what `plan.body` means at a
//! reduction without naming BOTH access shapes.
//!
//! This was the doc half of a guard in `crates/unpopped/tests/
//! the_declaration_form_is_a_seam.rs`. It lives here because that test read
//! `../../../docs/normative-seams.md`, a path outside its crate: Cargo cannot
//! package files above the crate root, so the test could not run from the
//! published `.crate`. This crate is never published and already reads
//! workspace files.
//!
//! The invariant is not a phrase, because each wrong version spelled it
//! differently. It is: every wrong version of the claim was true of exactly one
//! shape (`Access::Reduction` or `Access::RowReduce`) and stated unqualified.

use std::path::Path;

#[test]
fn a_doc_discussing_plan_body_names_both_access_shapes() {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../docs/normative-seams.md");
    let text =
        std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()));

    // Control: the doc DOES discuss `plan.body` with `epilogue`, so the
    // assertion below is engaged. Without this, a rewording that drops either
    // word would turn the guard into a `continue` that passes by finding nothing.
    assert!(
        text.contains("plan.body") && text.contains("epilogue"),
        "control: docs/normative-seams.md no longer mentions both `plan.body` \
         and `epilogue`, so this guard would pass by not engaging. Re-point it \
         at where the claim now lives"
    );

    assert!(
        text.contains("Access::Reduction") && text.contains("Access::RowReduce"),
        "docs/normative-seams.md discusses `plan.body` and `epilogue` together \
         but does not name BOTH access shapes. Every wrong version of this claim \
         was true of one shape and stated unqualified — naming both is what \
         makes it checkable"
    );
}
