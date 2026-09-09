//! **The caller-facing shape table must name every fold-shaped `Access`.**
//!
//! `ir.rs` carries two tables mapping `Access` shapes to the §6.16-0011 question:
//!
//! - one on [`is_bit_or_sign_move`]'s doc — the function a caller reaches FIRST,
//!   whose prose sends them onward with *"this function alone is not enough"*;
//! - one on `is_bit_move_fold_output`'s doc — the complete field map, on the page
//!   you only reach **once the first table has already sent you there**.
//!
//! Measured 2026-09-06 at `d5bda080`: **the caller-facing table listed
//! `Reduction` and `RowReduce`. The field map listed those two plus `Scan` and
//! `Window`.** A `Scan` or `Window` caller reading the entry point found no row
//! for their shape and no reason to read further.
//!
//! ⚠️ **THE TABLE ITSELF WAS REPAIRED BY THE LANE AT `146d26f`, INDEPENDENTLY AND
//! FIRST — this file no longer fixes anything. It pins what that repair cannot.**
//! `every_fold_shape_has_a_correct_predicate_call`, added by the same commit,
//! tests the PREDICATE'S BEHAVIOUR across shapes. **Forced, not argued: delete
//! the `Access::Scan` row from the caller-facing table on `146d26f` and that test
//! stays GREEN — 2 passed — while this one fails naming `["Scan"]`.** The
//! document can regress and the behavioural test cannot see it, because the
//! predicate keeps behaving correctly for a shape nobody is told to route.
//!
//! ⚠️ **THAT IS ROUND FIVE OF ONE DEFECT IN THIS FILE.** The predicate was renamed
//! from `is_bit_move_reduction_output` to `is_bit_move_fold_output` because
//! baracuda flagged that *"a caller would read 'reduction' and leave scan and
//! window on the old predicate"*. The rename shipped; the caller-facing table
//! kept two rows and the word *"reduction"*.
//! `the_reduction_field_table_is_not_duplicated_in_prose` guards a DIFFERENT
//! table in the same file, and its own comment records the claim escaping into
//! `docs/normative-seams.md` — *"a guard scoped to one file cannot see that."*
//! **A guard scoped to one TABLE, or to BEHAVIOUR, cannot see this.**
//!
//! # Why this reads the enum rather than comparing the two tables to each other
//!
//! Two tables agreeing proves they were copied, not that either is right. The
//! authority is the `Access` enum: a variant is fold-shaped when it carries a
//! **fold** (`op` or `stages`) and a **post** (`post` or `epilogue`), which is
//! the triple the predicate takes. Deriving from the code means adding a
//! fold-shaped variant reddens this test on the variant that was added.
//!
//! ⚠️ **`Access::Contraction` is deliberately OUT of that derivation and it is an
//! open question, not a settled exclusion.** It carries an `epilogue` over a
//! `Reduced(0)` bridge — the same bridge `RowReduce` uses — but no `op`/`stages`
//! field, because its fold is a fixed `Σ`. Under the §6.16 truth table a `Sum`
//! fold is COMPUTED whatever the post, so a row for it would always read the same
//! way; **that is a reason for a row that says so, not a reason for silence**, and
//! it is the predicate owner's call rather than this test's. Raised, not decided.

use std::collections::BTreeSet;

const IR: &str = include_str!("../src/ir.rs");

/// Top-level `Access` variants, with the field names each declares.
fn access_variants() -> Vec<(String, BTreeSet<String>)> {
    let start = IR
        .find("pub enum Access {")
        .expect("control: `pub enum Access {` must be present, or this test reads nothing");
    let body = &IR[start..];
    let end = body
        .find("\n}")
        .expect("control: the Access enum must have a closing brace");
    let body = &body[..end];

    let mut out: Vec<(String, BTreeSet<String>)> = Vec::new();
    let mut current: Option<(String, BTreeSet<String>)> = None;
    for line in body.lines() {
        // A variant opens at exactly four spaces of indent with a capital.
        let is_variant_head = line.len() > 4
            && line.starts_with("    ")
            && !line.starts_with("     ")
            && line.as_bytes()[4].is_ascii_uppercase();
        if is_variant_head {
            if let Some(v) = current.take() {
                out.push(v);
            }
            let name: String = line[4..]
                .chars()
                .take_while(|c| c.is_alphanumeric() || *c == '_')
                .collect();
            current = Some((name, BTreeSet::new()));
            continue;
        }
        // A field sits at eight spaces: `        name: Type,`
        if let Some((_, fields)) = current.as_mut() {
            let t = line.trim_start();
            if line.starts_with("        ")
                && !line.starts_with("         ")
                && !t.starts_with("///")
                && !t.starts_with("//")
            {
                if let Some((name, _)) = t.split_once(':') {
                    if !name.is_empty() && name.chars().all(|c| c.is_ascii_lowercase() || c == '_')
                    {
                        fields.insert(name.to_string());
                    }
                }
            }
        }
    }
    if let Some(v) = current.take() {
        out.push(v);
    }
    out
}

/// Carries the triple the predicate takes: a fold and a post.
fn is_fold_shaped(fields: &BTreeSet<String>) -> bool {
    let has_fold = fields.contains("op") || fields.contains("stages");
    let has_post = fields.contains("post") || fields.contains("epilogue");
    has_fold && has_post
}

/// The `Access::X` names appearing inside the caller-facing prescription table.
///
/// Anchored on the sentence that introduces it rather than on a line number, and
/// bounded by the fence that closes it.
fn caller_facing_rows() -> Vec<String> {
    let anchor = "this function alone is not enough";
    let start = IR.find(anchor).unwrap_or_else(|| {
        panic!(
            "control: the caller-facing table's introducing sentence ({anchor:?}) is gone. \
             This test then measures nothing — repoint it rather than deleting it."
        )
    });
    let rest = &IR[start..];
    let fence = rest.find("```text").expect("the table's opening fence");
    let after = &rest[fence + "```text".len()..];
    let close = after.find("```").expect("the table's closing fence");
    after[..close].lines().map(str::to_string).collect()
}

/// The `Access::X` names appearing in those rows.
fn caller_facing_table() -> BTreeSet<String> {
    caller_facing_rows()
        .iter()
        .filter_map(|l| l.split("Access::").nth(1))
        .map(|s| {
            s.chars()
                .take_while(|c| c.is_alphanumeric() || *c == '_')
                .collect::<String>()
        })
        .filter(|s| !s.is_empty())
        .collect()
}

/// The fold-shaped variant names. Shared, because Codacy noted the extraction
/// was cloned into both tests below and two copies of one derivation is the
/// defect this whole file exists to catch, one level down.
fn fold_shaped_variants() -> BTreeSet<String> {
    access_variants()
        .into_iter()
        .filter(|(_, f)| is_fold_shaped(f))
        .map(|(n, _)| n)
        .collect()
}

/// Every `access.<field>` a caller-facing row cites, keyed by the shape it names.
///
/// ⚠️ Deliberately only `access.`-prefixed citations. `plan.body` names a
/// different object, and `RowReduce`'s row passes destructured `&stages` /
/// `&epilogue` rather than reaching through an `access`, so neither is checkable
/// here and neither is silently counted as checked.
fn cited_access_fields() -> Vec<(String, BTreeSet<String>)> {
    let mut out = Vec::new();
    for line in caller_facing_rows() {
        let Some(rest) = line.split("Access::").nth(1) else {
            continue;
        };
        let shape: String = rest
            .chars()
            .take_while(|c| c.is_alphanumeric() || *c == '_')
            .collect();
        let fields: BTreeSet<String> = rest
            .split("access.")
            .skip(1)
            .map(|s| {
                s.chars()
                    .take_while(|c| c.is_alphanumeric() || *c == '_')
                    .collect::<String>()
            })
            .filter(|s| !s.is_empty())
            .collect();
        if !shape.is_empty() && !fields.is_empty() {
            out.push((shape, fields));
        }
    }
    out
}

#[test]
fn the_scan_and_window_shapes_exist_and_carry_the_triple() {
    // Non-vacuity first: if the parser stops finding variants, every assertion
    // below passes having examined nothing.
    let variants = access_variants();
    assert!(
        variants.len() >= 6,
        "only {} Access variants parsed — the enum parser is broken, and a broken \
         parser makes this whole file vacuous",
        variants.len()
    );

    let folds = fold_shaped_variants();

    // The discriminating half: it must EXCLUDE things too, or `is_fold_shaped`
    // is a function that always returns true.
    assert!(
        folds.contains("Reduction") && folds.contains("RowReduce"),
        "the two long-known fold shapes went missing: {folds:?}"
    );
    assert!(
        folds.contains("Scan") && folds.contains("Window"),
        "Scan/Window no longer read as fold-shaped: {folds:?}"
    );
    assert!(
        !folds.contains("Elementwise") && !folds.contains("Im2Col"),
        "a non-fold shape was classified as fold-shaped: {folds:?}"
    );
}

#[test]
fn the_caller_facing_table_names_every_fold_shaped_access() {
    let folds = fold_shaped_variants();
    let table = caller_facing_table();

    assert!(
        table.len() >= 2,
        "the caller-facing table parsed as {table:?} — fewer rows than it has ever \
         had, so the extractor is wrong rather than the table"
    );

    let missing: Vec<&String> = folds.difference(&table).collect();
    assert!(
        missing.is_empty(),
        "the caller-facing shape table omits fold-shaped Access variants: {missing:?}\n\
         table = {table:?}\nfold-shaped in the enum = {folds:?}\n\n\
         A caller reaches `is_bit_or_sign_move` FIRST. A shape with no row there \
         has no reason to read on to `is_bit_move_fold_output`, which is where the \
         complete field map lives. That asymmetry is what the rename from \
         `is_bit_move_reduction_output` was meant to prevent."
    );
}

#[test]
fn every_field_a_row_cites_exists_on_the_shape_it_names() {
    // ⚠️ ADOPTED FROM A CODACY REVIEW ON THIS FILE, which observed that checking
    // for *general* fold-like fields does not check the *specific* ones the table
    // tells a caller to pass. It is a different defect from a missing row and one
    // this file has had before: `ir.rs` records that "a caller that learned one
    // shape and generalised will guard the WRONG FIELD on the other".
    let fields: std::collections::BTreeMap<String, BTreeSet<String>> =
        access_variants().into_iter().collect();

    let cited = cited_access_fields();
    // Non-vacuity: `plan.body` and RowReduce's destructured `&stages`/`&epilogue`
    // are not `access.`-prefixed and are deliberately unchecked, so a shrinking
    // population here means the extractor broke rather than the table.
    assert!(
        cited.len() >= 3,
        "only {} rows cite an `access.<field>` — the row extractor is broken, and an \
         empty population passes this check having examined nothing: {cited:?}",
        cited.len()
    );

    let mut wrong = Vec::new();
    for (shape, cites) in &cited {
        let Some(actual) = fields.get(shape) else {
            wrong.push(format!(
                "row names `Access::{shape}`, which is not a variant"
            ));
            continue;
        };
        for f in cites {
            if !actual.contains(f) {
                wrong.push(format!(
                    "row for `Access::{shape}` cites `access.{f}`, which it does not \
                     have (it has {actual:?})"
                ));
            }
        }
    }
    assert!(
        wrong.is_empty(),
        "the caller-facing table tells a caller to pass a field that does not exist:\n  \
         {}\n\nA row can name the right shape and the wrong field, and that reads \
         as a working instruction right up until it does not compile.",
        wrong.join("\n  ")
    );
}
