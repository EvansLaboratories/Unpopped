//! Optimizing must not change the answer — checked on the values where a
//! rewrite *can* change it, which is the only place it ever does.
//!
//! # Why a random-input differential would pass while the defect was total
//!
//! Fuel's architect named this while we were working out which side owns which
//! check, and it is the sharpest thing anyone has said about testing a rewriter:
//!
//! > `fmaxf`-vs-`Max` diverges only on NaN. `x + 0.0` diverges only on `-0.0`.
//! > **The divergence set for these substitutions is a measure-zero subset that
//! > random probes never sample** — so a differential on ordinary inputs passes
//! > while the defect is total. **Enumerate the inputs on which the two forms
//! > can differ, and test exactly those.**
//!
//! An unsound rewrite is not usually wrong *somewhere in the number line*. It is
//! wrong on a handful of values with names — the zeros, the infinities, the
//! NaNs, the denormals — and correct everywhere else. Sampling `[-10, 10]`
//! uniformly finds none of them, ever, and reports green with total confidence.
//!
//! So this file's corpus **is** the divergence set: every input is a value some
//! plausible rewrite is wrong on.
//!
//! # Why it matters beyond this crate
//!
//! CireSnave is considering running every existing kernel through this generator
//! and handing the results back as additional candidates. That is safe on
//! *speed* — a slower kernel loses its benchmark — but not on *correctness*: a
//! consumer's benchmark rewards a kernel that is faster, including one that is
//! faster because it computes something slightly different.
//!
//! Fuel's admission gate cannot catch it either, and the reason is structural:
//! their reference is built from the candidate's own declared decomposition, so
//! the gate answers *"does this kernel compute what its recipe claims?"* and
//! never *"is that recipe faithful?"* The decomposition is the thing they
//! **trust**, not the thing they check. Their gate is also `#[cfg(feature =
//! "cuda")]`-only, so a Vulkane consumer gets no admission gate at all.
//!
//! **That leaves the semantics of a rewrite squarely here**, and it has to hold
//! without assuming any particular consumer's harness.
//!
//! # What this checks, exactly
//!
//! For each body: evaluate the body, evaluate `optimize(body)`, and require the
//! two to agree **bit for bit** — not within a tolerance. A tolerance would
//! absorb exactly the differences this exists to catch: `+0.0` vs `-0.0` differ
//! by nothing measurable and are different values, and a NaN's payload is not a
//! magnitude at all.
//!
//! Both sides go through the same oracle, so this is not a check that the oracle
//! is right. It is a check that the **rewrite is answer-preserving**, which is a
//! narrower claim and the one the optimizer actually makes.

use unpopped::ir::{BinaryOp, OpDef, ScalarExpr, UnaryOp, input, konst};
use unpopped::optimize::optimize;
use unpopped::oracle::{TypedBuffer, evaluate};
use unpopped::plan::build_plan;
use unpopped_vocab::{ArchSku, ElementKind, OpCategory, OperandDesc, structure_key};

/// Every value some plausible rewrite is wrong on, and nothing else.
///
/// Ordinary magnitudes (`1.0`, `-2.0`) are present as a control rather than as
/// coverage: if a body disagrees on those too, the fault is not subtle and the
/// message should say so.
///
/// # This list should not stay private, and that is a recorded debt
///
/// KISS ships an op-vector corpus (`kiss-op-manifest-v1`, generated from
/// `spec/ops.md`, vendored by Fuel at `fuel-dispatch/fixtures/kiss-corpus/`)
/// whose `tags` field already carries `signed-zero`. **When it can supply this,
/// `SPECIALS` should READ it rather than restate it.**
///
/// **It cannot today, and the reason is worse than thin coverage.** KISS
/// `origin/main` ships `ops-minmax-signed-zero.json` — 48 vectors over
/// `fmax_ieee` / `fmin_ieee` / `max_prop` / `min_prop`, exactly the four ops
/// this file exists for. It has **zero NaN vectors**, and KISS's architect
/// measured the consequence:
///
/// > **All 48 vectors pass for an implementation with `max_prop` and
/// > `fmax_ieee` SWAPPED.**
///
/// `ops.md` calls them four distinct ops *because* two propagate NaN and two
/// suppress it. So the one artefact enumerating all four **cannot distinguish
/// the property that is the stated reason they are four** — a green carrying no
/// information on its own axis. Filed as KISS #329.
///
/// **Consuming its inputs is right when they land. Treating its present
/// existence as coverage is the trap**, and it is the trap this file would have
/// walked into had the debt been written as "wait for KISS".
///
/// **Re-measured 2026-08-26 rather than updated from a relay.** A report reached
/// this repo that 96 NaN vectors had "merged tonight as kiss-ref `c6f89f84`",
/// which would have made this note stale. Measured at the refs instead:
///
/// - KISS `origin/main` `becf90f`: `conformance/corpus/ops-minmax-signed-zero.json`
///   still declares `number_of_vectors: 48`, every one tagged `signed-zero` /
///   `tie`, **no NaN inputs**.
/// - kiss-ref `c6f89f8`: landed a **generator and guard** for
///   `ops-minmax-nan.json`. The file itself does not exist in either tree.
///
/// So the note was correct and stayed. **A generator for an artefact is not the
/// artefact** — the same shape as a vendored copy read as the source, and it
/// would have cost a "correction" of a claim that was already true.
///
/// (An earlier revision of this note said "one op of 106" — Fuel's figure for
/// their VENDORED copy, which is 48 vectors and a whole file behind KISS. I
/// repeated it without its qualifier. The zero-NaN half was right about both.)
///
/// A private list of values that another project authoritatively defines is the
/// same hazard this workspace removed from `kiss_ref_diff::assert_conforming_eq`
/// last week, where a local mirror of kiss-ref's NaN-classing rule was replaced
/// by a call to `kiss_ref_core::ulp_distance_f32`. **A mirror cannot receive the
/// original's refinement**, and the divergence set is exactly the kind of thing
/// that gets refined — every new op with a discontinuity adds to it.
///
/// **What this file must NOT take from that corpus is the expected VALUES.** The
/// claim here is *preservation* — `eval(body)` equals `eval(optimize(body))` —
/// not correctness, so both sides go through this crate's own oracle and no
/// external authority is needed or wanted. The shared artifact is the INPUT SET,
/// nothing else.
const SPECIALS: &[f32] = &[
    0.0,
    -0.0,
    f32::NAN,
    -f32::NAN,
    f32::INFINITY,
    f32::NEG_INFINITY,
    f32::MIN_POSITIVE,           // smallest normal
    -f32::MIN_POSITIVE,          //
    f32::from_bits(1),           // smallest denormal
    f32::from_bits(0x8000_0001), // and its negation
    1.0,
    -1.0,
    2.0,
];

fn cross() -> (Vec<f32>, Vec<f32>) {
    let mut a = Vec::new();
    let mut b = Vec::new();
    for &x in SPECIALS {
        for &y in SPECIALS {
            a.push(x);
            b.push(y);
        }
    }
    (a, b)
}

fn eval_body(body: &ScalarExpr, n_inputs: u8, a: &[f32], b: &[f32]) -> Vec<f64> {
    let op = OpDef::elementwise("probe", n_inputs, &[ElementKind::F32], as_expr(body));
    let len = a.len() as i64;
    let d = OperandDesc::new(1, &[len], &[1], ElementKind::F32, 4);
    let operands: Vec<OperandDesc> = std::iter::repeat_n(d, usize::from(n_inputs) + 1).collect();
    let key = structure_key(
        if n_inputs == 1 {
            OpCategory::UnaryElementwise
        } else {
            OpCategory::BinaryElementwise
        },
        &operands,
        ArchSku::Sm89,
    );
    let plan = build_plan(&op, &key);
    let mut bufs = vec![TypedBuffer::from_f32(&[len], a)];
    if n_inputs > 1 {
        bufs.push(TypedBuffer::from_f32(&[len], b));
    }
    evaluate(&plan, &operands, &bufs, &[])
        .into_iter()
        .next()
        .expect("one output")
        .to_f64_vec()
}

/// `OpDef::elementwise` takes the builder's `Expr`, not a bare `ScalarExpr`.
/// This is the one place the two meet.
///
/// (Named `unsafe_wrap` in a first draft, which was a lie: there is nothing
/// unsafe here, and a name that says otherwise is the defect this workspace
/// spent a week removing from other people's code.)
fn as_expr(e: &ScalarExpr) -> unpopped::ir::Expr {
    unpopped::ir::Expr(e.clone())
}

fn abs(e: ScalarExpr) -> ScalarExpr {
    ScalarExpr::Unary(UnaryOp::Abs, Box::new(e))
}

fn neg(e: ScalarExpr) -> ScalarExpr {
    ScalarExpr::Sub(Box::new(konst(0.0).0), Box::new(e))
}

/// Bodies chosen for the rewrites they invite, including the ones that would be
/// **unsound** if the optimizer ever performed them.
///
/// The last four are the classic traps, and they are here precisely because a
/// rewriter that "simplifies" them looks obviously right and is obviously wrong:
///
/// - `x * 0.0` is NOT `0.0` — it is NaN for `x = inf`, and `-0.0` for `x < 0`
/// - `x - x` is NOT `0.0` — it is NaN for `x = inf` or `x = NaN`
/// - `x / x` is NOT `1.0` — it is NaN for `x = 0` and for `x = inf`
/// - `x + 0.0` is NOT `x` — it is `+0.0` when `x` is `-0.0`
fn corpus() -> Vec<(&'static str, u8, ScalarExpr)> {
    vec![
        ("mul_one", 1, (input(0) * konst(1.0)).0),
        ("div_two", 1, (input(0) / konst(2.0)).0),
        ("neg_neg", 1, neg(neg(ScalarExpr::Input(0)))),
        ("abs_abs", 1, abs(abs(ScalarExpr::Input(0)))),
        ("sub_zero", 1, (input(0) - konst(0.0)).0),
        ("add_sub", 2, ((input(0) + input(1)) - input(1)).0),
        ("untouched", 2, (input(0) + input(1)).0),
        // The four ops this file exists for. Absent from the first revision,
        // which is the same defect KISS #329 records one level up: a corpus
        // that omits the case it was built for.
        ("max_prop", 2, input(0).max(input(1)).0),
        ("min_prop", 2, input(0).min(input(1)).0),
        (
            "fmax_ieee",
            2,
            ScalarExpr::Binary(
                BinaryOp::FmaxIeee,
                Box::new(ScalarExpr::Input(0)),
                Box::new(ScalarExpr::Input(1)),
            ),
        ),
        (
            "fmin_ieee",
            2,
            ScalarExpr::Binary(
                BinaryOp::FminIeee,
                Box::new(ScalarExpr::Input(0)),
                Box::new(ScalarExpr::Input(1)),
            ),
        ),
        // the traps
        ("mul_zero", 1, (input(0) * konst(0.0)).0),
        ("self_sub", 1, (input(0) - input(0)).0),
        ("self_div", 1, (input(0) / input(0)).0),
        ("add_zero", 1, (input(0) + konst(0.0)).0),
    ]
}

#[test]
fn optimizing_agrees_bit_for_bit_on_every_value_a_rewrite_could_break() {
    let (a, b) = cross();
    let mut rewritten = 0;

    for (name, n_inputs, body) in corpus() {
        let opt = optimize(&body, ElementKind::F32);
        if opt != body {
            rewritten += 1;
        }

        let before = eval_body(&body, n_inputs, &a, &b);
        let after = eval_body(&opt, n_inputs, &a, &b);
        assert_eq!(before.len(), after.len(), "{name}: length");

        for (i, (x, y)) in before.iter().zip(&after).enumerate() {
            assert_eq!(
                x.to_bits(),
                y.to_bits(),
                "{name}: optimizing CHANGED THE ANSWER at input ({:?}, {:?}) — \
                 {x:?} (0x{:016x}) became {y:?} (0x{:016x}). This is the class of \
                 defect no random-input differential finds: the two forms agree \
                 everywhere except the values in SPECIALS.\n  before: {body:?}\n  \
                 after:  {opt:?}",
                a[i],
                b[i],
                x.to_bits(),
                y.to_bits()
            );
        }
    }

    // Vacuity control. If the optimizer rewrote nothing, every comparison is a
    // body against itself and the assertion above holds for an identity
    // function. The corpus is chosen to invite rewrites; if none fire, either
    // the optimizer is off or the corpus has drifted away from what it rewrites.
    assert!(
        rewritten >= 3,
        "only {rewritten} of the corpus was rewritten — the rest compares each \
         body against itself, so this cannot distinguish an answer-preserving \
         optimizer from one that does nothing"
    );
}

/// The corpus is the divergence set, not a sample of it.
///
/// A future edit that "tidies" `SPECIALS` down to a few round numbers would
/// leave the test above green and blind, because every defect it exists to
/// catch lives on exactly the values that would be removed.
#[test]
fn the_corpus_still_contains_the_values_rewrites_break_on() {
    let has = |p: fn(f32) -> bool| SPECIALS.iter().copied().any(p);

    assert!(has(|v| v == 0.0 && v.is_sign_positive()), "+0.0 missing");
    assert!(has(|v| v == 0.0 && v.is_sign_negative()), "-0.0 missing");
    assert!(has(f32::is_nan), "NaN missing");
    assert!(has(|v| v.is_infinite() && v > 0.0), "+inf missing");
    assert!(has(|v| v.is_infinite() && v < 0.0), "-inf missing");
    assert!(
        has(|v| v != 0.0 && v.abs() < f32::MIN_POSITIVE),
        "denormal missing — the value a flush-to-zero rewrite breaks on"
    );
    assert!(
        has(|v| v.is_finite() && v.abs() >= 1.0),
        "no ordinary magnitude — without one, a body that disagrees EVERYWHERE \
         is indistinguishable from one that disagrees only on the specials"
    );
}

/// **The corpus can tell the NaN-propagating ops from the NaN-suppressing ones.**
///
/// This is KISS #329's defect, checked against this file's own input set rather
/// than assumed absent from it.
///
/// KISS ships 48 vectors over `fmax_ieee` / `fmin_ieee` / `max_prop` /
/// `min_prop` and **all 48 pass with `max_prop` and `fmax_ieee` swapped**,
/// because it has no NaN vectors and the entire difference between the pairs is
/// NaN behaviour. An artefact that enumerates four ops and cannot distinguish
/// the property that makes them four is green carrying no information.
///
/// The preservation test above would inherit that defect silently: if `SPECIALS`
/// could not separate `Max` from `FmaxIeee`, then an optimizer that rewrote one
/// into the other would be *answer-preserving on this corpus* and pass. The
/// corpus would be endorsing the substitution it exists to catch.
///
/// So: require the two forms to actually disagree somewhere in `SPECIALS`. This
/// is not a claim about the optimizer at all — it is a claim about the inputs,
/// and it is the one that makes every claim about the optimizer meaningful.
#[test]
fn the_corpus_separates_nan_propagating_from_nan_suppressing() {
    let (a, b) = cross();
    let bin = |op: BinaryOp| {
        ScalarExpr::Binary(
            op,
            Box::new(ScalarExpr::Input(0)),
            Box::new(ScalarExpr::Input(1)),
        )
    };

    for (prop, ieee, name) in [
        (BinaryOp::Max, BinaryOp::FmaxIeee, "max"),
        (BinaryOp::Min, BinaryOp::FminIeee, "min"),
    ] {
        let p = eval_body(&bin(prop), 2, &a, &b);
        let i = eval_body(&bin(ieee), 2, &a, &b);
        // Restricted to inputs where a NaN is actually involved.
        //
        // An earlier revision counted EVERY differing position and PASSED WITH
        // NaN REMOVED FROM THE CORPUS — because these pairs also differ on
        // signed zero. It discriminated the ops, but on the wrong axis, so it
        // would have certified a NaN-blind corpus as adequate: the precise
        // shape of KISS #329, reproduced inside the test written to avoid it.
        //
        // Caught by mutating SPECIALS down to KISS's shape and watching this
        // test stay green while its sibling went red. A guard that passes for
        // an adjacent reason is the failure this file is about.
        let differing = p
            .iter()
            .zip(&i)
            .enumerate()
            .filter(|(k, _)| a[*k].is_nan() || b[*k].is_nan())
            .filter(|(_, (x, y))| x.to_bits() != y.to_bits())
            .count();

        assert!(
            differing > 0,
            "{name}_prop and f{name}_ieee produce IDENTICAL output on every input \
             in SPECIALS. The corpus cannot separate NaN-propagating from \
             NaN-suppressing, which is the exact property `ops.md` says makes \
             them distinct ops — so the preservation test above would pass an \
             optimizer that rewrote one into the other. This is KISS #329's \
             defect, in this file."
        );
    }
}
