//! The neutral driver spells no constant and no special float value — behaviourally.
//!
//! Backs **KISS-EMIT-6.4-0001** (constant spelling) and **KISS-EMIT-6.4-0002**
//! (special-float-value spelling), both of which say the same thing in two
//! scopes:
//!
//! > MUST be an emitter-must-supply decision … **the neutral driver MUST NOT
//! > spell** a constant / a special float value.
//!
//! # Why this file exists
//!
//! Both clauses were **de-credited** in KISS #270. Their only citing test
//! reddened when the *spec text* was mutated, never when the *emitter
//! obligation* was — so it backed a document-placement rule and not the clause,
//! and the honest record in `conformance/UNBACKED.tsv` says "a behavioral
//! emitter test is owed".
//!
//! The architect asked the editor a sharper question than "write it": is the
//! obligation **real and unexercised**, or **not exercisable as written**? The
//! second is a clause defect and goes back to them; the first is a debt.
//!
//! **This file is the answer, given by construction rather than by argument.**
//! If the test can be written, the clause admits one. Writing it is cheaper than
//! being sure, and it was the only way to be sure.
//!
//! # How it witnesses a negative
//!
//! The obligation is *not to spell*, and an absence is not directly observable.
//! So the emitter seam is handed a spelling **no driver could invent** — a
//! sentinel — and the emitted text is checked for it. If the driver had spelled
//! the value itself, the sentinel would be missing and a plausible literal would
//! be there instead.
//!
//! That makes the failure mode the *interesting* one: a driver that quietly
//! learns to spell constants does not merely differ, it stops calling the seam,
//! and the sentinel disappears. Mutation-proven below by lowering the same DAG
//! against a seam that spells honestly, and asserting the sentinel is then
//! absent — so the assertion is known to distinguish the two worlds rather than
//! merely to pass in one of them.
//!
//! # Scope, stated because the clause is wider than this crate
//!
//! This proves the obligation for **`unpopped`'s driver** — `lower_expr` /
//! `lower_dag` over the [`Lowering`] seam. It is one implementation's
//! conformance evidence, not the clause's only possible witness: any emitter
//! reaching the same seam is covered by the same argument, and an emitter that
//! does not use this driver needs its own. KISS-EMIT-6.4-0005 is deliberately
//! **not** here — see the note at the end of this file.

use unpopped::backend::{LowerError, Lowering, Spelling, lower_expr};
use unpopped::ir::ScalarExpr;

const SENTINEL: &str = "__EMITTER_SUPPLIED_CONSTANT__";

/// Every seam spells a sentinel the driver has no way to invent.
///
/// `leaf` is the only one that must stay plausible: the driver composes around
/// it, and a sentinel there would make an arithmetic failure look like a
/// constant failure.
fn sentinel_seam() -> (
    impl Fn(u8) -> Result<Spelling, LowerError>,
    impl Fn(f64) -> Result<Spelling, LowerError>,
) {
    (
        |i: u8| Ok(Spelling::Spelled(format!("in{i}"))),
        |v: f64| Ok(Spelling::Spelled(format!("{SENTINEL}({v:?})"))),
    )
}

/// An honest speller, for the negative control: this is what the emitted text
/// looks like when a constant is rendered as a literal rather than routed.
fn literal_seam() -> (
    impl Fn(u8) -> Result<Spelling, LowerError>,
    impl Fn(f64) -> Result<Spelling, LowerError>,
) {
    (
        |i: u8| Ok(Spelling::Spelled(format!("in{i}"))),
        |v: f64| Ok(Spelling::Spelled(format!("{v:?}f"))),
    )
}

fn lower_with(
    e: &ScalarExpr,
    leaf: &dyn Fn(u8) -> Result<Spelling, LowerError>,
    constant: &dyn Fn(f64) -> Result<Spelling, LowerError>,
) -> String {
    let never_un = |_, _| -> Result<Spelling, LowerError> { panic!("no unary in this corpus") };
    let never_bin =
        |_, _, _| -> Result<Spelling, LowerError> { panic!("no binary in this corpus") };
    let arith = |_op, a: String, b: String| Ok(Spelling::Spelled(format!("({a} + {b})")));

    // Built through the builder, not a struct literal: `Lowering` is
    // `#[non_exhaustive]`, which is exactly the constraint an out-of-crate
    // backend meets — so this test reaches the seam the way a real emitter does
    // rather than through a shortcut only an in-crate test would have.
    let lo = Lowering::builder(leaf, &never_un, &never_bin)
        .arith(&arith)
        .constant(constant)
        .build();
    match lower_expr(e, &lo).expect("this corpus lowers") {
        Spelling::Spelled(s) => s,
        other => panic!("unexpected non-spelling: {other:?}"),
    }
}

/// `in0 + <v>` — the smallest shape that forces the driver to compose a
/// constant with something else, so a driver that folded or elided it would be
/// visible.
fn input_plus(v: f64) -> ScalarExpr {
    ScalarExpr::Add(
        Box::new(ScalarExpr::Input(0)),
        Box::new(ScalarExpr::Const(v)),
    )
}

/// KISS-EMIT-6.4-0001 — the driver renders no ordinary constant itself.
#[test]
fn the_driver_routes_every_ordinary_constant_through_the_emitter_seam() {
    let (leaf, constant) = sentinel_seam();
    let mut checked = 0;

    for v in [0.5_f64, 1.0, -1.0, 2.0, 1e30, -1e-30, std::f64::consts::PI] {
        let out = lower_with(&input_plus(v), &leaf, &constant);
        assert!(
            out.contains(SENTINEL),
            "the driver spelled the constant {v} itself instead of asking the \
             emitter — KISS-EMIT-6.4-0001. Emitted: {out}"
        );
        checked += 1;
    }

    assert_eq!(checked, 7, "every constant in the corpus must be exercised");
}

/// KISS-EMIT-6.4-0002 — same obligation, over the values with no portable
/// decimal spelling, plus the finite edges the clause groups with them.
///
/// These are the ones where a driver that spelled locally would be *most*
/// tempting and *most* wrong: `inf` and NaN have no portable literal at all,
/// and `-0.0` is the value a careless render silently turns into `0.0`.
#[test]
fn the_driver_routes_every_special_float_value_through_the_emitter_seam() {
    let (leaf, constant) = sentinel_seam();

    let cases: [(&str, f64); 6] = [
        ("+inf", f64::INFINITY),
        ("-inf", f64::NEG_INFINITY),
        ("quiet NaN", f64::NAN),
        ("+0", 0.0),
        ("-0", -0.0),
        ("subnormal", f64::from_bits(1)),
    ];

    for (name, v) in cases {
        let out = lower_with(&input_plus(v), &leaf, &constant);
        assert!(
            out.contains(SENTINEL),
            "the driver spelled {name} itself instead of asking the emitter — \
             KISS-EMIT-6.4-0002. Emitted: {out}"
        );
    }
}

/// The negative control, without which neither assertion above means anything.
///
/// Both tests assert a substring is PRESENT. A test that only ever checks for
/// presence passes just as happily against a driver that emits the sentinel for
/// unrelated reasons, or against a `lower_expr` that returns its input. So:
/// lower the same expressions against a seam that spells honestly, and require
/// the sentinel to be **absent** and a real literal to be there.
///
/// This is the assertion that would go red if the driver ever learned to spell
/// constants — because then the sentinel seam's output would look like the
/// literal seam's, and these two tests could no longer disagree.
#[test]
fn the_sentinel_is_absent_when_the_emitter_spells_honestly() {
    let (leaf, literal) = literal_seam();
    let (_, sentinel) = sentinel_seam();

    for v in [0.5_f64, f64::INFINITY, -0.0] {
        let honest = lower_with(&input_plus(v), &leaf, &literal);
        let routed = lower_with(&input_plus(v), &leaf, &sentinel);

        assert!(
            !honest.contains(SENTINEL),
            "the honest seam must not produce the sentinel: {honest}"
        );
        assert_ne!(
            honest, routed,
            "the two seams produced identical output for {v}, so the presence \
             assertions above cannot tell a routed constant from a driver-spelled \
             one and prove nothing"
        );
    }
}

// KISS-EMIT-6.4-0005 IS NOT BACKED HERE, AND DELIBERATELY SO.
//
// It reads: any operator the neutrality audit (§6.5) did NOT prove universal
// across the emitter's claimed targets MUST be emitter-must-supply.
//
// Its subject is the COMPLEMENT of an audit result — and §6.5-0001 makes the
// recorded audit manifest a FREEZE PRECONDITION, so at Draft maturity no such
// manifest exists. The set of "operators the audit did not prove universal" has
// no value yet, so no test can construct a member of it, so no test can
// construct a violation.
//
// That is not the same debt as -0001 and -0002. Those were exercisable all
// along and nobody had written the test — this file is the proof, since it
// exists. -0005 is BLOCKED on a named artifact the spec itself sequences, and
// it becomes provable the moment that artifact lands. Recording it as `untested`
// invites someone to write a test that cannot exist; recording it as `blocked`
// with §6.5-0001 named says when it stops being blocked.
