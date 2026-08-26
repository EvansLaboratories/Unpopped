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

use unpopped::ir::{OpDef, ScalarExpr, UnaryOp, input, konst};
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
/// whose `tags` field already carries `signed-zero`. **Its coverage is one op of
/// 106 and it has zero NaN vectors today**, so it cannot yet supply this — but
/// when it can, `SPECIALS` should READ it rather than restate it.
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
