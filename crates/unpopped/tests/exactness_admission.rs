// SPDX-License-Identifier: MIT OR Apache-2.0
//! **What a ledger may admit, what it must decline, and what the file keeps.**
//!
//! `docs/design-per-device-decline.md`, sections 2 to 5. Every row that must decline has its
//! admitting control beside it, because a decline with no control also passes when the ledger is
//! never read at all.

use unpopped::exactness::{
    Admission, AdmissionTally, ArgsOffer, Coverage, DeviceStamp, ExactnessLedger, ExactnessRecord,
    ExactnessVerdict, Finding, LineError, Measured, Requirement, Withheld,
};
use unpopped::ir::{ScalarExpr, UnaryOp};
use unpopped_vocab::{ElementKind, HwStamp, TargetId};

const VK: &str = "vulkan:sg64.ops-abr.arith-f16.cm-none";
const PRESERVE: [&str; 2] = ["-denorm-mode-fp32", "preserve"];
const HASH: &str = "sha256:aa";
const COMPILER: &str = "slangc 2026.4";

fn stamp_with(compiler: &str, args: &[&str]) -> DeviceStamp {
    DeviceStamp::new(
        HwStamp {
            target: TargetId::parse(VK).unwrap(),
            device_name: "AMD Radeon 610M".into(),
            runtime_version: "1.3.280".into(),
            captured_unix_s: 7,
        },
        "24.1",
        compiler,
        args.iter().map(|s| (*s).to_string()).collect(),
    )
}

fn stamp(args: &[&str]) -> DeviceStamp {
    stamp_with(COMPILER, args)
}

fn relu() -> ScalarExpr {
    ScalarExpr::Unary(UnaryOp::Relu, Box::new(ScalarExpr::Input(0)))
}

fn rec(s: &DeviceStamp, verdict: ExactnessVerdict, hash: &str, cov: Coverage) -> ExactnessRecord {
    ExactnessRecord::new(
        s.clone(),
        &relu(),
        ElementKind::F32,
        verdict,
        hash,
        "unpopped-slang 0.21.0",
        cov,
    )
}

fn exhaustive() -> Coverage {
    Coverage::exhaustive(1 << 32)
}

fn ledger(records: Vec<ExactnessRecord>) -> ExactnessLedger {
    let mut l = ExactnessLedger::new();
    for r in records {
        l.push(r);
    }
    l
}

fn find(l: &ExactnessLedger, s: &DeviceStamp, hash: &str) -> Finding {
    l.find(s, hash, &relu(), ElementKind::F32)
}

fn admit(l: &ExactnessLedger, s: &DeviceStamp, hash: &str, req: Requirement) -> Admission {
    l.admit(s, hash, &relu(), ElementKind::F32, req)
}

// --- the record must name what it was baked against -----------------------------------------

#[test]
fn a_record_applies_only_to_the_kernel_it_measured() {
    let s = stamp(&PRESERVE);
    let l = ledger(vec![rec(&s, ExactnessVerdict::Exact, HASH, exhaustive())]);
    assert_eq!(find(&l, &s, HASH), Finding::Exact, "control: same hash");
    assert_eq!(
        find(&l, &s, "sha256:bb"),
        Finding::Withheld(Withheld::StaleKernel),
        "a re-emitted kernel cannot inherit the old Exact"
    );
}

#[test]
fn a_record_for_another_compiler_version_is_a_different_device() {
    let measured = stamp(&PRESERVE);
    let l = ledger(vec![rec(
        &measured,
        ExactnessVerdict::Exact,
        HASH,
        exhaustive(),
    )]);
    assert_eq!(find(&l, &measured, HASH), Finding::Exact, "control");
    let upgraded = stamp_with("slangc 2026.5", &PRESERVE);
    assert_eq!(
        find(&l, &upgraded, HASH),
        Finding::Withheld(Withheld::Unmeasured),
        "a compiler upgrade invalidates the record, even with the same arguments and hash"
    );
}

#[test]
fn only_an_exhaustive_run_can_admit_and_a_sampled_mismatch_still_declines() {
    let s = stamp(&PRESERVE);
    let sampled = Coverage::sampled(65_536, Some(1));
    let exact_sampled = ledger(vec![rec(&s, ExactnessVerdict::Exact, HASH, sampled)]);
    let exact_full = ledger(vec![rec(&s, ExactnessVerdict::Exact, HASH, exhaustive())]);
    assert_eq!(find(&exact_full, &s, HASH), Finding::Exact, "control");
    assert_eq!(
        find(&exact_sampled, &s, HASH),
        Finding::Withheld(Withheld::SampledOnly)
    );
    let flush_sampled = ledger(vec![rec(&s, ExactnessVerdict::Flushes, HASH, sampled)]);
    assert_eq!(
        find(&flush_sampled, &s, HASH),
        Finding::Withheld(Withheld::Flushes),
        "one mismatch is a mismatch, sampled or not"
    );
}

#[test]
fn a_flush_outweighs_an_exact_in_either_order_and_a_stale_flush_does_not() {
    let s = stamp(&PRESERVE);
    let e = || rec(&s, ExactnessVerdict::Exact, HASH, exhaustive());
    let f = || rec(&s, ExactnessVerdict::Flushes, HASH, exhaustive());
    for l in [ledger(vec![e(), f()]), ledger(vec![f(), e()])] {
        assert_eq!(find(&l, &s, HASH), Finding::Withheld(Withheld::Flushes));
    }
    let stale_flush = rec(&s, ExactnessVerdict::Flushes, "sha256:old", exhaustive());
    let l = ledger(vec![e(), stale_flush]);
    assert_eq!(
        find(&l, &s, HASH),
        Finding::Exact,
        "a mismatch on another kernel says nothing of this one"
    );
}

// --- admission -------------------------------------------------------------------------------

#[test]
fn unmeasured_and_flushes_are_different_answers() {
    let s = stamp(&[]);
    let flushed = ledger(vec![rec(&s, ExactnessVerdict::Flushes, HASH, exhaustive())]);
    let empty = ExactnessLedger::new();
    assert_eq!(
        admit(&flushed, &s, HASH, Requirement::BestEffort),
        Admission::ClaimWithheld(Withheld::Flushes)
    );
    assert_eq!(
        admit(&empty, &s, HASH, Requirement::BestEffort),
        Admission::ClaimWithheld(Withheld::Unmeasured),
        "collapsing these two hides that a device was never measured"
    );
}

#[test]
fn best_effort_keeps_the_kernel_and_exact_declines_it() {
    let s = stamp(&[]);
    let l = ledger(vec![rec(&s, ExactnessVerdict::Flushes, HASH, exhaustive())]);
    assert_eq!(
        admit(&l, &s, HASH, Requirement::BestEffort),
        Admission::ClaimWithheld(Withheld::Flushes)
    );
    assert_eq!(
        admit(&l, &s, HASH, Requirement::Exact),
        Admission::Declined(Withheld::Flushes),
        "a typed value, not an error"
    );
    let good = stamp(&PRESERVE);
    let l2 = ledger(vec![rec(
        &good,
        ExactnessVerdict::Exact,
        HASH,
        exhaustive(),
    )]);
    for req in [Requirement::BestEffort, Requirement::Exact] {
        assert_eq!(admit(&l2, &good, HASH, req), Admission::Admitted, "control");
    }
}

#[test]
fn a_flushing_default_offers_the_preserving_arguments() {
    let default = stamp(&[]);
    let preserve = stamp(&PRESERVE);
    let l = ledger(vec![
        rec(&default, ExactnessVerdict::Flushes, HASH, exhaustive()),
        rec(
            &preserve,
            ExactnessVerdict::Exact,
            "sha256:pp",
            exhaustive(),
        ),
    ]);
    assert_eq!(
        admit(&l, &default, HASH, Requirement::Exact),
        Admission::UseArgs(ArgsOffer {
            compiler_args: PRESERVE.iter().map(|s| (*s).to_string()).collect(),
            kernel_hash: "sha256:pp".into(),
        })
    );
    assert_eq!(
        admit(&l, &default, HASH, Requirement::BestEffort),
        Admission::ClaimWithheld(Withheld::Flushes),
        "an offer is made only to a caller who required Exact"
    );
}

/// A ledger with a flushing default and a variant record built by `variant`, asked for Exact.
fn offer_for(variant: Vec<ExactnessRecord>) -> Admission {
    let default = stamp(&[]);
    let mut records = vec![rec(&default, ExactnessVerdict::Flushes, HASH, exhaustive())];
    records.extend(variant);
    admit(&ledger(records), &default, HASH, Requirement::Exact)
}

fn variant_exact(s: &DeviceStamp, cov: Coverage) -> ExactnessRecord {
    rec(s, ExactnessVerdict::Exact, "sha256:pp", cov)
}

#[test]
fn a_sound_variant_record_is_offered() {
    let p = stamp(&PRESERVE);
    assert!(matches!(
        offer_for(vec![variant_exact(&p, exhaustive())]),
        Admission::UseArgs(_)
    ));
}

/// Each of the next three has the control above beside it: the same ledger with a sound record.
#[test]
fn a_sampled_only_variant_cannot_admit_so_it_is_not_offered() {
    let p = stamp(&PRESERVE);
    let r = offer_for(vec![variant_exact(&p, Coverage::sampled(10, None))]);
    assert_eq!(r, Admission::Declined(Withheld::Flushes));
}

#[test]
fn a_variant_outweighed_by_its_own_flushes_is_not_offered() {
    let p = stamp(&PRESERVE);
    let flush = rec(&p, ExactnessVerdict::Flushes, "sha256:pp", exhaustive());
    let r = offer_for(vec![variant_exact(&p, exhaustive()), flush]);
    assert_eq!(r, Admission::Declined(Withheld::Flushes));
}

#[test]
fn a_variant_measured_under_another_compiler_is_a_different_device() {
    let other = stamp_with("slangc 2026.5", &PRESERVE);
    let r = offer_for(vec![variant_exact(&other, exhaustive())]);
    assert_eq!(r, Admission::Declined(Withheld::Flushes));
}

#[test]
fn the_tally_counts_each_reason_per_stamp_and_not_admissions() {
    let default = stamp(&[]);
    let preserve = stamp(&PRESERVE);
    let l = ledger(vec![rec(
        &default,
        ExactnessVerdict::Flushes,
        HASH,
        exhaustive(),
    )]);
    let mut t = AdmissionTally::new();
    for _ in 0..2 {
        t.observe(
            &default,
            &admit(&l, &default, HASH, Requirement::BestEffort),
        );
    }
    t.observe(
        &preserve,
        &admit(&l, &preserve, HASH, Requirement::BestEffort),
    );
    t.observe(&default, &Admission::Admitted);
    assert_eq!(t.count(&default, Withheld::Flushes), 2);
    assert_eq!(t.count(&preserve, Withheld::Unmeasured), 1);
    assert_eq!(
        t.count(&default, Withheld::Unmeasured),
        0,
        "control: a reason never seen counts 0, not the total"
    );
    assert_eq!(t.total(), 3, "the admission counted nothing");
}

// --- the contract reads the hash -------------------------------------------------------------

#[test]
fn the_precision_claim_follows_the_kernel_hash() {
    use unpopped::contract::{AccuracyKey, MeasuredOn, precision_of_measured};
    let s = stamp(&PRESERVE);
    let l = ledger(vec![rec(&s, ExactnessVerdict::Exact, HASH, exhaustive())]);
    let key = AccuracyKey::for_target(s.hw.target);
    let m = |hash| Measured::new(&s, hash, &l);
    let ask = |hash| {
        let mm = m(hash);
        let on = MeasuredOn {
            stamp: mm.stamp,
            kernel_hash: mm.kernel_hash,
            ledger: mm.ledger,
            dtype: ElementKind::F32,
        };
        precision_of_measured(&relu(), &key, Some(&on))
    };
    assert_eq!(ask(HASH), ("correctly_rounded", Some(0)), "control");
    assert_eq!(
        ask("sha256:bb"),
        ("approximate", None),
        "a stale hash claims nothing"
    );
}

// --- the file --------------------------------------------------------------------------------

#[test]
fn the_file_round_trips() {
    let s = stamp(&PRESERVE);
    let l = ledger(vec![
        rec(&s, ExactnessVerdict::Exact, HASH, exhaustive()),
        rec(
            &stamp(&[]),
            ExactnessVerdict::Flushes,
            "sha256:bb",
            Coverage::sampled(256, Some(9)),
        ),
    ]);
    let text = l.to_jsonl();
    let read = ExactnessLedger::parse_jsonl(&text);
    assert!(read.rejected.is_empty());
    assert_eq!(read.ledger, l);
}

/// The wire form is a contract with the harness that wrote its lines before this schema moved
/// here, so the first field and the coverage spelling are pinned.
#[test]
fn the_wire_form_starts_with_the_schema_and_spells_coverage_as_the_harness_does() {
    let l = ledger(vec![rec(
        &stamp(&PRESERVE),
        ExactnessVerdict::Exact,
        HASH,
        exhaustive(),
    )]);
    let text = l.to_jsonl();
    let first = text.lines().next().unwrap();
    assert!(
        first.starts_with("{\"schema\":\"unpopped-exactness-ledger/1\",\"target\":\"vulkan:"),
        "schema first: {first}"
    );
    assert!(
        first
            .contains("\"coverage\":{\"kind\":\"exhaustive\",\"seed\":null,\"tuples\":4294967296}")
    );
}

#[test]
fn a_malformed_line_is_counted_with_its_number_and_never_dropped_silently() {
    let good = rec(
        &stamp(&PRESERVE),
        ExactnessVerdict::Exact,
        HASH,
        exhaustive(),
    )
    .to_line();
    let unknown = good.replace("exactness-ledger/1", "exactness-ledger/2");
    let no_hash = good.replace("\"kernel_hash\":\"sha256:aa\",", "");
    let text = format!("{good}\n\nnot json\n{unknown}\n{no_hash}\n{good}\n");
    let read = ExactnessLedger::parse_jsonl(&text);
    assert_eq!(read.ledger.len(), 2, "control: both good lines read");
    assert_eq!(
        read.rejected,
        vec![
            (3, LineError::NotJson(read_err(&read, 3))),
            (
                4,
                LineError::UnknownSchema("unpopped-exactness-ledger/2".into())
            ),
            (5, LineError::Field("kernel_hash")),
        ],
        "every non-blank line is either read or counted; the blank line is neither"
    );
    assert_eq!(
        read.ledger.len() + read.rejected.len(),
        5,
        "5 non-blank lines accounted for"
    );
}

fn read_err(read: &unpopped::exactness::LedgerRead, line: usize) -> String {
    match &read.rejected.iter().find(|(n, _)| *n == line).unwrap().1 {
        LineError::NotJson(m) => m.clone(),
        other => panic!("line {line} was {other:?}"),
    }
}

#[test]
fn a_field_this_version_does_not_know_survives_a_round_trip() {
    let line = rec(&stamp(&[]), ExactnessVerdict::Flushes, HASH, exhaustive()).to_line();
    let with_extra = format!(
        "{},\"added_later\":{{\"x\":1}}}}",
        line.trim_end_matches('}')
    );
    let read = ExactnessLedger::parse_jsonl(&with_extra);
    assert!(read.rejected.is_empty());
    let written = read.ledger.to_jsonl();
    assert!(
        written.contains("\"added_later\":{\"x\":1}"),
        "a newer writer's field was dropped: {written}"
    );
    let _ = ExactnessRecord::parse_line(written.trim_end()).expect("and it still parses");
}

#[test]
fn a_target_that_does_not_parse_is_a_counted_line_not_a_panic() {
    let good = rec(&stamp(&[]), ExactnessVerdict::Exact, HASH, exhaustive()).to_line();
    let bad = good.replace("\"target\":\"vulkan:", "\"target\":\"nonamespace");
    assert_ne!(bad, good, "the mutation applied");
    let read = ExactnessLedger::parse_jsonl(&bad);
    assert_eq!(read.rejected, vec![(1, LineError::Field("target"))]);
}

/// A line as `unpopped-harness` 0.0.9 wrote it, before this schema moved here: it must keep
/// reading, and writing it back must reproduce it byte for byte.
#[test]
fn a_line_the_harness_wrote_before_the_schema_moved_still_reads() {
    let line = concat!(
        "{\"schema\":\"unpopped-exactness-ledger/1\",\"target\":\"vulkan:sg64.ops-abr.arith-f16.cm-none\",",
        "\"device_name\":\"AMD Radeon 610M\",\"runtime_version\":\"1.3.280\",\"driver_version\":\"24.1\",",
        "\"compiler_args\":[\"-denorm-mode-fp32\",\"preserve\"],\"compiler_version\":\"slangc 2026.4\",",
        "\"key\":\"f32:{\\\"Unary\\\":[\\\"Relu\\\",{\\\"Input\\\":0}]}\",\"verdict\":\"exact\",\"kernel_hash\":\"sha256:aa\",",
        "\"emitter_version\":\"unpopped-slang 0.20.19\",",
        "\"coverage\":{\"kind\":\"exhaustive\",\"seed\":null,\"tuples\":4294967296},\"captured_unix_s\":7}"
    );
    let r = ExactnessRecord::parse_line(line).expect("a harness-0.0.9 line reads");
    assert_eq!(r.verdict, ExactnessVerdict::Exact);
    assert_eq!(r.stamp, stamp(&PRESERVE));
    assert_eq!(r.key, unpopped::body_key(&relu(), ElementKind::F32));
    assert_eq!(r.to_line(), line);
}

/// A stale kernel is not rescued by "use the arguments you already have": the record that went
/// stale names those same arguments, so offering them back would loop the adopter.
#[test]
fn a_stale_kernel_is_not_offered_its_own_arguments() {
    let s = stamp(&PRESERVE);
    let l = ledger(vec![rec(&s, ExactnessVerdict::Exact, HASH, exhaustive())]);
    assert_eq!(
        admit(&l, &s, HASH, Requirement::Exact),
        Admission::Admitted,
        "control"
    );
    assert_eq!(
        admit(&l, &s, "sha256:rebuilt", Requirement::Exact),
        Admission::Declined(Withheld::StaleKernel)
    );
}
