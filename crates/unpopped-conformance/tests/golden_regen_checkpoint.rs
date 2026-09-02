//! Section B is gated on a coordinated golden regen. This is the thing that
//! makes that gate fire.
//!
//! # The failure mode this exists to prevent
//!
//! `docs/deferred.md` Section B holds three items — the f16/bf16 spelling seam,
//! the temp-binding pass, and a Slang complex prelude — each correctly deferred
//! because it rewrites emitted text and would break byte-identity goldens
//! including baracuda's physical CUDA corpus.
//!
//! **They were gated on "a coordinated golden regen" with no owner and no date.**
//! That is not a deferral. It is a permanent hold wearing a deferral's clothes:
//! an item that reads as deliberate sequencing to whoever opens the list next,
//! and which nothing will ever dislodge, because the trigger is an event nobody
//! is responsible for causing.
//!
//! **A deferral tied to *"when the regen happens"* never fires if the regen never
//! happens. One tied to a date always fires.** That distinction is the whole
//! design of this file, and it is the portfolio PM's, not mine.
//!
//! # What to do when it goes red
//!
//! It is a decision point, not a bug. Exactly one of:
//!
//! 1. **Schedule the regen** — do the three items together, regenerate the
//!    goldens across this workspace and baracuda, and delete this file.
//! 2. **Move the date**, in a commit that says *why* the window slipped and who
//!    agreed. A moved date with a reason is a live deferral; a moved date without
//!    one is this defect coming back.
//!
//! **Deleting the constant to make CI green is the third option and it is the
//! one this file exists to make visible.**

/// The date Section B's hold comes back to the portfolio PM.
///
/// Set 2026-10-01 when `unpopped 0.7.0` published (2026-09-02) and the window
/// for a coordinated regen opened. **Owner: the portfolio PM**, who accepts the
/// regen proposal; this workspace drafts and implements it.
const REGEN_CHECKPOINT: (u32, u32, u32) = (2026, 10, 1);

/// Days since the Unix epoch for a civil date (proleptic Gregorian).
///
/// Hand-rolled rather than pulled in as a dependency: this crate is
/// `publish = false` and holds evidence, not machinery, and a date library is a
/// large surface for one comparison. Howard Hinnant's `days_from_civil`.
fn days_from_civil(y: i64, m: i64, d: i64) -> i64 {
    let y = if m <= 2 { y - 1 } else { y };
    let era = if y >= 0 { y } else { y - 399 } / 400;
    let yoe = y - era * 400;
    let mp = (m + 9) % 12;
    let doy = (153 * mp + 2) / 5 + d - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    era * 146_097 + doe - 719_468
}

fn today_utc() -> (u32, u32, u32) {
    let secs = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .expect("the clock is after 1970")
        .as_secs() as i64;
    let mut z = secs / 86_400 + 719_468;
    let era = if z >= 0 { z } else { z - 146_096 } / 146_097;
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    z = if m <= 2 { y + 1 } else { y };
    (z as u32, m as u32, d as u32)
}

#[test]
fn section_b_comes_back_to_the_pm_on_its_checkpoint_date() {
    let (cy, cm, cd) = REGEN_CHECKPOINT;
    let (ty, tm, td) = today_utc();
    let checkpoint = days_from_civil(i64::from(cy), i64::from(cm), i64::from(cd));
    let today = days_from_civil(i64::from(ty), i64::from(tm), i64::from(td));

    assert!(
        today < checkpoint,
        "SECTION B'S GOLDEN REGEN IS DUE ({cy}-{cm:02}-{cd:02}; today is \
         {ty}-{tm:02}-{td:02}).\n\n\
         Three items in docs/deferred.md Section B are held on a coordinated \
         golden regen: the f16/bf16 spelling seam, the temp-binding pass, and a \
         Slang complex prelude. Each rewrites emitted text and breaks \
         byte-identity goldens including baracuda's CUDA corpus, so they ride \
         one regen rather than three cleanup commits.\n\n\
         This is a DECISION POINT, not a bug. Either:\n  \
         (1) schedule the regen, do the three together, and delete this file; or\n  \
         (2) move REGEN_CHECKPOINT, in a commit that says why the window slipped \
         and who agreed.\n\n\
         NOT waiting on a decision — the shape is known. The override-mechanism \
         sub-question is settled (consumer-side shadow, no shared API); what remains \
         is that the neutral module must stop spelling a vendor type, adopting the \
         shape FP8 already proves: spell the STORAGE type and emit software helpers \
         into the kernel, no vendor intrinsics. FP8 got there first solely because it \
         had no goldens to rewrite.\n\n\
         Owner: the portfolio PM. A moved date with a reason is a live deferral; \
         a moved date without one is the hold-with-no-owner defect returning."
    );
}

/// The date arithmetic works, and the assertion above can actually fail.
///
/// Without this, a `days_from_civil` that returned a constant would make the
/// checkpoint permanently green and the guard permanently silent — which is the
/// same class of defect the guard itself exists to catch, one level down.
#[test]
fn the_date_comparison_discriminates() {
    // Known epoch anchors.
    assert_eq!(days_from_civil(1970, 1, 1), 0);
    assert_eq!(days_from_civil(2000, 3, 1), 11_017);
    // Ordering across the checkpoint, in both directions.
    let cp = days_from_civil(2026, 10, 1);
    assert!(
        days_from_civil(2026, 9, 30) < cp,
        "the day before is before"
    );
    assert!(days_from_civil(2026, 10, 2) > cp, "the day after is after");
    assert!(days_from_civil(2027, 1, 1) > cp, "a later year is after");
    // Leap-year handling, since a checkpoint straddling February would
    // otherwise be off by one for a whole month.
    assert_eq!(
        days_from_civil(2024, 3, 1) - days_from_civil(2024, 2, 28),
        2,
        "2024 is a leap year and Feb 29 was skipped"
    );
    // `today_utc` returns something plausible rather than a stub.
    let (y, m, d) = today_utc();
    assert!((2020..2100).contains(&y), "implausible year {y}");
    assert!((1..=12).contains(&m), "implausible month {m}");
    assert!((1..=31).contains(&d), "implausible day {d}");
}
