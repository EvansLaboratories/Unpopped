//! What this generator can actually **emit**, per dtype, per backend.
//!
//! # Why this is a test and not a line in a doc
//!
//! The deferred register carried these numbers in prose — "22/22 named, CpuC
//! lowers 9, Slang 5" — and by the time anyone read them they were wrong: the
//! measured figures are **8 and 4**. Not a large error, and that is the point.
//! A coverage claim decays silently, because nothing about adding or removing a
//! dtype arm forces the sentence describing it to change.
//!
//! So the claim lives here, as a table that has to be edited when the behaviour
//! moves. Adding a dtype to a backend fails this test until the table is
//! updated, which makes the update part of the change rather than a follow-up
//! nobody schedules.
//!
//! # Recognition is not lowering, and the gap is the useful number
//!
//! `unpopped-vocab` **recognizes** all 24 KISS §6.1 tokens and that is verified
//! against KISS's generated manifest. This file measures something different and
//! much smaller: for how many of those can this generator actually produce a
//! kernel. Conflating the two would let "we support 24 dtypes" mean two
//! incompatible things depending on who is asking — which is exactly the
//! distinction KISS draws between its recognition set and its usable set.
//!
//! A decline here is **conformant**, not a bug: a party that cannot lower a
//! dtype must still *name* it, so it can decline as a known dtype rather than as
//! an unknown token. This file pins which is which.

use unpopped::cpu_c::CpuC;
use unpopped::ir::{OpDef, input};
use unpopped::slang::Slang;
use unpopped::try_generate;
use unpopped_vocab::{ArchSku, ElementKind, OpCategory, OperandDesc, structure_key};

/// Every §6.1 dtype, with what each backend does with a plain elementwise `Add`.
///
/// `true` = lowers. Reserved dtypes are absent entirely: `f8e4m3fnuz` and
/// `f8e5m2fnuz` have no computation semantics at this schema version and must
/// never lower, which is asserted separately below rather than encoded as
/// `false` alongside merely-unimplemented ones. **"Forbidden" and "not done yet"
/// are different facts and should not share a column.**
const COVERAGE: &[(&str, ElementKind, bool, bool)] = &[
    // name          dtype                      CpuC   Slang
    ("f16", ElementKind::F16, false, false),
    ("bf16", ElementKind::Bf16, false, false),
    ("f32", ElementKind::F32, true, true),
    ("f64", ElementKind::F64, true, true),
    ("i8", ElementKind::I8, true, false),
    ("i16", ElementKind::I16, true, false),
    ("u8", ElementKind::U8, true, false),
    ("u16", ElementKind::U16, true, false),
    ("i32", ElementKind::I32, true, true),
    ("i64", ElementKind::I64, true, true),
    ("u32", ElementKind::U32, false, false),
    ("u64", ElementKind::U64, false, false),
    ("bool", ElementKind::Bool, false, false),
    ("f8e4m3fn", ElementKind::Fp8E4M3FN, false, false),
    ("f8e5m2", ElementKind::Fp8E5M2, false, false),
    ("f8e8m0", ElementKind::F8E8M0, false, false),
    ("f8e6m2", ElementKind::F8E6M2, false, false),
    ("i4", ElementKind::I4, false, false),
    ("u4", ElementKind::U4, false, false),
    ("b1", ElementKind::B1, false, false),
    ("c64", ElementKind::Complex64, false, false),
    ("c128", ElementKind::Complex128, false, false),
];

/// An extent and alignment that select the **scalar** schedule.
///
/// Deliberately 7 elements at 4-byte alignment: both backends lower only
/// `Schedule::Scalar`, and a divisible extent at a wide alignment elects a
/// vectorized schedule instead. Getting this wrong does not fail loudly — it
/// reports `UnsupportedSchedule` for *every* dtype and reads as "nothing is
/// supported". It cost me a 0/22 measurement before I noticed `f32` was
/// declining for a reason that had nothing to do with dtypes, which is why the
/// schedule is pinned by its own control below.
fn scalar_shape(dt: ElementKind) -> OperandDesc {
    OperandDesc::new(1, &[7], &[1], dt, 4)
}

fn lowers(dt: ElementKind, slang: bool) -> bool {
    let op = OpDef::elementwise("add", 2, &[dt], input(0) + input(1));
    let d = scalar_shape(dt);
    let key = structure_key(OpCategory::BinaryElementwise, &[d, d, d], ArchSku::Sm89);
    // A plan-gate rejection is a panic on this path; a backend decline is an
    // `Err`. Both mean "does not lower", and the distinction between them is a
    // separate concern from coverage.
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        if slang {
            try_generate(&op, &key, &Slang).is_ok()
        } else {
            try_generate(&op, &key, &CpuC).is_ok()
        }
    }))
    .unwrap_or(false)
}

#[test]
fn the_coverage_table_matches_what_the_backends_actually_do() {
    let mut wrong = Vec::new();
    let (mut c_n, mut s_n) = (0, 0);
    for &(name, dt, want_c, want_s) in COVERAGE {
        let (got_c, got_s) = (lowers(dt, false), lowers(dt, true));
        if got_c {
            c_n += 1;
        }
        if got_s {
            s_n += 1;
        }
        if got_c != want_c {
            wrong.push(format!(
                "{name}: CpuC table says {want_c}, actually {got_c}"
            ));
        }
        if got_s != want_s {
            wrong.push(format!(
                "{name}: Slang table says {want_s}, actually {got_s}"
            ));
        }
    }
    println!("\nmeasured lowering coverage — CpuC {c_n}/22, Slang {s_n}/22");
    for w in &wrong {
        println!("   {w}");
    }
    assert!(
        wrong.is_empty(),
        "{} entr(ies) stale. If you ADDED a dtype this is the reminder to \
         update the table; if you removed one, say why here.",
        wrong.len()
    );
}

/// The scalar-schedule control.
///
/// Every `false` above is only meaningful if the probe reaches the dtype gate at
/// all. A shape that elects a vectorized schedule declines for *every* dtype and
/// would make this whole file read as "nothing lowers" while asserting nothing —
/// the vacuous-green shape. `f32` lowering proves the probe gets through.
#[test]
fn the_probe_reaches_the_dtype_gate() {
    assert!(
        lowers(ElementKind::F32, false),
        "control: f32 must lower on CpuC, or the shape is selecting a schedule \
         no backend supports and every `false` above is measuring the wrong thing"
    );
    assert!(
        lowers(ElementKind::F32, true),
        "control: f32 must lower on Slang"
    );
}

/// The two RESERVED dtypes must never lower, at any schema version.
///
/// Separate from the table because it is a different kind of fact: the table
/// records what is *not done yet* and is expected to fill in over time, whereas
/// these are forbidden by KISS-CLASSIFY-6.1-0001 and filling them in would be a
/// conformance violation. A `false` that must stay `false` does not belong in a
/// column of `false`s that are all invitations.
#[test]
fn the_reserved_dtypes_never_lower() {
    for (name, dt) in [
        ("f8e4m3fnuz", ElementKind::Fp8E4M3FNUZ),
        ("f8e5m2fnuz", ElementKind::Fp8E5M2FNUZ),
    ] {
        assert!(dt.is_reserved(), "{name} must be marked reserved");
        assert!(
            !lowers(dt, false) && !lowers(dt, true),
            "{name} is RESERVED — no computation semantics at this schema version. \
             Lowering it is a conformance violation, not progress."
        );
    }
}

/// Coverage is a strict subset of recognition, and naming the gap is the point.
///
/// If these ever converge, this crate lowers everything it can name and the
/// assertion should be inverted rather than deleted.
#[test]
fn recognition_exceeds_lowering_and_the_gap_is_named() {
    let recognized = 24; // the closed §6.1 set, verified in unpopped-vocab
    let usable = 22; // recognized minus the two reserved
    assert_eq!(
        COVERAGE.len(),
        usable,
        "the table covers every usable dtype"
    );

    let c = COVERAGE.iter().filter(|e| e.2).count();
    let s = COVERAGE.iter().filter(|e| e.3).count();
    assert!(
        c < usable && s < usable,
        "lowering now covers every usable dtype — good news; invert this test"
    );
    println!(
        "\nrecognized {recognized} · usable {usable} · CpuC lowers {c} · Slang lowers {s}\n\
         not lowered by CpuC: {:?}",
        COVERAGE
            .iter()
            .filter(|e| !e.2)
            .map(|e| e.0)
            .collect::<Vec<_>>()
    );
}
