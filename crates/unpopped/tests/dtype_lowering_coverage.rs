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

/// What a backend does with a dtype, and **why**.
///
/// The distinctions among the non-lowering states are the point of this enum. A
/// single `false` column puts "nobody has written this yet", "this needs
/// something else first", and "this is decided and correct" in one cell — and
/// they call for completely different responses. Collapsed, the worklist
/// silently grows entries that will never be done, settled decisions get
/// re-litigated by whoever reads the table as a TODO list, and work that is
/// merely *ordered* looks impossible.
///
/// The `Blocked` variant exists because of a mistake this file made: Slang's
/// narrow integers were labelled `ByDesign` — permanently declined — on a
/// misreading of "depends on target + capabilities" as "cannot". They are
/// ordinary work behind one missing mechanism.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
enum Status {
    /// Lowers today.
    Lowers,
    /// Not implemented yet — a genuine worklist entry, nothing in the way.
    NotYet,
    /// Worklist, but a **prerequisite** has to land first. Still work, and the
    /// named blocker is what makes it schedulable in the right order.
    Blocked(&'static str),
    /// **Deliberately** declined, permanently, for the recorded reason. A
    /// typed decline here is conformant behaviour, not a gap.
    ByDesign(&'static str),
}
use Status::{Blocked, ByDesign, Lowers, NotYet};

/// Slang's own conformance docs: *"Only `int`/`int32_t` and `uint`/`uint32_t`
/// are universally supported; the others depend on target + capabilities."*
///
/// **Read carefully, that says Slang CAN spell these — on a target whose
/// capabilities allow it.** So this is not a limitation of Slang, and labelling
/// it one (as an earlier version of this file did) blames the target for a gap
/// that is ours.
///
/// The gap is that [`Backend::supports_dtype`] takes only a dtype:
/// `fn supports_dtype(&self, dtype: ElementKind) -> bool`. It has no target
/// parameter, so a backend can answer "always" or "never" and nothing in
/// between. Faced with a conditionally-available type the only *sound* answer is
/// "never" — claiming support unconditionally would emit `int8_t` for a target
/// that cannot compile it, which is the "decline, do not fall through" rule the
/// whole backend contract is built on.
///
/// The information is already there: `KernelPlan.key` carries the
/// `StructureKey`, and KISS's §6.8 target tokens encode capabilities directly —
/// `vulkan:sg64.ops-abr.arith-f16.cm-none` literally names an `arith-f16`
/// capability. The admissibility gate just cannot see it.
///
/// **Same root cause as the one vulkan vector excluded from the byte-match**:
/// `ArchSku` is a closed CUDA-only enum that cannot represent a `vulkan:` target
/// at all. Capability-aware dtype admission and a pluggable target namespace are
/// one piece of work, not two.
const SLANG_NARROW: &str = "Slang supports these on capable targets; supports_dtype has      no target parameter, so the only sound unconditional answer is no";

/// `U32` is this generator's index/address dtype — the gather/scatter index
/// operand pointer type — never a compute operand. No constructor builds a
/// compute cell keyed `U32`, so declining it as a compute dtype is the design,
/// not an omission. (Slang universally supports `uint32_t`; the decline here is
/// ours, not the target's.)
const U32_INDEX: &str = "U32 is the index/address dtype, never a compute operand";

/// Every §6.1 dtype, with what each backend does with a plain elementwise `Add`.
///
/// The two RESERVED dtypes are absent entirely rather than listed as declines:
/// `f8e4m3fnuz` / `f8e5m2fnuz` have no computation semantics at this schema
/// version and lowering them would be a conformance violation. That is a third
/// kind of fact again, and it is asserted separately below.
const COVERAGE: &[(&str, ElementKind, Status, Status)] = &[
    // name          dtype                      CpuC   Slang
    ("f16", ElementKind::F16, NotYet, NotYet),
    ("bf16", ElementKind::Bf16, NotYet, NotYet),
    ("f32", ElementKind::F32, Lowers, Lowers),
    ("f64", ElementKind::F64, Lowers, Lowers),
    ("i8", ElementKind::I8, Lowers, Blocked(SLANG_NARROW)),
    ("i16", ElementKind::I16, Lowers, Blocked(SLANG_NARROW)),
    ("u8", ElementKind::U8, Lowers, Blocked(SLANG_NARROW)),
    ("u16", ElementKind::U16, Lowers, Blocked(SLANG_NARROW)),
    ("i32", ElementKind::I32, Lowers, Lowers),
    ("i64", ElementKind::I64, Lowers, Lowers),
    (
        "u32",
        ElementKind::U32,
        ByDesign(U32_INDEX),
        ByDesign(U32_INDEX),
    ),
    ("u64", ElementKind::U64, NotYet, NotYet),
    ("bool", ElementKind::Bool, NotYet, NotYet),
    ("f8e4m3fn", ElementKind::Fp8E4M3FN, NotYet, NotYet),
    ("f8e5m2", ElementKind::Fp8E5M2, NotYet, NotYet),
    ("f8e8m0", ElementKind::F8E8M0, NotYet, NotYet),
    ("f8e6m2", ElementKind::F8E6M2, NotYet, NotYet),
    ("i4", ElementKind::I4, NotYet, NotYet),
    ("u4", ElementKind::U4, NotYet, NotYet),
    ("b1", ElementKind::B1, NotYet, NotYet),
    ("c64", ElementKind::Complex64, NotYet, NotYet),
    ("c128", ElementKind::Complex128, NotYet, NotYet),
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
        for (backend, slang, want) in [("CpuC", false, want_c), ("Slang", true, want_s)] {
            let got = lowers(dt, slang);
            if got {
                if slang { s_n += 1 } else { c_n += 1 }
            }
            if got != (want == Lowers) {
                wrong.push(format!(
                    "{name}: {backend} table says {want:?}, actually lowers={got}"
                ));
            }
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

/// Every deliberate decline carries a reason, and the reasons are real.
///
/// `ByDesign` is the entry that can rot worst: it looks settled, so nobody
/// re-checks it, and an empty or vague justification is indistinguishable from
/// a `NotYet` somebody wanted off the worklist. Each one is a claim about a
/// target or about this generator's design, and it should read as one.
#[test]
fn every_deliberate_decline_states_why() {
    let mut n = 0;
    for &(name, _, c, s) in COVERAGE {
        for st in [c, s] {
            if let ByDesign(why) | Blocked(why) = st {
                // Substance, not format. An earlier version of this also
                // demanded a colon, which failed a perfectly good reason for
                // being punctuated differently — the assertion was encoding my
                // habits rather than the property.
                assert!(
                    why.len() > 20,
                    "{name}: a ByDesign decline needs a real reason, got {why:?}"
                );
                assert!(
                    !why.to_lowercase().contains("todo") && !why.to_lowercase().contains("later"),
                    "{name}: {why:?} describes deferral, not a decision — that is `NotYet`"
                );
                n += 1;
            }
        }
    }
    assert!(
        n > 0,
        "no deliberate declines — has the enum stopped being used?"
    );
    println!(
        "
{n} deliberate decline(s), each with a stated reason"
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

    let c = COVERAGE.iter().filter(|e| e.2 == Lowers).count();
    let s = COVERAGE.iter().filter(|e| e.3 == Lowers).count();
    assert!(
        c < usable && s < usable,
        "lowering now covers every usable dtype — good news; invert this test"
    );

    // The worklist is `NotYet` only. A `ByDesign` decline is a conclusion, and
    // counting it as outstanding work would keep it on the list forever.
    // `Blocked` is still work — it is scheduled behind a prerequisite, not
    // excluded. Only `ByDesign` leaves the worklist.
    let todo: Vec<&str> = COVERAGE
        .iter()
        .filter(|e| matches!(e.2, NotYet | Blocked(_)))
        .map(|e| e.0)
        .collect();
    let settled: Vec<&str> = COVERAGE
        .iter()
        .filter(|e| matches!(e.2, ByDesign(_)))
        .map(|e| e.0)
        .collect();

    println!(
        "\nrecognized {recognized} · usable {usable} · CpuC lowers {c} · Slang lowers {s}\n\
         CpuC worklist ({}): {todo:?}\n\
         CpuC settled declines ({}): {settled:?}",
        todo.len(),
        settled.len()
    );
}
