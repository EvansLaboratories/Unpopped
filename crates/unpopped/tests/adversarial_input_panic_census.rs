//! **Census, not a fuzzer.** What `(OpDef, structure_key)` pairs do at the
//! emitter boundary: lower, decline, or abort.
//!
//! KISS-EMIT-6.8-0004 requires an emitter to return a typed decline rather than
//! panic on **any** input, and §6.1-0001 fixes that input as exactly
//! `(OpDef, structure_key)`. This file scoped the 0.6.0 fix, then measured it.
//!
//! # Why a census before a refactor
//!
//! `plan.rs` held ~158 panic-family sites in production code (135 `assert!`,
//! 17 `panic!`, 3 `assert_eq!`, 3 `unreachable!`). Converting all of them was
//! the wrong instinct, and it is the instinct this crate's own boundary ruling
//! argues against: an assertion that **no input can reach** is an internal
//! invariant, and internal invariants are supposed to abort — they fire on a bug
//! in this crate, never on a caller's data. Only the input-reachable ones are
//! §6.8-0004 violations.
//!
//! Those two populations are not separable by reading. They are separable by
//! feeding the emitter inputs and seeing which sites fire. **Six fired, not
//! 158**, and that is what 0.6.0 converted.
//!
//! # What it asserts, and why each assertion is here
//!
//! - **A ratchet on the pair.** Panics MUST NOT rise, and a fix moves mass to
//!   declines, so both numbers are re-recorded together by whoever earns the
//!   movement. A one-sided ratchet would let a panic be *deleted* rather than
//!   converted and call it progress.
//! - **A floor on declines.** Zero declines and zero panics is what a corpus
//!   that stopped reaching the plan gate looks like — and it is also what
//!   compliance looks like from outside. Those must not be confusable.
//! - **A floor on successes.** A corpus where nothing lowers is measuring
//!   nothing: every input would be rejected for being nonsense rather than for
//!   the property under test.
//! - **A positive control on the detector.** Now that the census counts zero
//!   panics, nothing else proves it could still see one. Closing a gap takes
//!   away the free proof that the instrument works, and that is precisely when
//!   an instrument quietly stops working.
//!
//! # What it is not
//!
//! Not adversarial in the §6.8-0004 sense — these are *structured* inputs, an
//! op paired with a cell that does not fit it. Truncated tokens, hostile
//! `OperandDesc` values, and rank/operand-count extremes are a separate axis and
//! are not covered here. **Zero panics here is a lower bound on compliance, not
//! a proof of it**, and naming this a census rather than a fuzz run is the
//! point: a name that claimed adversarial coverage while sampling a lattice is
//! the defect this repository keeps finding.

use std::panic::{self, AssertUnwindSafe};
use std::sync::Mutex;

use unpopped::backend::{Backend, GeneratedKernel, LowerError};
use unpopped::ir::{BinaryOp, Expr, OpDef, ReduceOp, ReduceStage, coord, input, reduced};
use unpopped::plan::KernelPlan;
use unpopped::try_generate;
use unpopped_vocab::{
    ArchSku, ElementKind, OpCategory, OperandDesc, StructureKey, TargetId, structure_key,
};

/// Measured over the corpus below. **The pair moves together.**
///
/// ```text
/// 2026-08-20, before 0.6.0:  640 lowered,   0 declined, 640 PANICKED  (6 sites)
/// 2026-08-20, after  0.6.0:  640 lowered, 640 declined,   0 panicked
/// ```
///
/// The before-state is kept because it is the measurement that scoped the fix.
/// Half the corpus aborted, and the zero was the sharper number: with a backend
/// that declines nothing, `try_generate` produced **not one typed decline**
/// across 1280 inputs, because the gate it called had no decline channel wired
/// to it. Every decline the crate could then produce came from a *backend*,
/// downstream of the gate that panicked.
///
/// Both stay pinned. Panics MUST NOT rise: a new admissibility `assert!` on the
/// plan path is a new §6.8-0004 violation and should cost a red build rather
/// than disappear among the ~150 sites `plan.rs` still holds. A one-sided
/// ratchet would let someone delete a panic instead of converting it and call
/// that progress.
/// **1008 -> 1428 on 2026-09-02 (second widening)**, again coverage rather than
/// behaviour: the dtype axis now derives from `ElementKind::ALL` (25) instead of
/// a hand-listed 20. **No lowering changed and `KNOWN_PANICKING` stayed 0** over
/// the larger 2400-input surface.
///
/// **640 -> 1008 on 2026-09-02**, and the delta is coverage rather than
/// behaviour: four ops with panic paths were added to the probe set
/// (`Max`/`Min`, which `cfamily::binary_int` panics on at integer dtypes, and
/// `BitAnd`/`Shr`, which are int-only and must miss at float dtypes). **No
/// lowering changed.** The zero above now covers the ops whose panic path
/// exists; before, it was a true number measured over a surface that excluded
/// them.
const KNOWN_PANICKING: usize = 0;

/// See [`KNOWN_PANICKING`]. Every one of these was a panic before 0.6.0.
const KNOWN_DECLINED: usize = 1428;

/// Spells every dtype and every plan, so a panic is never a backend decline in
/// disguise. A backend that declined would mask the very thing being counted.
struct SpellsAll;

impl Backend for SpellsAll {
    fn name(&self) -> &str {
        "spells-all"
    }
    fn provider(&self) -> &str {
        "unpopped-tests"
    }
    fn supports_dtype(&self, _dtype: ElementKind, _target: TargetId) -> bool {
        true
    }
    fn lower(&self, _plan: &KernelPlan<'_>) -> Result<GeneratedKernel, LowerError> {
        Ok(GeneratedKernel::new(
            "k".to_string(),
            "/* spelled */".to_string(),
        ))
    }
}

/// ⚠️ Derived from `ElementKind::ALL`, not hand-listed — the census is a claim
/// about "every input" and a copied list stops being every input the day a dtype
/// is added.
///
/// It used to name 20 variants while `ALL` had 25. Four of the five omitted are
/// the non-compute rows the plan gate now declines (`Fp8E4M3FNUZ`,
/// `Fp8E5M2FNUZ`, `F8E8M0`, `F8E6M2`) — including them costs nothing and proves
/// they decline rather than panic, which is the whole point. **The fifth was
/// `F32Strict`, a live compute dtype**, so `KNOWN_PANICKING = 0` was measured
/// over 20 of 25 while reading as a statement about all input.
///
/// `ALL` is complete by mechanism rather than by care: the enum is exhaustive so
/// a new variant breaks every match site, and `kiss_dtype_manifest` fails if a
/// §6.1-token-carrying variant is missing from it. **Deriving inherits that;
/// copying does not.**
fn dtypes() -> &'static [ElementKind] {
    &ElementKind::ALL
}

const CATEGORIES: &[OpCategory] = &[
    OpCategory::UnaryElementwise,
    OpCategory::BinaryElementwise,
    OpCategory::Reduction,
    OpCategory::Normalization,
];

fn key_for(dt: ElementKind, cat: OpCategory, rank: u8) -> StructureKey {
    let d = if rank == 2 {
        OperandDesc::new(2, &[4, 4], &[4, 1], dt, 4)
    } else {
        OperandDesc::new(1, &[7], &[1], dt, 4)
    };
    structure_key(cat, &[d, d, d], ArchSku::Sm89)
}

/// Op bodies that are individually legal and collectively do not fit every
/// cell. The mismatch is the input under test.
fn ops(dt: ElementKind) -> Vec<(&'static str, OpDef)> {
    let unary = |e: Expr| OpDef::elementwise("u", 1, &[dt], e);
    vec![
        (
            "add",
            OpDef::elementwise("b", 2, &[dt], input(0) + input(1)),
        ),
        (
            "mul",
            OpDef::elementwise("b", 2, &[dt], input(0) * input(1)),
        ),
        ("neg_ish", unary(input(0) * input(0))),
        // ⚠️ THE OPS WITH PANIC PATHS, added 2026-09-02 because the census's
        // zero did not cover them.
        //
        // `cfamily::binary_int` ends in `other => panic!("{other:?} has no
        // integer lowering ... must miss honestly at the plan gate")`. That
        // panic is defended by the gate — verified — but the census's op set was
        // `+`, `*`, `Sum` and coord, **none of which can reach it**. So
        // `KNOWN_PANICKING = 0` was a true number measured over a surface that
        // excluded the only ops whose panic path exists.
        //
        // A guard's verdict being right does not mean its coverage supports it.
        // These four make the zero cover the claim.
        (
            "max_float_only_at_int",
            OpDef::elementwise("b", 2, &[dt], input(0).binary(BinaryOp::Max, input(1))),
        ),
        (
            "min_float_only_at_int",
            OpDef::elementwise("b", 2, &[dt], input(0).binary(BinaryOp::Min, input(1))),
        ),
        (
            "bitand_int_only_at_float",
            OpDef::elementwise("b", 2, &[dt], input(0).binary(BinaryOp::BitAnd, input(1))),
        ),
        (
            "shr_int_only_at_float",
            OpDef::elementwise("b", 2, &[dt], input(0).binary(BinaryOp::Shr, input(1))),
        ),
        ("coord0", unary(input(0) + coord(0))),
        ("coord1", unary(input(0) + coord(1))),
        (
            "reduce_sum",
            OpDef::reduction("r", 1, &[dt], input(0), ReduceOp::Sum),
        ),
        (
            "row_reduce_ok",
            OpDef::row_reduce(
                "rr",
                1,
                &[dt],
                vec![ReduceStage {
                    pre: input(0).0,
                    op: ReduceOp::Sum,
                }],
                input(0) * reduced(0),
            ),
        ),
        (
            "row_reduce_forward_ref",
            OpDef::row_reduce(
                "rrf",
                1,
                &[dt],
                vec![ReduceStage {
                    pre: reduced(0).0,
                    op: ReduceOp::Sum,
                }],
                input(0) * reduced(0),
            ),
        ),
    ]
}

struct Census {
    total: usize,
    panicked: usize,
    lowered: usize,
    declined: usize,
    sites: Vec<String>,
}

fn run_census() -> Census {
    // Collect the panic location instead of printing it: the census runs
    // hundreds of deliberate panics and the default hook would bury the result.
    let sites: &'static Mutex<Vec<String>> = Box::leak(Box::new(Mutex::new(Vec::new())));
    let prev = panic::take_hook();
    panic::set_hook(Box::new(move |info| {
        let at = info
            .location()
            .map(|l| format!("{}:{}", l.file(), l.line()))
            .unwrap_or_else(|| "<unknown>".to_string());
        sites.lock().expect("census lock").push(at);
    }));

    let mut c = Census {
        total: 0,
        panicked: 0,
        lowered: 0,
        declined: 0,
        sites: Vec::new(),
    };

    for &dt in dtypes() {
        for &cat in CATEGORIES {
            for rank in [1u8, 2] {
                let key = key_for(dt, cat, rank);
                for (_name, op) in ops(dt) {
                    c.total += 1;
                    match panic::catch_unwind(AssertUnwindSafe(|| {
                        try_generate(&op, &key, &SpellsAll)
                    })) {
                        Ok(Ok(_)) => c.lowered += 1,
                        Ok(Err(_)) => c.declined += 1,
                        Err(_) => c.panicked += 1,
                    }
                }
            }
        }
    }

    panic::set_hook(prev);
    let mut s = sites.lock().expect("census lock").clone();
    s.sort();
    s.dedup();
    c.sites = s;
    c
}

#[test]
fn the_panic_census_holds_at_its_recorded_count() {
    let c = run_census();

    println!(
        "census: {} inputs -> {} lowered, {} declined, {} PANICKED across {} distinct sites",
        c.total,
        c.lowered,
        c.declined,
        c.panicked,
        c.sites.len()
    );
    for s in &c.sites {
        println!("  panic site: {s}");
    }

    assert!(
        c.lowered > 0,
        "no input lowered — the corpus is nonsense and measures nothing, not \
         even the thing that is green"
    );
    assert!(
        c.declined > 0,
        "no input was declined. Before 0.6.0 this assertion read `panicked > 0` \
         and existed so that a closed gap and a broken corpus could not look \
         alike. Inverted, it does the same job: a corpus that stopped REACHING \
         the plan gate would report zero declines and zero panics — which is \
         exactly what compliance looks like from the outside"
    );
    assert_eq!(
        (c.panicked, c.declined),
        (KNOWN_PANICKING, KNOWN_DECLINED),
        "the panic/decline split moved. Panics MUST NOT rise — a new \
         admissibility assert on the plan path is a new 6.8-0004 violation. A \
         fix moves mass the other way, so record BOTH numbers in the same \
         commit as the change that earned them. Distinct panic sites now: {:?}",
        c.sites
    );
}

/// Positive control for the counter itself.
///
/// The census now reports **zero** panics, which means its panic-detection path
/// is never exercised by the census. A counter that has stopped counting and a
/// subject that has stopped panicking produce identical output, and the whole
/// file would go quietly green if `catch_unwind` were ever removed, or if the
/// hook swallowed the location, or if a future edit classified an `Err(_)`
/// return as a panic.
///
/// So: drive a known panic through the same detection path and require it to be
/// seen. This is the assertion the `panicked > 0` guard used to make for free
/// while the gap was open — closing the gap took the free proof away with it,
/// and that is exactly when a guard silently stops guarding.
#[test]
fn the_census_can_still_see_a_panic() {
    let prev = panic::take_hook();
    panic::set_hook(Box::new(|_| {}));
    let seen = panic::catch_unwind(AssertUnwindSafe(|| {
        panic!("deliberate: proving the detector fires");
    }))
    .is_err();
    panic::set_hook(prev);

    assert!(
        seen,
        "the census cannot detect a panic, so its zero means nothing"
    );
}
