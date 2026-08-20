//! **Census, not a fuzzer.** How many `(OpDef, structure_key)` pairs make
//! `try_generate` panic, and at which lines.
//!
//! KISS-EMIT-6.8-0004 requires an emitter to return a typed decline rather than
//! panic on **any** input, and §6.1-0001 fixes that input as exactly
//! `(OpDef, structure_key)`. `crates/unpopped/tests/plan_gate_panics_on_malformed_input.rs`
//! pins that this crate does not comply. This file measures **how far** it does
//! not, so the fix can be scoped from a number instead of an estimate.
//!
//! # Why a census before a refactor
//!
//! `plan.rs` holds ~158 panic-family sites in production code (135 `assert!`,
//! 17 `panic!`, 3 `assert_eq!`, 3 `unreachable!`). Converting all of them is the
//! wrong instinct, and it is the instinct this crate's own boundary ruling
//! argues against: an assertion that **no input can reach** is an internal
//! invariant, and internal invariants are supposed to abort — they fire on a bug
//! in this crate, never on a caller's data. Only the input-reachable ones are
//! §6.8-0004 violations.
//!
//! Those two populations are not separable by reading. They are separable by
//! feeding the emitter inputs and seeing which sites fire. That is this file.
//!
//! # What it asserts, and why each assertion is here
//!
//! - **A ratchet on the count.** It MUST NOT rise. A new admissibility `assert!`
//!   on the plan path is a new violation, and it should cost whoever adds it a
//!   red build rather than going unnoticed among 158.
//! - **A floor on the corpus.** If the corpus stops producing panics the census
//!   is either fixed or broken, and those must not look alike — so the count
//!   going to zero fails too, loudly, and whoever fixed it edits this file
//!   deliberately.
//! - **A floor on successes.** A corpus where nothing lowers is measuring
//!   nothing: every input would be rejected for being nonsense rather than for
//!   the property under test.
//!
//! # What it is not
//!
//! Not adversarial in the §6.8-0004 sense — these are *structured* inputs, an
//! op paired with a cell that does not fit it. Truncated tokens, hostile
//! `OperandDesc` values, and rank/operand-count extremes are a separate axis and
//! are not covered here. **The number below is a lower bound on the gap**, and
//! naming it a census rather than a fuzz run is the point: a name that claimed
//! adversarial coverage while sampling a lattice is the defect this repository
//! keeps finding.

use std::panic::{self, AssertUnwindSafe};
use std::sync::Mutex;

use unpopped::backend::{Backend, GeneratedKernel, LowerError};
use unpopped::ir::{Expr, OpDef, ReduceOp, ReduceStage, coord, input, reduced};
use unpopped::plan::KernelPlan;
use unpopped::try_generate;
use unpopped_vocab::{
    ArchSku, ElementKind, OpCategory, OperandDesc, StructureKey, TargetId, structure_key,
};

/// Measured 2026-08-20 over the corpus below. **The pair moves together.**
///
/// 1280 structured inputs → **640 lowered, 640 panicked, 0 declined**, across
/// six distinct sites in `plan.rs`.
///
/// Half the corpus panics. The zero is the sharper number: with a backend that
/// declines nothing, `try_generate` produced **not one typed decline** across
/// 1280 inputs. The admissibility layer has no decline channel at all — it
/// passes or it aborts. Every decline this crate can currently produce comes
/// from a *backend*, downstream of the gate that panicked.
///
/// Fixing §6.8-0004 moves mass from `KNOWN_PANICKING` to `KNOWN_DECLINED`, one
/// site at a time. Both are pinned so that movement must be recorded here, in
/// the same commit, by whoever earns it.
const KNOWN_PANICKING: usize = 640;

/// See [`KNOWN_PANICKING`]. Zero today; every point it rises is a panic that
/// became a typed decline.
const KNOWN_DECLINED: usize = 0;

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

const DTYPES: &[ElementKind] = &[
    ElementKind::F32,
    ElementKind::F64,
    ElementKind::F16,
    ElementKind::Bf16,
    ElementKind::I32,
    ElementKind::I64,
    ElementKind::I8,
    ElementKind::U8,
    ElementKind::I16,
    ElementKind::U16,
    ElementKind::U32,
    ElementKind::U64,
    ElementKind::Bool,
    ElementKind::I4,
    ElementKind::U4,
    ElementKind::B1,
    ElementKind::Fp8E4M3FN,
    ElementKind::Fp8E5M2,
    ElementKind::Complex64,
    ElementKind::Complex128,
];

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

    for &dt in DTYPES {
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
        c.panicked > 0,
        "the census found no panics. Either KISS-EMIT 6.8-0004 is now satisfied \
         on this corpus — in which case delete this assertion deliberately and \
         retire the gap pin — or the corpus stopped reaching the plan gate. \
         Those must not look alike, so this fails rather than passing quietly"
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
