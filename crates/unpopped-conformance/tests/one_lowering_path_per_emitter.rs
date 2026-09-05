//! Each reference emitter dispatches exactly ONE schedule, and a second arm
//! expires a nil that was measured against the first.
//!
//! # The nil this protects
//!
//! `docs/normative-seams.md` records that neither reference emitter has the
//! multi-surface §6.16-0009 exposure baracuda found in theirs — they enumerated
//! their `Schedule::` dispatch and found FOUR defective lowering paths, three of
//! them discovered by tripping over them rather than by looking.
//!
//! That nil is true of **today's dispatch**, not of the crates. ⚠️ **A nil about a
//! dispatch decays silently the moment someone adds an arm**, and the person
//! adding it has no reason to connect a new `Schedule::Vectorized` arm to a
//! move/round-trip obligation recorded in a doc they are not reading.
//!
//! baracuda's prescription, taken verbatim: *"the expiry condition you wrote is
//! the part I would put in the code as a test, not in a message."*
//!
//! # Why the count and not the behaviour
//!
//! The behaviour is already covered — `narrow_float_moves_do_not_round` and
//! `a_sign_edit_agrees_with_the_oracle_on_every_fp8_byte` test the path that
//! exists. **Neither can notice a path that does not exist yet.** This asserts
//! the population those tests range over is still the whole population.

use std::path::Path;

/// Count `Schedule::<Name> =>` match arms, excluding the catch-all decline.
fn schedule_arms(src: &str) -> Vec<String> {
    src.lines()
        .filter_map(|l| {
            let t = l.trim();
            let rest = t.strip_prefix("Schedule::")?;
            let name = rest.split(&[' ', '=', '(', '{'][..]).next()?;
            t.contains("=>").then(|| name.to_string())
        })
        .collect()
}

#[test]
fn each_reference_emitter_serves_exactly_one_schedule() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("crates/");

    for krate in ["unpopped-cpu-c", "unpopped-slang"] {
        let path = root.join(krate).join("src/lib.rs");
        let src = std::fs::read_to_string(&path)
            .unwrap_or_else(|e| panic!("read {}: {e}", path.display()));

        let arms = schedule_arms(&src);

        // Control: the dispatch must be FOUND. A renamed enum or a moved match
        // would otherwise make this pass by measuring nothing.
        assert!(
            !arms.is_empty(),
            "{krate}: found no `Schedule::<X> =>` arms at all. Either the dispatch \
             moved out of src/lib.rs or the enum was renamed — this test measures \
             nothing until that is fixed, which is worse than a red"
        );

        assert_eq!(
            arms,
            vec!["Scalar".to_string()],
            "{krate} now dispatches {arms:?} rather than only Scalar.\n\n\
             ⚠️ THIS EXPIRES A RECORDED NIL. `docs/normative-seams.md` states that \
             neither reference emitter has the multi-surface KISS-OPS-6.16-0009 \
             exposure baracuda found in theirs (four defective lowering paths, \
             three found by tripping over them). That was measured against a \
             one-arm dispatch.\n\n\
             A new lowering path is a NEW SURFACE for the move/round-trip \
             obligation: ask whether it holds a running value in an accumulator \
             register whose type is chosen for arithmetic convenience rather than \
             for the storage type. That is the mechanism that predicts the defect; \
             \"is this op a move?\" is the obligation and does not.\n\n\
             Sweep the new path, then update the nil."
        );
    }
}
