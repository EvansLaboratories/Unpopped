// SPDX-License-Identifier: MIT OR Apache-2.0
//! The schedule carrier on a kernel (ROADMAP S2, `docs/design-tile-and-schedule-plan.md` section 3): `KernelFn::schedule`.
//! `check_kernel` checks the spec with `check_schedule`, the oracle ignores it (a schedule changes how fast, never what a
//! deterministic kernel computes), and a document without the field loads with `None`.

use unpopped::exec::{ExecId, LaunchShape};
use unpopped::ir::ScalarExpr;
use unpopped::kernel::{
    Block, BufferRole, KernelError, KernelFn, LocalDecl, Param, Place, Stmt, check_kernel,
};
use unpopped::oracle::TypedBuffer;
use unpopped::oracle::kernel_eval::KernelArg;
use unpopped::oracle::kernel_launch::run_launch;
use unpopped::schedule::{ScheduleSpec, schedule_fixtures};
use unpopped_vocab::ElementKind::F32;

/// `out[t] = in[t] * 2` over 8 threads.
fn doubling() -> KernelFn {
    let at = |buffer| Place::Logical {
        buffer,
        coords: vec![ScalarExpr::Exec(ExecId::ThreadIdx(0))],
    };
    KernelFn::new(
        "double",
        F32,
        vec![
            Param::buffer("in", F32, BufferRole::Read, Some(1)),
            Param::buffer("out", F32, BufferRole::Write, Some(1)),
        ],
        vec![LocalDecl::scalar(F32)],
        Block(vec![
            Stmt::Load {
                local: 0,
                place: at(0),
            },
            Stmt::Store {
                place: at(1),
                value: ScalarExpr::Mul(
                    Box::new(ScalarExpr::Local(0)),
                    Box::new(ScalarExpr::Const(2.0)),
                ),
            },
        ]),
    )
    .with_launch(LaunchShape::new([8, 1, 1], [1, 1, 1]))
}

fn run(k: &KernelFn) -> Vec<f64> {
    let input: Vec<f32> = (0..8).map(|i| i as f32 + 0.5).collect();
    let mut args = vec![
        KernelArg::Buffer(TypedBuffer::from_f32(&[8], &input)),
        KernelArg::Buffer(TypedBuffer::from_f32(&[8], &[0.0; 8])),
    ];
    run_launch(k, &mut args).expect("runs");
    let KernelArg::Buffer(o) = &args[1] else {
        unreachable!()
    };
    o.to_f64_vec()
}

#[test]
fn check_kernel_gives_the_verdict_check_schedule_gives_for_every_fixture() {
    assert_eq!(
        check_kernel(&doubling()),
        Ok(()),
        "the control: no schedule"
    );
    for (name, spec, want) in schedule_fixtures() {
        let got = check_kernel(&doubling().with_schedule(spec));
        match want {
            Ok(()) => assert_eq!(got, Ok(()), "{name}"),
            Err(e) => assert_eq!(got, Err(KernelError::Schedule(e)), "{name}"),
        }
    }
}

#[test]
fn the_oracle_ignores_a_schedule() {
    let plain = run(&doubling());
    let (_, full, _) = schedule_fixtures().remove(0);
    assert!(!full.set_fields().is_empty(), "the fixture sets fields");
    assert_eq!(run(&doubling().with_schedule(full)), plain);
    assert_eq!(plain[3], 7.0);
}

#[test]
fn the_field_round_trips_and_a_document_without_it_loads_with_none() {
    let (_, full, _) = schedule_fixtures().remove(0);
    let k = doubling().with_schedule(full.clone());
    let back: KernelFn = serde_json::from_str(&serde_json::to_string(&k).unwrap()).unwrap();
    assert_eq!(back.schedule.as_deref(), Some(&full));
    let plain = serde_json::to_value(doubling()).unwrap();
    assert!(plain.get("schedule").is_none(), "{plain}");
    let loaded: KernelFn = serde_json::from_value(plain).unwrap();
    assert_eq!(loaded.schedule, None);
    assert_eq!(ScheduleSpec::default().set_fields(), vec![]);
}

#[test]
fn a_partial_schedule_loads_with_the_unstated_fields_at_their_defaults() {
    // Pip's review of #498: a spec that states only the fields it sets used to fail to load
    let only_cluster: ScheduleSpec = serde_json::from_str(r#"{"cluster":[2,1,1]}"#).unwrap();
    assert_eq!(
        only_cluster,
        ScheduleSpec {
            cluster: Some([2, 1, 1]),
            ..ScheduleSpec::default()
        }
    );
    assert_eq!(
        serde_json::from_str::<ScheduleSpec>("{}").unwrap(),
        ScheduleSpec::default()
    );
    // a stated field still wins, and a wrong type is still a load error (the control: not everything loads)
    let staged: ScheduleSpec = serde_json::from_str(r#"{"stages":{"Fixed":3}}"#).unwrap();
    assert_eq!(staged.stages, unpopped::schedule::StageCount::Fixed(3));
    assert!(serde_json::from_str::<ScheduleSpec>(r#"{"cluster":"two"}"#).is_err());
}
