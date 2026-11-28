// SPDX-License-Identifier: MIT OR Apache-2.0
//! `unpopped-cpu-c` honours no schedule field (ROADMAP S2): a kernel that sets one is declined by the field's name, never run with
//! the field silently ignored. A kernel with no schedule, or a schedule that sets nothing, lowers.

use unpopped::backend::Backend;
use unpopped::exec::{ExecId, LaunchShape};
use unpopped::ir::ScalarExpr;
use unpopped::kernel::{Block, BufferRole, KernelFn, LocalDecl, Module, Param, Place, Stmt};
use unpopped::schedule::{
    EpilogueSchedule, ScheduleSpec, StageCount, TileDims, TileHierarchy, WorkDivision,
    schedule_fixtures,
};
use unpopped_cpu_c::{CpuC, lower_module};
use unpopped_vocab::ElementKind::F32;

fn copy() -> KernelFn {
    let at = |buffer| Place::Logical {
        buffer,
        coords: vec![ScalarExpr::Exec(ExecId::ThreadIdx(0))],
    };
    KernelFn::new(
        "copy",
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
                value: ScalarExpr::Local(0),
            },
        ]),
    )
    .with_launch(LaunchShape::new([8, 1, 1], [1, 1, 1]))
}

#[test]
fn every_set_field_is_declined_by_name_and_an_empty_schedule_lowers() {
    assert!(CpuC.lower_kernel(&copy()).is_ok(), "no schedule");
    assert!(
        CpuC.lower_kernel(&copy().with_schedule(ScheduleSpec::default()))
            .is_ok(),
        "a schedule that sets nothing"
    );
    let (_, full, _) = schedule_fixtures().remove(0);
    let fields = full.set_fields();
    assert!(
        fields.len() >= 5,
        "the fixture sets many fields: {fields:?}"
    );
    let err = CpuC
        .lower_kernel(&copy().with_schedule(full))
        .expect_err("a set field is declined");
    assert!(
        format!("{err:?}").contains(fields[0].name()),
        "{err:?} should name `{}`",
        fields[0].name()
    );
}

#[test]
fn each_field_alone_is_declined_under_its_own_name() {
    let tile = |warp, instr| TileHierarchy {
        cta: TileDims::new(128, 128, 64),
        warp,
        instr,
    };
    let one: Vec<ScheduleSpec> = vec![
        ScheduleSpec {
            tile: Some(tile(None, None)),
            ..ScheduleSpec::default()
        },
        ScheduleSpec {
            tile: Some(tile(Some(TileDims::new(64, 64, 64)), None)),
            ..ScheduleSpec::default()
        },
        ScheduleSpec {
            stages: StageCount::Fixed(2),
            ..ScheduleSpec::default()
        },
        ScheduleSpec {
            smem_carveout_bytes: Some(1024),
            ..ScheduleSpec::default()
        },
        ScheduleSpec {
            work: WorkDivision::SplitK { slices: 2 },
            ..ScheduleSpec::default()
        },
        ScheduleSpec {
            epilogue: Some(EpilogueSchedule {
                tile: Some([8, 8]),
                subtile: None,
            }),
            ..ScheduleSpec::default()
        },
        ScheduleSpec {
            cluster: Some([2, 1, 1]),
            ..ScheduleSpec::default()
        },
    ];
    for spec in one {
        let name = spec.set_fields()[0].name();
        let err = CpuC
            .lower_kernel(&copy().with_schedule(spec))
            .expect_err(name);
        assert!(
            format!("{err:?}").contains(name),
            "{err:?} should name `{name}`"
        );
    }
}

#[test]
fn a_module_function_that_sets_a_field_is_declined_too() {
    let (_, full, _) = schedule_fixtures().remove(0);
    let name = full.set_fields()[0].name();
    assert!(
        lower_module(&Module::new(vec![copy()]), 0).is_ok(),
        "the control"
    );
    let err = lower_module(&Module::new(vec![copy().with_schedule(full)]), 0)
        .expect_err("a module function with a schedule");
    assert!(format!("{err:?}").contains(name), "{err:?}");
}

/// An entry that calls a callee, the callee carrying `schedule` (and the entry none).
fn module_with_scheduled_callee(schedule: Option<ScheduleSpec>) -> Module {
    let entry = KernelFn::new(
        "entry",
        F32,
        vec![],
        vec![],
        Block(vec![Stmt::Call {
            callee: 1,
            args: vec![],
            result: None,
        }]),
    )
    .with_launch(LaunchShape::new([8, 1, 1], [1, 1, 1]));
    let mut callee = KernelFn::new("callee", F32, vec![], vec![], Block(vec![]));
    if let Some(s) = schedule {
        callee = callee.with_schedule(s);
    }
    Module::new(vec![entry, callee])
}

#[test]
fn a_called_function_that_sets_a_field_is_declined_not_lowered_silently() {
    // Pip's review of #498: the decline must reach every function of the module, not only the entry
    let (_, full, _) = schedule_fixtures().remove(0);
    let name = full.set_fields()[0].name();
    assert!(
        lower_module(&module_with_scheduled_callee(None), 0).is_ok(),
        "the control: the same call graph with no schedule lowers"
    );
    let err = lower_module(&module_with_scheduled_callee(Some(full)), 0)
        .expect_err("a scheduled callee is declined");
    assert!(
        format!("{err:?}").contains(name),
        "{err:?} should name `{name}`"
    );
}
