// SPDX-License-Identifier: MIT OR Apache-2.0
//! The schedule vocabulary (board 500 Q2 A, plan phase S1): `check_schedule` refuses a spec that is not self-consistent, and each
//! refusal has its accepted control beside it; `set_fields` lists exactly what a spec sets, so an emitter can decline it by name.

use unpopped::schedule::{
    EpilogueSchedule, ScheduleError, ScheduleField, ScheduleSpec, StageCount, TileDims,
    TileHierarchy, WorkDivision, check_schedule, schedule_fixtures,
};

fn hierarchy(cta: TileDims, warp: Option<TileDims>, instr: Option<TileDims>) -> ScheduleSpec {
    ScheduleSpec {
        tile: Some(TileHierarchy { cta, warp, instr }),
        ..ScheduleSpec::default()
    }
}

#[test]
fn the_default_schedule_states_nothing_and_passes() {
    let s = ScheduleSpec::default();
    assert_eq!(check_schedule(&s), Ok(()));
    assert!(s.set_fields().is_empty());
}

#[test]
fn a_tile_level_must_divide_the_one_above_and_the_control_passes() {
    let cta = TileDims::new(128, 128, 64);
    let ok = hierarchy(
        cta,
        Some(TileDims::new(64, 64, 64)),
        Some(TileDims::new(16, 8, 16)),
    );
    assert_eq!(check_schedule(&ok), Ok(()));
    // a warp tile that does not divide the block tile
    let bad = hierarchy(cta, Some(TileDims::new(96, 64, 64)), None);
    assert_eq!(
        check_schedule(&bad),
        Err(ScheduleError::TileDoesNotNest(ScheduleField::WarpTile))
    );
    // an instruction tile that does not divide the warp tile (and with no warp tile, the block tile)
    let bad = hierarchy(
        cta,
        Some(TileDims::new(64, 64, 64)),
        Some(TileDims::new(48, 8, 16)),
    );
    assert_eq!(
        check_schedule(&bad),
        Err(ScheduleError::TileDoesNotNest(ScheduleField::InstrTile))
    );
    let bad = hierarchy(cta, None, Some(TileDims::new(48, 8, 16)));
    assert_eq!(
        check_schedule(&bad),
        Err(ScheduleError::TileDoesNotNest(ScheduleField::InstrTile))
    );
    assert_eq!(
        check_schedule(&hierarchy(cta, None, Some(TileDims::new(16, 8, 16)))),
        Ok(())
    );
}

#[test]
fn a_zero_extent_is_refused_at_each_level() {
    let cta = TileDims::new(128, 128, 64);
    assert_eq!(
        check_schedule(&hierarchy(TileDims::new(0, 128, 64), None, None)),
        Err(ScheduleError::ZeroExtent(ScheduleField::Tile))
    );
    assert_eq!(
        check_schedule(&hierarchy(cta, Some(TileDims::new(64, 0, 64)), None)),
        Err(ScheduleError::ZeroExtent(ScheduleField::WarpTile))
    );
    assert_eq!(
        check_schedule(&hierarchy(cta, None, Some(TileDims::new(16, 8, 0)))),
        Err(ScheduleError::ZeroExtent(ScheduleField::InstrTile))
    );
}

#[test]
fn counts_that_mean_nothing_are_refused_and_the_neighbouring_values_pass() {
    let with = |f: fn(&mut ScheduleSpec)| {
        let mut s = ScheduleSpec::default();
        f(&mut s);
        s
    };
    assert_eq!(
        check_schedule(&with(|s| s.stages = StageCount::Fixed(0))),
        Err(ScheduleError::NoStages)
    );
    assert_eq!(
        check_schedule(&with(|s| s.stages = StageCount::Fixed(1))),
        Ok(())
    );
    assert_eq!(
        check_schedule(&with(|s| s.smem_carveout_bytes = Some(0))),
        Err(ScheduleError::ZeroCarveout)
    );
    assert_eq!(
        check_schedule(&with(|s| s.smem_carveout_bytes = Some(1))),
        Ok(())
    );
    assert_eq!(
        check_schedule(&with(
            |s| s.work = WorkDivision::Persistent { blocks: Some(0) }
        )),
        Err(ScheduleError::NoPersistentBlocks)
    );
    assert_eq!(
        check_schedule(&with(|s| s.work = WorkDivision::Persistent { blocks: None })),
        Ok(())
    );
    assert_eq!(
        check_schedule(&with(|s| s.work = WorkDivision::SplitK { slices: 1 })),
        Err(ScheduleError::SplitKNeedsTwoSlices)
    );
    assert_eq!(
        check_schedule(&with(|s| s.work = WorkDivision::SplitK { slices: 2 })),
        Ok(())
    );
    assert_eq!(
        check_schedule(&with(|s| s.cluster = Some([2, 0, 1]))),
        Err(ScheduleError::ZeroCluster)
    );
    assert_eq!(
        check_schedule(&with(|s| s.cluster = Some([2, 1, 1]))),
        Ok(())
    );
}

#[test]
fn the_epilogue_subtile_must_divide_the_tile() {
    let with = |tile, subtile| ScheduleSpec {
        epilogue: Some(EpilogueSchedule { tile, subtile }),
        ..ScheduleSpec::default()
    };
    assert_eq!(
        check_schedule(&with(Some([64, 64]), Some([64, 16]))),
        Ok(())
    );
    assert_eq!(
        check_schedule(&with(Some([64, 64]), Some([48, 16]))),
        Err(ScheduleError::BadEpilogue)
    );
    assert_eq!(
        check_schedule(&with(Some([0, 64]), None)),
        Err(ScheduleError::BadEpilogue)
    );
}

#[test]
fn set_fields_lists_exactly_what_a_spec_sets_in_a_fixed_order() {
    let (_, full, _) = schedule_fixtures()
        .into_iter()
        .find(|(n, _, _)| *n == "FullySet")
        .expect("the fully set fixture");
    assert_eq!(
        full.set_fields(),
        vec![
            ScheduleField::Tile,
            ScheduleField::WarpTile,
            ScheduleField::InstrTile,
            ScheduleField::Stages,
            ScheduleField::SmemCarveout,
            ScheduleField::WorkDivision,
            ScheduleField::Epilogue,
            ScheduleField::Cluster,
        ]
    );
    // each name is distinct: an emitter's decline names exactly one field
    let names: Vec<&str> = full.set_fields().iter().map(|f| f.name()).collect();
    let mut sorted = names.clone();
    sorted.sort_unstable();
    sorted.dedup();
    assert_eq!(sorted.len(), names.len());
    // the control: data-parallel work and an automatic stage count set nothing
    let mut quiet = full.clone();
    quiet.tile = None;
    quiet.stages = StageCount::Auto;
    quiet.smem_carveout_bytes = None;
    quiet.work = WorkDivision::DataParallel;
    quiet.epilogue = None;
    quiet.cluster = None;
    assert!(quiet.set_fields().is_empty());
}

#[test]
fn every_schedule_fixture_gives_the_result_it_states() {
    for (name, spec, expected) in schedule_fixtures() {
        assert_eq!(check_schedule(&spec), expected, "{name}");
    }
}

#[test]
fn a_spec_round_trips_through_json() {
    for (name, spec, _) in schedule_fixtures() {
        let text = serde_json::to_string(&spec).expect("serialises");
        let back: ScheduleSpec = serde_json::from_str(&text).expect("parses");
        assert_eq!(back, spec, "{name}");
    }
}
