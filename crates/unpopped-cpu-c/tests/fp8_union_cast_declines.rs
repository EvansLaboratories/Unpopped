// SPDX-License-Identifier: MIT OR Apache-2.0
//! Plan E1b (KISS 0.24.0): `Cast` into or out of `F8E3M4` / `F8UE4M3` runs on the oracle, and `unpopped-cpu-c` DECLINES it by
//! name: no C codec for them is emitted, and a declined lowering is a typed `LowerError`, never C that computes the wrong thing.
//! The control is the same kernel through `F16`, which cpu-c lowers.

use unpopped::backend::Backend;
use unpopped::exec::{ExecId, LaunchShape};
use unpopped::ir::{CastSpec, RoundMode, Saturation, ScalarExpr};
use unpopped::kernel::{Block, BufferRole, KernelFn, LocalDecl, Param, Place, Stmt};
use unpopped_cpu_c::CpuC;
use unpopped_vocab::ElementKind;

fn round_trip(via: ElementKind, sat: Saturation) -> KernelFn {
    let tid = ScalarExpr::Exec(ExecId::ThreadIdx(0));
    let at = |b: u32| Place::Logical {
        buffer: b,
        coords: vec![tid.clone()],
    };
    let cast = |e, to, sat| {
        ScalarExpr::Cast(
            Box::new(e),
            CastSpec {
                to,
                round: RoundMode::NearestEven,
                sat,
            },
        )
    };
    KernelFn::new(
        "rt",
        ElementKind::F32,
        vec![
            Param::buffer("in", ElementKind::F32, BufferRole::Read, Some(1)),
            Param::buffer("out", ElementKind::F32, BufferRole::ReadWrite, Some(1)),
        ],
        vec![LocalDecl::scalar(ElementKind::F32)],
        Block(vec![
            Stmt::Load {
                local: 0,
                place: at(0),
            },
            Stmt::Store {
                place: at(1),
                value: cast(
                    cast(ScalarExpr::Local(0), via, sat),
                    ElementKind::F32,
                    Saturation::Wrap,
                ),
            },
        ]),
    )
    .with_launch(LaunchShape::new([4, 1, 1], [1, 1, 1]))
}

#[test]
fn cpu_c_declines_a_cast_through_the_two_new_8_bit_formats_and_lowers_the_f16_control() {
    for (via, sat) in [
        (ElementKind::F8E3M4, Saturation::Wrap),
        (ElementKind::F8UE4M3, Saturation::ClampFinite),
    ] {
        let r = CpuC.lower_kernel(&round_trip(via, sat));
        println!("{via:?}: {:?}", r.as_ref().map(|g| g.source.len()));
        assert!(r.is_err(), "{via:?} must be declined, got a lowering");
    }
    // the control: through F16 it lowers
    assert!(
        CpuC.lower_kernel(&round_trip(ElementKind::F16, Saturation::Wrap))
            .is_ok()
    );
}
