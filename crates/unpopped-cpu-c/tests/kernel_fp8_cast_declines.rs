// SPDX-License-Identifier: MIT OR Apache-2.0
//! `unpopped-cpu-c` has no spelling for a `Cast` to or from the OCP FP8 formats (ROADMAP U68 gave the kernel layer those endpoints): it
//! declines such a kernel by name, never lowers it to a wrong conversion. The same kernel with `F32` buffers lowers.

use unpopped::backend::Backend;
use unpopped::exec::{ExecId, LaunchShape};
use unpopped::ir::{CastSpec, RoundMode, Saturation, ScalarExpr};
use unpopped::kernel::{Block, BufferRole, KernelFn, LocalDecl, Param, Place, Stmt, check_kernel};
use unpopped_cpu_c::CpuC;
use unpopped_vocab::ElementKind::{self, F32, Fp8E4M3FN, Fp8E5M2};

fn convert(to: ElementKind) -> KernelFn {
    let at = |buffer| Place::Logical {
        buffer,
        coords: vec![ScalarExpr::Exec(ExecId::ThreadIdx(0))],
    };
    KernelFn::new(
        "cvt",
        F32,
        vec![
            Param::buffer("in", F32, BufferRole::Read, Some(1)),
            Param::buffer("out", to, BufferRole::Write, Some(1)),
        ],
        vec![LocalDecl::scalar(F32)],
        Block(vec![
            Stmt::Load {
                local: 0,
                place: at(0),
            },
            Stmt::Store {
                place: at(1),
                value: ScalarExpr::Cast(
                    Box::new(ScalarExpr::Local(0)),
                    CastSpec {
                        to,
                        round: RoundMode::NearestEven,
                        sat: Saturation::ClampFinite,
                    },
                ),
            },
        ]),
    )
    .with_launch(LaunchShape::new([8, 1, 1], [1, 1, 1]))
}

#[test]
fn a_cast_into_an_fp8_format_is_declined_by_name_and_the_f32_control_lowers() {
    // the control: the same kernel widening to F64 lowers (so the decline is about FP8, not the kernel shape)
    assert!(CpuC.lower_kernel(&convert(ElementKind::F64)).is_ok());
    for fp8 in [Fp8E4M3FN, Fp8E5M2] {
        let k = convert(fp8);
        assert_eq!(check_kernel(&k), Ok(()), "{fp8:?} is well formed");
        let err = CpuC
            .lower_kernel(&k)
            .expect_err("an FP8 buffer or Cast is not spelled");
        let text = format!("{err:?}").to_lowercase();
        assert!(
            text.contains("fp8") || text.contains("f8"),
            "{fp8:?}: {err:?}"
        );
    }
}
