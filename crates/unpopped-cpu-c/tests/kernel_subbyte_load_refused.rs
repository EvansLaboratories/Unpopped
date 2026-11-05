// SPDX-License-Identifier: MIT OR Apache-2.0
//! **The C emitter does not spell a sub-byte kernel buffer yet** (ROADMAP U9b, `docs/options-u9b-subbyte-dequant.md`): a kernel
//! that loads an `I4` buffer element and widens it is refused with a typed error naming the buffer, although the plan path already
//! has software codecs for `i4`/`u4`/`b1`. Pinned so that spelling them (slice S2 of the note) turns this red on purpose.

use unpopped::backend::{Backend, LowerError};
use unpopped::exec::{ExecId, LaunchShape};
use unpopped::ir::{CastSpec, RoundMode, Saturation, ScalarExpr};
use unpopped::kernel::{Block, BufferRole, KernelFn, LocalDecl, Param, Place, Stmt};
use unpopped_cpu_c::CpuC;
use unpopped_vocab::ElementKind;

const I4: ElementKind = ElementKind::I4;

#[test]
fn an_i4_buffer_is_refused_with_a_typed_error() {
    let at = |b| Place::Logical {
        buffer: b,
        coords: vec![ScalarExpr::Exec(ExecId::ThreadIdx(0))],
    };
    let k = KernelFn::new(
        "i4_copy",
        ElementKind::I32,
        vec![
            Param::buffer("w", I4, BufferRole::Read, Some(1)),
            Param::buffer("out", ElementKind::I32, BufferRole::ReadWrite, Some(1)),
        ],
        vec![LocalDecl::scalar(I4)],
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
                        to: ElementKind::I32,
                        round: RoundMode::NearestEven,
                        sat: Saturation::Wrap,
                    },
                ),
            },
        ]),
    )
    .with_launch(LaunchShape::new([8, 1, 1], [1, 1, 1]));
    match CpuC.lower_kernel(&k) {
        Err(LowerError::UnsupportedKernel { detail }) => {
            assert!(detail.contains("buffer `w` has dtype I4"), "{detail}");
        }
        other => panic!("expected a typed refusal, got {other:?}"),
    }
}
