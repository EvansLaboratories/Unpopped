// SPDX-License-Identifier: MIT OR Apache-2.0
//! **What the kernel layer does today with a sub-byte (`I4`) buffer element** (ROADMAP U9b, `docs/options-u9b-subbyte-dequant.md`):
//! a `Load` and `Store` of an `I4` element work in the checker and the oracle, packed two per byte, low nibble first; a `Cast`
//! from `I4` (the widening a dequantizing GEMM needs) is a typed refusal ("not defined yet"). Both facts are pinned so that the
//! day the `Cast` endpoints are extended this file says so loudly and the dequant kernel (U9b) can be built.

use unpopped::exec::{ExecId, LaunchShape};
use unpopped::ir::{CastSpec, RoundMode, Saturation, ScalarExpr};
use unpopped::kernel::{Block, BufferRole, KernelFn, LocalDecl, Param, Place, Stmt, check_kernel};
use unpopped::oracle::TypedBuffer;
use unpopped::oracle::kernel_eval::KernelArg;
use unpopped::oracle::kernel_launch::run_launch;
use unpopped::types::TypeError;
use unpopped_vocab::ElementKind;

const I32: ElementKind = ElementKind::I32;
const I4: ElementKind = ElementKind::I4;

fn at(buffer: u32) -> Place {
    Place::Logical {
        buffer,
        coords: vec![ScalarExpr::Exec(ExecId::ThreadIdx(0))],
    }
}

/// `out[t] = value(w[t])` where `value` is the local itself (`I4` out) or its widening cast (`I32` out).
fn kernel(widen: bool) -> KernelFn {
    let value = if widen {
        ScalarExpr::Cast(
            Box::new(ScalarExpr::Local(0)),
            CastSpec {
                to: I32,
                round: RoundMode::NearestEven,
                sat: Saturation::Wrap,
            },
        )
    } else {
        ScalarExpr::Local(0)
    };
    let out = if widen { I32 } else { I4 };
    KernelFn::new(
        "i4",
        I32,
        vec![
            Param::buffer("w", I4, BufferRole::Read, Some(1)),
            Param::buffer("out", out, BufferRole::ReadWrite, Some(1)),
        ],
        vec![LocalDecl::scalar(I4)],
        Block(vec![
            Stmt::Load {
                local: 0,
                place: at(0),
            },
            Stmt::Store {
                place: at(1),
                value,
            },
        ]),
    )
    .with_launch(LaunchShape::new([8, 1, 1], [1, 1, 1]))
}

#[test]
fn a_sub_byte_load_and_store_round_trip_in_the_oracle() {
    let k = kernel(false);
    assert_eq!(check_kernel(&k), Ok(()));
    // nibbles 8, F, 0, 1, 7, 3, C, 5 (as signed i4: -8, -1, 0, 1, 7, 3, -4, 5), low nibble first in each byte
    let nibbles: [u8; 8] = [0x8, 0xF, 0x0, 0x1, 0x7, 0x3, 0xC, 0x5];
    let bytes: Vec<u8> = nibbles.chunks(2).map(|p| p[0] | p[1] << 4).collect();
    let mut args = [
        KernelArg::Buffer(TypedBuffer::new(I4, vec![8], vec![1], bytes.clone())),
        KernelArg::Buffer(TypedBuffer::new(I4, vec![8], vec![1], vec![0; 4])),
    ];
    run_launch(&k, &mut args).expect("the oracle runs it");
    let KernelArg::Buffer(out) = &args[1] else {
        unreachable!("a buffer")
    };
    assert_eq!(out.bytes, bytes, "every nibble came back where it went");
    assert_eq!(
        bytes,
        [0xF8, 0x10, 0x37, 0x5C],
        "the packing pinned: low nibble first"
    );
}

#[test]
fn a_widening_cast_from_a_sub_byte_type_is_a_typed_refusal_for_now() {
    let err = check_kernel(&kernel(true)).expect_err("the cast is not defined yet");
    let text = format!("{err:?}");
    assert!(
        matches!(
            err,
            unpopped::kernel::KernelError::Type(TypeError::IllegalCast { from: I4, .. })
        ),
        "{text}"
    );
    assert!(text.contains("not defined yet"), "{text}");
}
