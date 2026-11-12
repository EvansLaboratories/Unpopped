// SPDX-License-Identifier: MIT OR Apache-2.0
//! **A float-to-integer `Cast` in the kernel layer** (found while building ROADMAP U14): the checker admits a `Cast` of an
//! `F32` local to an integer (`Saturation::Clamp`), and the oracle used to stop on it with "integer expression outside the
//! v0 subset", because its integer evaluator read the operand as an integer. The conversion is now the float node's:
//! rounded by the mode, saturated into the destination, NaN to 0 (`docs` of [`Saturation::Clamp`]).

use unpopped::ir::{CastSpec, RoundMode, Saturation, ScalarExpr};
use unpopped::kernel::{Block, BufferRole, KernelFn, LocalDecl, Param, Place, Stmt, check_kernel};
use unpopped::oracle::TypedBuffer;
use unpopped::oracle::kernel_eval::{KernelArg, run_kernel};
use unpopped_vocab::ElementKind;

const F32: ElementKind = ElementKind::F32;
const I32: ElementKind = ElementKind::I32;
const U32: ElementKind = ElementKind::U32;

const INPUTS: [f32; 10] = [
    2.5,
    -2.5,
    3.5,
    0.9,
    -0.9,
    f32::NAN,
    1.0e10,
    -1.0e10,
    f32::INFINITY,
    f32::NEG_INFINITY,
];

fn at(buffer: u32) -> Place {
    Place::Logical {
        buffer,
        coords: vec![ScalarExpr::Local(0)],
    }
}

/// `for i in 0..10 { out[i] = (to) in[i] }`, through an `F32` local.
fn kernel(to: ElementKind, round: RoundMode) -> KernelFn {
    let spec = CastSpec {
        to,
        round,
        sat: Saturation::Clamp,
    };
    let ci = |v: i64| ScalarExpr::ConstInt(v, I32);
    KernelFn::new(
        "float_to_int",
        F32,
        vec![
            Param::buffer("in", F32, BufferRole::Read, Some(1)),
            Param::buffer("out", to, BufferRole::Write, Some(1)),
        ],
        vec![LocalDecl::scalar(I32), LocalDecl::scalar(F32)],
        Block(vec![Stmt::For {
            local: 0,
            start: ci(0),
            end: ci(INPUTS.len() as i64),
            step: ci(1),
            body: Block(vec![
                Stmt::Load {
                    local: 1,
                    place: at(0),
                },
                Stmt::Store {
                    place: at(1),
                    value: ScalarExpr::Cast(Box::new(ScalarExpr::Local(1)), spec),
                },
            ]),
        }]),
    )
}

fn convert(to: ElementKind, round: RoundMode) -> Vec<i64> {
    let k = kernel(to, round);
    assert_eq!(check_kernel(&k), Ok(()), "well formed");
    let bytes = INPUTS.iter().flat_map(|x| x.to_le_bytes()).collect();
    let n = INPUTS.len() as i64;
    let mut args = vec![
        KernelArg::Buffer(TypedBuffer::new(F32, vec![n], vec![1], bytes)),
        KernelArg::Buffer(TypedBuffer::new(
            to,
            vec![n],
            vec![1],
            vec![0; 4 * INPUTS.len()],
        )),
    ];
    run_kernel(&k, &mut args).expect("the oracle converts it");
    let KernelArg::Buffer(out) = &args[1] else {
        unreachable!("buffer argument");
    };
    out.bytes
        .chunks(4)
        .map(|c| {
            let b: [u8; 4] = c.try_into().expect("four bytes");
            if to == I32 {
                i64::from(i32::from_le_bytes(b))
            } else {
                i64::from(u32::from_le_bytes(b))
            }
        })
        .collect()
}

#[test]
fn a_float_to_signed_cast_rounds_by_the_mode_saturates_and_maps_nan_to_zero() {
    let (min, max) = (i64::from(i32::MIN), i64::from(i32::MAX));
    assert_eq!(
        convert(I32, RoundMode::TowardZero),
        [2, -2, 3, 0, 0, 0, max, min, max, min]
    );
    assert_eq!(
        convert(I32, RoundMode::NearestEven),
        [2, -2, 4, 1, -1, 0, max, min, max, min]
    );
}

#[test]
fn a_float_to_unsigned_cast_clamps_below_at_zero_and_above_at_the_maximum() {
    let max = i64::from(u32::MAX);
    assert_eq!(
        convert(U32, RoundMode::TowardZero),
        [2, 0, 3, 0, 0, 0, max, 0, max, 0]
    );
}
