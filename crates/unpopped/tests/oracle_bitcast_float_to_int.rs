// SPDX-License-Identifier: MIT OR Apache-2.0
//! **The oracle evaluates an integer `Bitcast` of a float** (found 2026-11-23 lifting the `__float_as_int` family: the kernel passed
//! `check_kernel`, then the oracle answered `Unsupported("integer expression outside the v0 subset")`). The integer evaluator reads only
//! integer operands, so a `Bitcast` whose operand is a float node went to it and failed; it now evaluates the operand at its own type and
//! reads the bits as the integer type. A float-to-float and an int-to-float `Bitcast` already worked and are the controls.

use unpopped::exec::LaunchShape;
use unpopped::ir::ScalarExpr;
use unpopped::kernel::{Block, BufferRole, KernelFn, LocalDecl, Param, Place, Stmt, check_kernel};
use unpopped::oracle::TypedBuffer;
use unpopped::oracle::kernel_eval::KernelArg;
use unpopped::oracle::kernel_launch::run_launch;
use unpopped_vocab::ElementKind as E;

/// `f = in[0]; out[0] = bitcast<to>(f);` for an F32 `in` and an `out` of type `to` (computed at F32).
fn kernel(to: E) -> KernelFn {
    let at = |buffer: u32| Place::Logical {
        buffer,
        coords: vec![ScalarExpr::ConstInt(0, E::I32)],
    };
    KernelFn::new(
        "bits",
        E::F32,
        vec![
            Param::buffer("out", to, BufferRole::ReadWrite, Some(1)),
            Param::buffer("in", E::F32, BufferRole::Read, Some(1)),
        ],
        vec![LocalDecl::scalar(E::F32), LocalDecl::scalar(to)],
        Block(vec![
            Stmt::Load {
                local: 0,
                place: at(1),
            },
            Stmt::Let {
                local: 1,
                init: ScalarExpr::Bitcast(Box::new(ScalarExpr::Local(0)), to),
            },
            Stmt::Store {
                place: at(0),
                value: ScalarExpr::Local(1),
            },
        ]),
    )
    .with_launch(LaunchShape::new([1, 1, 1], [1, 1, 1]))
}

fn run(to: E, input: f32, out: TypedBuffer) -> TypedBuffer {
    let k = kernel(to);
    check_kernel(&k).expect("a same-width Bitcast checks");
    let mut args = vec![
        KernelArg::Buffer(out),
        KernelArg::Buffer(TypedBuffer::from_f32(&[1], &[input])),
    ];
    run_launch(&k, &mut args).expect("runs");
    let KernelArg::Buffer(b) = args.swap_remove(0) else {
        panic!()
    };
    b
}

#[test]
fn a_float_read_as_i32_and_as_u32_gives_its_bit_pattern() {
    // 1.5f is 0x3FC00000; -2.0f is 0xC0000000, negative as an i32 and large as a u32
    let b = run(E::I32, 1.5, TypedBuffer::from_i32(&[1], &[0]));
    assert_eq!(b.to_i128_vec(), vec![i128::from(0x3FC0_0000i32)]);
    let b = run(E::I32, -2.0, TypedBuffer::from_i32(&[1], &[0]));
    assert_eq!(b.to_i128_vec(), vec![i128::from(0xC000_0000u32 as i32)]);
    let b = run(E::U32, -2.0, TypedBuffer::from_u32(&[1], &[0]));
    assert_eq!(b.to_i128_vec(), vec![i128::from(0xC000_0000u32)]);
}

#[test]
fn the_sign_of_zero_and_a_nan_payload_survive() {
    let b = run(E::U32, -0.0, TypedBuffer::from_u32(&[1], &[0]));
    assert_eq!(b.to_i128_vec(), vec![i128::from(0x8000_0000u32)], "-0.0");
    let nan = f32::from_bits(0x7fc0_1234);
    let b = run(E::U32, nan, TypedBuffer::from_u32(&[1], &[0]));
    assert_eq!(
        b.to_i128_vec(),
        vec![i128::from(0x7fc0_1234u32)],
        "a NaN payload"
    );
}
