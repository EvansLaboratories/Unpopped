// SPDX-License-Identifier: MIT OR Apache-2.0
//! **Census wall found by real instantiations (34 instantiations, `IllegalIntLeaf` U8/I8/U16/I16): a kernel over narrow-integer buffers
//! whose compute type is that narrow type** (`uint8_t` buffers, `row < M && col < N`). The constants the walker makes itself (the 1 and 0 of
//! `&&`, `||`, `!`, a comparison) were written at the compute type, and an integer leaf is `I32`, `I64`, `U32` or `U64` only; they are
//! `I32`, C's `int`. Oracle runs.

use unpopped::exec::LaunchShape;
use unpopped::kernel::KernelFn;
use unpopped::oracle::TypedBuffer;
use unpopped::oracle::kernel_eval::KernelArg;
use unpopped::oracle::kernel_launch::run_launch;
use unpopped_convert::{Frontend, LiftContext, lift_kernel_in};

fn parse(src: &str) -> Option<tree_sitter::Tree> {
    let mut p = tree_sitter::Parser::new();
    p.set_language(&tree_sitter_cpp::LANGUAGE.into()).ok()?;
    p.parse(src, None)
}

const C: Frontend = Frontend::new("c", "void", &[], "out", "in", parse);

fn lift(src: &str) -> KernelFn {
    lift_kernel_in(
        &C,
        src,
        "k",
        &LiftContext::new(LaunchShape::new([1, 1, 1], [1, 1, 1])),
    )
    .expect("lifts and passes the checker")
}

fn run(k: &KernelFn) -> Vec<f64> {
    let mut args = vec![
        KernelArg::Buffer(TypedBuffer::from_u8(&[4], &[0; 4])),
        KernelArg::Buffer(TypedBuffer::from_u8(&[4], &[7, 8, 9, 10])),
    ];
    run_launch(k, &mut args).expect("runs");
    let KernelArg::Buffer(o) = &args[0] else {
        panic!()
    };
    o.to_f64_vec()
}

#[test]
fn logical_operators_and_a_guarded_select_over_u8_buffers() {
    let k = lift(
        "void k(unsigned char* out, const unsigned char* in) { for (int i = 0; i < 4; ++i) { \
         unsigned char v = in[i]; bool ok = i < 3 && v > 7; bool either = i == 0 || !ok; \
         out[i] = ok ? in[i] : (either ? (unsigned char)1 : (unsigned char)0); } }",
    );
    // i = 0: ok false (7 > 7 no), either true -> 1; i = 1: ok (8 > 7) -> 8; i = 2: ok -> 9; i = 3: ok false (i < 3 no), either true -> 1
    assert_eq!(run(&k), vec![1.0, 8.0, 9.0, 1.0]);
}
