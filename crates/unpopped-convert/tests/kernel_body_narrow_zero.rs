// SPDX-License-Identifier: MIT OR Apache-2.0
//! **Census wall (5 kernels `Invalid(IllegalIntLeaf U8)`): the zero of a narrow integer.** `uint8_t sc, m;` (no initialiser) and a
//! function returning `uint8_t` start from zero, which was written as a `U8` constant leaf, and an integer leaf is `I32`, `I64`,
//! `U32` or `U64` only. The zero is now an `I32` zero converted to the narrow type. Oracle runs against Rust references.

use unpopped::exec::LaunchShape;
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

fn run(src: &str, n: usize) -> Vec<f64> {
    let k = lift_kernel_in(
        &C,
        src,
        "k",
        &LiftContext::new(LaunchShape::new([1, 1, 1], [1, 1, 1])),
    )
    .expect("lifts and passes the checker");
    let mut args = vec![
        KernelArg::Buffer(TypedBuffer::from_f32(&[n as i64], &vec![9.0; n])),
        KernelArg::Buffer(TypedBuffer::from_u8(&[4], &[0, 5, 200, 7])),
    ];
    run_launch(&k, &mut args).expect("runs");
    let KernelArg::Buffer(o) = &args[0] else {
        panic!()
    };
    o.to_f64_vec()
}

#[test]
fn an_uninitialised_narrow_local_and_a_narrow_return_start_from_a_checked_zero() {
    let src = "unsigned char truthy(unsigned char x) { return x != 0 ? 1u : 0u; }\n\
        void k(float* out, const unsigned char* in) { unsigned char sc, m; \
        if (in[1] > 3) { sc = in[1] & 63; m = in[2] >> 4; } \
        out[0] = (float)sc; out[1] = (float)m; out[2] = (float)truthy(in[0]); out[3] = (float)truthy(in[3]); \
        unsigned short h; out[4] = (float)h; short s; out[5] = (float)s; }";
    // sc = 5, m = 200 >> 4 = 12; truthy(0) = 0, truthy(7) = 1; the never-assigned locals read as zero
    assert_eq!(run(src, 6), vec![5.0, 12.0, 0.0, 1.0, 0.0, 0.0]);
}
