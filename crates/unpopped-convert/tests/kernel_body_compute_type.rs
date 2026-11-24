// SPDX-License-Identifier: MIT OR Apache-2.0
//! **The compute type of a kernel with a byte buffer first and a narrow float elsewhere** (`const void* vx, __half* dst`; 24
//! gguf kernels). A narrow float anywhere makes the arithmetic `float`: the leading byte buffer no longer gives the kernel `U8`
//! arithmetic (and so a refusal of every `float` literal). Oracle runs against Rust references.

use unpopped::exec::LaunchShape;
use unpopped::oracle::TypedBuffer;
use unpopped::oracle::kernel_eval::KernelArg;
use unpopped::oracle::kernel_launch::run_launch;
use unpopped_convert::{Frontend, lift_kernel_launched};
use unpopped_vocab::ElementKind;

fn parse(src: &str) -> Option<tree_sitter::Tree> {
    let mut p = tree_sitter::Parser::new();
    p.set_language(&tree_sitter_cpp::LANGUAGE.into()).ok()?;
    p.parse(src, None)
}

const C: Frontend = Frontend::new("c", "void", &[], "out", "in", parse);

#[test]
fn a_half_output_after_a_byte_buffer_computes_in_float() {
    let src = "void k(const void* vx, __half* dst) { dst[0] = __float2half(__half2float(dst[0]) * 2.0f + 0.25f); }";
    let k = lift_kernel_launched(&C, src, "k", LaunchShape::new([1, 1, 1], [1, 1, 1])).unwrap();
    assert_eq!(k.compute, ElementKind::F32);
    let mut args = vec![
        KernelArg::Buffer(TypedBuffer::from_u8(&[4], &[0, 0, 0, 0])),
        KernelArg::Buffer(TypedBuffer::from_f16_bits(&[1], &[0x3e00])), // 1.5
    ];
    run_launch(&k, &mut args).expect("runs");
    let KernelArg::Buffer(o) = &args[1] else {
        panic!()
    };
    assert_eq!(o.to_f64_vec(), vec![3.25]);
}

#[test]
fn an_integer_kernel_with_no_narrow_float_keeps_its_integer_compute_type() {
    let src = "void k(const int* a, int* b) { b[0] = a[0] + 1; }";
    let k = lift_kernel_launched(&C, src, "k", LaunchShape::new([1, 1, 1], [1, 1, 1])).unwrap();
    assert_eq!(k.compute, ElementKind::I32);
}
