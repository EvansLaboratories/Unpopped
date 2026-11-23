// SPDX-License-Identifier: MIT OR Apache-2.0
//! **Census wall: a kernel parameter with a default value** (`int max_entries = 0`, `int64_t* profiler_buf = nullptr`; 6 Baracuda
//! kernels). A kernel is launched with every argument, so the default is never used: the parameter is the same parameter.
//! Oracle runs against Rust references.

use unpopped::exec::LaunchShape;
use unpopped::kernel::KernelFn;
use unpopped::oracle::TypedBuffer;
use unpopped::oracle::kernel_eval::KernelArg;
use unpopped::oracle::kernel_launch::run_launch;
use unpopped_convert::{Frontend, KernelLiftError, lift_kernel_launched};

fn parse(src: &str) -> Option<tree_sitter::Tree> {
    let mut p = tree_sitter::Parser::new();
    p.set_language(&tree_sitter_cpp::LANGUAGE.into()).ok()?;
    p.parse(src, None)
}

const C: Frontend = Frontend::new("c", "void", &[], "out", "in", parse);

fn lift(src: &str) -> Result<KernelFn, KernelLiftError> {
    lift_kernel_launched(&C, src, "k", LaunchShape::new([1, 1, 1], [1, 1, 1]))
}

#[test]
fn a_defaulted_scalar_parameter_takes_the_launched_value_not_the_default() {
    let k =
        lift("void k(float* out, int n, int scale = 5) { out[0] = (float)(n * scale); }").unwrap();
    let mut args = vec![
        KernelArg::Buffer(TypedBuffer::from_f32(&[1], &[0.0])),
        KernelArg::Int(3),
        KernelArg::Int(7),
    ];
    run_launch(&k, &mut args).expect("runs");
    let KernelArg::Buffer(o) = &args[0] else {
        panic!()
    };
    // 3 * 7, not 3 * 5
    assert_eq!(o.to_f64_vec(), vec![21.0]);
}

#[test]
fn a_defaulted_pointer_parameter_is_a_buffer() {
    let k = lift("void k(float* out, const float* extra = nullptr) { out[0] = extra[0] + 1.0f; }")
        .unwrap();
    let mut args = vec![
        KernelArg::Buffer(TypedBuffer::from_f32(&[1], &[0.0])),
        KernelArg::Buffer(TypedBuffer::from_f32(&[1], &[41.0])),
    ];
    run_launch(&k, &mut args).expect("runs");
    let KernelArg::Buffer(o) = &args[0] else {
        panic!()
    };
    assert_eq!(o.to_f64_vec(), vec![42.0]);
}
