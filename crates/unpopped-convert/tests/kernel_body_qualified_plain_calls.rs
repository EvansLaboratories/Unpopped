// SPDX-License-Identifier: MIT OR Apache-2.0
//! **Census wall: a namespace-qualified plain call** (`baracuda::block_reduce_max_f32(v, smem)`, 4 Baracuda kernels). A call
//! whose name is qualified and whose last segment is a function this source defines is that function (the walker is
//! namespace-blind, as for qualified template calls); an undefined qualified name still declines. Oracle runs against Rust
//! references.

use unpopped::exec::LaunchShape;
use unpopped::kernel::KernelFn;
use unpopped::oracle::TypedBuffer;
use unpopped::oracle::kernel_eval::KernelArg;
use unpopped::oracle::kernel_launch::run_launch;
use unpopped_convert::{Frontend, KernelLiftError, lift_kernel_launched};

fn parse(src: &str) -> Option<tree_sitter::Tree> {
    let b = src.replace("__device__", "          ");
    let mut p = tree_sitter::Parser::new();
    p.set_language(&tree_sitter_cpp::LANGUAGE.into()).ok()?;
    p.parse(b, None)
}

const C: Frontend = Frontend::new("c", "void", &[], "out", "in", parse);

fn lift(src: &str) -> Result<KernelFn, KernelLiftError> {
    lift_kernel_launched(&C, src, "k", LaunchShape::new([1, 1, 1], [1, 1, 1]))
}

fn run(k: &KernelFn) -> Vec<f64> {
    let mut args = vec![
        KernelArg::Buffer(TypedBuffer::from_f32(&[2], &[0.0, 0.0])),
        KernelArg::Buffer(TypedBuffer::from_f32(&[2], &[3.0, 4.0])),
    ];
    run_launch(k, &mut args).expect("runs");
    let KernelArg::Buffer(o) = &args[0] else {
        panic!()
    };
    o.to_f64_vec()
}

const NS: &str = "namespace baracuda { namespace util {\n\
    __device__ float scale2(float v) { return v * 2.0f; }\n\
    __device__ void bump(float& acc, float v) { acc += v; }\n\
    } }\n";

#[test]
fn a_qualified_call_as_a_value_and_as_a_statement_runs_the_function() {
    let src = format!(
        "{NS}void k(float* out, const float* in) {{ float acc = 1.0f; baracuda::util::bump(acc, in[1]); \
         out[0] = baracuda::util::scale2(in[0]) + acc; out[1] = scale2(in[0]); }}"
    );
    // 3 * 2 + (1 + 4); the bare spelling is the same function
    assert_eq!(run(&lift(&src).unwrap()), vec![11.0, 6.0]);
}

#[test]
fn a_qualified_name_the_source_does_not_define_still_declines() {
    let src = format!(
        "{NS}void k(float* out, const float* in) {{ out[0] = baracuda::util::nowhere(in[0]); }}"
    );
    assert!(matches!(
        lift(&src),
        Err(KernelLiftError::NotYetLifted { .. })
    ));
}
