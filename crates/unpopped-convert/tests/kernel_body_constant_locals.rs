// SPDX-License-Identifier: MIT OR Apache-2.0
//! **Census wall: an array sized by a constant local** (`static constexpr int kBr = TileShape<T>::Br; __shared__ CT s[kBr];`,
//! 5 Baracuda kernels). A `const`/`constexpr` integer local with a value known at lift time is a constant expression where C
//! needs one: an array extent. Oracle runs against Rust references.

use unpopped::exec::LaunchShape;
use unpopped::kernel::KernelFn;
use unpopped::oracle::TypedBuffer;
use unpopped::oracle::kernel_eval::KernelArg;
use unpopped::oracle::kernel_launch::run_launch;
use unpopped_convert::{Frontend, KernelLiftError, lift_kernel_launched};

fn parse(src: &str) -> Option<tree_sitter::Tree> {
    let b = src.replace("__shared__", "volatile  ");
    let mut p = tree_sitter::Parser::new();
    p.set_language(&tree_sitter_cpp::LANGUAGE.into()).ok()?;
    p.parse(b, None)
}

const C: Frontend = Frontend::new("c", "void", &[], "out", "in", parse);

fn lift(src: &str, block: u32) -> Result<KernelFn, KernelLiftError> {
    lift_kernel_launched(&C, src, "k", LaunchShape::new([block, 1, 1], [1, 1, 1]))
}

fn run(k: &KernelFn, n: usize) -> Vec<f64> {
    let mut args = vec![KernelArg::Buffer(TypedBuffer::from_f32(
        &[n as i64],
        &vec![0.0; n],
    ))];
    run_launch(k, &mut args).expect("runs");
    let KernelArg::Buffer(o) = &args[0] else {
        panic!()
    };
    o.to_f64_vec()
}

#[test]
fn a_constexpr_local_sizes_a_shared_array_and_a_private_one() {
    // 4 threads fill a shared array of kN = 2 * 2 floats, then each reads its right neighbour's slot
    // (the struct-trait form, `constexpr int kN = Shape<2>::N;`, is the census kernels' shape: tests/kernel_body_trait_types.rs)
    let src = "void k(float* out) { constexpr int kN = 4; const int m = kN / 2 + 1; __shared__ float s[kN]; float priv[m]; \
        s[threadIdx.x] = (float)(threadIdx.x + 1); priv[2] = 100.0f; __syncthreads(); \
        out[threadIdx.x] = s[(threadIdx.x + 1) % 4] + priv[2]; }";
    let k = lift(src, 4).unwrap();
    assert_eq!(run(&k, 4), vec![102.0, 103.0, 104.0, 101.0]);
}

#[test]
fn a_local_that_is_not_constant_still_cannot_size_an_array() {
    // not `const`
    let a = "void k(float* out) { int n = 4; float priv[n]; priv[0] = 1.0f; out[0] = priv[0]; }";
    // `const`, from a run-time value
    let b = "void k(float* out) { const int n = (int)threadIdx.x + 1; float priv[n]; priv[0] = 1.0f; out[0] = priv[0]; }";
    for src in [a, b] {
        assert!(
            matches!(lift(src, 1), Err(KernelLiftError::NotYetLifted { .. })),
            "{src}"
        );
    }
}
