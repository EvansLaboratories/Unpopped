// SPDX-License-Identifier: MIT OR Apache-2.0
//! **Census wall: `auto x = e;`** (`auto result = warp_buf[0];` in the block reductions; 8 kernels). The local takes the type of its
//! initialiser, evaluated once. An `auto` with no initialiser, a pointer or an array still declines. Oracle runs.

use unpopped::exec::LaunchShape;
use unpopped::kernel::KernelFn;
use unpopped::oracle::TypedBuffer;
use unpopped::oracle::kernel_eval::KernelArg;
use unpopped::oracle::kernel_launch::run_launch;
use unpopped_convert::{Frontend, KernelLiftError, LiftContext, lift_kernel_in};

fn parse(src: &str) -> Option<tree_sitter::Tree> {
    let mut p = tree_sitter::Parser::new();
    p.set_language(&tree_sitter_cpp::LANGUAGE.into()).ok()?;
    p.parse(src, None)
}

const C: Frontend = Frontend::new("c", "void", &[], "out", "in", parse);

fn lift(src: &str) -> Result<KernelFn, KernelLiftError> {
    lift_kernel_in(
        &C,
        src,
        "k",
        &LiftContext::new(LaunchShape::new([1, 1, 1], [1, 1, 1])),
    )
}

#[test]
fn auto_takes_the_type_of_its_initialiser() {
    // float stays float (0.5 survives), int stays int (7 / 2 = 3), a double expression stays double
    let k = lift(
        "void k(double* out, const float* in) { auto a = in[0]; auto n = 7; auto h = n / 2; const auto d = (double)a * 2.0; \
         out[0] = (double)a; out[1] = (double)h; out[2] = d; }",
    )
    .unwrap();
    let mut args = vec![
        KernelArg::Buffer(TypedBuffer::from_f64(&[3], &[0.0; 3])),
        KernelArg::Buffer(TypedBuffer::from_f32(&[1], &[0.5])),
    ];
    run_launch(&k, &mut args).expect("runs");
    let KernelArg::Buffer(o) = &args[0] else {
        panic!()
    };
    assert_eq!(o.to_f64_vec(), vec![0.5, 3.0, 1.0]);
}

#[test]
fn an_auto_without_an_initialiser_a_pointer_or_an_array_declines() {
    for body in ["auto x;", "auto p = in;", "auto a[2] = {1, 2};"] {
        let r = lift(&format!(
            "void k(float* out, const float* in) {{ {body} out[0] = 1.0f; }}"
        ));
        assert!(
            matches!(r, Err(KernelLiftError::NotYetLifted { .. })),
            "{body}: {r:?}"
        );
    }
}
