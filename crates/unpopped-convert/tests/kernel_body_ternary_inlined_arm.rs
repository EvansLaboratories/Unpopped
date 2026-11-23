// SPDX-License-Identifier: MIT OR Apache-2.0
//! **Walker bug found by real instantiations (census `Invalid(UseBeforeLet)`): a conditional whose arm is an inlined call** (`keep ?
//! input[k] : zero_of<T>()`). An inlined call flushes the statements hoisted so far, so the positions the conditional kept to split
//! its two arms were wrong: the callee's `Let` landed in the other arm and the checker refused the kernel. Each arm's statements are
//! now taken apart as they arrive. Oracle runs: both arms taken, the arm not taken reads nothing.

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
        KernelArg::Buffer(TypedBuffer::from_f32(&[4], &[0.0; 4])),
        KernelArg::Buffer(TypedBuffer::from_f32(&[4], &[1.0, 2.0, 3.0, 4.0])),
    ];
    run_launch(k, &mut args).expect("runs");
    let KernelArg::Buffer(o) = &args[0] else {
        panic!()
    };
    o.to_f64_vec()
}

#[test]
fn an_arm_that_is_an_inlined_call_and_an_arm_that_loads_keep_their_own_statements() {
    let k = lift(
        "float twice(float x) { float y = x * 2.0f; return y; }\n\
         void k(float* out, const float* in) { for (int i = 0; i < 4; ++i) { bool keep = i < 2; \
         out[i] = keep ? in[i] : twice(in[i] + 10.0f); } }",
    );
    // i = 0, 1 keep the load; i = 2, 3 take the call: 2 * (3 + 10), 2 * (4 + 10)
    assert_eq!(run(&k), vec![1.0, 2.0, 26.0, 28.0]);
    // the call in the FIRST arm, the load in the second
    let k = lift(
        "float twice(float x) { float y = x * 2.0f; return y; }\n\
         void k(float* out, const float* in) { for (int i = 0; i < 4; ++i) { bool keep = i < 2; \
         out[i] = keep ? twice(in[i]) : in[i] + 100.0f; } }",
    );
    assert_eq!(run(&k), vec![2.0, 4.0, 103.0, 104.0]);
}

#[test]
fn a_call_in_each_arm_and_no_memory_read_still_lifts_as_a_select() {
    let k = lift(
        "float inc(float x) { float y = x + 1.0f; return y; }\n\
         float dec(float x) { float y = x - 1.0f; return y; }\n\
         void k(float* out, const float* in) { for (int i = 0; i < 4; ++i) { float v = (float)i; out[i] = i < 2 ? inc(v) : dec(v); } }",
    );
    assert_eq!(run(&k), vec![1.0, 2.0, 1.0, 2.0]);
}
