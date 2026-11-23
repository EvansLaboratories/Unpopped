// SPDX-License-Identifier: MIT OR Apache-2.0
//! **Census wall: the optional-pointer idiom `T* xa = (da != nullptr) ? da + off : nullptr;`** (loss backward kernels, ring
//! attention finalize: 10 real instantiations). Whether a buffer is null is known when lifting (`LiftContext::with_null_buffer`), so
//! the condition is a constant that picks its arm. A chosen `nullptr` stands for the other arm's own base when that base is a
//! buffer the caller declared null (a later `if (xa)` is then false); a condition that is not known when lifting, and a
//! `nullptr` that is not a declared-null buffer, decline by name. Oracle runs against Rust references.

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

fn ctx() -> LiftContext {
    LiftContext::new(LaunchShape::new([1, 1, 1], [1, 1, 1]))
}

fn lift(src: &str, c: &LiftContext) -> Result<KernelFn, KernelLiftError> {
    lift_kernel_in(&C, src, "k", c)
}

/// (out, da) after a run with `in = [3, 4]` and `da` all 7s.
fn run(k: &KernelFn, n: i32) -> (Vec<f64>, Vec<f64>) {
    let mut args = vec![
        KernelArg::Buffer(TypedBuffer::from_f32(&[2], &[0.0, 0.0])),
        KernelArg::Buffer(TypedBuffer::from_f32(&[2], &[3.0, 4.0])),
        KernelArg::Buffer(TypedBuffer::from_f32(&[6], &[7.0; 6])),
        KernelArg::Int(i128::from(n)),
    ];
    run_launch(k, &mut args).expect("runs");
    let get = |i: usize| {
        let KernelArg::Buffer(b) = &args[i] else {
            panic!()
        };
        b.to_f64_vec()
    };
    (get(0), get(2))
}

const WRITES: &str = "void k(float* out, const float* in, float* da, int n) { \
    float* xa = (da != nullptr) ? da + 2 : nullptr; \
    if (xa != nullptr) { xa[0] = in[0] * 2.0f; xa[1] = in[1]; } out[0] = 1.0f; out[1] = (xa == nullptr) ? 5.0f : 6.0f; }";

#[test]
fn a_present_buffer_takes_the_pointer_arm_and_a_declared_null_one_takes_the_nullptr_arm() {
    let present = lift(WRITES, &ctx()).expect("lifts (was a decline)");
    let (out, da) = run(&present, 0);
    assert_eq!(out, vec![1.0, 6.0]);
    assert_eq!(da, vec![7.0, 7.0, 6.0, 4.0, 7.0, 7.0]);
    let absent = lift(WRITES, &ctx().with_null_buffer("da")).expect("lifts with da null");
    let (out, da) = run(&absent, 0);
    assert_eq!(out, vec![1.0, 5.0]);
    assert_eq!(da, vec![7.0; 6], "the guarded stores never run");
}

#[test]
fn the_nullptr_arm_may_come_first() {
    let src = "void k(float* out, const float* in, float* da, int n) { \
        float* xa = (da == nullptr) ? nullptr : da + 2; if (xa) { xa[0] = in[0]; } out[0] = (xa != nullptr) ? 1.0f : 2.0f; }";
    let (out, da) = run(&lift(src, &ctx()).unwrap(), 0);
    assert_eq!((out[0], da[2]), (1.0, 3.0));
    let (out, da) = run(&lift(src, &ctx().with_null_buffer("da")).unwrap(), 0);
    assert_eq!((out[0], da[2]), (2.0, 7.0));
}

#[test]
fn what_is_not_known_when_lifting_declines_by_name() {
    for (src, why) in [
        // the condition depends on a runtime value
        (
            "void k(float* out, const float* in, float* da, int n) { float* xa = (n > 0) ? da + 2 : nullptr; out[0] = 1.0f; }",
            "not known when lifting",
        ),
        // `nullptr` chosen, but `da` is a real buffer: the null has no representation here
        (
            "void k(float* out, const float* in, float* da, int n) { float* xa = (da != nullptr) ? nullptr : da + 2; out[0] = 1.0f; }",
            "not a buffer the caller declared null",
        ),
    ] {
        match lift(src, &ctx()) {
            Err(KernelLiftError::NotYetLifted { construct, .. }) => {
                assert!(
                    construct.contains(why),
                    "{src}: `{construct}` lacks `{why}`"
                );
            }
            other => panic!("{src}: expected a decline, got {other:?}"),
        }
    }
}
