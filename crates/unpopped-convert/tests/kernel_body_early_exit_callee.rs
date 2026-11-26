// SPDX-License-Identifier: MIT OR Apache-2.0
//! **Census wall (found by the census as `Invalid: BarrierDivergent`, the gguf mmvq kernels (19 early-return-then-collective kernels in baracuda source at eae3a6dcf, none deriving the guard from `threadIdx.x`; a census figure of 57 is not reproduced by source text and may count instantiations)): a guard `return` in a `void` function
//! the kernel calls last, followed by a warp shuffle.** `if (row >= nrows) return; ...; __shfl_xor_sync(..)` inlined as
//! `if (c) {} else { rest }` put the shuffle under a branch, which the checker rightly refuses. When the call is the last
//! statement of the kernel's own body the callee's `return` leaves the kernel, so it lowers as `Stmt::Return` and the rest stays in
//! line. A call that is NOT last (more kernel statements follow) keeps the `else` form: its `return` must not end the kernel.
//! Oracle runs against Rust references.

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
    let ctx = LiftContext::new(LaunchShape::new([32, 1, 1], [1, 1, 1]).with_subgroup(32, 32));
    lift_kernel_in(&C, src, "k", &ctx)
}

fn run(k: &KernelFn, nrows: i32) -> Vec<f64> {
    let input: Vec<f32> = (1..=32).map(|v| v as f32).collect();
    let mut args = vec![
        KernelArg::Buffer(TypedBuffer::from_f32(&[2], &[0.0, 0.0])),
        KernelArg::Buffer(TypedBuffer::from_f32(&[32], &input)),
        KernelArg::Int(i128::from(nrows)),
    ];
    run_launch(k, &mut args).expect("runs");
    let KernelArg::Buffer(o) = &args[0] else {
        panic!()
    };
    o.to_f64_vec()
}

const INNER: &str = "static void inner(float* out, const float* in, int nrows) {\n\
    const int row = blockIdx.x * blockDim.y + threadIdx.y;\n\
    if (row >= nrows) return;\n\
    float t = in[threadIdx.x];\n\
    for (int mask = 16; mask > 0; mask >>= 1) { t += __shfl_xor_sync(0xffffffff, t, mask, 32); }\n\
    if (threadIdx.x == 0) { out[0] = t; }\n\
}\n";

#[test]
fn a_guard_return_before_a_warp_shuffle_in_the_last_called_function_lifts_and_runs() {
    let k = lift(&format!(
        "{INNER}extern \"C\" void k(float* out, const float* in, int nrows) {{ inner(out, in, nrows); }}"
    ))
    .expect("lifts (was Invalid(BarrierDivergent))");
    // the guard passes: the sum 1..=32; the guard stops the kernel: nothing written
    assert_eq!(run(&k, 1), vec![528.0, 0.0]);
    assert_eq!(run(&k, 0), vec![0.0, 0.0]);
}

#[test]
fn a_call_that_is_not_last_does_not_end_the_kernel_with_its_return() {
    // the guard `return` leaves only `inner`: the kernel's own `out[1] = 5` still runs when the guard stops the callee. The
    // inlined form is `if (row >= nrows) {} else { rest }` with a shuffle in the else arm. `row` is built from `blockIdx.x` and
    // `threadIdx.y`, which are the same for every lane of a subgroup when the launch states the subgroup size (ROADMAP U63), so
    // that arm is warp-uniform and the kernel lifts, and it must lift AS the else form (the guard stops only the callee), never
    // as an early kernel exit: the trailing `out[1] = 5` runs on both paths. (Before U63 the checker read `threadIdx.y` as
    // lane-divergent and refused this lift.)
    let src = format!(
        "{INNER}extern \"C\" void k(float* out, const float* in, int nrows) {{ inner(out, in, nrows); if (threadIdx.x == 0) out[1] = 5.0f; }}"
    );
    let k = lift(&src).expect("lifts as the else form since the guard is warp-uniform");
    assert_eq!(run(&k, 1), vec![528.0, 5.0], "the guard passes: the sum, then the kernel's own statement");
    assert_eq!(run(&k, 0), vec![0.0, 5.0], "the guard stops only the callee: the kernel's own statement still runs");
    // without a barrier in the callee the else form is fine and the trailing statement runs either way
    let k = lift(
        "static void inner(float* out, const float* in, int nrows) { if (nrows <= 0) return; if (threadIdx.x == 0) out[0] = in[0]; }\n\
         extern \"C\" void k(float* out, const float* in, int nrows) { inner(out, in, nrows); if (threadIdx.x == 0) out[1] = 5.0f; }",
    )
    .unwrap();
    assert_eq!(run(&k, 0), vec![0.0, 5.0]);
    assert_eq!(run(&k, 1), vec![1.0, 5.0]);
    // a helper's own last call is not the kernel's last: the guard in `leaf` leaves `leaf` (and `mid`), not the kernel
    let k = lift(
        "static void leaf(float* out, const float* in, int nrows) { if (nrows <= 0) return; if (threadIdx.x == 0) out[0] = in[0]; }
         static void mid(float* out, const float* in, int nrows) { leaf(out, in, nrows); }
         extern \"C\" void k(float* out, const float* in, int nrows) { mid(out, in, nrows); if (threadIdx.x == 0) out[1] = 5.0f; }",
    )
    .unwrap();
    assert_eq!(run(&k, 0), vec![0.0, 5.0]);
    assert_eq!(run(&k, 1), vec![1.0, 5.0]);
}

#[test]
fn a_call_inside_a_branch_or_a_loop_is_never_a_kernel_exit() {
    // the call is the LAST statement of its own block, but that block is an `if` / `for` body: the callee's guard `return`
    // leaves only the callee, and the kernel's later `out[1] = 5` still runs
    for wrapper in [
        "if (nrows < 100) { inner(out, in, nrows); }",
        "for (int r = 0; r < 1; ++r) { inner(out, in, nrows); }",
        "{ inner(out, in, nrows); }",
    ] {
        let k = lift(&format!(
            "static void inner(float* out, const float* in, int nrows) {{ if (nrows <= 0) return; if (threadIdx.x == 0) out[0] = in[0]; }}
             extern \"C\" void k(float* out, const float* in, int nrows) {{ {wrapper} if (threadIdx.x == 0) out[1] = 5.0f; }}"
        ))
        .unwrap();
        assert_eq!(run(&k, 0), vec![0.0, 5.0], "{wrapper}");
        assert_eq!(run(&k, 1), vec![1.0, 5.0], "{wrapper}");
    }
}
