// SPDX-License-Identifier: MIT OR Apache-2.0
//! **Census wall: `do { body } while (c);`** (10 real instantiations once the `switch` wall was gone). It lowers as
//! `while (1) { body; if (!c) break; }`: the body runs before the first test, a `break` leaves the loop, and a `continue` in the body
//! declines by name (it would skip the test at the end). The condition's loads run at the end of every pass. Oracle runs against
//! Rust references.

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

const IN: [f32; 6] = [3.0, 2.0, 1.0, 0.0, 5.0, 5.0];

fn run(k: &KernelFn, n: i32, outs: usize) -> Vec<f64> {
    let mut args = vec![
        KernelArg::Buffer(TypedBuffer::from_f32(&[outs as i64], &vec![0.0; outs])),
        KernelArg::Buffer(TypedBuffer::from_f32(&[6], &IN)),
        KernelArg::Int(i128::from(n)),
    ];
    run_launch(k, &mut args).expect("runs");
    let KernelArg::Buffer(o) = &args[0] else {
        panic!()
    };
    o.to_f64_vec()
}

#[test]
fn the_body_runs_before_the_first_test_and_the_test_ends_the_loop() {
    let k = lift(
        "void k(float* out, const float* in, int n) { int c = 0; do { c = c + 1; } while (c < n); out[0] = (float)c; \
           int d = 0; int s = 0; do { s = s + d; d++; } while (d < 4); out[1] = (float)s; out[2] = (float)d; }",
    )
    .expect("lifts (was a decline)");
    for n in [-2, 0, 1, 5] {
        // the do loop always runs once: n <= 1 leaves c at 1
        let want = f64::from(n.max(1));
        assert_eq!(run(&k, n, 3), vec![want, 6.0, 4.0], "n={n}");
    }
}

#[test]
fn a_condition_that_reads_memory_is_read_every_pass_and_break_leaves_the_loop() {
    // walks the input while the element (as an int) is nonzero: 3, 2, 1 then 0 stops it
    let k = lift(
        "void k(float* out, const float* in, int n) { int i = 0; float s = 0.0f; \
           do { s += in[i]; i = i + 1; } while ((int)in[i] != 0); out[0] = s; out[1] = (float)i; \
           int j = 0; int hits = 0; do { j = j + 1; if (j == n) { break; } hits = hits + 1; } while (j < 100); out[2] = (float)hits; out[3] = (float)j; }",
    )
    .expect("lifts");
    let r = run(&k, 4, 4);
    assert_eq!(r, vec![6.0, 3.0, 3.0, 4.0]);
}

#[test]
fn a_return_in_a_do_loop_of_an_inlined_function_uses_the_done_flag() {
    let k = lift(
        "static int first_zero(const float* in) { int i = 0; do { if ((int)in[i] == 0) { return i; } i = i + 1; } while (i < 6); return -1; }\n\
         void k(float* out, const float* in, int n) { out[0] = (float)first_zero(in); }",
    )
    .expect("lifts");
    assert_eq!(run(&k, 0, 1), vec![3.0]);
}

#[test]
fn a_continue_in_the_body_declines_by_name() {
    let r = lift(
        "void k(float* out, const float* in, int n) { int i = 0; do { i = i + 1; if (i == 2) { continue; } } while (i < 5); out[0] = (float)i; }",
    );
    match r {
        Err(KernelLiftError::NotYetLifted { construct, .. }) => {
            assert!(construct.contains("`continue` in a `do`"), "{construct}");
        }
        other => panic!("expected a decline, got {other:?}"),
    }
}
