// SPDX-License-Identifier: MIT OR Apache-2.0
//! **Census wall: `switch`** (the nf4 codebook, a `switch (idx & 0xF) { case 0: return ..; ... default: return ..; }` in an inlined
//! function: 11 real instantiations). A `switch` whose cases each end in `break`/`return`/`continue` and whose `default` is last is
//! rewritten to an `if` chain before the walk: the selector held in one local, stacked labels sharing a body, the trailing `break` dropped. A
//! fall-through, a `break` from the middle of a case, and a `default` that is not last still decline by name. Oracle runs against
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

fn run(k: &KernelFn, n: i32, outs: usize) -> Vec<f64> {
    let mut args = vec![
        KernelArg::Buffer(TypedBuffer::from_f32(&[outs as i64], &vec![0.0; outs])),
        KernelArg::Int(i128::from(n)),
    ];
    run_launch(k, &mut args).expect("runs");
    let KernelArg::Buffer(o) = &args[0] else {
        panic!()
    };
    o.to_f64_vec()
}

fn declines(src: &str) {
    let r = lift(src);
    assert!(
        matches!(r, Err(KernelLiftError::NotYetLifted { .. })),
        "{src}: {r:?}"
    );
}

#[test]
fn a_codebook_switch_of_returns_in_an_inlined_function() {
    let k = lift(
        "static float code(int i) { switch (i & 3) { case 0: return 1.5f; case 1: return -2.0f; case 2: return 4.0f; default: return 0.25f; } }\n\
         void k(float* out, int n) { for (int j = 0; j < 6; ++j) { out[j] = code(n + j); } }",
    )
    .expect("lifts (was a decline)");
    let table = [1.5, -2.0, 4.0, 0.25];
    for n in [0, 1, 7] {
        let want: Vec<f64> = (0..6).map(|j| table[((n + j) & 3) as usize]).collect();
        assert_eq!(run(&k, n, 6), want, "n={n}");
    }
}

#[test]
fn breaks_stacked_labels_and_a_default_in_the_kernel_body() {
    let k = lift(
        "void k(float* out, int n) { float acc = 0.0f; \
           switch (n * 2 + 1) { case 1: case 3: acc = 10.0f; break; case 5: { acc = 20.0f; acc += 1.0f; break; } default: acc = -1.0f; } \
           out[0] = acc; \
           float z = 7.0f; switch (n) { case 0: z = 1.0f; break; case 1: z = 2.0f; break; } out[1] = z; }",
    )
    .expect("lifts");
    let a = |n: i32| match n * 2 + 1 {
        1 | 3 => 10.0,
        5 => 21.0,
        _ => -1.0,
    };
    let z = |n: i32| match n {
        0 => 1.0,
        1 => 2.0,
        _ => 7.0,
    };
    for n in 0..5 {
        assert_eq!(run(&k, n, 2), vec![a(n), z(n)], "n={n}");
    }
}

#[test]
fn a_continue_in_a_case_reaches_the_enclosing_loop() {
    let k = lift(
        "void k(float* out, int n) { float s = 0.0f; int c = 0; \
           for (int i = 0; i < 6; ++i) { c = c + 1; switch (i % 3) { case 0: continue; case 1: s += 1.0f; break; default: s += 10.0f; } s += 100.0f; } \
           out[0] = s; out[1] = (float)c; }",
    )
    .expect("lifts");
    // i = 0, 3: continue (no 100); i = 1, 4: +1 +100; i = 2, 5: +10 +100
    assert_eq!(run(&k, 0, 2), vec![2.0 * 101.0 + 2.0 * 110.0, 6.0]);
}

#[test]
fn what_is_not_a_clean_switch_still_declines_by_name() {
    // fall-through into the next group
    declines(
        "void k(float* out, int n) { float x = 0.0f; switch (n) { case 0: x = 1.0f; case 1: x = 2.0f; break; } out[0] = x; }",
    );
    // a `break` from the middle of a case
    declines(
        "void k(float* out, int n) { float x = 0.0f; switch (n) { case 0: if (n > 3) { break; } x = 1.0f; break; default: x = 2.0f; } out[0] = x; }",
    );
    // a `default` that is not last
    declines(
        "void k(float* out, int n) { float x = 0.0f; switch (n) { default: x = 2.0f; break; case 0: x = 1.0f; break; } out[0] = x; }",
    );
}
