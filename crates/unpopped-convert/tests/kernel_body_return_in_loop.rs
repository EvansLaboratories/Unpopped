// SPDX-License-Identifier: MIT OR Apache-2.0
//! **Census wall: a `return` inside a loop of an inlined function** (20 real instantiations, plus 3 `return` followed by
//! statements). The function lowers with a result local and a `done` flag: a `return` assigns the result, sets `done` and breaks
//! out of the open loops; what follows a statement that may have returned runs under `if (done == 0)`. Oracle runs against Rust
//! references (the same search written in Rust).

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

const IN: [f32; 8] = [3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0];

fn run(k: &KernelFn, t: i32) -> Vec<f64> {
    let mut args = vec![
        KernelArg::Buffer(TypedBuffer::from_f32(&[4], &[0.0; 4])),
        KernelArg::Buffer(TypedBuffer::from_f32(&[8], &IN)),
        KernelArg::Int(i128::from(t)),
    ];
    run_launch(k, &mut args).expect("runs");
    let KernelArg::Buffer(o) = &args[0] else {
        panic!()
    };
    o.to_f64_vec()
}

/// first index whose element exceeds `t`, or -1
fn first_above(t: i32) -> f64 {
    IN.iter()
        .position(|&v| v > t as f32)
        .map_or(-1.0, |p| p as f64)
}

#[test]
fn a_value_return_in_a_for_loop_stops_the_search_and_the_fallthrough_value_is_kept() {
    let k = lift(
        "static int find(const float* in, int t) { for (int i = 0; i < 8; ++i) { if (in[i] > (float)t) { return i; } } return -1; }\n\
         void k(float* out, const float* in, int t) { out[0] = (float)find(in, t); out[1] = 7.0f; }",
    )
    .expect("lifts (was a decline)");
    for t in [0, 3, 5, 8, 9] {
        let r = run(&k, t);
        assert_eq!(r[0], first_above(t), "t={t}");
        // the statement after the call still runs
        assert_eq!(r[1], 7.0, "t={t}");
    }
}

#[test]
fn a_return_in_a_nested_loop_stops_both_loops() {
    // returns the flattened index of the first (i, j) with in[i] + in[j] > t, scanning i then j
    let k = lift(
        "static int find2(const float* in, int t) { for (int i = 0; i < 4; ++i) { for (int j = 0; j < 4; ++j) { if (in[i] + in[j + 4] > (float)t) { return i * 10 + j; } } } return -1; }\n\
         void k(float* out, const float* in, int t) { out[0] = (float)find2(in, t); }",
    )
    .expect("lifts");
    for t in [0, 6, 9, 12, 15, 30] {
        let mut want = -1.0;
        'o: for i in 0..4 {
            for j in 0..4 {
                if IN[i] + IN[j + 4] > t as f32 {
                    want = (i * 10 + j) as f64;
                    break 'o;
                }
            }
        }
        assert_eq!(run(&k, t)[0], want, "t={t}");
    }
}

#[test]
fn a_while_loop_return_and_work_after_the_loop_runs_only_when_nothing_returned() {
    let k = lift(
        "static float scan(const float* in, int t) { int i = 0; while (i < 8) { if (in[i] > (float)t) { return in[i]; } i = i + 1; } float s = 0.0f; for (int j = 0; j < 8; ++j) { s += in[j]; } return -s; }\n\
         void k(float* out, const float* in, int t) { out[0] = scan(in, t); }",
    )
    .expect("lifts");
    for t in [0, 4, 8, 9] {
        let want = IN
            .iter()
            .find(|&&v| v > t as f32)
            .map_or(-IN.iter().sum::<f32>(), |&v| v);
        assert_eq!(run(&k, t)[0], f64::from(want), "t={t}");
    }
}

#[test]
fn a_return_followed_by_statements_and_a_void_return_in_a_loop() {
    // `if (c) { x = 1; } if (t > 3) { return 5; } x = 9; return x;`: the return is not last in its block
    let k = lift(
        "static int pick(int t) { int x = 2; if (t > 3) { return 5; } x = x + t; return x; }\n\
         static void put(float* out, const float* in, int t) { for (int i = 0; i < 8; ++i) { if (in[i] > (float)t) { out[2] = (float)i; return; } } out[2] = -1.0f; }\n\
         void k(float* out, const float* in, int t) { out[0] = (float)pick(t); put(out, in, t); out[3] = 1.0f; }",
    )
    .expect("lifts");
    for t in [0, 3, 4, 9] {
        let r = run(&k, t);
        assert_eq!(r[0], if t > 3 { 5.0 } else { f64::from(2 + t) }, "t={t}");
        assert_eq!(r[2], first_above(t), "t={t}");
        // the kernel's own statement after the void call still runs
        assert_eq!(r[3], 1.0, "t={t}");
    }
}

#[test]
fn a_return_whose_value_does_not_match_the_type_still_declines() {
    let r = lift(
        "static int f(const float* in) { for (int i = 0; i < 8; ++i) { if (in[i] > 1.0f) { return; } } return 1; }\n\
         void k(float* out, const float* in, int t) { out[0] = (float)f(in); }",
    );
    assert!(
        matches!(r, Err(KernelLiftError::NotYetLifted { .. })),
        "{r:?}"
    );
}
