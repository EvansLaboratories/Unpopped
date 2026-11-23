// SPDX-License-Identifier: MIT OR Apache-2.0
//! **Counted loops that count down** (`for (int d = rank - 1; d >= 0; --d)`, found by the struct-array slice: it is the loop
//! Baracuda's strided kernels run over their `shape.v[d]` arrays). `i > b` and `i >= b` with `--` or `-= s` lower to a `For`
//! with a negative step (the IR's limit is exclusive, so `>= b` stops above `b - 1`). An unsigned counter declines: in C,
//! `i >= 0` on an unsigned never ends.

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

/// The digits a loop visits, as a number: `acc = acc * 10 + i`.
fn digits(head: &str, n: i128) -> f64 {
    let k = lift(&format!(
        "void k(float* out, int n) {{ int acc = 0; {head} {{ acc = acc * 10 + i; }} out[0] = (float)acc; }}"
    ))
    .unwrap();
    let mut args = vec![
        KernelArg::Buffer(TypedBuffer::from_f32(&[1], &[0.0])),
        KernelArg::Int(n),
    ];
    run_launch(&k, &mut args).expect("runs");
    let KernelArg::Buffer(o) = &args[0] else {
        panic!()
    };
    o.to_f64_vec()[0]
}

#[test]
fn a_descending_loop_visits_the_counter_from_the_start_down_to_the_limit() {
    assert_eq!(digits("for (int i = 4; i >= 0; --i)", 0), 43210.0);
    assert_eq!(digits("for (int i = 5; i >= 2; --i)", 0), 5432.0); // `>= 2` includes 2
    assert_eq!(digits("for (int i = 5; i > 2; --i)", 0), 543.0); // `> 2` stops before it
    assert_eq!(
        digits("for (int i = 7; i > 0; i -= 2)", 0),
        753.0 * 10.0 + 1.0
    ); // 7, 5, 3, 1
}

#[test]
fn a_runtime_start_and_an_empty_descending_loop() {
    let head = "for (int i = n - 1; i >= 0; --i)";
    assert_eq!(digits(head, 3), 210.0);
    assert_eq!(digits(head, 1), 0.0);
    assert_eq!(digits(head, 0), 0.0, "n = 0 runs no iteration");
    // control: the ascending form is unchanged
    assert_eq!(digits("for (int i = 0; i < n; ++i)", 4), 123.0);
}

#[test]
fn an_unsigned_counter_never_ends_in_c_and_is_not_run_as_a_finite_loop() {
    // `i >= 0` is always true of an unsigned: the counter wraps to 4294967295. A constant-trip unroller that ran it as the four
    // trips 3, 2, 1, 0 would lift a loop C never leaves (found 2026-11-22 by this test); it now wraps and hits the trip limit.
    let r = lift(
        "void k(float* out) { int a = 0; for (unsigned int i = 3; i >= 0; --i) { a += 1; } out[0] = (float)a; }",
    );
    assert!(
        matches!(&r, Err(KernelLiftError::NotYetLifted { construct, .. }) if construct.contains("more than 64 trips")),
        "{r:?}"
    );
    // a bound outside the counter's range is compared after C's conversion, which is not modelled
    let r = lift(
        "void k(float* out) { int a = 0; for (unsigned int i = 3; i > -1; --i) { a += 1; } out[0] = (float)a; }",
    );
    assert!(
        matches!(&r, Err(KernelLiftError::NotYetLifted { construct, .. }) if construct.contains("outside the range")),
        "{r:?}"
    );
    // control: a signed counter ends at the limit
    assert_eq!(digits("for (int i = 3; i >= 0; --i)", 0), 3210.0);
}

#[test]
fn a_mismatched_update_still_declines() {
    for body in [
        "for (int i = 3; i >= 0; ++i)",
        "for (int i = 0; i < 3; --i)",
    ] {
        let r = lift(&format!(
            "void k(float* out) {{ int a = 0; {body} {{ a += 1; }} out[0] = (float)a; }}"
        ));
        assert!(
            matches!(&r, Err(KernelLiftError::NotYetLifted { construct, .. })
                if construct.contains("counted loop") || construct.contains("more than 64 trips")),
            "{body}: {r:?}"
        );
    }
}
