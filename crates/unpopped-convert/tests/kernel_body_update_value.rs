// SPDX-License-Identifier: MIT OR Apache-2.0
//! **Census wall: `x++` / `++x` / `x--` / `--x` used as a value** (`c = bcast_coord[bi++];`, 12 real instantiations). The update is
//! hoisted before the statement and the value is a copy of the local taken before it (postfix) or after it (prefix). An update in
//! the right operand of `&&`/`||` declines (it runs only sometimes), and so does one on something that is not an integer local.
//! Oracle runs against Rust references.

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

const IN: [f32; 6] = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0];

fn run(k: &KernelFn, outs: usize) -> Vec<f64> {
    let mut args = vec![
        KernelArg::Buffer(TypedBuffer::from_f32(&[outs as i64], &vec![0.0; outs])),
        KernelArg::Buffer(TypedBuffer::from_f32(&[6], &IN)),
    ];
    run_launch(k, &mut args).expect("runs");
    let KernelArg::Buffer(o) = &args[0] else {
        panic!()
    };
    o.to_f64_vec()
}

#[test]
fn postfix_and_prefix_updates_as_index_values() {
    let k = lift(
        "void k(float* out, const float* in) { int bi = 0; int d = 5; \
           out[0] = in[bi++]; out[1] = in[bi++]; out[2] = in[++bi]; out[3] = in[d--]; out[4] = in[--d]; \
           out[5] = (float)bi; out[6] = (float)d; }",
    )
    .expect("lifts (was a decline)");
    // bi: 0 -> in[0], 1 -> in[1], then ++bi = 3 -> in[3]; d: in[5], d = 4, then --d = 3 -> in[3]; bi = 3, d = 3
    assert_eq!(
        run(&k, 7),
        vec![
            f64::from(IN[0]),
            f64::from(IN[1]),
            f64::from(IN[3]),
            f64::from(IN[5]),
            f64::from(IN[3]),
            3.0,
            3.0
        ]
    );
}

#[test]
fn an_update_in_a_loop_body_and_a_ternary_arm_runs_when_the_c_runs_it() {
    let k = lift(
        "void k(float* out, const float* in) { int i = 0; float s = 0.0f; \
           for (int j = 0; j < 3; ++j) { s += in[i++]; } out[0] = s; out[1] = (float)i; \
           int p = 0; int q = 0; float t = (p == 0) ? in[q++] : in[4]; out[2] = t; out[3] = (float)q; \
           int r = 1; float u = (r == 0) ? in[q++] : in[4]; out[4] = u; out[5] = (float)q; }",
    )
    .expect("lifts");
    // the loop adds in[0..3] and leaves i = 3; the taken arm bumps q to 1; the untaken arm leaves it at 1
    assert_eq!(
        run(&k, 6),
        vec![
            f64::from(IN[0] + IN[1] + IN[2]),
            3.0,
            f64::from(IN[0]),
            1.0,
            f64::from(IN[4]),
            1.0
        ]
    );
}

#[test]
fn an_update_that_may_not_run_or_is_not_an_integer_local_declines() {
    for src in [
        // the right operand of `&&` runs only when the left is true, and that of `||` only when it is false
        "void k(float* out, const float* in) { int i = 0; int a = 0; if (a > 0 && i++ > 0) { out[0] = 1.0f; } out[1] = (float)i; }",
        "void k(float* out, const float* in) { int i = 0; int a = 1; if (a > 0 || i++ > 0) { out[0] = 1.0f; } out[1] = (float)i; }",
        // a float is not an integer local
        "void k(float* out, const float* in) { float f = 1.0f; out[0] = f++; }",
        // a kernel parameter is read-only
        "void k(float* out, const float* in, int n) { out[0] = (float)(n++); }",
    ] {
        let r = lift(src);
        assert!(
            matches!(r, Err(KernelLiftError::NotYetLifted { .. })),
            "{src}: {r:?}"
        );
    }
}
