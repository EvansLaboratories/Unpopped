// SPDX-License-Identifier: MIT OR Apache-2.0
//! **U19 slice 3: math calls and atomics** (`docs/options-u19-cfamily-body-frontend.md`). A call is lifted only through the
//! frontend's own intrinsic tables (a fast-math spelling is not in them, so it declines rather than lifts as the accurate op);
//! an atomic is one relaxed `AtomicRmw`/`AtomicCas` at CUDA's scope. Each lifted kernel runs on the oracle against a Rust
//! reference.

use unpopped::exec::LaunchShape;
use unpopped::kernel::KernelFn;
use unpopped::oracle::TypedBuffer;
use unpopped::oracle::kernel_eval::{KernelArg, run_kernel};
use unpopped::oracle::kernel_launch::run_launch;
use unpopped_convert::{Frontend, KernelLiftError, lift_kernel, lift_kernel_launched};

fn parse(src: &str) -> Option<tree_sitter::Tree> {
    let mut p = tree_sitter::Parser::new();
    p.set_language(&tree_sitter_cpp::LANGUAGE.into()).ok()?;
    p.parse(src, None)
}

const C: Frontend = Frontend::new("c", "void", &[], "out", "in", parse);

fn one_thread(src: &str) -> Result<KernelFn, KernelLiftError> {
    lift_kernel_launched(&C, src, "k", LaunchShape::new([1, 1, 1], [1, 1, 1]))
}

fn f32s(v: &[f32]) -> KernelArg {
    KernelArg::Buffer(TypedBuffer::from_f32(&[v.len() as i64], v))
}

fn i32s(v: &[i32]) -> KernelArg {
    KernelArg::Buffer(TypedBuffer::from_i32(&[v.len() as i64], v))
}

fn buf(a: &KernelArg) -> Vec<f64> {
    let KernelArg::Buffer(b) = a else {
        panic!("a buffer")
    };
    b.to_f64_vec()
}

#[test]
fn math_calls_lift_through_the_frontends_table_with_c_semantics() {
    let k = lift_kernel(
        &C,
        "void k(float* out, const float* a, const float* b, int n) { for (int i = 0; i < n; i++) { \
         out[i] = sqrtf(fabsf(a[i])) + fmaxf(a[i], b[i]) * 2.0f - fminf(a[i], b[i]) + floorf(a[i]) + ceilf(b[i]) + truncf(a[i]); } }",
        "k",
    )
    .unwrap();
    let a = [4.0f32, -2.5, f32::NAN, 9.75, -0.5, 6.25];
    let b = [1.5f32, -3.25, 2.0, f32::NAN, 0.75, 6.25];
    let mut args = vec![f32s(&[0.0; 6]), f32s(&a), f32s(&b), KernelArg::Int(6)];
    run_kernel(&k, &mut args).expect("runs");
    let want: Vec<f64> = (0..6)
        .map(|i| {
            let (x, y) = (a[i], b[i]);
            f64::from(x.abs().sqrt() + x.max(y) * 2.0 - x.min(y) + x.floor() + y.ceil() + x.trunc())
        })
        .collect();
    let got = buf(&args[0]);
    for i in 0..6 {
        assert!(
            (got[i].is_nan() && want[i].is_nan()) || got[i] == want[i],
            "element {i}: {} vs {}",
            got[i],
            want[i]
        );
    }
    // the NaN-suppressing max is exercised: a NaN operand gives the other one
    assert!(want.iter().any(|v| !v.is_nan()) && a[2].is_nan() && !f32::NAN.max(2.0).is_nan());
}

#[test]
fn a_fast_math_spelling_and_unknown_calls_decline_instead_of_lifting_as_the_accurate_op() {
    for (call, needle) in [
        ("__expf(a[0])", "`__expf`"),
        ("frobnicate(a[0])", "`frobnicate`"),
        ("sqrtf(a[0], a[1])", "2 argument"),
    ] {
        let src = format!("void k(float* out, const float* a) {{ out[0] = {call}; }}");
        match lift_kernel(&C, &src, "k") {
            Err(KernelLiftError::NotYetLifted { construct, .. }) => {
                assert!(construct.contains(needle), "`{construct}` lacks `{needle}`");
            }
            other => panic!("{call}: {other:?}"),
        }
    }
    // Control: the accurate spelling lifts.
    assert!(
        lift_kernel(
            &C,
            "void k(float* out, const float* a) { out[0] = sqrtf(a[0]); }",
            "k"
        )
        .is_ok()
    );
}

#[test]
fn an_integer_through_a_float_math_function_converts_as_c_does() {
    let k = lift_kernel(&C, "void k(float* out, int n) { out[0] = sqrtf(n); }", "k").unwrap();
    let mut args = vec![f32s(&[0.0]), KernelArg::Int(81)];
    run_kernel(&k, &mut args).expect("runs");
    assert_eq!(buf(&args[0]), vec![9.0]);
}

#[test]
fn atomics_return_the_old_value_and_apply_the_op() {
    let k = one_thread(
        "void k(int* out, int* c) { int a = atomicAdd(&c[0], 5); int b = atomicMax(&c[1], 7); int d = atomicExch(&c[2], 9); \
         int e = atomicCAS(&c[3], 4, 40); int f = atomicCAS(&c[4], 4, 40); atomicOr(&c[5], 6); atomicAnd(&c[6], 3); \
         out[0] = a; out[1] = b; out[2] = d; out[3] = e; out[4] = f; }",
    )
    .unwrap();
    let mut args = vec![i32s(&[0; 5]), i32s(&[10, 3, 8, 4, 5, 1, 7])];
    run_launch(&k, &mut args).expect("runs");
    assert_eq!(buf(&args[0]), vec![10.0, 3.0, 8.0, 4.0, 5.0]);
    // c[0]+5, max(3,7), exch 9, cas hit -> 40, cas miss stays 5, 1|6, 7&3
    assert_eq!(buf(&args[1]), vec![15.0, 7.0, 9.0, 40.0, 5.0, 7.0, 3.0]);
}

#[test]
fn every_thread_of_a_block_adds_into_one_counter() {
    let k = lift_kernel_launched(
        &C,
        "void k(int* c, const int* a) { atomicAdd(&c[0], a[threadIdx.x]); }",
        "k",
        LaunchShape::new([8, 1, 1], [1, 1, 1]),
    )
    .unwrap();
    let a = [3, -1, 4, 1, -5, 9, 2, 6];
    let mut args = vec![i32s(&[100]), i32s(&a)];
    run_launch(&k, &mut args).expect("runs");
    assert_eq!(
        buf(&args[0]),
        vec![100.0 + f64::from(a.iter().sum::<i32>())]
    );
}

#[test]
fn atomics_that_are_not_a_plain_address_decline() {
    for (call, needle) in [
        ("atomicAdd((unsigned*)c, 1u)", "through a pointer"),
        ("atomicInc(&c[0], 3)", "`atomicInc`"),
        ("atomicAdd(&c[0])", "1 argument"),
    ] {
        let src = format!("void k(int* c) {{ {call}; }}");
        match one_thread(&src) {
            Err(KernelLiftError::NotYetLifted { construct, .. }) => {
                assert!(construct.contains(needle), "`{construct}` lacks `{needle}`");
            }
            other => panic!("{call}: {other:?}"),
        }
    }
    assert!(one_thread("void k(int* c) { atomicAdd(&c[0], 1); }").is_ok());
    // since the census wall slice, a pointer that names an element of its own base is that element
    assert!(one_thread("void k(int* c) { atomicAdd(c + 1, 1); }").is_ok());
}

#[test]
fn an_atomic_in_a_conditional_arm_runs_only_when_that_arm_is_taken() {
    // since slice 5 a `?:` arm with effects lifts as `If (c) { arm } else { arm }`: the atomic is in the taken arm only
    let k =
        one_thread("void k(int* out, int* c, int n) { out[0] = n > 0 ? atomicAdd(&c[0], 1) : 0; }")
            .unwrap();
    for (n, want_out, want_c) in [(1, 5.0, 6.0), (0, 0.0, 5.0)] {
        let mut args = vec![i32s(&[0]), i32s(&[5]), KernelArg::Int(n)];
        run_launch(&k, &mut args).expect("runs");
        assert_eq!(
            (buf(&args[0]), buf(&args[1])),
            (vec![want_out], vec![want_c]),
            "n = {n}"
        );
    }
    // the `&&` operand keeps its guard: an effect there is not hoisted unconditionally
    let r = one_thread(
        "void k(int* out, int* c, int n) { out[0] = n > 0 && atomicAdd(&c[0], 1) > 0; }",
    );
    assert!(
        matches!(r, Err(KernelLiftError::NotYetLifted { .. })),
        "{r:?}"
    );
}
