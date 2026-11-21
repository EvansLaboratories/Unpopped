// SPDX-License-Identifier: MIT OR Apache-2.0
//! **U19 slice 1, the statement spine** (`docs/options-u19-cfamily-body-frontend.md`). Each lifted kernel is run on the oracle
//! and compared with an independent Rust reference on distinct data (a differential, not golden text); each decline names the
//! construct. The tests parse with `tree-sitter-cpp`, so the CUDA qualifiers (`__global__`) are Baracuda's leg, not this one.

use unpopped::kernel::KernelFn;
use unpopped::oracle::TypedBuffer;
use unpopped::oracle::kernel_eval::{KernelArg, run_kernel};
use unpopped_convert::{Frontend, KernelLiftError, lift_kernel};

fn parse(src: &str) -> Option<tree_sitter::Tree> {
    let mut p = tree_sitter::Parser::new();
    p.set_language(&tree_sitter_cpp::LANGUAGE.into()).ok()?;
    p.parse(src, None)
}

const C: Frontend = Frontend::new("c", "void", &[], "out", "in", parse);

fn lift(src: &str) -> Result<KernelFn, KernelLiftError> {
    lift_kernel(&C, src, "k")
}

fn run_f32(k: &KernelFn, bufs: &[&[f32]], scalars: &[KernelArg], out_len: usize) -> Vec<f64> {
    // parameters are (out, inputs..., scalars...) in each source below
    let mut args = vec![KernelArg::Buffer(TypedBuffer::from_f32(
        &[out_len as i64],
        &vec![0.0; out_len],
    ))];
    for b in bufs {
        args.push(KernelArg::Buffer(TypedBuffer::from_f32(
            &[b.len() as i64],
            b,
        )));
    }
    args.extend(scalars.iter().map(clone_arg));
    run_kernel(k, &mut args).expect("runs");
    let KernelArg::Buffer(o) = &args[0] else {
        panic!("out first")
    };
    o.to_f64_vec()
}

fn clone_arg(a: &KernelArg) -> KernelArg {
    match a {
        KernelArg::Int(v) => KernelArg::Int(*v),
        KernelArg::Float(v) => KernelArg::Float(*v),
        _ => panic!("scalars only"),
    }
}

const X: [f32; 6] = [3.5, -1.25, 0.5, 8.0, -7.75, 2.0];
const Y: [f32; 6] = [0.125, 4.0, -2.5, 1.5, 6.0, -0.75];

#[test]
fn a_counted_loop_with_a_scalar_parameter() {
    let k = lift("void k(float* out, const float* x, const float* y, float a, int n) { for (int i = 0; i < n; i++) { out[i] = a * x[i] + y[i]; } }").unwrap();
    let got = run_f32(
        &k,
        &[&X, &Y],
        &[KernelArg::Float(2.5), KernelArg::Int(6)],
        6,
    );
    let want: Vec<f64> = (0..6).map(|i| f64::from(2.5f32 * X[i] + Y[i])).collect();
    assert_eq!(got, want);
    // Control: a different `n` stores fewer elements, so the bound is read.
    let part = run_f32(
        &k,
        &[&X, &Y],
        &[KernelArg::Float(2.5), KernelArg::Int(3)],
        6,
    );
    assert_eq!(&part[3..], &[0.0, 0.0, 0.0]);
    assert_eq!(&part[..3], &want[..3]);
}

#[test]
fn branches_ternary_and_boolean_operators() {
    let src = "void k(float* out, const float* x, int n) { for (int i = 0; i < n; i++) { float v = x[i]; \
        if (v > 0.0f && v < 5.0f) { out[i] = v * 2.0f; } else if (!(v > 0.0f) || v == 8.0f) { out[i] = v > -5.0f ? -1.0f : -2.0f; } else { out[i] = 100.0f; } } }";
    let k = lift(src).unwrap();
    let got = run_f32(&k, &[&X], &[KernelArg::Int(6)], 6);
    let want: Vec<f64> = X
        .iter()
        .map(|&v| {
            if v > 0.0 && v < 5.0 {
                f64::from(v * 2.0)
            } else if v <= 0.0 || v == 8.0 {
                if v > -5.0 { -1.0 } else { -2.0 }
            } else {
                100.0
            }
        })
        .collect();
    assert_eq!(got, want);
    // every arm is taken by the data (3.5 first, -1.25 second, -7.75 the -2.0, 8.0 the `== 8`)
    for w in [-2.0, -1.0, 7.0] {
        assert!(want.contains(&w), "arm with {w} not exercised");
    }
}

#[test]
fn compound_assignment_to_an_element_and_to_a_local() {
    let k = lift("void k(float* out, const float* x, int n) { for (int i = 0; i < n; i += 2) { out[i] = 1.0f; out[i] += x[i]; out[i] *= 2.0f; float t = 0.5f; t -= x[i]; out[i] += t; } }").unwrap();
    let got = run_f32(&k, &[&X], &[KernelArg::Int(6)], 6);
    let want: Vec<f64> = (0..6)
        .map(|i| {
            if i % 2 == 0 {
                f64::from((1.0f32 + X[i]) * 2.0 + (0.5 - X[i]))
            } else {
                0.0
            }
        })
        .collect();
    assert_eq!(got, want);
}

#[test]
fn a_while_loop_that_reads_memory_is_reread_every_iteration() {
    let k = lift("void k(float* out, const float* x) { int i = 0; while (x[i] > 0.0f) { out[i] = x[i] + 1.0f; i++; } }").unwrap();
    let data = [2.0f32, 4.5, 1.0, -3.0, 9.0, 9.0];
    let got = run_f32(&k, &[&data], &[], 6);
    assert_eq!(got, vec![3.0, 5.5, 2.0, 0.0, 0.0, 0.0]);
}

#[test]
fn integer_kernels_promote_and_convert_as_c_does() {
    let src = "void k(int* out, const int* a, int n) { for (int i = 0; i < n; i++) { int v = a[i]; \
        out[i] = (v / 2) % 5 + (v << 2) - (v & 3) + (v > 2 ? 1 : 0); } }";
    let k = lift(src).unwrap();
    let a = [7i32, -9, 4, 13, 0, 21];
    let mut args = vec![
        KernelArg::Buffer(TypedBuffer::from_i32(&[6], &[0; 6])),
        KernelArg::Buffer(TypedBuffer::from_i32(&[6], &a)),
        KernelArg::Int(6),
    ];
    run_kernel(&k, &mut args).expect("runs");
    let KernelArg::Buffer(o) = &args[0] else {
        panic!()
    };
    let want: Vec<f64> = a
        .iter()
        .map(|&v| f64::from((v / 2) % 5 + (v << 2) - (v & 3) + i32::from(v > 2)))
        .collect();
    assert_eq!(o.to_f64_vec(), want);
}

#[test]
fn a_cast_to_float_and_back() {
    let k = lift(
        "void k(float* out, const int* a, int n) { for (int i = 0; i < n; i++) { out[i] = (float)a[i] * 0.5f + (float)(int)out[i]; } }",
    );
    // `out` is Write-only? no: it is non-const, so ReadWrite: reading it is legal
    let k = k.unwrap();
    let a = [3i32, -8, 5, 0, 11, 2];
    let mut args = vec![
        KernelArg::Buffer(TypedBuffer::from_f32(
            &[6],
            &[1.9, -2.9, 0.0, 7.0, 0.5, -0.5],
        )),
        KernelArg::Buffer(TypedBuffer::from_i32(&[6], &a)),
        KernelArg::Int(6),
    ];
    run_kernel(&k, &mut args).expect("runs");
    let KernelArg::Buffer(o) = &args[0] else {
        panic!()
    };
    let prior = [1.9f32, -2.9, 0.0, 7.0, 0.5, -0.5];
    let want: Vec<f64> = (0..6)
        .map(|i| f64::from(a[i] as f32 * 0.5 + (prior[i] as i32) as f32))
        .collect();
    assert_eq!(o.to_f64_vec(), want);
}

fn declined(src: &str) -> KernelLiftError {
    lift(src).expect_err("must decline")
}

fn not_yet(src: &str, needle: &str) {
    match declined(src) {
        KernelLiftError::NotYetLifted { construct, .. } => {
            assert!(construct.contains(needle), "`{construct}` lacks `{needle}`");
        }
        other => panic!("expected NotYetLifted({needle}), got {other:?}"),
    }
}

#[test]
fn constructs_outside_the_spine_decline_by_name() {
    not_yet(
        "void k(float* out, float x) { out[0] = frobnicate(x); }",
        "`frobnicate`",
    );
    not_yet("void k(float* out) { out[0] = 1.0; }", "F64 literal");
    not_yet(
        "void k(float* out, int n) { n = 3; out[0] = 1.0f; }",
        "parameter `n`",
    );
    not_yet(
        "void k(float* out, int n) { for (int i = 0; i < n; i++) { i = 2; } }",
        "loop counter",
    );
    not_yet(
        "void k(float* out, int n) { for (int i = 0; i < n; i++) { n = 2; } out[0] = 1.0f; }",
        "assigns",
    );
    not_yet("void k(float* out, long n) { out[0] = 1.0f; }", "`long`");
    not_yet("void k(float* out) { float* p = out; }", "pointer");
    not_yet(
        "void k(float* out) { float v; out[0] = 1.0f; }",
        "initializer",
    );
    not_yet("void k(float* out) { return 1; }", "return");
    not_yet(
        "void k(float* out, const float* x) { out[0] = x[0] > 0.0f ? x[1] : 0.0f; }",
        "?:",
    );
    not_yet(
        "void k(float* out, const float* x) { out[0] = x[0] > 0.0f && x[1] > 0.0f; }",
        "&&",
    );
    not_yet(
        "void k(float* out, int n) { for (int i = 0; i * 2 < n; i++) { out[i] = 1.0f; } }",
        "counted loop",
    );
}

#[test]
fn a_decline_is_the_whole_kernel_never_a_partial_one() {
    // the first statements lift; the third does not: nothing comes back
    let r =
        lift("void k(float* out, float x) { out[0] = x; out[1] = x + 1.0f; out[2] = frobnicate(x); }");
    assert!(
        matches!(r, Err(KernelLiftError::NotYetLifted { line: 1, .. })),
        "{r:?}"
    );
}

#[test]
fn the_irs_own_refusals_are_outside_ir_not_not_yet() {
    // integer division in a kernel whose compute type is a float: the IR's type rules refuse it
    let r = declined(
        "void k(float* out, const float* x, int n) { for (int i = 0; i < n / 2; i++) { out[i] = x[i]; } }",
    );
    assert!(matches!(r, KernelLiftError::OutsideIr { .. }), "{r:?}");
    // Control: the same division in an integer kernel lifts (see integer_kernels_promote_and_convert_as_c_does).
    assert!(lift("void k(int* out, const int* x, int n) { for (int i = 0; i < n / 2; i++) { out[i] = x[i]; } }").is_ok());
}

#[test]
fn c_remainder_truncates_toward_zero_and_is_total() {
    // `a % b`: sign of the dividend (asymmetric negative operands), and 0 for a zero divisor and for MIN % -1.
    let k = lift("void k(int* out, const int* a, const int* b, int n) { for (int i = 0; i < n; i++) { out[i] = a[i] % b[i]; } }").unwrap();
    let a = [7i32, -7, 7, -7, 9, i32::MIN, 5, -6];
    let b = [3i32, 3, -3, -3, 0, -1, 5, 4];
    let mut args = vec![
        KernelArg::Buffer(TypedBuffer::from_i32(&[8], &[99; 8])),
        KernelArg::Buffer(TypedBuffer::from_i32(&[8], &a)),
        KernelArg::Buffer(TypedBuffer::from_i32(&[8], &b)),
        KernelArg::Int(8),
    ];
    run_kernel(&k, &mut args).expect("runs");
    let KernelArg::Buffer(o) = &args[0] else {
        panic!()
    };
    let want: Vec<f64> = a
        .iter()
        .zip(&b)
        .map(|(&x, &y)| f64::from(x.checked_rem(y).unwrap_or(0)))
        .collect();
    assert_eq!(want, vec![1.0, -1.0, 1.0, -1.0, 0.0, 0.0, 0.0, -2.0]);
    assert_eq!(o.to_f64_vec(), want);
}

#[test]
fn remainder_on_floats_is_not_c() {
    let r = declined("void k(float* out, const float* a) { out[0] = a[0] % 2.0f; }");
    assert!(
        matches!(&r, KernelLiftError::OutsideIr { construct, .. } if construct.contains("floating-point")),
        "{r:?}"
    );
}

#[test]
fn a_load_in_a_loop_start_runs_once_before_the_loop() {
    let k =
        lift("void k(int* out, const int* a) { for (int i = a[0]; i < 5; i++) { out[i] = i; } }")
            .unwrap();
    let mut args = vec![
        KernelArg::Buffer(TypedBuffer::from_i32(&[6], &[0; 6])),
        KernelArg::Buffer(TypedBuffer::from_i32(&[1], &[2])),
    ];
    run_kernel(&k, &mut args).expect("runs");
    let KernelArg::Buffer(o) = &args[0] else {
        panic!()
    };
    assert_eq!(o.to_f64_vec(), vec![0.0, 0.0, 2.0, 3.0, 4.0, 0.0]);
}

#[test]
fn a_missing_function_and_a_syntax_error_are_distinct() {
    assert_eq!(
        lift_kernel(&C, "void other(float* out) { out[0] = 1.0f; }", "k"),
        Err(KernelLiftError::NoSuchFunction("k".into()))
    );
    assert!(matches!(
        lift_kernel(&C, "void k(float* out) { out[0] = ; }", "k"),
        Err(KernelLiftError::SyntaxError { line: 1 })
    ));
    assert_eq!(
        lift_kernel(&C, "int x;", "k"),
        Err(KernelLiftError::NotAKernel)
    );
}

#[test]
fn unary_minus_and_unsigned_conversions() {
    // float negate, integer negate (0 - x), and an unsigned operand meeting a signed one of the same width
    let k = lift("void k(float* out, const float* x, int s, unsigned int u) { out[0] = -x[0]; out[1] = (float)(-s); out[2] = (float)(s + u); }").unwrap();
    let got = run_f32(&k, &[&[2.5]], &[KernelArg::Int(-3), KernelArg::Int(1)], 3);
    // s + u is unsigned: -3 + 1 wraps to 4294967294
    assert_eq!(got, vec![-2.5, 3.0, f64::from(4_294_967_294u32 as f32)]);
}
