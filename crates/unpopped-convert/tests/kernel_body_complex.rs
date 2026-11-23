// SPDX-License-Identifier: MIT OR Apache-2.0
//! **Census wall: `cuComplex.h`** (`cuFloatComplex`, `cuDoubleComplex`, `make_cuFloatComplex`, `cuCaddf`, `cuCsubf`, `cuCmulf`, `cuConjf`,
//! `cuCrealf`, `cuCimagf` and the `double` forms: 12 real instantiations). The header is not in the census corpus, so the names are
//! builtins pinned to the text of CUDA 13.3's `include/cuComplex.h` (formulas and line numbers in `kernel_body/complex.rs`). The
//! typedefs are the vector types `float2`/`double2`. `cuCmulf` is `((x.x * y.x) - (x.y * y.y), (x.x * y.y) + (x.y * y.x))` with NO
//! fused multiply-add: the oracle evaluates each `Mul` and `Add` on its own, and the first test below has values that a fused
//! evaluation would get visibly wrong. `NaN` and infinity follow the formula. `cuCdivf` and `cuCabsf` decline by name. Oracle runs
//! against Rust references written from the header's formulas.

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

fn run_f32(k: &KernelFn, input: &[f32], outs: usize) -> Vec<f64> {
    let mut args = vec![
        KernelArg::Buffer(TypedBuffer::from_f32(&[outs as i64], &vec![0.0; outs])),
        KernelArg::Buffer(TypedBuffer::from_f32(&[input.len() as i64], input)),
    ];
    run_launch(k, &mut args).expect("runs");
    let KernelArg::Buffer(o) = &args[0] else {
        panic!()
    };
    o.to_f64_vec()
}

fn declines(src: &str, needle: &str) {
    match lift(src) {
        Err(KernelLiftError::NotYetLifted { construct, .. }) => {
            assert!(construct.contains(needle), "`{construct}` lacks `{needle}`");
        }
        other => panic!("{src}: expected a decline, got {other:?}"),
    }
}

const DECLS: &str = "cuFloatComplex a = make_cuFloatComplex(in[0], in[1]); cuFloatComplex b = make_cuFloatComplex(in[2], in[3]);";

#[test]
fn add_sub_mul_conj_real_imag_follow_the_header_formulas() {
    let k = lift(&format!(
        "void k(float* out, const float* in) {{ {DECLS} \
         cuFloatComplex s = cuCaddf(a, b); cuFloatComplex d = cuCsubf(a, b); cuFloatComplex p = cuCmulf(a, b); cuFloatComplex c = cuConjf(a); \
         out[0] = cuCrealf(s); out[1] = cuCimagf(s); out[2] = cuCrealf(d); out[3] = cuCimagf(d); \
         out[4] = cuCrealf(p); out[5] = cuCimagf(p); out[6] = cuCrealf(c); out[7] = cuCimagf(c); \
         cuComplex n = cuCmulf(cuCaddf(a, b), cuConjf(b)); out[8] = cuCrealf(n); out[9] = cuCimagf(n); \
         cuFloatComplex z = a; z = cuCmulf(z, b); out[10] = cuCrealf(z); out[11] = z.y; }}"
    ))
    .expect("lifts (was a decline: the type `cuFloatComplex`)");
    // a = (1, 2), b = (3, 4)
    let r = run_f32(&k, &[1.0, 2.0, 3.0, 4.0], 12);
    assert_eq!(
        r,
        vec![
            4.0, 6.0, // a + b
            -2.0, -2.0, // a - b
            -5.0, 10.0, // (1*3 - 2*4, 1*4 + 2*3)
            1.0, -2.0, // conj a
            36.0, 2.0, // (4,6) * (3,-4) = (12 + 24, -16 + 18)
            -5.0, 10.0, // z = a * b
        ]
    );
}

#[test]
fn the_multiply_is_unfused_two_roundings_where_a_fused_one_would_give_one() {
    // x = 1 + 2^-12: x * x = 1 + 2^-11 + 2^-24 needs 25 bits, so an f32 multiply rounds it (a tie, to even).
    // real = x*x - x*x: two equal roundings cancel to exactly 0.0; a fused (x*x) - round(x*x) would be 2^-24.
    // imag = x*x + x*x = 2 * round(x*x); fused fma(x, x, round(x*x)) would add the unrounded product once.
    let x = 1.0f32 + 2f32.powi(-12);
    let k = lift(&format!(
        "void k(float* out, const float* in) {{ {DECLS} cuFloatComplex p = cuCmulf(a, b); out[0] = cuCrealf(p); out[1] = cuCimagf(p); }}"
    ))
    .expect("lifts");
    let r = run_f32(&k, &[x, x, x, x], 2);
    let rounded = x * x;
    assert_eq!(r[0], 0.0, "unfused: round(x*x) - round(x*x)");
    assert_eq!(r[1], f64::from(rounded + rounded));
    // the fused values, to show the test is not vacuous
    let exact = f64::from(x) * f64::from(x);
    assert_ne!(exact - f64::from(rounded), 0.0);
}

#[test]
fn infinity_and_nan_follow_the_formula_not_a_complex_convention() {
    let k = lift(&format!(
        "void k(float* out, const float* in) {{ {DECLS} cuFloatComplex p = cuCmulf(a, b); out[0] = cuCrealf(p); out[1] = cuCimagf(p); }}"
    ))
    .expect("lifts");
    // (inf, 0) * (0, inf): real = inf*0 - 0*inf = NaN - NaN = NaN; imag = inf*inf + 0*0 = inf
    let r = run_f32(&k, &[f32::INFINITY, 0.0, 0.0, f32::INFINITY], 2);
    assert!(r[0].is_nan(), "{r:?}");
    assert_eq!(r[1], f64::INFINITY);
}

#[test]
fn the_double_forms_compute_in_double() {
    let k = lift(
        "void k(double* out, const double* in) { cuDoubleComplex a = make_cuDoubleComplex(in[0], in[1]); \
         cuDoubleComplex b = make_cuDoubleComplex(in[2], in[3]); cuDoubleComplex p = cuCmul(a, b); cuDoubleComplex s = cuCadd(a, cuConj(b)); \
         out[0] = cuCreal(p); out[1] = cuCimag(p); out[2] = cuCreal(s); out[3] = cuCimag(s); }",
    )
    .expect("lifts");
    let mut args = vec![
        KernelArg::Buffer(TypedBuffer::from_f64(&[4], &[0.0; 4])),
        KernelArg::Buffer(TypedBuffer::from_f64(&[4], &[1.5, -2.0, 0.25, 4.0])),
    ];
    run_launch(&k, &mut args).expect("runs");
    let KernelArg::Buffer(o) = &args[0] else {
        panic!()
    };
    // (1.5, -2) * (0.25, 4) = (0.375 + 8, 6 - 0.5) = (8.375, 5.5); (1.5, -2) + (0.25, -4) = (1.75, -6)
    assert_eq!(o.to_f64_vec(), vec![8.375, 5.5, 1.75, -6.0]);
}

#[test]
fn division_and_magnitude_decline_by_name_and_a_precision_mismatch_does_too() {
    declines(
        &format!(
            "void k(float* out, const float* in) {{ {DECLS} cuFloatComplex q = cuCdivf(a, b); out[0] = cuCrealf(q); }}"
        ),
        "cuCdivf",
    );
    declines(
        &format!("void k(float* out, const float* in) {{ {DECLS} out[0] = cuCabsf(a); }}"),
        "cuCabsf",
    );
    // a `cuCmul` (double) result where a float2 is wanted
    declines(
        &format!(
            "void k(float* out, const float* in) {{ {DECLS} cuFloatComplex p = cuCmul(a, b); out[0] = cuCrealf(p); }}"
        ),
        "lanes",
    );
}
