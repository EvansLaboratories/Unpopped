// SPDX-License-Identifier: MIT OR Apache-2.0
//! **U19 slice 4c-4: symbolic pointers** (pm-kernels ruling Q6; `docs/options-u19-slice5-decode-census.md`). A pointer is a base
//! (buffer or allocation), a pointee type and a byte offset resolved in the walker; every dereference is a raw place. Source
//! bytes are chosen so a wrong offset, width or scale shows in the value read. The test parser blanks `__shared__` and
//! `__device__`/`__forceinline__` to same-length spaces.

use unpopped::exec::LaunchShape;
use unpopped::kernel::{KernelFn, Place, Stmt};
use unpopped::oracle::TypedBuffer;
use unpopped::oracle::kernel_eval::{KernelArg, run_kernel};
use unpopped::oracle::kernel_launch::run_launch;
use unpopped_convert::{Frontend, KernelLiftError, lift_kernel, lift_kernel_launched};

fn parse(src: &str) -> Option<tree_sitter::Tree> {
    let mut blanked = src.to_string();
    for q in ["__shared__", "__device__", "__forceinline__"] {
        blanked = blanked.replace(q, &" ".repeat(q.len()));
    }
    let mut p = tree_sitter::Parser::new();
    p.set_language(&tree_sitter_cpp::LANGUAGE.into()).ok()?;
    p.parse(blanked, None)
}

const C: Frontend = Frontend::new("c", "void", &[], "out", "in", parse);

fn lift(src: &str) -> Result<KernelFn, KernelLiftError> {
    lift_kernel(&C, src, "k")
}

fn run(k: &KernelFn, out: usize, a: &[f32]) -> Vec<f64> {
    let mut args = vec![KernelArg::Buffer(TypedBuffer::from_f32(
        &[out as i64],
        &vec![0.0; out],
    ))];
    if !a.is_empty() {
        args.push(KernelArg::Buffer(TypedBuffer::from_f32(
            &[a.len() as i64],
            a,
        )));
    }
    run_kernel(k, &mut args).expect("runs");
    let KernelArg::Buffer(o) = &args[0] else {
        panic!()
    };
    o.to_f64_vec()
}

const A: [f32; 8] = [1.5, -2.0, 3.25, 0.5, 7.0, -9.5, 2.75, 100.0];

#[test]
fn pointer_arithmetic_and_dereference_scale_by_the_pointee() {
    let k = lift(
        "void k(float* out, const float* a) { const float* q = a + 2; const float* r = &a[5]; \
         out[0] = q[1]; out[1] = *q; out[2] = *(q + 3); out[3] = r[2]; out[4] = (r - 1)[0]; out[5] = (2 + q)[0]; }",
    )
    .unwrap();
    assert_eq!(
        run(&k, 6, &A),
        vec![A[3], A[2], A[5], A[7], A[4], A[4]]
            .into_iter()
            .map(f64::from)
            .collect::<Vec<_>>()
    );
}

#[test]
fn a_cast_reads_the_bytes_as_another_type() {
    // 1.5f = 0x3FC00000: read as u32 it is 1069547520; as u16 pieces (little endian) 0 then 16320
    let k = lift(
        "void k(float* out, const float* a) { const unsigned int* u = (const unsigned int*)a; const unsigned short* h = (const unsigned short*)a; \
         out[0] = (float)u[0]; out[1] = (float)h[0]; out[2] = (float)h[1]; }",
    )
    .unwrap();
    assert_eq!(
        run(&k, 3, &A),
        vec![f64::from(1_069_547_520u32 as f32), 0.0, 16320.0]
    );
}

#[test]
fn a_pointer_is_fixed_when_it_is_declared() {
    // `q` points at a[1]; changing `i` afterwards must not move it
    let k = lift(
        "void k(float* out, const float* a) { int i = 1; const float* q = a + i; i = 3; out[0] = q[0]; out[1] = a[i]; }",
    )
    .unwrap();
    assert_eq!(run(&k, 2, &A), vec![f64::from(A[1]), f64::from(A[3])]);
}

#[test]
fn a_cast_pointer_into_a_shared_array_addresses_its_bytes() {
    let k = lift_kernel_launched(
        &C,
        "void k(float* out) { __shared__ unsigned char raw[16]; float* f = (float*)raw; f[1] = 3.5f; out[0] = (float)raw[7]; out[1] = (float)raw[4]; out[2] = f[1]; }",
        "k",
        LaunchShape::new([1, 1, 1], [1, 1, 1]),
    )
    .unwrap();
    // 3.5f = 0x40600000: little-endian bytes 00 00 60 40 at 4..8
    let mut args = vec![KernelArg::Buffer(TypedBuffer::from_f32(&[3], &[0.0; 3]))];
    run_launch(&k, &mut args).expect("runs");
    let KernelArg::Buffer(o) = &args[0] else {
        panic!()
    };
    assert_eq!(o.to_f64_vec(), vec![64.0, 0.0, 3.5]);
}

#[test]
fn pointer_parameters_of_inlined_functions_alias_the_callers_memory() {
    let k = lift(
        "__device__ void fill(float* dst, float v, int n) { for (int i = 0; i < n; i++) { dst[i] = v; } }\n\
         __device__ float total(const float* p, int n) { float s = 0.0f; for (int i = 0; i < n; i++) { s += p[i]; } return s; }\n\
         void k(float* out, const float* a) { fill(out + 2, 7.0f, 3); out[0] = total(a + 4, 3); out[1] = total(a, 2); }",
    )
    .unwrap();
    let want = [A[4] + A[5] + A[6], A[0] + A[1], 7.0, 7.0, 7.0];
    assert_eq!(
        run(&k, 5, &A),
        want.iter().map(|&v| f64::from(v)).collect::<Vec<_>>()
    );
}

#[test]
fn ldg_takes_a_pointer_expression_and_a_plain_element_keeps_its_typed_place() {
    let k = lift("void k(float* out, const float* a) { out[0] = __ldg(a + 6); out[1] = a[1]; }")
        .unwrap();
    assert_eq!(run(&k, 2, &A), vec![f64::from(A[6]), f64::from(A[1])]);
    // a plain `a[i]` is still a Logical place; only pointer forms are Raw
    let places: Vec<&Place> = k
        .body
        .0
        .iter()
        .filter_map(|s| match s {
            Stmt::Load { place, .. } => Some(place),
            _ => None,
        })
        .collect();
    assert!(matches!(places[0], Place::Raw { .. }), "{places:?}");
    assert!(matches!(places[1], Place::Logical { .. }), "{places:?}");
}

fn not_yet(src: &str, needle: &str) {
    match lift(src) {
        Err(KernelLiftError::NotYetLifted { construct, .. }) => {
            assert!(construct.contains(needle), "`{construct}` lacks `{needle}`");
        }
        other => panic!("expected NotYetLifted({needle}), got {other:?}"),
    }
}

#[test]
fn a_pointer_that_must_survive_as_a_value_declines() {
    let h = "void k(float* out, const float* a) {";
    not_yet(
        &format!("{h} const float* q = a; q = a + 1; out[0] = q[0]; }}"),
        "not a scalar local",
    );
    not_yet(
        &format!("{h} const float* q = a; out[0] = q; }}"),
        "array or pointer `q`",
    );
    not_yet(
        &format!("{h} const float* q = a + 1; const float* r = a; out[0] = (float)(q - r); }}"),
        "used as a value",
    );
    not_yet(
        &format!("{h} const float* q = (const float*)3; out[0] = q[0]; }}"),
        "not a pointer",
    );
    not_yet(
        &format!("{h} const int* q = a; out[0] = 1.0f; }}"),
        "implicit pointer conversion",
    );
    not_yet(
        "void k(int* c) { int* q = c + 1; atomicAdd(q, 1); }",
        "through a pointer",
    );
    not_yet(
        "void k(int* c) { atomicAdd(&(c + 1)[0], 1); }",
        "through a pointer",
    );
    // Control: the plain spelling lifts.
    assert!(lift(&format!("{h} const float* q = a + 1; out[0] = q[0]; }}")).is_ok());
}
