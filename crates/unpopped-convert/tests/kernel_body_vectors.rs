// SPDX-License-Identifier: MIT OR Apache-2.0
//! **U19 slice 4b: vector locals** (`float2`/`float4`/`int2`/`int4`/`uint2`/`uint4`/`double2`/`double4`) as **scalarized** locals (one scalar
//! local per lane: the oracle cannot run an IR packed local, `tests/kernel_packed.rs`, and scalarizing is exact) (`docs/options-u19-cfamily-body-frontend.md`): `make_<type>(...)`, whole-vector copies, member reads, and member
//! writes (the vector rebuilt with one lane replaced). Vectors in **memory** (`float4*` parameters, `((float4*)p)[i]`) are
//! not lifted: they decline by name.

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

#[test]
fn a_float4_is_built_copied_read_and_written_by_lane() {
    let k = lift(
        "void k(float* out, const float* a) { float4 v = make_float4(a[0], a[1], a[2], a[3]); float4 w = v; \
         v.x = v.y + 1.0f; v.w *= 2.0f; v.z -= a[0]; \
         out[0] = v.x; out[1] = v.y; out[2] = v.z; out[3] = v.w; out[4] = w.x; out[5] = w.w; }",
    )
    .unwrap();
    let a = [1.5f32, -2.0, 3.25, 0.75];
    let mut args = vec![
        KernelArg::Buffer(TypedBuffer::from_f32(&[6], &[0.0; 6])),
        KernelArg::Buffer(TypedBuffer::from_f32(&[4], &a)),
    ];
    run_kernel(&k, &mut args).expect("runs");
    let KernelArg::Buffer(o) = &args[0] else {
        panic!()
    };
    // v = (a0,a1,a2,a3) with x = a1 + 1, w = a3 * 2, z = a2 - a0; the copy `w` keeps the originals (value semantics)
    let want = [a[1] + 1.0, a[1], a[2] - a[0], a[3] * 2.0, a[0], a[3]];
    assert_eq!(
        o.to_f64_vec(),
        want.iter().map(|&v| f64::from(v)).collect::<Vec<_>>()
    );
}

#[test]
fn an_int2_in_an_integer_kernel_and_a_whole_assignment() {
    let k = lift(
        "void k(int* out, const int* a) { int2 p = make_int2(a[0], a[1]); int2 q = make_int2(7, 9); p = q; q.x = 100; \
         out[0] = p.x; out[1] = p.y; out[2] = q.x; out[3] = q.y; }",
    )
    .unwrap();
    let mut args = vec![
        KernelArg::Buffer(TypedBuffer::from_i32(&[4], &[0; 4])),
        KernelArg::Buffer(TypedBuffer::from_i32(&[2], &[1, 2])),
    ];
    run_kernel(&k, &mut args).expect("runs");
    let KernelArg::Buffer(o) = &args[0] else {
        panic!()
    };
    // `p = q` copies by value, so the later write to q.x does not reach p
    assert_eq!(o.to_f64_vec(), vec![7.0, 9.0, 100.0, 9.0]);
}

#[test]
fn a_whole_assignment_reads_the_old_lanes() {
    // `v = make_float2(v.y, v.x)` is a swap: both lanes are read before either is written.
    let k = lift("void k(float* out) { float2 v = make_float2(1.5f, -2.5f); v = make_float2(v.y, v.x); out[0] = v.x; out[1] = v.y; }").unwrap();
    let mut args = vec![KernelArg::Buffer(TypedBuffer::from_f32(&[2], &[0.0; 2]))];
    run_kernel(&k, &mut args).expect("runs");
    let KernelArg::Buffer(o) = &args[0] else {
        panic!()
    };
    assert_eq!(o.to_f64_vec(), vec![-2.5, 1.5]);
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
fn what_is_not_a_vector_local_declines_by_name() {
    not_yet(
        "void k(float* out) { float3 v = make_float3(1.0f, 2.0f, 3.0f); out[0] = v.x; }",
        "`float3`",
    );
    not_yet(
        "void k(float* out) { float2 v = make_float2(1.0f, 2.0f); out[0] = v.z; }",
        "member `z`",
    );
    not_yet(
        "void k(float* out) { float4 v = make_float4(1.0f, 2.0f); out[0] = v.x; }",
        "2 argument",
    );
    not_yet(
        "void k(float* out) { float2 v = make_int2(1, 2); out[0] = v.x; }",
        "neither",
    );
    not_yet(
        "void k(float* out) { float2 v = make_float2(1.0f, 2.0f); out[0] = v; }",
        "vector `v` used as a scalar",
    );
    not_yet(
        "void k(float* out) { float2 v = make_float2(1.0f, 2.0f); v += v; out[0] = v.x; }",
        "compound assignment",
    );
    not_yet("void k(float4* out) { out[0].x = 1.0f; }", "`float4`");
    // Control: the accepted spelling lifts.
    assert!(
        lift("void k(float* out) { float2 v = make_float2(1.0f, 2.0f); out[0] = v.y; }").is_ok()
    );
}
