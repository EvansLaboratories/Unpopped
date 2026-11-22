// SPDX-License-Identifier: MIT OR Apache-2.0
//! **U19 slice 4c-1: local arrays** (`float v[8];`, `alignas(16) float v[8];`) as `Private` allocations, one copy per thread
//! (`docs/options-u19-slice5-decode-census.md`). The oracle runs a launched kernel; each thread must see its own array.

use unpopped::exec::LaunchShape;
use unpopped::kernel::KernelFn;
use unpopped::oracle::TypedBuffer;
use unpopped::oracle::kernel_eval::KernelArg;
use unpopped::oracle::kernel_launch::run_launch;
use unpopped_convert::{Frontend, KernelLiftError, lift_kernel, lift_kernel_launched};

fn parse(src: &str) -> Option<tree_sitter::Tree> {
    let mut p = tree_sitter::Parser::new();
    p.set_language(&tree_sitter_cpp::LANGUAGE.into()).ok()?;
    p.parse(src, None)
}

const C: Frontend = Frontend::new("c", "void", &[], "out", "in", parse);

fn launched(src: &str, block: u32) -> Result<KernelFn, KernelLiftError> {
    lift_kernel_launched(&C, src, "k", LaunchShape::new([block, 1, 1], [1, 1, 1]))
}

fn f32s(v: &[f32]) -> KernelArg {
    KernelArg::Buffer(TypedBuffer::from_f32(&[v.len() as i64], v))
}

#[test]
fn a_local_array_is_filled_and_summed() {
    let k = launched(
        "void k(float* out, const float* a) { alignas(16) float v[8]; float w[4]; \
         for (int i = 0; i < 8; i++) { v[i] = a[i] * 2.0f; } \
         for (int j = 0; j < 4; j++) { w[j] = v[j] + v[j + 4]; } \
         float s = 0.0f; for (int j = 0; j < 4; j++) { s += w[j]; } out[0] = s; out[1] = w[3]; }",
        1,
    )
    .unwrap();
    let a = [1.5f32, -2.0, 3.25, 0.5, 4.0, -1.0, 2.5, 7.0];
    let mut args = vec![f32s(&[0.0; 2]), f32s(&a)];
    run_launch(&k, &mut args).expect("runs");
    let v: Vec<f32> = a.iter().map(|x| x * 2.0).collect();
    let w: Vec<f32> = (0..4).map(|j| v[j] + v[j + 4]).collect();
    let KernelArg::Buffer(o) = &args[0] else {
        panic!()
    };
    assert_eq!(
        o.to_f64_vec(),
        vec![f64::from(w.iter().sum::<f32>()), f64::from(w[3])]
    );
}

#[test]
fn every_thread_has_its_own_array() {
    // If the array were shared, the four threads would overwrite each other's slot 0.
    let k = launched(
        "void k(float* out) { float v[2]; v[0] = (float)threadIdx.x; v[1] = 100.0f; out[threadIdx.x] = v[0] + v[1]; }",
        4,
    )
    .unwrap();
    let mut args = vec![f32s(&[0.0; 4])];
    run_launch(&k, &mut args).expect("runs");
    let KernelArg::Buffer(o) = &args[0] else {
        panic!()
    };
    assert_eq!(o.to_f64_vec(), vec![100.0, 101.0, 102.0, 103.0]);
}

fn not_yet(r: Result<KernelFn, KernelLiftError>, needle: &str) {
    match r {
        Err(KernelLiftError::NotYetLifted { construct, .. }) => {
            assert!(construct.contains(needle), "`{construct}` lacks `{needle}`");
        }
        other => panic!("expected NotYetLifted({needle}), got {other:?}"),
    }
}

#[test]
fn what_is_not_a_plain_sized_array_declines() {
    not_yet(
        lift_kernel(&C, "void k(float* out) { float v[4]; out[0] = 1.0f; }", "k"),
        "a local array without a launch",
    );
    not_yet(
        launched(
            "void k(float* out, int n) { float v[n]; out[0] = 1.0f; }",
            1,
        ),
        "positive integer literal",
    );
    not_yet(
        launched("void k(float* out) { float v[0]; out[0] = 1.0f; }", 1),
        "positive integer literal",
    );
    not_yet(
        launched(
            "void k(float* out) { float v[2] = {1.0f, 2.0f, 3.0f}; out[0] = v[0]; }",
            1,
        ),
        // a brace initialiser lifts since census slice 6 (`tests/kernel_body_array_init.rs`); one with too many elements does not
        "3 initialisers for an array of 2 elements",
    );
    not_yet(
        launched("void k(float* out) { float v[2][2]; out[0] = 1.0f; }", 1),
        "more than one dimension",
    );
    // Control: the plain spelling lifts.
    assert!(
        launched(
            "void k(float* out) { float v[2]; v[0] = 1.0f; out[0] = v[0]; }",
            1
        )
        .is_ok()
    );
}
