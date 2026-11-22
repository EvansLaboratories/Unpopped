// SPDX-License-Identifier: MIT OR Apache-2.0
//! **Census wall: a multi-dimensional array** (`__shared__ uint8_t smem_A[M_TILE][K_TILE_BYTES];`, 8 Baracuda kernels). One flat
//! allocation of the product of the extents, indexed row-major (`a[i][j]` is element `i * J + j`), as C lays it out. Every index
//! must be given; a row (`a[i]`) declines. Oracle runs against Rust references.

use unpopped::exec::LaunchShape;
use unpopped::kernel::KernelFn;
use unpopped::oracle::TypedBuffer;
use unpopped::oracle::kernel_eval::KernelArg;
use unpopped::oracle::kernel_launch::run_launch;
use unpopped_convert::{Frontend, KernelLiftError, lift_kernel_launched};

fn parse(src: &str) -> Option<tree_sitter::Tree> {
    // a same-length qualifier keeps the declaration node starting where `__shared__` does (the walker reads the original text)
    let b = src.replace("__shared__", "volatile  ");
    let mut p = tree_sitter::Parser::new();
    p.set_language(&tree_sitter_cpp::LANGUAGE.into()).ok()?;
    p.parse(b, None)
}

const C: Frontend = Frontend::new("c", "void", &[], "out", "in", parse);

fn lift(src: &str, block: u32) -> Result<KernelFn, KernelLiftError> {
    lift_kernel_launched(&C, src, "k", LaunchShape::new([block, 1, 1], [1, 1, 1]))
}

fn run(k: &KernelFn, n: usize) -> Vec<f64> {
    let mut args = vec![KernelArg::Buffer(TypedBuffer::from_f32(
        &[n as i64],
        &vec![0.0; n],
    ))];
    run_launch(k, &mut args).expect("runs");
    let KernelArg::Buffer(o) = &args[0] else {
        panic!()
    };
    o.to_f64_vec()
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
fn a_private_two_dimensional_array_is_row_major() {
    // 3 rows by 4 columns: a[r][c] = 10 r + c; reading a[1][2] and the last element a[2][3] pins the stride
    let src = "void k(float* out) { float a[3][4]; \
        for (int r = 0; r < 3; r++) { for (int c = 0; c < 4; c++) { a[r][c] = (float)(10 * r + c); } } \
        out[0] = a[1][2]; out[1] = a[2][3]; out[2] = a[0][3]; out[3] = a[2][0]; }";
    assert_eq!(run(&lift(src, 1).unwrap(), 4), vec![12.0, 23.0, 3.0, 20.0]);
}

#[test]
fn a_shared_tile_is_shared_across_threads_with_a_barrier() {
    // thread t writes row t of a 4 x 3 tile, then reads the column the NEXT row wrote: wrong without the right strides
    let src = "void k(float* out) { __shared__ float t[4][3]; \
        for (int c = 0; c < 3; c++) { t[threadIdx.x][c] = (float)(threadIdx.x * 3 + c); } __syncthreads(); \
        out[threadIdx.x] = t[(threadIdx.x + 1) % 4][2] + t[threadIdx.x][0]; }";
    let k = lift(src, 4).unwrap();
    let want: Vec<f64> = (0..4)
        .map(|t: i32| f64::from(((t + 1) % 4) * 3 + 2 + t * 3))
        .collect();
    assert_eq!(run(&k, 4), want);
}

#[test]
fn three_dimensions_and_constant_extents() {
    let src = "void k(float* out) { float a[2][3][2]; \
        for (int i = 0; i < 2; i++) { for (int j = 0; j < 3; j++) { for (int l = 0; l < 2; l++) { a[i][j][l] = (float)(100 * i + 10 * j + l); } } } \
        out[0] = a[1][2][1]; out[1] = a[1][0][1]; out[2] = a[0][2][0]; }";
    assert_eq!(run(&lift(src, 1).unwrap(), 3), vec![121.0, 101.0, 20.0]);
}

#[test]
fn what_is_not_a_full_index_or_a_plain_array_declines() {
    // a row used where an element is needed
    not_yet(
        lift(
            "void k(float* out) { float a[2][2]; a[0][0] = 1.0f; out[0] = a[1]; }",
            1,
        ),
        "1 subscript(s) on a 2-dimensional array",
    );
    // too many subscripts
    not_yet(
        lift(
            "void k(float* out) { float a[2][2]; a[0][0] = 1.0f; out[0] = a[0][0][0]; }",
            1,
        ),
        "subscript",
    );
    // an initialiser of a multi-dimensional array
    not_yet(
        lift(
            "void k(float* out) { float a[2][2] = {{1.0f, 2.0f}, {3.0f, 4.0f}}; out[0] = a[1][1]; }",
            1,
        ),
        "more than one dimension and an initialiser",
    );
    // a zero extent
    not_yet(
        lift("void k(float* out) { float a[2][0]; out[0] = 1.0f; }", 1),
        "positive integer literal",
    );
    // control: the one-dimensional array still lifts
    assert!(
        lift(
            "void k(float* out) { float a[2]; a[1] = 1.0f; out[0] = a[1]; }",
            1
        )
        .is_ok()
    );
}
