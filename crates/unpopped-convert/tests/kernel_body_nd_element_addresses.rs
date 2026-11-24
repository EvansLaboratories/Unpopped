// SPDX-License-Identifier: MIT OR Apache-2.0
//! **Census wall: the address of an element of a multi-dimensional array** (`*reinterpret_cast<const uint32_t*>(&smem_A[m][k])`,
//! the fp8 MMA fragment read; 5 Baracuda kernels). The array is one row-major allocation, so `&a[i][j]` is a pointer at flat
//! element `i * J + j`; the cast and the dereference then lift as for a one-dimensional array. Oracle runs against Rust
//! references (little-endian words).

use unpopped::exec::LaunchShape;
use unpopped::kernel::KernelFn;
use unpopped::oracle::TypedBuffer;
use unpopped::oracle::kernel_eval::KernelArg;
use unpopped::oracle::kernel_launch::run_launch;
use unpopped_convert::{Frontend, KernelLiftError, lift_kernel_launched};

fn parse(src: &str) -> Option<tree_sitter::Tree> {
    // a same-length qualifier keeps the declaration node starting where `__shared__` does
    let b = src.replace("__shared__", "volatile  ");
    let mut p = tree_sitter::Parser::new();
    p.set_language(&tree_sitter_cpp::LANGUAGE.into()).ok()?;
    p.parse(b, None)
}

const C: Frontend = Frontend::new("c", "void", &[], "out", "in", parse);

fn lift(src: &str) -> Result<KernelFn, KernelLiftError> {
    lift_kernel_launched(&C, src, "k", LaunchShape::new([1, 1, 1], [1, 1, 1]))
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

const FILL: &str = "__shared__ uint8_t t[4][8]; \
    for (int r = 0; r < 4; r++) { for (int c = 0; c < 8; c++) { t[r][c] = (uint8_t)(r * 16 + c); } } ";

#[test]
fn a_word_read_through_the_address_of_a_two_dimensional_element() {
    let src = format!(
        "void k(float* out) {{ {FILL} \
         uint32_t w = *reinterpret_cast<const uint32_t*>(&t[1][4]); out[0] = (float)(w & 65535u); out[1] = (float)(w >> 16); }}"
    );
    // t[1][4..8] = 0x14, 0x15, 0x16, 0x17: little endian 0x17161514
    let w: u32 = 0x1716_1514;
    assert_eq!(
        run(&lift(&src).unwrap(), 2),
        vec![f64::from(w & 0xffff), f64::from(w >> 16)]
    );
}

#[test]
fn a_pointer_made_from_the_address_indexes_the_flat_array() {
    // `&t[2][3]` is flat element 19; p[2] is element 21 = t[2][5] = 0x25; p[-1] is t[2][2]
    let src = format!(
        "void k(float* out) {{ {FILL} const uint8_t* p = &t[2][3]; out[0] = (float)p[2]; out[1] = (float)(*(p - 1)); }}"
    );
    assert_eq!(
        run(&lift(&src).unwrap(), 2),
        vec![f64::from(0x25), f64::from(0x22)]
    );
}

#[test]
fn a_row_address_or_a_one_dimensional_address_is_unchanged() {
    // `&t[1]` of a two-dimensional array is a row, not an element: still declines
    let row =
        format!("void k(float* out) {{ {FILL} const uint8_t* p = &t[1]; out[0] = (float)p[0]; }}");
    assert!(matches!(
        lift(&row),
        Err(KernelLiftError::NotYetLifted { .. })
    ));
    // control: a one-dimensional array keeps working
    let flat = "void k(float* out) { __shared__ uint8_t u[8]; u[3] = 9; const uint8_t* p = &u[2]; out[0] = (float)p[1]; }";
    assert_eq!(run(&lift(flat).unwrap(), 1), vec![9.0]);
}

#[test]
fn the_address_scales_by_the_element_size() {
    // a 4-byte element: `&t[1][2]` is byte 24 of 32; `p[1]` is t[1][3], `p[-2]` is t[1][0]
    let src = "void k(float* out) { __shared__ int t[2][4]; \
        for (int r = 0; r < 2; r++) { for (int c = 0; c < 4; c++) { t[r][c] = r * 10 + c; } } \
        const int* p = &t[1][2]; out[0] = (float)p[1]; out[1] = (float)p[-2]; }";
    assert_eq!(run(&lift(src).unwrap(), 2), vec![13.0, 10.0]);
}
