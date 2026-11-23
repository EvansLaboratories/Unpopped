// SPDX-License-Identifier: MIT OR Apache-2.0
//! **Census wall: `memcpy(&x, p, sizeof(x))` into a scalar local** (the gguf `memcpy(&qh, x[ib].qh, sizeof(qh))` gathering four bytes
//! into a `uint32_t`; 12 kernels). It loads `x`'s type from the byte pointer, little-endian, at any byte offset. Any other shape
//! (a size that is not the local's, a non-scalar destination, a source that is not a pointer) declines by name. Oracle runs
//! against Rust references (`u32::from_le_bytes`, `f32::from_le_bytes`).

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

const BYTES: [u8; 12] = [1, 2, 3, 4, 5, 6, 7, 8, 0x00, 0x00, 0xC0, 0x3F];

fn run(k: &KernelFn, n: usize) -> Vec<f64> {
    let mut args = vec![
        KernelArg::Buffer(TypedBuffer::from_f32(&[n as i64], &vec![0.0; n])),
        KernelArg::Buffer(TypedBuffer::from_u8(&[12], &BYTES)),
    ];
    run_launch(k, &mut args).expect("runs");
    let KernelArg::Buffer(o) = &args[0] else {
        panic!()
    };
    o.to_f64_vec()
}

fn declines(src: &str) {
    let r = lift(src);
    assert!(
        matches!(r, Err(KernelLiftError::NotYetLifted { .. })),
        "{src}: {r:?}"
    );
}

#[test]
fn four_bytes_gather_little_endian_at_any_offset_and_a_float_takes_its_bit_pattern() {
    let k = lift(
        "void k(float* out, const unsigned char* in) { unsigned int qh; memcpy(&qh, in + 1, sizeof(qh)); \
         out[0] = (float)(qh & 0xffff); out[1] = (float)(qh >> 16); \
         float f; memcpy(&f, in + 8, sizeof(f)); out[2] = f; \n         unsigned long long w; memcpy(&w, in, sizeof(w)); out[3] = (float)(w >> 40); out[4] = (float)(w & 0xff); }",
    )
    .unwrap();
    let qh = u32::from_le_bytes([BYTES[1], BYTES[2], BYTES[3], BYTES[4]]);
    let f = f32::from_le_bytes([BYTES[8], BYTES[9], BYTES[10], BYTES[11]]);
    let w = u64::from_le_bytes([
        BYTES[0], BYTES[1], BYTES[2], BYTES[3], BYTES[4], BYTES[5], BYTES[6], BYTES[7],
    ]);
    assert_eq!(
        run(&k, 5),
        vec![
            f64::from(qh & 0xffff),
            f64::from(qh >> 16),
            f64::from(f),
            (w >> 40) as f64,
            (w & 0xff) as f64,
        ]
    );
}

#[test]
fn a_size_that_is_not_the_locals_or_a_non_scalar_destination_declines() {
    declines(
        "void k(float* out, const unsigned char* in) { unsigned int qh; memcpy(&qh, in, 2); out[0] = (float)qh; }",
    );
    declines(
        "void k(float* out, const unsigned char* in) { unsigned int qh; memcpy(&qh, in, sizeof(float) * 2); out[0] = (float)qh; }",
    );
    declines(
        "void k(float* out, const unsigned char* in) { unsigned int q[2]; memcpy(q, in, 8); out[0] = (float)q[0]; }",
    );
    declines(
        "void k(float* out, const unsigned char* in) { unsigned int qh; memcpy(&qh, 5, sizeof(qh)); out[0] = (float)qh; }",
    );
}
