// SPDX-License-Identifier: MIT OR Apache-2.0
//! **Census wall: the address of a scalar local** (`T zero; { unsigned char* p = reinterpret_cast<unsigned char*>(&zero); for (size_t
//! i = 0; i < sizeof(T); ++i) p[i] = 0; }`, the embedding kernels: 16 real instantiations stopped at "the address of something that is
//! not an array or pointer element"). A scalar local whose address is taken (other than by `memcpy(&x, ...)`, which has its own slice)
//! lives in memory: a `Private` allocation of one element, so a byte pointer into it reads and writes the same bytes the name does.
//! Reads of the name are loads, assignments are stores (`+=` reads first). Oracle runs against Rust references on the bit patterns.

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

fn run(k: &KernelFn, outs: usize) -> Vec<f64> {
    let mut args = vec![KernelArg::Buffer(TypedBuffer::from_f32(
        &[outs as i64],
        &vec![0.0; outs],
    ))];
    run_launch(k, &mut args).expect("runs");
    let KernelArg::Buffer(o) = &args[0] else {
        panic!()
    };
    o.to_f64_vec()
}

#[test]
fn zeroing_a_scalar_through_a_byte_pointer_is_the_memset_idiom() {
    let k = lift(
        "void k(float* out) { float zero = 5.0f; \
         { unsigned char* p = reinterpret_cast<unsigned char*>(&zero); for (size_t i = 0; i < sizeof(float); ++i) p[i] = 0; } \
         out[0] = zero; }",
    )
    .expect("lifts (was a decline: the address of something that is not an array or pointer element)");
    assert_eq!(run(&k, 1), vec![0.0]);
}

#[test]
fn the_bytes_written_are_the_bytes_the_name_reads_little_endian() {
    // 1.5f is 0x3FC00000: the bytes, least significant first, are 00 00 C0 3F
    let k = lift(
        "void k(float* out) { float z; unsigned char* p = reinterpret_cast<unsigned char*>(&z); \
         p[0] = 0; p[1] = 0; p[2] = 192; p[3] = 63; out[0] = z; \
         z = 2.5f; out[1] = (float)p[3]; out[2] = (float)p[2]; z += 1.0f; out[3] = z; }",
    )
    .expect("lifts");
    let two_and_a_half = 2.5f32.to_bits().to_le_bytes();
    assert_eq!(
        run(&k, 4),
        vec![
            1.5,
            f64::from(two_and_a_half[3]),
            f64::from(two_and_a_half[2]),
            3.5
        ]
    );
}

#[test]
fn an_integer_scalar_and_a_scalar_whose_address_is_never_taken() {
    let k = lift(
        "void k(float* out) { unsigned int u = 0xffffffffu; unsigned char* b = (unsigned char*)&u; b[0] = 1; b[3] = 0; out[0] = (float)u; \
         float plain = 4.0f; plain += 1.0f; out[1] = plain; }",
    )
    .expect("lifts");
    // bytes ff ff ff ff -> 01 ff ff 00 little endian = 0x00ffff01
    assert_eq!(run(&k, 2), vec![f64::from(0x00ff_ff01u32), 5.0]);
    // control: `plain` was never address-taken, so it is an ordinary local: the lowered body has exactly one Private allocation
    assert_eq!(k.allocs.len(), 1, "{:?}", k.allocs);
}
