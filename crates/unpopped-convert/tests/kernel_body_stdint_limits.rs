// SPDX-License-Identifier: MIT OR Apache-2.0
//! **Census wall: the `<stdint.h>` limits** (`INT8_MAX` ... `INT64_MIN`, `UINT8_MAX`..`UINT32_MAX`; the reduce-axis integer functors'
//! `init()`: 12 real instantiations once their member-specialisation wall was gone). Each is the constant the C macro expands to, an
//! `int` (`unsigned int` for `UINT32_MAX`, `long long` for the 64-bit ones). Oracle runs against Rust's `MAX`/`MIN`.

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

#[test]
fn each_limit_is_the_constant_the_c_macro_expands_to() {
    let k = lift(
        "void k(double* out) { out[0] = (double)INT8_MAX; out[1] = (double)INT8_MIN; out[2] = (double)UINT8_MAX; \
         out[3] = (double)INT16_MAX; out[4] = (double)INT16_MIN; out[5] = (double)UINT16_MAX; \
         out[6] = (double)INT32_MAX; out[7] = (double)INT32_MIN; out[8] = (double)UINT32_MAX; \
         out[9] = (double)INT64_MAX; out[10] = (double)INT64_MIN; }",
    )
    .expect("lifts (was a decline: the name `INT8_MAX`)");
    let mut args = vec![KernelArg::Buffer(TypedBuffer::from_f64(&[11], &[0.0; 11]))];
    run_launch(&k, &mut args).expect("runs");
    let KernelArg::Buffer(o) = &args[0] else {
        panic!()
    };
    let want = [
        f64::from(i8::MAX),
        f64::from(i8::MIN),
        f64::from(u8::MAX),
        f64::from(i16::MAX),
        f64::from(i16::MIN),
        f64::from(u16::MAX),
        f64::from(i32::MAX),
        f64::from(i32::MIN),
        f64::from(u32::MAX),
        i64::MAX as f64,
        i64::MIN as f64,
    ];
    assert_eq!(o.to_f64_vec(), want);
}

#[test]
fn a_narrow_limit_converts_to_the_narrow_type_it_is_stored_in() {
    // `uint8_t x = UINT8_MAX; x + 1` wraps in the narrow type; an `int` limit fills an `int` local exactly
    let k = lift(
        "void k(double* out) { unsigned char b = UINT8_MAX; b = b + 1; out[0] = (double)b; int m = INT32_MAX; out[1] = (double)(m - 1); }",
    )
    .expect("lifts");
    let mut args = vec![KernelArg::Buffer(TypedBuffer::from_f64(&[2], &[0.0; 2]))];
    run_launch(&k, &mut args).expect("runs");
    let KernelArg::Buffer(o) = &args[0] else {
        panic!()
    };
    assert_eq!(o.to_f64_vec(), vec![0.0, f64::from(i32::MAX - 1)]);
}
