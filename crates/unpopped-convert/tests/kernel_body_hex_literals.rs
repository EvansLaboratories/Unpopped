// SPDX-License-Identifier: MIT OR Apache-2.0
//! **Census wall: hexadecimal integer literals** (`0xffffffff`, the shuffle mask and every bit-twiddling constant; 7 Baracuda
//! kernels stopped at one). C types an unsuffixed hex literal as the first of `int`, `unsigned int`, `long long`, `unsigned
//! long long` that holds it, so `0xffffffff` is a 32-bit *unsigned* and `0xffffffff + 1` wraps to 0, where the decimal
//! `4294967295 + 1` is a 64-bit signed 4294967296. Oracle runs against Rust references.

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

fn out(src: &str, n: usize) -> Vec<f64> {
    let k = lift(src).unwrap();
    let mut args = vec![KernelArg::Buffer(TypedBuffer::from_f32(
        &[n as i64],
        &vec![0.0; n],
    ))];
    run_launch(&k, &mut args).expect("runs");
    let KernelArg::Buffer(o) = &args[0] else {
        panic!()
    };
    o.to_f64_vec()
}

#[test]
fn a_hex_literal_has_the_value_it_spells() {
    let src = "void k(float* out) { out[0] = (float)0xff; out[1] = (float)0XA0; out[2] = (float)0xffff; out[3] = (float)0x1F; }";
    assert_eq!(out(src, 4), vec![255.0, 160.0, 65535.0, 31.0]);
}

#[test]
fn the_type_is_the_first_that_holds_it() {
    // 0xffffffff is unsigned int: + 1 wraps. Decimal 4294967295 is a 64-bit signed: + 1 does not. (Both cast through int64 so
    // the f32 output is exact for these values.)
    let src = "void k(float* out) { out[0] = (float)(0xffffffff + 1); out[1] = (float)(4294967295 + 1); \
        out[2] = (float)(0x80000000 > 0); out[3] = (float)(0x100000000 + 1 == 4294967297); out[4] = (float)(0xffffffffu + 1u); }";
    assert_eq!(out(src, 5), vec![0.0, 4294967296.0, 1.0, 1.0, 0.0]);
}

#[test]
fn a_hex_mask_selects_bits() {
    let src = "void k(float* out) { unsigned int x = 0xdeadbeefu; out[0] = (float)((x >> 16) & 0xffff); out[1] = (float)(x & 0xf); \
        out[2] = (float)((x ^ 0xffffffff) >> 28); }";
    let x: u32 = 0xdead_beef;
    assert_eq!(
        out(src, 3),
        vec![
            f64::from((x >> 16) & 0xffff),
            f64::from(x & 0xf),
            f64::from((x ^ 0xffff_ffff) >> 28)
        ]
    );
}

#[test]
fn a_value_that_does_not_fit_a_signed_64_bit_integer_declines() {
    match lift("void k(float* out) { out[0] = (float)(0xffffffffffffffff > 0); }") {
        Err(KernelLiftError::NotYetLifted { construct, .. }) => {
            assert!(construct.contains("the literal"), "{construct}");
        }
        other => panic!("{other:?}"),
    }
    // a bare prefix is not a literal
    assert!(lift("void k(float* out) { out[0] = (float)0x; }").is_err());
}
