// SPDX-License-Identifier: MIT OR Apache-2.0
//! **`ParamsBlock::of_struct`** (ROADMAP U23, derivation slice): the layout of a struct parameter, derived from its fields
//! with the natural C rules (each field aligned to its own size, the block aligned to its widest field). No thread-set, memory-kind or copy-count
//! annotation is carried (the U23 guard, `docs/PATENT-SWEEP.md`). Consumers: fishstringer and Baracuda.
//!
//! The expected offsets are written out by hand from the C rules, with distinct sizes so a wrong alignment cannot hide.

use unpopped::kernel::{Param, StructField};
use unpopped::kernel_interface::{ParamsBlock, ParamsBlockError};
use unpopped_vocab::ElementKind as E;

fn st(tys: &[E]) -> Param {
    let fields = tys
        .iter()
        .enumerate()
        .map(|(i, t)| StructField::new(format!("f{i}"), *t, i as u32))
        .collect();
    Param::structure("cfg", fields)
}

fn offsets(b: &ParamsBlock) -> Vec<u32> {
    b.fields.iter().map(|f| f.offset).collect()
}

#[test]
fn padding_follows_the_c_rules() {
    // u8 @0, i32 @4 (pad 3), u16 @8, f64 @16 (pad 6): size 24, align 8.
    let b = ParamsBlock::of_struct(&st(&[E::U8, E::I32, E::U16, E::F64])).unwrap();
    assert_eq!(offsets(&b), [0, 4, 8, 16]);
    assert_eq!((b.size, b.align, b.by_address), (24, 8, false));
    let names: Vec<_> = b.fields.iter().map(|f| f.name.as_str()).collect();
    assert_eq!(names, ["f0", "f1", "f2", "f3"]);
    assert_eq!(b.fields[3].ty, E::F64);
}

#[test]
fn tail_padding_rounds_the_size_to_the_alignment() {
    // f64 @0, u8 @8: raw end 9, rounded to 16.
    let b = ParamsBlock::of_struct(&st(&[E::F64, E::U8])).unwrap();
    assert_eq!((b.size, b.align), (16, 8));
    // Control: a packed-by-size struct needs no padding.
    let c = ParamsBlock::of_struct(&st(&[E::I32, E::F32])).unwrap();
    assert_eq!(offsets(&c), [0, 4]);
    assert_eq!((c.size, c.align), (8, 4));
}

#[test]
fn complex_is_a_pair_aligned_to_one_half() {
    let b = ParamsBlock::of_struct(&st(&[E::U8, E::Complex64, E::Complex128])).unwrap();
    assert_eq!(offsets(&b), [0, 4, 16]);
    assert_eq!((b.size, b.align), (32, 8));
}

#[test]
fn empty_struct_is_size_zero_align_one() {
    let b = ParamsBlock::of_struct(&st(&[])).unwrap();
    assert_eq!((b.size, b.align, b.fields.len()), (0, 1, 0));
}

#[test]
fn by_address_is_the_callers_choice() {
    let b = ParamsBlock::of_struct(&st(&[E::I32]))
        .unwrap()
        .passed_by_address();
    assert!(b.by_address);
}

#[test]
fn declines_are_typed() {
    assert_eq!(
        ParamsBlock::of_struct(&Param::scalar("x", E::I32, 0)),
        Err(ParamsBlockError::NotAStruct)
    );
    for ty in [E::I4, E::U4, E::B1] {
        let e = ParamsBlock::of_struct(&st(&[E::I32, ty])).unwrap_err();
        assert_eq!(
            e,
            ParamsBlockError::Unsized {
                field: "f1".into(),
                ty
            }
        );
        assert!(e.to_string().contains("f1"));
    }
}
