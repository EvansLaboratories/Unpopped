// SPDX-License-Identifier: MIT OR Apache-2.0
//! **Struct-typed scalar parameters** (ROADMAP U23, `docs/PATENT-SWEEP.md` U23 section). A `ParamKind::Struct` groups
//! scalar fields the kernel reads as locals, so an emitter can pass the group by value (`__grid_constant__`, Slang
//! `ConstantBuffer`) instead of flattening it. It describes a layout and carries no thread-set, memory-kind or copy-count
//! annotation (`kernel_interface::layout_only_guard`).
//!
//! Each behaviour has a program that shows it and a control that does not.

use unpopped::ir::ScalarExpr;
use unpopped::kernel::{
    Block, BufferRole, KernelError, KernelFn, LocalDecl, Param, Place, Stmt, StructField,
    check_kernel,
};
use unpopped::oracle::TypedBuffer;
use unpopped::oracle::kernel_eval::{KernelArg, KernelTrap, run_kernel};
use unpopped_vocab::ElementKind;

const F32: ElementKind = ElementKind::F32;
const I32: ElementKind = ElementKind::I32;
const A: u32 = 0;
const B: u32 = 1;

fn local(i: u32) -> ScalarExpr {
    ScalarExpr::Local(i)
}

fn out_at(i: i64) -> Place {
    Place::Logical {
        buffer: 0,
        coords: vec![ScalarExpr::ConstInt(i, I32)],
    }
}

/// `out[0] = cfg.a; out[1] = cfg.b * 2`, with `cfg` a struct parameter of two `F32` fields bound to locals `A` and `B`.
fn sum_kernel(fields: Vec<StructField>, locals: Vec<LocalDecl>) -> KernelFn {
    KernelFn::new(
        "sum",
        F32,
        vec![
            Param::buffer("out", F32, BufferRole::Write, Some(1)),
            Param::structure("cfg", fields),
        ],
        locals,
        Block(vec![
            Stmt::Store {
                place: out_at(0),
                value: local(A),
            },
            Stmt::Store {
                place: out_at(1),
                value: ScalarExpr::Add(Box::new(local(B)), Box::new(local(B))),
            },
        ]),
    )
}

fn two_f32_fields() -> Vec<StructField> {
    vec![StructField::new("a", F32, A), StructField::new("b", F32, B)]
}

fn baseline() -> KernelFn {
    sum_kernel(
        two_f32_fields(),
        vec![LocalDecl::scalar(F32), LocalDecl::scalar(F32)],
    )
}

fn args(a: f64, b: f64) -> Vec<KernelArg> {
    vec![
        KernelArg::Buffer(TypedBuffer::from_f32(&[2], &[0.0, 0.0])),
        KernelArg::Struct(vec![KernelArg::Float(a), KernelArg::Float(b)]),
    ]
}

#[test]
fn the_baseline_is_valid() {
    assert_eq!(check_kernel(&baseline()), Ok(()));
}

#[test]
fn a_kernel_reads_each_field_as_its_local() {
    let mut a = args(1.5, 2.25);
    run_kernel(&baseline(), &mut a).expect("runs");
    let KernelArg::Buffer(out) = &a[0] else {
        panic!("buffer first")
    };
    assert_eq!(out.to_f64_vec(), vec![1.5, 4.5]);
    // Control: different field values give a different result, so the fields are not ignored.
    let mut c = args(10.0, 0.5);
    run_kernel(&baseline(), &mut c).expect("runs");
    let KernelArg::Buffer(out) = &c[0] else {
        panic!("buffer first")
    };
    assert_eq!(out.to_f64_vec(), vec![10.0, 1.0]);
}

#[test]
fn a_field_local_that_is_unknown_is_refused() {
    // Control: the baseline above passes with both locals declared.
    let k = sum_kernel(two_f32_fields(), vec![LocalDecl::scalar(F32)]);
    assert_eq!(
        check_kernel(&k),
        Err(KernelError::UnknownLocal { local: B })
    );
}

#[test]
fn two_fields_on_one_local_are_refused() {
    let fields = vec![StructField::new("a", F32, A), StructField::new("b", F32, A)];
    let k = sum_kernel(fields, vec![LocalDecl::scalar(F32), LocalDecl::scalar(F32)]);
    assert_eq!(check_kernel(&k), Err(KernelError::Redeclared { local: A }));
}

#[test]
fn a_field_whose_local_has_another_type_is_refused() {
    let k = sum_kernel(
        two_f32_fields(),
        vec![LocalDecl::scalar(F32), LocalDecl::scalar(I32)],
    );
    assert!(matches!(
        check_kernel(&k),
        Err(KernelError::TypeMismatch { .. })
    ));
}

#[test]
fn an_argument_with_the_wrong_field_count_is_a_named_mismatch() {
    let mut a = vec![
        KernelArg::Buffer(TypedBuffer::from_f32(&[2], &[0.0, 0.0])),
        KernelArg::Struct(vec![KernelArg::Float(1.0)]),
    ];
    let r = run_kernel(&baseline(), &mut a);
    assert!(matches!(r, Err(KernelTrap::ArgMismatch(_))), "{r:?}");
}
