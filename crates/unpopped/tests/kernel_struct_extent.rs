// SPDX-License-Identifier: MIT OR Apache-2.0
//! **A struct field as a dynamic allocation extent** (ROADMAP U23 slice 2). An integer field of a `ParamKind::Struct`
//! may size an `Extent::Param` allocation, exactly as a scalar parameter does. Layout only: no storage location is
//! chosen per thread set (`kernel_interface::layout_only_guard`).
//!
//! Each behaviour has a program that shows it and a control that does not.

use unpopped::exec::{AllocDecl, Extent, LaunchShape, MemorySpace};
use unpopped::ir::ScalarExpr;
use unpopped::kernel::{
    Block, BufferRole, KernelError, KernelFn, LocalDecl, Param, Place, Stmt, StructField,
    check_kernel,
};
use unpopped::kernel_interface::KernelInterface;
use unpopped::oracle::TypedBuffer;
use unpopped::oracle::kernel_eval::{KernelArg, KernelTrap};
use unpopped::oracle::kernel_launch::run_launch;
use unpopped_vocab::ElementKind;

const F32: ElementKind = ElementKind::F32;
const I32: ElementKind = ElementKind::I32;
const U32: ElementKind = ElementKind::U32;
const COUNT: u32 = 0;
const SCALE: u32 = 1;

/// `dyn[5] = 1.0` over a Shared allocation sized by `cfg.<field 0>` (local `COUNT`); `cfg` is `{ n: <ty>, s: f32 }`.
fn kernel(count_ty: ElementKind) -> KernelFn {
    KernelFn::new(
        "ext",
        F32,
        vec![
            Param::buffer("out", F32, BufferRole::Write, Some(1)),
            Param::structure(
                "cfg",
                vec![
                    StructField::new("n", count_ty, COUNT),
                    StructField::new("s", F32, SCALE),
                ],
            ),
        ],
        vec![LocalDecl::scalar(count_ty), LocalDecl::scalar(F32)],
        Block(vec![Stmt::Store {
            place: Place::Alloc {
                alloc: 0,
                index: ScalarExpr::ConstInt(5, U32),
            },
            value: ScalarExpr::Const(1.0),
        }]),
    )
    .with_launch(LaunchShape::new([1, 1, 1], [1, 1, 1]).with_subgroup(4, 8))
    .with_allocs(vec![AllocDecl::new(
        "dyn",
        MemorySpace::Shared,
        F32,
        Extent::Param(COUNT),
    )])
}

fn args(n: i128) -> Vec<KernelArg> {
    vec![
        KernelArg::Buffer(TypedBuffer::from_f32(&[1], &[0.0])),
        KernelArg::Struct(vec![KernelArg::Int(n), KernelArg::Float(2.0)]),
    ]
}

#[test]
fn an_integer_field_may_size_an_allocation() {
    assert_eq!(check_kernel(&kernel(I32)), Ok(()));
}

#[test]
fn a_float_field_may_not() {
    assert_eq!(
        check_kernel(&kernel(F32)),
        Err(KernelError::AllocExtent { alloc: 0 })
    );
}

#[test]
fn the_extent_comes_from_the_field_value() {
    let k = kernel(I32);
    run_launch(&k, &mut args(6)).expect("control: n = 6 holds index 5");
    assert_eq!(
        run_launch(&k, &mut args(5)),
        Err(KernelTrap::AllocOutOfBounds { alloc: 0 })
    );
    assert!(matches!(
        run_launch(&k, &mut args(0)),
        Err(KernelTrap::ArgMismatch(_))
    ));
}

#[test]
fn the_interface_names_the_struct_and_the_field() {
    let i = KernelInterface::of(&kernel(I32));
    assert_eq!(i.dynamic_shared.len(), 1);
    let d = &i.dynamic_shared[0];
    assert_eq!(d.size_param, 1, "the struct parameter");
    assert_eq!(d.size_field, Some(0), "its first field");
}

#[test]
fn a_scalar_extent_has_no_field() {
    let k = KernelFn::new(
        "plain",
        F32,
        vec![Param::scalar("n", I32, COUNT)],
        vec![LocalDecl::scalar(I32)],
        Block(vec![]),
    )
    .with_allocs(vec![AllocDecl::new(
        "dyn",
        MemorySpace::Shared,
        F32,
        Extent::Param(COUNT),
    )]);
    assert_eq!(KernelInterface::of(&k).dynamic_shared[0].size_field, None);
}
