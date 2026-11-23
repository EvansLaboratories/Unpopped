// SPDX-License-Identifier: MIT OR Apache-2.0
//! **A struct field as a launch grid extent** (ROADMAP U23 slice 3). An integer field of a `ParamKind::Struct` may be the
//! `GridDim::Param` / `GridDim::CeilDiv` source of a grid (ROADMAP U22), exactly as a scalar parameter is. The U23 guard holds: no
//! storage location is chosen per thread set (`kernel_interface::layout_only_guard`).
//!
//! Each behaviour has a program that shows it and a control that does not.

use unpopped::exec::{ExecId, GridDim, LaunchShape};
use unpopped::ir::ScalarExpr;
use unpopped::kernel::{
    Block, BufferRole, KernelError, KernelFn, LocalDecl, Param, Place, Stmt, StructField,
    check_kernel,
};
use unpopped::oracle::TypedBuffer;
use unpopped::oracle::kernel_eval::{KernelArg, KernelTrap};
use unpopped::oracle::kernel_launch::run_launch;
use unpopped_vocab::ElementKind;

const F32: ElementKind = ElementKind::F32;
const I32: ElementKind = ElementKind::I32;
const COUNT: u32 = 0;
const SCALE: u32 = 1;
const OUT: usize = 8;

/// `out[blockIdx.x] = 1.0`, one thread per block, over a grid read from `cfg.n` (local `COUNT`); `cfg` is
/// `{ s: f32, n: <ty> }`, so the count is NOT field 0.
fn kernel(count_ty: ElementKind, grid: GridDim) -> KernelFn {
    KernelFn::new(
        "sg",
        F32,
        vec![
            Param::buffer("out", F32, BufferRole::Write, Some(1)),
            Param::structure(
                "cfg",
                vec![
                    StructField::new("s", F32, SCALE),
                    StructField::new("n", count_ty, COUNT),
                ],
            ),
        ],
        vec![LocalDecl::scalar(count_ty), LocalDecl::scalar(F32)],
        Block(vec![Stmt::Store {
            place: Place::Logical {
                buffer: 0,
                coords: vec![ScalarExpr::Exec(ExecId::BlockIdx(0))],
            },
            value: ScalarExpr::Const(1.0),
        }]),
    )
    .with_launch(LaunchShape::new([1, 1, 1], [1, 1, 1]).with_grid_dims([
        grid,
        GridDim::Static(1),
        GridDim::Static(1),
    ]))
}

fn args(n: i128) -> Vec<KernelArg> {
    vec![
        KernelArg::Buffer(TypedBuffer::from_f32(&[OUT as i64], &[0.0; OUT])),
        KernelArg::Struct(vec![KernelArg::Float(2.0), KernelArg::Int(n)]),
    ]
}

fn written(a: &[KernelArg]) -> usize {
    let KernelArg::Buffer(b) = &a[0] else {
        unreachable!("a buffer")
    };
    b.bytes
        .chunks(4)
        .filter(|c| f32::from_le_bytes((*c).try_into().unwrap()) == 1.0)
        .count()
}

#[test]
fn an_integer_field_may_be_the_grid() {
    assert_eq!(check_kernel(&kernel(I32, GridDim::Param(COUNT))), Ok(()));
}

#[test]
fn a_float_field_may_not() {
    assert_eq!(
        check_kernel(&kernel(F32, GridDim::Param(COUNT))),
        Err(KernelError::BadLaunch)
    );
}

#[test]
fn the_grid_comes_from_the_field_value() {
    let k = kernel(I32, GridDim::Param(COUNT));
    for n in [1, 3, 5] {
        let mut a = args(n);
        run_launch(&k, &mut a).expect("runs");
        assert_eq!(written(&a), n as usize, "n = {n}");
    }
}

#[test]
fn a_ceil_div_grid_reads_the_field_too() {
    let k = kernel(
        I32,
        GridDim::CeilDiv {
            param: COUNT,
            by: 2,
        },
    );
    let mut a = args(5);
    run_launch(&k, &mut a).expect("runs");
    assert_eq!(written(&a), 3, "ceil(5 / 2)");
}

#[test]
fn a_missing_or_non_positive_field_value_is_refused() {
    let k = kernel(I32, GridDim::Param(COUNT));
    assert!(matches!(
        run_launch(&k, &mut args(0)),
        Err(KernelTrap::ArgMismatch(_))
    ));
    let mut short = args(3);
    short[1] = KernelArg::Struct(vec![KernelArg::Float(2.0)]);
    assert!(matches!(
        run_launch(&k, &mut short),
        Err(KernelTrap::ArgMismatch(_))
    ));
}
