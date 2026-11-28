// SPDX-License-Identifier: MIT OR Apache-2.0
//! **A specialization constant is a scalar parameter fixed when the kernel is built** (`KernelFn::spec_consts`, additive).
//! The oracle binds the default whatever the launch slot holds; the checker refuses a constant that names no scalar parameter,
//! names one twice, or carries a default of the wrong kind or range; and a kernel without one serializes exactly as before.

use unpopped::ir::ScalarExpr;
use unpopped::kernel::{
    Block, BufferRole, KernelError, KernelFn, LocalDecl, Param, Place, SpecValue, Stmt,
    check_kernel,
};
use unpopped::oracle::TypedBuffer;
use unpopped::oracle::kernel_eval::{KernelArg, run_kernel};
use unpopped_vocab::ElementKind;

const I32: ElementKind = ElementKind::I32;

/// `out[0] = n + n`, with `n` the scalar parameter 1 (local 0).
fn doubles() -> KernelFn {
    KernelFn::new(
        "doubles",
        I32,
        vec![
            Param::buffer("out", I32, BufferRole::Write, Some(1)),
            Param::scalar("n", I32, 0),
        ],
        vec![LocalDecl::scalar(I32)],
        Block(vec![Stmt::Store {
            place: Place::Logical {
                buffer: 0,
                coords: vec![ScalarExpr::ConstInt(0, I32)],
            },
            value: ScalarExpr::Add(
                Box::new(ScalarExpr::Local(0)),
                Box::new(ScalarExpr::Local(0)),
            ),
        }]),
    )
}

fn run(k: &KernelFn, n: i128) -> f64 {
    let mut args = vec![
        KernelArg::Buffer(TypedBuffer::from_i32(&[1], &[0])),
        KernelArg::Int(n),
    ];
    run_kernel(k, &mut args).expect("runs");
    let KernelArg::Buffer(b) = &args[0] else {
        panic!("a buffer")
    };
    b.to_f64_vec()[0]
}

#[test]
fn the_oracle_binds_the_default_not_the_launch_slot() {
    let k = doubles().with_spec_const(1, SpecValue::Int(21));
    assert_eq!(check_kernel(&k), Ok(()));
    assert_eq!(run(&k, 5), 42.0, "the default 21, not the slot's 5");
    // control: without the constant the launch value is used
    assert_eq!(run(&doubles(), 5), 10.0);
}

#[test]
fn a_bad_spec_const_is_refused_and_the_good_one_is_the_control() {
    let bad = |k: KernelFn| check_kernel(&k);
    // names a buffer parameter, a parameter that is not there, the same parameter twice
    assert_eq!(
        bad(doubles().with_spec_const(0, SpecValue::Int(1))),
        Err(KernelError::BadSpecConst { index: 0 })
    );
    assert_eq!(
        bad(doubles().with_spec_const(9, SpecValue::Int(1))),
        Err(KernelError::BadSpecConst { index: 0 })
    );
    assert_eq!(
        bad(doubles()
            .with_spec_const(1, SpecValue::Int(1))
            .with_spec_const(1, SpecValue::Int(2))),
        Err(KernelError::BadSpecConst { index: 1 })
    );
    // a float default for an integer parameter, and an integer out of the type's range
    assert_eq!(
        bad(doubles().with_spec_const(1, SpecValue::Float(1.5))),
        Err(KernelError::BadSpecConst { index: 0 })
    );
    assert_eq!(
        bad(doubles().with_spec_const(1, SpecValue::Int(i128::from(i32::MAX) + 1))),
        Err(KernelError::BadSpecConst { index: 0 })
    );
    // control: the in-range integer default on the scalar parameter passes
    assert_eq!(
        bad(doubles().with_spec_const(1, SpecValue::Int(i128::from(i32::MAX)))),
        Ok(())
    );
}

#[test]
fn a_kernel_without_one_serializes_as_before_and_one_with_it_round_trips() {
    let plain = serde_json::to_string(&doubles()).unwrap();
    assert!(!plain.contains("spec_consts"), "{plain}");
    // an old document without the field still reads
    let back: KernelFn = serde_json::from_str(&plain).unwrap();
    assert_eq!(back, doubles());
    let k = doubles()
        .with_spec_const(1, SpecValue::Int(7))
        .with_spec_const(0, SpecValue::Float(0.5));
    let text = serde_json::to_string(&k).unwrap();
    assert!(text.contains("spec_consts"));
    assert_eq!(serde_json::from_str::<KernelFn>(&text).unwrap(), k);
}
