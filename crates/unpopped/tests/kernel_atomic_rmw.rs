// SPDX-License-Identifier: MIT OR Apache-2.0
//! **`Stmt::AtomicRmw` (relaxed, global integer element)**: the checker admits only the v1 form, the
//! launch interpreter applies the combine exactly and reports a `DataRace` only for a conflict that
//! involves a plain access (`docs/ir-kernel-layer.md`, "Atomics").

use unpopped::exec::{ExecId, LaunchShape};
use unpopped::ir::{BinaryOp, ScalarExpr};
use unpopped::kernel::{
    AtomicOp, BufferRole, KernelError, KernelFn, Param, Place, Stmt, check_kernel,
};
use unpopped::oracle::TypedBuffer;
use unpopped::oracle::kernel_eval::{KernelArg, KernelTrap};
use unpopped::oracle::kernel_launch::run_launch;
use unpopped_vocab::ElementKind;

const I32: ElementKind = ElementKind::I32;

fn cell() -> Place {
    Place::Logical {
        buffer: 0,
        coords: vec![ScalarExpr::ConstInt(0, I32)],
    }
}

fn atomic(op: AtomicOp, v: i64) -> Stmt {
    Stmt::AtomicRmw {
        op,
        place: cell(),
        value: ScalarExpr::ConstInt(v, I32),
    }
}

fn kernel(elem: ElementKind, role: BufferRole, body: Vec<Stmt>) -> KernelFn {
    let mut k = KernelFn::new(
        "a",
        I32,
        vec![Param::buffer("c", elem, role, Some(1))],
        vec![],
        unpopped::kernel::Block(body),
    );
    k.launch = Some(LaunchShape::new([4, 1, 1], [3, 1, 1]));
    k
}

fn launch(k: &KernelFn, start: i32) -> Result<i32, KernelTrap> {
    let buf = TypedBuffer::new(I32, vec![1], vec![1], start.to_le_bytes().to_vec());
    let mut args = vec![KernelArg::Buffer(buf)];
    run_launch(k, &mut args)?;
    let KernelArg::Buffer(b) = &args[0] else {
        unreachable!()
    };
    Ok(i32::from_le_bytes(b.bytes[..4].try_into().unwrap()))
}

#[test]
fn twelve_threads_adding_one_each_make_twelve_with_no_race() {
    let k = kernel(I32, BufferRole::ReadWrite, vec![atomic(AtomicOp::Add, 1)]);
    assert_eq!(check_kernel(&k), Ok(()));
    assert_eq!(launch(&k, 5), Ok(17));
}

#[test]
fn max_and_min_pick_the_extremum_and_add_wraps() {
    let k = |op, v| kernel(I32, BufferRole::ReadWrite, vec![atomic(op, v)]);
    assert_eq!(launch(&k(AtomicOp::Max, 9), 3), Ok(9));
    assert_eq!(launch(&k(AtomicOp::Max, 1), 3), Ok(3));
    assert_eq!(launch(&k(AtomicOp::Min, -4), 3), Ok(-4));
    let wrap = k(AtomicOp::Add, i64::from(i32::MAX));
    assert_eq!(launch(&wrap, 0), Ok(i32::MAX.wrapping_mul(12)));
}

#[test]
fn an_atomic_and_a_plain_access_to_one_element_race() {
    let plain_store = Stmt::Store {
        place: cell(),
        value: ScalarExpr::ConstInt(1, I32),
    };
    let load = Stmt::Load {
        local: 0,
        place: cell(),
    };
    for plain in [plain_store, load] {
        let mut k = kernel(
            I32,
            BufferRole::ReadWrite,
            vec![plain, atomic(AtomicOp::Add, 1)],
        );
        k.locals = vec![unpopped::kernel::LocalDecl::scalar(I32)];
        assert_eq!(check_kernel(&k), Ok(()));
        assert_eq!(launch(&k, 0), Err(KernelTrap::DataRace));
    }
}

/// The atomic comes first, and the plain access is made by the last thread of the launch, so
/// nothing runs after it: only the earlier atomics can make it race.
#[test]
fn a_plain_access_after_an_atomic_races_too() {
    let store = Stmt::Store {
        place: cell(),
        value: ScalarExpr::ConstInt(1, I32),
    };
    let load = Stmt::Load {
        local: 0,
        place: cell(),
    };
    let is = |id, v| {
        ScalarExpr::Binary(
            BinaryOp::CmpEq,
            Box::new(ScalarExpr::Exec(id)),
            Box::new(ScalarExpr::ConstInt(v, ElementKind::U32)),
        )
    };
    for plain in [store, load] {
        let last = Stmt::If {
            cond: is(ExecId::BlockIdx(0), 2),
            then: unpopped::kernel::Block(vec![Stmt::If {
                cond: is(ExecId::ThreadIdx(0), 3),
                then: unpopped::kernel::Block(vec![plain]),
                els: unpopped::kernel::Block(vec![]),
            }]),
            els: unpopped::kernel::Block(vec![]),
        };
        let mut k = kernel(
            I32,
            BufferRole::ReadWrite,
            vec![atomic(AtomicOp::Add, 1), last],
        );
        k.locals = vec![unpopped::kernel::LocalDecl::scalar(I32)];
        assert_eq!(check_kernel(&k), Ok(()));
        assert_eq!(launch(&k, 0), Err(KernelTrap::DataRace));
    }
}

#[test]
fn plain_stores_alone_still_race() {
    let store = Stmt::Store {
        place: cell(),
        value: ScalarExpr::ConstInt(1, I32),
    };
    let k = kernel(I32, BufferRole::ReadWrite, vec![store]);
    assert_eq!(launch(&k, 0), Err(KernelTrap::DataRace));
}

#[test]
fn the_checker_admits_only_the_v1_form() {
    let bad = |elem, role| check_kernel(&kernel(elem, role, vec![atomic(AtomicOp::Add, 1)]));
    assert!(matches!(
        bad(I32, BufferRole::Read),
        Err(KernelError::AtomicUnsupported { .. })
    ));
    assert!(matches!(
        bad(I32, BufferRole::Write),
        Err(KernelError::AtomicUnsupported { .. })
    ));
    assert!(matches!(
        bad(ElementKind::F32, BufferRole::ReadWrite),
        Err(KernelError::AtomicUnsupported { .. })
    ));
    let wrong_value = kernel(
        I32,
        BufferRole::ReadWrite,
        vec![Stmt::AtomicRmw {
            op: AtomicOp::Add,
            place: cell(),
            value: ScalarExpr::ConstInt(1, ElementKind::I64),
        }],
    );
    assert!(check_kernel(&wrong_value).is_err());
}
