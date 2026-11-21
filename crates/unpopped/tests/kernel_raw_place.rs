// SPDX-License-Identifier: MIT OR Apache-2.0
//! **`Place::Raw`: a typed access at a byte offset** (ROADMAP U9c slice 1, `docs/options-u9c-pointer-values.md`). The source buffer
//! holds the bytes 1, 2, 3 ... 16, so every byte is distinct: a wrong offset, a wrong width or a wrong byte order shows in the
//! value read. Misalignment and bounds are typed traps; the race detector sees the bytes through the buffer's own elements.

use unpopped::exec::{ExecId, LaunchShape};
use unpopped::ir::{BinaryOp, CastSpec, RoundMode, Saturation, ScalarExpr};
use unpopped::kernel::{
    AtomicOp, Block, BufferRole, KernelError, KernelFn, LocalDecl, Param, Place, Stmt, check_kernel,
};
use unpopped::oracle::TypedBuffer;
use unpopped::oracle::kernel_eval::{KernelArg, KernelTrap};
use unpopped::oracle::kernel_launch::run_launch;
use unpopped_vocab::ElementKind;

const U8: ElementKind = ElementKind::U8;
const U16: ElementKind = ElementKind::U16;
const U32: ElementKind = ElementKind::U32;
const U64: ElementKind = ElementKind::U64;
const F32: ElementKind = ElementKind::F32;

/// 16 bytes, byte `i` holding `i + 1`, as four little-endian `u32`.
fn source() -> TypedBuffer {
    TypedBuffer::from_u32(&[4], &[0x0403_0201, 0x0807_0605, 0x0C0B_0A09, 0x100F_0E0D])
}

/// `v` as a constant of the narrow integer type `to` (an integer leaf is I32, I64, U32 or U64, so a narrower one is a cast).
fn narrow(v: i64, to: ElementKind) -> ScalarExpr {
    ScalarExpr::Cast(
        Box::new(ScalarExpr::ConstInt(v, U32)),
        CastSpec {
            to,
            round: RoundMode::NearestEven,
            sat: Saturation::Wrap,
        },
    )
}

fn raw(off: i64, elem: ElementKind) -> Place {
    Place::Raw {
        buffer: 0,
        byte_offset: ScalarExpr::ConstInt(off, ElementKind::I32),
        elem,
    }
}

fn out_cell(buffer: u32) -> Place {
    Place::Logical {
        buffer,
        coords: vec![ScalarExpr::ConstInt(0, U32)],
    }
}

fn launch_of(k: &KernelFn, args: &mut [KernelArg]) -> Result<(), KernelTrap> {
    run_launch(k, args).map(|_| ())
}

/// One thread loads `elem` at byte `off` of `src` into local 0 and stores it to element 0 of an `elem` output buffer.
fn load_at(src: TypedBuffer, off: i64, elem: ElementKind) -> Result<u128, KernelTrap> {
    let k = KernelFn::new(
        "raw_load",
        F32,
        vec![
            Param::buffer("src", U32, BufferRole::Read, Some(1)),
            Param::buffer("out", elem, BufferRole::Write, Some(1)),
        ],
        vec![LocalDecl::scalar(elem)],
        Block(vec![
            Stmt::Load {
                local: 0,
                place: raw(off, elem),
            },
            Stmt::Store {
                place: out_cell(1),
                value: ScalarExpr::Local(0),
            },
        ]),
    )
    .with_launch(LaunchShape::new([1, 1, 1], [1, 1, 1]));
    check_kernel(&k).map_err(KernelTrap::Malformed)?;
    let n = unpopped::types::raw_access_bytes(elem).expect("a whole-byte type") as usize;
    let mut args = vec![
        KernelArg::Buffer(src),
        KernelArg::Buffer(TypedBuffer::new(elem, vec![1], vec![1], vec![0; n])),
    ];
    launch_of(&k, &mut args)?;
    let KernelArg::Buffer(o) = &args[1] else {
        unreachable!("a buffer")
    };
    Ok(o.bytes
        .iter()
        .rev()
        .fold(0u128, |a, b| (a << 8) | u128::from(*b)))
}

#[test]
fn a_raw_load_reads_the_named_bytes_at_the_named_width() {
    assert_eq!(load_at(source(), 5, U8), Ok(0x06));
    assert_eq!(load_at(source(), 6, U16), Ok(0x0807));
    assert_eq!(load_at(source(), 4, U32), Ok(0x0807_0605));
    assert_eq!(load_at(source(), 8, U64), Ok(0x100F_0E0D_0C0B_0A09));
    // the last byte, and the last aligned word: the bound is the buffer's bytes, inclusive of the end
    assert_eq!(load_at(source(), 15, U8), Ok(0x10));
    assert_eq!(load_at(source(), 12, U32), Ok(0x100F_0E0D));
}

#[test]
fn byte_zero_is_the_views_first_element_not_the_storage_start() {
    // a view of 3 elements starting one element in: byte 0 is storage byte 4
    let mut view = source();
    view.shape = vec![3];
    view.base_offset = 1;
    assert_eq!(load_at(view.clone(), 0, U32), Ok(0x0807_0605));
    // the bound is the storage's end, not the view's shape (raw ignores shape): 12 bytes remain
    assert_eq!(load_at(view.clone(), 8, U32), Ok(0x100F_0E0D));
    assert_eq!(
        load_at(view, 12, U32),
        Err(KernelTrap::OutOfBounds { buffer: 0 })
    );
}

#[test]
fn a_misaligned_or_negative_offset_is_a_typed_trap() {
    let bad = |off| KernelTrap::MisalignedAccess {
        buffer: 0,
        byte_offset: off,
    };
    assert_eq!(load_at(source(), 2, U32), Err(bad(2)));
    assert_eq!(load_at(source(), 1, U16), Err(bad(1)));
    assert_eq!(load_at(source(), 4, U64), Err(bad(4)));
    assert_eq!(load_at(source(), -4, U32), Err(bad(-4)));
}

#[test]
fn an_access_past_the_buffers_bytes_is_out_of_bounds() {
    let oob = Err(KernelTrap::OutOfBounds { buffer: 0 });
    assert_eq!(load_at(source(), 16, U8), oob);
    assert_eq!(load_at(source(), 16, U32), oob);
    assert_eq!(load_at(source(), 8 + 8, U64), oob);
    // not misaligned, only too wide for what remains
    assert_eq!(load_at(source(), 8, U64), Ok(0x100F_0E0D_0C0B_0A09));
}

#[test]
fn a_raw_store_writes_only_its_bytes_little_endian() {
    let k = KernelFn::new(
        "raw_store",
        F32,
        vec![Param::buffer("buf", U32, BufferRole::ReadWrite, Some(1))],
        vec![],
        Block(vec![Stmt::Store {
            place: raw(2, U16),
            value: narrow(0xBEEF, U16),
        }]),
    )
    .with_launch(LaunchShape::new([1, 1, 1], [1, 1, 1]));
    assert_eq!(check_kernel(&k), Ok(()));
    let mut args = vec![KernelArg::Buffer(source())];
    assert_eq!(launch_of(&k, &mut args), Ok(()));
    let KernelArg::Buffer(b) = &args[0] else {
        unreachable!("a buffer")
    };
    let expect: Vec<u8> = (1..=16u8)
        .map(|v| match v {
            3 => 0xEF,
            4 => 0xBE,
            v => v,
        })
        .collect();
    assert_eq!(b.bytes, expect);
}

/// Two threads: thread 0 runs `a`, thread 1 runs `b`.
fn two_threads(a: Stmt, b: Stmt) -> Result<(), KernelTrap> {
    let tid_is = |v| {
        ScalarExpr::Binary(
            BinaryOp::CmpEq,
            Box::new(ScalarExpr::Exec(ExecId::ThreadIdx(0))),
            Box::new(ScalarExpr::ConstInt(v, U32)),
        )
    };
    let only = |v, st| Stmt::If {
        cond: tid_is(v),
        then: Block(vec![st]),
        els: Block(vec![]),
    };
    let k = KernelFn::new(
        "raw_race",
        F32,
        vec![Param::buffer("buf", U32, BufferRole::ReadWrite, Some(1))],
        vec![LocalDecl::scalar(U8)],
        Block(vec![only(0, a), only(1, b)]),
    )
    .with_launch(LaunchShape::new([2, 1, 1], [1, 1, 1]));
    check_kernel(&k).map_err(KernelTrap::Malformed)?;
    launch_of(&k, &mut [KernelArg::Buffer(source())])
}

fn logical_store(v: i64) -> Stmt {
    Stmt::Store {
        place: Place::Logical {
            buffer: 0,
            coords: vec![ScalarExpr::ConstInt(1, U32)],
        },
        value: ScalarExpr::ConstInt(v, U32),
    }
}

fn raw_load_u8(off: i64) -> Stmt {
    Stmt::Load {
        local: 0,
        place: raw(off, U8),
    }
}

#[test]
fn a_raw_read_of_a_byte_another_thread_wrote_through_the_element_races() {
    // element 1 is bytes 4..8: thread 0 writes it, thread 1 reads byte 7 of it
    assert_eq!(
        two_threads(logical_store(9), raw_load_u8(7)),
        Err(KernelTrap::DataRace)
    );
    assert_eq!(
        two_threads(logical_store(9), raw_load_u8(4)),
        Err(KernelTrap::DataRace)
    );
}

#[test]
fn a_raw_read_of_a_neighbouring_element_does_not_race() {
    // bytes 3 and 8 belong to elements 0 and 2; thread 0 wrote element 1 only
    assert_eq!(two_threads(logical_store(9), raw_load_u8(3)), Ok(()));
    assert_eq!(two_threads(logical_store(9), raw_load_u8(8)), Ok(()));
}

#[test]
fn a_wide_raw_access_covers_every_element_its_bytes_touch() {
    // a 4-byte access at byte 4 is element 1 exactly; at byte 6 it would be misaligned, so use a U64 over elements 0 and 1
    let wide = Stmt::Load {
        local: 0,
        place: raw(0, U64),
    };
    let r = {
        let tid_is = |v| {
            ScalarExpr::Binary(
                BinaryOp::CmpEq,
                Box::new(ScalarExpr::Exec(ExecId::ThreadIdx(0))),
                Box::new(ScalarExpr::ConstInt(v, U32)),
            )
        };
        let only = |v, st| Stmt::If {
            cond: tid_is(v),
            then: Block(vec![st]),
            els: Block(vec![]),
        };
        let k = KernelFn::new(
            "raw_wide_race",
            F32,
            vec![Param::buffer("buf", U32, BufferRole::ReadWrite, Some(1))],
            vec![LocalDecl::scalar(U64)],
            Block(vec![only(0, wide), only(1, logical_store(9))]),
        )
        .with_launch(LaunchShape::new([2, 1, 1], [1, 1, 1]));
        assert_eq!(check_kernel(&k), Ok(()));
        launch_of(&k, &mut [KernelArg::Buffer(source())])
    };
    // the U64 read at byte 0 touches elements 0 and 1; thread 1 writes element 1
    assert_eq!(r, Err(KernelTrap::DataRace));
}

#[test]
fn two_raw_byte_accesses_to_one_wider_element_are_reported_conservatively() {
    // pinned, documented limit: the detector works at the buffer's element granularity, so bytes 4 and 5 (different bytes,
    // one u32 element) race although the hardware would not
    let store = |off| Stmt::Store {
        place: raw(off, U8),
        value: narrow(1, U8),
    };
    assert_eq!(two_threads(store(4), store(5)), Err(KernelTrap::DataRace));
    // bytes in different elements are independent
    assert_eq!(two_threads(store(3), store(4)), Ok(()));
}

#[test]
fn the_checker_applies_the_buffers_role_and_refuses_what_has_no_raw_rules() {
    let with = |role, st| {
        KernelFn::new(
            "k",
            F32,
            vec![Param::buffer("buf", U32, role, Some(1))],
            vec![LocalDecl::scalar(U32)],
            Block(vec![st]),
        )
        .with_launch(LaunchShape::new([1, 1, 1], [1, 1, 1]))
    };
    let store = Stmt::Store {
        place: raw(0, U32),
        value: ScalarExpr::ConstInt(1, U32),
    };
    assert_eq!(
        check_kernel(&with(BufferRole::Read, store.clone())),
        Err(KernelError::WriteToReadOnly { buffer: 0 })
    );
    assert_eq!(check_kernel(&with(BufferRole::ReadWrite, store)), Ok(()));
    let load = Stmt::Load {
        local: 0,
        place: raw(0, U32),
    };
    assert_eq!(
        check_kernel(&with(BufferRole::Write, load)),
        Err(KernelError::ReadFromWriteOnly { buffer: 0 })
    );
    let atomic = Stmt::AtomicRmw {
        op: AtomicOp::Add,
        place: raw(0, U32),
        value: ScalarExpr::ConstInt(1, U32),
        old: None,
        order: unpopped::kernel::MemoryOrder::Relaxed,
        scope: unpopped::exec::Scope::Device,
    };
    assert!(matches!(
        check_kernel(&with(BufferRole::ReadWrite, atomic)),
        Err(KernelError::TypeMismatch { .. })
    ));
}

#[test]
fn the_checker_refuses_a_fractional_element_and_a_non_integer_byte_offset() {
    let with = |role, st| {
        KernelFn::new(
            "k",
            F32,
            vec![Param::buffer("buf", U32, role, Some(1))],
            vec![LocalDecl::scalar(U32)],
            Block(vec![st]),
        )
        .with_launch(LaunchShape::new([1, 1, 1], [1, 1, 1]))
    };
    // an element that is not a whole number of bytes picks a layout: refused by name
    let fp8 = Stmt::Load {
        local: 0,
        place: raw(0, ElementKind::Fp8E4M3FN),
    };
    assert!(matches!(
        check_kernel(&with(BufferRole::Read, fp8)),
        Err(KernelError::TypeMismatch { .. })
    ));
    // a byte offset that is not an integer
    let fl = Stmt::Load {
        local: 0,
        place: Place::Raw {
            buffer: 0,
            byte_offset: ScalarExpr::Const(0.0),
            elem: U32,
        },
    };
    assert!(check_kernel(&with(BufferRole::Read, fl)).is_err());
}
