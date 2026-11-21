// SPDX-License-Identifier: MIT OR Apache-2.0
//! **`CopyAsync` from a `Place::Raw` source** (ROADMAP U9c slice 2b, `docs/options-u9c-pointer-values.md`). The source buffer is
//! 32 bytes holding 1, 2, 3 ... 32, so the bytes read, their order and the offset all show in the values that land. `elems`
//! counts values of the raw place's type; the bytes read are `elems * size`, aligned to one value and inside the buffer.

use unpopped::exec::{AllocDecl, ExecId, Extent, LaunchShape, MemorySpace, Scope};
use unpopped::ir::{BinaryOp, ScalarExpr};
use unpopped::kernel::{
    BarrierKind, Block, BufferRole, CacheHint, KernelError, KernelFn, LocalDecl, Param, Place, Stmt,
    check_kernel,
};
use unpopped::oracle::TypedBuffer;
use unpopped::oracle::kernel_eval::{KernelArg, KernelTrap};
use unpopped::oracle::kernel_launch::run_launch;
use unpopped_vocab::ElementKind;

const U8: ElementKind = ElementKind::U8;
const U32: ElementKind = ElementKind::U32;
const F32: ElementKind = ElementKind::F32;

fn cu(v: i64) -> ScalarExpr {
    ScalarExpr::ConstInt(v, U32)
}

fn raw(off: i64, elem: ElementKind) -> Place {
    Place::Raw {
        buffer: 0,
        byte_offset: ScalarExpr::ConstInt(off, ElementKind::I32),
        elem,
    }
}

fn shared(i: i64) -> Place {
    Place::Alloc {
        alloc: 0,
        index: cu(i),
    }
}

fn copy(dst: Place, src: Place, elems: u32, valid: Option<ScalarExpr>) -> Stmt {
    Stmt::CopyAsync {
        valid,
        cache: CacheHint::Auto,
        l2_prefetch: None,
        dst,
        src,
        elems,
    }
}

/// The bytes 1..=32.
fn bytes32() -> TypedBuffer {
    TypedBuffer::new(U8, vec![32], vec![1], (1..=32u8).collect())
}

/// One thread: `body`, commit, wait for everything, then the eight `Shared` U32 elements go to `out`.
fn run(body: Vec<Stmt>) -> Result<Vec<u32>, KernelTrap> {
    // zero the whole array first, so an element no copy lands on reads as 0 rather than trapping uninitialized
    let mut stmts: Vec<Stmt> = (0..8)
        .map(|i| Stmt::Store {
            place: shared(i),
            value: cu(0),
        })
        .collect();
    stmts.extend(body);
    stmts.push(Stmt::AsyncCommit);
    stmts.push(Stmt::AsyncWait { pending: 0 });
    for i in 0..8 {
        stmts.push(Stmt::Load {
            local: i as u32,
            place: shared(i),
        });
        stmts.push(Stmt::Store {
            place: Place::Logical {
                buffer: 1,
                coords: vec![cu(i)],
            },
            value: ScalarExpr::Local(i as u32),
        });
    }
    let k = KernelFn::new(
        "k",
        F32,
        vec![
            Param::buffer("src", U8, BufferRole::Read, Some(1)),
            Param::buffer("out", U32, BufferRole::Write, Some(1)),
        ],
        (0..8).map(|_| LocalDecl::scalar(U32)).collect(),
        Block(stmts),
    )
    .with_allocs(vec![AllocDecl::new(
        "s",
        MemorySpace::Shared,
        U32,
        Extent::Static(8),
    )])
    .with_launch(LaunchShape::new([1, 1, 1], [1, 1, 1]));
    check_kernel(&k).map_err(KernelTrap::Malformed)?;
    let mut args = [
        KernelArg::Buffer(bytes32()),
        KernelArg::Buffer(TypedBuffer::new(U32, vec![8], vec![1], vec![0; 32])),
    ];
    run_launch(&k, &mut args)?;
    let KernelArg::Buffer(o) = &args[1] else {
        unreachable!("a buffer")
    };
    Ok(o.bytes
        .chunks(4)
        .map(|c| u32::from_le_bytes(c.try_into().unwrap()))
        .collect())
}

/// The little-endian word whose bytes are `b, b+1, b+2, b+3` (the source holds `i + 1` at byte `i`).
fn word_at(byte: u32) -> u32 {
    (0..4).fold(0, |a, i| a | (byte + i + 1) << (8 * i))
}

#[test]
fn a_raw_source_lands_consecutive_values_from_the_named_bytes() {
    // four U32 values from byte 4 land at Shared elements 2..6
    let got = run(vec![copy(shared(2), raw(4, U32), 4, None)]).unwrap();
    assert_eq!(
        got,
        vec![
            0,
            0,
            word_at(4),
            word_at(8),
            word_at(12),
            word_at(16),
            0,
            0
        ]
    );
    // the last 16 bytes of the buffer are in range
    let end = run(vec![copy(shared(4), raw(16, U32), 4, None)]).unwrap();
    assert_eq!(&end[4..], &[word_at(16), word_at(20), word_at(24), word_at(28)]);
}

#[test]
fn a_misaligned_or_out_of_range_raw_source_is_a_typed_trap() {
    assert_eq!(
        run(vec![copy(shared(0), raw(2, U32), 4, None)]),
        Err(KernelTrap::MisalignedAccess {
            buffer: 0,
            byte_offset: 2
        })
    );
    // 20 + 16 bytes > 32
    assert_eq!(
        run(vec![copy(shared(0), raw(20, U32), 4, None)]),
        Err(KernelTrap::OutOfBounds { buffer: 0 })
    );
}

#[test]
fn a_valid_count_copies_that_many_values_and_zero_fills_the_rest() {
    let got = run(vec![copy(shared(0), raw(0, U32), 4, Some(cu(2)))]).unwrap();
    assert_eq!(&got[..4], &[word_at(0), word_at(4), 0, 0]);
}

/// Thread 0 copies bytes 0..16 of a read-write source; thread 1 stores `Logical` element `elem` of it. No barrier between.
fn race_with_write_to(elem: i64) -> Result<(), KernelTrap> {
    let tid_is = |v| {
        ScalarExpr::Binary(
            BinaryOp::CmpEq,
            Box::new(ScalarExpr::Exec(ExecId::ThreadIdx(0))),
            Box::new(cu(v)),
        )
    };
    let only = |v, body: Vec<Stmt>| Stmt::If {
        cond: tid_is(v),
        then: Block(body),
        els: Block(vec![]),
    };
    let k = KernelFn::new(
        "race",
        F32,
        vec![Param::buffer("src", U32, BufferRole::ReadWrite, Some(1))],
        vec![],
        Block(vec![
            only(
                0,
                vec![
                    copy(shared(0), raw(0, U32), 4, None),
                    Stmt::AsyncCommit,
                    Stmt::AsyncWait { pending: 0 },
                ],
            ),
            only(
                1,
                vec![Stmt::Store {
                    place: Place::Logical {
                        buffer: 0,
                        coords: vec![cu(elem)],
                    },
                    value: cu(9),
                }],
            ),
        ]),
    )
    .with_allocs(vec![AllocDecl::new(
        "s",
        MemorySpace::Shared,
        U32,
        Extent::Static(8),
    )])
    .with_launch(LaunchShape::new([2, 1, 1], [1, 1, 1]));
    check_kernel(&k).map_err(KernelTrap::Malformed)?;
    let mut args = [KernelArg::Buffer(TypedBuffer::new(
        U32,
        vec![8],
        vec![1],
        (0..32u8).collect(),
    ))];
    run_launch(&k, &mut args).map(|_| ())
}

#[test]
fn every_element_a_raw_source_reads_is_logged_for_the_race_detector() {
    // the copy reads elements 0..4; another thread writing element 3 (its last) races, element 4 does not
    assert_eq!(race_with_write_to(3), Err(KernelTrap::DataRace));
    assert_eq!(race_with_write_to(0), Err(KernelTrap::DataRace));
    assert_eq!(race_with_write_to(4), Ok(()));
}

#[test]
fn the_checker_takes_a_raw_source_of_the_destinations_type_and_nothing_else() {
    let k = |dst, src| {
        KernelFn::new(
            "k",
            F32,
            vec![Param::buffer("src", U8, BufferRole::Read, Some(1))],
            vec![],
            Block(vec![copy(dst, src, 4, None)]),
        )
        .with_allocs(vec![AllocDecl::new(
            "s",
            MemorySpace::Shared,
            U32,
            Extent::Static(8),
        )])
        .with_launch(LaunchShape::new([1, 1, 1], [1, 1, 1]))
    };
    assert_eq!(check_kernel(&k(shared(0), raw(0, U32))), Ok(()));
    // an element type that is not the destination's
    assert!(matches!(
        check_kernel(&k(shared(0), raw(0, F32))),
        Err(KernelError::TypeMismatch { .. })
    ));
    // a raw destination is not a Shared allocation place
    assert!(matches!(
        check_kernel(&k(raw(0, U32), raw(0, U32))),
        Err(KernelError::AsyncUnsupported { .. })
    ));
    // a byte-addressed allocation destination is declined by name too (the landing of a pun is not modelled)
    let araw = Place::AllocRaw {
        alloc: 0,
        byte_offset: ScalarExpr::ConstInt(0, ElementKind::I32),
        elem: U32,
    };
    assert!(matches!(
        check_kernel(&k(araw, raw(0, U32))),
        Err(KernelError::AsyncUnsupported { .. })
    ));
    // 2 values of 4 bytes is 8 bytes (fine), 3 is not a legal size
    let three = KernelFn::new(
        "k",
        F32,
        vec![Param::buffer("src", U8, BufferRole::Read, Some(1))],
        vec![],
        Block(vec![copy(shared(0), raw(0, U32), 3, None)]),
    )
    .with_allocs(vec![AllocDecl::new(
        "s",
        MemorySpace::Shared,
        U32,
        Extent::Static(8),
    )])
    .with_launch(LaunchShape::new([1, 1, 1], [1, 1, 1]));
    assert!(matches!(
        check_kernel(&three),
        Err(KernelError::AsyncUnsupported { .. })
    ));
    let _ = (Scope::Block, BarrierKind::Full);
}
