// SPDX-License-Identifier: MIT OR Apache-2.0
//! **`Place::AllocRaw`: a typed access at a byte offset of a `Shared` or `Private` allocation** (ROADMAP U9c slice 2,
//! `docs/options-u9c-pointer-values.md`). The allocation is filled with the bytes 1, 2, 3 ... 16 (four little-endian `u32`), so
//! a wrong offset, width or byte order shows in the value read. An element is initialized only once a store covers all of it.

use unpopped::exec::{AllocDecl, ExecId, Extent, LaunchShape, MemorySpace, Scope};
use unpopped::ir::{BinaryOp, CastSpec, RoundMode, Saturation, ScalarExpr};
use unpopped::kernel::{
    Block, BufferRole, KernelError, KernelFn, LocalDecl, Param, Place, Stmt, check_kernel,
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

fn araw(off: i64, elem: ElementKind) -> Place {
    Place::AllocRaw {
        alloc: 0,
        byte_offset: ScalarExpr::ConstInt(off, ElementKind::I32),
        elem,
    }
}

fn cell(i: i64) -> Place {
    Place::Alloc {
        alloc: 0,
        index: ScalarExpr::ConstInt(i, U32),
    }
}

fn out_cell() -> Place {
    Place::Logical {
        buffer: 0,
        coords: vec![ScalarExpr::ConstInt(0, U32)],
    }
}

/// Stores 0x04030201, 0x08070605, 0x0C0B0A09, 0x100F0E0D to elements 0..4 of allocation 0.
fn fill() -> Vec<Stmt> {
    [0x0403_0201i64, 0x0807_0605, 0x0C0B_0A09, 0x100F_0E0D]
        .iter()
        .enumerate()
        .map(|(i, v)| Stmt::Store {
            place: cell(i as i64),
            value: ScalarExpr::ConstInt(*v, U32),
        })
        .collect()
}

/// One thread runs `body` over a 4-element `U32` allocation in `space`, then loads `elem` at byte `off` and writes it to `out`.
fn run_in(
    space: MemorySpace,
    body: Vec<Stmt>,
    off: i64,
    elem: ElementKind,
) -> Result<u128, KernelTrap> {
    let mut stmts = body;
    stmts.push(Stmt::Load {
        local: 0,
        place: araw(off, elem),
    });
    stmts.push(Stmt::Store {
        place: out_cell(),
        value: ScalarExpr::Local(0),
    });
    let k = KernelFn::new(
        "alloc_raw",
        F32,
        vec![Param::buffer("out", elem, BufferRole::Write, Some(1))],
        vec![LocalDecl::scalar(elem)],
        Block(stmts),
    )
    .with_launch(LaunchShape::new([1, 1, 1], [1, 1, 1]))
    .with_allocs(vec![AllocDecl::new("a", space, U32, Extent::Static(4))]);
    check_kernel(&k).map_err(KernelTrap::Malformed)?;
    let n = unpopped::types::raw_access_bytes(elem).expect("a whole-byte type") as usize;
    let mut args = vec![KernelArg::Buffer(TypedBuffer::new(
        elem,
        vec![1],
        vec![1],
        vec![0; n],
    ))];
    run_launch(&k, &mut args).map(|_| ())?;
    let KernelArg::Buffer(o) = &args[0] else {
        unreachable!("a buffer")
    };
    Ok(o.bytes
        .iter()
        .rev()
        .fold(0u128, |a, b| (a << 8) | u128::from(*b)))
}

fn run(body: Vec<Stmt>, off: i64, elem: ElementKind) -> Result<u128, KernelTrap> {
    run_in(MemorySpace::Shared, body, off, elem)
}

#[test]
fn a_raw_load_from_an_allocation_reads_the_named_bytes_at_the_named_width() {
    for space in [MemorySpace::Shared, MemorySpace::Private] {
        assert_eq!(run_in(space, fill(), 5, U8), Ok(0x06));
        assert_eq!(run_in(space, fill(), 6, U16), Ok(0x0807));
        assert_eq!(run_in(space, fill(), 4, U32), Ok(0x0807_0605));
        assert_eq!(run_in(space, fill(), 8, U64), Ok(0x100F_0E0D_0C0B_0A09));
        assert_eq!(run_in(space, fill(), 15, U8), Ok(0x10));
    }
}

#[test]
fn a_raw_store_into_an_allocation_lands_little_endian_and_a_wide_one_covers_two_elements() {
    // a U64 store at byte 8 covers elements 2 and 3 completely; a U32 load of element 3 sees its high half
    let body = vec![
        Stmt::Store {
            place: cell(0),
            value: ScalarExpr::ConstInt(0, U32),
        },
        Stmt::Store {
            place: cell(1),
            value: ScalarExpr::ConstInt(0, U32),
        },
        Stmt::Store {
            place: araw(8, U64),
            value: ScalarExpr::Cast(
                Box::new(ScalarExpr::ConstInt(
                    0x1122_3344_5566_7788,
                    ElementKind::I64,
                )),
                CastSpec {
                    to: U64,
                    round: RoundMode::NearestEven,
                    sat: Saturation::Wrap,
                },
            ),
        },
    ];
    assert_eq!(run(body.clone(), 12, U32), Ok(0x1122_3344));
    assert_eq!(run(body, 8, U32), Ok(0x5566_7788));
}

#[test]
fn a_misaligned_or_out_of_range_access_is_a_typed_trap() {
    let bad = |off| KernelTrap::MisalignedAccess {
        buffer: 0,
        byte_offset: off,
    };
    assert_eq!(run(fill(), 2, U32), Err(bad(2)));
    assert_eq!(run(fill(), 1, U16), Err(bad(1)));
    assert_eq!(run(fill(), -4, U32), Err(bad(-4)));
    let oob = Err(KernelTrap::AllocOutOfBounds { alloc: 0 });
    assert_eq!(run(fill(), 16, U8), oob);
    assert_eq!(run(fill(), 16, U32), oob);
    assert_eq!(run(fill(), 12, U64), Err(bad(12)));
    // not misaligned, only too wide for what remains
    assert_eq!(run(fill(), 8, U64), Ok(0x100F_0E0D_0C0B_0A09));
}

#[test]
fn an_element_is_initialized_only_when_a_store_covers_all_of_it() {
    let uninit = Err(KernelTrap::UninitializedShared { alloc: 0 });
    // nothing stored: any raw load traps
    assert_eq!(run(vec![], 0, U8), uninit);
    // one byte store into element 0 leaves it uninitialized (the other three bytes are unspecified): conservative, pinned
    let byte = |off| Stmt::Store {
        place: araw(off, U8),
        value: narrow(7, U8),
    };
    assert_eq!(run(vec![byte(0)], 0, U8), uninit);
    // a whole-word raw store initializes it, and a byte load then reads one byte of it
    let word = Stmt::Store {
        place: araw(0, U32),
        value: ScalarExpr::ConstInt(0x0403_0201, U32),
    };
    assert_eq!(run(vec![word.clone()], 2, U8), Ok(0x03));
    // a U64 load over a stored element and an unstored one traps
    assert_eq!(run(vec![word], 0, U64), uninit);
    // and the Private flavour names Private
    assert_eq!(
        run_in(MemorySpace::Private, vec![], 0, U8),
        Err(KernelTrap::UninitializedPrivate { alloc: 0 })
    );
}

/// Block of two threads: both fill the allocation (thread 0 does the stores), a barrier, then thread 0 runs `a` and thread 1 `b`.
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
    let mut stmts = vec![only(
        0,
        Stmt::If {
            cond: ScalarExpr::ConstInt(1, U32),
            then: Block(fill()),
            els: Block(vec![]),
        },
    )];
    stmts.push(Stmt::barrier(Scope::Block));
    stmts.push(only(0, a));
    stmts.push(only(1, b));
    let k = KernelFn::new(
        "alloc_raw_race",
        F32,
        vec![],
        vec![LocalDecl::scalar(U8)],
        Block(stmts),
    )
    .with_launch(LaunchShape::new([2, 1, 1], [1, 1, 1]))
    .with_allocs(vec![AllocDecl::new(
        "a",
        MemorySpace::Shared,
        U32,
        Extent::Static(4),
    )]);
    check_kernel(&k).map_err(KernelTrap::Malformed)?;
    run_launch(&k, &mut []).map(|_| ())
}

fn elem_store(i: i64) -> Stmt {
    Stmt::Store {
        place: cell(i),
        value: ScalarExpr::ConstInt(9, U32),
    }
}

fn raw_load_u8(off: i64) -> Stmt {
    Stmt::Load {
        local: 0,
        place: araw(off, U8),
    }
}

#[test]
fn a_raw_read_of_a_byte_of_an_element_another_thread_wrote_races() {
    // element 1 is bytes 4..8
    assert_eq!(
        two_threads(elem_store(1), raw_load_u8(7)),
        Err(KernelTrap::DataRace)
    );
    assert_eq!(
        two_threads(elem_store(1), raw_load_u8(4)),
        Err(KernelTrap::DataRace)
    );
    // a neighbouring element's byte does not
    assert_eq!(two_threads(elem_store(1), raw_load_u8(3)), Ok(()));
    assert_eq!(two_threads(elem_store(1), raw_load_u8(8)), Ok(()));
}

#[test]
fn a_wide_raw_access_covers_every_allocation_element_its_bytes_touch() {
    let wide = Stmt::Load {
        local: 0,
        place: araw(0, U64),
    };
    let k_local = |a, b| {
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
        let mut stmts = vec![only(
            0,
            Stmt::If {
                cond: ScalarExpr::ConstInt(1, U32),
                then: Block(fill()),
                els: Block(vec![]),
            },
        )];
        stmts.push(Stmt::barrier(Scope::Block));
        stmts.push(only(0, a));
        stmts.push(only(1, b));
        KernelFn::new(
            "wide",
            F32,
            vec![],
            vec![LocalDecl::scalar(U64)],
            Block(stmts),
        )
        .with_launch(LaunchShape::new([2, 1, 1], [1, 1, 1]))
        .with_allocs(vec![AllocDecl::new(
            "a",
            MemorySpace::Shared,
            U32,
            Extent::Static(4),
        )])
    };
    // thread 0 reads bytes 0..8 (elements 0 and 1); thread 1 writes element 1
    let k = k_local(wide, elem_store(1));
    assert_eq!(check_kernel(&k), Ok(()));
    assert_eq!(
        run_launch(&k, &mut []).map(|_| ()),
        Err(KernelTrap::DataRace)
    );
}

#[test]
fn the_checker_refuses_constant_tensor_and_what_has_no_raw_rules() {
    let k = |space, st| {
        KernelFn::new(
            "k",
            F32,
            vec![],
            vec![LocalDecl::scalar(U32)],
            Block(vec![st]),
        )
        .with_launch(LaunchShape::new([1, 1, 1], [1, 1, 1]))
        .with_allocs(vec![AllocDecl::new("a", space, U32, Extent::Static(4))])
    };
    let load = Stmt::Load {
        local: 0,
        place: araw(0, U32),
    };
    assert_eq!(check_kernel(&k(MemorySpace::Shared, load.clone())), Ok(()));
    assert!(matches!(
        check_kernel(&k(MemorySpace::Constant, load.clone())),
        Err(KernelError::TypeMismatch { .. })
    ));
    assert!(matches!(
        check_kernel(&k(MemorySpace::Tensor, load)),
        Err(KernelError::Tcgen05Unsupported { .. })
    ));
    // an element that is not a whole number of bytes
    let fp8 = Stmt::Load {
        local: 0,
        place: araw(0, ElementKind::Fp8E4M3FN),
    };
    assert!(matches!(
        check_kernel(&k(MemorySpace::Shared, fp8)),
        Err(KernelError::TypeMismatch { .. })
    ));
    // an alloc that does not exist
    let ghost = Stmt::Load {
        local: 0,
        place: Place::AllocRaw {
            alloc: 3,
            byte_offset: ScalarExpr::ConstInt(0, U32),
            elem: U32,
        },
    };
    assert_eq!(
        check_kernel(&k(MemorySpace::Shared, ghost)),
        Err(KernelError::BadPlace { buffer: 3 })
    );
    // an atomic on it has no raw rules
    let atomic = Stmt::atomic_rmw(
        unpopped::kernel::AtomicOp::Add,
        araw(0, U32),
        ScalarExpr::ConstInt(1, U32),
    );
    assert!(matches!(
        check_kernel(&k(MemorySpace::Shared, atomic)),
        Err(KernelError::TypeMismatch { .. })
    ));
}
