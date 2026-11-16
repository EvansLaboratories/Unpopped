// SPDX-License-Identifier: MIT OR Apache-2.0
//! **The `For` unroll hint (ROADMAP U18) and the `CopyAsync` forms (U13)**. The hint is non-semantic: every value of it
//! gives the same result. The copy forms: a `valid` element count (`cp.async ... src-size`: the first `valid` elements
//! are read, the rest of the destination is zero-filled, and `valid` 0 reads nothing at all, which is how a predicated
//! copy is spelled), a cache hint (`.ca`, `.cg`; the oracle ignores it, the checker holds `.cg` to 16 bytes) and an L2
//! prefetch size (64, 128 or 256). Each form has a program that shows it and a control that does not.

use unpopped::exec::{AllocDecl, Extent, LaunchShape, MemorySpace};
use unpopped::ir::ScalarExpr;
use unpopped::kernel::{
    Block, BufferRole, CacheHint, KernelError, KernelFn, LocalDecl, Param, Place, Stmt, Unroll,
    check_kernel,
};
use unpopped::oracle::TypedBuffer;
use unpopped::oracle::kernel_eval::{KernelArg, KernelTrap};
use unpopped::oracle::kernel_launch::run_launch;
use unpopped_vocab::ElementKind;

const U32: ElementKind = ElementKind::U32;

fn u(v: i64) -> ScalarExpr {
    ScalarExpr::ConstInt(v, U32)
}
fn tile(i: ScalarExpr) -> Place {
    Place::Alloc { alloc: 0, index: i }
}
fn buf(buffer: u32, i: ScalarExpr) -> Place {
    Place::Logical {
        buffer,
        coords: vec![i],
    }
}

/// The copy under test, then a loop (with the unroll hint under test) that copies the tile to `out`.
fn kernel(copy: Stmt, unroll: Unroll) -> KernelFn {
    let body = vec![
        copy,
        Stmt::AsyncCommit,
        Stmt::AsyncWait { pending: 0 },
        Stmt::For {
            local: 0,
            start: u(0),
            end: u(4),
            step: u(1),
            unroll,
            body: Block(vec![
                Stmt::Load {
                    local: 1,
                    place: tile(ScalarExpr::Local(0)),
                },
                Stmt::Store {
                    place: buf(1, ScalarExpr::Local(0)),
                    value: ScalarExpr::Local(1),
                },
            ]),
        },
    ];
    let mut k = KernelFn::new(
        "forms",
        U32,
        vec![
            Param::buffer("in", U32, BufferRole::Read, Some(1)),
            Param::buffer("out", U32, BufferRole::ReadWrite, Some(1)),
        ],
        vec![LocalDecl::scalar(U32), LocalDecl::scalar(U32)],
        Block(body),
    )
    .with_launch(LaunchShape::new([1, 1, 1], [1, 1, 1]));
    k.allocs = vec![AllocDecl::new(
        "tile",
        MemorySpace::Shared,
        U32,
        Extent::Static(4),
    )];
    k
}

fn copy(src: i64, elems: u32, valid: Option<i64>, cache: CacheHint, l2: Option<u32>) -> Stmt {
    Stmt::CopyAsync {
        dst: tile(u(0)),
        src: buf(0, u(src)),
        elems,
        valid: valid.map(u),
        cache,
        l2_prefetch: l2,
    }
}
fn plain() -> Stmt {
    copy(0, 4, None, CacheHint::Auto, None)
}

fn run(k: &KernelFn) -> Result<Vec<u32>, KernelTrap> {
    assert_eq!(check_kernel(k), Ok(()));
    let words: Vec<u8> = [1u32, 2, 4, 8]
        .iter()
        .flat_map(|w| w.to_le_bytes())
        .collect();
    let mut args = [
        KernelArg::Buffer(TypedBuffer::new(U32, vec![4], vec![1], words)),
        KernelArg::Buffer(TypedBuffer::new(U32, vec![4], vec![1], vec![0; 16])),
    ];
    run_launch(k, &mut args)?;
    let KernelArg::Buffer(b) = &args[1] else {
        unreachable!("a buffer")
    };
    Ok(b.bytes
        .chunks(4)
        .map(|c| u32::from_le_bytes(c.try_into().unwrap()))
        .collect())
}

#[test]
fn the_unroll_hint_does_not_change_the_result() {
    for h in [
        Unroll::None,
        Unroll::Full,
        Unroll::Count(1),
        Unroll::Count(4),
    ] {
        assert_eq!(run(&kernel(plain(), h)), Ok(vec![1, 2, 4, 8]), "{h:?}");
    }
}

#[test]
fn the_unroll_hint_survives_serialisation_and_a_zero_count_is_refused() {
    let k = kernel(plain(), Unroll::Count(4));
    let json = serde_json::to_string(&k).unwrap();
    let back: KernelFn = serde_json::from_str(&json).unwrap();
    assert_eq!(back, k);
    assert!(json.contains("Count"), "the hint is in the serial form");
    assert!(matches!(
        check_kernel(&kernel(plain(), Unroll::Count(0))),
        Err(KernelError::UnrollUnsupported { .. })
    ));
}

#[test]
fn valid_reads_that_many_elements_and_zero_fills_the_rest() {
    let two = kernel(copy(0, 4, Some(2), CacheHint::Auto, None), Unroll::None);
    assert_eq!(run(&two), Ok(vec![1, 2, 0, 0]));
    let none = kernel(copy(0, 4, Some(0), CacheHint::Auto, None), Unroll::None);
    assert_eq!(run(&none), Ok(vec![0, 0, 0, 0]));
    // the control: with every element valid, the tile is the source
    let all = kernel(copy(0, 4, Some(4), CacheHint::Auto, None), Unroll::None);
    assert_eq!(run(&all), Ok(vec![1, 2, 4, 8]));
}

#[test]
fn a_predicated_off_copy_reads_nothing_so_its_source_may_be_out_of_range() {
    let off = kernel(copy(100, 4, Some(0), CacheHint::Auto, None), Unroll::None);
    assert_eq!(run(&off), Ok(vec![0, 0, 0, 0]));
    let on = kernel(copy(100, 4, None, CacheHint::Auto, None), Unroll::None);
    assert_eq!(run(&on), Err(KernelTrap::OutOfBounds { buffer: 0 }));
}

#[test]
fn a_valid_count_above_the_copy_size_traps() {
    let k = kernel(copy(0, 4, Some(5), CacheHint::Auto, None), Unroll::None);
    assert!(matches!(run(&k), Err(KernelTrap::Unsupported(_))));
}

#[test]
fn cache_hints_and_l2_prefetch_do_not_change_the_result() {
    for hint in [CacheHint::Auto, CacheHint::All, CacheHint::Global] {
        for l2 in [None, Some(64), Some(128), Some(256)] {
            let k = kernel(copy(0, 4, None, hint, l2), Unroll::None);
            assert_eq!(run(&k), Ok(vec![1, 2, 4, 8]), "{hint:?} {l2:?}");
        }
    }
}

#[test]
fn the_checker_refuses_what_a_copy_form_cannot_be() {
    let refused = |c: Stmt| {
        matches!(
            check_kernel(&kernel(c, Unroll::None)),
            Err(KernelError::AsyncUnsupported { .. })
        )
    };
    // `.cg` is 16 bytes only
    assert!(refused(copy(0, 2, None, CacheHint::Global, None)), ".cg 8B");
    assert!(refused(copy(0, 4, None, CacheHint::Auto, Some(100))), "l2");
    let float_valid = Stmt::CopyAsync {
        dst: tile(u(0)),
        src: buf(0, u(0)),
        elems: 4,
        valid: Some(ScalarExpr::Const(1.0)),
        cache: CacheHint::Auto,
        l2_prefetch: None,
    };
    let mut k = kernel(float_valid, Unroll::None);
    k.compute = ElementKind::F32;
    assert!(matches!(
        check_kernel(&k),
        Err(KernelError::AsyncUnsupported { .. })
    ));
    // the control: `.ca` at 8 bytes is fine
    assert_eq!(
        check_kernel(&kernel(
            copy(0, 2, None, CacheHint::All, None),
            Unroll::None
        )),
        Ok(())
    );
}
