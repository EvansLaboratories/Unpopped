// SPDX-License-Identifier: MIT OR Apache-2.0
//! `run_launch_lenient` (U19 slice 5, pm-kernels 2026-11-22): the opt-in that lets the oracle run a kernel that does two things
//! hardware shrugs off, with the strict `run_launch` unchanged. **Each allowance is pinned from both sides**: what it lets
//! through, and what it must still refuse, so a test goes red if it is made to allow too little or too much.
//!
//! * `same_value_writes`: two unordered stores of the **same bits** to one element are not a race; of different bits they still
//!   are. Compared as bits, never as a float `==` (`+0.0` and `-0.0` differ, a NaN equals itself).
//! * `zero_shared` / `shared_pattern`: a read of a never-stored shared element is a zero, or a known non-zero pattern.

use unpopped::exec::{AllocDecl, ExecId, Extent, LaunchShape, MemorySpace, Scope};
use unpopped::ir::{CastSpec, RoundMode, Saturation, ScalarExpr};
use unpopped::kernel::{BarrierKind, Block, BufferRole, KernelFn, LocalDecl, Param, Place, Stmt};
use unpopped::oracle::TypedBuffer;
use unpopped::oracle::kernel_eval::{KernelArg, KernelTrap};
use unpopped::oracle::kernel_launch::{Leniency, run_launch, run_launch_lenient};
use unpopped_vocab::ElementKind;

const F32: ElementKind = ElementKind::F32;
const U32: ElementKind = ElementKind::U32;

fn tid_f32() -> ScalarExpr {
    ScalarExpr::Cast(
        Box::new(ScalarExpr::Exec(ExecId::ThreadIdx(0))),
        CastSpec {
            to: F32,
            round: RoundMode::NearestEven,
            sat: Saturation::Wrap,
        },
    )
}

fn c(v: f64) -> ScalarExpr {
    ScalarExpr::Const(v)
}

fn u(v: i64) -> ScalarExpr {
    ScalarExpr::ConstInt(v, U32)
}

fn sub(a: ScalarExpr, b: ScalarExpr) -> ScalarExpr {
    ScalarExpr::Sub(Box::new(a), Box::new(b))
}

fn mul(a: ScalarExpr, b: ScalarExpr) -> ScalarExpr {
    ScalarExpr::Mul(Box::new(a), Box::new(b))
}

fn out_place(i: ScalarExpr) -> Place {
    Place::Logical {
        buffer: 0,
        coords: vec![i],
    }
}

fn shared0() -> Place {
    Place::Alloc {
        alloc: 0,
        index: u(0),
    }
}

/// The same cells reached by byte offset: `Place::Raw` (global) and `Place::AllocRaw` (shared), which the decode kernel's
/// pointer accesses lower to.
fn raw_out0() -> Place {
    Place::Raw {
        buffer: 0,
        byte_offset: ScalarExpr::ConstInt(0, ElementKind::I64),
        elem: F32,
    }
}

fn raw_shared0() -> Place {
    Place::AllocRaw {
        alloc: 0,
        byte_offset: ScalarExpr::ConstInt(0, ElementKind::I64),
        elem: F32,
    }
}

fn kernel(body: Vec<Stmt>, shared: bool) -> KernelFn {
    let k = KernelFn::new(
        "lenient",
        F32,
        vec![Param::buffer("out", F32, BufferRole::ReadWrite, Some(1))],
        vec![LocalDecl::scalar(F32)],
        Block(body),
    )
    .with_launch(LaunchShape::new([2, 1, 1], [1, 1, 1]));
    if shared {
        k.with_allocs(vec![AllocDecl::new(
            "s",
            MemorySpace::Shared,
            F32,
            Extent::Static(1),
        )])
    } else {
        k
    }
}

/// Both threads store `value` to ONE shared cell; after a barrier each reads it into `out[tid]`.
fn shared_race(value: ScalarExpr) -> KernelFn {
    shared_race_at(shared0(), value)
}

fn shared_race_at(cell: Place, value: ScalarExpr) -> KernelFn {
    kernel(
        vec![
            Stmt::Store {
                place: cell.clone(),
                value,
            },
            Stmt::Barrier {
                scope: Scope::Block,
                kind: BarrierKind::Full,
            },
            Stmt::Load {
                local: 0,
                place: cell,
            },
            Stmt::Store {
                place: out_place(ScalarExpr::Exec(ExecId::ThreadIdx(0))),
                value: ScalarExpr::Local(0),
            },
        ],
        true,
    )
}

/// Both threads store `value` to `out[0]`, a GLOBAL cell.
fn global_race(value: ScalarExpr) -> KernelFn {
    global_race_at(out_place(u(0)), value)
}

fn global_race_at(cell: Place, value: ScalarExpr) -> KernelFn {
    kernel(vec![Stmt::Store { place: cell, value }], false)
}

/// The race on one cell four ways: shared and global, typed and raw.
fn races(value: impl Fn() -> ScalarExpr) -> Vec<(&'static str, KernelFn)> {
    vec![
        ("shared", shared_race(value())),
        ("global", global_race(value())),
        ("shared raw", shared_race_at(raw_shared0(), value())),
        ("global raw", global_race_at(raw_out0(), value())),
    ]
}

fn out() -> Vec<KernelArg> {
    vec![KernelArg::Buffer(TypedBuffer::from_f32(&[2], &[0.0, 0.0]))]
}

fn bits(args: &[KernelArg]) -> Vec<u32> {
    let KernelArg::Buffer(b) = &args[0] else {
        panic!()
    };
    b.bytes
        .chunks(4)
        .map(|c| u32::from_le_bytes(c.try_into().unwrap()))
        .collect()
}

const SAME: Leniency = Leniency::strict().with_same_value_writes();

/// The values: both threads agree / differ / differ only in the sign of zero / are the same NaN.
fn agree() -> ScalarExpr {
    c(1.5)
}
fn differ() -> ScalarExpr {
    ScalarExpr::Add(Box::new(c(1.0)), Box::new(tid_f32()))
}
fn signed_zero() -> ScalarExpr {
    // 0.0 * (1 - 2*tid): +0.0 for thread 0, -0.0 for thread 1
    mul(c(0.0), sub(c(1.0), mul(c(2.0), tid_f32())))
}
fn same_nan() -> ScalarExpr {
    ScalarExpr::Div(Box::new(c(0.0)), Box::new(c(0.0)))
}

#[test]
fn strict_run_launch_refuses_even_two_equal_stores() {
    for (what, k) in races(agree) {
        assert_eq!(
            run_launch(&k, &mut out()),
            Err(KernelTrap::DataRace),
            "{what}"
        );
    }
}

#[test]
fn equal_stores_pass_when_allowed_in_shared_and_in_global_typed_and_raw() {
    for (what, k) in races(agree) {
        let mut a = out();
        run_launch_lenient(&k, &mut a, SAME)
            .unwrap_or_else(|e| panic!("{what}: equal bits are not a race: {e:?}"));
        assert_eq!(bits(&a)[0], 1.5f32.to_bits(), "{what}");
    }
}

#[test]
fn different_stores_still_race_under_the_allowance_in_every_form() {
    for (what, k) in races(differ) {
        assert_eq!(
            run_launch_lenient(&k, &mut out(), Leniency::all()),
            Err(KernelTrap::DataRace),
            "{what}"
        );
    }
}

#[test]
fn equality_is_of_bits_so_a_signed_zero_still_races_and_a_nan_does_not() {
    for (what, k) in races(signed_zero) {
        assert_eq!(
            run_launch_lenient(&k, &mut out(), SAME),
            Err(KernelTrap::DataRace),
            "{what}: +0.0 and -0.0 are equal as floats and different as bits"
        );
    }
    for (what, k) in races(same_nan) {
        let mut a = out();
        run_launch_lenient(&k, &mut a, SAME).unwrap_or_else(|e| panic!("{what}: {e:?}"));
        assert!(f32::from_bits(bits(&a)[0]).is_nan(), "{what}");
    }
}

#[test]
fn the_allowance_off_means_strict_even_through_run_launch_lenient() {
    let off = Leniency::strict().with_zero_shared();
    for (what, k) in races(agree) {
        assert_eq!(
            run_launch_lenient(&k, &mut out(), off),
            Err(KernelTrap::DataRace),
            "{what}"
        );
    }
}

/// Each thread reads the never-stored shared cell into `out[tid]`.
fn read_unwritten() -> KernelFn {
    kernel(
        vec![
            Stmt::Load {
                local: 0,
                place: shared0(),
            },
            Stmt::Store {
                place: out_place(ScalarExpr::Exec(ExecId::ThreadIdx(0))),
                value: ScalarExpr::Local(0),
            },
        ],
        true,
    )
}

#[test]
fn an_unwritten_shared_read_is_refused_zero_or_a_known_pattern() {
    let k = read_unwritten();
    assert!(matches!(
        run_launch(&k, &mut out()),
        Err(KernelTrap::UninitializedShared { .. })
    ));
    let mut z = out();
    run_launch_lenient(&k, &mut z, Leniency::strict().with_zero_shared()).unwrap();
    assert_eq!(bits(&z), vec![0, 0]);
    // the pattern: byte i is 1 + (37 i + seed) % 58, little-endian f32 from bytes 0..4
    let mut p = out();
    let seed = 9u8;
    run_launch_lenient(&k, &mut p, Leniency::strict().with_shared_pattern(seed)).unwrap();
    let byte = |i: u64| (1 + ((37 * i + u64::from(seed)) % 58)) as u8;
    let want = u32::from_le_bytes([byte(0), byte(1), byte(2), byte(3)]);
    assert_eq!(bits(&p), vec![want, want]);
    assert!(f32::from_bits(want).is_finite() && want != 0);
    // a different seed is a different pattern
    let mut q = out();
    run_launch_lenient(&k, &mut q, Leniency::strict().with_shared_pattern(seed + 1)).unwrap();
    assert_ne!(bits(&q), bits(&p));
}
