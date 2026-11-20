// SPDX-License-Identifier: MIT OR Apache-2.0
//! **Packed values inside a kernel** (ROADMAP U15, the kernel-interpreter slice). `Extract` of a `Pack`, of a lane-wise
//! operator over `Pack`s, and of a `BitcastPacked` evaluates in a kernel body as it does in a plan expression
//! (`oracle::expr_eval::packed_lane`). Each lane has a distinct value so a wrong lane shows.

use unpopped::exec::LaunchShape;
use unpopped::ir::{CastSpec, RoundMode, Saturation, ScalarExpr};
use unpopped::kernel::{Block, BufferRole, KernelFn, LocalDecl, Param, Place, Stmt, check_kernel};
use unpopped::oracle::TypedBuffer;
use unpopped::oracle::kernel_eval::KernelArg;
use unpopped::oracle::kernel_launch::run_launch;
use unpopped_vocab::ElementKind;

const F32: ElementKind = ElementKind::F32;
const U8: ElementKind = ElementKind::U8;
const U32: ElementKind = ElementKind::U32;

fn c(v: f64) -> ScalarExpr {
    ScalarExpr::Const(v)
}

fn extract(x: ScalarExpr, i: u8) -> ScalarExpr {
    ScalarExpr::Extract(Box::new(x), i)
}

fn to_f32(x: ScalarExpr) -> ScalarExpr {
    ScalarExpr::Cast(
        Box::new(x),
        CastSpec {
            to: F32,
            round: RoundMode::NearestEven,
            sat: Saturation::Wrap,
        },
    )
}

/// `out[0] = <value>` where `value` is an F32 expression over packed forms; the kernel's one thread runs it.
fn run_value(value: ScalarExpr) -> f32 {
    let k = KernelFn::new(
        "packed",
        F32,
        vec![Param::buffer("out", F32, BufferRole::Write, Some(1))],
        vec![LocalDecl::scalar(F32)],
        Block(vec![
            Stmt::Let {
                local: 0,
                init: value,
            },
            Stmt::Store {
                place: Place::Logical {
                    buffer: 0,
                    coords: vec![ScalarExpr::ConstInt(0, U32)],
                },
                value: ScalarExpr::Local(0),
            },
        ]),
    )
    .with_launch(LaunchShape::new([1, 1, 1], [1, 1, 1]));
    check_kernel(&k).expect("well formed");
    let mut args = vec![KernelArg::Buffer(TypedBuffer::from_f32(&[1], &[-1.0]))];
    run_launch(&k, &mut args).expect("runs");
    let KernelArg::Buffer(b) = &args[0] else {
        unreachable!("a buffer")
    };
    f32::from_le_bytes(b.bytes[..4].try_into().unwrap())
}

#[test]
fn a_lane_of_a_pack_is_that_lane() {
    let p = || ScalarExpr::Pack(vec![c(1.5), c(2.5), c(3.5), c(4.5)]);
    assert_eq!(run_value(extract(p(), 0)), 1.5);
    assert_eq!(run_value(extract(p(), 2)), 3.5);
}

#[test]
fn a_lane_wise_operator_applies_to_the_same_lane() {
    let sum = ScalarExpr::Add(
        Box::new(ScalarExpr::Pack(vec![c(1.0), c(2.0)])),
        Box::new(ScalarExpr::Pack(vec![c(10.0), c(20.0)])),
    );
    assert_eq!(run_value(extract(sum.clone(), 0)), 11.0);
    assert_eq!(run_value(extract(sum, 1)), 22.0);
}

#[test]
fn a_lane_of_a_bitcast_is_sliced_from_its_bits_lane_zero_least_significant() {
    // 0x04030201 as four u8 lanes: lane 0 = 1, lane 2 = 3
    let word = || ScalarExpr::ConstInt(0x0403_0201, U32);
    let lane = |i| {
        to_f32(extract(
            ScalarExpr::BitcastPacked(Box::new(word()), U8, 4),
            i,
        ))
    };
    assert_eq!(run_value(lane(0)), 1.0);
    assert_eq!(run_value(lane(2)), 3.0);
    assert_eq!(run_value(lane(3)), 4.0);
}

/// `let p: packed = <packed>; out[0] = <value over Local(0)>` (local 0 is the packed one, local 1 the F32 result).
fn run_packed_local(packed: ScalarExpr, lanes: u8, value: ScalarExpr) -> Result<f32, String> {
    let k = KernelFn::new(
        "packed_local",
        F32,
        vec![Param::buffer("out", F32, BufferRole::Write, Some(1))],
        vec![LocalDecl::packed(F32, lanes), LocalDecl::scalar(F32)],
        Block(vec![
            Stmt::Let {
                local: 0,
                init: packed,
            },
            Stmt::Let {
                local: 1,
                init: value,
            },
            Stmt::Store {
                place: Place::Logical {
                    buffer: 0,
                    coords: vec![ScalarExpr::ConstInt(0, U32)],
                },
                value: ScalarExpr::Local(1),
            },
        ]),
    )
    .with_launch(LaunchShape::new([1, 1, 1], [1, 1, 1]));
    check_kernel(&k).map_err(|e| format!("check: {e:?}"))?;
    let mut args = vec![KernelArg::Buffer(TypedBuffer::from_f32(&[1], &[-1.0]))];
    run_launch(&k, &mut args).map_err(|e| format!("run: {e:?}"))?;
    let KernelArg::Buffer(b) = &args[0] else {
        unreachable!("a buffer")
    };
    Ok(f32::from_le_bytes(b.bytes[..4].try_into().unwrap()))
}

/// The gap, pinned: a packed LOCAL (`Let` of a `Pack`, read back by `Extract(Local(_), i)`) passes `check_kernel` but the
/// interpreter traps `Unsupported("packed value")`. Closing it needs a pass that scalarizes packed locals into one scalar
/// local per lane (`Let`/`Assign` per lane, `Extract(Local(p), i)` to `Local(p_i)`); there is no `Stmt` mapper to build it
/// on (38 variants). When it lands this test flips to `Ok(2.5)`.
#[test]
fn a_packed_local_is_a_pinned_gap() {
    let p = ScalarExpr::Pack(vec![c(1.5), c(2.5), c(3.5), c(4.5)]);
    let lane = |i| extract(ScalarExpr::Local(0), i);
    let r = run_packed_local(p, 4, lane(1));
    assert_eq!(r, Err("run: Unsupported(\"packed value\")".to_string()));
}
