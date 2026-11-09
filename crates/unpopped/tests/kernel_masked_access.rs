// SPDX-License-Identifier: MIT OR Apache-2.0
//! **Masked loads and stores** (ROADMAP R5.7 C4): `Stmt::MaskedLoad { local, place, mask, fill }` declares `local` as the
//! element at `place` where `mask != 0` and as `fill` (any value of the element type, not only zero) where it is zero;
//! `Stmt::MaskedStore { place, value, mask }` writes only where the mask holds. Where a mask is false **the memory is not
//! touched**: no bounds check and no race record, so an index outside the buffer under a false mask is legal, and a false
//! lane cannot race with a true one. Each claim has a control that shows the check can fire.

use unpopped::exec::{AllocDecl, ExecId, Extent, LaunchShape, MemorySpace};
use unpopped::ir::{BinaryOp, ScalarExpr};
use unpopped::kernel::{
    Block, BufferRole, KernelError, KernelFn, LocalDecl, Param, Place, Stmt, check_kernel,
};
use unpopped::oracle::TypedBuffer;
use unpopped::oracle::kernel_eval::{KernelArg, KernelTrap, run_kernel};
use unpopped::oracle::kernel_launch::run_launch;
use unpopped_vocab::ElementKind;

const F32: ElementKind = ElementKind::F32;
const I32: ElementKind = ElementKind::I32;
const U32: ElementKind = ElementKind::U32;
const IN_LEN: usize = 5;
const OUT_LEN: usize = 8;

fn ci(v: i64) -> ScalarExpr {
    ScalarExpr::ConstInt(v, I32)
}

fn at(buffer: u32, i: ScalarExpr) -> Place {
    Place::Logical {
        buffer,
        coords: vec![i],
    }
}

fn lt(a: ScalarExpr, b: ScalarExpr) -> ScalarExpr {
    ScalarExpr::Binary(BinaryOp::CmpLt, Box::new(a), Box::new(b))
}

fn floats(v: &[f32]) -> KernelArg {
    let bytes = v.iter().flat_map(|x| x.to_le_bytes()).collect();
    KernelArg::Buffer(TypedBuffer::new(F32, vec![v.len() as i64], vec![1], bytes))
}

fn out_floats(a: &KernelArg) -> Vec<f32> {
    let KernelArg::Buffer(b) = a else {
        unreachable!()
    };
    b.bytes
        .chunks(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect()
}

/// `for i in 0..8 { out[i] = (i < 5 ? in[i] : fill) }` with the load masked (locals: 0 the counter, 1 the value).
fn masked_load_loop(mask: ScalarExpr, fill: ScalarExpr) -> KernelFn {
    let body = vec![Stmt::For {
        local: 0,
        start: ci(0),
        end: ci(OUT_LEN as i64),
        step: ci(1),
        body: Block(vec![
            Stmt::MaskedLoad {
                local: 1,
                place: at(0, ScalarExpr::Local(0)),
                mask,
                fill,
            },
            Stmt::Store {
                place: at(1, ScalarExpr::Local(0)),
                value: ScalarExpr::Local(1),
            },
        ]),
    }];
    KernelFn::new(
        "mload",
        F32,
        vec![
            Param::buffer("in", F32, BufferRole::Read, Some(1)),
            Param::buffer("out", F32, BufferRole::Write, Some(1)),
        ],
        vec![LocalDecl::scalar(I32), LocalDecl::scalar(F32)],
        Block(body),
    )
}

fn run_load(k: &KernelFn) -> Result<Vec<f32>, KernelTrap> {
    let mut args = vec![
        floats(&[10.0, 11.0, 12.0, 13.0, 14.0]),
        floats(&[0.0; OUT_LEN]),
    ];
    run_kernel(k, &mut args)?;
    Ok(out_floats(&args[1]))
}

fn in_range() -> ScalarExpr {
    lt(ScalarExpr::Local(0), ci(IN_LEN as i64))
}

#[test]
fn a_false_mask_gives_the_fill_and_never_reads_the_memory() {
    let k = masked_load_loop(in_range(), ScalarExpr::Const(-7.5));
    assert_eq!(check_kernel(&k), Ok(()));
    // lanes 5..8 index past the end of `in`; under the false mask that is legal
    assert_eq!(
        run_load(&k),
        Ok(vec![10.0, 11.0, 12.0, 13.0, 14.0, -7.5, -7.5, -7.5])
    );
    // the control: the same loop with a mask that is always true reads out of bounds
    let all = masked_load_loop(ci(1), ScalarExpr::Const(-7.5));
    assert!(matches!(
        run_load(&all),
        Err(KernelTrap::OutOfBounds { buffer: 0 })
    ));
}

#[test]
fn the_fill_is_any_value_the_element_type_holds() {
    for fill in [0.0, 1.0, -0.0, f64::INFINITY, 3.25] {
        let k = masked_load_loop(in_range(), ScalarExpr::Const(fill));
        let out = run_load(&k).expect("runs");
        for v in &out[IN_LEN..] {
            assert_eq!(v.to_bits(), (fill as f32).to_bits(), "fill {fill}");
        }
    }
    let nan = run_load(&masked_load_loop(in_range(), ScalarExpr::Const(f64::NAN))).expect("runs");
    assert!(nan[IN_LEN..].iter().all(|v| v.is_nan()));
    // a fill that is an expression, not a literal: the loop counter
    let k = masked_load_loop(
        in_range(),
        ScalarExpr::Cast(
            Box::new(ScalarExpr::Local(0)),
            unpopped::ir::CastSpec {
                to: F32,
                round: unpopped::ir::RoundMode::NearestEven,
                sat: unpopped::ir::Saturation::Wrap,
            },
        ),
    );
    assert_eq!(
        run_load(&k),
        Ok(vec![10.0, 11.0, 12.0, 13.0, 14.0, 5.0, 6.0, 7.0])
    );
}

#[test]
fn any_scalar_type_is_a_mask_and_nonzero_means_true() {
    // an F32 mask: 1.0 for i < 5, else 0.0 (a comparison yields the compute type's 0/1)
    let k = masked_load_loop(in_range(), ScalarExpr::Const(-1.0));
    assert_eq!(run_load(&k).expect("runs")[IN_LEN], -1.0);
    // an integer mask `5 - i`: non-zero for i != 5, so lane 5 is the false one, and lanes 6 and 7 are true (and out of
    // bounds): the trap shows the mask is tested `!= 0`, not `> 0`
    let diff = ScalarExpr::Sub(Box::new(ci(5)), Box::new(ScalarExpr::Local(0)));
    let k = masked_load_loop(diff, ScalarExpr::Const(-1.0));
    assert!(matches!(
        run_load(&k),
        Err(KernelTrap::OutOfBounds { buffer: 0 })
    ));
}

#[test]
fn a_masked_store_writes_only_where_the_mask_holds() {
    // `for i in 0..8 { if-masked out[i] = 9.0 }` with the mask `i < 5`
    let body = vec![Stmt::For {
        local: 0,
        start: ci(0),
        end: ci(OUT_LEN as i64),
        step: ci(1),
        body: Block(vec![Stmt::MaskedStore {
            place: at(0, ScalarExpr::Local(0)),
            value: ScalarExpr::Const(9.0),
            mask: in_range(),
        }]),
    }];
    let k = KernelFn::new(
        "mstore",
        F32,
        vec![Param::buffer("out", F32, BufferRole::Write, Some(1))],
        vec![LocalDecl::scalar(I32)],
        Block(body),
    );
    assert_eq!(check_kernel(&k), Ok(()));
    // the buffer has 5 elements: lanes 5..8 are out of bounds, legal under the false mask
    let mut args = vec![floats(&[0.0; IN_LEN])];
    run_kernel(&k, &mut args).expect("runs");
    assert_eq!(out_floats(&args[0]), vec![9.0; IN_LEN]);
    // the control: a true mask over the same range traps
    let mut all = k.clone();
    let Stmt::For { body, .. } = &mut all.body.0[0] else {
        unreachable!()
    };
    let Stmt::MaskedStore { mask, .. } = &mut body.0[0] else {
        unreachable!()
    };
    *mask = ci(1);
    let mut args = vec![floats(&[0.0; IN_LEN])];
    assert!(matches!(
        run_kernel(&all, &mut args),
        Err(KernelTrap::OutOfBounds { buffer: 0 })
    ));
}

/// Two threads both store `out[0]`; with `masked` only thread 0 does (`ThreadIdx.x == 0`).
fn two_writers(masked: bool) -> KernelFn {
    let place = Place::Logical {
        buffer: 0,
        coords: vec![ScalarExpr::ConstInt(0, U32)],
    };
    let value = ScalarExpr::Const(5.0);
    let store = if masked {
        let first = ScalarExpr::Binary(
            BinaryOp::CmpEq,
            Box::new(ScalarExpr::Exec(ExecId::ThreadIdx(0))),
            Box::new(ScalarExpr::ConstInt(0, U32)),
        );
        Stmt::MaskedStore {
            place,
            value,
            mask: first,
        }
    } else {
        Stmt::Store { place, value }
    };
    KernelFn::new(
        "writers",
        F32,
        vec![Param::buffer("out", F32, BufferRole::Write, Some(1))],
        vec![],
        Block(vec![store]),
    )
    .with_launch(LaunchShape::new([2, 1, 1], [1, 1, 1]))
}

#[test]
fn a_false_lane_leaves_no_race_record() {
    let run = |k: &KernelFn| {
        let mut args = vec![floats(&[0.0])];
        run_launch(k, &mut args).map(|()| out_floats(&args[0]))
    };
    // the control: two unmasked stores to one element race
    assert_eq!(run(&two_writers(false)), Err(KernelTrap::DataRace));
    // masked, only thread 0 writes, so there is nothing to race with
    assert_eq!(run(&two_writers(true)), Ok(vec![5.0]));
}

#[test]
fn the_checker_types_the_fill_the_mask_and_the_buffer_role() {
    // fill of another type than the element
    let bad_fill = masked_load_loop(in_range(), ci(0));
    assert!(matches!(
        check_kernel(&bad_fill),
        Err(KernelError::TypeMismatch {
            what: "MaskedLoad fill"
        })
    ));
    // a load from a write-only buffer, masked or not
    let mut wo = masked_load_loop(in_range(), ScalarExpr::Const(0.0));
    wo.params[0] = Param::buffer("in", F32, BufferRole::Write, Some(1));
    assert!(matches!(
        check_kernel(&wo),
        Err(KernelError::ReadFromWriteOnly { buffer: 0 })
    ));
    // a store to a read-only buffer
    let mut ro = KernelFn::new(
        "ro",
        F32,
        vec![Param::buffer("out", F32, BufferRole::Read, Some(1))],
        vec![],
        Block(vec![Stmt::MaskedStore {
            place: at(0, ci(0)),
            value: ScalarExpr::Const(1.0),
            mask: ci(1),
        }]),
    );
    assert!(matches!(
        check_kernel(&ro),
        Err(KernelError::WriteToReadOnly { buffer: 0 })
    ));
    // control: with the role right it checks
    ro.params[0] = Param::buffer("out", F32, BufferRole::Write, Some(1));
    assert_eq!(check_kernel(&ro), Ok(()));
}

/// A launched kernel reading a `Constant` array (the one load that is uniform for a uniform index) through a masked load:
/// `v = masked_load(c[0], mask, fill); if v { barrier }`. Local 0 is a uniform scalar parameter `n`.
fn barrier_under_a_masked_value(mask: ScalarExpr, fill: ScalarExpr) -> KernelFn {
    let mut k = KernelFn::new(
        "divergent",
        I32,
        vec![Param::scalar("n", I32, 0)],
        vec![LocalDecl::scalar(I32), LocalDecl::scalar(I32)],
        Block(vec![
            Stmt::MaskedLoad {
                local: 1,
                place: Place::Alloc {
                    alloc: 0,
                    index: ci(0),
                },
                mask,
                fill,
            },
            Stmt::If {
                cond: ScalarExpr::Local(1),
                then: Block(vec![Stmt::Barrier {
                    scope: unpopped::exec::Scope::Block,
                    kind: unpopped::kernel::BarrierKind::Full,
                }]),
                els: Block(vec![]),
            },
        ]),
    )
    .with_launch(LaunchShape::new([4, 1, 1], [1, 1, 1]));
    k.allocs = vec![AllocDecl::new(
        "c",
        MemorySpace::Constant,
        I32,
        Extent::Static(4),
    )];
    k
}

#[test]
fn a_masked_value_is_divergent_when_its_mask_or_its_fill_is() {
    let uniform = ScalarExpr::Local(0);
    let by_thread = ScalarExpr::Cast(
        Box::new(ScalarExpr::Exec(ExecId::ThreadIdx(0))),
        unpopped::ir::CastSpec {
            to: I32,
            round: unpopped::ir::RoundMode::NearestEven,
            sat: unpopped::ir::Saturation::Wrap,
        },
    );
    // the control: a uniform mask and fill make the value uniform, and the barrier under it is accepted
    assert_eq!(
        check_kernel(&barrier_under_a_masked_value(uniform.clone(), ci(0))),
        Ok(())
    );
    for (what, k) in [
        (
            "mask",
            barrier_under_a_masked_value(by_thread.clone(), ci(0)),
        ),
        ("fill", barrier_under_a_masked_value(uniform, by_thread)),
    ] {
        assert_eq!(
            check_kernel(&k),
            Err(KernelError::BarrierDivergent),
            "a thread-dependent {what}"
        );
    }
}

#[test]
fn a_mask_is_checked_like_any_expression() {
    // a mask that reads a local nobody declared, on a load and on a store
    let load = masked_load_loop(ScalarExpr::Local(9), ScalarExpr::Const(0.0));
    assert!(matches!(
        check_kernel(&load),
        Err(KernelError::UnknownLocal { .. })
    ));
    let store = KernelFn::new(
        "s",
        F32,
        vec![Param::buffer("out", F32, BufferRole::Write, Some(1))],
        vec![],
        Block(vec![Stmt::MaskedStore {
            place: at(0, ci(0)),
            value: ScalarExpr::Const(1.0),
            mask: ScalarExpr::Local(9),
        }]),
    );
    assert!(matches!(
        check_kernel(&store),
        Err(KernelError::UnknownLocal { .. })
    ));
}
