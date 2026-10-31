// SPDX-License-Identifier: MIT OR Apache-2.0
//! **E1d-6: a launched Scatter (atomic combines) equals the serial kernel**
//! (`docs/e1d-parallel-mapping.md`, "E1d-6 as built").
//!
//! Integer `AtomicAdd`, `AtomicMax` and `AtomicMin` are order-independent, so the destination must
//! be bit-exact on every geometry even though sources name one target many times and some targets
//! fall outside the destination. A launched `Assign` and a float add decline; a plain load and
//! store in place of the atomic is a `DataRace` (the reason the statement exists).

use unpopped::ir::{OobPolicy, OpDef, WriteCombine, WriteIndex};
use unpopped::kernel::{KernelFn, Stmt, check_kernel};
use unpopped::kernel_lift::{LiftDecline, lift_scatter_opdef, lift_scatter_opdef_launched};
use unpopped::oracle::TypedBuffer;
use unpopped::oracle::kernel_eval::{KernelArg, KernelTrap, run_kernel};
use unpopped::oracle::kernel_launch::run_launch;
use unpopped_vocab::ElementKind;

const I32: ElementKind = ElementKind::I32;
const GEOMETRIES: [(u32, u32); 5] = [(1, 1), (2, 3), (3, 8), (4, 32), (16, 64)];
const SRC: [i64; 2] = [9, 4];
const DST: [i64; 2] = [5, 4];

fn i32s(shape: &[i64], v: impl Fn(usize) -> i32) -> TypedBuffer {
    let n: i64 = shape.iter().product();
    let bytes = (0..n as usize).flat_map(|i| v(i).to_le_bytes()).collect();
    TypedBuffer::new(I32, shape.to_vec(), vec![shape[1], 1], bytes)
}

fn op(combine: WriteCombine, dtype: ElementKind) -> OpDef {
    let base = OpDef::scatter("s", &[dtype], 0, I32);
    match combine {
        WriteCombine::AtomicAdd => OpDef::scatter_add("s", &[dtype], 0, I32),
        WriteCombine::Assign => base,
        c => base.with_scatter(WriteIndex::ScatterIndexed {
            index_operand: 1,
            axis: 0,
            combine: c,
            oob: OobPolicy::Skip,
            index_dtype: I32,
        }),
    }
}

/// Run `k` on updates, a full-shape index with duplicates and out-of-range targets, and a
/// pre-filled destination.
fn run(k: &KernelFn, launch: bool) -> Result<Vec<u8>, KernelTrap> {
    let updates = i32s(&SRC, |i| (i as i32 * 7 % 11) - 5);
    let targets = [0, 4, 4, 1, 7, -2, 0, 2, 4, 3, 0, 1, 9, 2, 2, 4, 0, 1];
    let index = i32s(&SRC, |i| targets[(i * 5 + i / 4) % targets.len()]);
    let dst = i32s(&DST, |i| i as i32 - 3);
    let mut args = vec![
        KernelArg::Buffer(updates),
        KernelArg::Buffer(index),
        KernelArg::Buffer(dst),
        KernelArg::Int(i128::from(SRC[0])),
        KernelArg::Int(i128::from(SRC[1])),
        KernelArg::Int(i128::from(DST[0])),
    ];
    if launch {
        run_launch(k, &mut args)?;
    } else {
        run_kernel(k, &mut args)?;
    }
    let KernelArg::Buffer(b) = &args[2] else {
        unreachable!()
    };
    Ok(b.bytes.clone())
}

#[test]
fn every_integer_combine_is_bit_exact_on_every_geometry() {
    for combine in [
        WriteCombine::AtomicAdd,
        WriteCombine::AtomicMax,
        WriteCombine::AtomicMin,
    ] {
        let o = op(combine, I32);
        let want = run(&lift_scatter_opdef(&o, I32, 2).unwrap(), false).unwrap();
        for (grid, block) in GEOMETRIES {
            let k = lift_scatter_opdef_launched(&o, I32, 2, block, grid).unwrap();
            assert_eq!(check_kernel(&k), Ok(()), "well formed");
            assert_eq!(
                run(&k, true),
                Ok(want.clone()),
                "{combine:?} {grid}x{block}"
            );
        }
    }
}

#[test]
fn assign_and_a_float_add_decline() {
    let assign = lift_scatter_opdef_launched(&op(WriteCombine::Assign, I32), I32, 2, 2, 2);
    assert!(matches!(assign, Err(LiftDecline::UnsupportedForm(_))));
    let f32_add = op(WriteCombine::AtomicAdd, ElementKind::F32);
    let float = lift_scatter_opdef_launched(&f32_add, ElementKind::F32, 2, 2, 2);
    assert!(matches!(float, Err(LiftDecline::UnsupportedForm(_))));
    let zero = lift_scatter_opdef_launched(&op(WriteCombine::AtomicAdd, I32), I32, 2, 0, 1);
    assert_eq!(zero.err(), Some(LiftDecline::BadLaunch));
}

/// Swap each atomic for the serial lift's load and store (what the launched kernel would be
/// without the statement): two threads reaching one target must race.
#[test]
fn without_the_atomic_statement_duplicate_targets_race() {
    let o = op(WriteCombine::AtomicAdd, I32);
    let plain = serial_body_as_grid_stride(&o);
    assert_eq!(run(&plain, true), Err(KernelTrap::DataRace));
}

fn serial_body_as_grid_stride(o: &OpDef) -> KernelFn {
    let serial = lift_scatter_opdef(o, I32, 2).unwrap();
    let launched = lift_scatter_opdef_launched(o, I32, 2, 4, 2).unwrap();
    let Some(Stmt::For { start, step, .. }) = launched.body.0.first().cloned() else {
        panic!("a loop")
    };
    let mut k = serial;
    let Some(Stmt::For {
        start: s0,
        step: st0,
        ..
    }) = k.body.0.first_mut()
    else {
        panic!("a loop")
    };
    *s0 = start;
    *st0 = step;
    k.launch = launched.launch;
    k
}
