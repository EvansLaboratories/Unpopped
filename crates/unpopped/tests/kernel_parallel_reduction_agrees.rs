// SPDX-License-Identifier: MIT OR Apache-2.0
//! **E1d-1: a block-tree reduction computes what the serial reduction computes**
//! (`docs/e1d-parallel-mapping.md`).
//!
//! `lift_reduction_opdef_launched` has each thread fold a strided slice of the reduced axis, then
//! combines the partials in a log-step tree with a `Barrier` between rounds. It is held to the
//! serial kernel's bytes (itself proven equal to `oracle::evaluate` in `kernel_reduction_agrees`)
//! for the reductions whose result does not depend on order: integer `Sum`/`Prod`/`Max`/`Min` and
//! float `Max`/`Min`. The corpora carry NaN, infinities and wrapping magnitudes, and the
//! geometries include blocks wider than the row (threads with nothing to fold, whose `have` flag
//! the combine must respect) and fewer blocks than rows (the row loop must stride).
//!
//! The corpus has no `-0.0`: a `Max`/`Min` over both zeros is order-dependent in the sign of the
//! zero (documented in `kernel_lift/block_reduce.rs`), so it is not claimed here.
//!
//! Six broken copies of the launched kernel (the tests after the geometry sweep) must **disagree** (or trap), each by breaking one
//! thing; a `Barrier` removed must be reported by the race checker, not by luck.

use unpopped::ir::{Expr, OpDef, ReduceOp, ScalarExpr, input};
use unpopped::kernel::{Block, KernelFn, Stmt, check_kernel};
use unpopped::kernel_lift::{LiftDecline, lift_reduction_opdef, lift_reduction_opdef_launched};
use unpopped::oracle::TypedBuffer;
use unpopped::oracle::kernel_eval::{KernelArg, KernelTrap, run_kernel};
use unpopped::oracle::kernel_launch::run_launch;
use unpopped_vocab::{AxisMask, ElementKind};

const ROWS: i64 = 9;
const COLS: i64 = 37;
const F32: ElementKind = ElementKind::F32;
const I32: ElementKind = ElementKind::I32;

fn f32_corpus() -> Vec<f32> {
    let base = [
        0.0,
        1.0,
        -1.0,
        0.5,
        2.0,
        -2.5,
        1.0e-40,
        3.0e38,
        -3.0e38,
        f32::INFINITY,
        f32::NEG_INFINITY,
        0.1,
        -7.25,
    ];
    (0..(ROWS * COLS) as usize)
        .map(|i| {
            // A NaN in a few rows, at a different column in each, so that the threads holding it
            // differ from row to row.
            let (r, c) = (i / COLS as usize, i % COLS as usize);
            if r % 4 == 1 && c == (r * 5) % COLS as usize {
                f32::NAN
            } else {
                base[(i * 7 + r) % base.len()]
            }
        })
        .collect()
}

fn i32_corpus() -> Vec<i32> {
    let base = [
        0,
        1,
        -1,
        7,
        -300,
        i32::MAX,
        i32::MIN,
        65_537,
        -2,
        3,
        1_000_003,
    ];
    (0..(ROWS * COLS) as usize)
        .map(|i| base[(i * 5 + i / COLS as usize) % base.len()])
        .collect()
}

fn input_buffer(dtype: ElementKind) -> TypedBuffer {
    let bytes: Vec<u8> = if dtype == F32 {
        f32_corpus().iter().flat_map(|v| v.to_le_bytes()).collect()
    } else {
        i32_corpus().iter().flat_map(|v| v.to_le_bytes()).collect()
    };
    TypedBuffer::new(dtype, vec![ROWS, COLS], vec![COLS, 1], bytes)
}

/// The op, its output shape and strides, for a reduction of the last axis or axis 0.
fn case(
    rop: ReduceOp,
    dtype: ElementKind,
    axis0: bool,
    keepdim: bool,
) -> (OpDef, Vec<i64>, Vec<i64>) {
    let mut axes = AxisMask::EMPTY;
    if axis0 {
        axes.set(0);
    }
    let op = OpDef::reduction_axes("r", 1, &[dtype], Expr(input(0).0), rop, axes, keepdim);
    let (shape, strides) = match (axis0, keepdim) {
        (false, false) => (vec![ROWS], vec![1]),
        (false, true) => (vec![ROWS, 1], vec![1, 1]),
        (true, false) => (vec![COLS], vec![1]),
        (true, true) => (vec![1, COLS], vec![COLS, 1]),
    };
    (op, shape, strides)
}

fn run(
    k: &KernelFn,
    dtype: ElementKind,
    shape: &[i64],
    strides: &[i64],
    launch: bool,
) -> Result<Vec<u8>, KernelTrap> {
    let len = shape.iter().product::<i64>() as usize * 4;
    let mut a = vec![
        KernelArg::Buffer(input_buffer(dtype)),
        KernelArg::Buffer(TypedBuffer::new(
            dtype,
            shape.to_vec(),
            strides.to_vec(),
            vec![0u8; len],
        )),
        KernelArg::Int(i128::from(ROWS)),
        KernelArg::Int(i128::from(COLS)),
    ];
    if launch {
        run_launch(k, &mut a)?;
    } else {
        run_kernel(k, &mut a)?;
    }
    let KernelArg::Buffer(out) = &a[1] else {
        unreachable!("the output follows the input")
    };
    Ok(out.bytes[..len].to_vec())
}

/// The reductions held to the serial bytes.
fn held() -> Vec<(ReduceOp, ElementKind)> {
    vec![
        (ReduceOp::Sum, I32),
        (ReduceOp::Prod, I32),
        (ReduceOp::Max, I32),
        (ReduceOp::Min, I32),
        (ReduceOp::Max, F32),
        (ReduceOp::Min, F32),
    ]
}

/// grid x block: one thread; two; fewer blocks than rows; a block wider than the row (64 > 37);
/// more blocks than rows.
const GEOMETRIES: [(u32, u32); 7] = [(1, 1), (1, 2), (2, 4), (3, 8), (4, 32), (1, 64), (12, 16)];

#[test]
fn every_geometry_gives_the_serial_bytes() {
    for (rop, dtype) in held() {
        for (axis0, keepdim) in [(false, false), (false, true), (true, false)] {
            let (op, shape, strides) = case(rop, dtype, axis0, keepdim);
            let serial = lift_reduction_opdef(&op, dtype, 2).expect("the reduction lifts");
            let want =
                run(&serial, dtype, &shape, &strides, false).expect("the serial kernel runs");
            for (grid, block) in GEOMETRIES {
                let k = lift_reduction_opdef_launched(&op, dtype, 2, block, grid).expect("lifts");
                assert_eq!(check_kernel(&k), Ok(()), "well formed");
                assert_eq!(
                    run(&k, dtype, &shape, &strides, true).expect("the launch runs"),
                    want,
                    "{rop:?} {dtype:?} axis0={axis0} keepdim={keepdim} on grid {grid} x block {block}"
                );
            }
        }
    }
}

#[test]
fn a_form_with_neither_order_independence_nor_a_bound_declines() {
    let (sum, ..) = case(ReduceOp::Prod, F32, false, false);
    let (mean, ..) = case(ReduceOp::Mean, I32, false, false);
    let (max, ..) = case(ReduceOp::Max, F32, false, false);
    for (op, dtype) in [(&sum, F32), (&mean, I32)] {
        assert!(
            matches!(
                lift_reduction_opdef_launched(op, dtype, 2, 8, 2),
                Err(LiftDecline::UnsupportedForm(_))
            ),
            "{dtype:?} {:?}",
            op.name
        );
    }
    for (block, grid) in [(0, 2), (6, 2), (8, 0)] {
        assert!(
            matches!(
                lift_reduction_opdef_launched(&max, F32, 2, block, grid),
                Err(LiftDecline::BadLaunch)
            ),
            "block {block} grid {grid}"
        );
    }
    // No kept axis (a rank-1 reduction) is outside this form.
    let one_d = OpDef::reduction_axes(
        "r",
        1,
        &[F32],
        Expr(input(0).0),
        ReduceOp::Max,
        AxisMask::EMPTY,
        false,
    );
    assert!(matches!(
        lift_reduction_opdef_launched(&one_d, F32, 1, 8, 2),
        Err(LiftDecline::UnsupportedForm(_))
    ));
}

/// The block that holds the `Barrier`s directly.
fn row_block(b: &mut Block) -> Option<&mut Block> {
    if b.0.iter().any(|s| matches!(s, Stmt::Barrier { .. })) {
        return Some(b);
    }
    for s in &mut b.0 {
        if let Stmt::For { body, .. } | Stmt::While { body, .. } = s {
            if let Some(r) = row_block(body) {
                return Some(r);
            }
        }
    }
    None
}

fn barrier_positions(b: &Block) -> Vec<usize> {
    (0..b.0.len())
        .filter(|&i| matches!(b.0[i], Stmt::Barrier { .. }))
        .collect()
}

fn mutate(k: &KernelFn, f: impl FnOnce(&mut Block)) -> KernelFn {
    let mut m = k.clone();
    f(&mut m.body);
    m
}

/// A launched kernel with its serial answer, so a broken copy can be compared to it.
struct Rig {
    kernel: KernelFn,
    dtype: ElementKind,
    shape: Vec<i64>,
    strides: Vec<i64>,
    want: Vec<u8>,
}

impl Rig {
    fn new(rop: ReduceOp, dtype: ElementKind, block: u32, grid: u32) -> Self {
        let (op, shape, strides) = case(rop, dtype, false, false);
        let serial = lift_reduction_opdef(&op, dtype, 2).expect("lifts");
        let want = run(&serial, dtype, &shape, &strides, false).expect("the serial kernel runs");
        let kernel = lift_reduction_opdef_launched(&op, dtype, 2, block, grid).expect("lifts");
        let rig = Self {
            kernel,
            dtype,
            shape,
            strides,
            want,
        };
        // Control: the unbroken kernel agrees, or every check below proves nothing.
        assert_eq!(rig.launch(&rig.kernel), Ok(rig.want.clone()));
        rig
    }

    fn launch(&self, k: &KernelFn) -> Result<Vec<u8>, KernelTrap> {
        run(k, self.dtype, &self.shape, &self.strides, true)
    }

    fn broken(&self, f: impl FnOnce(&mut Block)) -> Result<Vec<u8>, KernelTrap> {
        self.launch(&mutate(&self.kernel, f))
    }

    fn disagrees(&self, f: impl FnOnce(&mut Block)) -> bool {
        self.broken(f) != Ok(self.want.clone())
    }
}

fn remove_barrier(b: &mut Block, nth: usize) {
    let rb = row_block(b).expect("a row block");
    let at = barrier_positions(rb)[nth];
    rb.0.remove(at);
}

/// `Sum` over `I32` on 8-thread blocks: a double count or a missed partial changes the answer.
fn sum_rig() -> Rig {
    Rig::new(ReduceOp::Sum, I32, 8, 3)
}

#[test]
fn a_removed_first_barrier_reads_unwritten_shared_memory() {
    // Before any round has published, the reader finds the shared array unwritten.
    assert!(matches!(
        sum_rig().broken(|b| remove_barrier(b, 0)),
        Err(KernelTrap::UninitializedShared { .. })
    ));
}

#[test]
fn a_removed_round_barrier_is_a_data_race() {
    // Between two rounds, a thread reads a slot another thread is writing: a race, by name.
    assert!(matches!(
        sum_rig().broken(|b| remove_barrier(b, 1)),
        Err(KernelTrap::DataRace)
    ));
}

#[test]
fn a_dropped_last_round_disagrees() {
    // Thread 0 never combines thread 1's partial.
    assert!(sum_rig().disagrees(|b| {
        let rb = row_block(b).expect("a row block");
        let last = *barrier_positions(rb).last().expect("barriers");
        rb.0.drain(last - 1..=last);
    }));
}

#[test]
fn an_unstrided_slice_double_counts() {
    // Every thread folds the whole row (step 1): a Sum counts each element once per thread.
    assert!(sum_rig().disagrees(|b| {
        let rb = row_block(b).expect("a row block");
        let Some(Stmt::For { step, .. }) = rb.0.iter_mut().find(|s| matches!(s, Stmt::For { .. }))
        else {
            panic!("the slice loop")
        };
        *step = ScalarExpr::ConstInt(1, I32);
    }));
}

#[test]
fn blocks_starting_at_row_zero_skip_rows() {
    // With 3 blocks, rows other than 0, 3, 6 are never written.
    assert!(sum_rig().disagrees(|b| {
        let Some(Stmt::For { start, .. }) = b.0.first_mut() else {
            panic!("the row loop")
        };
        *start = ScalarExpr::ConstInt(0, I32);
    }));
}

/// A combine that ignores the partner's `have` flag lets a thread with nothing to fold (block 64 >
/// 37 columns) contribute its identity 0, which beats every negative `Max`. The shared corpus has
/// positives in every row, so this needs an all-negative input of its own.
#[test]
fn a_combine_that_ignores_have_picks_up_an_empty_threads_zero() {
    let (op, ..) = case(ReduceOp::Max, F32, false, false);
    let launched = lift_reduction_opdef_launched(&op, F32, 2, 64, 3).expect("lifts");
    let serial = lift_reduction_opdef(&op, F32, 2).expect("lifts");
    let ignores_have = mutate(&launched, |b| {
        let rb = row_block(b).expect("a row block");
        for s in &mut rb.0 {
            let Stmt::If { then, .. } = s else { continue };
            for t in &mut then.0 {
                if let Stmt::If {
                    cond: ScalarExpr::Local(_),
                    then: inner,
                    ..
                } = t
                {
                    *t = Stmt::If {
                        cond: ScalarExpr::ConstInt(1, I32),
                        then: Block(std::mem::take(&mut inner.0)),
                        els: Block(vec![]),
                    };
                }
            }
        }
    });
    let negative = |k: &KernelFn, launch: bool| {
        let vals: Vec<u8> = (0..ROWS * COLS)
            .flat_map(|i| (-1.0f32 - (i % 5) as f32).to_le_bytes())
            .collect();
        let mut a = vec![
            KernelArg::Buffer(TypedBuffer::new(F32, vec![ROWS, COLS], vec![COLS, 1], vals)),
            KernelArg::Buffer(TypedBuffer::new(F32, vec![ROWS], vec![1], vec![0u8; 36])),
            KernelArg::Int(i128::from(ROWS)),
            KernelArg::Int(i128::from(COLS)),
        ];
        if launch {
            run_launch(k, &mut a)?;
        } else {
            run_kernel(k, &mut a)?;
        }
        let KernelArg::Buffer(o) = &a[1] else {
            unreachable!("the output follows the input")
        };
        Ok::<_, KernelTrap>(o.bytes.clone())
    };
    let want = negative(&serial, false).expect("serial");
    assert_eq!(negative(&launched, true), Ok(want.clone()), "the control");
    assert_ne!(negative(&ignores_have, true), Ok(want));
}
