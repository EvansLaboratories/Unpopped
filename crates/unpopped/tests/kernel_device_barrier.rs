// SPDX-License-Identifier: MIT OR Apache-2.0
//! ROADMAP U4, slice S6a (`docs/options-u4-sync-and-atomics.md`): a `Barrier` of `Device` scope synchronizes the whole
//! grid. The oracle used to run blocks one after another, so it had nothing to wait for; now every block is resident:
//! blocks run in order, each to its end or to a `Device` barrier, and when every block waits at the same barrier they all
//! go on. Every access before a `Device` barrier is ordered before every access after it, whatever the blocks (so the
//! classic "write a partial per block, barrier, read them all" is race-free, and a `DataRace` without the barrier).
//!
//! `Cluster` scope and named barriers are still refused (S6b); `unpopped-cpu-c` still spells only `Block` barriers.

use unpopped::exec::{ExecId, LaunchShape, Scope};
use unpopped::ir::{BinaryOp, CastSpec, RoundMode, Saturation, ScalarExpr};
use unpopped::kernel::{
    Block, BufferRole, KernelError, KernelFn, LocalDecl, Param, Place, Stmt, check_kernel,
};
use unpopped::oracle::TypedBuffer;
use unpopped::oracle::kernel_eval::{KernelArg, KernelTrap};
use unpopped::oracle::kernel_launch::run_launch;
use unpopped_vocab::ElementKind;

const I32: ElementKind = ElementKind::I32;
const U32: ElementKind = ElementKind::U32;
const GRID: u32 = 3;

fn c(v: i64) -> ScalarExpr {
    ScalarExpr::ConstInt(v, I32)
}

fn local(i: u32) -> ScalarExpr {
    ScalarExpr::Local(i)
}

fn block_idx() -> ScalarExpr {
    ScalarExpr::Exec(ExecId::BlockIdx(0))
}

/// `blockIdx + 1` as an `I32`.
fn mine() -> ScalarExpr {
    let spec = CastSpec {
        to: I32,
        round: RoundMode::NearestEven,
        sat: Saturation::Wrap,
    };
    ScalarExpr::Add(
        Box::new(ScalarExpr::Cast(Box::new(block_idx()), spec)),
        Box::new(c(1)),
    )
}

fn at(buffer: u32, i: ScalarExpr) -> Place {
    Place::Logical {
        buffer,
        coords: vec![i],
    }
}

fn at_const(buffer: u32, i: u32) -> Place {
    at(buffer, ScalarExpr::ConstInt(i64::from(i), U32))
}

fn barrier(scope: Scope) -> Stmt {
    Stmt::barrier(scope)
}

/// Thread 0 of each block: `then`.
fn thread0(then: Vec<Stmt>) -> Stmt {
    Stmt::If {
        cond: ScalarExpr::Binary(
            BinaryOp::CmpEq,
            Box::new(ScalarExpr::Exec(ExecId::ThreadIdx(0))),
            Box::new(ScalarExpr::ConstInt(0, U32)),
        ),
        then: Block(then),
        els: Block(vec![]),
    }
}

/// `dst = src[0] + src[1] + src[2]`, through locals `1..=3` (declared by these loads).
fn sum_all(src: u32, dst: u32) -> Vec<Stmt> {
    let mut v: Vec<Stmt> = (0..GRID)
        .map(|i| Stmt::Load {
            local: 1 + i,
            place: at_const(src, i),
        })
        .collect();
    v.push(Stmt::Store {
        place: at(dst, block_idx()),
        value: ScalarExpr::Add(
            Box::new(ScalarExpr::Add(Box::new(local(1)), Box::new(local(2)))),
            Box::new(local(3)),
        ),
    });
    v
}

fn kernel(buffers: u32, block: u32, body: Vec<Stmt>) -> KernelFn {
    let params = (0..buffers)
        .map(|i| Param::buffer(format!("b{i}"), I32, BufferRole::ReadWrite, Some(1)))
        .collect();
    let locals = (0..4).map(|_| LocalDecl::scalar(I32)).collect();
    KernelFn::new("grid", I32, params, locals, Block(body))
        .with_launch(LaunchShape::new([block, 1, 1], [GRID, 1, 1]))
}

fn run(k: &KernelFn, buffers: u32) -> Result<Vec<Vec<i32>>, KernelTrap> {
    assert_eq!(check_kernel(k), Ok(()), "well formed");
    let mut args: Vec<KernelArg> = (0..buffers)
        .map(|_| KernelArg::Buffer(TypedBuffer::new(I32, vec![4], vec![1], vec![0; 16])))
        .collect();
    run_launch(k, &mut args)?;
    Ok(args
        .iter()
        .map(|a| {
            let KernelArg::Buffer(b) = a else {
                unreachable!()
            };
            b.bytes
                .chunks(4)
                .map(|w| i32::from_le_bytes(w.try_into().unwrap()))
                .collect()
        })
        .collect())
}

/// Each block writes `blockIdx + 1` into `part[blockIdx]`, then (after `between`) sums all three into `out[blockIdx]`.
fn partials_then_sum(between: Vec<Stmt>) -> KernelFn {
    let mut body = vec![thread0(vec![Stmt::Store {
        place: at(0, block_idx()),
        value: mine(),
    }])];
    body.extend(between);
    body.push(thread0(sum_all(0, 1)));
    kernel(2, 2, body)
}

#[test]
fn a_device_barrier_makes_every_blocks_partial_visible_to_every_block() {
    let k = partials_then_sum(vec![barrier(Scope::Device)]);
    let out = run(&k, 2).expect("race free");
    assert_eq!(out[1][..3], [6, 6, 6]);
}

#[test]
fn without_it_the_same_kernel_is_a_data_race() {
    for between in [vec![], vec![barrier(Scope::Block)]] {
        let k = partials_then_sum(between);
        assert_eq!(run(&k, 2), Err(KernelTrap::DataRace));
    }
}

#[test]
fn a_thread_keeps_its_locals_across_the_barrier() {
    // local 0 is set before the barrier and stored after it
    let k = kernel(
        2,
        2,
        vec![
            Stmt::Let {
                local: 0,
                init: mine(),
            },
            barrier(Scope::Device),
            thread0(vec![Stmt::Store {
                place: at(1, block_idx()),
                value: local(0),
            }]),
        ],
    );
    let out = run(&k, 2).expect("runs");
    assert_eq!(out[1][..3], [1, 2, 3]);
}

#[test]
fn two_device_barriers_in_sequence_are_two_phases() {
    // phase A: part[b] = b + 1; phase B: tmp[b] = part[0] + part[1] + part[2]; phase C: out[b] = tmp[0] + tmp[1] + tmp[2]
    let body = |second: bool| {
        let mut v = vec![
            thread0(vec![Stmt::Store {
                place: at(0, block_idx()),
                value: mine(),
            }]),
            barrier(Scope::Device),
            thread0(sum_all(0, 1)),
        ];
        if second {
            v.push(barrier(Scope::Device));
        }
        v.push(thread0(sum_all(1, 2)));
        v
    };
    let out = run(&kernel(3, 2, body(true)), 3).expect("two phases");
    assert_eq!(out[2][..3], [18, 18, 18]);
    // dropping the second barrier lets block 0 read tmp[1] and tmp[2] before blocks 1 and 2 write them
    assert_eq!(
        run(&kernel(3, 2, body(false)), 3),
        Err(KernelTrap::DataRace)
    );
}

#[test]
fn a_block_barrier_and_a_device_barrier_mix() {
    let k = partials_then_sum(vec![
        barrier(Scope::Block),
        barrier(Scope::Device),
        barrier(Scope::Block),
    ]);
    assert_eq!(run(&k, 2).expect("mixed")[1][..3], [6, 6, 6]);
}

#[test]
fn a_device_barrier_in_a_single_block_grid_is_fine() {
    let k = KernelFn::new(
        "one",
        I32,
        vec![Param::buffer("o", I32, BufferRole::ReadWrite, Some(1))],
        vec![],
        Block(vec![
            barrier(Scope::Device),
            thread0(vec![Stmt::Store {
                place: at_const(0, 0),
                value: c(5),
            }]),
        ]),
    )
    .with_launch(LaunchShape::new([4, 1, 1], [1, 1, 1]));
    assert_eq!(run(&k, 1).expect("one block")[0][0], 5);
}

#[test]
fn a_device_barrier_under_control_that_differs_between_blocks_is_divergent() {
    // blockIdx is uniform for a Block barrier but not for a Device one
    let under = |scope| {
        kernel(
            1,
            2,
            vec![Stmt::If {
                cond: ScalarExpr::Binary(
                    BinaryOp::CmpEq,
                    Box::new(block_idx()),
                    Box::new(ScalarExpr::ConstInt(0, U32)),
                ),
                then: Block(vec![barrier(scope)]),
                els: Block(vec![]),
            }],
        )
    };
    assert_eq!(
        check_kernel(&under(Scope::Device)),
        Err(KernelError::BarrierDivergent)
    );
    assert_eq!(check_kernel(&under(Scope::Block)), Ok(()));
}

#[test]
fn cluster_scope_is_still_refused() {
    let k = kernel(1, 2, vec![barrier(Scope::Cluster)]);
    assert_eq!(check_kernel(&k), Err(KernelError::BadBarrierScope));
}
