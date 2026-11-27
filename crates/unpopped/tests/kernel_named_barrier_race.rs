// SPDX-License-Identifier: MIT OR Apache-2.0
//! **A named barrier orders only what was released before it opened.** A thread that runs on into the next phase and releases
//! there must not order a waiter of the earlier phase after what it did (found by the Hillis-Steele conformance kernel, #457:
//! a missing sync between a step's read and write ran to completion with wrong values and no `DataRace`).
//!
//! Thirty-two threads (one warp; a named barrier counts whole warps) on one named barrier (`bar.sync 1, 32`): each stores its input to a shared tile, syncs, then adds its left
//! neighbour in place. With the second sync the read of `s[i-1]` is ordered before the neighbour's write; without it the
//! neighbour's write and the read are unordered and the launch is a `DataRace`.

use unpopped::exec::{AllocDecl, ExecId, Extent, LaunchShape, MemorySpace, Scope};
use unpopped::ir::{BinaryOp, ScalarExpr};
use unpopped::kernel::{
    BarrierKind, Block, BufferRole, KernelFn, LocalDecl, Param, Place, Stmt, check_kernel,
};
use unpopped::oracle::TypedBuffer;
use unpopped::oracle::kernel_eval::{KernelArg, KernelTrap};
use unpopped::oracle::kernel_launch::run_launch;
use unpopped_vocab::ElementKind;

const F32: ElementKind = ElementKind::F32;
const N: u32 = 32;

fn tid() -> ScalarExpr {
    ScalarExpr::Exec(ExecId::ThreadIdx(0))
}
fn tile(i: ScalarExpr) -> Place {
    Place::Alloc { alloc: 0, index: i }
}
fn sync() -> Stmt {
    Stmt::Barrier {
        scope: Scope::Block,
        kind: BarrierKind::Named { id: 1, count: N },
    }
}

/// `between`: whether the sync between the neighbour read and the in-place write is present.
fn kernel(between: bool) -> KernelFn {
    let u = |v: i64| ScalarExpr::ConstInt(v, ElementKind::U32);
    let mut body = vec![
        Stmt::Load {
            local: 0,
            place: Place::Logical {
                buffer: 0,
                coords: vec![tid()],
            },
        },
        Stmt::Store {
            place: tile(tid()),
            value: ScalarExpr::Local(0),
        },
        sync(),
        Stmt::Let {
            local: 1,
            init: ScalarExpr::Const(0.0),
        },
        Stmt::If {
            cond: ScalarExpr::Binary(BinaryOp::CmpGe, Box::new(tid()), Box::new(u(1))),
            then: Block(vec![
                Stmt::Load {
                    local: 2,
                    place: tile(ScalarExpr::Sub(Box::new(tid()), Box::new(u(1)))),
                },
                Stmt::Assign {
                    local: 1,
                    value: ScalarExpr::Local(2),
                },
            ]),
            els: Block(vec![]),
        },
    ];
    if between {
        body.push(sync());
    }
    body.push(Stmt::Load {
        local: 3,
        place: tile(tid()),
    });
    body.push(Stmt::Store {
        place: tile(tid()),
        value: ScalarExpr::Add(
            Box::new(ScalarExpr::Local(3)),
            Box::new(ScalarExpr::Local(1)),
        ),
    });
    body.push(sync());
    body.push(Stmt::Load {
        local: 4,
        place: tile(tid()),
    });
    body.push(Stmt::Store {
        place: Place::Logical {
            buffer: 1,
            coords: vec![tid()],
        },
        value: ScalarExpr::Local(4),
    });
    let params = vec![
        Param::buffer("input", F32, BufferRole::Read, Some(1)),
        Param::buffer("out", F32, BufferRole::Write, Some(1)),
    ];
    let locals = vec![LocalDecl::scalar(F32); 5];
    KernelFn::new("named_neighbour", F32, params, locals, Block(body))
        .with_launch(LaunchShape::new([N, 1, 1], [1, 1, 1]))
        .with_allocs(vec![AllocDecl::new(
            "tile",
            MemorySpace::Shared,
            F32,
            Extent::Static(N),
        )])
}

fn run(between: bool) -> Result<Vec<f32>, KernelTrap> {
    let k = kernel(between);
    assert_eq!(check_kernel(&k), Ok(()));
    let buf = |v: &[f32]| {
        KernelArg::Buffer(TypedBuffer::new(
            F32,
            vec![v.len() as i64],
            vec![1],
            v.iter().flat_map(|x| x.to_le_bytes()).collect(),
        ))
    };
    let input: Vec<f32> = (1..=N).map(|i| i as f32).collect();
    let mut args = vec![buf(&input), buf(&[0.0; N as usize])];
    run_launch(&k, &mut args)?;
    let KernelArg::Buffer(o) = &args[1] else {
        unreachable!("a buffer")
    };
    Ok(o.bytes
        .chunks(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect())
}

#[test]
fn the_synced_kernel_runs_and_adds_the_left_neighbour() {
    let want: Vec<f32> = (1..=N)
        .map(|i| if i == 1 { 1.0 } else { (2 * i - 1) as f32 })
        .collect();
    assert_eq!(run(true), Ok(want));
}

#[test]
fn a_missing_sync_between_the_read_and_the_write_is_a_data_race() {
    assert_eq!(run(false), Err(KernelTrap::DataRace));
}
