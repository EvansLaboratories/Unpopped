// SPDX-License-Identifier: MIT OR Apache-2.0
//! **The serialized thread-loop C agrees with the launch interpreter** (ROADMAP U4, the cpu-c
//! reference; `docs/options-u4-sync-and-atomics.md`, "Who could use it"). A launched `KernelFn`
//! (execution ids, `Shared` allocations, `Barrier`, `AtomicRmw`) is lowered by `CpuC::lower_kernel` to C
//! that runs the blocks one after another and, inside a block, the threads one after another, splitting
//! the thread loop at each barrier; it is compiled with a real host C compiler, run, and its written
//! buffers compared byte for byte with `oracle::kernel_launch::run_launch`.
//!
//! The old value of an atomic depends on the order threads reach the element, which the IR does not
//! specify, so those kernels are compared on properties (a permutation), never on that order.

use unpopped::exec::{AllocDecl, ExecId, Extent, LaunchShape, MemorySpace, Scope};
use unpopped::ir::{BinaryOp, Expr, OpDef, ReduceOp, ScalarExpr, input};
use unpopped::kernel::{
    AtomicOp, Block, BufferRole, KernelFn, LocalDecl, MemoryOrder, Param, Place, Stmt,
};
use unpopped::kernel_lift::lift_reduction_opdef_launched;
use unpopped::oracle::TypedBuffer;
use unpopped::oracle::kernel_eval::KernelArg;
use unpopped_vocab::{AxisMask, ElementKind};

#[path = "support/cc.rs"]
mod cc;
#[path = "support/kernel_run.rs"]
mod kernel_run;
#[path = "support/launch_run.rs"]
mod launch_run;
use launch_run::agree_launch;

const I32: ElementKind = ElementKind::I32;
const U32: ElementKind = ElementKind::U32;

fn tid() -> ScalarExpr {
    ScalarExpr::Exec(ExecId::ThreadIdx(0))
}

fn u(v: i64) -> ScalarExpr {
    ScalarExpr::ConstInt(v, U32)
}

fn ci(v: i64) -> ScalarExpr {
    ScalarExpr::ConstInt(v, I32)
}

fn ibuf(vals: &[i32]) -> TypedBuffer {
    let n = vals.len() as i64;
    let bytes: Vec<u8> = vals.iter().flat_map(|v| v.to_le_bytes()).collect();
    TypedBuffer::new(I32, vec![n], vec![1], bytes)
}

/// `if tid < 1 { then }`.
fn thread_zero(then: Vec<Stmt>) -> Stmt {
    Stmt::If {
        cond: ScalarExpr::Binary(BinaryOp::CmpLt, Box::new(tid()), Box::new(u(1))),
        then: Block(then),
        els: Block(vec![]),
    }
}

fn shared_cell(index: ScalarExpr) -> Place {
    Place::Alloc { alloc: 0, index }
}

fn out_at(index: ScalarExpr) -> Place {
    Place::Logical {
        buffer: 0,
        coords: vec![index],
    }
}

/// A block of `threads` threads and `blocks` blocks, one `ReadWrite` `I32` buffer, a `Shared`
/// allocation of `bins` `I32` cells, one `I32` local.
fn histogram_kernel(threads: u32, blocks: u32, bins: usize, body: Vec<Stmt>) -> KernelFn {
    let mut k = KernelFn::new(
        "h",
        I32,
        vec![Param::buffer("out", I32, BufferRole::ReadWrite, Some(1))],
        vec![LocalDecl::scalar(I32)],
        Block(body),
    )
    .with_allocs(vec![AllocDecl::new(
        "bins",
        MemorySpace::Shared,
        I32,
        Extent::Static(bins as u32),
    )]);
    k.launch = Some(LaunchShape::new([threads, 1, 1], [blocks, 1, 1]));
    k
}

/// Each thread counts itself into bin `min(tid, 3)` of a Shared histogram; thread `t < 4` then adds the
/// bin to `out[t]`. Two barriers: after the zeroing, and before the read.
#[test]
fn a_shared_histogram_with_two_barriers_agrees() {
    let zero = thread_zero(
        (0..4)
            .map(|b| Stmt::Store {
                place: shared_cell(u(b)),
                value: ci(0),
            })
            .collect(),
    );
    let bin = ScalarExpr::Binary(BinaryOp::Min, Box::new(tid()), Box::new(u(3)));
    let count = Stmt::atomic_rmw(AtomicOp::Add, shared_cell(bin), ci(1));
    let publish = Stmt::If {
        cond: ScalarExpr::Binary(BinaryOp::CmpLt, Box::new(tid()), Box::new(u(4))),
        then: Block(vec![
            Stmt::Load {
                local: 0,
                place: shared_cell(tid()),
            },
            Stmt::atomic_rmw(AtomicOp::Add, out_at(tid()), ScalarExpr::Local(0)),
        ]),
        els: Block(vec![]),
    };
    let body = vec![
        zero,
        Stmt::barrier(Scope::Block),
        count,
        Stmt::barrier(Scope::Block),
        publish,
    ];
    for (threads, blocks) in [(8, 1), (12, 3), (4, 2)] {
        let k = histogram_kernel(threads, blocks, 4, body.clone());
        agree_launch(
            &format!("hist_{threads}x{blocks}"),
            &k,
            vec![KernelArg::Buffer(ibuf(&[100, 200, 300, 400]))],
        );
    }
}

/// The block-tree reductions the lift builds (a `For` holding a `Barrier` per round, a `Shared`
/// accumulator, a `have` flag for `Max`/`Min`) agree across geometries, including a block wider than
/// the row and more blocks than rows.
#[test]
fn the_block_tree_reductions_agree_across_geometries() {
    const ROWS: i64 = 9;
    const COLS: i64 = 37;
    let vals: Vec<i32> = (0..(ROWS * COLS) as i32)
        .map(|i| (i * 37 + 11) % 211 - 100)
        .collect();
    let input_buf = TypedBuffer::new(
        I32,
        vec![ROWS, COLS],
        vec![COLS, 1],
        vals.iter().flat_map(|v| v.to_le_bytes()).collect(),
    );
    for rop in [ReduceOp::Sum, ReduceOp::Max, ReduceOp::Min] {
        let op = OpDef::reduction_axes(
            "r",
            1,
            &[I32],
            Expr(input(0).0),
            rop,
            AxisMask::EMPTY,
            false,
        );
        for (grid, block) in [(1, 1), (1, 2), (2, 4), (3, 8), (1, 64), (12, 16)] {
            let k = lift_reduction_opdef_launched(&op, I32, 2, block, grid).expect("lifts");
            let args = vec![
                KernelArg::Buffer(input_buf.clone()),
                KernelArg::Buffer(TypedBuffer::new(
                    I32,
                    vec![ROWS],
                    vec![1],
                    vec![0; 4 * ROWS as usize],
                )),
                KernelArg::Int(i128::from(ROWS)),
                KernelArg::Int(i128::from(COLS)),
            ];
            agree_launch(&format!("tree_{rop:?}_{grid}x{block}"), &k, args);
        }
    }
}

/// `old = fetch_add(cell, 1); out[tid] = old` over one block: the old values are a permutation of
/// `start..start+n` in the C reference as in the interpreter.
#[test]
fn a_fetch_add_hands_out_distinct_slots_in_the_c_reference() {
    let cell = Place::Logical {
        buffer: 0,
        coords: vec![u(0)],
    };
    let slot = Place::Logical {
        buffer: 1,
        coords: vec![tid()],
    };
    let body = vec![
        Stmt::AtomicRmw {
            op: AtomicOp::Add,
            place: cell,
            value: ci(1),
            old: Some(0),
            order: MemoryOrder::Relaxed,
            scope: Scope::Device,
        },
        Stmt::Store {
            place: slot,
            value: ScalarExpr::Local(0),
        },
    ];
    let mut k = KernelFn::new(
        "f",
        I32,
        vec![
            Param::buffer("c", I32, BufferRole::ReadWrite, Some(1)),
            Param::buffer("o", I32, BufferRole::ReadWrite, Some(1)),
        ],
        vec![LocalDecl::scalar(I32)],
        Block(body),
    );
    k.launch = Some(LaunchShape::new([12, 1, 1], [1, 1, 1]));
    let args = vec![
        KernelArg::Buffer(ibuf(&[100])),
        KernelArg::Buffer(ibuf(&[0; 12])),
    ];
    let Some(c) = kernel_run::run_c("fetch_add_slots", &k, &args) else {
        return;
    };
    assert_eq!(c.status, 0);
    let word =
        |bytes: &[u8], i: usize| i32::from_le_bytes(bytes[4 * i..4 * i + 4].try_into().unwrap());
    assert_eq!(word(&c.outs[0], 0), 112, "the cell");
    let mut olds: Vec<i32> = (0..12).map(|i| word(&c.outs[1], i)).collect();
    olds.sort_unstable();
    assert_eq!(olds, (100..112).collect::<Vec<_>>());
}
