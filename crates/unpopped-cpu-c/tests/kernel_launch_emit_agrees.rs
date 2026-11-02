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
use unpopped::ir::{OobPolicy, WriteCombine, WriteIndex};
use unpopped::kernel::{
    AtomicOp, Block, BufferRole, KernelFn, LocalDecl, MemoryOrder, Param, Place, Stmt,
};
use unpopped::kernel_lift::{lift_reduction_opdef_launched, lift_scatter_opdef_launched};
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

/// Every thread of a 3-D block in a 2-D grid stores a code of all its ids at its global linear
/// position, computed from `BlockDim`, `GridDim`, `ThreadIdx` and `BlockIdx`: each id is spelled
/// right only if the whole output matches.
#[test]
fn three_dimensional_ids_agree() {
    let id = |e: ExecId| ScalarExpr::Exec(e);
    let add = |a, b| ScalarExpr::Add(Box::new(a), Box::new(b));
    let mul = |a, b| ScalarExpr::Mul(Box::new(a), Box::new(b));
    let block_lin = add(
        id(ExecId::ThreadIdx(0)),
        mul(
            id(ExecId::BlockDim(0)),
            add(
                id(ExecId::ThreadIdx(1)),
                mul(id(ExecId::BlockDim(1)), id(ExecId::ThreadIdx(2))),
            ),
        ),
    );
    let grid_lin = add(
        id(ExecId::BlockIdx(0)),
        mul(id(ExecId::GridDim(0)), id(ExecId::BlockIdx(1))),
    );
    let nt = mul(
        id(ExecId::BlockDim(0)),
        mul(id(ExecId::BlockDim(1)), id(ExecId::BlockDim(2))),
    );
    let at = add(mul(grid_lin, nt), block_lin);
    // 1 + tx + 10 ty + 100 tz + 1000 bx + 10000 by, so a swapped axis changes the bytes.
    let code = [
        (ExecId::ThreadIdx(0), 1),
        (ExecId::ThreadIdx(1), 10),
        (ExecId::ThreadIdx(2), 100),
        (ExecId::BlockIdx(0), 1000),
        (ExecId::BlockIdx(1), 10000),
    ]
    .into_iter()
    .fold(u(1), |acc, (e, w)| add(acc, mul(id(e), u(w))));
    let place = Place::Logical {
        buffer: 0,
        coords: vec![at],
    };
    let mut k = KernelFn::new(
        "ids",
        U32,
        vec![Param::buffer("o", U32, BufferRole::Write, Some(1))],
        vec![],
        Block(vec![Stmt::Store { place, value: code }]),
    );
    k.launch = Some(LaunchShape::new([4, 3, 2], [2, 3, 1]));
    let n = 4 * 3 * 2 * 2 * 3;
    let out = TypedBuffer::new(U32, vec![n], vec![1], vec![0; 4 * n as usize]);
    agree_launch("ids_3d", &k, vec![KernelArg::Buffer(out)]);
}

fn cell_kernel(op: AtomicOp, value: i64) -> KernelFn {
    let cell = Place::Logical {
        buffer: 0,
        coords: vec![u(0)],
    };
    let mut k = KernelFn::new(
        "c",
        I32,
        vec![Param::buffer("c", I32, BufferRole::ReadWrite, Some(1))],
        vec![],
        Block(vec![Stmt::atomic_rmw(op, cell, ci(value))]),
    );
    k.launch = Some(LaunchShape::new([6, 1, 1], [2, 1, 1]));
    k
}

/// The associative combines reach their final value whatever the order, so the C reference and the
/// interpreter agree byte for byte (the old value is not read here).
#[test]
fn the_associative_atomic_combines_agree() {
    for (op, v, start) in [
        (AtomicOp::Add, 7, 5),
        (AtomicOp::Max, 9, 3),
        (AtomicOp::Min, -4, 3),
        (AtomicOp::And, 0b1010, 0b0110),
        (AtomicOp::Or, 0b0101, 0b0010),
        (AtomicOp::Xor, 0b0110, 0b0011),
    ] {
        let k = cell_kernel(op, v);
        agree_launch(
            &format!("combine_{op:?}"),
            &k,
            vec![KernelArg::Buffer(ibuf(&[start]))],
        );
    }
}

/// `Exchange` is order dependent, so the C reference is held to the property the IR promises: the
/// values read back plus the final cell are the start value and every value written, once each.
/// `old = exchange(cell, tid + 1); out[tid] = old` over one block of 12 threads.
fn exchange_kernel() -> KernelFn {
    let cell = Place::Logical {
        buffer: 0,
        coords: vec![u(0)],
    };
    let slot = Place::Logical {
        buffer: 1,
        coords: vec![tid()],
    };
    let value = ScalarExpr::Cast(
        Box::new(ScalarExpr::Add(Box::new(tid()), Box::new(u(1)))),
        unpopped::ir::CastSpec {
            to: I32,
            round: unpopped::ir::RoundMode::NearestEven,
            sat: unpopped::ir::Saturation::Wrap,
        },
    );
    let body = vec![
        Stmt::AtomicRmw {
            op: AtomicOp::Exchange,
            place: cell,
            value,
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
        "x",
        I32,
        vec![
            Param::buffer("c", I32, BufferRole::ReadWrite, Some(1)),
            Param::buffer("o", I32, BufferRole::ReadWrite, Some(1)),
        ],
        vec![LocalDecl::scalar(I32)],
        Block(body),
    );
    k.launch = Some(LaunchShape::new([12, 1, 1], [1, 1, 1]));
    k
}

#[test]
fn exchange_conserves_every_value_in_the_c_reference() {
    let k = exchange_kernel();
    let args = vec![
        KernelArg::Buffer(ibuf(&[1000])),
        KernelArg::Buffer(ibuf(&[0; 12])),
    ];
    let Some(c) = kernel_run::run_c("exchange_conserved", &k, &args) else {
        return;
    };
    let word =
        |bytes: &[u8], i: usize| i32::from_le_bytes(bytes[4 * i..4 * i + 4].try_into().unwrap());
    let mut seen: Vec<i32> = (0..12).map(|i| word(&c.outs[1], i)).collect();
    seen.push(word(&c.outs[0], 0));
    seen.sort_unstable();
    let mut want: Vec<i32> = (1..=12).collect();
    want.push(1000);
    want.sort_unstable();
    assert_eq!(seen, want);
}

/// The launched integer scatter (atomic combines, duplicate targets, out-of-range targets skipped)
/// agrees with the interpreter on two geometries.
#[test]
fn the_launched_integer_scatters_agree() {
    const SRC: [i64; 2] = [9, 4];
    const DST: [i64; 2] = [5, 4];
    let i32s = |shape: &[i64], v: &dyn Fn(usize) -> i32| {
        let n: i64 = shape.iter().product();
        let bytes = (0..n as usize).flat_map(|i| v(i).to_le_bytes()).collect();
        TypedBuffer::new(I32, shape.to_vec(), vec![shape[1], 1], bytes)
    };
    let targets = [0, 4, 4, 1, 7, -2, 0, 2, 4, 3, 0, 1, 9, 2, 2, 4, 0, 1];
    for combine in [
        WriteCombine::AtomicAdd,
        WriteCombine::AtomicMax,
        WriteCombine::AtomicMin,
    ] {
        let op = match combine {
            WriteCombine::AtomicAdd => OpDef::scatter_add("s", &[I32], 0, I32),
            c => OpDef::scatter("s", &[I32], 0, I32).with_scatter(WriteIndex::ScatterIndexed {
                index_operand: 1,
                axis: 0,
                combine: c,
                oob: OobPolicy::Skip,
                index_dtype: I32,
            }),
        };
        for (grid, block) in [(2, 3), (4, 8)] {
            let k = lift_scatter_opdef_launched(&op, I32, 2, block, grid).expect("lifts");
            let args = vec![
                KernelArg::Buffer(i32s(&SRC, &|i| (i as i32 * 7 % 11) - 5)),
                KernelArg::Buffer(i32s(&SRC, &|i| targets[(i * 5 + i / 4) % targets.len()])),
                KernelArg::Buffer(i32s(&DST, &|i| i as i32 - 3)),
                KernelArg::Int(i128::from(SRC[0])),
                KernelArg::Int(i128::from(SRC[1])),
                KernelArg::Int(i128::from(DST[0])),
            ];
            agree_launch(&format!("scatter_{combine:?}_{grid}x{block}"), &k, args);
        }
    }
}

fn refused(k: &KernelFn) -> String {
    use unpopped::backend::{Backend, LowerError};
    match unpopped_cpu_c::CpuC.lower_kernel(k) {
        Err(LowerError::UnsupportedKernel { detail }) => detail,
        other => panic!("expected a typed refusal, got {other:?}"),
    }
}

fn with_body(body: Vec<Stmt>, threads: u32) -> KernelFn {
    let mut k = KernelFn::new(
        "r",
        I32,
        vec![Param::buffer("o", I32, BufferRole::ReadWrite, Some(1))],
        vec![LocalDecl::scalar(I32)],
        Block(body),
    )
    .with_allocs(vec![AllocDecl::new(
        "sh",
        MemorySpace::Shared,
        I32,
        Extent::Static(4),
    )]);
    k.launch = Some(LaunchShape::new([threads, 1, 1], [1, 1, 1]));
    k
}

/// A launch that requires a subgroup, reading `LaneId`.
fn lane_id_kernel() -> KernelFn {
    let lane = ScalarExpr::Cast(
        Box::new(ScalarExpr::Exec(ExecId::LaneId)),
        unpopped::ir::CastSpec {
            to: I32,
            round: unpopped::ir::RoundMode::NearestEven,
            sat: unpopped::ir::Saturation::Wrap,
        },
    );
    let mut k = with_body(
        vec![Stmt::Let {
            local: 0,
            init: lane,
        }],
        4,
    );
    k.launch = Some(LaunchShape::new([4, 1, 1], [1, 1, 1]).with_subgroup(4, 4));
    k
}

/// What the thread loop cannot spell is refused with a typed error naming it, not mis-spelled.
#[test]
fn what_the_thread_loop_cannot_spell_is_a_typed_refusal() {
    let barrier = |scope| Stmt::barrier(scope);
    let ret = Stmt::Return;
    let while_with_barrier = Stmt::While {
        cond: ci(0),
        body: Block(vec![barrier(Scope::Block)]),
    };
    let cases: Vec<(&str, KernelFn, &str)> = vec![
        (
            "subgroup barrier",
            with_body(vec![barrier(Scope::Subgroup)], 4),
            "Subgroup Barrier",
        ),
        (
            "while holding a barrier",
            with_body(vec![while_with_barrier], 4),
            "While loop that holds a Barrier",
        ),
        (
            "return with a barrier",
            with_body(vec![barrier(Scope::Block), ret.clone()], 4),
            "Return in a kernel that has a Barrier",
        ),
        ("over 1024 threads", with_body(vec![], 2048), "2048 threads"),
        ("lane id", lane_id_kernel(), "execution id"),
    ];
    for (name, k, want) in cases {
        let d = refused(&k);
        assert!(d.contains(want), "{name}: {d}");
    }
}

/// A `Return` in a kernel with no barrier ends only that thread: threads below 3 stop, the rest store.
/// (A `Return` spelled as leaving the whole kernel would stop everything at thread 0.)
#[test]
fn an_early_return_ends_one_thread_not_the_kernel() {
    let guard = Stmt::If {
        cond: ScalarExpr::Binary(BinaryOp::CmpLt, Box::new(tid()), Box::new(u(3))),
        then: Block(vec![Stmt::Return]),
        els: Block(vec![]),
    };
    let store = Stmt::Store {
        place: out_at(tid()),
        value: ci(7),
    };
    let mut k = KernelFn::new(
        "early",
        I32,
        vec![Param::buffer("o", I32, BufferRole::ReadWrite, Some(1))],
        vec![],
        Block(vec![guard, store]),
    );
    k.launch = Some(LaunchShape::new([8, 1, 1], [1, 1, 1]));
    agree_launch("early_return", &k, vec![KernelArg::Buffer(ibuf(&[0; 8]))]);
}

/// A kernel with no launch shape is spelled exactly as before: the serial C still agrees with the plain
/// interpreter (`kernel_emit_agrees.rs` holds the full corpus; this keeps the launch path honest about it).
#[test]
fn an_unlaunched_kernel_still_agrees() {
    let op = OpDef::elementwise(
        "add",
        2,
        &[I32],
        Expr(ScalarExpr::Add(Box::new(input(0).0), Box::new(input(1).0))),
    );
    let k = unpopped::kernel_lift::lift_elementwise_opdef(&op, I32, 1).expect("lifts");
    let args = vec![
        KernelArg::Buffer(ibuf(&[1, 2, 3, 4, 5])),
        KernelArg::Buffer(ibuf(&[10, 20, 30, 40, 50])),
        KernelArg::Buffer(ibuf(&[0; 5])),
        KernelArg::Int(5),
    ];
    kernel_run::agree("unlaunched_add", &k, args);
}
