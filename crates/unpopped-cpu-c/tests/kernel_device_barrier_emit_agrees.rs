// SPDX-License-Identifier: MIT OR Apache-2.0
//! **The serialized thread-loop C agrees with the launch interpreter across a `Device` barrier** (ROADMAP U4 S6a, the
//! cpu-c reference): a top-level `Device` barrier cuts the body into phases of the whole grid; every block's locals and
//! `Shared` memory persist across it, and each phase is one pass over the blocks (`unpopped-cpu-c/src/launch.rs`). The C
//! is compiled with a host compiler, run, and its written buffers compared byte for byte with `run_launch`.
//!
//! The kernels: each block writes a partial, a `Device` barrier, every block sums all the partials (the write side and the
//! read side are in different phases, so a C that ran each block to its end would read zeros); a thread's local and a
//! block's `Shared` array carried across the barrier; two barriers; a `Block` barrier mixed in. A `Device` barrier under
//! control, and one over too many resident threads, are refused typed.

use unpopped::backend::Backend;
use unpopped::exec::{AllocDecl, ExecId, LaunchShape, MemorySpace, Scope};
use unpopped::ir::{BinaryOp, CastSpec, RoundMode, Saturation, ScalarExpr};
use unpopped::kernel::{Block, BufferRole, KernelFn, LocalDecl, Param, Place, Stmt, check_kernel};
use unpopped::oracle::kernel_eval::KernelArg;
use unpopped_cpu_c::CpuC;
use unpopped_vocab::ElementKind;

#[path = "support/cc.rs"]
mod cc;
#[path = "support/kernel_data.rs"]
mod kernel_data;
// The shared compile-and-run drivers; this file uses a subset of them.
#[allow(dead_code)]
#[path = "support/kernel_run.rs"]
mod kernel_run;
#[allow(dead_code)]
#[path = "support/launch_run.rs"]
mod launch_run;
use kernel_data::zeroed;
use launch_run::agree_launch;

const I32: ElementKind = ElementKind::I32;
const U32: ElementKind = ElementKind::U32;
const GRID: u32 = 3;
const T: u32 = 4;

fn c(v: i64) -> ScalarExpr {
    ScalarExpr::ConstInt(v, I32)
}

fn local(i: u32) -> ScalarExpr {
    ScalarExpr::Local(i)
}

fn as_i32(e: ScalarExpr) -> ScalarExpr {
    let spec = CastSpec {
        to: I32,
        round: RoundMode::NearestEven,
        sat: Saturation::Wrap,
    };
    ScalarExpr::Cast(Box::new(e), spec)
}

fn block_idx() -> ScalarExpr {
    ScalarExpr::Exec(ExecId::BlockIdx(0))
}

fn thread_idx() -> ScalarExpr {
    ScalarExpr::Exec(ExecId::ThreadIdx(0))
}

/// `blockIdx.y * 100 + blockIdx.x * 10 + threadIdx + 1`, an `I32` distinct per thread of any grid up to 10 wide.
fn tag() -> ScalarExpr {
    let y = ScalarExpr::Mul(
        Box::new(as_i32(ScalarExpr::Exec(ExecId::BlockIdx(1)))),
        Box::new(c(100)),
    );
    let x = ScalarExpr::Mul(Box::new(as_i32(block_idx())), Box::new(c(10)));
    ScalarExpr::Add(
        Box::new(ScalarExpr::Add(
            Box::new(ScalarExpr::Add(Box::new(y), Box::new(x))),
            Box::new(as_i32(thread_idx())),
        )),
        Box::new(c(1)),
    )
}

/// The block's linear id on a 3-wide grid: `blockIdx.y * 3 + blockIdx.x`.
fn linear_block() -> ScalarExpr {
    ScalarExpr::Add(
        Box::new(ScalarExpr::Mul(
            Box::new(ScalarExpr::Exec(ExecId::BlockIdx(1))),
            Box::new(ScalarExpr::ConstInt(3, U32)),
        )),
        Box::new(block_idx()),
    )
}

fn at(buffer: u32, i: ScalarExpr) -> Place {
    Place::Logical {
        buffer,
        coords: vec![i],
    }
}

fn thread0(then: Vec<Stmt>) -> Stmt {
    Stmt::If {
        cond: ScalarExpr::Binary(
            BinaryOp::CmpEq,
            Box::new(thread_idx()),
            Box::new(ScalarExpr::ConstInt(0, U32)),
        ),
        then: Block(then),
        els: Block(vec![]),
    }
}

fn kernel(buffers: u32, locals: u32, allocs: Vec<AllocDecl>, body: Vec<Stmt>) -> KernelFn {
    kernel_on([GRID, 1, 1], buffers, locals, allocs, body)
}

fn kernel_on(
    grid: [u32; 3],
    buffers: u32,
    locals: u32,
    allocs: Vec<AllocDecl>,
    body: Vec<Stmt>,
) -> KernelFn {
    let params = (0..buffers)
        .map(|i| Param::buffer(format!("b{i}"), I32, BufferRole::ReadWrite, Some(1)))
        .collect();
    let decls = (0..locals).map(|_| LocalDecl::scalar(I32)).collect();
    KernelFn::new("grid", I32, params, decls, Block(body))
        .with_allocs(allocs)
        .with_launch(LaunchShape::new([T, 1, 1], grid))
}

fn args(buffers: u32) -> Vec<KernelArg> {
    (0..buffers)
        .map(|_| KernelArg::Buffer(zeroed(I32, &[i64::from(6 * T)])))
        .collect()
}

fn check(tag: &str, k: &KernelFn, buffers: u32) {
    assert_eq!(check_kernel(k), Ok(()), "{tag}: well formed");
    agree_launch(tag, k, args(buffers));
}

/// `dst[blockIdx] = src[0] + src[1] + src[2]`, through locals `1..=3`.
fn sum_all(src: u32, dst: u32) -> Vec<Stmt> {
    let mut v: Vec<Stmt> = (0..GRID)
        .map(|i| Stmt::Load {
            local: 1 + i,
            place: at(src, ScalarExpr::ConstInt(i64::from(i), U32)),
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

fn write_partial() -> Stmt {
    thread0(vec![Stmt::Store {
        place: at(0, block_idx()),
        value: tag(),
    }])
}

#[test]
fn partials_then_a_sum_across_the_grid() {
    let k = kernel(
        2,
        4,
        vec![],
        vec![
            write_partial(),
            Stmt::barrier(Scope::Device),
            thread0(sum_all(0, 1)),
        ],
    );
    check("dev_sum", &k, 2);
}

#[test]
fn two_phases_and_a_block_barrier_mixed_in() {
    let k = kernel(
        3,
        4,
        vec![],
        vec![
            write_partial(),
            Stmt::barrier(Scope::Block),
            Stmt::barrier(Scope::Device),
            thread0(sum_all(0, 1)),
            Stmt::barrier(Scope::Device),
            thread0(sum_all(1, 2)),
        ],
    );
    check("dev_two_phases", &k, 3);
}

#[test]
fn a_local_and_a_shared_array_survive_the_barrier() {
    // local 0 (a per-thread tag) and sh0[t] (a per-block array) are set before the barrier and read after it
    let sh = AllocDecl::new(
        "sh",
        MemorySpace::Shared,
        I32,
        unpopped::exec::Extent::Static(T),
    );
    let sh_at = Place::Alloc {
        alloc: 0,
        index: thread_idx(),
    };
    let k = kernel_on(
        [3, 2, 1],
        1,
        2,
        vec![sh],
        vec![
            Stmt::Let {
                local: 0,
                init: tag(),
            },
            Stmt::Store {
                place: sh_at.clone(),
                value: ScalarExpr::Add(Box::new(local(0)), Box::new(c(100))),
            },
            Stmt::barrier(Scope::Device),
            Stmt::Load {
                local: 1,
                place: sh_at,
            },
            Stmt::Store {
                place: at(
                    0,
                    ScalarExpr::Add(
                        Box::new(ScalarExpr::Mul(
                            Box::new(linear_block()),
                            Box::new(ScalarExpr::ConstInt(i64::from(T), U32)),
                        )),
                        Box::new(thread_idx()),
                    ),
                ),
                value: ScalarExpr::Add(Box::new(local(1)), Box::new(local(0))),
            },
        ],
    );
    check("dev_local_shared", &k, 1);
}

#[test]
fn a_two_dimensional_grid_and_a_scalar_parameter_across_the_barrier() {
    // grid [3, 2] (unequal extents, so a swapped x and y shows): block (bx, by) writes part[3 * by + bx]; after the
    // barrier every block sums the six and adds the
    // scalar parameter `k` (local 0, set once before phase 0 and read in phase 1)
    let id = ScalarExpr::Add(
        Box::new(ScalarExpr::Mul(
            Box::new(as_i32(ScalarExpr::Exec(ExecId::BlockIdx(1)))),
            Box::new(c(3)),
        )),
        Box::new(as_i32(block_idx())),
    );
    let loads = (0..6).map(|i| Stmt::Load {
        local: 1 + i,
        place: at(0, ScalarExpr::ConstInt(i64::from(i), U32)),
    });
    let total = (1..=6).map(local).fold(local(0), |acc, l| {
        ScalarExpr::Add(Box::new(acc), Box::new(l))
    });
    let mut after: Vec<Stmt> = loads.collect();
    after.push(Stmt::Store {
        place: at(1, id.clone()),
        value: total,
    });
    let body = vec![
        thread0(vec![Stmt::Store {
            place: at(0, id.clone()),
            value: ScalarExpr::Add(Box::new(id), Box::new(c(1))),
        }]),
        Stmt::barrier(Scope::Device),
        thread0(after),
    ];
    let mut params: Vec<Param> = (0..2)
        .map(|i| Param::buffer(format!("b{i}"), I32, BufferRole::ReadWrite, Some(1)))
        .collect();
    params.push(Param::scalar("k", I32, 0));
    let locals = (0..7).map(|_| LocalDecl::scalar(I32)).collect();
    let k = KernelFn::new("grid2", I32, params, locals, Block(body))
        .with_launch(LaunchShape::new([T, 1, 1], [3, 2, 1]));
    assert_eq!(check_kernel(&k), Ok(()));
    let mut a = args(2);
    a.push(KernelArg::Int(7));
    agree_launch("dev_grid2d", &k, a);
}

#[test]
fn the_phases_are_visible_in_the_c() {
    let k = kernel(
        2,
        4,
        vec![],
        vec![
            write_partial(),
            Stmt::barrier(Scope::Device),
            thread0(sum_all(0, 1)),
        ],
    );
    let g = CpuC.lower_kernel(&k).expect("a Device barrier lowers");
    assert!(g.source.contains("phase 0") && g.source.contains("phase 1"));
    assert!(g.source.contains("l0_all"), "per-grid local storage");
}

#[test]
fn a_device_barrier_under_control_is_refused_typed() {
    let nested = kernel(
        1,
        0,
        vec![],
        vec![Stmt::If {
            cond: c(1),
            then: Block(vec![Stmt::barrier(Scope::Device)]),
            els: Block(vec![]),
        }],
    );
    let err = CpuC.lower_kernel(&nested).expect_err("nested refused");
    assert!(format!("{err:?}").contains("Device"), "{err:?}");
}

#[test]
fn a_grid_too_big_to_keep_resident_is_refused_typed() {
    let big = KernelFn::new(
        "big",
        I32,
        vec![Param::buffer("b", I32, BufferRole::ReadWrite, Some(1))],
        vec![LocalDecl::scalar(I32)],
        Block(vec![Stmt::barrier(Scope::Device)]),
    )
    .with_launch(LaunchShape::new([1024, 1, 1], [65, 1, 1]));
    let err = CpuC.lower_kernel(&big).expect_err("too many resident");
    assert!(format!("{err:?}").contains("resident"), "{err:?}");
}
