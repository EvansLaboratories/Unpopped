// SPDX-License-Identifier: MIT OR Apache-2.0
//! ROADMAP U4, slice S6b-1 (`docs/options-u4-s6b-cluster-and-named-barriers.md`): the `Cluster` scope. A launch may name
//! cluster dimensions (`LaunchShape::with_cluster`; the grid must be a multiple of them); a group of that many blocks is a
//! cluster. A `Cluster` barrier synchronizes the blocks of one cluster, orders their accesses (and only theirs), and clusters
//! are independent of each other; a `Cluster`-scope atomic or fence reaches the blocks of its cluster and no further.
//!
//! The launch is 4 blocks of 2 threads in clusters of 2 blocks: clusters `{0, 1}` and `{2, 3}`.

use unpopped::exec::{ExecId, LaunchShape, Scope};
use unpopped::ir::{BinaryOp, CastSpec, RoundMode, Saturation, ScalarExpr};
use unpopped::kernel::{
    AtomicOp, Block, BufferRole, KernelError, KernelFn, LocalDecl, MemoryOrder, Param, Place, Stmt,
    check_kernel,
};
use unpopped::oracle::TypedBuffer;
use unpopped::oracle::kernel_eval::{KernelArg, KernelTrap};
use unpopped::oracle::kernel_launch::run_launch;
use unpopped_vocab::ElementKind;

const I32: ElementKind = ElementKind::I32;
const U32: ElementKind = ElementKind::U32;
const BLOCKS: u32 = 4;

fn c(v: i64) -> ScalarExpr {
    ScalarExpr::ConstInt(v, I32)
}

fn u(v: i64) -> ScalarExpr {
    ScalarExpr::ConstInt(v, U32)
}

fn local(i: u32) -> ScalarExpr {
    ScalarExpr::Local(i)
}

fn block_idx() -> ScalarExpr {
    ScalarExpr::Exec(ExecId::BlockIdx(0))
}

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

fn thread0(then: Vec<Stmt>) -> Stmt {
    Stmt::If {
        cond: ScalarExpr::Binary(
            BinaryOp::CmpEq,
            Box::new(ScalarExpr::Exec(ExecId::ThreadIdx(0))),
            Box::new(u(0)),
        ),
        then: Block(then),
        els: Block(vec![]),
    }
}

fn clustered(body: Vec<Stmt>) -> KernelFn {
    launch(Some([2, 1, 1]), body)
}

fn launch(cluster: Option<[u32; 3]>, body: Vec<Stmt>) -> KernelFn {
    let params = (0..2)
        .map(|i| Param::buffer(format!("b{i}"), I32, BufferRole::ReadWrite, Some(1)))
        .collect();
    let locals = (0..4).map(|_| LocalDecl::scalar(I32)).collect();
    let mut shape = LaunchShape::new([2, 1, 1], [BLOCKS, 1, 1]);
    if let Some(d) = cluster {
        shape = shape.with_cluster(d);
    }
    KernelFn::new("cl", I32, params, locals, Block(body)).with_launch(shape)
}

fn run(k: &KernelFn) -> Result<Vec<Vec<i32>>, KernelTrap> {
    assert_eq!(check_kernel(k), Ok(()), "well formed");
    let mut args: Vec<KernelArg> = (0..2)
        .map(|_| KernelArg::Buffer(TypedBuffer::new(I32, vec![8], vec![1], vec![0; 32])))
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

fn write_partial() -> Stmt {
    thread0(vec![Stmt::Store {
        place: at(0, block_idx()),
        value: mine(),
    }])
}

/// `out[blockIdx] = part[first] + part[first + 1]` where `first` is `base` (a U32 expression), through locals `1` and `2`.
fn sum_pair(base: ScalarExpr) -> Stmt {
    thread0(vec![
        Stmt::Load {
            local: 1,
            place: at(0, base.clone()),
        },
        Stmt::Load {
            local: 2,
            place: at(0, ScalarExpr::Add(Box::new(base), Box::new(c(1)))),
        },
        Stmt::Store {
            place: at(1, block_idx()),
            value: ScalarExpr::Add(Box::new(local(1)), Box::new(local(2))),
        },
    ])
}

/// The first block of this block's cluster: `(blockIdx / 2) * 2`.
fn cluster_base() -> ScalarExpr {
    ScalarExpr::Mul(Box::new(cluster_id()), Box::new(c(2)))
}

/// This block's cluster: `blockIdx / 2`, as an `I32` (the integer operators work at the compute type).
fn cluster_id() -> ScalarExpr {
    let spec = CastSpec {
        to: I32,
        round: RoundMode::NearestEven,
        sat: Saturation::Wrap,
    };
    ScalarExpr::Div(
        Box::new(ScalarExpr::Cast(Box::new(block_idx()), spec)),
        Box::new(c(2)),
    )
}

#[test]
fn a_cluster_barrier_makes_the_partials_of_a_cluster_visible_inside_it() {
    let k = clustered(vec![
        write_partial(),
        Stmt::barrier(Scope::Cluster),
        sum_pair(cluster_base()),
    ]);
    // partials are 1, 2, 3, 4: cluster {0,1} sums 1 + 2, cluster {2,3} sums 3 + 4
    assert_eq!(run(&k).expect("race free")[1][..4], [3, 3, 7, 7]);
}

#[test]
fn clusters_tile_the_grid_on_every_axis() {
    // grid [2, 4], clusters [1, 2]: block (bx, by) is in cluster (bx, by / 2); its cluster-mate is (bx, by ^ 1). Each
    // block writes part[2 * by + bx], a Cluster barrier, then reads its mate's partial.
    let spec = CastSpec {
        to: I32,
        round: RoundMode::NearestEven,
        sat: Saturation::Wrap,
    };
    let by = ScalarExpr::Cast(Box::new(ScalarExpr::Exec(ExecId::BlockIdx(1))), spec);
    let bx = ScalarExpr::Cast(Box::new(block_idx()), spec);
    let lin = |y: ScalarExpr| {
        ScalarExpr::Add(
            Box::new(ScalarExpr::Mul(Box::new(y), Box::new(c(2)))),
            Box::new(bx.clone()),
        )
    };
    let mate = ScalarExpr::Binary(BinaryOp::BitXor, Box::new(by.clone()), Box::new(c(1)));
    let k = |barrier: bool| {
        let mut body = vec![thread0(vec![Stmt::Store {
            place: at(0, lin(by.clone())),
            value: ScalarExpr::Add(Box::new(lin(by.clone())), Box::new(c(1))),
        }])];
        if barrier {
            body.push(Stmt::barrier(Scope::Cluster));
        }
        body.push(thread0(vec![
            Stmt::Load {
                local: 1,
                place: at(0, lin(mate.clone())),
            },
            Stmt::Store {
                place: at(1, lin(by.clone())),
                value: local(1),
            },
        ]));
        let params = (0..2)
            .map(|i| Param::buffer(format!("b{i}"), I32, BufferRole::ReadWrite, Some(1)))
            .collect();
        let locals = (0..4).map(|_| LocalDecl::scalar(I32)).collect();
        KernelFn::new("cl2", I32, params, locals, Block(body))
            .with_launch(LaunchShape::new([2, 1, 1], [2, 4, 1]).with_cluster([1, 2, 1]))
    };
    // out[2 * by + bx] = mate's partial = 2 * (by ^ 1) + bx + 1
    let out = run(&k(true)).expect("mates are ordered")[1].clone();
    assert_eq!(out, [3, 4, 1, 2, 7, 8, 5, 6]);
    // without the barrier the mate's partial is unordered
    assert_eq!(run(&k(false)), Err(KernelTrap::DataRace));
}

#[test]
fn it_orders_nothing_across_clusters() {
    // block 0 reads block 2's partial: another cluster, so the Cluster barrier does not order it
    let k = clustered(vec![
        write_partial(),
        Stmt::barrier(Scope::Cluster),
        sum_pair(c(1)),
    ]);
    assert_eq!(run(&k), Err(KernelTrap::DataRace));
    // a Device barrier does
    let k = clustered(vec![
        write_partial(),
        Stmt::barrier(Scope::Device),
        sum_pair(c(1)),
    ]);
    assert_eq!(run(&k).expect("device")[1][0], 2 + 3);
    // and with no barrier at all the in-cluster read is a race too
    let k = clustered(vec![write_partial(), sum_pair(cluster_base())]);
    assert_eq!(run(&k), Err(KernelTrap::DataRace));
}

#[test]
fn two_cluster_barriers_are_two_phases_of_each_cluster() {
    let tmp_sum = |src: u32, dst: u32, base: ScalarExpr| {
        thread0(vec![
            Stmt::Load {
                local: 1,
                place: at(src, base.clone()),
            },
            Stmt::Load {
                local: 2,
                place: at(src, ScalarExpr::Add(Box::new(base), Box::new(c(1)))),
            },
            Stmt::Store {
                place: at(dst, block_idx()),
                value: ScalarExpr::Add(Box::new(local(1)), Box::new(local(2))),
            },
        ])
    };
    // phase 1 writes part, phase 2 sums the pair into out, phase 3 sums out's pair again (each block reads its
    // cluster-mate's phase-2 result: ordered only by the second barrier)
    let body = |second: bool| {
        let mut v = vec![
            write_partial(),
            Stmt::barrier(Scope::Cluster),
            tmp_sum(0, 1, cluster_base()),
        ];
        if second {
            v.push(Stmt::barrier(Scope::Cluster));
        }
        v.push(thread0(vec![
            Stmt::Load {
                local: 3,
                place: at(1, cluster_base()),
            },
            Stmt::Store {
                place: at(0, block_idx()),
                value: local(3),
            },
        ]));
        v
    };
    assert!(run(&clustered(body(true))).is_ok());
    assert_eq!(run(&clustered(body(false))), Err(KernelTrap::DataRace));
}

#[test]
fn a_cluster_barrier_under_control_that_differs_between_blocks_is_divergent() {
    let under = |scope| {
        clustered(vec![Stmt::If {
            cond: ScalarExpr::Binary(BinaryOp::CmpEq, Box::new(block_idx()), Box::new(u(0))),
            then: Block(vec![Stmt::barrier(scope)]),
            els: Block(vec![]),
        }])
    };
    assert_eq!(
        check_kernel(&under(Scope::Cluster)),
        Err(KernelError::BarrierDivergent)
    );
    assert_eq!(check_kernel(&under(Scope::Block)), Ok(()));
}

#[test]
fn a_cluster_atomic_reaches_its_clusters_blocks_and_no_further() {
    // every block adds 1 to cell (blockIdx / 2): the two blocks of a cluster share a cell
    let add = |scope| {
        clustered(vec![thread0(vec![Stmt::AtomicRmw {
            op: AtomicOp::Add,
            place: at(0, cluster_id()),
            value: c(1),
            old: None,
            order: MemoryOrder::Relaxed,
            scope,
        }])])
    };
    assert_eq!(
        run(&add(Scope::Cluster)).expect("same cluster")[0][..2],
        [2, 2]
    );
    // every block adds to cell 0 instead: blocks of different clusters meet at Cluster scope
    let all_one_cell = |scope| {
        clustered(vec![thread0(vec![Stmt::AtomicRmw {
            op: AtomicOp::Add,
            place: at(0, u(0)),
            value: c(1),
            old: None,
            order: MemoryOrder::Relaxed,
            scope,
        }])])
    };
    assert_eq!(
        run(&all_one_cell(Scope::Cluster)),
        Err(KernelTrap::DataRace)
    );
    assert_eq!(run(&all_one_cell(Scope::Device)).expect("device")[0][0], 4);
}

#[test]
fn release_acquire_at_cluster_scope_orders_blocks_of_one_cluster() {
    // block 0 publishes data and a flag; block 1 (same cluster) acquires the flag and reads the data
    let k = clustered(vec![
        Stmt::If {
            cond: ScalarExpr::Binary(BinaryOp::CmpEq, Box::new(block_idx()), Box::new(u(0))),
            then: Block(vec![thread0(vec![
                Stmt::Store {
                    place: at(0, u(0)),
                    value: c(42),
                },
                Stmt::AtomicRmw {
                    op: AtomicOp::Exchange,
                    place: at(1, u(0)),
                    value: c(1),
                    old: None,
                    order: MemoryOrder::Release,
                    scope: Scope::Cluster,
                },
            ])]),
            els: Block(vec![]),
        },
        Stmt::If {
            cond: ScalarExpr::Binary(BinaryOp::CmpEq, Box::new(block_idx()), Box::new(u(1))),
            then: Block(vec![thread0(vec![
                Stmt::AtomicRmw {
                    op: AtomicOp::Or,
                    place: at(1, u(0)),
                    value: c(0),
                    old: Some(0),
                    order: MemoryOrder::Acquire,
                    scope: Scope::Cluster,
                },
                Stmt::If {
                    cond: ScalarExpr::Binary(BinaryOp::CmpEq, Box::new(local(0)), Box::new(c(1))),
                    then: Block(vec![
                        Stmt::Load {
                            local: 1,
                            place: at(0, u(0)),
                        },
                        Stmt::Store {
                            place: at(0, u(1)),
                            value: local(1),
                        },
                    ]),
                    els: Block(vec![]),
                },
            ])]),
            els: Block(vec![]),
        },
    ]);
    assert_eq!(run(&k).expect("ordered inside the cluster")[0][1], 42);
}

#[test]
fn the_launch_and_the_scope_are_checked() {
    let barrier = vec![Stmt::barrier(Scope::Cluster)];
    // no cluster dimensions: a Cluster barrier, fence or atomic has no meaning
    let none = launch(None, barrier.clone());
    assert_eq!(check_kernel(&none), Err(KernelError::BadBarrierScope));
    let fence = launch(
        None,
        vec![Stmt::Fence {
            scope: Scope::Cluster,
            order: MemoryOrder::Release,
        }],
    );
    assert!(matches!(
        check_kernel(&fence),
        Err(KernelError::SyncUnsupported { .. })
    ));
    let atomic = launch(
        None,
        vec![Stmt::AtomicRmw {
            op: AtomicOp::Add,
            place: at(0, u(0)),
            value: c(1),
            old: None,
            order: MemoryOrder::Relaxed,
            scope: Scope::Cluster,
        }],
    );
    assert!(matches!(
        check_kernel(&atomic),
        Err(KernelError::AtomicUnsupported { .. })
    ));
    // with dimensions they check
    assert_eq!(
        check_kernel(&launch(Some([2, 1, 1]), barrier.clone())),
        Ok(())
    );
    // the grid (4 x 1 x 1) must be a multiple of the cluster on every axis, and no extent may be zero
    for bad in [[3, 1, 1], [2, 2, 1], [0, 1, 1], [8, 1, 1]] {
        assert_eq!(
            check_kernel(&launch(Some(bad), barrier.clone())),
            Err(KernelError::BadLaunch),
            "{bad:?}"
        );
    }
}
