// SPDX-License-Identifier: MIT OR Apache-2.0
//! U3 row 4: the barrier-uniformity rule (decision D-R4-5, design note section 4).
//!
//! Three groups. **Accepted** programs show the rule is not "reject every barrier under control
//! flow" (a vacuous rule passes the next group). **Rejected** programs are genuinely divergent and
//! each reaches `BarrierDivergent` by a different route. **Correct but rejected** programs are the
//! four the note names: they are uniform in fact, the conservative rule refuses them, and each is
//! pinned here so a later relaxation flips a test on purpose instead of changing behaviour
//! silently.

use unpopped::exec::{AllocDecl, ExecId, Extent, LaunchShape, MemorySpace, Scope};
use unpopped::ir::{CastSpec, RoundMode, Saturation, ScalarExpr};
use unpopped::kernel::{
    Block, BufferRole, KernelError, KernelFn, LocalDecl, Param, Place, Stmt, check_kernel,
};
use unpopped_vocab::ElementKind;

const F32: ElementKind = ElementKind::F32;
const I32: ElementKind = ElementKind::I32;
const U32: ElementKind = ElementKind::U32;

const N: u32 = 0; // scalar param n: i32 (uniform)
const T: u32 = 1; // u32
const I: u32 = 2; // i32 counter
const V: u32 = 3; // i32 scratch
const W: u32 = 4; // i32 scratch

fn ci(v: i64) -> ScalarExpr {
    ScalarExpr::ConstInt(v, I32)
}

fn local(i: u32) -> ScalarExpr {
    ScalarExpr::Local(i)
}

fn to_i32(e: ScalarExpr) -> ScalarExpr {
    ScalarExpr::Cast(
        Box::new(e),
        CastSpec {
            to: I32,
            round: RoundMode::NearestEven,
            sat: Saturation::Wrap,
        },
    )
}

fn exec(id: ExecId) -> ScalarExpr {
    ScalarExpr::Exec(id)
}

/// A condition built from a `U32` id: non-zero when the thread index differs from `1`.
fn tid_cond() -> ScalarExpr {
    ScalarExpr::Sub(
        Box::new(exec(ExecId::ThreadIdx(0))),
        Box::new(ScalarExpr::ConstInt(1, U32)),
    )
}

fn barrier() -> Stmt {
    Stmt::Barrier {
        scope: Scope::Block,
    }
}

fn if_(cond: ScalarExpr, then: Vec<Stmt>) -> Stmt {
    Stmt::If {
        cond,
        then: Block(then),
        els: Block(vec![]),
    }
}

fn for_(local: u32, end: ScalarExpr, body: Vec<Stmt>) -> Stmt {
    Stmt::For {
        local,
        start: ci(0),
        end,
        step: ci(1),
        body: Block(body),
    }
}

fn load_flag(into: u32) -> Stmt {
    Stmt::Load {
        local: into,
        place: Place::Logical {
            buffer: 0,
            coords: vec![ci(0)],
        },
    }
}

fn kernel(body: Vec<Stmt>) -> KernelFn {
    KernelFn::new(
        "uniform",
        F32,
        vec![
            Param::buffer("flags", I32, BufferRole::Read, Some(1)),
            Param::scalar("n", I32, N),
        ],
        vec![
            LocalDecl::scalar(I32),
            LocalDecl::scalar(U32),
            LocalDecl::scalar(I32),
            LocalDecl::scalar(I32),
            LocalDecl::scalar(I32),
        ],
        Block(body),
    )
    .with_launch(LaunchShape::new([32, 1, 1], [1, 1, 1]).with_subgroup(32, 32))
}

fn ok(body: Vec<Stmt>) {
    assert_eq!(check_kernel(&kernel(body)), Ok(()));
}

fn divergent(body: Vec<Stmt>) {
    assert_eq!(
        check_kernel(&kernel(body)),
        Err(KernelError::BarrierDivergent)
    );
}

// ---- accepted: the rule is not "reject every barrier under control" ----------------------------

#[test]
fn a_barrier_at_the_top_level_is_accepted() {
    ok(vec![barrier()]);
}

#[test]
fn a_barrier_in_a_loop_bounded_by_a_parameter_is_accepted() {
    ok(vec![for_(I, local(N), vec![barrier()])]);
}

#[test]
fn a_barrier_under_a_condition_on_a_parameter_is_accepted() {
    ok(vec![if_(local(N), vec![barrier()])]);
}

#[test]
fn a_barrier_in_a_while_on_a_parameter_set_local_is_accepted() {
    // v = n; while v { barrier; v = v - 1 }: every thread runs the same count.
    ok(vec![
        Stmt::Let {
            local: V,
            init: local(N),
        },
        Stmt::While {
            cond: local(V),
            body: Block(vec![
                barrier(),
                Stmt::Assign {
                    local: V,
                    value: ScalarExpr::Sub(Box::new(local(V)), Box::new(ci(1))),
                },
            ]),
        },
    ]);
}

#[test]
fn a_barrier_after_a_loop_with_a_uniform_break_is_accepted() {
    ok(vec![
        for_(I, local(N), vec![if_(local(N), vec![Stmt::Break])]),
        barrier(),
    ]);
}

#[test]
fn a_divergent_branch_with_no_barrier_inside_is_accepted() {
    // Control for the rejected group: the same divergent condition, no barrier under it.
    ok(vec![if_(tid_cond(), vec![Stmt::Return]), barrier()]);
}

// ---- rejected: each is divergent by a different route -----------------------------------------

#[test]
fn a_barrier_under_a_thread_dependent_if_is_rejected() {
    divergent(vec![if_(tid_cond(), vec![barrier()])]);
}

#[test]
fn a_barrier_in_the_else_arm_of_a_divergent_if_is_rejected() {
    divergent(vec![Stmt::If {
        cond: tid_cond(),
        then: Block(vec![]),
        els: Block(vec![barrier()]),
    }]);
}

#[test]
fn a_barrier_in_a_loop_with_a_thread_dependent_bound_is_rejected() {
    divergent(vec![for_(
        I,
        to_i32(exec(ExecId::ThreadIdx(0))),
        vec![barrier()],
    )]);
}

#[test]
fn a_barrier_in_a_while_on_a_thread_dependent_condition_is_rejected() {
    divergent(vec![Stmt::While {
        cond: tid_cond(),
        body: Block(vec![barrier(), Stmt::Break]),
    }]);
}

#[test]
fn a_barrier_under_a_local_derived_from_an_exec_id_is_rejected() {
    divergent(vec![
        Stmt::Let {
            local: T,
            init: exec(ExecId::ThreadIdx(0)),
        },
        Stmt::Let {
            local: V,
            init: to_i32(local(T)),
        },
        if_(local(V), vec![barrier()]),
    ]);
}

#[test]
fn a_value_assigned_under_divergent_control_is_divergent() {
    // v = 0; if tid { v = 1 }; if v { barrier }: v carries the divergence by control, not data.
    divergent(vec![
        Stmt::Let {
            local: V,
            init: ci(0),
        },
        if_(
            tid_cond(),
            vec![Stmt::Assign {
                local: V,
                value: ci(1),
            }],
        ),
        if_(local(V), vec![barrier()]),
    ]);
}

#[test]
fn a_divergence_carried_round_a_loop_is_seen() {
    // v = 0; for i in 0..n { if v { barrier }; v = tid }: v is uniform on pass one only.
    divergent(vec![
        Stmt::Let {
            local: V,
            init: ci(0),
        },
        for_(
            I,
            local(N),
            vec![
                if_(local(V), vec![barrier()]),
                Stmt::Assign {
                    local: V,
                    value: to_i32(exec(ExecId::ThreadIdx(0))),
                },
            ],
        ),
    ]);
}

#[test]
fn a_divergence_that_needs_two_rounds_to_reach_the_barrier_is_seen() {
    // for i in 0..n { if a { barrier }; a = b; b = tid }: b is divergent after one walk, a only
    // after two, and the barrier reads a. One walk plus the check would miss it.
    divergent(vec![
        Stmt::Let {
            local: V,
            init: ci(0),
        },
        Stmt::Let {
            local: W,
            init: ci(0),
        },
        for_(
            I,
            local(N),
            vec![
                if_(local(V), vec![barrier()]),
                Stmt::Assign {
                    local: V,
                    value: local(W),
                },
                Stmt::Assign {
                    local: W,
                    value: to_i32(exec(ExecId::ThreadIdx(0))),
                },
            ],
        ),
    ]);
}

#[test]
fn a_barrier_before_a_divergent_break_in_the_same_loop_is_rejected() {
    // for i in 0..n { barrier; if tid { break } }: pass two starts with fewer threads.
    divergent(vec![for_(
        I,
        local(N),
        vec![barrier(), if_(tid_cond(), vec![Stmt::Break])],
    )]);
}

#[test]
fn a_barrier_after_a_divergent_continue_is_rejected() {
    divergent(vec![for_(
        I,
        local(N),
        vec![if_(tid_cond(), vec![Stmt::Continue]), barrier()],
    )]);
}

#[test]
fn a_value_loaded_from_shared_or_private_memory_is_divergent() {
    for space in [MemorySpace::Shared, MemorySpace::Private] {
        let mut k = kernel(vec![
            Stmt::Load {
                local: V,
                place: Place::Alloc {
                    alloc: 0,
                    index: ci(0),
                },
            },
            if_(local(V), vec![barrier()]),
        ]);
        k.allocs = vec![AllocDecl::new("a", space, I32, Extent::Static(4))];
        assert_eq!(
            check_kernel(&k),
            Err(KernelError::BarrierDivergent),
            "{space:?}"
        );
    }
}

#[test]
fn a_value_loaded_from_a_constant_allocation_at_a_uniform_index_is_uniform() {
    // Control for the load rule: a Constant array read at a parameter index is the one load that
    // is not divergent; the same kernel with a thread-dependent index is.
    let build = |index: ScalarExpr| {
        let mut k = kernel(vec![
            Stmt::Load {
                local: V,
                place: Place::Alloc { alloc: 0, index },
            },
            if_(local(V), vec![barrier()]),
        ]);
        k.allocs = vec![AllocDecl::new(
            "c",
            MemorySpace::Constant,
            I32,
            Extent::Static(4),
        )];
        k
    };
    assert_eq!(check_kernel(&build(local(N))), Ok(()));
    assert_eq!(
        check_kernel(&build(exec(ExecId::ThreadIdx(0)))),
        Err(KernelError::BarrierDivergent)
    );
}

#[test]
fn an_earlier_error_wins_over_divergence() {
    // The first violation in program order is reported; uniformity runs after the scope check.
    let mut body = vec![if_(tid_cond(), vec![barrier()])];
    body.insert(
        0,
        Stmt::Assign {
            local: W,
            value: ci(0),
        },
    );
    assert_eq!(
        check_kernel(&kernel(body)),
        Err(KernelError::UseBeforeLet { local: W })
    );
}

// ---- correct but rejected: pinned so a relaxation is a deliberate flip -------------------------

#[test]
fn rejected_1_a_condition_that_is_always_true_but_reads_an_exec_id() {
    // if tid.x < blockDim.x { barrier }: always true, yet reads an Exec leaf.
    divergent(vec![if_(
        ScalarExpr::Sub(
            Box::new(exec(ExecId::BlockDim(0))),
            Box::new(exec(ExecId::ThreadIdx(0))),
        ),
        vec![barrier()],
    )]);
}

#[test]
fn rejected_2_a_loop_bounded_by_a_value_every_thread_loads_identically() {
    // for i in 0..flags[0] { barrier }: data-dependent but uniform trip count.
    divergent(vec![load_flag(V), for_(I, local(V), vec![barrier()])]);
}

#[test]
fn rejected_3_a_local_assigned_from_subgroup_size() {
    // v = subgroup_size; if v { barrier }: uniform within the subgroup.
    divergent(vec![
        Stmt::Let {
            local: T,
            init: exec(ExecId::SubgroupSize),
        },
        Stmt::Let {
            local: V,
            init: to_i32(local(T)),
        },
        if_(local(V), vec![barrier()]),
    ]);
}

#[test]
fn rejected_4_a_barrier_after_a_loop_with_a_break_guarded_by_a_uniform_loaded_flag() {
    // for i in 0..n { f = flags[0]; if f { break } }; barrier
    divergent(vec![
        for_(
            I,
            local(N),
            vec![load_flag(W), if_(local(W), vec![Stmt::Break])],
        ),
        barrier(),
    ]);
}
