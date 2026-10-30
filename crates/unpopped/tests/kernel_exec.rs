// SPDX-License-Identifier: MIT OR Apache-2.0
//! U3 row 4: execution ids, the launch shape and memory-space allocations in the kernel checker
//! (`docs/u3-row4-scopes-and-memory-spaces.md` sections 2-3).
//!
//! The baseline is a valid launched kernel; each test changes exactly one thing and names the one
//! error it must produce. The unmutated baseline passing (and the `control` assertions, which show
//! the nearest legal variant passing) is the positive control the mutations share.

use unpopped::exec::{AllocDecl, ExecId, Extent, LaunchShape, MemorySpace, Scope};
use unpopped::ir::ScalarExpr;
use unpopped::kernel::{
    Block, BufferRole, KernelError, KernelFn, LocalDecl, Param, Place, Stmt, check_kernel,
};
use unpopped::oracle::TypedBuffer;
use unpopped::oracle::kernel_eval::{KernelArg, KernelTrap, run_kernel};
use unpopped_vocab::ElementKind;

const F32: ElementKind = ElementKind::F32;
const I32: ElementKind = ElementKind::I32;
const U32: ElementKind = ElementKind::U32;

const N: u32 = 0; // scalar param n: i32
const T: u32 = 1; // thread index: u32
const X: u32 = 2; // loaded element: f32
const Y: u32 = 3; // element read back from shared: f32
const SPARE: u32 = 4; // an unused i32 local (mutations that need a fresh id)

const TILE: u32 = 0; // Shared F32 Static(16)
const TABLE: u32 = 1; // Constant F32 Static(4)

fn tid() -> ScalarExpr {
    ScalarExpr::Exec(ExecId::ThreadIdx(0))
}

fn local(i: u32) -> ScalarExpr {
    ScalarExpr::Local(i)
}

fn buf(buffer: u32, c: ScalarExpr) -> Place {
    Place::Logical {
        buffer,
        coords: vec![c],
    }
}

fn alloc(alloc: u32, index: ScalarExpr) -> Place {
    Place::Alloc { alloc, index }
}

fn allocs() -> Vec<AllocDecl> {
    vec![
        AllocDecl::new("tile", MemorySpace::Shared, F32, Extent::Static(16)),
        AllocDecl::new("table", MemorySpace::Constant, F32, Extent::Static(4)),
    ]
}

/// `t = threadIdx.x; x = in[t]; tile[t] = x; barrier; y = tile[t]; out[t] = y`
fn baseline_body() -> Vec<Stmt> {
    vec![
        Stmt::Let {
            local: T,
            init: tid(),
        },
        Stmt::Load {
            local: X,
            place: buf(0, local(T)),
        },
        Stmt::Store {
            place: alloc(TILE, local(T)),
            value: local(X),
        },
        Stmt::Barrier {
            scope: Scope::Block,
        },
        Stmt::Load {
            local: Y,
            place: alloc(TILE, local(T)),
        },
        Stmt::Store {
            place: buf(1, local(T)),
            value: local(Y),
        },
    ]
}

fn base_launch() -> LaunchShape {
    LaunchShape::new([16, 1, 1], [1, 1, 1]).with_subgroup(32, 32)
}

fn kernel(body: Vec<Stmt>) -> KernelFn {
    KernelFn::new(
        "exec",
        F32,
        vec![
            Param::buffer("in", F32, BufferRole::Read, Some(1)),
            Param::buffer("out", F32, BufferRole::Write, Some(1)),
            Param::scalar("n", I32, N),
        ],
        vec![
            LocalDecl::scalar(I32),
            LocalDecl::scalar(U32),
            LocalDecl::scalar(F32),
            LocalDecl::scalar(F32),
            LocalDecl::scalar(I32),
        ],
        Block(body),
    )
    .with_launch(base_launch())
    .with_allocs(allocs())
}

fn baseline() -> KernelFn {
    kernel(baseline_body())
}

fn with_launch(launch: LaunchShape) -> KernelFn {
    baseline().with_launch(launch)
}

/// The baseline with its first statement replaced by `t = <id>`.
fn reading(id: ExecId) -> Vec<Stmt> {
    let mut body = baseline_body();
    body[0] = Stmt::Let {
        local: T,
        init: ScalarExpr::Exec(id),
    };
    body
}

#[test]
fn the_baseline_is_valid() {
    assert_eq!(check_kernel(&baseline()), Ok(()));
}

// ---- execution ids ----------------------------------------------------------------------------

#[test]
fn an_exec_id_without_a_launch_is_refused() {
    let mut k = baseline();
    k.launch = None;
    assert_eq!(check_kernel(&k), Err(KernelError::ExecWithoutLaunch));
}

#[test]
fn every_exec_id_checks_under_a_launch_with_a_subgroup() {
    let ids = [
        ExecId::ThreadIdx(2),
        ExecId::BlockIdx(1),
        ExecId::BlockDim(0),
        ExecId::GridDim(2),
        ExecId::LaneId,
        ExecId::SubgroupId,
        ExecId::SubgroupSize,
    ];
    for id in ids {
        assert_eq!(check_kernel(&kernel(reading(id))), Ok(()), "{id:?}");
    }
}

#[test]
fn an_exec_axis_above_two_is_refused() {
    for id in [
        ExecId::ThreadIdx(3),
        ExecId::BlockIdx(3),
        ExecId::BlockDim(3),
        ExecId::GridDim(255),
    ] {
        let axis = id.axis().expect("an axis id");
        assert_eq!(
            check_kernel(&kernel(reading(id))),
            Err(KernelError::ExecAxisOutOfRange { axis }),
            "{id:?}"
        );
    }
}

#[test]
fn a_subgroup_id_needs_a_launch_that_names_a_subgroup() {
    for id in [ExecId::LaneId, ExecId::SubgroupId, ExecId::SubgroupSize] {
        let mut k = kernel(reading(id));
        k.launch = Some(LaunchShape::new([16, 1, 1], [1, 1, 1]));
        assert_eq!(
            check_kernel(&k),
            Err(KernelError::SubgroupIdWithoutSubgroup),
            "{id:?}"
        );
    }
}

#[test]
fn a_block_id_does_not_need_a_subgroup() {
    let mut k = baseline();
    k.launch = Some(LaunchShape::new([16, 1, 1], [1, 1, 1]));
    assert_eq!(check_kernel(&k), Ok(()));
}

#[test]
fn an_exec_id_types_as_u32() {
    // Control: the U32 local T takes the id (the baseline). Mutation: an I32 local does not.
    let mut body = baseline_body();
    body[0] = Stmt::Let {
        local: SPARE,
        init: tid(),
    };
    assert_eq!(
        check_kernel(&kernel(body)),
        Err(KernelError::TypeMismatch {
            what: "declaration and its local"
        })
    );
}

// ---- launch shape -----------------------------------------------------------------------------

#[test]
fn a_zero_extent_is_a_bad_launch() {
    for l in [
        LaunchShape::new([0, 1, 1], [1, 1, 1]),
        LaunchShape::new([16, 0, 1], [1, 1, 1]),
        LaunchShape::new([16, 1, 0], [1, 1, 1]),
        LaunchShape::new([16, 1, 1], [0, 1, 1]),
        LaunchShape::new([16, 1, 1], [1, 1, 0]),
    ] {
        assert_eq!(check_kernel(&with_launch(l)), Err(KernelError::BadLaunch));
    }
}

#[test]
fn a_bad_subgroup_range_is_a_bad_launch() {
    for (min, max) in [(0, 32), (0, 0), (33, 32)] {
        let l = LaunchShape::new([16, 1, 1], [1, 1, 1]).with_subgroup(min, max);
        assert_eq!(
            check_kernel(&with_launch(l)),
            Err(KernelError::BadLaunch),
            "{min}..={max}"
        );
    }
}

#[test]
fn a_subgroup_range_is_accepted() {
    let l = LaunchShape::new([16, 1, 1], [1, 1, 1]).with_subgroup(16, 64);
    assert_eq!(check_kernel(&with_launch(l)), Ok(()));
}

// ---- allocations ------------------------------------------------------------------------------

#[test]
fn a_global_allocation_is_refused() {
    let mut k = baseline();
    k.allocs[0] = AllocDecl::new("g", MemorySpace::Global, F32, Extent::Static(16));
    assert_eq!(check_kernel(&k), Err(KernelError::AllocSpace { alloc: 0 }));
}

#[test]
fn a_zero_static_extent_is_refused() {
    let mut k = baseline();
    k.allocs[1] = AllocDecl::new("table", MemorySpace::Constant, F32, Extent::Static(0));
    assert_eq!(check_kernel(&k), Err(KernelError::AllocExtent { alloc: 1 }));
}

#[test]
fn a_param_extent_must_be_an_integer_scalar_parameter() {
    // Control: dynamic shared memory sized by the scalar parameter `n`.
    let mut k = baseline();
    k.allocs[0] = AllocDecl::new("tile", MemorySpace::Shared, F32, Extent::Param(N));
    assert_eq!(check_kernel(&k), Ok(()));
    // Mutations: a local that is not a parameter; a local that does not exist.
    for bad in [T, 99] {
        let mut k = baseline();
        k.allocs[0] = AllocDecl::new("tile", MemorySpace::Shared, F32, Extent::Param(bad));
        assert_eq!(check_kernel(&k), Err(KernelError::AllocExtent { alloc: 0 }));
    }
}

#[test]
fn a_float_scalar_parameter_cannot_size_an_allocation() {
    let mut k = baseline();
    k.params.push(Param::scalar("f", F32, Y));
    k.allocs[0] = AllocDecl::new("tile", MemorySpace::Shared, F32, Extent::Param(Y));
    assert_eq!(check_kernel(&k), Err(KernelError::AllocExtent { alloc: 0 }));
}

// ---- Place::Alloc -----------------------------------------------------------------------------

#[test]
fn a_constant_allocation_can_be_read_but_not_written() {
    let mut body = baseline_body();
    body[4] = Stmt::Load {
        local: Y,
        place: alloc(TABLE, local(T)),
    };
    assert_eq!(check_kernel(&kernel(body)), Ok(()), "control: a read");
    let mut body = baseline_body();
    body[2] = Stmt::Store {
        place: alloc(TABLE, local(T)),
        value: local(X),
    };
    assert_eq!(
        check_kernel(&kernel(body)),
        Err(KernelError::WriteToConstant { alloc: TABLE })
    );
}

#[test]
fn shared_memory_needs_a_launch_and_private_does_not() {
    let unlaunched = |space| {
        let mut k = KernelFn::new(
            "p",
            F32,
            vec![],
            vec![],
            Block(vec![Stmt::Store {
                place: alloc(0, ScalarExpr::ConstInt(0, I32)),
                value: ScalarExpr::Const(1.0),
            }]),
        );
        k.allocs = vec![AllocDecl::new("a", space, F32, Extent::Static(4))];
        k
    };
    assert_eq!(
        check_kernel(&unlaunched(MemorySpace::Private)),
        Ok(()),
        "control"
    );
    assert_eq!(
        check_kernel(&unlaunched(MemorySpace::Shared)),
        Err(KernelError::SharedWithoutLaunch { alloc: 0 })
    );
}

#[test]
fn an_unknown_allocation_is_a_bad_place() {
    let mut body = baseline_body();
    body[2] = Stmt::Store {
        place: alloc(7, local(T)),
        value: local(X),
    };
    assert_eq!(
        check_kernel(&kernel(body)),
        Err(KernelError::BadPlace { buffer: 7 })
    );
}

#[test]
fn an_allocation_index_must_be_an_integer() {
    let mut body = baseline_body();
    body[2] = Stmt::Store {
        place: alloc(TILE, local(X)),
        value: local(X),
    };
    assert_eq!(
        check_kernel(&kernel(body)),
        Err(KernelError::BadPlace { buffer: TILE })
    );
}

#[test]
fn a_store_to_an_allocation_must_match_its_element_type() {
    let mut body = baseline_body();
    body[2] = Stmt::Store {
        place: alloc(TILE, local(T)),
        value: local(T),
    };
    assert_eq!(
        check_kernel(&kernel(body)),
        Err(KernelError::TypeMismatch { what: "Store" })
    );
}

#[test]
fn a_load_from_an_allocation_declares_its_element_type() {
    let mut body = baseline_body();
    body[4] = Stmt::Load {
        local: SPARE, // an I32 local cannot take an F32 element
        place: alloc(TILE, local(T)),
    };
    assert_eq!(
        check_kernel(&kernel(body)),
        Err(KernelError::TypeMismatch {
            what: "declaration and its local"
        })
    );
}

// ---- Barrier shape ----------------------------------------------------------------------------

#[test]
fn a_barrier_scope_must_be_subgroup_or_block() {
    for s in [Scope::Subgroup, Scope::Block] {
        let mut body = baseline_body();
        body[3] = Stmt::Barrier { scope: s };
        assert_eq!(check_kernel(&kernel(body)), Ok(()), "{s:?}");
    }
    for s in [Scope::Thread, Scope::Cluster, Scope::Device] {
        let mut body = baseline_body();
        body[3] = Stmt::Barrier { scope: s };
        assert_eq!(
            check_kernel(&kernel(body)),
            Err(KernelError::BadBarrierScope),
            "{s:?}"
        );
    }
}

#[test]
fn a_barrier_needs_a_launch() {
    let mut k = KernelFn::new(
        "b",
        F32,
        vec![],
        vec![],
        Block(vec![Stmt::Barrier {
            scope: Scope::Block,
        }]),
    );
    assert_eq!(check_kernel(&k), Err(KernelError::ExecWithoutLaunch));
    k.launch = Some(base_launch());
    assert_eq!(check_kernel(&k), Ok(()), "control: the same body, launched");
}

// ---- the interpreter says no, by name ---------------------------------------------------------

#[test]
fn the_interpreter_declines_a_launched_kernel_by_name() {
    let mut args = vec![
        KernelArg::Buffer(TypedBuffer::from_f32(&[16], &[0.0; 16])),
        KernelArg::Buffer(TypedBuffer::from_f32(&[16], &[0.0; 16])),
        KernelArg::Int(16),
    ];
    let r = run_kernel(&baseline(), &mut args);
    assert!(matches!(r, Err(KernelTrap::Unsupported(_))), "{r:?}");
    // Control: the same arguments run an unlaunched kernel (the whole of U2).
    let mut plain = KernelFn::new(
        "plain",
        F32,
        vec![
            Param::buffer("in", F32, BufferRole::Read, Some(1)),
            Param::buffer("out", F32, BufferRole::Write, Some(1)),
            Param::scalar("n", I32, N),
        ],
        vec![LocalDecl::scalar(I32)],
        Block(vec![Stmt::Store {
            place: buf(1, ScalarExpr::ConstInt(0, I32)),
            value: ScalarExpr::Const(1.0),
        }]),
    );
    plain.launch = None;
    assert_eq!(run_kernel(&plain, &mut args), Ok(()));
}
