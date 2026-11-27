// SPDX-License-Identifier: MIT OR Apache-2.0
//! **One checker-valid kernel (or module) per [`Stmt`] variant** (ROADMAP U60, asked by vulkane), for an emitter's
//! decline test: the checker passes the fixture, so the refusal an emitter's test sees is the EMITTER's own text.
//!
//! Behind the `fixtures` cargo feature (and in this crate's own tests); a normal build does not carry it.
//!
//! * [`fixtures`] lists `(variant name, Fixture)` in the order of the `Stmt` enum; [`fixture`] looks one up.
//! * [`needs`] says in a few words what a fixture needs that is unusual (a module, a launch, tensor memory, ...).
//! * [`variant_name`] is an **exhaustive `match` with no wildcard arm**: a new `Stmt` variant is a compile error
//!   under the feature until it is listed in the `variants!` call below, and [`ALL`] is generated from that same list.
//!   The unit tests then assert that every name in [`ALL`] has exactly one fixture, that every fixture passes
//!   [`crate::kernel::check_kernel`] / [`crate::kernel::check_module`], and that each fixture really contains a statement of its own variant.
//!
//! The fixtures are small and are not meant to compute anything useful (the oracle may trap on some, for example a
//! fragment read before it is written): they are meant to be *valid* and to contain the one statement.

use crate::exec::{AllocDecl, ExecId, Extent, LaunchShape, MemorySpace, Scope};
use crate::ir::{BinaryOp, ReduceOp, ScalarExpr};
use crate::kernel::{
    AtomicOp, Block, BodyText, BufferRole, CacheHint, CallArg, CollectiveKind, Effects, ExternDecl,
    KernelError, KernelFn, LocalDecl, MemoryOrder, Module, OperandLayout, Param, Place, Stmt,
    TargetBody, Unroll,
};
use crate::mma::MmaAtom;
use unpopped_vocab::ElementKind;

const F32: ElementKind = ElementKind::F32;
const F16: ElementKind = ElementKind::F16;
const BF16: ElementKind = ElementKind::Bf16;
const I32: ElementKind = ElementKind::I32;
const U32: ElementKind = ElementKind::U32;
const U64: ElementKind = ElementKind::U64;
const U8: ElementKind = ElementKind::U8;

/// A fixture: a kernel `check_kernel` accepts, or a module `check_module` accepts at `entry`.
#[derive(Clone, Debug, PartialEq)]
#[non_exhaustive]
pub enum Fixture {
    /// A single kernel, checked with `check_kernel`.
    Kernel(KernelFn),
    /// A module, checked with `check_module(&module, entry)`.
    Module {
        /// The functions and declarations.
        module: Module,
        /// The entry function's index in `module.fns`.
        entry: u32,
    },
}

impl Fixture {
    /// Run the checker that suits this fixture.
    ///
    /// # Errors
    /// The checker's error, if the fixture is not valid (a fixture of this module always is).
    pub fn check(&self) -> Result<(), crate::kernel::KernelError> {
        match self {
            Self::Kernel(k) => crate::kernel::check_kernel(k),
            Self::Module { module, entry } => crate::kernel::check_module(module, *entry),
        }
    }

    /// Every function of the fixture: the one kernel, or every function of the module.
    #[must_use]
    pub fn functions(&self) -> Vec<&KernelFn> {
        match self {
            Self::Kernel(k) => vec![k],
            Self::Module { module, .. } => module.fns.iter().collect(),
        }
    }
}

/// The completeness guard: list every `Stmt` variant here, in enum order (`{ .. }` for a struct-like variant, nothing
/// for a unit one). The `match` has no wildcard arm, so a variant missing from the list does not compile; a name
/// that is listed twice or is not a variant does not compile either.
macro_rules! variants {
    ($($name:ident $({ $($dots:tt)* })?),* $(,)?) => {
        /// The name of the `Stmt` variant of `st`.
        #[must_use]
        pub fn variant_name(st: &Stmt) -> &'static str {
            match st {
                $(Stmt::$name $({ $($dots)* })? => stringify!($name),)*
            }
        }

        /// Every `Stmt` variant name, in the order of the enum.
        pub const ALL: &[&str] = &[$(stringify!($name)),*];
    };
}

variants! {
    Let { .. },
    Load { .. },
    Assign { .. },
    Store { .. },
    MaskedLoad { .. },
    MaskedStore { .. },
    If { .. },
    For { .. },
    While { .. },
    Break,
    Continue,
    Return,
    Barrier { .. },
    AtomicRmw { .. },
    AtomicCas { .. },
    MulticastLoadReduce { .. },
    GridDepWait,
    GridDepLaunchDependents,
    Fence { .. },
    Collective { .. },
    CollectiveTuple { .. },
    Mma { .. },
    MatrixLoad { .. },
    MbarInit { .. },
    MbarArrive { .. },
    MbarCompleteTx { .. },
    MbarWait { .. },
    MatrixStore { .. },
    TileLoad { .. },
    TileMma { .. },
    TileFill { .. },
    TileStore { .. },
    ProxyFence { .. },
    Wgmma { .. },
    WgmmaCommit,
    WgmmaWait { .. },
    Tcgen05Mma { .. },
    Tcgen05Commit { .. },
    Tcgen05Alloc { .. },
    Tcgen05Dealloc { .. },
    Tcgen05RelinquishPermit,
    Tcgen05Ld { .. },
    CopyAsync { .. },
    CopyBulk { .. },
    CopyUnpack { .. },
    AsyncCommit,
    AsyncWait { .. },
    Call { .. },
    CallTuple { .. },
    ExternCall { .. },
}

// ---- small constructors ----------------------------------------------------------------------------------------

fn c(v: f64) -> ScalarExpr {
    ScalarExpr::Const(v)
}

fn u(v: i64) -> ScalarExpr {
    ScalarExpr::ConstInt(v, U32)
}

fn ci(v: i64) -> ScalarExpr {
    ScalarExpr::ConstInt(v, I32)
}

fn l(i: u32) -> ScalarExpr {
    ScalarExpr::Local(i)
}

fn tid() -> ScalarExpr {
    ScalarExpr::Exec(ExecId::ThreadIdx(0))
}

fn mul(a: ScalarExpr, b: ScalarExpr) -> ScalarExpr {
    ScalarExpr::Mul(Box::new(a), Box::new(b))
}

fn add(a: ScalarExpr, b: ScalarExpr) -> ScalarExpr {
    ScalarExpr::Add(Box::new(a), Box::new(b))
}

/// Element `index` of buffer parameter 0.
fn buf_at(index: ScalarExpr) -> Place {
    Place::Logical {
        buffer: 0,
        coords: vec![index],
    }
}

/// Element 0 of buffer parameter 0.
fn buf0() -> Place {
    buf_at(u(0))
}

/// Element `index` of allocation `alloc`.
fn alloc_at(alloc: u32, index: ScalarExpr) -> Place {
    Place::Alloc { alloc, index }
}

fn buffer(elem: ElementKind, role: BufferRole) -> Param {
    Param::buffer("buf", elem, role, Some(1))
}

fn locals(tys: &[ElementKind]) -> Vec<LocalDecl> {
    tys.iter().copied().map(LocalDecl::scalar).collect()
}

/// A kernel computing in `compute`, with one rank-1 buffer parameter `buf` of `elem`/`role` when `buf` is given.
fn kernel(
    name: &str,
    compute: ElementKind,
    buf: Option<(ElementKind, BufferRole)>,
    locals: &[ElementKind],
    body: Vec<Stmt>,
) -> KernelFn {
    KernelFn::new(
        name,
        compute,
        buf.map(|(e, r)| buffer(e, r)).into_iter().collect(),
        self::locals(locals),
        Block(body),
    )
}

/// An `F32` kernel over one read-write `F32` buffer.
fn f32_kernel(name: &str, locals: &[ElementKind], body: Vec<Stmt>) -> KernelFn {
    kernel(name, F32, Some((F32, BufferRole::ReadWrite)), locals, body)
}

/// One block of 32 threads.
fn block32() -> LaunchShape {
    LaunchShape::new([32, 1, 1], [1, 1, 1])
}

/// One warp: 32 threads with a subgroup of 32.
fn warp() -> LaunchShape {
    block32().with_subgroup(32, 32)
}

/// One warpgroup: 128 threads, subgroups of 32.
fn warpgroup() -> LaunchShape {
    LaunchShape::new([128, 1, 1], [1, 1, 1]).with_subgroup(32, 32)
}

fn alloc(name: &str, space: MemorySpace, elem: ElementKind, len: u32) -> AllocDecl {
    AllocDecl::new(name, space, elem, Extent::Static(len))
}

/// The `Shared` 64-bit element an `mbarrier` lives in.
fn mbar_alloc() -> AllocDecl {
    alloc("bar", MemorySpace::Shared, U64, 1)
}

fn bar() -> Place {
    alloc_at(0, u(0))
}

fn on_thread0(body: Vec<Stmt>) -> Stmt {
    Stmt::If {
        cond: ScalarExpr::Binary(BinaryOp::CmpEq, Box::new(tid()), Box::new(u(0))),
        then: Block(body),
        els: Block(vec![]),
    }
}

// ---- the fixtures, in the order of the enum ---------------------------------------------------------------------

/// `Let`: declare a local.
fn let_() -> KernelFn {
    f32_kernel(
        "let",
        &[F32],
        vec![Stmt::Let {
            local: 0,
            init: c(1.0),
        }],
    )
}

/// `Load`: read a buffer element into a local.
fn load() -> KernelFn {
    f32_kernel(
        "load",
        &[F32],
        vec![Stmt::Load {
            local: 0,
            place: buf0(),
        }],
    )
}

/// `Assign`: mutate a declared local.
fn assign() -> KernelFn {
    f32_kernel(
        "assign",
        &[F32],
        vec![
            Stmt::Let {
                local: 0,
                init: c(1.0),
            },
            Stmt::Assign {
                local: 0,
                value: c(2.0),
            },
        ],
    )
}

/// `Store`: write a buffer element.
fn store() -> KernelFn {
    f32_kernel(
        "store",
        &[],
        vec![Stmt::Store {
            place: buf0(),
            value: c(1.0),
        }],
    )
}

/// `MaskedLoad`: a predicated read with a fill value.
fn masked_load() -> KernelFn {
    f32_kernel(
        "masked_load",
        &[F32],
        vec![Stmt::MaskedLoad {
            local: 0,
            place: buf0(),
            mask: u(1),
            fill: c(0.0),
        }],
    )
}

/// `MaskedStore`: a predicated write.
fn masked_store() -> KernelFn {
    f32_kernel(
        "masked_store",
        &[],
        vec![Stmt::MaskedStore {
            place: buf0(),
            value: c(1.0),
            mask: u(1),
        }],
    )
}

/// `If`: a branch.
fn if_() -> KernelFn {
    f32_kernel(
        "if",
        &[],
        vec![Stmt::If {
            cond: u(1),
            then: Block(vec![Stmt::Store {
                place: buf0(),
                value: c(1.0),
            }]),
            els: Block(vec![]),
        }],
    )
}

/// `For`: a counted loop.
fn for_() -> KernelFn {
    f32_kernel(
        "for",
        &[U32],
        vec![Stmt::For {
            local: 0,
            start: u(0),
            end: u(4),
            step: u(1),
            unroll: Unroll::None,
            body: Block(vec![Stmt::Store {
                place: buf_at(l(0)),
                value: c(1.0),
            }]),
        }],
    )
}

/// `While`: a conditional loop.
fn while_() -> KernelFn {
    f32_kernel(
        "while",
        &[],
        vec![Stmt::While {
            cond: u(0),
            body: Block(vec![Stmt::Store {
                place: buf0(),
                value: c(1.0),
            }]),
        }],
    )
}

/// `Break`: leave the innermost loop (the loop is the only other statement).
fn break_() -> KernelFn {
    f32_kernel(
        "break",
        &[],
        vec![Stmt::While {
            cond: u(1),
            body: Block(vec![Stmt::Break]),
        }],
    )
}

/// `Continue`: start the next pass of the innermost loop.
fn continue_() -> KernelFn {
    f32_kernel(
        "continue",
        &[U32],
        vec![Stmt::For {
            local: 0,
            start: u(0),
            end: u(2),
            step: u(1),
            unroll: Unroll::None,
            body: Block(vec![Stmt::Continue]),
        }],
    )
}

/// `Return`: leave the kernel.
fn return_() -> KernelFn {
    f32_kernel("return", &[], vec![Stmt::Return])
}

/// `Barrier`: a block barrier. Needs a launch.
fn barrier() -> KernelFn {
    f32_kernel("barrier", &[], vec![Stmt::barrier(Scope::Block)]).with_launch(block32())
}

/// `AtomicRmw`: an atomic add on a global buffer element.
fn atomic_rmw() -> KernelFn {
    f32_kernel(
        "atomic_rmw",
        &[],
        vec![Stmt::atomic_rmw(AtomicOp::Add, buf0(), c(1.0))],
    )
    .with_launch(block32())
}

/// `AtomicCas`: a compare-and-exchange on an integer buffer element.
fn atomic_cas() -> KernelFn {
    kernel(
        "atomic_cas",
        I32,
        Some((I32, BufferRole::ReadWrite)),
        &[I32],
        vec![Stmt::AtomicCas {
            place: buf0(),
            expected: ci(0),
            desired: ci(1),
            old: Some(0),
            order: MemoryOrder::Relaxed,
            scope: Scope::Device,
        }],
    )
    .with_launch(block32())
}

/// `MulticastLoadReduce`: a reducing read over every device's copy. Needs a launch of several devices.
fn multicast_load_reduce() -> KernelFn {
    f32_kernel(
        "multicast_load_reduce",
        &[F32],
        vec![Stmt::MulticastLoadReduce {
            local: 0,
            place: Place::Multicast {
                buffer: 0,
                coords: vec![u(0)],
            },
            op: AtomicOp::Add,
            order: MemoryOrder::Relaxed,
        }],
    )
    .with_launch(block32().with_devices(2))
}

/// `GridDepWait`: `griddepcontrol.wait`.
fn grid_dep_wait() -> KernelFn {
    f32_kernel("grid_dep_wait", &[], vec![Stmt::GridDepWait]).with_launch(block32())
}

/// `GridDepLaunchDependents`: `griddepcontrol.launch_dependents`.
fn grid_dep_launch_dependents() -> KernelFn {
    f32_kernel(
        "grid_dep_launch_dependents",
        &[],
        vec![Stmt::GridDepLaunchDependents],
    )
    .with_launch(block32())
}

/// `Fence`: a block-scope acquire-release fence.
fn fence() -> KernelFn {
    f32_kernel(
        "fence",
        &[],
        vec![Stmt::Fence {
            scope: Scope::Block,
            order: MemoryOrder::AcqRel,
        }],
    )
    .with_launch(block32())
}

/// `Collective`: a subgroup sum. Needs a warp launch.
fn collective() -> KernelFn {
    f32_kernel(
        "collective",
        &[F32, F32],
        vec![
            Stmt::Let {
                local: 0,
                init: c(1.0),
            },
            Stmt::Collective {
                kind: CollectiveKind::Reduce(ReduceOp::Sum),
                value: l(0),
                result: 1,
                scope: Scope::Subgroup,
            },
        ],
    )
    .with_launch(warp())
}

/// `CollectiveTuple`: a block reduce of a `(F32, F32)` state with a module function. Needs a module.
fn collective_tuple() -> Fixture {
    // `pair_sum(a0, a1, b0, b1) -> (a0 + b0, a1 + b1)`: locals 0..=3 the parameters, 4 and 5 the results.
    let combine = KernelFn::new(
        "pair_sum",
        F32,
        vec![
            Param::scalar("a0", F32, 0),
            Param::scalar("a1", F32, 1),
            Param::scalar("b0", F32, 2),
            Param::scalar("b1", F32, 3),
        ],
        locals(&[F32; 6]),
        Block(vec![
            Stmt::Let {
                local: 4,
                init: add(l(0), l(2)),
            },
            Stmt::Let {
                local: 5,
                init: add(l(1), l(3)),
            },
        ]),
    )
    .with_result(4)
    .with_results([5]);
    let entry = f32_kernel(
        "collective_tuple",
        &[F32; 4],
        vec![
            Stmt::Let {
                local: 0,
                init: c(1.0),
            },
            Stmt::Let {
                local: 1,
                init: c(2.0),
            },
            Stmt::CollectiveTuple {
                kind: CollectiveKind::ReduceWith { combine: 1 },
                values: vec![l(0), l(1)],
                results: vec![2, 3],
                scope: Scope::Block,
            },
        ],
    )
    .with_launch(block32());
    Fixture::Module {
        module: Module::new(vec![entry, combine]),
        entry: 0,
    }
}

/// `Mma`: an `m16n8k16` f16 -> f32 warp MMA on `Private` fragments. Needs a warp launch and the atom's fragments.
fn mma() -> KernelFn {
    let atom = MmaAtom::nvidia_m16n8k16_f16_f32();
    let len = |which| atom.values(which);
    f32_kernel(
        "mma",
        &[],
        vec![Stmt::Mma {
            atom: atom.clone(),
            a: 0,
            b: 1,
            c: 2,
            d: 3,
            extra: vec![],
        }],
    )
    .with_launch(warp())
    .with_allocs(vec![
        alloc("a", MemorySpace::Private, F16, len(0)),
        alloc("b", MemorySpace::Private, F16, len(1)),
        alloc("c", MemorySpace::Private, F32, len(2)),
        alloc("d", MemorySpace::Private, F32, len(2)),
    ])
}

/// `MatrixLoad`: `ldmatrix` of one 8 x 8 tile of `F16`. Needs a warp launch and a `Shared` tile.
fn matrix_load() -> KernelFn {
    f32_kernel(
        "matrix_load",
        &[],
        vec![Stmt::MatrixLoad {
            count: 1,
            trans: false,
            src: alloc_at(0, mul(tid(), u(8))),
            dst: 1,
        }],
    )
    .with_launch(warp())
    .with_allocs(vec![
        alloc("tile", MemorySpace::Shared, F16, 64),
        alloc("frag", MemorySpace::Private, F16, 2),
    ])
}

/// `MbarInit`: initialise an `mbarrier`. Needs a `Shared` 64-bit element.
fn mbar_init() -> KernelFn {
    f32_kernel(
        "mbar_init",
        &[],
        vec![Stmt::MbarInit {
            bar: bar(),
            count: u(1),
        }],
    )
    .with_launch(block32())
    .with_allocs(vec![mbar_alloc()])
}

/// `MbarArrive`: arrive at an `mbarrier`. Needs a `Shared` 64-bit element.
fn mbar_arrive() -> KernelFn {
    f32_kernel(
        "mbar_arrive",
        &[],
        vec![Stmt::MbarArrive {
            bar: bar(),
            tx: None,
        }],
    )
    .with_launch(block32())
    .with_allocs(vec![mbar_alloc()])
}

/// `MbarCompleteTx`: complete bytes of the transaction count. Needs a `Shared` 64-bit element.
fn mbar_complete_tx() -> KernelFn {
    f32_kernel(
        "mbar_complete_tx",
        &[],
        vec![Stmt::MbarCompleteTx {
            bar: bar(),
            bytes: u(4),
        }],
    )
    .with_launch(block32())
    .with_allocs(vec![mbar_alloc()])
}

/// `MbarWait`: wait for a phase of an `mbarrier`. Needs a `Shared` 64-bit element.
fn mbar_wait() -> KernelFn {
    f32_kernel(
        "mbar_wait",
        &[],
        vec![Stmt::MbarWait {
            bar: bar(),
            parity: u(0),
        }],
    )
    .with_launch(block32())
    .with_allocs(vec![mbar_alloc()])
}

/// `MatrixStore`: `stmatrix` of one 8 x 8 tile of `F16`. Needs a warp launch and a `Shared` tile.
fn matrix_store() -> KernelFn {
    f32_kernel(
        "matrix_store",
        &[],
        vec![Stmt::MatrixStore {
            count: 1,
            trans: false,
            dst: alloc_at(0, mul(tid(), u(8))),
            src: 1,
        }],
    )
    .with_launch(warp())
    .with_allocs(vec![
        alloc("tile", MemorySpace::Shared, F16, 64),
        alloc("frag", MemorySpace::Private, F16, 2),
    ])
}

/// A 2 x 2 `F32` opaque tile (ROADMAP T2).
fn tile(name: &str) -> AllocDecl {
    AllocDecl::tile(name, F32, 2, 2)
}

/// The tile statements' kernel: launch, a tile `0`, a `Shared` `F32` array `1` and the tiles `2..4` for an MMA.
fn tile_kernel(name: &str, body: Vec<Stmt>) -> KernelFn {
    f32_kernel(name, &[], body)
        .with_launch(warp())
        .with_allocs(vec![
            tile("t"),
            alloc("s", MemorySpace::Shared, F32, 16),
            tile("a"),
            tile("b"),
        ])
}

/// `TileLoad`: read a 2 x 2 tile from a `Shared` array with a row stride of 4. Needs a launch and a `Tile` allocation.
fn tile_load() -> KernelFn {
    tile_kernel(
        "tile_load",
        vec![Stmt::TileLoad {
            d: 0,
            src: alloc_at(1, u(0)),
            elements_per_row: u(4),
            origin: [u(0), u(0)],
        }],
    )
}

/// `TileMma`: `t = a * b + t` over 2 x 2 `F32` tiles.
fn tile_mma() -> KernelFn {
    tile_kernel(
        "tile_mma",
        vec![Stmt::TileMma {
            d: 0,
            a: 2,
            b: 3,
            c: 0,
        }],
    )
}

/// `TileFill`: set every element of a tile.
fn tile_fill() -> KernelFn {
    tile_kernel(
        "tile_fill",
        vec![Stmt::TileFill {
            d: 0,
            value: c(1.0),
        }],
    )
}

/// `TileStore`: write a 2 x 2 tile to a `Shared` array with a row stride of 4.
fn tile_store() -> KernelFn {
    tile_kernel(
        "tile_store",
        vec![Stmt::TileStore {
            a: 0,
            dst: alloc_at(1, u(0)),
            elements_per_row: u(4),
            origin: [u(0), u(0)],
        }],
    )
}

/// `ProxyFence`: `fence.proxy.async.shared::cta`.
fn proxy_fence() -> KernelFn {
    f32_kernel(
        "proxy_fence",
        &[],
        vec![Stmt::ProxyFence {
            space: MemorySpace::Shared,
        }],
    )
    .with_launch(block32())
}

/// `Wgmma`: an `m64n8k16` `Bf16` -> `F32` warpgroup MMA with row-major (oracle-only) operand layouts. Needs a
/// 128-thread launch, `Shared` tiles and a `Private` fragment.
fn wgmma() -> KernelFn {
    f32_kernel(
        "wgmma",
        &[],
        vec![Stmt::Wgmma {
            n: 8,
            ab: BF16,
            acc: F32,
            a: alloc_at(0, u(0)),
            b: alloc_at(1, u(0)),
            d: 2,
            scale_d: u(0),
            a_layout: OperandLayout::RowMajor,
            b_layout: OperandLayout::RowMajor,
            a_regs: None,
        }],
    )
    .with_launch(warpgroup())
    .with_allocs(vec![
        alloc("a", MemorySpace::Shared, BF16, 64 * 16),
        alloc("b", MemorySpace::Shared, BF16, 8 * 16),
        alloc("d", MemorySpace::Private, F32, 4),
    ])
}

/// `WgmmaCommit`: `wgmma.commit_group`. Needs a 128-thread launch.
fn wgmma_commit() -> KernelFn {
    f32_kernel("wgmma_commit", &[], vec![Stmt::WgmmaCommit]).with_launch(warpgroup())
}

/// `WgmmaWait`: `wgmma.wait_group`. Needs a 128-thread launch.
fn wgmma_wait() -> KernelFn {
    f32_kernel("wgmma_wait", &[], vec![Stmt::WgmmaWait { pending: 0 }]).with_launch(warpgroup())
}

/// `Tcgen05Mma`: an `n = 8` `tcgen05.mma` into tensor memory, issued by thread 0. Needs `Shared` tiles, a `Tensor`
/// allocation and row-major (oracle-only) operand layouts.
fn tcgen05_mma() -> KernelFn {
    f32_kernel(
        "tcgen05_mma",
        &[],
        vec![on_thread0(vec![Stmt::Tcgen05Mma {
            n: 8,
            ab: BF16,
            acc: F32,
            a: alloc_at(0, u(0)),
            b: alloc_at(1, u(0)),
            d: 2,
            d_col: u(0),
            scale_d: u(0),
            a_layout: OperandLayout::RowMajor,
            b_layout: OperandLayout::RowMajor,
            m: 128,
            d_lane: 0,
            b_ty: None,
            negate_a: false,
            negate_b: false,
        }])],
    )
    .with_launch(warpgroup())
    .with_allocs(vec![
        alloc("a", MemorySpace::Shared, BF16, 128 * 16),
        alloc("b", MemorySpace::Shared, BF16, 8 * 16),
        alloc("tmem", MemorySpace::Tensor, F32, 128 * 8),
    ])
}

/// `Tcgen05Commit`: `tcgen05.commit` onto an `mbarrier`. Needs a `Shared` 64-bit element.
fn tcgen05_commit() -> KernelFn {
    f32_kernel(
        "tcgen05_commit",
        &[],
        vec![on_thread0(vec![Stmt::Tcgen05Commit { bar: bar() }])],
    )
    .with_launch(warpgroup())
    .with_allocs(vec![mbar_alloc()])
}

/// `Tcgen05Ld`: `tcgen05.ld.32x32b` of 8 columns. Needs a `Tensor` allocation.
fn tcgen05_ld() -> KernelFn {
    f32_kernel(
        "tcgen05_ld",
        &[],
        vec![Stmt::Tcgen05Ld {
            src: 0,
            col: u(0),
            count: 8,
            dst: 1,
        }],
    )
    .with_launch(warpgroup())
    .with_allocs(vec![
        alloc("tmem", MemorySpace::Tensor, F32, 128 * 8),
        alloc("frag", MemorySpace::Private, F32, 8),
    ])
}

/// `CopyAsync`: a 16-byte `cp.async` from a buffer into `Shared`.
fn copy_async() -> KernelFn {
    kernel(
        "copy_async",
        F32,
        Some((F32, BufferRole::Read)),
        &[],
        vec![Stmt::CopyAsync {
            dst: alloc_at(0, u(0)),
            src: buf0(),
            elems: 4,
            valid: None,
            cache: CacheHint::Auto,
            l2_prefetch: None,
        }],
    )
    .with_launch(block32())
    .with_allocs(vec![alloc("dst", MemorySpace::Shared, F32, 4)])
}

/// `CopyBulk`: a 16-byte bulk copy from a buffer into `Shared`, completing on an `mbarrier`.
fn copy_bulk() -> KernelFn {
    kernel(
        "copy_bulk",
        F32,
        Some((F32, BufferRole::Read)),
        &[],
        vec![Stmt::CopyBulk {
            dst: alloc_at(0, u(0)),
            src: buf0(),
            elems: 4,
            bar: alloc_at(1, u(0)),
        }],
    )
    .with_launch(block32())
    .with_allocs(vec![
        alloc("dst", MemorySpace::Shared, F32, 4),
        alloc("bar", MemorySpace::Shared, U64, 1),
    ])
}

/// `Tcgen05Alloc`: `tcgen05.alloc.cta_group::1` of 32 columns. Needs a `Tensor` allocation of 128 lanes x 32 columns.
fn tcgen05_alloc() -> KernelFn {
    f32_kernel(
        "tcgen05_alloc",
        &[],
        vec![on_thread0(vec![Stmt::Tcgen05Alloc { d: 0, ncols: 32 }])],
    )
    .with_launch(warpgroup())
    .with_allocs(vec![alloc("tmem", MemorySpace::Tensor, F32, 128 * 32)])
}

/// `Tcgen05Dealloc`: `tcgen05.dealloc.cta_group::1` of 32 columns. Needs a `Tensor` allocation.
fn tcgen05_dealloc() -> KernelFn {
    f32_kernel(
        "tcgen05_dealloc",
        &[],
        vec![on_thread0(vec![Stmt::Tcgen05Dealloc { d: 0, ncols: 32 }])],
    )
    .with_launch(warpgroup())
    .with_allocs(vec![alloc("tmem", MemorySpace::Tensor, F32, 128 * 32)])
}

/// `Tcgen05RelinquishPermit`: `tcgen05.relinquish_alloc_permit.cta_group::1`.
fn tcgen05_relinquish_permit() -> KernelFn {
    f32_kernel(
        "tcgen05_relinquish_permit",
        &[],
        vec![on_thread0(vec![Stmt::Tcgen05RelinquishPermit])],
    )
    .with_launch(warpgroup())
}

/// `CopyUnpack`: one group of 16 `F4E2M1` elements from a global buffer into a 16-byte `U8` `Shared` slot, completing on
/// an `mbarrier`. Needs a `Shared` byte tile and a `Shared` 64-bit element.
fn copy_unpack() -> KernelFn {
    kernel(
        "copy_unpack",
        F32,
        Some((ElementKind::F4E2M1, BufferRole::Read)),
        &[],
        vec![Stmt::CopyUnpack {
            dst: alloc_at(0, u(0)),
            src: buf0(),
            groups: 1,
            bar: alloc_at(1, u(0)),
        }],
    )
    .with_launch(block32())
    .with_allocs(vec![
        alloc("dst", MemorySpace::Shared, U8, 16),
        alloc("bar", MemorySpace::Shared, U64, 1),
    ])
}

/// `AsyncCommit`: `cp.async.commit_group`.
fn async_commit() -> KernelFn {
    f32_kernel("async_commit", &[], vec![Stmt::AsyncCommit]).with_launch(block32())
}

/// `AsyncWait`: `cp.async.wait_group`.
fn async_wait() -> KernelFn {
    f32_kernel("async_wait", &[], vec![Stmt::AsyncWait { pending: 0 }]).with_launch(block32())
}

/// `id(x) -> x`: a callee of one `F32` parameter and one result (locals 0 and 1).
fn identity_callee() -> KernelFn {
    KernelFn::new(
        "id",
        F32,
        vec![Param::scalar("x", F32, 0)],
        locals(&[F32, F32]),
        Block(vec![Stmt::Let {
            local: 1,
            init: l(0),
        }]),
    )
    .with_result(1)
}

/// `Call`: call a module function. Needs a module.
fn call() -> Fixture {
    let entry = f32_kernel(
        "call",
        &[F32],
        vec![Stmt::Call {
            callee: 1,
            args: vec![CallArg::Scalar(c(1.0))],
            result: Some(0),
        }],
    );
    Fixture::Module {
        module: Module::new(vec![entry, identity_callee()]),
        entry: 0,
    }
}

/// `CallTuple`: call a module function that returns two values. Needs a module.
fn call_tuple() -> Fixture {
    // `swap(x, y) -> (y, x)`: locals 0 and 1 the parameters, 2 and 3 the results.
    let callee = KernelFn::new(
        "swap",
        F32,
        vec![Param::scalar("x", F32, 0), Param::scalar("y", F32, 1)],
        locals(&[F32; 4]),
        Block(vec![
            Stmt::Let {
                local: 2,
                init: l(1),
            },
            Stmt::Let {
                local: 3,
                init: l(0),
            },
        ]),
    )
    .with_result(2)
    .with_results([3]);
    let entry = f32_kernel(
        "call_tuple",
        &[F32, F32],
        vec![Stmt::CallTuple {
            callee: 1,
            args: vec![CallArg::Scalar(c(1.0)), CallArg::Scalar(c(2.0))],
            results: vec![0, 1],
        }],
    );
    Fixture::Module {
        module: Module::new(vec![entry, callee]),
        entry: 0,
    }
}

/// `ExternCall`: call a pure opaque declaration. Needs a module (the declaration lives in it).
fn extern_call() -> Fixture {
    let decl =
        ExternDecl::new("inc", vec![U32], vec![U32], Effects::pure()).with_body(TargetBody::new(
            "cuda-ptx",
            BodyText::Asm {
                template: "add.u32 {1}, {0}, 1;".into(),
                constraints: "=r,r".into(),
            },
        ));
    let entry = kernel(
        "extern_call",
        U32,
        Some((U32, BufferRole::ReadWrite)),
        &[U32],
        vec![Stmt::ExternCall {
            decl: 0,
            args: vec![u(1)],
            results: vec![0],
        }],
    );
    Fixture::Module {
        module: Module::new(vec![entry]).with_externs(vec![decl]),
        entry: 0,
    }
}

// ---- the public listing -----------------------------------------------------------------------------------------

/// One fixture per `Stmt` variant, as `(variant name, fixture)`, in the order of the enum.
#[must_use]
pub fn fixtures() -> Vec<(&'static str, Fixture)> {
    let k = Fixture::Kernel;
    vec![
        ("Let", k(let_())),
        ("Load", k(load())),
        ("Assign", k(assign())),
        ("Store", k(store())),
        ("MaskedLoad", k(masked_load())),
        ("MaskedStore", k(masked_store())),
        ("If", k(if_())),
        ("For", k(for_())),
        ("While", k(while_())),
        ("Break", k(break_())),
        ("Continue", k(continue_())),
        ("Return", k(return_())),
        ("Barrier", k(barrier())),
        ("AtomicRmw", k(atomic_rmw())),
        ("AtomicCas", k(atomic_cas())),
        ("MulticastLoadReduce", k(multicast_load_reduce())),
        ("GridDepWait", k(grid_dep_wait())),
        ("GridDepLaunchDependents", k(grid_dep_launch_dependents())),
        ("Fence", k(fence())),
        ("Collective", k(collective())),
        ("CollectiveTuple", collective_tuple()),
        ("Mma", k(mma())),
        ("MatrixLoad", k(matrix_load())),
        ("MbarInit", k(mbar_init())),
        ("MbarArrive", k(mbar_arrive())),
        ("MbarCompleteTx", k(mbar_complete_tx())),
        ("MbarWait", k(mbar_wait())),
        ("MatrixStore", k(matrix_store())),
        ("TileLoad", k(tile_load())),
        ("TileMma", k(tile_mma())),
        ("TileFill", k(tile_fill())),
        ("TileStore", k(tile_store())),
        ("ProxyFence", k(proxy_fence())),
        ("Wgmma", k(wgmma())),
        ("WgmmaCommit", k(wgmma_commit())),
        ("WgmmaWait", k(wgmma_wait())),
        ("Tcgen05Mma", k(tcgen05_mma())),
        ("Tcgen05Commit", k(tcgen05_commit())),
        ("Tcgen05Alloc", k(tcgen05_alloc())),
        ("Tcgen05Dealloc", k(tcgen05_dealloc())),
        ("Tcgen05RelinquishPermit", k(tcgen05_relinquish_permit())),
        ("Tcgen05Ld", k(tcgen05_ld())),
        ("CopyAsync", k(copy_async())),
        ("CopyBulk", k(copy_bulk())),
        ("CopyUnpack", k(copy_unpack())),
        ("AsyncCommit", k(async_commit())),
        ("AsyncWait", k(async_wait())),
        ("Call", call()),
        ("CallTuple", call_tuple()),
        ("ExternCall", extern_call()),
    ]
}

/// The fixture for the `Stmt` variant `name`, or `None` for a name that is not a variant.
#[must_use]
pub fn fixture(name: &str) -> Option<Fixture> {
    fixtures()
        .into_iter()
        .find(|(n, _)| *n == name)
        .map(|(_, f)| f)
}

/// Kernels `check_kernel` **refuses**, each with the error it gives: a `Subgroup`-scope collective in control that may differ
/// between the lanes of one subgroup (a decline test for a *checker* rule, not an emitter's; ROADMAP U63, board 506). The oracle
/// runs a collective over every lane of its subgroup, a device over the lanes still active, so a kernel whose lanes may not all
/// arrive has no single meaning. `"CollectiveUnderDivergentIf"` puts it under `if (threadIdx.x & 1)` (`BarrierDivergent`);
/// `"CollectiveAfterDivergentReturn"` puts a `Return` under that condition before it (`CollectiveAfterReturn`). The uniform
/// control of both is the `Collective` fixture of [`fixtures`].
#[must_use]
pub fn divergent_fixtures() -> Vec<(&'static str, KernelFn, KernelError)> {
    let odd = || ScalarExpr::Binary(BinaryOp::BitAnd, Box::new(tid()), Box::new(u(1)));
    let sum = || Stmt::Collective {
        kind: CollectiveKind::Reduce(ReduceOp::Sum),
        value: l(0),
        result: 1,
        scope: Scope::Subgroup,
    };
    let set = || Stmt::Let {
        local: 0,
        init: c(1.0),
    };
    let when_odd = |then: Vec<Stmt>| Stmt::If {
        cond: odd(),
        then: Block(then),
        els: Block(vec![]),
    };
    let build =
        |name: &str, body: Vec<Stmt>| f32_kernel(name, &[F32, F32], body).with_launch(warp());
    vec![
        (
            "CollectiveUnderDivergentIf",
            build(
                "collective_under_divergent_if",
                vec![set(), when_odd(vec![sum()])],
            ),
            KernelError::BarrierDivergent,
        ),
        (
            "CollectiveAfterDivergentReturn",
            build(
                "collective_after_divergent_return",
                vec![set(), when_odd(vec![Stmt::Return]), sum()],
            ),
            KernelError::CollectiveAfterReturn,
        ),
    ]
}

/// What the fixture for `name` needs that is unusual, in a few words, or `""` for a plain single-thread kernel:
/// `"module"` (a `check_module` fixture), `"launch"`, `"warp launch"`, `"warpgroup launch"`, `"multi-device launch"`,
/// `"tensor memory"`, `"operand layouts"` (a tile layout, here the oracle-only row-major one), `"mbarrier"`
/// (a `Shared` 64-bit element), `"shared tile"`. Several are joined with a comma.
#[must_use]
pub fn needs(name: &str) -> &'static str {
    match name {
        "Barrier"
        | "AtomicRmw"
        | "AtomicCas"
        | "GridDepWait"
        | "GridDepLaunchDependents"
        | "Fence"
        | "ProxyFence"
        | "AsyncCommit"
        | "AsyncWait" => "launch",
        "MulticastLoadReduce" => "multi-device launch",
        "Collective" => "warp launch",
        "Mma" => "warp launch, register fragments",
        "MatrixLoad" | "MatrixStore" => "warp launch, shared tile",
        "TileLoad" | "TileMma" | "TileFill" | "TileStore" => "launch, tile allocations",
        "MbarInit" | "MbarArrive" | "MbarCompleteTx" | "MbarWait" => "launch, mbarrier",
        "CopyAsync" => "launch, shared tile",
        "CopyBulk" | "CopyUnpack" => "launch, shared tile, mbarrier",
        "Wgmma" => "warpgroup launch, shared tiles, operand layouts",
        "WgmmaCommit" | "WgmmaWait" => "warpgroup launch",
        "Tcgen05Mma" => "warpgroup launch, shared tiles, tensor memory, operand layouts",
        "Tcgen05Commit" => "warpgroup launch, mbarrier",
        "Tcgen05Ld" => "warpgroup launch, tensor memory",
        "Tcgen05Alloc" | "Tcgen05Dealloc" => "warpgroup launch, tensor memory",
        "Tcgen05RelinquishPermit" => "warpgroup launch",
        "CollectiveTuple" | "Call" | "CallTuple" | "ExternCall" => "module",
        _ => "",
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernel::BarrierKind;

    /// Every statement of `body`, recursing into `If`, `For` and `While`.
    fn walk<'a>(body: &'a [Stmt], out: &mut Vec<&'a Stmt>) {
        for st in body {
            out.push(st);
            match st {
                Stmt::If { then, els, .. } => {
                    walk(&then.0, out);
                    walk(&els.0, out);
                }
                Stmt::For { body, .. } | Stmt::While { body, .. } => walk(&body.0, out),
                _ => {}
            }
        }
    }

    #[test]
    fn every_variant_has_exactly_one_fixture_and_no_fixture_is_extra() {
        let all = fixtures();
        for name in ALL {
            let n = all.iter().filter(|(f, _)| f == name).count();
            assert_eq!(n, 1, "variant {name} has {n} fixtures, want exactly 1");
        }
        for (name, _) in &all {
            assert!(ALL.contains(name), "fixture {name} is not a Stmt variant");
        }
        assert_eq!(all.len(), ALL.len());
        // the order is the enum's
        let names: Vec<&str> = all.iter().map(|(n, _)| *n).collect();
        assert_eq!(names, ALL);
        // the positive control: the enum has 50 variants on this base
        assert_eq!(ALL.len(), 50, "the Stmt variants this list was written for");
    }

    #[test]
    fn every_divergent_fixture_is_refused_with_its_error_and_its_uniform_control_is_admitted() {
        let divergent = divergent_fixtures();
        assert_eq!(divergent.len(), 2);
        for (name, k, err) in divergent {
            assert_eq!(crate::kernel::check_kernel(&k), Err(err), "{name}");
        }
        // the control: the uniform form of the same collective passes
        let Some(Fixture::Kernel(k)) = fixture("Collective") else {
            panic!("the Collective fixture is a kernel");
        };
        assert_eq!(crate::kernel::check_kernel(&k), Ok(()));
    }

    #[test]
    fn fixture_names_are_unique() {
        let mut names: Vec<&str> = fixtures().iter().map(|(n, _)| *n).collect();
        names.sort_unstable();
        let before = names.len();
        names.dedup();
        assert_eq!(names.len(), before, "a fixture name is listed twice");
        let mut all = ALL.to_vec();
        all.sort_unstable();
        all.dedup();
        assert_eq!(all.len(), ALL.len(), "a variant is listed twice in ALL");
    }

    #[test]
    fn every_fixture_passes_its_checker() {
        let mut failed = Vec::new();
        for (name, f) in fixtures() {
            if let Err(e) = f.check() {
                eprintln!("FIXTURE {name} FAILS: {e:?}");
                failed.push(name);
            }
        }
        assert!(
            failed.is_empty(),
            "fixtures the checker refuses: {failed:?}"
        );
    }

    #[test]
    fn every_fixture_really_contains_its_variant() {
        for (name, f) in fixtures() {
            let mut stmts = Vec::new();
            for func in f.functions() {
                walk(&func.body.0, &mut stmts);
            }
            assert!(
                stmts.iter().any(|st| variant_name(st) == name),
                "the fixture for {name} contains no {name} statement (it has {:?})",
                stmts.iter().map(|s| variant_name(s)).collect::<Vec<_>>()
            );
        }
    }

    #[test]
    fn lookup_and_needs_agree_with_the_list() {
        for (name, f) in fixtures() {
            assert_eq!(fixture(name), Some(f.clone()), "{name}");
            let module = matches!(f, Fixture::Module { .. });
            assert_eq!(
                needs(name).contains("module"),
                module,
                "{name}: needs() and the fixture disagree on a module"
            );
        }
        assert_eq!(fixture("NotAVariant"), None);
    }

    #[test]
    fn barrier_kind_is_full_in_the_barrier_fixture() {
        // the one place a fixture pins a field the checker could vary
        let Some(Fixture::Kernel(k)) = fixture("Barrier") else {
            panic!("a kernel fixture")
        };
        assert!(matches!(
            k.body.0[0],
            Stmt::Barrier {
                kind: BarrierKind::Full,
                ..
            }
        ));
    }
}
