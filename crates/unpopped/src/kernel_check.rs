// SPDX-License-Identifier: MIT OR Apache-2.0
//! The scope and type checker for [`crate::kernel::KernelFn`] ([`check_kernel`]), split from
//! `kernel.rs` so neither file outgrows the size the analysers accept.

mod async_copy;
mod collective;
mod extern_call;
mod matrix;
mod mbar;
mod mma;
mod tcgen05;
mod tile;
mod wgmma;

use unpopped_vocab::ElementKind;

use crate::exec::{Extent, GridDim, MemorySpace, Scope as ExecScope};
use crate::ir::ScalarExpr;
use crate::kernel::{
    AtomicOp, BarrierKind, Block, BufferRole, CallArg, KernelError, KernelFn, MemoryOrder, Param,
    ParamKind, Place, SpecValue, Stmt,
};
use crate::kernel_extern::ExternSig;
use crate::types::{self, Ty};

/// Per-walk state: which locals are in scope, which are protected from `Assign`.
struct Scope {
    in_scope: Vec<bool>,
    ever_declared: Vec<bool>,
    param_local: Vec<bool>,
    counter_local: Vec<bool>,
    tys: Vec<Ty>,
    loop_depth: u32,
    /// The signatures of the module's functions; empty outside a module (a `Call` is then refused).
    callees: Vec<CalleeSig>,
    /// The signatures of the module's extern declarations.
    externs: Vec<ExternSig>,
}

/// What a `Call` is checked against: a callee's parameters and the type of its result local.
#[derive(Clone)]
pub(crate) struct CalleeSig {
    pub(crate) params: Vec<ParamKind>,
    pub(crate) result: Option<Ty>,
    /// The types of the values after `result` (`KernelFn::results`), for a `CallTuple`.
    pub(crate) extra: Vec<Ty>,
    /// The function reaches a `Barrier`, directly or through its own calls: a call of it is a barrier site.
    pub(crate) barrier: bool,
}

/// Check that `k` is well formed (`docs/ir-kernel-layer.md` section 4).
///
/// # Errors
///
/// The first [`KernelError`], walking statements in program order and each statement's
/// expressions before its effects.
pub fn check_kernel(k: &KernelFn) -> Result<(), KernelError> {
    check_kernel_with(k, Vec::new(), Vec::new(), true)
}

/// [`check_kernel`] with the signatures of the module the function lives in, so a `Call` resolves.
///
/// `entry` is whether nothing in the module calls `k`: only then are its buffer parameters bound by the launcher (distinct
/// storage, `docs/ir-kernel-layer.md`), so only then may the barrier-uniformity check call a load from a `Read` buffer uniform.
/// A function reached through a `Call` may receive the same buffer under two parameters (`CallArg::Buffer` aliases).
pub(crate) fn check_kernel_with(
    k: &KernelFn,
    callees: Vec<CalleeSig>,
    externs: Vec<ExternSig>,
    entry: bool,
) -> Result<(), KernelError> {
    check_launch(k)?;
    if let Some(s) = &k.schedule {
        crate::schedule::check_schedule(s).map_err(KernelError::Schedule)?;
    }
    check_declared_dtypes(k)?;
    let n = k.locals.len();
    let mut sc = Scope {
        in_scope: vec![false; n],
        ever_declared: vec![false; n],
        param_local: vec![false; n],
        counter_local: vec![false; n],
        tys: k.locals.iter().map(|d| d.ty()).collect(),
        loop_depth: 0,
        callees,
        externs,
    };
    for p in &k.params {
        match &p.kind {
            ParamKind::Scalar { ty, local } => declare_param_local(k, &mut sc, *ty, *local)?,
            ParamKind::Struct { fields } => {
                for f in fields {
                    declare_param_local(k, &mut sc, f.ty, f.local)?;
                }
            }
            ParamKind::Buffer { .. } => {}
        }
    }
    check_allocs(k)?;
    check_spec_consts(k)?;
    check_block(k, &k.body, &mut sc)?;
    let barrier_callees: Vec<bool> = sc.callees.iter().map(|c| c.barrier).collect();
    crate::kernel_uniform::check_barriers(k, &barrier_callees, &sc.externs, entry)
}

/// A kernel declares no dtype of union batch 1 (`types::is_union_batch1`): they are storage and key identity only until the casts,
/// the wide integer and the software binary128 land, so one is declined by name here and the oracle never meets a value it cannot
/// hold. Covers the compute type, every parameter (buffer element, scalar, struct field), every local and every allocation.
fn check_declared_dtypes(k: &KernelFn) -> Result<(), KernelError> {
    let refuse = |dtype| Err(KernelError::DtypeNotComputed { dtype });
    // the four wide integers are computed (ROADMAP E2) and so is `F128` (ROADMAP E3: the soft binary128 of `oracle/kernel_f128.rs`,
    // which runs every node `types::check_f128_node` admits and nothing else); the rest of union batch 1 is declined by name (the two
    // 6-bit floats are held in an 8-bit container by the oracle since f6-oracle-storage and are no longer declined)
    let declined = |t: ElementKind| {
        types::is_union_batch1(t) && !types::is_wide_int(t) && t != ElementKind::F128
    };
    if declined(k.compute) {
        return refuse(k.compute);
    }
    for p in &k.params {
        let tys: Vec<ElementKind> = match &p.kind {
            ParamKind::Buffer { elem, .. } => vec![*elem],
            ParamKind::Scalar { ty, .. } => vec![*ty],
            ParamKind::Struct { fields } => fields.iter().map(|f| f.ty).collect(),
        };
        if let Some(t) = tys.into_iter().find(|t| declined(*t)) {
            return refuse(t);
        }
    }
    if let Some(l) = k.locals.iter().find(|l| declined(l.elem)) {
        return refuse(l.elem);
    }
    if let Some(a) = k.allocs.iter().find(|a| declined(a.elem)) {
        return refuse(a.elem);
    }
    Ok(())
}

/// A launch has no zero extent, a subgroup range is non-empty and starts at 1 or more, and clusters (if any) tile the grid.
fn check_launch(k: &KernelFn) -> Result<(), KernelError> {
    let Some(l) = &k.launch else { return Ok(()) };
    let sg_ok = l.subgroup.is_none_or(|s| s.min >= 1 && s.min <= s.max);
    // clusters tile the grid: every cluster extent is positive and divides the grid's
    if l.devices == 0 {
        return Err(KernelError::BadLaunch);
    }
    let cluster_ok = l
        .cluster
        .is_none_or(|c| (0..3).all(|a| c[a] >= 1 && l.grid[a] % c[a] == 0));
    if l.block.contains(&0) || l.grid.contains(&0) || !sg_ok || !cluster_ok {
        return Err(KernelError::BadLaunch);
    }
    if let Some(dims) = &l.grid_dims {
        // a grid read from scalar arguments: each extent names an integer scalar parameter, a divisor is positive, and a
        // cluster cannot be checked against a grid that is not known until launch
        let ok = l.cluster.is_none()
            && dims.iter().all(|d| {
                let by_ok = match *d {
                    GridDim::Static(n) => n >= 1,
                    GridDim::CeilDiv { by, .. } => by >= 1,
                    _ => false,
                };
                // a scalar parameter or an integer field of a struct parameter (ROADMAP U23)
                let param_ok = d.param().is_none_or(|p| {
                    k.argument_of_local(p)
                        .is_some_and(|(_, _, ty)| types::int_range(ty).is_some())
                });
                (by_ok || matches!(d, GridDim::Param(_))) && param_ok
            });
        if !ok {
            return Err(KernelError::BadLaunch);
        }
    }
    Ok(())
}

/// Bind scalar parameter local `local` (of type `ty`) in `sc`: it must exist, be declared once and carry that type.
fn declare_param_local(
    k: &KernelFn,
    sc: &mut Scope,
    ty: ElementKind,
    local: u32,
) -> Result<(), KernelError> {
    let i = local as usize;
    let Some(decl) = k.locals.get(i) else {
        return Err(KernelError::UnknownLocal { local });
    };
    if sc.in_scope[i] {
        return Err(KernelError::Redeclared { local });
    }
    if decl.ty() != Ty::Scalar(ty) {
        return Err(KernelError::TypeMismatch {
            what: "scalar parameter and its local",
        });
    }
    sc.in_scope[i] = true;
    sc.ever_declared[i] = true;
    sc.param_local[i] = true;
    Ok(())
}

/// Every allocation is in a declarable space and has a usable extent.
fn check_allocs(k: &KernelFn) -> Result<(), KernelError> {
    for (i, a) in k.allocs.iter().enumerate() {
        let alloc = i as u32;
        if a.space == MemorySpace::Global {
            return Err(KernelError::AllocSpace { alloc });
        }
        let ok = match a.extent {
            Extent::Static(n) => n > 0,
            Extent::Param(local) => k
                .argument_of_local(local)
                .is_some_and(|(_, _, ty)| types::int_range(ty).is_some()),
        };
        if !ok {
            return Err(KernelError::AllocExtent { alloc });
        }
        check_alloc_layout(k, a, alloc)?;
        tile::check_tile_alloc(a, alloc)?;
    }
    Ok(())
}

/// The largest layout the checker proves injective by enumerating it (4 Mi elements, far above any shared array).
const LAYOUT_ENUMERATION_CAP: i64 = 1 << 22;

/// 3 and 4. range and injective, by one enumeration: every offset lies below the extent and no two logical indices share an
/// offset (the swizzle is a bijection, so injectivity is the layout's alone).
pub(crate) fn check_layout_offsets(
    layout: &unpopped_vocab::layout_algebra::Layout,
    extent: u32,
) -> Result<(), &'static str> {
    if layout.size() > LAYOUT_ENUMERATION_CAP {
        return Err("the layout is too large to prove injective");
    }
    let mut seen = vec![false; extent as usize];
    for i in 0..layout.size() {
        let Ok(o) = layout.call(i) else {
            return Err("the layout is not defined over its own size");
        };
        match usize::try_from(o).ok().and_then(|o| seen.get_mut(o)) {
            Some(slot) if !*slot => *slot = true,
            Some(_) => return Err("the layout is not injective: two indices share an offset"),
            None => return Err("the layout's cosize exceeds the allocation's extent"),
        }
    }
    Ok(())
}

/// The rules of `docs/design-swizzled-allocation.md` section 3 for one allocation's `align_bytes` and `layout`, each with its
/// accepted control in `tests/kernel_alloc_layout.rs`.
fn check_alloc_layout(
    k: &KernelFn,
    a: &crate::exec::AllocDecl,
    alloc: u32,
) -> Result<(), KernelError> {
    let bad = |why: &'static str| Err(KernelError::AllocLayout { alloc, why });
    if let Some(al) = a.align_bytes {
        if al == 0 || !al.is_power_of_two() {
            return bad("align_bytes is a power of two");
        }
    }
    if let Some(d) = &a.dyn_layout {
        return check_dyn_layout(k, a, d, alloc);
    }
    let Some(l) = &a.layout else { return Ok(()) };
    // 1. space: a layout (and a swizzle) belongs to a `Shared` array; the other spaces keep the identity until a row asks
    if !matches!(a.space, MemorySpace::Shared | MemorySpace::Private) {
        return bad(
            "a layout is only on a Shared or Private allocation (a Global array is a buffer's own shape and stride)",
        );
    }
    if a.space == MemorySpace::Private && l.swizzle.is_some() {
        return bad("a swizzle is a shared-memory bank pattern: a Private array has no banks");
    }
    // 2. size: a Static extent equal to the layout's size (a Param extent is not known here)
    let Extent::Static(n) = a.extent else {
        return bad("a layout needs a Static extent (a dynamic extent is not known at check time)");
    };
    if i64::from(n) != l.layout.size() {
        return bad("the layout's size must equal the allocation's extent");
    }
    if let Err(why) = check_layout_offsets(&l.layout, n) {
        return bad(why);
    }
    let Some(sw) = l.swizzle else { return Ok(()) };
    // swizzle rules: the element width must not be above the swizzle base, the period must divide the extent, and the base
    // address must be a multiple of the period (the pattern is anchored at address zero)
    let width = u32::try_from(types::elem_size(a.elem)).unwrap_or(0);
    let Ok(in_elems) = sw.in_elements(width) else {
        return bad(
            "the swizzle base is below the element width (or the width is not a power of two)",
        );
    };
    let period_elems =
        1u64 << (in_elems.base() + in_elems.shift().unsigned_abs() + in_elems.bits());
    if u64::from(n) % period_elems != 0 {
        return bad(
            "the swizzle period does not divide the extent, so a swizzled offset could leave it",
        );
    }
    let period_bytes = 1u64 << (sw.base() + sw.shift().unsigned_abs() + sw.bits());
    match a.align_bytes {
        Some(al) if u64::from(al) % period_bytes == 0 => Ok(()),
        _ => bad(
            "a swizzled allocation states an align_bytes that is a multiple of the swizzle period",
        ),
    }
}

/// The static half of the rules of `docs/design-dynamic-layout.md` section 3 for a `dyn_layout`: the launch binds the arguments and
/// runs the enumerating checker ([`check_layout_offsets`]) on the bound layout, so nothing here needs a value. Every parameter leaf
/// names an integer scalar parameter, and the allocation carries one layout, on a `Shared` space (a swizzle is not offered: its
/// period must divide an extent that is a value here, so it is refused until a row needs it).
fn check_dyn_layout(
    k: &KernelFn,
    a: &crate::exec::AllocDecl,
    d: &unpopped_vocab::layout_algebra::DynLayout,
    alloc: u32,
) -> Result<(), KernelError> {
    let bad = |why: &'static str| Err(KernelError::AllocLayout { alloc, why });
    if a.layout.is_some() {
        return bad("an allocation carries a layout or a dyn_layout, not both");
    }
    if !matches!(a.space, MemorySpace::Shared | MemorySpace::Private) {
        return bad(
            "a dyn_layout is only on a Shared or Private allocation (a Global array is a buffer's own shape and stride)",
        );
    }
    if a.space == MemorySpace::Private && d.swizzle().is_some() {
        return bad("a swizzle is a shared-memory bank pattern: a Private array has no banks");
    }
    for local in d.params() {
        let integer = k
            .argument_of_local(local)
            .is_some_and(|(_, _, ty)| types::int_range(ty).is_some());
        if !integer {
            return bad("a dyn_layout parameter names no integer scalar parameter of the kernel");
        }
    }
    // a byte swizzle (section 8): the rules that need no value are checked here, the period against the extent at the launch
    if let Some(sw) = d.swizzle() {
        let width = u32::try_from(types::elem_size(a.elem)).unwrap_or(0);
        if sw.in_elements(width).is_err() {
            return bad(
                "the swizzle base is below the element width (or the width is not a power of two)",
            );
        }
        let period_bytes = 1u64 << (sw.base() + sw.shift().unsigned_abs() + sw.bits());
        match a.align_bytes {
            Some(al) if u64::from(al) % period_bytes == 0 => {}
            _ => {
                return bad(
                    "a swizzled allocation states an align_bytes that is a multiple of the swizzle period",
                );
            }
        }
    }
    Ok(())
}

/// A specialization constant names a distinct scalar parameter and carries a default of that parameter's kind (an integer in
/// the type's range for an integer type, a float for a floating one).
fn check_spec_consts(k: &KernelFn) -> Result<(), KernelError> {
    for (i, c) in k.spec_consts.iter().enumerate() {
        let index = i as u32;
        let ty = match k.params.get(c.param as usize).map(|p| &p.kind) {
            Some(ParamKind::Scalar { ty, .. }) => *ty,
            _ => return Err(KernelError::BadSpecConst { index }),
        };
        if k.spec_consts[..i].iter().any(|d| d.param == c.param) {
            return Err(KernelError::BadSpecConst { index });
        }
        let ok = match (c.default, types::int_range(ty)) {
            (SpecValue::Int(v), Some((lo, hi))) => (lo..=hi).contains(&v),
            (SpecValue::Float(_), None) => true,
            _ => false,
        };
        if !ok {
            return Err(KernelError::BadSpecConst { index });
        }
    }
    Ok(())
}

fn check_block(k: &KernelFn, b: &Block, sc: &mut Scope) -> Result<(), KernelError> {
    let mut declared_here: Vec<usize> = Vec::new();
    let mut terminated = false;
    for st in &b.0 {
        if terminated {
            return Err(KernelError::Unreachable);
        }
        terminated = check_stmt(k, st, sc, &mut declared_here)?;
    }
    for i in declared_here {
        sc.in_scope[i] = false;
    }
    Ok(())
}

/// An `AtomicRmw` or an `AtomicCas`.
fn check_atomic_stmt(
    k: &KernelFn,
    st: &Stmt,
    sc: &mut Scope,
    declared_here: &mut Vec<usize>,
) -> Result<(), KernelError> {
    match st {
        Stmt::AtomicRmw {
            op,
            place,
            value,
            old,
            order,
            scope,
        } => {
            if matches!(place, Place::Multicast { .. }) {
                let red = MulticastRed {
                    op: *op,
                    old: *old,
                    order: *order,
                    scope: *scope,
                };
                return check_multicast_red(k, place, value, red, sc);
            }
            let form = AtomicForm {
                op: *op,
                site: AtomicSite {
                    old: *old,
                    scope: *scope,
                },
            };
            check_atomic(k, place, value, form, sc, declared_here)?;
        }
        Stmt::AtomicCas {
            place,
            expected,
            desired,
            old,
            scope,
            ..
        } => {
            if matches!(place, Place::Multicast { .. }) {
                return Err(KernelError::MulticastUnsupported {
                    what: "an AtomicCas on a multicast place (multimem has no compare-and-swap)",
                });
            }
            let site = AtomicSite {
                old: *old,
                scope: *scope,
            };
            check_cas(k, place, [expected, desired], site, sc, declared_here)?;
        }
        _ => unreachable!("check_atomic_stmt is called for atomic statements only"),
    }
    Ok(())
}

/// The statements that synchronize or call: atomics, fences, calls and barriers.
fn check_sync_stmt(
    k: &KernelFn,
    st: &Stmt,
    sc: &mut Scope,
    declared_here: &mut Vec<usize>,
) -> Result<(), KernelError> {
    match st {
        Stmt::AtomicRmw { .. } | Stmt::AtomicCas { .. } => {
            check_atomic_stmt(k, st, sc, declared_here)?;
        }
        Stmt::Fence { scope, order } => {
            if matches!(order, MemoryOrder::Relaxed) {
                return Err(KernelError::SyncUnsupported {
                    what: "a Relaxed Fence (it would order nothing; use Acquire, Release, AcqRel or SeqCst)",
                });
            }
            if !scope_ok(k, *scope) {
                return Err(KernelError::SyncUnsupported {
                    what: "a Fence scope other than Subgroup, Block, Device or (with launch cluster dims) Cluster",
                });
            }
        }
        Stmt::ProxyFence { space } => {
            if *space != MemorySpace::Shared {
                return Err(KernelError::SyncUnsupported {
                    what: "a ProxyFence over a space other than Shared (the .global and .shared::cluster forms are not modelled)",
                });
            }
        }
        Stmt::Collective { .. } => collective::check_collective(k, st, sc, declared_here)?,
        Stmt::CollectiveTuple { .. } => {
            collective::check_collective_tuple(k, st, sc, declared_here)?;
        }
        Stmt::Mma { .. } => mma::check_mma(k, st)?,
        Stmt::TileLoad { .. }
        | Stmt::TileMma { .. }
        | Stmt::TileFill { .. }
        | Stmt::TileStore { .. } => tile::check_tile(k, st, sc)?,
        Stmt::Wgmma { .. } => wgmma::check_wgmma(k, st, sc)?,
        Stmt::WgmmaCommit | Stmt::WgmmaWait { .. } => {}
        Stmt::GridDepWait | Stmt::GridDepLaunchDependents => {}
        Stmt::Tcgen05Mma { .. }
        | Stmt::Tcgen05Ld { .. }
        | Stmt::Tcgen05Alloc { .. }
        | Stmt::Tcgen05Dealloc { .. }
        | Stmt::Tcgen05RelinquishPermit => tcgen05::check_tcgen05(k, st, sc)?,
        Stmt::MatrixLoad { .. } => matrix::check_matrix_load(k, st, sc)?,
        Stmt::MbarInit { .. }
        | Stmt::MbarArrive { .. }
        | Stmt::MbarCompleteTx { .. }
        | Stmt::Tcgen05Commit { .. }
        | Stmt::MbarWait { .. } => mbar::check_mbar(k, st, sc)?,
        Stmt::MatrixStore { .. } => matrix::check_matrix_store(k, st, sc)?,
        Stmt::CopyAsync { .. } | Stmt::AsyncCommit | Stmt::AsyncWait { .. } => {
            async_copy::check_async(k, st, sc)?;
        }
        Stmt::CopyBulk { .. } => mbar::check_copy_bulk(k, st, sc)?,
        Stmt::CopyUnpack { .. } => mbar::check_copy_unpack(k, st, sc)?,
        Stmt::ExternCall {
            decl,
            args,
            results,
        } => extern_call::check_extern_call(k, (*decl, args, results), sc, declared_here)?,
        Stmt::Call {
            callee,
            args,
            result,
        } => check_call(k, (*callee, args, *result), sc, declared_here)?,
        Stmt::CallTuple {
            callee,
            args,
            results,
        } => check_call_tuple(k, (*callee, args, results), sc, declared_here)?,
        Stmt::Barrier { scope, kind } => {
            if !scope_ok(k, *scope) {
                return Err(KernelError::BadBarrierScope);
            }
            let Some(launch) = k.launch.as_ref() else {
                return Err(KernelError::ExecWithoutLaunch);
            };
            if let BarrierKind::Named { id, count } | BarrierKind::NamedArrive { id, count } = kind
            {
                check_named(*scope, (*id, *count), launch.block)?;
            }
        }
        _ => unreachable!("check_sync_stmt is called for sync statements only"),
    }
    Ok(())
}

/// Whether a barrier, fence or atomic may name `scope`: `Subgroup`, `Block` and `Device` always, `Cluster` only in a
/// launch that has cluster dimensions; `Thread` has no semantics.
fn scope_ok(k: &KernelFn, scope: ExecScope) -> bool {
    match scope {
        ExecScope::Subgroup | ExecScope::Block | ExecScope::Device => true,
        ExecScope::Cluster => k.launch.is_some_and(|l| l.cluster.is_some()),
        ExecScope::System => k.launch.is_some_and(|l| l.devices > 1),
        _ => false,
    }
}

/// A `Call` against the signature of its callee: arity, each argument's kind and type, the buffer roles
/// the caller really has, and the result local's type (`docs/options-u3-calls.md`, Q2 and Q5).
fn check_call(
    k: &KernelFn,
    (callee, args, result): (u32, &[CallArg], Option<u32>),
    sc: &mut Scope,
    declared_here: &mut Vec<usize>,
) -> Result<(), KernelError> {
    let sig = check_callee_args(k, (callee, args), sc)?;
    match (result, sig.result) {
        (Some(local), Some(ty)) => declare(k, local, ty, sc, declared_here)?,
        (Some(_), None) => {
            return Err(KernelError::CallResult {
                what: "a result is asked of a callee that has none",
            });
        }
        (None, _) => {}
    }
    Ok(())
}

/// A `CallTuple`: the same argument checks as a `Call`, and one result local per value the callee returns, each of
/// the value's type.
fn check_call_tuple(
    k: &KernelFn,
    (callee, args, results): (u32, &[CallArg], &[u32]),
    sc: &mut Scope,
    declared_here: &mut Vec<usize>,
) -> Result<(), KernelError> {
    let sig = check_callee_args(k, (callee, args), sc)?;
    let tys: Vec<Ty> = sig.result.into_iter().chain(sig.extra).collect();
    if results.len() != tys.len() {
        return Err(KernelError::CallResult {
            what: "a CallTuple that does not name one local per value its callee returns",
        });
    }
    for (local, ty) in results.iter().zip(tys) {
        declare(k, *local, ty, sc, declared_here)?;
    }
    Ok(())
}

/// Resolve the callee and check the arguments against its parameters (arity, kind, type, buffer role).
fn check_callee_args(
    k: &KernelFn,
    (callee, args): (u32, &[CallArg]),
    sc: &Scope,
) -> Result<CalleeSig, KernelError> {
    if sc.callees.is_empty() {
        return Err(KernelError::CallUnsupported {
            what: "a Call outside a module: check it with check_module",
        });
    }
    let Some(sig) = sc.callees.get(callee as usize).cloned() else {
        return Err(KernelError::BadCallee { callee });
    };
    if args.len() != sig.params.len() {
        return Err(KernelError::CallArity {
            expected: sig.params.len(),
            found: args.len(),
        });
    }
    for (i, (a, p)) in args.iter().zip(&sig.params).enumerate() {
        check_call_arg(k, i, (a, p), sc)?;
    }
    Ok(sig)
}

/// One argument against the parameter in its position.
fn check_call_arg(
    k: &KernelFn,
    i: usize,
    (arg, param): (&CallArg, &ParamKind),
    sc: &Scope,
) -> Result<(), KernelError> {
    let bad = |what| Err(KernelError::CallArgMismatch { arg: i, what });
    match (arg, param) {
        (CallArg::Scalar(e), ParamKind::Scalar { ty, .. }) => {
            if check_expr(k, e, sc)? == Ty::Scalar(*ty) {
                Ok(())
            } else {
                bad("a scalar argument of a different type than its parameter")
            }
        }
        (CallArg::Buffer(b), ParamKind::Buffer { elem, role, rank }) => {
            match k.params.get(*b as usize).map(|p| &p.kind) {
                Some(ParamKind::Buffer {
                    elem: e,
                    role: r,
                    rank: n,
                }) => buffer_fits((*elem, *role, *rank), (*e, *r, *n)).map_or(Ok(()), bad),
                _ => bad("a buffer argument that is not a buffer parameter of the caller"),
            }
        }
        (CallArg::Alloc(a), ParamKind::Buffer { elem, role, rank }) => {
            let Some(d) = k.allocs.get(*a as usize) else {
                return bad("an allocation the caller does not declare");
            };
            alloc_fits((*elem, *role, *rank), (d.elem, d.space)).map_or(Ok(()), bad)
        }
        (CallArg::Struct(vals), ParamKind::Struct { fields }) => {
            if vals.len() != fields.len() {
                return bad(
                    "a struct argument with a different number of fields than its parameter",
                );
            }
            for (e, f) in vals.iter().zip(fields) {
                if check_expr(k, e, sc)? != Ty::Scalar(f.ty) {
                    return bad("a struct field of a different type than its parameter's field");
                }
            }
            Ok(())
        }
        _ => bad("an argument of a different kind than its parameter"),
    }
}

/// Why the caller's buffer `(elem, role, rank)` cannot be passed to the callee's `(elem, role, rank)`.
fn buffer_fits(
    callee: (ElementKind, BufferRole, Option<u8>),
    caller: (ElementKind, BufferRole, Option<u8>),
) -> Option<&'static str> {
    let ((ce, cr, cn), (e, r, n)) = (callee, caller);
    if ce != e {
        Some("a buffer of a different element type than its parameter")
    } else if (cr.reads() && !r.reads()) || (cr.writes() && !r.writes()) {
        Some("a buffer without the role the callee uses it with")
    } else if matches!((cn, n), (Some(x), Some(y)) if x != y) {
        Some("a buffer of a different rank than its parameter")
    } else {
        None
    }
}

/// Why the caller's allocation `(elem, space)` cannot be passed to the callee's buffer parameter.
fn alloc_fits(
    callee: (ElementKind, BufferRole, Option<u8>),
    alloc: (ElementKind, MemorySpace),
) -> Option<&'static str> {
    let ((ce, cr, cn), (e, space)) = (callee, alloc);
    if ce != e {
        Some("an allocation of a different element type than its parameter")
    } else if space == MemorySpace::Constant && cr.writes() {
        Some("a Constant allocation passed to a callee that writes it")
    } else if cn != Some(1) {
        Some("an allocation is one-dimensional, its parameter must declare rank 1")
    } else {
        None
    }
}

/// Check one statement; `Ok(true)` when it unconditionally leaves the block (`Break`,
/// `Continue`, `Return`).
fn check_stmt(
    k: &KernelFn,
    st: &Stmt,
    sc: &mut Scope,
    declared_here: &mut Vec<usize>,
) -> Result<bool, KernelError> {
    match st {
        Stmt::Let { local, init } => {
            let ty = check_expr(k, init, sc)?;
            declare(k, *local, ty, sc, declared_here)?;
        }
        Stmt::Load { local, place } => check_load(k, *local, place, sc, declared_here)?,
        Stmt::MulticastLoadReduce { .. } => check_multicast_load_reduce(k, st, sc, declared_here)?,
        Stmt::Assign { local, value } => check_assign(k, *local, value, sc)?,
        Stmt::Store { place, value } => check_store(k, place, value, sc)?,
        Stmt::MaskedLoad {
            local,
            place,
            mask,
            fill,
        } => {
            check_mask(k, mask, sc)?;
            let elem = check_load_elem(k, place, sc)?;
            if check_expr(k, fill, sc)? != Ty::Scalar(elem) {
                return Err(KernelError::TypeMismatch {
                    what: "MaskedLoad fill",
                });
            }
            declare(k, *local, Ty::Scalar(elem), sc, declared_here)?;
        }
        Stmt::MaskedStore { place, value, mask } => {
            if matches!(place, Place::Multicast { .. }) {
                return Err(KernelError::MulticastUnsupported {
                    what: "a MaskedStore on a multicast place (guard a Store with an If)",
                });
            }
            check_mask(k, mask, sc)?;
            check_store(k, place, value, sc)?;
        }
        Stmt::AtomicRmw { .. }
        | Stmt::AtomicCas { .. }
        | Stmt::Fence { .. }
        | Stmt::ProxyFence { .. }
        | Stmt::Collective { .. }
        | Stmt::CollectiveTuple { .. }
        | Stmt::Mma { .. }
        | Stmt::Wgmma { .. }
        | Stmt::WgmmaCommit
        | Stmt::WgmmaWait { .. }
        | Stmt::GridDepWait
        | Stmt::GridDepLaunchDependents
        | Stmt::Tcgen05Mma { .. }
        | Stmt::Tcgen05Commit { .. }
        | Stmt::Tcgen05Ld { .. }
        | Stmt::Tcgen05Alloc { .. }
        | Stmt::Tcgen05Dealloc { .. }
        | Stmt::Tcgen05RelinquishPermit
        | Stmt::MatrixLoad { .. }
        | Stmt::MbarInit { .. }
        | Stmt::MbarArrive { .. }
        | Stmt::MbarCompleteTx { .. }
        | Stmt::MbarWait { .. }
        | Stmt::MatrixStore { .. }
        | Stmt::TileLoad { .. }
        | Stmt::TileMma { .. }
        | Stmt::TileFill { .. }
        | Stmt::TileStore { .. }
        | Stmt::CopyBulk { .. }
        | Stmt::CopyUnpack { .. }
        | Stmt::CopyAsync { .. }
        | Stmt::AsyncCommit
        | Stmt::AsyncWait { .. }
        | Stmt::Call { .. }
        | Stmt::CallTuple { .. }
        | Stmt::ExternCall { .. }
        | Stmt::Barrier { .. } => check_sync_stmt(k, st, sc, declared_here)?,
        Stmt::Assume { cond } => scalar_cond(k, cond, sc)?,
        Stmt::If { cond, then, els } => {
            scalar_cond(k, cond, sc)?;
            check_block(k, then, sc)?;
            check_block(k, els, sc)?;
        }
        Stmt::For {
            local,
            start,
            end,
            step,
            body,
            unroll,
        } => {
            if matches!(unroll, crate::kernel::Unroll::Count(0)) {
                return Err(KernelError::UnrollUnsupported {
                    what: "an unroll factor of 0",
                });
            }
            check_for(k, *local, [start, end, step], body, sc)?;
        }
        Stmt::While { cond, body } => {
            scalar_cond(k, cond, sc)?;
            in_loop(k, body, sc)?;
        }
        Stmt::Break | Stmt::Continue => {
            if sc.loop_depth == 0 {
                return Err(KernelError::BreakOutsideLoop);
            }
            return Ok(true);
        }
        Stmt::Return => return Ok(true),
    }
    Ok(false)
}

fn check_load(
    k: &KernelFn,
    local: u32,
    place: &Place,
    sc: &mut Scope,
    declared_here: &mut Vec<usize>,
) -> Result<(), KernelError> {
    let elem = check_load_elem(k, place, sc)?;
    declare(k, local, Ty::Scalar(elem), sc, declared_here)
}

/// The element type of a readable `place`.
fn check_load_elem(k: &KernelFn, place: &Place, sc: &Scope) -> Result<ElementKind, KernelError> {
    if matches!(place, Place::Multicast { .. }) {
        return Err(KernelError::MulticastUnsupported {
            what: "a Load of a multicast place (a reducing read is a MulticastLoadReduce)",
        });
    }
    let (elem, role) = check_place_raw_ok(k, place, sc)?;
    if !role.reads() {
        return Err(match place {
            Place::Logical { buffer, .. }
            | Place::Peer { buffer, .. }
            | Place::Raw { buffer, .. }
            | Place::Multicast { buffer, .. } => KernelError::ReadFromWriteOnly { buffer: *buffer },
            Place::Alloc { alloc, .. }
            | Place::AllocRaw { alloc, .. }
            | Place::ClusterPeer { alloc, .. } => KernelError::BadPlace { buffer: *alloc },
        });
    }
    Ok(elem)
}

/// A mask is a scalar of any type (tested `!= 0`, as a `Select` condition is).
fn check_mask(k: &KernelFn, mask: &ScalarExpr, sc: &Scope) -> Result<(), KernelError> {
    match check_expr(k, mask, sc)? {
        Ty::Scalar(_) => Ok(()),
        Ty::Packed(..) => Err(KernelError::TypeMismatch {
            what: "a packed mask",
        }),
    }
}

fn check_store(
    k: &KernelFn,
    place: &Place,
    value: &ScalarExpr,
    sc: &Scope,
) -> Result<(), KernelError> {
    let (elem, role) = check_place_raw_ok(k, place, sc)?;
    if !role.writes() {
        return Err(match place {
            Place::Logical { buffer, .. }
            | Place::Peer { buffer, .. }
            | Place::Raw { buffer, .. }
            | Place::Multicast { buffer, .. } => KernelError::WriteToReadOnly { buffer: *buffer },
            Place::Alloc { alloc, .. }
            | Place::AllocRaw { alloc, .. }
            | Place::ClusterPeer { alloc, .. } => KernelError::WriteToConstant { alloc: *alloc },
        });
    }
    if matches!(place, Place::Multicast { .. }) && !multicast_store_elem(elem) {
        return Err(KernelError::MulticastUnsupported {
            what: "a multicast Store of an element other than I32, U32, F32, F16 or Bf16",
        });
    }
    if types::is_subbyte_int(elem) && !is_private_alloc_place(k, place) {
        // a sub-byte Store is a read-modify-write of a byte two threads share (ROADMAP U9b, ruled 2026-11-05); a `Private`
        // allocation is one thread's, so no byte is shared and the store is allowed (an MMA fragment of 4-bit elements)
        return Err(KernelError::TypeMismatch {
            what: "a Store of a sub-byte element outside a Private allocation (shared and global sub-byte buffers are read-only in kernels)",
        });
    }
    if check_expr(k, value, sc)? != Ty::Scalar(elem) {
        return Err(KernelError::TypeMismatch { what: "Store" });
    }
    Ok(())
}

/// Whether `place` is an element of a `Private` allocation (one thread's own storage).
fn is_private_alloc_place(k: &KernelFn, place: &Place) -> bool {
    matches!(place, Place::Alloc { alloc, .. }
        if k.allocs.get(*alloc as usize).is_some_and(|a| a.space == MemorySpace::Private))
}

/// The parts of an `AtomicRmw` on a multicast place other than its place and operand.
#[derive(Clone, Copy)]
struct MulticastRed {
    op: AtomicOp,
    old: Option<u32>,
    order: MemoryOrder,
    scope: ExecScope,
}

/// An `AtomicRmw` on a `Place::Multicast` (`multimem.red`, ROADMAP U12 slice 3): a `ReadWrite` buffer in a launch of
/// several devices, `System` scope, `Release` order, no old value, and an op and element `multicast_red_ok` admits.
fn check_multicast_red(
    k: &KernelFn,
    place: &Place,
    value: &ScalarExpr,
    red: MulticastRed,
    sc: &Scope,
) -> Result<(), KernelError> {
    let refuse = |what| Err(KernelError::MulticastUnsupported { what });
    let (elem, role) = check_place(k, place, sc)?;
    if !(role.reads() && role.writes()) {
        return Err(KernelError::AtomicUnsupported {
            what: "an atomic on a buffer that is not ReadWrite",
        });
    }
    if red.scope != ExecScope::System {
        return refuse("a multicast AtomicRmw at a scope other than System");
    }
    if red.order != MemoryOrder::Release {
        return refuse("a multicast AtomicRmw with an order other than Release");
    }
    if red.old.is_some() {
        return refuse(
            "a multicast AtomicRmw with an old value (a reduction has no single old value)",
        );
    }
    if !multicast_red_ok(red.op, elem) {
        return refuse("a multicast AtomicRmw other than Add, Min or Max on I32/U32 or Add on F32");
    }
    if check_expr(k, value, sc)? != Ty::Scalar(elem) {
        return Err(KernelError::TypeMismatch { what: "AtomicRmw" });
    }
    Ok(())
}

/// A `MulticastLoadReduce` (`multimem.ld_reduce`, ROADMAP U12 slice 4): a readable [`Place::Multicast`] in a launch of
/// several devices, a `Relaxed` or `Acquire` order, and an op and element `multicast_ld_reduce_ok` admits; it declares
/// `local` with the element's type.
fn check_multicast_load_reduce(
    k: &KernelFn,
    stmt: &Stmt,
    sc: &mut Scope,
    declared_here: &mut Vec<usize>,
) -> Result<(), KernelError> {
    let Stmt::MulticastLoadReduce {
        local,
        place,
        op,
        order,
    } = stmt
    else {
        unreachable!("called for a MulticastLoadReduce");
    };
    let (local, op, order) = (*local, *op, *order);
    let refuse = |what| Err(KernelError::MulticastUnsupported { what });
    if !matches!(place, Place::Multicast { .. }) {
        return refuse("a MulticastLoadReduce on a place other than a multicast one");
    }
    let (elem, role) = check_place(k, place, sc)?;
    if !role.reads() {
        return Err(KernelError::ReadFromWriteOnly {
            buffer: match place {
                Place::Multicast { buffer, .. } => *buffer,
                _ => unreachable!("checked to be a multicast place"),
            },
        });
    }
    if !matches!(order, MemoryOrder::Relaxed | MemoryOrder::Acquire) {
        return refuse("a MulticastLoadReduce with an order other than Relaxed or Acquire");
    }
    if !multicast_ld_reduce_ok(op, elem) {
        return refuse(
            "a MulticastLoadReduce other than Add, Min or Max on I32/U32/F16/Bf16 or Add on F32",
        );
    }
    declare(k, local, Ty::Scalar(elem), sc, declared_here)
}

/// The ops and elements `multimem.ld_reduce` combines: `Add`, `Min`, `Max` on `I32`, `U32`, `F16`, `Bf16`; `Add` on `F32`.
fn multicast_ld_reduce_ok(op: AtomicOp, e: ElementKind) -> bool {
    match e {
        ElementKind::I32 | ElementKind::U32 | ElementKind::F16 | ElementKind::Bf16 => {
            matches!(op, AtomicOp::Add | AtomicOp::Min | AtomicOp::Max)
        }
        ElementKind::F32 => matches!(op, AtomicOp::Add),
        _ => false,
    }
}

/// The elements `multimem.st` writes (ROADMAP U12 slice 3).
fn multicast_store_elem(e: ElementKind) -> bool {
    matches!(
        e,
        ElementKind::I32
            | ElementKind::U32
            | ElementKind::F32
            | ElementKind::F16
            | ElementKind::Bf16
    )
}

/// The ops and elements `multimem.red` combines (ROADMAP U12 slice 3): `Add`, `Min`, `Max` on a 32-bit integer, `Add`
/// on `F32`, `F16` and `Bf16` (ROADMAP U12b: the oracle sums exactly and rounds once).
fn multicast_red_ok(op: AtomicOp, e: ElementKind) -> bool {
    match e {
        ElementKind::I32 | ElementKind::U32 => {
            matches!(op, AtomicOp::Add | AtomicOp::Min | AtomicOp::Max)
        }
        ElementKind::F32 | ElementKind::F16 | ElementKind::Bf16 => matches!(op, AtomicOp::Add),
        _ => false,
    }
}

/// The parts of an `AtomicRmw` other than its place and operand.
#[derive(Clone, Copy)]
struct AtomicForm {
    op: AtomicOp,
    site: AtomicSite,
}

/// An `AtomicRmw` (v1): a `ReadWrite` global buffer or `Shared` element, a value of the element's type,
/// a `Subgroup`, `Block` or `Device` scope, any memory order, and an `old` local (if any) of the element's type that this
/// statement declares.
fn check_atomic(
    k: &KernelFn,
    place: &Place,
    value: &ScalarExpr,
    form: AtomicForm,
    sc: &mut Scope,
    declared_here: &mut Vec<usize>,
) -> Result<(), KernelError> {
    let AtomicForm { op, site } = form;
    let elem = check_atomic_site(k, place, site, sc)?;
    let float_ok = matches!(op, AtomicOp::Add | AtomicOp::Exchange)
        && matches!(
            elem,
            ElementKind::F32 | ElementKind::F64 | ElementKind::F16 | ElementKind::Bf16
        );
    if crate::types::int_range(elem).is_none() && !float_ok {
        return Err(KernelError::AtomicUnsupported {
            what: "an atomic on a non-integer element (only an F16/Bf16/F32/F64 Add or Exchange is admitted)",
        });
    }
    if matches!(op, AtomicOp::Inc | AtomicOp::Dec) && elem != ElementKind::U32 {
        return Err(KernelError::AtomicUnsupported {
            what: "an Inc or Dec atomic outside a U32 element (CUDA atomicInc/atomicDec, PTX atom.inc.u32/atom.dec.u32)",
        });
    }
    if check_expr(k, value, sc)? != Ty::Scalar(elem) {
        return Err(KernelError::TypeMismatch { what: "AtomicRmw" });
    }
    if let Some(local) = site.old {
        declare(k, local, Ty::Scalar(elem), sc, declared_here)?;
    }
    Ok(())
}

/// An `AtomicCas`: an integer element at an atomic site, `expected` and `desired` of that type, and an
/// `old` local (if any) of that type.
fn check_cas(
    k: &KernelFn,
    place: &Place,
    [expected, desired]: [&ScalarExpr; 2],
    site: AtomicSite,
    sc: &mut Scope,
    declared_here: &mut Vec<usize>,
) -> Result<(), KernelError> {
    let elem = check_atomic_site(k, place, site, sc)?;
    if crate::types::int_range(elem).is_none() {
        return Err(KernelError::AtomicUnsupported {
            what: "an AtomicCas on a non-integer element (compare-and-exchange is integer only)",
        });
    }
    for e in [expected, desired] {
        if check_expr(k, e, sc)? != Ty::Scalar(elem) {
            return Err(KernelError::TypeMismatch { what: "AtomicCas" });
        }
    }
    if let Some(local) = site.old {
        declare(k, local, Ty::Scalar(elem), sc, declared_here)?;
    }
    Ok(())
}

/// What every atomic statement states about where it runs: the old-value local and the scope (the memory
/// order needs no check: every order is meaningful on a read-modify-write).
#[derive(Clone, Copy)]
struct AtomicSite {
    old: Option<u32>,
    scope: ExecScope,
}

/// The checks every atomic shares: a `Subgroup`, `Block` or `Device` scope, a `ReadWrite` global buffer or a
/// `Shared` allocation element. Returns the element type.
fn check_atomic_site(
    k: &KernelFn,
    place: &Place,
    site: AtomicSite,
    sc: &Scope,
) -> Result<ElementKind, KernelError> {
    // every memory order is meaningful on a read-modify-write (`SeqCst` is treated as `AcqRel`)
    if !scope_ok(k, site.scope) {
        return Err(KernelError::AtomicUnsupported {
            what: "an atomic scope other than Subgroup, Block, Device or (with launch cluster dims) Cluster",
        });
    }
    let outside_shared = match place {
        Place::Alloc { alloc, .. } => k
            .allocs
            .get(*alloc as usize)
            .is_some_and(|a| a.space != MemorySpace::Shared),
        _ => false,
    };
    if outside_shared {
        return Err(KernelError::AtomicUnsupported {
            what: "an atomic on an allocation outside Shared memory",
        });
    }
    let (elem, role) = check_place(k, place, sc)?;
    if !(role.reads() && role.writes()) {
        return Err(KernelError::AtomicUnsupported {
            what: "an atomic on a buffer that is not ReadWrite",
        });
    }
    Ok(elem)
}

fn check_assign(
    k: &KernelFn,
    local: u32,
    value: &ScalarExpr,
    sc: &Scope,
) -> Result<(), KernelError> {
    let ty = check_expr(k, value, sc)?;
    use_local(k, local, sc)?;
    let i = local as usize;
    if sc.param_local[i] {
        return Err(KernelError::ParamAssigned { local });
    }
    if sc.counter_local[i] {
        return Err(KernelError::CounterAssigned { local });
    }
    if sc.tys[i] != ty {
        return Err(KernelError::TypeMismatch { what: "Assign" });
    }
    Ok(())
}

/// A `For`: three integer bounds of one type, a counter declared by the loop itself, and a body
/// that may `Break` and `Continue`.
fn check_for(
    k: &KernelFn,
    local: u32,
    bounds: [&ScalarExpr; 3],
    body: &Block,
    sc: &mut Scope,
) -> Result<(), KernelError> {
    let t0 = check_expr(k, bounds[0], sc)?;
    let t1 = check_expr(k, bounds[1], sc)?;
    let t2 = check_expr(k, bounds[2], sc)?;
    let int = matches!(t0, Ty::Scalar(e) if types::int_range(e).is_some());
    if t0 != t1 || t1 != t2 || !int {
        return Err(KernelError::TypeMismatch { what: "For bounds" });
    }
    let mut counter = Vec::new();
    declare(k, local, t0, sc, &mut counter)?;
    sc.counter_local[local as usize] = true;
    let r = in_loop(k, body, sc);
    sc.counter_local[local as usize] = false;
    sc.in_scope[local as usize] = false;
    r
}

/// Check `body` one loop deeper.
fn in_loop(k: &KernelFn, body: &Block, sc: &mut Scope) -> Result<(), KernelError> {
    sc.loop_depth += 1;
    let r = check_block(k, body, sc);
    sc.loop_depth -= 1;
    r
}

fn declare(
    k: &KernelFn,
    local: u32,
    ty: Ty,
    sc: &mut Scope,
    declared_here: &mut Vec<usize>,
) -> Result<(), KernelError> {
    let i = local as usize;
    if i >= k.locals.len() {
        return Err(KernelError::UnknownLocal { local });
    }
    if sc.in_scope[i] {
        return Err(KernelError::Redeclared { local });
    }
    if sc.tys[i] != ty {
        return Err(KernelError::TypeMismatch {
            what: "declaration and its local",
        });
    }
    sc.in_scope[i] = true;
    sc.ever_declared[i] = true;
    declared_here.push(i);
    Ok(())
}

fn use_local(k: &KernelFn, local: u32, sc: &Scope) -> Result<(), KernelError> {
    let i = local as usize;
    if i >= k.locals.len() {
        return Err(KernelError::UnknownLocal { local });
    }
    if sc.in_scope[i] {
        Ok(())
    } else if sc.ever_declared[i] {
        Err(KernelError::OutOfScope { local })
    } else {
        Err(KernelError::UseBeforeLet { local })
    }
}

fn scalar_cond(k: &KernelFn, cond: &ScalarExpr, sc: &Scope) -> Result<(), KernelError> {
    match check_expr(k, cond, sc)? {
        Ty::Scalar(_) => Ok(()),
        Ty::Packed(..) => Err(KernelError::TypeMismatch {
            what: "condition (a packed value has no truth value)",
        }),
    }
}

/// The place of a statement other than a plain `Load`/`Store` (and their masked forms): a [`Place::Raw`] is refused here by
/// name, since the atomics, the async copies, the matrix and barrier statements have no raw-byte rules yet (ROADMAP U9c).
fn check_place(
    k: &KernelFn,
    p: &Place,
    sc: &Scope,
) -> Result<(ElementKind, BufferRole), KernelError> {
    if matches!(p, Place::Raw { .. } | Place::AllocRaw { .. }) {
        return Err(KernelError::TypeMismatch {
            what: "a Raw place here (only Load, Store, MaskedLoad and MaskedStore take one)",
        });
    }
    check_place_raw_ok(k, p, sc)
}

fn check_place_raw_ok(
    k: &KernelFn,
    p: &Place,
    sc: &Scope,
) -> Result<(ElementKind, BufferRole), KernelError> {
    let (buffer, coords, device) = match p {
        Place::Logical { buffer, coords } => (buffer, coords, None),
        Place::Peer {
            buffer,
            device,
            coords,
        } => (buffer, coords, Some(device)),
        Place::Multicast { buffer, coords } => {
            if !k.launch.is_some_and(|l| l.devices > 1) {
                return Err(KernelError::PeerUnsupported {
                    what: "a multicast place in a launch that spans one device",
                });
            }
            (buffer, coords, None)
        }
        Place::Alloc { alloc, index } => return check_alloc_place(k, *alloc, index, sc),
        Place::Raw {
            buffer,
            byte_offset,
            elem,
        } => return check_raw_place(k, *buffer, byte_offset, *elem, sc),
        Place::AllocRaw {
            alloc,
            byte_offset,
            elem,
        } => return check_alloc_raw_place(k, *alloc, byte_offset, *elem, sc),
        Place::ClusterPeer {
            alloc,
            block,
            index,
        } => return check_peer_place(k, *alloc, (block, index), sc),
    };
    check_logical_place(k, *buffer, coords, device, sc)
}

/// A `Logical`, `Peer` or `Multicast` place: a buffer parameter indexed by integer coordinates (and a peer device).
fn check_logical_place(
    k: &KernelFn,
    buffer: u32,
    coords: &[ScalarExpr],
    device: Option<&ScalarExpr>,
    sc: &Scope,
) -> Result<(ElementKind, BufferRole), KernelError> {
    let bad = KernelError::BadPlace { buffer };
    let Some(Param {
        kind: ParamKind::Buffer { elem, role, rank },
        ..
    }) = k.params.get(buffer as usize)
    else {
        return Err(bad);
    };
    if rank.is_some_and(|r| usize::from(r) != coords.len()) {
        return Err(bad);
    }
    if let Some(d) = device {
        if !k.launch.is_some_and(|l| l.devices > 1) {
            return Err(KernelError::PeerUnsupported {
                what: "a peer place in a launch that spans one device",
            });
        }
        let int = matches!(check_expr(k, d, sc)?, Ty::Scalar(e) if types::int_range(e).is_some());
        if !int {
            return Err(bad);
        }
    }
    for c in coords {
        let int = matches!(check_expr(k, c, sc)?, Ty::Scalar(e) if types::int_range(e).is_some());
        if !int {
            return Err(bad);
        }
    }
    Ok((*elem, *role))
}

/// A `Raw` place (ROADMAP U9c): a buffer parameter, an integer byte offset, and an access type of a whole number of bytes. It
/// reads and writes as the buffer's role says; the access type need not be the buffer's element type.
fn check_raw_place(
    k: &KernelFn,
    buffer: u32,
    byte_offset: &ScalarExpr,
    elem: ElementKind,
    sc: &Scope,
) -> Result<(ElementKind, BufferRole), KernelError> {
    let bad = KernelError::BadPlace { buffer };
    let Some(Param {
        kind: ParamKind::Buffer { role, .. },
        ..
    }) = k.params.get(buffer as usize)
    else {
        return Err(bad);
    };
    if types::raw_access_bytes(elem).is_none() {
        return Err(KernelError::TypeMismatch {
            what: "a Raw place of an element that is not a whole number of bytes",
        });
    }
    let int =
        matches!(check_expr(k, byte_offset, sc)?, Ty::Scalar(e) if types::int_range(e).is_some());
    if !int {
        return Err(bad);
    }
    Ok((elem, *role))
}

/// An `AllocRaw` place (ROADMAP U9c): an existing `Shared` or `Private` allocation (`Shared` needs a launch), an integer byte
/// offset and an access type of a whole number of bytes. `Constant` (the oracle holds no initializer) and `Tensor` are refused.
fn check_alloc_raw_place(
    k: &KernelFn,
    alloc: u32,
    byte_offset: &ScalarExpr,
    elem: ElementKind,
    sc: &Scope,
) -> Result<(ElementKind, BufferRole), KernelError> {
    let Some(a) = k.allocs.get(alloc as usize) else {
        return Err(KernelError::BadPlace { buffer: alloc });
    };
    match a.space {
        MemorySpace::Tensor => {
            return Err(KernelError::Tcgen05Unsupported {
                what: "an ordinary access to tensor memory",
            });
        }
        MemorySpace::Tile => {
            return Err(KernelError::Tile {
                alloc,
                why: "an ordinary access to a Tile (only the tile statements reach one)",
            });
        }
        MemorySpace::Constant => {
            return Err(KernelError::TypeMismatch {
                what: "an AllocRaw place on a Constant allocation",
            });
        }
        MemorySpace::Shared if k.launch.is_none() => {
            return Err(KernelError::SharedWithoutLaunch { alloc });
        }
        _ => {}
    }
    if types::raw_access_bytes(elem).is_none() || types::elem_bits(a.elem) % 8 != 0 {
        return Err(KernelError::TypeMismatch {
            what: "an AllocRaw place of an element, or on an allocation, that is not a whole number of bytes",
        });
    }
    let int =
        matches!(check_expr(k, byte_offset, sc)?, Ty::Scalar(e) if types::int_range(e).is_some());
    if !int {
        return Err(KernelError::BadPlace { buffer: alloc });
    }
    Ok((elem, BufferRole::ReadWrite))
}

/// A `ClusterPeer` place (ROADMAP U12 slice 1): a `Shared` allocation, a launch that names cluster dimensions, and an
/// integer rank and index. It reads and writes like the allocation's own element.
fn check_peer_place(
    k: &KernelFn,
    alloc: u32,
    (block, index): (&ScalarExpr, &ScalarExpr),
    sc: &Scope,
) -> Result<(ElementKind, BufferRole), KernelError> {
    let Some(a) = k.allocs.get(alloc as usize) else {
        return Err(KernelError::BadPlace { buffer: alloc });
    };
    if a.space != MemorySpace::Shared {
        return Err(KernelError::ClusterPeerUnsupported {
            what: "a cluster-peer place on an allocation outside Shared memory",
        });
    }
    if !k.launch.is_some_and(|l| l.cluster.is_some()) {
        return Err(KernelError::ClusterPeerUnsupported {
            what: "a cluster-peer place in a launch that names no cluster dimensions",
        });
    }
    for e in [block, index] {
        let int = matches!(check_expr(k, e, sc)?, Ty::Scalar(t) if types::int_range(t).is_some());
        if !int {
            return Err(KernelError::BadPlace { buffer: alloc });
        }
    }
    Ok((a.elem, BufferRole::ReadWrite))
}

/// An `Alloc` place: the allocation exists, `Shared` has a launch, the index is an integer. A
/// `Constant` allocation reads like a read-only buffer; the others like a read-write one.
fn check_alloc_place(
    k: &KernelFn,
    alloc: u32,
    index: &ScalarExpr,
    sc: &Scope,
) -> Result<(ElementKind, BufferRole), KernelError> {
    let Some(a) = k.allocs.get(alloc as usize) else {
        return Err(KernelError::BadPlace { buffer: alloc });
    };
    if a.space == MemorySpace::Tensor {
        return Err(KernelError::Tcgen05Unsupported {
            what: "an ordinary access to tensor memory",
        });
    }
    if a.space == MemorySpace::Tile {
        return Err(KernelError::Tile {
            alloc,
            why: "an ordinary access to a Tile (only the tile statements reach one)",
        });
    }
    if a.space == MemorySpace::Shared && k.launch.is_none() {
        return Err(KernelError::SharedWithoutLaunch { alloc });
    }
    let int = matches!(check_expr(k, index, sc)?, Ty::Scalar(e) if types::int_range(e).is_some());
    if !int {
        return Err(KernelError::BadPlace { buffer: alloc });
    }
    let role = if a.space == MemorySpace::Constant {
        BufferRole::Read
    } else {
        BufferRole::ReadWrite
    };
    Ok((a.elem, role))
}

/// Type an expression inside a kernel: reject intent-layer leaves, check every `Local` is in
/// scope, then defer to [`types::infer_ty_in`] with the declared local types.
fn check_expr(k: &KernelFn, e: &ScalarExpr, sc: &Scope) -> Result<Ty, KernelError> {
    walk(k, e, sc)?;
    types::infer_ty_in(e, k.compute, &sc.tys).map_err(KernelError::Type)
}

/// An `Exec` leaf needs a launch, an axis in `0..=2`, and (for the subgroup ids) a launch that
/// names a subgroup size.
fn check_exec(k: &KernelFn, id: crate::exec::ExecId) -> Result<(), KernelError> {
    let Some(l) = &k.launch else {
        return Err(KernelError::ExecWithoutLaunch);
    };
    if let Some(axis) = id.axis().filter(|a| *a > 2) {
        return Err(KernelError::ExecAxisOutOfRange { axis });
    }
    if id.needs_subgroup() && l.subgroup.is_none() {
        return Err(KernelError::SubgroupIdWithoutSubgroup);
    }
    Ok(())
}

fn walk(k: &KernelFn, e: &ScalarExpr, sc: &Scope) -> Result<(), KernelError> {
    match e {
        ScalarExpr::Local(i) => return use_local(k, *i, sc),
        ScalarExpr::Exec(id) => return check_exec(k, *id),
        ScalarExpr::Input(_)
        | ScalarExpr::Param(_)
        | ScalarExpr::Reduced(_)
        | ScalarExpr::Coord(_)
        | ScalarExpr::CoordInt(..) => {
            return Err(KernelError::IntentLeaf {
                node: types::node_name(e),
            });
        }
        _ => {}
    }
    for x in types::operands(e) {
        walk(k, x, sc)?;
        // a sub-byte value is read, widened and narrowed, never computed on (ROADMAP U9b, ruled 2026-11-05)
        if !matches!(e, ScalarExpr::Cast(..))
            && matches!(types::infer_ty_in(x, k.compute, &sc.tys), Ok(Ty::Scalar(t)) if types::is_subbyte_int(t))
        {
            return Err(KernelError::TypeMismatch {
                what: "arithmetic on a sub-byte value (widen it with a Cast first)",
            });
        }
    }
    Ok(())
}

/// A named barrier: `Block` scope, an id of 0 to 15, and a count of whole warps that fits in the block.
fn check_named(
    scope: crate::exec::Scope,
    (id, count): (u32, u32),
    block: [u32; 3],
) -> Result<(), KernelError> {
    let refuse = |what| Err(KernelError::NamedBarrierUnsupported { what });
    if scope != crate::exec::Scope::Block {
        return refuse("a named barrier of a scope other than Block");
    }
    if id > 15 {
        return refuse("a barrier id above 15");
    }
    if count == 0 || count % 32 != 0 || count > block.iter().product::<u32>() {
        return refuse("a thread count that is not whole warps within the block");
    }
    Ok(())
}
