// SPDX-License-Identifier: MIT OR Apache-2.0
//! `AllocDecl::dyn_layout` as C (`docs/design-dynamic-layout.md` section 5): the physical offset of a logical index is C arithmetic over
//! the kernel's scalar parameters, and the launch-time proof the oracle runs (`bind_dyn_layouts`) is emitted at the top of the kernel
//! function, so a bind the oracle refuses returns **status 4** before any thread runs.
//!
//! The offset is the layout's own: the colexicographic coordinates of the index over the flattened shape (the last mode takes the
//! rest), each times its stride, summed. The proof, with the oracle's order of refusals collapsed into one status: every shape leaf
//! at least 1 and no stride negative, the shape's size equal to the extent, every offset (so the cosize) inside the extent, the extent within the
//! enumeration cap (4 Mi, as `check_layout_offsets`), and no two indices sharing an offset (a byte map of `extent` entries).

use unpopped::backend::LowerError;
use unpopped::exec::{AllocDecl, AllocLayout};
use unpopped::kernel::{KernelFn, Stmt};
use unpopped_vocab::ElementKind;
use unpopped_vocab::layout_algebra::{Dim, DynLayout, DynTuple, IntTuple};

use crate::base::refuse;
use crate::dynamic::{extent_expr, param_of_local};

/// The status a launch returns when a `dyn_layout` does not bind to a layout the checker accepts.
pub(crate) const STATUS: i32 = 4;

fn lift(t: &IntTuple) -> DynTuple {
    match t {
        IntTuple::Int(v) => DynTuple::Leaf(Dim::Static(*v)),
        IntTuple::Tuple(x) => DynTuple::Tuple(x.iter().map(lift).collect()),
    }
}

/// A static allocation layout as the dynamic one with no parameter leaf, so one offset spelling serves both
/// (`check_kernel` has already proved the static layout injective and inside the extent, so no launch check is emitted for it).
pub(crate) fn static_as_dyn(l: &AllocLayout) -> Result<DynLayout, LowerError> {
    let d = DynLayout::new(lift(l.layout.shape()), lift(l.layout.stride()))
        .map_err(|_| refuse("a static layout this emitter cannot lift"))?;
    Ok(match l.swizzle {
        Some(sw) => d.with_swizzle(sw),
        None => d,
    })
}

fn leaves(t: &DynTuple, out: &mut Vec<Dim>) {
    match t {
        DynTuple::Leaf(d) => out.push(*d),
        DynTuple::Tuple(x) => x.iter().for_each(|y| leaves(y, out)),
    }
}

/// The `(shape, stride)` leaves of `d`, in order, as C `long long` expressions over the scalar parameters.
fn flat(k: &KernelFn, d: &DynLayout) -> Result<Vec<(String, String)>, LowerError> {
    let (mut s, mut t) = (Vec::new(), Vec::new());
    leaves(d.shape(), &mut s);
    leaves(d.stride(), &mut t);
    let c = |dim: Dim| -> Result<String, LowerError> {
        match dim {
            Dim::Static(v) => Ok(format!("{v}LL")),
            Dim::Param(l) => param_of_local(k, l)
                .map(|i| format!("((long long)p{i})"))
                .ok_or_else(|| refuse("a dyn_layout parameter that is no plain scalar parameter")),
        }
    };
    s.into_iter()
        .zip(t)
        .map(|(a, b)| Ok((c(a)?, c(b)?)))
        .collect()
}

/// The element-form swizzle of `d` for `elem`, with its period in elements (`None` for a layout without a swizzle). The swizzle is
/// XOR-bit data only: the offset is XORed with a shifted mask of its own bits, nothing is looked up and nothing is transferred.
/// The byte width of the element types this emitter gives a `dyn_layout` swizzle (the others are refused by name).
pub(crate) fn width_of(elem: ElementKind) -> Result<u32, LowerError> {
    match elem {
        ElementKind::I8 | ElementKind::U8 | ElementKind::Bool => Ok(1),
        ElementKind::F16 | ElementKind::Bf16 | ElementKind::I16 | ElementKind::U16 => Ok(2),
        ElementKind::F32 | ElementKind::I32 | ElementKind::U32 => Ok(4),
        ElementKind::F64 | ElementKind::I64 | ElementKind::U64 => Ok(8),
        other => Err(refuse(format!(
            "a dyn_layout swizzle on {other:?} elements"
        ))),
    }
}

fn swizzle_of(d: &DynLayout, elem: ElementKind) -> Result<Option<(i64, i32, u64)>, LowerError> {
    let Some(sw) = d.swizzle() else {
        return Ok(None);
    };
    let width = width_of(elem)?;
    let e = sw
        .in_elements(width)
        .map_err(|_| refuse("a dyn_layout swizzle whose base is below the element width"))?;
    let src = ((1i64 << e.bits()) - 1) << (e.base() + e.shift().max(0).unsigned_abs());
    let period = 1u64 << (e.base() + e.shift().unsigned_abs() + e.bits());
    Ok(Some((src, e.shift(), period)))
}

/// The statements that set `out` (a fresh `long long`) to the physical offset of the logical index in `idx`; `tmp` is scratch.
pub(crate) fn offset_lines(
    k: &KernelFn,
    d: &DynLayout,
    elem: ElementKind,
    (idx, tmp, out): (&str, &str, &str),
) -> Result<Vec<String>, LowerError> {
    let modes = flat(k, d)?;
    let mut lines = vec![format!("long long {tmp} = {idx}; long long {out} = 0;")];
    for (j, (s, t)) in modes.iter().enumerate() {
        if j + 1 == modes.len() {
            lines.push(format!("{out} += {tmp} * {t};"));
        } else {
            lines.push(format!("{out} += ({tmp} % {s}) * {t}; {tmp} /= {s};"));
        }
    }
    if let Some((src, shift, _)) = swizzle_of(d, elem)? {
        // `Swizzle::apply`: `x ^ ((x & src) >> shift)`, or `<<` for a negative shift
        let moved = if shift >= 0 {
            format!("(({out} & {src}LL) >> {shift})")
        } else {
            format!("(({out} & {src}LL) << {})", -shift)
        };
        lines.push(format!("{out} ^= {moved};"));
    }
    Ok(lines)
}

/// The launch-time proof for allocation `i`, as C statements that `return` [`STATUS`] (or 3 when the byte map cannot be allocated).
pub(crate) fn launch_check(
    k: &KernelFn,
    i: usize,
    a: &AllocDecl,
    d: &DynLayout,
) -> Result<Vec<String>, LowerError> {
    let modes = flat(k, d)?;
    let ext = extent_expr(k, a)?;
    let r = STATUS;
    let mut l = vec![
        format!("{{ /* dyn_layout of allocation {i} */"),
        format!("long long ext{i} = {ext}; long long sz{i} = 1;"),
        format!("if (ext{i} < 1 || ext{i} > 4194304LL) return {r};"),
    ];
    for (s, t) in &modes {
        // a leaf the layout cannot hold, or one that already overshoots the extent, is refused before it can overflow
        l.push(format!(
            "if ({s} < 1 || {t} < 0 || {s} > ext{i} || ({s} > 1 && {t} > ext{i})) return {r};"
        ));
        l.push(format!("sz{i} *= {s}; if (sz{i} > ext{i}) return {r};"));
    }
    l.push(format!("if (sz{i} != ext{i}) return {r};"));
    if let Some((_, _, period)) = swizzle_of(d, a.elem)? {
        // the static rule on the bound extent: a swizzled offset must not leave it
        l.push(format!("if (ext{i} % {period}LL != 0) return {r};"));
    }
    l.push(format!(
        "unsigned char *seen{i} = (unsigned char *)calloc((size_t)ext{i}, 1); if (!seen{i}) return 3;"
    ));
    l.push(format!(
        "for (long long q{i} = 0; q{i} < ext{i}; ++q{i}) {{"
    ));
    l.extend(offset_lines(
        k,
        d,
        a.elem,
        (&format!("q{i}"), &format!("t{i}"), &format!("po{i}")),
    )?);
    l.push(format!(
        "if (po{i} >= ext{i} || seen{i}[po{i}]) {{ free(seen{i}); return {r}; }} seen{i}[po{i}] = 1;"
    ));
    l.push("}".to_string());
    l.push(format!("free(seen{i}); }}"));
    Ok(l)
}

/// Whether `b` holds a statement that reaches an allocation by its storage, not through `address` (a matrix or mma fragment move, a
/// call that passes the array): such a statement would bypass the layout, so a kernel with a `dyn_layout` holding one is refused.
fn bypasses_the_layout(b: &[Stmt]) -> bool {
    b.iter().any(|s| match s {
        Stmt::Mma { .. }
        | Stmt::MatrixLoad { .. }
        | Stmt::MatrixStore { .. }
        | Stmt::Call { .. }
        | Stmt::CallTuple { .. } => true,
        Stmt::If { then, els, .. } => bypasses_the_layout(&then.0) || bypasses_the_layout(&els.0),
        Stmt::For { body, .. } | Stmt::While { body, .. } => bypasses_the_layout(&body.0),
        _ => false,
    })
}

/// What this emitter cannot spell about a `dyn_layout`, refused before any text is written (a no-op for a kernel with none).
pub(crate) fn admit(k: &KernelFn, in_module: bool, single_phase: bool) -> Result<(), LowerError> {
    let Some(first) = k
        .allocs
        .iter()
        .position(|a| a.dyn_layout.is_some() || a.layout.is_some())
    else {
        return Ok(());
    };
    let name = &k.allocs[first].name;
    if in_module {
        return Err(refuse(format!(
            "allocation `{name}` states a `layout` or `dyn_layout` in a callee"
        )));
    }
    if !single_phase {
        return Err(refuse(format!(
            "allocation `{name}` states a `layout` or `dyn_layout` in a kernel with a Device barrier, a peer access or several devices"
        )));
    }
    if bypasses_the_layout(&k.body.0) {
        return Err(refuse(format!(
            "allocation `{name}` states a `layout` or `dyn_layout` in a kernel that moves fragments or calls a callee (they address the storage, not the layout)"
        )));
    }
    for a in k.allocs.iter().filter_map(|a| a.dyn_layout.as_ref()) {
        // every parameter leaf must name a plain scalar parameter (a struct field is not spelled)
        flat(k, a)?;
    }
    Ok(())
}
