// SPDX-License-Identifier: MIT OR Apache-2.0
//! Hierarchical layouts (`shape : stride`) and their algebra — pure vocabulary.
//!
//! A [`Layout`] is a function from a logical index to an integer offset, written
//! `shape : stride` with both sides an [`IntTuple`] (an integer, or a nested tuple
//! of them) of the same profile. `(2,(3,4)):(1,(2,6))` is a layout of size 24 whose
//! second mode is itself a 3x4 layout. This is the value the CuTe layout algebra
//! (NVIDIA CUTLASS `media/docs/cpp/cute/01_layout.md`, `02_layout_algebra.md`)
//! operates on; the operations here follow that documentation, and every one of
//! them is checked against a brute-force reading of its definition by
//! `tests/layout_algebra_properties.rs`, not against CuTe itself.
//!
//! **Pure by construction.** No dependency on the IR, an op definition, a
//! frontend or a backend; no `unsafe`; no I/O. All arithmetic is checked: an
//! operation that cannot be carried out returns a [`LayoutError`], never a wrong
//! layout and never a panic.
//!
//! **What an index means.** An integer index into a layout is decomposed
//! colexicographically (first mode fastest) over the *flattened* shape. As in
//! CuTe, the last flattened mode is unbounded: [`Layout::call`] accepts an index
//! at or beyond [`Layout::size`] and extrapolates through that mode. Composition
//! relies on this.
//!
//! **Domain restrictions (this module's, not CuTe's).** Shape leaves are `>= 1`
//! and stride leaves are `>= 0`: a negative stride is declined
//! ([`LayoutError::NegativeStride`]) rather than half-supported, because
//! [`Layout::complement`] orders modes by stride. Nesting depth is capped at
//! [`MAX_DEPTH`].
//!
//! Not here yet: TMA basis strides.

use std::fmt;

mod slice;
mod swizzle;

pub use slice::SliceCrd;
pub use swizzle::{ComposedLayout, Swizzle};

/// Deepest nesting an [`IntTuple`] may have inside a [`Layout`]. A bare integer
/// has depth 0, `(a,b)` depth 1. Bounds recursion on untrusted input; real CuTe
/// layouts are far shallower.
pub const MAX_DEPTH: usize = 16;

/// Why a layout operation could not be carried out.
#[derive(Clone, Debug, Eq, PartialEq, Hash)]
#[non_exhaustive]
pub enum LayoutError {
    /// Shape and stride do not have the same profile (nesting structure).
    NotCongruent,
    /// A tuple with no elements.
    EmptyTuple,
    /// A shape leaf was `< 1`.
    NonPositiveShape(i64),
    /// A stride leaf was negative; strides here are `>= 0`.
    NegativeStride(i64),
    /// Nesting deeper than [`MAX_DEPTH`].
    TooDeep,
    /// An integer computation overflowed `i64`.
    Overflow,
    /// A coordinate or index was out of range or negative.
    OutOfRange,
    /// A coordinate's profile does not fit the shape it addresses.
    ProfileMismatch,
    /// Composition's divisibility condition failed: the left layout's modes do
    /// not divide evenly by the right layout's stride or shape.
    NotDivisible,
    /// Complement of a layout with a size-greater-than-1, stride-0 mode, which
    /// maps many indices to one offset.
    ZeroStrideMode,
    /// Complement of a layout whose modes overlap or whose strides are not
    /// successive multiples (each stride must be a multiple of the extent
    /// covered by the smaller strides).
    NotComplementable,
    /// A cotarget (or a tiler/product size) that is not positive.
    NonPositiveCotarget(i64),
    /// A swizzle parameter set that is not a bijection (`|shift| < bits`) or
    /// would not fit in `i64`.
    BadSwizzle,
}

impl fmt::Display for LayoutError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NotCongruent => write!(f, "shape and stride have different profiles"),
            Self::EmptyTuple => write!(f, "empty tuple"),
            Self::NonPositiveShape(v) => write!(f, "shape leaf {v} is not >= 1"),
            Self::NegativeStride(v) => write!(f, "stride leaf {v} is negative"),
            Self::TooDeep => write!(f, "nesting deeper than MAX_DEPTH ({MAX_DEPTH})"),
            Self::Overflow => write!(f, "integer overflow"),
            Self::OutOfRange => write!(f, "coordinate or index out of range"),
            Self::ProfileMismatch => write!(f, "coordinate profile does not fit the shape"),
            Self::NotDivisible => write!(f, "composition divisibility condition failed"),
            Self::ZeroStrideMode => write!(f, "complement of a stride-0 mode of extent > 1"),
            Self::NotComplementable => write!(f, "layout modes overlap or are not stride-nested"),
            Self::NonPositiveCotarget(v) => write!(f, "cotarget {v} is not >= 1"),
            Self::BadSwizzle => write!(f, "swizzle parameters are not a valid bijection"),
        }
    }
}

impl std::error::Error for LayoutError {}

/// An integer, or a nested tuple of them: the type of a shape, a stride, or a
/// coordinate. Tuples are never empty inside a [`Layout`].
#[derive(Clone, Debug, Eq, PartialEq, Hash)]
pub enum IntTuple {
    /// A leaf integer.
    Int(i64),
    /// A tuple of sub-tuples.
    Tuple(Vec<IntTuple>),
}

impl From<i64> for IntTuple {
    fn from(v: i64) -> Self {
        Self::Int(v)
    }
}

impl IntTuple {
    /// A tuple of the given elements.
    pub fn tuple(items: impl IntoIterator<Item = IntTuple>) -> Self {
        Self::Tuple(items.into_iter().collect())
    }

    /// A flat tuple of integers.
    pub fn ints(items: impl IntoIterator<Item = i64>) -> Self {
        Self::Tuple(items.into_iter().map(Self::Int).collect())
    }

    /// Number of top-level elements; a bare integer has rank 1.
    pub fn rank(&self) -> usize {
        match self {
            Self::Int(_) => 1,
            Self::Tuple(t) => t.len(),
        }
    }

    /// Nesting depth: 0 for an integer, `1 + max(child depth)` for a tuple.
    pub fn depth(&self) -> usize {
        match self {
            Self::Int(_) => 0,
            Self::Tuple(t) => 1 + t.iter().map(Self::depth).max().unwrap_or(0),
        }
    }

    /// The leaf integers in order (the flattening).
    pub fn leaves(&self) -> Vec<i64> {
        let mut out = Vec::new();
        self.collect_leaves(&mut out);
        out
    }

    fn collect_leaves(&self, out: &mut Vec<i64>) {
        match self {
            Self::Int(v) => out.push(*v),
            Self::Tuple(t) => t.iter().for_each(|c| c.collect_leaves(out)),
        }
    }

    /// Whether `other` has exactly this nesting structure.
    pub fn congruent(&self, other: &IntTuple) -> bool {
        match (self, other) {
            (Self::Int(_), Self::Int(_)) => true,
            (Self::Tuple(a), Self::Tuple(b)) => {
                a.len() == b.len() && a.iter().zip(b).all(|(x, y)| x.congruent(y))
            }
            _ => false,
        }
    }

    /// Product of all leaves, checked.
    pub fn product(&self) -> Result<i64, LayoutError> {
        self.leaves().into_iter().try_fold(1i64, |acc, v| {
            acc.checked_mul(v).ok_or(LayoutError::Overflow)
        })
    }

    /// The natural coordinate of linear index `idx` within this *shape*,
    /// congruent with it (first mode fastest). `idx` is not range-checked: the
    /// last leaf absorbs any excess.
    pub fn idx2crd(&self, idx: i64) -> Result<IntTuple, LayoutError> {
        if idx < 0 {
            return Err(LayoutError::OutOfRange);
        }
        match self {
            Self::Int(_) => Ok(Self::Int(idx)),
            Self::Tuple(t) => {
                let mut rest = idx;
                let mut out = Vec::with_capacity(t.len());
                for (k, sub) in t.iter().enumerate() {
                    if k + 1 == t.len() {
                        out.push(sub.idx2crd(rest)?);
                    } else {
                        let sz = sub.product()?;
                        out.push(sub.idx2crd(rest % sz)?);
                        rest /= sz;
                    }
                }
                Ok(Self::Tuple(out))
            }
        }
    }

    fn fmt_into(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Int(v) => write!(f, "{v}"),
            Self::Tuple(t) => {
                write!(f, "(")?;
                for (k, c) in t.iter().enumerate() {
                    if k > 0 {
                        write!(f, ",")?;
                    }
                    c.fmt_into(f)?;
                }
                write!(f, ")")
            }
        }
    }
}

impl fmt::Display for IntTuple {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.fmt_into(f)
    }
}

/// A hierarchical `shape : stride` layout. Fields are private so every value
/// has passed [`Layout::new`]'s checks.
#[derive(Clone, Debug, Eq, PartialEq, Hash)]
pub struct Layout {
    shape: IntTuple,
    stride: IntTuple,
}

impl fmt::Display for Layout {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.shape.fmt_into(f)?;
        write!(f, ":")?;
        self.stride.fmt_into(f)
    }
}

fn check_nonempty(t: &IntTuple) -> bool {
    match t {
        IntTuple::Int(_) => true,
        IntTuple::Tuple(v) => !v.is_empty() && v.iter().all(check_nonempty),
    }
}

fn node(mut modes: Vec<(IntTuple, IntTuple)>) -> (IntTuple, IntTuple) {
    if modes.len() == 1 {
        return modes.remove(0);
    }
    let (s, d): (Vec<_>, Vec<_>) = modes.into_iter().unzip();
    (IntTuple::Tuple(s), IntTuple::Tuple(d))
}

/// The `(shape, stride)` trees of a flat mode list; `1:0` when empty.
fn flat_node(modes: &[(i64, i64)]) -> (IntTuple, IntTuple) {
    if modes.is_empty() {
        return (IntTuple::Int(1), IntTuple::Int(0));
    }
    node(
        modes
            .iter()
            .map(|&(s, d)| (IntTuple::Int(s), IntTuple::Int(d)))
            .collect(),
    )
}

impl Layout {
    /// Validate and build a layout: congruent profiles, no empty tuples, depth
    /// at most [`MAX_DEPTH`], shape leaves `>= 1`, stride leaves `>= 0`, and
    /// [`Layout::size`] and [`Layout::cosize`] representable in `i64`.
    pub fn new(shape: IntTuple, stride: IntTuple) -> Result<Self, LayoutError> {
        if !shape.congruent(&stride) {
            return Err(LayoutError::NotCongruent);
        }
        if !check_nonempty(&shape) {
            return Err(LayoutError::EmptyTuple);
        }
        if shape.depth() > MAX_DEPTH {
            return Err(LayoutError::TooDeep);
        }
        let mut cosize: i64 = 1;
        for (s, d) in shape.leaves().into_iter().zip(stride.leaves()) {
            if s < 1 {
                return Err(LayoutError::NonPositiveShape(s));
            }
            if d < 0 {
                return Err(LayoutError::NegativeStride(d));
            }
            cosize = (s - 1)
                .checked_mul(d)
                .and_then(|x| cosize.checked_add(x))
                .ok_or(LayoutError::Overflow)?;
        }
        shape.product()?;
        Ok(Self { shape, stride })
    }

    /// The single-mode layout `s : d`.
    pub fn leaf(s: i64, d: i64) -> Result<Self, LayoutError> {
        Self::new(IntTuple::Int(s), IntTuple::Int(d))
    }

    /// Compact column-major layout over `shape`: stride 1 on the first flattened
    /// mode, each next stride the product of the extents before it.
    pub fn col_major(shape: IntTuple) -> Result<Self, LayoutError> {
        fn go(t: &IntTuple, run: &mut i64) -> Result<IntTuple, LayoutError> {
            match t {
                IntTuple::Int(s) => {
                    let d = *run;
                    *run = run.checked_mul(*s).ok_or(LayoutError::Overflow)?;
                    Ok(IntTuple::Int(d))
                }
                IntTuple::Tuple(v) => Ok(IntTuple::Tuple(
                    v.iter().map(|c| go(c, run)).collect::<Result<_, _>>()?,
                )),
            }
        }
        let mut run = 1;
        let stride = go(&shape, &mut run)?;
        Self::new(shape, stride)
    }

    /// The layout whose modes are `modes`, in order (CuTe `make_layout(A, B, ..)`).
    pub fn from_modes(modes: &[Layout]) -> Result<Self, LayoutError> {
        if modes.is_empty() {
            return Err(LayoutError::EmptyTuple);
        }
        let (s, d) = modes
            .iter()
            .map(|m| (m.shape.clone(), m.stride.clone()))
            .unzip::<_, _, Vec<_>, Vec<_>>();
        Self::new(IntTuple::Tuple(s), IntTuple::Tuple(d))
    }

    /// The shape.
    pub fn shape(&self) -> &IntTuple {
        &self.shape
    }

    /// The stride.
    pub fn stride(&self) -> &IntTuple {
        &self.stride
    }

    /// Number of top-level modes.
    pub fn rank(&self) -> usize {
        self.shape.rank()
    }

    /// Nesting depth of the shape.
    pub fn depth(&self) -> usize {
        self.shape.depth()
    }

    /// Product of all shape leaves: the number of valid indices.
    pub fn size(&self) -> i64 {
        // `new` proved this fits.
        self.shape.product().unwrap_or(i64::MAX)
    }

    /// One past the largest offset the layout can produce:
    /// `1 + sum((s-1)*d)` over flattened modes.
    pub fn cosize(&self) -> i64 {
        // `new` proved this fits.
        self.flat_modes().iter().fold(1i64, |acc, (s, d)| {
            acc.saturating_add((s - 1).saturating_mul(*d))
        })
    }

    /// The `(shape, stride)` leaf pairs in order.
    pub fn flat_modes(&self) -> Vec<(i64, i64)> {
        self.shape
            .leaves()
            .into_iter()
            .zip(self.stride.leaves())
            .collect()
    }

    /// The top-level modes as layouts (a single-mode layout is its own mode).
    pub fn modes(&self) -> Vec<Layout> {
        match (&self.shape, &self.stride) {
            (IntTuple::Tuple(s), IntTuple::Tuple(d)) => s
                .iter()
                .zip(d)
                .map(|(s, d)| Layout {
                    shape: s.clone(),
                    stride: d.clone(),
                })
                .collect(),
            _ => vec![self.clone()],
        }
    }

    fn from_flat(modes: &[(i64, i64)]) -> Layout {
        // Callers pass modes already known valid.
        let (shape, stride) = flat_node(modes);
        Layout { shape, stride }
    }

    /// The same layout with every nesting level removed.
    pub fn flatten(&self) -> Layout {
        Self::from_flat(&self.flat_modes())
    }

    /// The offset for linear index `idx` (colexicographic over the flattened
    /// shape). `idx >= size()` extrapolates through the last mode.
    pub fn call(&self, idx: i64) -> Result<i64, LayoutError> {
        if idx < 0 {
            return Err(LayoutError::OutOfRange);
        }
        let modes = self.flat_modes();
        let mut rest = idx;
        let mut off: i64 = 0;
        for (k, (s, d)) in modes.iter().enumerate() {
            let c = if k + 1 == modes.len() { rest } else { rest % s };
            off = c
                .checked_mul(*d)
                .and_then(|x| off.checked_add(x))
                .ok_or(LayoutError::Overflow)?;
            rest /= s;
        }
        Ok(off)
    }

    /// The offset for coordinate `crd`. An integer coordinate against a tuple
    /// shape is a linear index into that sub-layout; a tuple coordinate must
    /// match the shape's arity at every level. Leaves are range-checked.
    pub fn crd2idx(&self, crd: &IntTuple) -> Result<i64, LayoutError> {
        fn go(crd: &IntTuple, shape: &IntTuple, stride: &IntTuple) -> Result<i64, LayoutError> {
            match (crd, shape, stride) {
                (IntTuple::Int(c), IntTuple::Int(s), IntTuple::Int(d)) => {
                    if *c < 0 || c >= s {
                        return Err(LayoutError::OutOfRange);
                    }
                    c.checked_mul(*d).ok_or(LayoutError::Overflow)
                }
                (IntTuple::Int(c), _, _) => {
                    let sub = Layout {
                        shape: shape.clone(),
                        stride: stride.clone(),
                    };
                    if *c < 0 || *c >= sub.size() {
                        return Err(LayoutError::OutOfRange);
                    }
                    sub.call(*c)
                }
                (IntTuple::Tuple(cs), IntTuple::Tuple(ss), IntTuple::Tuple(ds))
                    if cs.len() == ss.len() =>
                {
                    let mut off: i64 = 0;
                    for ((c, s), d) in cs.iter().zip(ss).zip(ds) {
                        off = off.checked_add(go(c, s, d)?).ok_or(LayoutError::Overflow)?;
                    }
                    Ok(off)
                }
                _ => Err(LayoutError::ProfileMismatch),
            }
        }
        go(crd, &self.shape, &self.stride)
    }

    /// Merge adjacent flattened modes that are contiguous and drop extent-1
    /// modes. Same function (`coalesce(L)(i) == L(i)` for every `i`), same size,
    /// depth at most 1. A layout of size 1 becomes `1:0`.
    pub fn coalesce(&self) -> Layout {
        let mut out: Vec<(i64, i64)> = Vec::new();
        for (s, d) in self.flat_modes() {
            if s == 1 {
                continue;
            }
            match out.last_mut() {
                Some((ps, pd)) if ps.checked_mul(*pd) == Some(d) => {
                    // `size()` was proved to fit, so the merged extent does too.
                    *ps *= s;
                }
                _ => out.push((s, d)),
            }
        }
        Self::from_flat(&out)
    }

    /// `self ∘ rhs`: the layout `C` with `C(i) == self(rhs(i))` for every
    /// `i < size(rhs)`, with `rhs`'s profile. `self` is read as a function of its
    /// linear index (its own hierarchy is ignored). Fails with
    /// [`LayoutError::NotDivisible`] when CuTe's divisibility condition is not
    /// met; that condition is sufficient, not necessary, so some expressible
    /// compositions are declined (`(4,2):(1,4) ∘ 2:3` is one).
    pub fn composition(&self, rhs: &Layout) -> Result<Layout, LayoutError> {
        let a = self.flat_modes();
        let (shape, stride) = compose_tree(&a, &rhs.shape, &rhs.stride)?;
        Layout::new(shape, stride)
    }

    /// The layout `R` that fills in what `self` leaves out of `[0, cotarget)`.
    ///
    /// `R`'s strides increase, `(self, R)` is a bijection onto
    /// `[0, size(self)*size(R))`, and `size(self)*size(R) >= cotarget` whenever
    /// `cotarget` is at least the extent `self`'s modes span (otherwise `R` just
    /// fills the gaps below that extent). Extent-1 modes are ignored; a stride-0
    /// mode of larger extent, an overlap, or strides that are not successive
    /// multiples are declined.
    pub fn complement(&self, cotarget: i64) -> Result<Layout, LayoutError> {
        if cotarget < 1 {
            return Err(LayoutError::NonPositiveCotarget(cotarget));
        }
        let mut modes: Vec<(i64, i64)> = self
            .flat_modes()
            .into_iter()
            .filter(|&(s, _)| s > 1)
            .collect();
        if modes.iter().any(|&(_, d)| d == 0) {
            return Err(LayoutError::ZeroStrideMode);
        }
        modes.sort_by_key(|&(s, d)| (d, s));
        let mut cur: i64 = 1;
        let mut out: Vec<(i64, i64)> = Vec::new();
        for (s, d) in modes {
            if d % cur != 0 {
                return Err(LayoutError::NotComplementable);
            }
            let gap = d / cur;
            if gap > 1 {
                out.push((gap, cur));
            }
            cur = s.checked_mul(d).ok_or(LayoutError::Overflow)?;
        }
        let rest = cotarget / cur + i64::from(cotarget % cur != 0);
        if rest > 1 {
            out.push((rest, cur));
        }
        let r = Self::from_flat(&out);
        Layout::new(r.shape, r.stride)
    }

    /// `self / tiler`: `composition(self, (tiler, complement(tiler, size(self))))`.
    /// A rank-2 layout, mode 0 the elements of one tile, mode 1 the tiles.
    /// Exact when `tiler`'s tiling divides `size(self)`; otherwise the last
    /// tile extrapolates past `self` (the "round-up residue" CuTe masks).
    pub fn logical_divide(&self, tiler: &Layout) -> Result<Layout, LayoutError> {
        let rest = tiler.complement(self.size())?;
        self.composition(&Layout::from_modes(&[tiler.clone(), rest])?)
    }

    /// A layout `R` with `self(R(i)) == i` for every `i < size(R)`, where `size(R)`
    /// is as large as `self`'s modes allow: starting from offset 1, it chains the
    /// modes whose stride is exactly the extent covered so far. Offsets beyond
    /// `size(R)` are not claimed. For a layout that is a bijection onto
    /// `[0, size(self))` this is the two-sided inverse. `1:0` when no mode has
    /// stride 1.
    pub fn right_inverse(&self) -> Layout {
        let modes = self.flat_modes();
        let mut index_stride = Vec::with_capacity(modes.len());
        let mut run: i64 = 1;
        for &(s, _) in &modes {
            index_stride.push(run);
            run = run.saturating_mul(s);
        }
        let mut used = vec![false; modes.len()];
        let mut covered: i64 = 1;
        let mut out: Vec<(i64, i64)> = Vec::new();
        while let Some(k) =
            (0..modes.len()).find(|&k| !used[k] && modes[k].0 > 1 && modes[k].1 == covered)
        {
            used[k] = true;
            out.push((modes[k].0, index_stride[k]));
            match covered.checked_mul(modes[k].0) {
                Some(c) => covered = c,
                None => break,
            }
        }
        Self::from_flat(&out).coalesce()
    }

    /// For an injective `self`, a layout `L` with `L(self(i)) == i` for every
    /// `i < size(self)` and `size(L) >= cosize(self)`:
    /// `right_inverse((self, complement(self, cosize(self))))`. Declines what
    /// [`Layout::complement`] declines (stride-0 or overlapping modes).
    pub fn left_inverse(&self) -> Result<Layout, LayoutError> {
        let rest = self.complement(self.cosize())?;
        Ok(Layout::from_modes(&[self.clone(), rest])?.right_inverse())
    }

    /// `self x rhs`: `(self, composition(complement(self, size(self)*cosize(rhs)), rhs))`.
    /// A rank-2 layout: mode 0 is `self`, mode 1 places copies of it where `rhs`
    /// says, never overlapping `self`'s own image when both are injective.
    pub fn logical_product(&self, rhs: &Layout) -> Result<Layout, LayoutError> {
        let target = self
            .size()
            .checked_mul(rhs.cosize())
            .ok_or(LayoutError::Overflow)?;
        let rest = self.complement(target)?.composition(rhs)?;
        Layout::from_modes(&[self.clone(), rest])
    }
}

fn compose_tree(
    a: &[(i64, i64)],
    shape: &IntTuple,
    stride: &IntTuple,
) -> Result<(IntTuple, IntTuple), LayoutError> {
    match (shape, stride) {
        (IntTuple::Int(s), IntTuple::Int(d)) => compose_leaf(a, *s, *d),
        (IntTuple::Tuple(ss), IntTuple::Tuple(ds)) => {
            let mut s_out = Vec::with_capacity(ss.len());
            let mut d_out = Vec::with_capacity(ss.len());
            for (s, d) in ss.iter().zip(ds) {
                let (cs, cd) = compose_tree(a, s, d)?;
                s_out.push(cs);
                d_out.push(cd);
            }
            Ok((IntTuple::Tuple(s_out), IntTuple::Tuple(d_out)))
        }
        _ => Err(LayoutError::NotCongruent),
    }
}

/// `a ∘ (s : d)` for flattened `a`: walk `a`'s modes consuming the stride `d`
/// (skipping whole modes it jumps over) and then the extent `s`.
fn compose_leaf(a: &[(i64, i64)], s: i64, d: i64) -> Result<(IntTuple, IntTuple), LayoutError> {
    if s == 1 {
        return Ok((IntTuple::Int(1), IntTuple::Int(0)));
    }
    if d == 0 {
        return Ok((IntTuple::Int(s), IntTuple::Int(0)));
    }
    let mut rest_shape = s;
    let mut rest_stride = d;
    let mut out: Vec<(i64, i64)> = Vec::new();
    for (k, &(ak_s, ak_d)) in a.iter().enumerate() {
        if rest_shape == 1 {
            break;
        }
        let last = k + 1 == a.len();
        if !last && rest_stride >= ak_s {
            if rest_stride % ak_s != 0 {
                return Err(LayoutError::NotDivisible);
            }
            rest_stride /= ak_s;
            continue;
        }
        let take = if last {
            rest_shape
        } else {
            if ak_s % rest_stride != 0 {
                return Err(LayoutError::NotDivisible);
            }
            let avail = ak_s / rest_stride;
            let take = avail.min(rest_shape);
            if rest_shape % take != 0 {
                return Err(LayoutError::NotDivisible);
            }
            take
        };
        if take > 1 {
            out.push((
                take,
                ak_d.checked_mul(rest_stride).ok_or(LayoutError::Overflow)?,
            ));
        }
        rest_shape /= take;
        rest_stride = 1;
    }
    Ok(flat_node(&out))
}
