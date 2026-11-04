// SPDX-License-Identifier: MIT OR Apache-2.0
//! **Matrix-multiply-accumulate atoms as data** (ROADMAP U7 S2; `docs/options-u7-mma-atoms.md`, layer 2).
//!
//! An [`MmaAtom`] describes one collective `D = A.B + C` that a group of lanes computes together: the logical
//! `M x N x K`, the operand and accumulator types, the lane count, and for each of A, B and C/D the **thread-value
//! layout**: where in the logical matrix each (lane, value) fragment element lives. The atom is data, not an enum
//! of named hardware ops, so a vendor is a constructor and never touches [`crate::kernel::Stmt`].
//!
//! # Conventions
//!
//! - The logical matrices are column-major: `A` is `M x K` (element `(r, c)` has id `r + M*c`), `B` is `K x N`
//!   (`k + K*n`), `C` and `D` are `M x N` (`m + M*n`).
//! - A thread-value layout maps the linear index `lane + lanes * value` (colexicographic over its shape, lane
//!   fastest, exactly [`Layout::call`]) to that id. A well-formed atom's layout is a **bijection** from
//!   `lanes * values` fragment elements onto `0..rows*cols`: every logical element lives in exactly one slot of
//!   exactly one lane. [`MmaAtom::new`] checks it, so a wrong lane layout (the likeliest defect in an atom
//!   definition) is refused at construction and nothing downstream can run it.
//! - The layouts are stored as flat `(shape, stride)` mode lists so the atom serializes with the kernel; the
//!   nesting of a [`Layout`] does not change what [`Layout::call`] returns.
//!
//! Shipped: the PTX `mma.sync.aligned` atoms `m16n8k16` (f16 and bf16 into f32) and `m16n8k32` (s8 into s32). The
//! `m16n8k8` tf32 atom waits for a tf32 operand notion: there is no tf32 `ElementKind`, and "the operands are
//! rounded to tf32 first" belongs to the compute rule, not to the layout.

use serde::{Deserialize, Serialize};
use unpopped_vocab::ElementKind;
use unpopped_vocab::layout_algebra::Layout;

use crate::exec::Scope;

/// Why an [`MmaAtom`] was refused.
#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum MmaAtomError {
    /// A dimension or the lane count is zero.
    ZeroDimension,
    /// A scope other than `Subgroup`: the register-fragment atoms are warp-level.
    BadScope,
    /// A layout is not valid, or its size is not the logical matrix size.
    BadLayout(&'static str),
    /// A layout maps two fragment elements to one logical id, or past the matrix.
    NotABijection(&'static str),
}

/// One flat `(shape, stride)` mode list: a [`Layout`] in a serializable form.
type Modes = Vec<(i64, i64)>;

/// What an atom is, apart from its fragment layouts: the arguments of [`MmaAtom::new`] as one record.
#[derive(Clone, Copy, Debug)]
pub struct AtomSpec<'a> {
    /// The atom's name, for diagnostics.
    pub name: &'a str,
    /// The logical `(M, N, K)`.
    pub mnk: (u32, u32, u32),
    /// The element types of A, B and of the accumulator (C and D).
    pub dtypes: (ElementKind, ElementKind, ElementKind),
    /// The group that takes part (the register-fragment atoms are `Subgroup`).
    pub scope: Scope,
    /// The number of lanes in the group.
    pub lanes: u32,
}

/// A matrix-multiply-accumulate atom: see the [module docs](self).
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub struct MmaAtom {
    name: String,
    m: u32,
    n: u32,
    k: u32,
    #[serde(with = "crate::text::ek")]
    a: ElementKind,
    #[serde(with = "crate::text::ek")]
    b: ElementKind,
    #[serde(with = "crate::text::ek")]
    acc: ElementKind,
    scope: Scope,
    lanes: u32,
    layouts: [Modes; 3],
}

impl MmaAtom {
    /// Build an atom after checking its layouts. `layouts` are the thread-value layouts of A, B and C/D, each
    /// over `lanes` lanes (the fastest modes) and a number of values per lane.
    ///
    /// # Errors
    /// [`MmaAtomError`] for a zero dimension, a scope other than `Subgroup`, an invalid layout, or a layout that
    /// is not a bijection onto its logical matrix.
    pub fn new(spec: AtomSpec<'_>, layouts: [&Layout; 3]) -> Result<Self, MmaAtomError> {
        let AtomSpec {
            name,
            mnk: (m, n, k),
            dtypes: (a, b, acc),
            scope,
            lanes,
        } = spec;
        if m == 0 || n == 0 || k == 0 || lanes == 0 {
            return Err(MmaAtomError::ZeroDimension);
        }
        if scope != Scope::Subgroup {
            return Err(MmaAtomError::BadScope);
        }
        let atom = Self {
            name: name.to_owned(),
            m,
            n,
            k,
            a,
            b,
            acc,
            scope,
            lanes,
            layouts: layouts.map(Layout::flat_modes),
        };
        for (i, what) in ["A", "B", "C"].into_iter().enumerate() {
            atom.check_bijection(i, what)?;
        }
        Ok(atom)
    }

    /// The atom's name, for diagnostics.
    #[must_use]
    pub fn name(&self) -> &str {
        &self.name
    }

    /// The logical `(M, N, K)`.
    #[must_use]
    pub const fn mnk(&self) -> (u32, u32, u32) {
        (self.m, self.n, self.k)
    }

    /// The element types of A, B and of the accumulator (C and D).
    #[must_use]
    pub const fn dtypes(&self) -> (ElementKind, ElementKind, ElementKind) {
        (self.a, self.b, self.acc)
    }

    /// The group that takes part.
    #[must_use]
    pub const fn scope(&self) -> Scope {
        self.scope
    }

    /// The number of lanes in the group.
    #[must_use]
    pub const fn lanes(&self) -> u32 {
        self.lanes
    }

    /// The rows and columns of operand `which` (0 = A, 1 = B, 2 = C/D) as its column-major logical matrix.
    const fn dims(&self, which: usize) -> (u32, u32) {
        match which {
            0 => (self.m, self.k),
            1 => (self.k, self.n),
            _ => (self.m, self.n),
        }
    }

    /// Fragment elements per lane for operand `which` (0 = A, 1 = B, 2 = C/D).
    #[must_use]
    pub const fn values(&self, which: usize) -> u32 {
        let (r, c) = self.dims(which);
        (r * c) / self.lanes
    }

    /// The logical column-major id of fragment element `value` of `lane` in operand `which` (0 = A, 1 = B,
    /// 2 = C/D), or `None` when `which`, `lane` or `value` is out of range.
    #[must_use]
    pub fn id(&self, which: usize, lane: u32, value: u32) -> Option<u32> {
        if which > 2 || lane >= self.lanes || value >= self.values(which) {
            return None;
        }
        let layout = self.layout(which).ok()?;
        let idx = i64::from(lane) + i64::from(self.lanes) * i64::from(value);
        u32::try_from(layout.call(idx).ok()?).ok()
    }

    fn layout(&self, which: usize) -> Result<Layout, MmaAtomError> {
        let leaves: Result<Vec<Layout>, _> = self.layouts[which]
            .iter()
            .map(|&(s, d)| Layout::leaf(s, d))
            .collect();
        leaves
            .ok()
            .and_then(|l| Layout::from_modes(&l).ok())
            .ok_or(MmaAtomError::BadLayout("a layout that does not rebuild"))
    }

    fn check_bijection(&self, which: usize, what: &'static str) -> Result<(), MmaAtomError> {
        let (r, c) = self.dims(which);
        let total = u64::from(r) * u64::from(c);
        if total % u64::from(self.lanes) != 0 {
            return Err(MmaAtomError::BadLayout(
                "a matrix whose size is not a multiple of the lane count",
            ));
        }
        let layout = self.layout(which)?;
        if u64::try_from(layout.size()).ok() != Some(total) {
            return Err(MmaAtomError::BadLayout(
                "a layout whose size is not the matrix size",
            ));
        }
        let mut seen = vec![false; total as usize];
        for idx in 0..total {
            let id = layout
                .call(idx as i64)
                .ok()
                .and_then(|v| usize::try_from(v).ok())
                .filter(|v| *v < seen.len())
                .ok_or(MmaAtomError::NotABijection(what))?;
            if std::mem::replace(&mut seen[id], true) {
                return Err(MmaAtomError::NotABijection(what));
            }
        }
        Ok(())
    }
}

fn flat(modes: &[(i64, i64)]) -> Layout {
    let leaves: Vec<Layout> = modes
        .iter()
        .map(|&(s, d)| Layout::leaf(s, d).expect("a constant mode is valid"))
        .collect();
    Layout::from_modes(&leaves).expect("a constant layout is valid")
}

/// The C/D layout of every `m16n8` atom: `(lane = tig + 4*gid, value = b0 + 2*b1)` holds row `gid + 8*b1`,
/// column `2*tig + b0` of a `16 x 8` column-major matrix (`tig` = lane % 4, `gid` = lane / 4).
fn c_m16n8() -> Layout {
    flat(&[(4, 32), (8, 1), (2, 16), (2, 8)])
}

impl MmaAtom {
    /// PTX `mma.sync.aligned.m16n8k16` with f16 inputs and an f32 accumulator.
    #[must_use]
    pub fn nvidia_m16n8k16_f16_f32() -> Self {
        Self::m16n8k16(ElementKind::F16, "nvidia.mma.sync.m16n8k16.f16.f32")
    }

    /// PTX `mma.sync.aligned.m16n8k16` with bf16 inputs and an f32 accumulator.
    #[must_use]
    pub fn nvidia_m16n8k16_bf16_f32() -> Self {
        Self::m16n8k16(ElementKind::Bf16, "nvidia.mma.sync.m16n8k16.bf16.f32")
    }

    fn m16n8k16(input: ElementKind, name: &str) -> Self {
        let a = flat(&[(4, 32), (8, 1), (2, 16), (2, 8), (2, 128)]);
        let b = flat(&[(4, 2), (8, 16), (2, 1), (2, 8)]);
        Self::new(
            AtomSpec {
                name,
                mnk: (16, 8, 16),
                dtypes: (input, input, ElementKind::F32),
                scope: Scope::Subgroup,
                lanes: 32,
            },
            [&a, &b, &c_m16n8()],
        )
        .expect("the PTX m16n8k16 fragment layouts are bijections")
    }

    /// PTX `mma.sync.aligned.m16n8k32` with s8 inputs and an s32 accumulator.
    #[must_use]
    pub fn nvidia_m16n8k32_s8_s32() -> Self {
        let a = flat(&[(4, 64), (8, 1), (4, 16), (2, 8), (2, 256)]);
        let b = flat(&[(4, 4), (8, 32), (4, 1), (2, 16)]);
        Self::new(
            AtomSpec {
                name: "nvidia.mma.sync.m16n8k32.s8.s32",
                mnk: (16, 8, 32),
                dtypes: (ElementKind::I8, ElementKind::I8, ElementKind::I32),
                scope: Scope::Subgroup,
                lanes: 32,
            },
            [&a, &b, &c_m16n8()],
        )
        .expect("the PTX m16n8k32 fragment layouts are bijections")
    }
}
