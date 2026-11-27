// SPDX-License-Identifier: MIT OR Apache-2.0
//! The double-precision NVIDIA `mma.sync.aligned` atoms `m16n8k4`, `m16n8k8` and `m16n8k16` as data (ROADMAP U7 theme 3):
//! f64 inputs and an f64 accumulator. Additive: constructors only (no new statement, flag or element kind). The `m8n8k4`
//! f64 atom is in `mma_sm80.rs`.
//!
//! **Sources.** Public PTX ISA 9.5 (text of the PDF, outside every repo): `m16n8k4` section 9.7.16.5.6 (pp. 549-554;
//! Figures 69, 71, 73), `m16n8k8` section 9.7.16.5.7 (pp. 557-562; Figures 76, 79, 81), `m16n8k16` section 9.7.16.5.8
//! (A pp. 564-565, Figure 83; B p. 566, Figure 85; C p. 567, Figure 86). The k16 A column formula is split across a page
//! break in the extracted text (`(i * 2) + threadID_in_group` for even `i`, `(i * 2) - 2 + threadID_in_group` for odd `i`
//! as read), which is `4 * (i / 2) + threadID_in_group`: a bijection, and the same as the CUTLASS layout. The even/odd
//! pairing of the two branches is this lane's reading of interleaved text, not a quotation. Second source for all three: CUTLASS
//! `include/cute/atom/mma_traits_sm90.hpp` (`SM90_16x8x{4,8,16}_F64F64F64F64_TN`, `main` 5c179ede); a CuTe `BLayout`
//! indexes `(N, K)`, so B strides are recomputed. PTX gives these f64 shapes a `sm_90` target.

use unpopped_vocab::ElementKind::F64;
use unpopped_vocab::layout_algebra::Layout;

use crate::exec::Scope;
use crate::mma::{AtomSpec, MmaAtom, c_m16n8, flat};

fn atom(name: &str, k: u32, a: &Layout, b: &Layout) -> MmaAtom {
    MmaAtom::new(
        AtomSpec {
            name,
            mnk: (16, 8, k),
            dtypes: (F64, F64, F64),
            scope: Scope::Subgroup,
            lanes: 32,
        },
        [a, b, &c_m16n8()],
    )
    .expect("the f64 fragment layouts are bijections")
}

impl MmaAtom {
    /// `mma.sync.aligned.m16n8k4` f64 (CuTe `SM90_16x8x4_F64F64F64F64_TN`): A is rows `gid`, `gid + 8`, column `tig`.
    #[must_use]
    pub fn nvidia_m16n8k4_f64_f64() -> Self {
        let a = flat(&[(4, 16), (8, 1), (2, 8)]);
        let b = flat(&[(4, 1), (8, 4)]);
        atom("nvidia.mma.sync.m16n8k4.f64.f64", 4, &a, &b)
    }

    /// `mma.sync.aligned.m16n8k8` f64 (CuTe `SM90_16x8x8_F64F64F64F64_TN`).
    #[must_use]
    pub fn nvidia_m16n8k8_f64_f64() -> Self {
        let a = flat(&[(4, 16), (8, 1), (2, 8), (2, 64)]);
        let b = flat(&[(4, 1), (8, 8), (2, 4)]);
        atom("nvidia.mma.sync.m16n8k8.f64.f64", 8, &a, &b)
    }

    /// `mma.sync.aligned.m16n8k16` f64 (CuTe `SM90_16x8x16_F64F64F64F64_TN`); A, B and C checked against PTX Figures 83, 85, 86.
    #[must_use]
    pub fn nvidia_m16n8k16_f64_f64() -> Self {
        let a = flat(&[(4, 16), (8, 1), (2, 8), (4, 64)]);
        let b = flat(&[(4, 1), (8, 16), (4, 4)]);
        atom("nvidia.mma.sync.m16n8k16.f64.f64", 16, &a, &b)
    }
}
