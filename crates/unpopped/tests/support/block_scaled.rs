// SPDX-License-Identifier: MIT OR Apache-2.0
//! A block-scaled GEMM whose scale factors sit in the 512-byte basic-block layout (CUTLASS survey row 22). The scale-factor
//! atom holds the scales of 128 rows by 4 scale columns (each scale covers 16 elements of `K`): row `r` and scale column `c`
//! live at offset `(r % 32) * 16 + (r / 32) * 4 + c`. This is `Sm1xxBlockScaledConfig`'s `SfKMajorAtom`,
//! `((32, 4), (SFVecSize, 4)) : ((16, 4), (0, 1))`, recalled from `cutlass/detail/blockscaled_layout.hpp` and UNVERIFIED
//! against a checkout; only a SINGLE atom (128 rows, 4 scale columns, `K = 64`) is covered, not the tiling of atoms.
//!
//! `out[m][n] = sum_kb sfa[m][kb] * sfb[n][kb] * sum_{j < 16} a[m][16 kb + j] * b[n][16 kb + j]`, the scales read through the atom
//! offset by index arithmetic in the kernel. `F32` data and power-of-two scales, so the comparison is exact.

use unpopped::exec::{ExecId, LaunchShape};
use unpopped::ir::{CastSpec, RoundMode, Saturation, ScalarExpr};
use unpopped::kernel::{Block, BufferRole, KernelFn, LocalDecl, Param, Place, Stmt, Unroll};
use unpopped_vocab::ElementKind;

pub(crate) const F32: ElementKind = ElementKind::F32;
pub(crate) const I32: ElementKind = ElementKind::I32;
/// Rows of `A` (a full atom), the columns of `B` used, the contraction, the scale vector width, the scale columns.
pub(crate) const M: usize = 128;
pub(crate) const N: usize = 4;
pub(crate) const K: usize = 64;
pub(crate) const VEC: usize = 16;
pub(crate) const SFC: usize = K / VEC;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum Mutant {
    None,
    /// The row factors take each other's strides: `(r % 32) * 4 + (r / 32) * 16 + c`.
    SwappedRowFactors,
    /// The scale column stride is 4 and the row-block stride is 1 (a column-major atom).
    ColumnMajorAtom,
}

fn ci(v: usize) -> ScalarExpr {
    ScalarExpr::ConstInt(v as i64, I32)
}
fn l(i: u32) -> ScalarExpr {
    ScalarExpr::Local(i)
}
fn add(a: ScalarExpr, b: ScalarExpr) -> ScalarExpr {
    ScalarExpr::Add(Box::new(a), Box::new(b))
}
fn sub(a: ScalarExpr, b: ScalarExpr) -> ScalarExpr {
    ScalarExpr::Sub(Box::new(a), Box::new(b))
}
fn mul(a: ScalarExpr, b: ScalarExpr) -> ScalarExpr {
    ScalarExpr::Mul(Box::new(a), Box::new(b))
}
fn div(a: ScalarExpr, b: ScalarExpr) -> ScalarExpr {
    ScalarExpr::Div(Box::new(a), Box::new(b))
}
fn exec_i32(id: ExecId) -> ScalarExpr {
    ScalarExpr::Cast(
        Box::new(ScalarExpr::Exec(id)),
        CastSpec {
            to: I32,
            round: RoundMode::NearestEven,
            sat: Saturation::Wrap,
        },
    )
}
fn flat(buffer: u32, i: ScalarExpr) -> Place {
    Place::Logical {
        buffer,
        coords: vec![i],
    }
}
fn load(local: u32, buffer: u32, i: ScalarExpr) -> Stmt {
    Stmt::Load {
        local,
        place: flat(buffer, i),
    }
}
fn counted(local: u32, end: usize, body: Vec<Stmt>) -> Stmt {
    Stmt::For {
        unroll: Unroll::None,
        local,
        start: ci(0),
        end: ci(end),
        step: ci(1),
        body: Block(body),
    }
}

/// The offset of scale `(row, col)` in the atom, as the mutant reads it.
fn atom_offset(row: &ScalarExpr, col: ScalarExpr, mutant: Mutant) -> ScalarExpr {
    let (lo, hi) = (rem32(row), div(row.clone(), ci(32)));
    match mutant {
        Mutant::SwappedRowFactors => add(add(mul(lo, ci(4)), mul(hi, ci(16))), col),
        Mutant::ColumnMajorAtom => add(add(mul(lo, ci(4)), hi), mul(col, ci(128))),
        Mutant::None => add(add(mul(lo, ci(16)), mul(hi, ci(4))), col),
    }
}

fn rem32(x: &ScalarExpr) -> ScalarExpr {
    sub(x.clone(), mul(div(x.clone(), ci(32)), ci(32)))
}

/// One scale block: the 16-element dot of local 3 (locals 4, 5 the operands), then the two atom scales (locals 6, 7) applied.
fn scale_block(m: &ScalarExpr, n: &ScalarExpr, mutant: Mutant) -> Vec<Stmt> {
    let k_at = |row: &ScalarExpr| add(mul(row.clone(), ci(K)), add(mul(l(1), ci(VEC)), l(2)));
    let dot = counted(
        2,
        VEC,
        vec![
            load(4, 0, k_at(m)),
            load(5, 1, k_at(n)),
            Stmt::Assign {
                local: 3,
                value: add(l(3), mul(l(4), l(5))),
            },
        ],
    );
    let block = vec![
        Stmt::Let {
            local: 3,
            init: ScalarExpr::Const(0.0),
        },
        dot,
        load(6, 2, atom_offset(m, l(1), mutant)),
        load(7, 3, atom_offset(n, l(1), mutant)),
        Stmt::Assign {
            local: 0,
            value: add(l(0), mul(mul(l(6), l(7)), l(3))),
        },
    ];
    block
}

/// The buffer parameters.
fn scale_params() -> Vec<Param> {
    ["a", "b", "sfa", "sfb", "out"]
        .iter()
        .enumerate()
        .map(|(i, name)| {
            let role = if i == 4 {
                BufferRole::ReadWrite
            } else {
                BufferRole::Read
            };
            Param::buffer(*name, F32, role, Some(1))
        })
        .collect()
}

/// Buffers: 0 `a` `[M*K]`, 1 `b` `[N*K]`, 2 `sfa` `[512]` (the atom), 3 `sfb` `[512]` (read), 4 `out` `[M*N]` (write). Thread
/// `(x, y)` is output `(n, m)`. Locals: 0 `acc`, 1 `kb`, 2 `j`, 3 `inner`, 4 `x`, 5 `y`, 6 `sa`, 7 `sb`.
pub(crate) fn block_scaled_gemm(mutant: Mutant) -> KernelFn {
    let (n, m) = (
        exec_i32(ExecId::ThreadIdx(0)),
        exec_i32(ExecId::ThreadIdx(1)),
    );
    let block = scale_block(&m, &n, mutant);
    let body = vec![
        Stmt::Let {
            local: 0,
            init: ScalarExpr::Const(0.0),
        },
        counted(1, SFC, block),
        Stmt::Store {
            place: flat(4, add(mul(m, ci(N)), n)),
            value: l(0),
        },
    ];
    let mut locals = vec![LocalDecl::scalar(F32); 8];
    locals[1] = LocalDecl::scalar(I32);
    locals[2] = LocalDecl::scalar(I32);
    KernelFn::new(
        "block_scaled_gemm",
        F32,
        scale_params(),
        locals,
        Block(body),
    )
    .with_launch(LaunchShape::new([N as u32, M as u32, 1], [1, 1, 1]))
}
