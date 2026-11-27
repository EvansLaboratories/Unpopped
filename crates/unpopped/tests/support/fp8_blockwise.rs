// SPDX-License-Identifier: MIT OR Apache-2.0
//! A blockwise-scaled FP8 GEMM at the kernel layer (CUTLASS survey row 24, the DeepSeek-V3 geometry): `A` is `[M, K]` and `B` is
//! `[N, K]` holding FP8-representable values (see below); the scale of `A` is per row and per 128-wide block of `K` (1x128), the scale of `B` is per 128 rows of `N`
//! and per 128-wide block of `K` (128x128), both `F32`. Per K-block the products are accumulated unscaled in `F32` and the
//! partial sum is then multiplied by the two scales and added to the running total (the promotion form):
//! `out[m][n] = sum_kb sfa[m][kb] * sfb[n / 128][kb] * sum_{j < 128} a[m][128 kb + j] * b[n][128 kb + j]`.
//!
//! The operands are `F32` buffers holding FP8-representable values, not FP8 buffers: the oracle does not compute an FP8 dtype in the
//! kernel layer (`check_kernel` answers `DtypeNotComputed { dtype: F8E3M4 }` for an `F8E3M4` buffer or local, and "a Cast endpoint
//! outside {...} is not defined yet" for a `Cast` from `Fp8E4M3FN`). That is a finding for the primary; this file exercises the
//! blockwise-scaling geometry (1x128 and 128x128, `F32` scales, promotion accumulation), which is what row 24 leaves open besides the dtype.
//!
//! The geometry follows the DeepSeek-V3 report's fine-grained quantization (arXiv:2412.19437, section 3.3.2) and CUTLASS's
//! blockwise-scaled GEMM example; UNVERIFIED against either's source. The hold on 4x-ratio matrix kernels does not apply (scalar
//! loops, no matrix atom); where the patent is cited: US12314717B2, method claim, "findings not advice", lifted by the PM
//! 2026-11-27 under board 519 B, not a clearance, no attorney.

use unpopped::exec::{ExecId, LaunchShape};
use unpopped::ir::{CastSpec, RoundMode, Saturation, ScalarExpr};
use unpopped::kernel::{Block, BufferRole, KernelFn, LocalDecl, Param, Place, Stmt, Unroll};
use unpopped_vocab::ElementKind;

pub(crate) const F32: ElementKind = ElementKind::F32;
pub(crate) const I32: ElementKind = ElementKind::I32;
pub(crate) const M: usize = 2;
pub(crate) const N: usize = 256;
pub(crate) const K: usize = 256;
pub(crate) const BLK: usize = 128;
pub(crate) const KB: usize = K / BLK;
pub(crate) const NB: usize = N / BLK;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum Mutant {
    None,
    /// The scales of K-block `1 - kb` are used (the blocks swapped).
    SwappedKBlock,
    /// `B`'s scale block is chosen by `n % 2` rather than `n / 128`.
    NBlockByParity,
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

/// The two scale loads of K-block local 1 for output `(m, n)` into locals 6 (`sfa`) and 7 (`sfb`), as the mutant reads them.
fn scale_loads(m: &ScalarExpr, n: &ScalarExpr, mutant: Mutant) -> Vec<Stmt> {
    let kb = if mutant == Mutant::SwappedKBlock {
        sub(ci(KB - 1), l(1))
    } else {
        l(1)
    };
    let n_block = if mutant == Mutant::NBlockByParity {
        sub(n.clone(), mul(div(n.clone(), ci(2)), ci(2)))
    } else {
        div(n.clone(), ci(BLK))
    };
    vec![
        load(6, 2, add(mul(m.clone(), ci(KB)), kb.clone())),
        load(7, 3, add(mul(n_block, ci(KB)), kb)),
    ]
}

/// The unscaled FP8 dot of one K-block into local 3 (locals 4 and 5 hold the two operands).
fn block_dot(m: &ScalarExpr, n: &ScalarExpr) -> Vec<Stmt> {
    let k_at = |row: &ScalarExpr| add(mul(row.clone(), ci(K)), add(mul(l(1), ci(BLK)), l(2)));
    vec![
        Stmt::Let {
            local: 3,
            init: ScalarExpr::Const(0.0),
        },
        counted(
            2,
            BLK,
            vec![
                load(4, 0, k_at(m)),
                load(5, 1, k_at(n)),
                Stmt::Assign {
                    local: 3,
                    value: add(l(3), mul(l(4), l(5))),
                },
            ],
        ),
    ]
}

/// Buffers: 0 `a` `[M*K]` and 1 `b` `[N*K]` (`F32`), 2 `sfa` `[M*KB]` and 3 `sfb` `[NB*KB]` (`F32`) (read), 4 `out` `[M*N]`
/// (write). Thread `(x, y)` is output `(n, m)`. Locals: 0 `acc`, 1 `kb`, 2 `j`, 3 `inner`, 4 and 5 the FP8 operands, 6 `sfa`, 7 `sfb`.
pub(crate) fn fp8_blockwise_gemm(mutant: Mutant) -> KernelFn {
    let (n, m) = (
        exec_i32(ExecId::ThreadIdx(0)),
        exec_i32(ExecId::ThreadIdx(1)),
    );
    let mut block = block_dot(&m, &n);
    block.extend(scale_loads(&m, &n, mutant));
    block.push(Stmt::Assign {
        local: 0,
        value: add(l(0), mul(mul(l(6), l(7)), l(3))),
    });
    let body = vec![
        Stmt::Let {
            local: 0,
            init: ScalarExpr::Const(0.0),
        },
        counted(1, KB, block),
        Stmt::Store {
            place: flat(4, add(mul(m, ci(N)), n)),
            value: l(0),
        },
    ];
    KernelFn::new("fp8_blockwise_gemm", F32, params(), locals(), Block(body))
        .with_launch(LaunchShape::new([N as u32, M as u32, 1], [1, 1, 1]))
}

fn params() -> Vec<Param> {
    let kinds = [F32; 5];
    ["a", "b", "sfa", "sfb", "out"]
        .iter()
        .zip(kinds)
        .enumerate()
        .map(|(i, (name, kind))| {
            let role = if i == 4 {
                BufferRole::ReadWrite
            } else {
                BufferRole::Read
            };
            Param::buffer(*name, kind, role, Some(1))
        })
        .collect()
}

fn locals() -> Vec<LocalDecl> {
    [F32, I32, I32, F32, F32, F32, F32, F32]
        .map(LocalDecl::scalar)
        .to_vec()
}
