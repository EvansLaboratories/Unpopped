// SPDX-License-Identifier: MIT OR Apache-2.0
//! The `Stmt::MatrixLoad` conformance kernel of ROADMAP U8b (`docs/options-u8-copy-atoms.md` S2), shared by the oracle
//! test (`kernel_matrix_load.rs`) and the C test (`unpopped-cpu-c/tests/kernel_matrix_load_emit_agrees.rs`, which keeps a
//! copy: change both). One warp stages `count` 8 x 8 tiles of 16-bit elements into `Shared`, loads them with `ldmatrix` semantics
//! and writes every lane's fragment out, so the output shows exactly which element each lane received.

use unpopped::exec::{AllocDecl, ExecId, Extent, LaunchShape, MemorySpace};
use unpopped::ir::{BinaryOp, ScalarExpr};
use unpopped::kernel::{Block, BufferRole, KernelFn, LocalDecl, Param, Place, Stmt};
use unpopped_vocab::ElementKind;

/// The element type: 16 bits, moved as bits (the data are bit patterns, never read as numbers). `F16` because the serial C
/// spells no 16-bit integer compute type.
pub(crate) const H16: ElementKind = ElementKind::F16;
const U32: ElementKind = ElementKind::U32;

fn u(v: i64) -> ScalarExpr {
    ScalarExpr::ConstInt(v, U32)
}
fn local(i: u32) -> ScalarExpr {
    ScalarExpr::Local(i)
}
fn add(a: ScalarExpr, b: ScalarExpr) -> ScalarExpr {
    ScalarExpr::Add(Box::new(a), Box::new(b))
}
fn mul(a: ScalarExpr, b: ScalarExpr) -> ScalarExpr {
    ScalarExpr::Mul(Box::new(a), Box::new(b))
}
fn lane() -> ScalarExpr {
    ScalarExpr::Exec(ExecId::ThreadIdx(0))
}
fn buf(buffer: u32, index: ScalarExpr) -> Place {
    Place::Logical {
        buffer,
        coords: vec![index],
    }
}
fn alloc(alloc: u32, index: ScalarExpr) -> Place {
    Place::Alloc { alloc, index }
}
fn counted(var: u32, end: i64, body: Vec<Stmt>) -> Stmt {
    Stmt::For {
        local: var,
        start: u(0),
        end: u(end),
        step: u(1),
        body: Block(body),
    }
}

/// Params: 0 `src` `[count * 64]` read, 1 `out` `[32 * 8]` read-write. Allocs: 0 `tiles` Shared `[count * 64]`, 1 `frag`
/// Private `[2 * count]`. Locals: 0 `x`, 1 `j`, 2 `v`. Lane `l` stages row `l` (8 elements) of the `count * 8` rows, loads, and
/// stores its `2 * count` fragment elements at `out[l * 8 ..]`. Without `barrier` the kernel also stages the row of its
/// mirrored lane (`8 * count - 1 - l`), so each lane reads a row another lane wrote with nothing between: a data race.
pub(crate) fn matrix_load_kernel(count: u32, trans: bool, barrier: bool) -> KernelFn {
    let staged = if barrier {
        lane()
    } else {
        ScalarExpr::Sub(Box::new(u(i64::from(8 * count) - 1)), Box::new(lane()))
    };
    let stage = Stmt::If {
        cond: ScalarExpr::Binary(
            BinaryOp::CmpLt,
            Box::new(lane()),
            Box::new(u(i64::from(8 * count))),
        ),
        then: Block(vec![counted(
            1,
            8,
            vec![
                Stmt::Load {
                    local: 0,
                    place: buf(0, add(mul(staged.clone(), u(8)), local(1))),
                },
                Stmt::Store {
                    place: alloc(0, add(mul(staged, u(8)), local(1))),
                    value: local(0),
                },
            ],
        )]),
        els: Block(vec![]),
    };
    let mut body = vec![stage];
    if barrier {
        body.push(Stmt::barrier(unpopped::exec::Scope::Block));
    }
    body.push(Stmt::MatrixLoad {
        count,
        trans,
        src: alloc(0, mul(lane(), u(8))),
        dst: 1,
    });
    body.push(counted(
        2,
        i64::from(2 * count),
        vec![
            Stmt::Load {
                local: 0,
                place: alloc(1, local(2)),
            },
            Stmt::Store {
                place: buf(1, add(mul(lane(), u(8)), local(2))),
                value: local(0),
            },
        ],
    ));
    let mut k = KernelFn::new(
        "matrix_load",
        ElementKind::F32,
        vec![
            Param::buffer("src", H16, BufferRole::Read, Some(1)),
            Param::buffer("out", H16, BufferRole::ReadWrite, Some(1)),
        ],
        vec![
            LocalDecl::scalar(H16),
            LocalDecl::scalar(U32),
            LocalDecl::scalar(U32),
        ],
        Block(body),
    )
    .with_launch(LaunchShape::new([32, 1, 1], [1, 1, 1]).with_subgroup(32, 32));
    k.allocs = vec![
        AllocDecl::new(
            "tiles",
            MemorySpace::Shared,
            H16,
            Extent::Static(64 * count),
        ),
        AllocDecl::new("frag", MemorySpace::Private, H16, Extent::Static(2 * count)),
    ];
    k
}

/// The staged data: element `i` is `i + 1`, so every tile cell is distinct and non-zero.
pub(crate) fn matrix_input(count: u32) -> Vec<u16> {
    (0..64 * count).map(|i| i as u16 + 1).collect()
}

/// What each lane must receive, written from the PTX ISA's description (tile `t`, row `r` is `src[t * 64 + r * 8 ..]`;
/// lane `l` gets two consecutive elements of its row, or of its column when transposed), as `out[l * 8 + v]`.
pub(crate) fn matrix_expected(count: u32, trans: bool) -> Vec<u16> {
    let src = matrix_input(count);
    let mut out = vec![0u16; 32 * 8];
    for l in 0..32usize {
        for t in 0..count as usize {
            for e in 0..2usize {
                let (row, col) = if trans {
                    (2 * (l % 4) + e, l / 4)
                } else {
                    (l / 4, 2 * (l % 4) + e)
                };
                out[l * 8 + 2 * t + e] = src[t * 64 + row * 8 + col];
            }
        }
    }
    out
}
