// SPDX-License-Identifier: MIT OR Apache-2.0
//! The `Stmt::MatrixStore` conformance kernel of ROADMAP U8b slice 2 (`docs/options-u8-copy-atoms.md` S2), shared by the
//! oracle test (`kernel_matrix_store.rs`) and the C test (`unpopped-cpu-c/tests/kernel_matrix_store_emit_agrees.rs`, which
//! keeps a copy: change both). Every lane loads its fragment (`2 * count` elements) from `frag_in[lane * 8 ..]`, the warp
//! stores the tiles into `Shared` with `stmatrix` semantics, and the tiles are copied out, so the output shows exactly
//! which lane's element landed in each tile cell.

use unpopped::exec::{AllocDecl, ExecId, Extent, LaunchShape, MemorySpace};
use unpopped::ir::ScalarExpr;
use unpopped::kernel::{Block, BufferRole, KernelFn, LocalDecl, Param, Place, Stmt};
use unpopped_vocab::ElementKind;

/// The element type: 16 bits moved as bits; `F16` because the serial C spells no 16-bit integer compute type.
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

/// Params: 0 `frag_in` `[32 * 8]` read, 1 `out` `[count * 64]` read-write. Allocs: 0 `tiles` Shared `[count * 64]`, 1 `frag`
/// Private `[2 * count]`. Locals: 0 `x`, 1 `j`, 2 `i`. Without `barrier` each lane copies out the mirrored cell
/// (`64 * count - 1 - i`) with no barrier after the store, so it reads what another lane just wrote: a data race.
pub(crate) fn matrix_store_kernel(count: u32, trans: bool, barrier: bool) -> KernelFn {
    let mut body = vec![load_fragment(count)];
    body.push(Stmt::MatrixStore {
        count,
        trans,
        dst: alloc(0, mul(lane(), u(8))),
        src: 1,
    });
    if barrier {
        body.push(Stmt::barrier(unpopped::exec::Scope::Block));
    }
    body.push(copy_out(count, barrier));
    let mut k = KernelFn::new(
        "matrix_store",
        ElementKind::F32,
        vec![
            Param::buffer("frag_in", H16, BufferRole::Read, Some(1)),
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

/// Every lane reads its `2 * count` fragment elements from `frag_in[lane * 8 ..]`.
fn load_fragment(count: u32) -> Stmt {
    counted(
        2,
        i64::from(2 * count),
        vec![
            Stmt::Load {
                local: 0,
                place: buf(0, add(mul(lane(), u(8)), local(2))),
            },
            Stmt::Store {
                place: alloc(1, local(2)),
                value: local(0),
            },
        ],
    )
}

/// Lane `l` copies cells `l`, `l + 32`, ... of `tiles` to `out` (the mirrored cell when `barrier` is false).
fn copy_out(count: u32, barrier: bool) -> Stmt {
    let cell = add(lane(), mul(local(1), u(32)));
    let from = if barrier {
        cell.clone()
    } else {
        ScalarExpr::Sub(Box::new(u(i64::from(64 * count) - 1)), Box::new(cell.clone()))
    };
    counted(
        1,
        i64::from(2 * count),
        vec![
            Stmt::Load {
                local: 0,
                place: alloc(0, from),
            },
            Stmt::Store {
                place: buf(1, cell),
                value: local(0),
            },
        ],
    )
}

/// Lane `l`'s fragment input: element `i` of lane `l` is `1 + l * 8 + i`, so every element is distinct and non-zero.
pub(crate) fn fragment_input() -> Vec<u16> {
    (0..32 * 8).map(|i| i as u16 + 1).collect()
}

/// What the tiles must hold, written from the PTX ISA's description: tile `t` row `r` column `c` is, without `trans`,
/// element `2 t + c % 2` of lane `4 r + c / 2`; transposed, element `2 t + r % 2` of lane `r / 2 + 4 c`. As `out[t * 64 + r * 8 + c]`.
pub(crate) fn tiles_expected(count: u32, trans: bool) -> Vec<u16> {
    let input = fragment_input();
    let mut out = vec![0u16; 64 * count as usize];
    for t in 0..count as usize {
        for r in 0..8usize {
            for c in 0..8usize {
                let (l, e) = if trans {
                    (r / 2 + 4 * c, r % 2)
                } else {
                    (4 * r + c / 2, c % 2)
                };
                out[t * 64 + r * 8 + c] = input[l * 8 + 2 * t + e];
            }
        }
    }
    out
}
