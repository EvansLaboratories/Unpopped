// COPY of `crates/unpopped/tests/support/grouped.rs`: a test may not `#[path]` into another crate's directory (sweep SUP-004).
// Change both files together.
// SPDX-License-Identifier: MIT OR Apache-2.0
//! The grouped-GEMM conformance kernel of ROADMAP U9 K5 (`docs/options-u9-gemm-family.md`): several GEMMs of different
//! shapes in one launch, described by a problem table, the faithful kernel-layer form of CUTLASS's grouped GEMM before
//! pointer values exist (the operands live in one buffer each and the table holds element offsets, not pointers).
//! Shared by the oracle test (`kernel_gemm_grouped.rs`) and the C test
//! (`unpopped-cpu-c/tests/kernel_gemm_grouped_emit_agrees.rs`, which keeps a copy: change both).

use unpopped::exec::{ExecId, LaunchShape};
use unpopped::ir::{BinaryOp, ScalarExpr};
use unpopped::kernel::{Block, BufferRole, KernelFn, LocalDecl, Param, Place, Stmt};
use unpopped_vocab::ElementKind;

pub(crate) const I32: ElementKind = ElementKind::I32;
pub(crate) const U32: ElementKind = ElementKind::U32;
/// Table columns: `m, n, k, a_off, b_off, d_off`.
pub(crate) const COLS: usize = 6;
/// The largest `m` and `n` of any problem: the block is `MAXN x MAXM`.
pub(crate) const MAXM: u32 = 4;
pub(crate) const MAXN: u32 = 4;

/// `(m, n, k)` of each problem; different shapes on purpose, none larger than the block.
pub(crate) const PROBLEMS: [(u32, u32, u32); 3] = [(3, 2, 4), (4, 4, 2), (2, 4, 3)];

fn cu(v: i64) -> ScalarExpr {
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

fn flat(buffer: u32, i: ScalarExpr) -> Place {
    Place::Logical {
        buffer,
        coords: vec![i],
    }
}

fn lt(a: ScalarExpr, b: ScalarExpr) -> ScalarExpr {
    ScalarExpr::Binary(BinaryOp::CmpLt, Box::new(a), Box::new(b))
}

/// The table row of this block's problem into locals 0..=5 (`m, n, k, a_off, b_off, d_off`).
fn read_row() -> Vec<Stmt> {
    (0..COLS as u32)
        .map(|j| Stmt::Load {
            local: j,
            place: Place::Logical {
                buffer: 0,
                coords: vec![ScalarExpr::Exec(ExecId::BlockIdx(0)), cu(i64::from(j))],
            },
        })
        .collect()
}

/// `acc = sum over kk of A[a_off + r*k + kk] * B[b_off + kk*n + c]`, then `D[d_off + r*n + c] = acc` (locals 6..=9).
fn problem_body(r: &ScalarExpr, c: &ScalarExpr) -> Vec<Stmt> {
    vec![
        Stmt::Let {
            local: 6,
            init: ScalarExpr::ConstInt(0, I32),
        },
        Stmt::For {
            local: 7,
            start: cu(0),
            end: local(2),
            step: cu(1),
            body: Block(vec![
                Stmt::Load {
                    local: 8,
                    place: flat(1, add(add(local(3), mul(r.clone(), local(2))), local(7))),
                },
                Stmt::Load {
                    local: 9,
                    place: flat(2, add(add(local(4), mul(local(7), local(1))), c.clone())),
                },
                Stmt::Assign {
                    local: 6,
                    value: add(local(6), mul(local(8), local(9))),
                },
            ]),
        },
        Stmt::Store {
            place: flat(3, add(add(local(5), mul(r.clone(), local(1))), c.clone())),
            value: local(6),
        },
    ]
}

/// Buffers: 0 `tbl` `[P, 6]` `U32`, 1 `a`, 2 `b` (flat `I32`, read), 3 `d` (flat `I32`). Block `p` computes problem `p`,
/// one thread per `(r, c)`; a thread outside the problem's `m x n` does nothing.
pub(crate) fn grouped_gemm() -> KernelFn {
    let r = ScalarExpr::Exec(ExecId::ThreadIdx(1));
    let c = ScalarExpr::Exec(ExecId::ThreadIdx(0));
    let mut body = read_row();
    body.push(Stmt::If {
        cond: lt(r.clone(), local(0)),
        then: Block(vec![Stmt::If {
            cond: lt(c.clone(), local(1)),
            then: Block(problem_body(&r, &c)),
            els: Block(vec![]),
        }]),
        els: Block(vec![]),
    });
    let mut locals: Vec<LocalDecl> = (0..6).map(|_| LocalDecl::scalar(U32)).collect();
    locals.extend([I32, U32, I32, I32].map(LocalDecl::scalar));
    KernelFn::new(
        "grouped_gemm",
        I32,
        vec![
            Param::buffer("tbl", U32, BufferRole::Read, Some(2)),
            Param::buffer("a", I32, BufferRole::Read, Some(1)),
            Param::buffer("b", I32, BufferRole::Read, Some(1)),
            Param::buffer("d", I32, BufferRole::ReadWrite, Some(1)),
        ],
        locals,
        Block(body),
    )
    .with_launch(LaunchShape::new(
        [MAXN, MAXM, 1],
        [PROBLEMS.len() as u32, 1, 1],
    ))
}

/// The table, the three flat operands and the host-computed `D` of [`PROBLEMS`]: `(tbl, a, b, d_zero_len, d_ref)`.
pub(crate) fn grouped_data() -> (Vec<u32>, Vec<i32>, Vec<i32>, usize, Vec<i32>) {
    let (mut tbl, mut a, mut b, mut d) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
    for (p, (m, n, k)) in PROBLEMS.into_iter().enumerate() {
        let (ao, bo, dof) = (a.len(), b.len(), d.len());
        tbl.extend([m, n, k, ao as u32, bo as u32, dof as u32]);
        let p = p as i32;
        a.extend((0..(m * k) as i32).map(|i| 3 * p + i % 5 - 2));
        b.extend((0..(k * n) as i32).map(|i| 7 - p - i % 4));
        for r in 0..m as usize {
            for c in 0..n as usize {
                let acc: i32 = (0..k as usize)
                    .map(|kk| a[ao + r * k as usize + kk] * b[bo + kk * n as usize + c])
                    .sum();
                d.push(acc);
            }
        }
    }
    let len = d.len();
    (tbl, a, b, len, d)
}
