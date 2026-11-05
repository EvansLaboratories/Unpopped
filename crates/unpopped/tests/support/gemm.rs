// SPDX-License-Identifier: MIT OR Apache-2.0
//! The GEMM conformance kernels of ROADMAP U9 (kernel-layer part, `docs/options-u9-gemm-family.md`): a small integer
//! GEMM `D = epilogue(A.B)` written with the kernel layer alone (one thread per output element, `I32` so every result
//! is exact and a plain host loop is the reference). Shared by the oracle test (`kernel_gemm_family.rs`) and the C test
//! (`unpopped-cpu-c/tests/kernel_gemm_family_emit_agrees.rs`, which keeps a copy: change both).

use unpopped::exec::{ExecId, LaunchShape};
use unpopped::ir::{BinaryOp, ScalarExpr};
use unpopped::kernel::{Block, BufferRole, KernelFn, LocalDecl, Param, Place, Stmt};
use unpopped_vocab::ElementKind;

pub(crate) const I32: ElementKind = ElementKind::I32;
pub(crate) const U32: ElementKind = ElementKind::U32;
pub(crate) const M: u32 = 4;
pub(crate) const N: u32 = 4;
pub(crate) const K: u32 = 4;
pub(crate) const ALPHA: i64 = 2;
pub(crate) const BETA: i64 = 3;

fn ci(v: i64) -> ScalarExpr {
    ScalarExpr::ConstInt(v, I32)
}

fn cu(v: i64) -> ScalarExpr {
    ScalarExpr::ConstInt(v, U32)
}

fn local(i: u32) -> ScalarExpr {
    ScalarExpr::Local(i)
}

fn bin(op: BinaryOp, a: ScalarExpr, b: ScalarExpr) -> ScalarExpr {
    ScalarExpr::Binary(op, Box::new(a), Box::new(b))
}

fn mul(a: ScalarExpr, b: ScalarExpr) -> ScalarExpr {
    ScalarExpr::Mul(Box::new(a), Box::new(b))
}

fn add(a: ScalarExpr, b: ScalarExpr) -> ScalarExpr {
    ScalarExpr::Add(Box::new(a), Box::new(b))
}

fn at2(buffer: u32, r: ScalarExpr, c: ScalarExpr) -> Place {
    Place::Logical {
        buffer,
        coords: vec![r, c],
    }
}

/// `relu(x)` for an `I32` expression.
fn relu(x: ScalarExpr) -> ScalarExpr {
    ScalarExpr::Select(
        Box::new(bin(BinaryOp::CmpLt, x.clone(), ci(0))),
        Box::new(ci(0)),
        Box::new(x),
    )
}

fn row() -> ScalarExpr {
    ScalarExpr::Exec(ExecId::ThreadIdx(1))
}

fn col() -> ScalarExpr {
    ScalarExpr::Exec(ExecId::ThreadIdx(0))
}

/// `acc = sum over k of A[r][k] * B[k][c]` into local 0 (locals 1, 2 the operands, 5 the K counter).
fn mainloop() -> Vec<Stmt> {
    vec![
        Stmt::Let {
            local: 0,
            init: ci(0),
        },
        Stmt::For {
            local: 5,
            start: cu(0),
            end: cu(i64::from(K)),
            step: cu(1),
            body: Block(vec![
                Stmt::Load {
                    local: 1,
                    place: at2(0, row(), local(5)),
                },
                Stmt::Load {
                    local: 2,
                    place: at2(1, local(5), col()),
                },
                Stmt::Assign {
                    local: 0,
                    value: add(local(0), mul(local(1), local(2))),
                },
            ]),
        },
    ]
}

/// The epilogue: read `C` and the column's bias, combine, clamp, store (locals 3, 4, 6).
fn epilogue() -> Vec<Stmt> {
    vec![
        Stmt::Load {
            local: 3,
            place: at2(2, row(), col()),
        },
        Stmt::Load {
            local: 4,
            place: Place::Logical {
                buffer: 3,
                coords: vec![col()],
            },
        },
        Stmt::Let {
            local: 6,
            init: add(
                add(mul(ci(ALPHA), local(0)), mul(ci(BETA), local(3))),
                local(4),
            ),
        },
        Stmt::Store {
            place: at2(4, row(), col()),
            value: relu(local(6)),
        },
    ]
}

/// Buffers: 0 `a` `[M, K]`, 1 `b` `[K, N]`, 2 `c` `[M, N]`, 3 `bias` `[N]` (all read), 4 `out` `[M, N]`.
/// `out[r][c] = relu(ALPHA * (A.B)[r][c] + BETA * C[r][c] + bias[c])`, one thread per `(r, c)`.
pub(crate) fn fused_epilogue() -> KernelFn {
    let mut body = mainloop();
    body.extend(epilogue());
    let p = |name: &str, role: BufferRole, rank: u8| Param::buffer(name, I32, role, Some(rank));
    let mut locals: Vec<LocalDecl> = (0..7).map(|_| LocalDecl::scalar(I32)).collect();
    locals[5] = LocalDecl::scalar(U32);
    KernelFn::new(
        "gemm_epilogue",
        I32,
        vec![
            p("a", BufferRole::Read, 2),
            p("b", BufferRole::Read, 2),
            p("c", BufferRole::Read, 2),
            p("bias", BufferRole::Read, 1),
            p("out", BufferRole::ReadWrite, 2),
        ],
        locals,
        Block(body),
    )
    .with_launch(LaunchShape::new([N, M, 1], [1, 1, 1]))
}

/// The host reference for [`fused_epilogue`].
pub(crate) fn fused_epilogue_ref(a: &[i32], b: &[i32], c: &[i32], bias: &[i32]) -> Vec<i32> {
    let (m, n, k) = (M as usize, N as usize, K as usize);
    let mut out = vec![0; m * n];
    for r in 0..m {
        for col in 0..n {
            let acc: i32 = (0..k).map(|i| a[r * k + i] * b[i * n + col]).sum();
            let v = ALPHA as i32 * acc + BETA as i32 * c[r * n + col] + bias[col];
            out[r * n + col] = v.max(0);
        }
    }
    out
}
