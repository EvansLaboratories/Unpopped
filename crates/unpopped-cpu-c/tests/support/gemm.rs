// COPY of `crates/unpopped/tests/support/gemm.rs`: a test may not `#[path]` into another crate's directory (sweep SUP-004).
// Change both files together.
// SPDX-License-Identifier: MIT OR Apache-2.0
//! The GEMM conformance kernels of ROADMAP U9 (kernel-layer part, `docs/options-u9-gemm-family.md`): a small integer
//! GEMM `D = epilogue(A.B)` written with the kernel layer alone (one thread per output element, `I32` so every result
//! is exact and a plain host loop is the reference). Shared by the oracle test (`kernel_gemm_family.rs`) and the C test
//! (`unpopped-cpu-c/tests/kernel_gemm_family_emit_agrees.rs`, which keeps a copy: change both).

use unpopped::exec::Scope;
use unpopped::exec::{ExecId, LaunchShape};
use unpopped::ir::{BinaryOp, ScalarExpr};
use unpopped::kernel::{
    AtomicOp, Block, BufferRole, KernelFn, LocalDecl, MemoryOrder, Param, Place, Stmt,
};
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

/// An atomic read-modify-write of `place` at device scope, result unused.
fn atomic(op: AtomicOp, place: Place, value: ScalarExpr) -> Stmt {
    Stmt::AtomicRmw {
        op,
        place,
        value,
        old: None,
        order: MemoryOrder::Relaxed,
        scope: Scope::Device,
    }
}

/// The aux epilogue: `D = acc`, the mask `acc > 0`, the per-column sum of `relu(acc)` (`dbias`) and the maximum of
/// `acc` (`amax`), the last two by atomics (locals 6 and 7).
fn aux_epilogue() -> Vec<Stmt> {
    vec![
        Stmt::Store {
            place: at2(2, row(), col()),
            value: local(0),
        },
        Stmt::Let {
            local: 6,
            init: ScalarExpr::Select(
                Box::new(bin(BinaryOp::CmpLt, ci(0), local(0))),
                Box::new(ci(1)),
                Box::new(ci(0)),
            ),
        },
        Stmt::Store {
            place: at2(3, row(), col()),
            value: local(6),
        },
        Stmt::Let {
            local: 7,
            init: relu(local(0)),
        },
        atomic(
            AtomicOp::Add,
            Place::Logical {
                buffer: 4,
                coords: vec![col()],
            },
            local(7),
        ),
        atomic(
            AtomicOp::Max,
            Place::Logical {
                buffer: 5,
                coords: vec![cu(0)],
            },
            local(0),
        ),
    ]
}

/// Buffers: 0 `a` `[M, K]`, 1 `b` `[K, N]` (read); 2 `d` `[M, N]` (`A.B`), 3 `mask` `[M, N]` (`acc > 0`), 4 `dbias` `[N]`
/// (per-column sum of `relu(acc)`, start at zero), 5 `amax` `[1]` (the maximum of `acc`, start at `i32::MIN`).
pub(crate) fn aux_outputs() -> KernelFn {
    let mut body = mainloop();
    body.extend(aux_epilogue());
    let p = |name: &str, role: BufferRole, rank: u8| Param::buffer(name, I32, role, Some(rank));
    let mut locals: Vec<LocalDecl> = (0..8).map(|_| LocalDecl::scalar(I32)).collect();
    locals[5] = LocalDecl::scalar(U32);
    KernelFn::new(
        "gemm_aux",
        I32,
        vec![
            p("a", BufferRole::Read, 2),
            p("b", BufferRole::Read, 2),
            p("d", BufferRole::ReadWrite, 2),
            p("mask", BufferRole::ReadWrite, 2),
            p("dbias", BufferRole::ReadWrite, 1),
            p("amax", BufferRole::ReadWrite, 1),
        ],
        locals,
        Block(body),
    )
    .with_launch(LaunchShape::new([N, M, 1], [1, 1, 1]))
}

/// The host reference for [`aux_outputs`]: `(D, mask, dbias, amax)`.
pub(crate) fn aux_outputs_ref(a: &[i32], b: &[i32]) -> (Vec<i32>, Vec<i32>, Vec<i32>, i32) {
    let (m, n, k) = (M as usize, N as usize, K as usize);
    let d: Vec<i32> = (0..m * n)
        .map(|i| (0..k).map(|t| a[(i / n) * k + t] * b[t * n + i % n]).sum())
        .collect();
    let mask = d.iter().map(|v| i32::from(*v > 0)).collect();
    let dbias = (0..n)
        .map(|c| (0..m).map(|r| d[r * n + c].max(0)).sum())
        .collect();
    let amax = *d.iter().max().expect("a non-empty product");
    (d, mask, dbias, amax)
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
