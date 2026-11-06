// COPY of `crates/unpopped/tests/support/attention.rs`: a test may not `#[path]` into another crate's directory (sweep SUP-004).
// Change both files together.
// SPDX-License-Identifier: MIT OR Apache-2.0
//! The attention conformance kernels of ROADMAP U10 (`docs/options-u10-attention.md`), written with the kernel layer alone.
//! A1 is the naive reference: one thread per query row computes the scores `S = scale * Q K^T` into a global scratch buffer, the
//! row max and sum of `exp(S - max)`, then `O = softmax(S) V`. Shared by the oracle test (`kernel_attention_naive.rs`) and the C
//! test (`unpopped-cpu-c/tests/kernel_attention_naive_emit_agrees.rs`, which keeps a copy: change both).

use unpopped::exec::{ExecId, LaunchShape};
use unpopped::ir::{BinaryOp, ScalarExpr, UnaryOp};
use unpopped::kernel::{Block, BufferRole, KernelFn, LocalDecl, Param, Place, Stmt};
use unpopped_vocab::ElementKind;

pub(crate) const F32: ElementKind = ElementKind::F32;
pub(crate) const U32: ElementKind = ElementKind::U32;
/// Query rows (one thread each), key/value rows, head dimension of Q and K, head dimension of V.
pub(crate) const LQ: usize = 4;
pub(crate) const LK: usize = 6;
pub(crate) const D: usize = 4;
pub(crate) const DV: usize = 3;
/// The score scale. Large on purpose: with these inputs some scores pass 89, where `exp` overflows an `f32`, so a kernel that
/// does not subtract the row max produces infinities and NaN.
pub(crate) const SCALE: f64 = 6.0;

fn cu(v: usize) -> ScalarExpr {
    ScalarExpr::ConstInt(v as i64, U32)
}

fn local(i: u32) -> ScalarExpr {
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

fn exp(x: ScalarExpr) -> ScalarExpr {
    ScalarExpr::Unary(UnaryOp::Exp, Box::new(x))
}

fn flat(buffer: u32, i: ScalarExpr) -> Place {
    Place::Logical {
        buffer,
        coords: vec![i],
    }
}

fn counted(local_id: u32, end: usize, body: Vec<Stmt>) -> Stmt {
    Stmt::For {
        local: local_id,
        start: cu(0),
        end: cu(end),
        step: cu(1),
        body: Block(body),
    }
}

fn row() -> ScalarExpr {
    ScalarExpr::Exec(ExecId::ThreadIdx(0))
}

fn load(local: u32, buffer: u32, i: ScalarExpr) -> Stmt {
    Stmt::Load {
        local,
        place: flat(buffer, i),
    }
}

// Locals: 0 acc, 1 j, 2 d, 3 q, 4 k, 5 max, 6 sum, 7 score, 8 v.

/// `S[row][j] = SCALE * dot(Q[row], K[j])` for every `j`.
fn scores() -> Stmt {
    counted(
        1,
        LK,
        vec![
            Stmt::Let {
                local: 0,
                init: ScalarExpr::Const(0.0),
            },
            counted(
                2,
                D,
                vec![
                    load(3, 0, add(mul(row(), cu(D)), local(2))),
                    load(4, 1, add(mul(local(1), cu(D)), local(2))),
                    Stmt::Assign {
                        local: 0,
                        value: add(local(0), mul(local(3), local(4))),
                    },
                ],
            ),
            Stmt::Store {
                place: flat(3, add(mul(row(), cu(LK)), local(1))),
                value: mul(local(0), ScalarExpr::Const(SCALE)),
            },
        ],
    )
}

/// `max` and `sum = sum_j exp(S[row][j] - max)` of the row (locals 5 and 6).
fn softmax_stats() -> Vec<Stmt> {
    let s_at = |j: ScalarExpr| add(mul(row(), cu(LK)), j);
    vec![
        load(5, 3, s_at(cu(0))),
        counted(
            1,
            LK,
            vec![
                load(7, 3, s_at(local(1))),
                Stmt::Assign {
                    local: 5,
                    value: ScalarExpr::Binary(
                        BinaryOp::Max,
                        Box::new(local(5)),
                        Box::new(local(7)),
                    ),
                },
            ],
        ),
        Stmt::Let {
            local: 6,
            init: ScalarExpr::Const(0.0),
        },
        counted(
            1,
            LK,
            vec![
                load(7, 3, s_at(local(1))),
                Stmt::Assign {
                    local: 6,
                    value: add(local(6), exp(sub(local(7), local(5)))),
                },
            ],
        ),
    ]
}

/// `O[row][d] = sum_j (exp(S[row][j] - max) / sum) * V[j][d]`.
fn weighted_values() -> Stmt {
    counted(
        2,
        DV,
        vec![
            Stmt::Let {
                local: 0,
                init: ScalarExpr::Const(0.0),
            },
            counted(
                1,
                LK,
                vec![
                    load(7, 3, add(mul(row(), cu(LK)), local(1))),
                    load(8, 2, add(mul(local(1), cu(DV)), local(2))),
                    Stmt::Assign {
                        local: 0,
                        value: add(
                            local(0),
                            mul(div(exp(sub(local(7), local(5))), local(6)), local(8)),
                        ),
                    },
                ],
            ),
            Stmt::Store {
                place: flat(4, add(mul(row(), cu(DV)), local(2))),
                value: local(0),
            },
        ],
    )
}

/// Buffers: 0 `q` `[LQ*D]`, 1 `k` `[LK*D]`, 2 `v` `[LK*DV]` (read), 3 `s` `[LQ*LK]` scratch scores, 4 `out` `[LQ*DV]`; all `F32`.
pub(crate) fn naive_attention() -> KernelFn {
    let mut body = vec![scores()];
    body.extend(softmax_stats());
    body.push(weighted_values());
    KernelFn::new(
        "attention_naive",
        F32,
        vec![
            Param::buffer("q", F32, BufferRole::Read, Some(1)),
            Param::buffer("k", F32, BufferRole::Read, Some(1)),
            Param::buffer("v", F32, BufferRole::Read, Some(1)),
            Param::buffer("s", F32, BufferRole::ReadWrite, Some(1)),
            Param::buffer("out", F32, BufferRole::ReadWrite, Some(1)),
        ],
        vec![
            LocalDecl::scalar(F32),
            LocalDecl::scalar(U32),
            LocalDecl::scalar(U32),
            LocalDecl::scalar(F32),
            LocalDecl::scalar(F32),
            LocalDecl::scalar(F32),
            LocalDecl::scalar(F32),
            LocalDecl::scalar(F32),
            LocalDecl::scalar(F32),
        ],
        Block(body),
    )
    .with_launch(LaunchShape::new([LQ as u32, 1, 1], [1, 1, 1]))
}

/// `(q, k, v)`: small integers, so every dot product is exact before the scale; scores reach past 89.
pub(crate) fn attention_data() -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let q = (0..LQ * D)
        .map(|i| ((i / D * 3 + i % D * 5) % 7) as f32 - 3.0)
        .collect();
    let k = (0..LK * D)
        .map(|i| ((i / D * 5 + i % D * 3) % 9) as f32 - 4.0)
        .collect();
    let v = (0..LK * DV)
        .map(|i| ((i / DV * 7 + i % DV) % 5) as f32 - 2.0)
        .collect();
    (q, k, v)
}

/// The host result `(scores, out)`, in the same `f32` operation order as the kernel.
pub(crate) fn attention_reference() -> (Vec<f32>, Vec<f32>) {
    let (q, k, v) = attention_data();
    let mut s = vec![0f32; LQ * LK];
    let mut out = vec![0f32; LQ * DV];
    for i in 0..LQ {
        for j in 0..LK {
            let mut acc = 0f32;
            for d in 0..D {
                acc += q[i * D + d] * k[j * D + d];
            }
            s[i * LK + j] = acc * SCALE as f32;
        }
        let m = s[i * LK..(i + 1) * LK]
            .iter()
            .fold(s[i * LK], |a, &b| a.max(b));
        let l: f32 = s[i * LK..(i + 1) * LK].iter().map(|x| (x - m).exp()).sum();
        for d in 0..DV {
            let mut acc = 0f32;
            for j in 0..LK {
                acc += ((s[i * LK + j] - m).exp() / l) * v[j * DV + d];
            }
            out[i * DV + d] = acc;
        }
    }
    (s, out)
}
