// COPY of `crates/unpopped/tests/support/attention.rs`: a test may not `#[path]` into another crate's directory (sweep SUP-004).
// Change both files together.
// SPDX-License-Identifier: MIT OR Apache-2.0
//! The attention conformance kernels of ROADMAP U10 (`docs/options-u10-attention.md`), written with the kernel layer alone.
//! A1 is the naive reference: one thread per query row computes the scores `S = scale * Q K^T` into a global scratch buffer, the
//! row max and sum of `exp(S - max)`, then `O = softmax(S) V`. Shared by the oracle test (`kernel_attention_naive.rs`) and the C
//! test (`unpopped-cpu-c/tests/kernel_attention_naive_emit_agrees.rs`, which keeps a copy: change both).

use unpopped::exec::{ExecId, LaunchShape};
use unpopped::ir::{BinaryOp, CastSpec, RoundMode, Saturation, ScalarExpr, UnaryOp};
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

// A2 locals: 0 dot, 1 j, 2 d (output column), 3 q, 4 k, 5 m, 6 l, 7 s, 8 v, 9 m_new, 10 alpha, 11 p, 12 acc, 13 e (head dim).

/// The most negative finite `f32` stands for "no score yet", so the first rescale `exp(m - m_new)` is exactly 0.
pub(crate) const NO_SCORE: f64 = -3.0e38;

/// `s = SCALE * dot(Q[row], K[j])` into local 7, without storing it anywhere.
fn score_inline(k_row: ScalarExpr) -> Vec<Stmt> {
    vec![
        Stmt::Let {
            local: 0,
            init: ScalarExpr::Const(0.0),
        },
        counted(
            13,
            D,
            vec![
                load(3, 0, add(mul(row(), cu(D)), local(13))),
                load(4, 1, add(mul(k_row, cu(D)), local(13))),
                Stmt::Assign {
                    local: 0,
                    value: add(local(0), mul(local(3), local(4))),
                },
            ],
        ),
        Stmt::Let {
            local: 7,
            init: mul(local(0), ScalarExpr::Const(SCALE)),
        },
    ]
}

/// The online-softmax update for one key: `m' = max(m, s)`, `alpha = exp(m - m')`, `p = exp(s - m')`,
/// `l = l * alpha + p`, `acc = acc * alpha + p * V[j][d]`, `m = m'`.
fn online_step() -> Vec<Stmt> {
    let mut b = score_inline(local(1));
    b.extend(online_tail(local(1)));
    b
}

/// The update half of [`online_step`], given the score in local 7.
fn online_tail(v_row: ScalarExpr) -> Vec<Stmt> {
    vec![
        load(8, 2, add(mul(v_row, cu(DV)), local(2))),
        Stmt::Let {
            local: 9,
            init: ScalarExpr::Binary(BinaryOp::Max, Box::new(local(5)), Box::new(local(7))),
        },
        Stmt::Let {
            local: 10,
            init: exp(sub(local(5), local(9))),
        },
        Stmt::Let {
            local: 11,
            init: exp(sub(local(7), local(9))),
        },
        Stmt::Assign {
            local: 6,
            value: add(mul(local(6), local(10)), local(11)),
        },
        Stmt::Assign {
            local: 12,
            value: add(mul(local(12), local(10)), mul(local(11), local(8))),
        },
        Stmt::Assign {
            local: 5,
            value: local(9),
        },
    ]
}

/// Flash-style forward: no score matrix. Per query row and output column, one pass over the keys keeps `(m, l, acc)`.
/// Buffers: 0 `q`, 1 `k`, 2 `v` (read), 3 `out` `[LQ*DV]`; all `F32`.
pub(crate) fn flash_attention() -> KernelFn {
    let column = counted(
        2,
        DV,
        vec![
            Stmt::Let {
                local: 5,
                init: ScalarExpr::Const(NO_SCORE),
            },
            Stmt::Let {
                local: 6,
                init: ScalarExpr::Const(0.0),
            },
            Stmt::Let {
                local: 12,
                init: ScalarExpr::Const(0.0),
            },
            counted(1, LK, online_step()),
            Stmt::Store {
                place: flat(3, add(mul(row(), cu(DV)), local(2))),
                value: div(local(12), local(6)),
            },
        ],
    );
    let mut locals = vec![LocalDecl::scalar(F32); 14];
    for u in [1, 2, 13] {
        locals[u] = LocalDecl::scalar(U32);
    }
    KernelFn::new(
        "attention_flash",
        F32,
        vec![
            Param::buffer("q", F32, BufferRole::Read, Some(1)),
            Param::buffer("k", F32, BufferRole::Read, Some(1)),
            Param::buffer("v", F32, BufferRole::Read, Some(1)),
            Param::buffer("out", F32, BufferRole::ReadWrite, Some(1)),
        ],
        locals,
        Block(vec![column]),
    )
    .with_launch(LaunchShape::new([LQ as u32, 1, 1], [1, 1, 1]))
}

/// The ALiBi bias on a visible score: `s -= slope * (row - j)`.
fn alibi_bias(slope: f64) -> Stmt {
    let dist = ScalarExpr::Cast(
        Box::new(sub(row(), local(1))),
        CastSpec {
            to: F32,
            round: RoundMode::NearestEven,
            sat: Saturation::Wrap,
        },
    );
    Stmt::Assign {
        local: 7,
        value: sub(local(7), mul(ScalarExpr::Const(slope), dist)),
    }
}

/// The per-output-column skeleton of a streamed softmax: `(m, l, acc)` start empty, `scan` updates them, and the column
/// `out[row][d]` is `acc / l`, or 0 when no key was visible (`l == 0`). `out` is the output buffer's index.
fn softmax_column(scan: Vec<Stmt>, out: u32) -> Stmt {
    let nonempty = ScalarExpr::Binary(
        BinaryOp::CmpGt,
        Box::new(local(6)),
        Box::new(ScalarExpr::Const(0.0)),
    );
    let mut body = vec![
        Stmt::Let {
            local: 5,
            init: ScalarExpr::Const(NO_SCORE),
        },
        Stmt::Let {
            local: 6,
            init: ScalarExpr::Const(0.0),
        },
        Stmt::Let {
            local: 12,
            init: ScalarExpr::Const(0.0),
        },
    ];
    body.extend(scan);
    body.push(Stmt::Store {
        place: flat(out, add(mul(row(), cu(DV)), local(2))),
        value: ScalarExpr::Select(
            Box::new(nonempty),
            Box::new(div(local(12), local(6))),
            Box::new(ScalarExpr::Const(0.0)),
        ),
    });
    counted(2, DV, body)
}

/// A one-thread-per-row kernel over `rows` rows: `n_locals` `F32` locals except the listed integer ones.
fn decode_kernel(
    name: &str,
    params: Vec<Param>,
    int_locals: &[(usize, ElementKind)],
    n_locals: usize,
    column: Stmt,
    rows: usize,
) -> KernelFn {
    let mut locals = vec![LocalDecl::scalar(F32); n_locals];
    for &(i, ty) in int_locals {
        locals[i] = LocalDecl::scalar(ty);
    }
    KernelFn::new(name, F32, params, locals, Block(vec![column]))
        .with_launch(LaunchShape::new([rows as u32, 1, 1], [1, 1, 1]))
}

/// A3: key `j` is visible to query row `i` when `j + lag <= i` (`lag = 0` is causal; `lag = 1` hides the diagonal, so row 0 sees
/// nothing). With a `slope`, a visible score also loses `slope * (i - j)` (the ALiBi bias). A row that sees no key is all zeros.
pub(crate) fn masked_attention(lag: usize, slope: Option<f64>) -> KernelFn {
    let mut step = score_inline(local(1));
    step.extend(slope.map(alibi_bias));
    step.extend(online_tail(local(1)));
    let visible = ScalarExpr::Binary(
        BinaryOp::CmpLe,
        Box::new(add(local(1), cu(lag))),
        Box::new(row()),
    );
    let guarded = Stmt::If {
        cond: visible,
        then: Block(step),
        els: Block(vec![]),
    };
    let column = softmax_column(vec![counted(1, LK, vec![guarded])], 3);
    let params = vec![
        Param::buffer("q", F32, BufferRole::Read, Some(1)),
        Param::buffer("k", F32, BufferRole::Read, Some(1)),
        Param::buffer("v", F32, BufferRole::Read, Some(1)),
        Param::buffer("out", F32, BufferRole::ReadWrite, Some(1)),
    ];
    decode_kernel(
        "attention_masked",
        params,
        &[(1, U32), (2, U32), (13, U32)],
        14,
        column,
        LQ,
    )
}

/// The host result of [`masked_attention`], in the same `f32` operation order.
pub(crate) fn masked_reference(lag: usize, slope: Option<f64>) -> Vec<f32> {
    let (q, k, v) = attention_data();
    let mut out = vec![0f32; LQ * DV];
    for i in 0..LQ {
        for d in 0..DV {
            let (mut m, mut l, mut acc) = (NO_SCORE as f32, 0f32, 0f32);
            for j in (0..LK).filter(|&j| j + lag <= i) {
                let mut dot = 0f32;
                for e in 0..D {
                    dot += q[i * D + e] * k[j * D + e];
                }
                let mut s = dot * SCALE as f32;
                if let Some(a) = slope {
                    s -= a as f32 * (i - j) as f32;
                }
                let mn = m.max(s);
                let alpha = (m - mn).exp();
                let p = (s - mn).exp();
                l = l * alpha + p;
                acc = acc * alpha + p * v[j * DV + d];
                m = mn;
            }
            out[i * DV + d] = if l > 0.0 { acc / l } else { 0.0 };
        }
    }
    out
}

// A4: a paged KV cache. One query token per sequence; K and V live in a pool of PAGE-token pages, and an `I32` block table maps
// (sequence, logical page) to a pool page. `lens` holds each sequence's token count. Locals: 14 page (`I32`), 15 slot, 16 len (`I32`).

pub(crate) const I32: ElementKind = ElementKind::I32;
/// Sequences (one thread each), tokens per page, logical pages per sequence, pool pages.
pub(crate) const NSEQ: usize = 3;
pub(crate) const PAGE: usize = 2;
pub(crate) const MAXP: usize = 4;
pub(crate) const NPAGES: usize = 9;
/// Token counts: a partial last page, exactly one full page, an odd count.
pub(crate) const LENS: [i32; NSEQ] = [7, 2, 5];
/// The block table, row per sequence. Unused entries name page 8, whose contents are poison.
pub(crate) const TABLE: [i32; NSEQ * MAXP] = [5, 2, 7, 0, 3, 8, 8, 8, 6, 1, 4, 8];

fn to_u32(x: ScalarExpr) -> ScalarExpr {
    ScalarExpr::Cast(
        Box::new(x),
        CastSpec {
            to: U32,
            round: RoundMode::NearestEven,
            sat: Saturation::Wrap,
        },
    )
}

/// Paged decode attention, flash-style. Buffers: 0 `q` `[NSEQ*D]`, 1 `k` pool `[NPAGES*PAGE*D]`, 2 `v` pool `[NPAGES*PAGE*DV]`,
/// 3 `table` `I32` `[NSEQ*MAXP]`, 4 `lens` `I32` `[NSEQ]`, 5 `out` `[NSEQ*DV]`.
pub(crate) fn paged_attention() -> KernelFn {
    let pool_row = add(mul(to_u32(local(14)), cu(PAGE)), local(15));
    let mut step = score_inline(pool_row.clone());
    step.extend(online_tail(pool_row));
    let token = add(mul(local(1), cu(PAGE)), local(15));
    let visible = ScalarExpr::Binary(
        BinaryOp::CmpLt,
        Box::new(token),
        Box::new(to_u32(local(16))),
    );
    let page_loop = counted(
        1,
        MAXP,
        vec![
            load(14, 3, add(mul(row(), cu(MAXP)), local(1))),
            counted(
                15,
                PAGE,
                vec![Stmt::If {
                    cond: visible,
                    then: Block(step),
                    els: Block(vec![]),
                }],
            ),
        ],
    );
    let column = softmax_column(vec![load(16, 4, row()), page_loop], 5);
    let params = vec![
        Param::buffer("q", F32, BufferRole::Read, Some(1)),
        Param::buffer("k", F32, BufferRole::Read, Some(1)),
        Param::buffer("v", F32, BufferRole::Read, Some(1)),
        Param::buffer("table", I32, BufferRole::Read, Some(1)),
        Param::buffer("lens", I32, BufferRole::Read, Some(1)),
        Param::buffer("out", F32, BufferRole::ReadWrite, Some(1)),
    ];
    let ints = [
        (1, U32),
        (2, U32),
        (13, U32),
        (14, I32),
        (15, U32),
        (16, I32),
    ];
    decode_kernel("attention_paged", params, &ints, 17, column, NSEQ)
}

/// `(q, k_pool, v_pool)`: quarter-integer queries (so the softmax is not one-hot and K matters), small-integer pools; page 8 is poison (huge), so a read of an unused table entry or past a length shows.
pub(crate) fn paged_data() -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let (q, _, _) = attention_data();
    let pool = |width: usize, mul_a: usize, mod_a: usize, skew: usize| -> Vec<f32> {
        (0..NPAGES * PAGE * width)
            .map(|i| {
                if i / (PAGE * width) == 8 {
                    1000.0
                } else {
                    ((i * mul_a + i / width * skew) % mod_a) as f32 - (mod_a / 2) as f32
                }
            })
            .collect()
    };
    (
        q[..NSEQ * D].iter().map(|x| x * 0.25).collect(),
        pool(D, 1, 5, 3),
        pool(DV, 1, 5, 3),
    )
}

/// The host result, gathering each sequence's tokens through the table, in the kernel's `f32` order.
pub(crate) fn paged_reference() -> Vec<f32> {
    let (q, k, v) = paged_data();
    let mut out = vec![0f32; NSEQ * DV];
    for i in 0..NSEQ {
        for d in 0..DV {
            let (mut m, mut l, mut acc) = (NO_SCORE as f32, 0f32, 0f32);
            for t in 0..LENS[i] as usize {
                let r = TABLE[i * MAXP + t / PAGE] as usize * PAGE + t % PAGE;
                let mut dot = 0f32;
                for e in 0..D {
                    dot += q[i * D + e] * k[r * D + e];
                }
                let s = dot * SCALE as f32;
                let mn = m.max(s);
                let alpha = (m - mn).exp();
                let p = (s - mn).exp();
                l = l * alpha + p;
                acc = acc * alpha + p * v[r * DV + d];
                m = mn;
            }
            out[i * DV + d] = if l > 0.0 { acc / l } else { 0.0 };
        }
    }
    out
}
