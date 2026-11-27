// SPDX-License-Identifier: MIT OR Apache-2.0
//! Decode attention with grouped KV heads (GQA) and with a latent KV cache (MLA), at the kernel layer (CUTLASS survey row 92;
//! the style of `support/attention.rs`, which this file does not edit). One query token, one thread per query head, a streamed
//! (online) softmax over `L` cached tokens.
//!
//! GQA (Ainslie et al. 2023, arXiv:2305.13245): `HQ` query heads share `HKV` key/value heads, `G = HQ / HKV` query heads per
//! KV head; query head `h` reads KV head `h / G`. The cache is `[L, HKV, D]`.
//!
//! MLA (DeepSeek-V2, arXiv:2405.04434, the absorbed decode form): the cache holds a latent `c[j]` of width `R` and a shared
//! rope key `kr[j]` of width `RP`. With the key up-projection absorbed into the query (`q_abs[h] = W_UK[h]^T q_nope[h]`, done
//! on the host), `score[h][j] = scale * (q_abs[h] . c[j] + q_rope[h] . kr[j])`, the softmax-weighted latent is
//! `lat[h] = sum_j p[h][j] c[j]`, and the output is up-projected `out[h][d] = sum_r W_UV[h][d][r] lat[h][r]`.
//! UNVERIFIED against the DeepSeek or FlashMLA kernels: only the papers' algebra is used.

use unpopped::exec::{ExecId, LaunchShape};
use unpopped::ir::{BinaryOp, ScalarExpr, UnaryOp};
use unpopped::kernel::{Block, BufferRole, KernelFn, LocalDecl, Param, Place, Stmt, Unroll};
use unpopped_vocab::ElementKind;

pub(crate) const F32: ElementKind = ElementKind::F32;
pub(crate) const U32: ElementKind = ElementKind::U32;
/// GQA: query heads, KV heads, head dim of K, head dim of V, cached tokens.
pub(crate) const HQ: usize = 4;
pub(crate) const HKV: usize = 2;
pub(crate) const D: usize = 4;
pub(crate) const DV: usize = 3;
pub(crate) const L: usize = 5;
/// MLA: heads, latent width, rope width (`L` and `DV` as above).
pub(crate) const H: usize = 3;
pub(crate) const R: usize = 4;
pub(crate) const RP: usize = 2;
pub(crate) const SCALE: f32 = 0.5;
const NO_SCORE: f64 = -3.0e38;

/// What the kernel is built wrong, to show the test can fail.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum Mutant {
    None,
    /// GQA: KV head `h % HKV` instead of `h / G` (interleaved rather than grouped).
    InterleavedKvHead,
    /// GQA: every query head reads KV head 0 (the MQA reading).
    SingleKvHead,
    /// MLA: the rope score term is dropped.
    NoRope,
    /// MLA: the up-projection reads `W_UV[h][r][d]` (transposed).
    TransposedUp,
}

fn cu(v: usize) -> ScalarExpr {
    ScalarExpr::ConstInt(v as i64, U32)
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
fn exp(x: ScalarExpr) -> ScalarExpr {
    ScalarExpr::Unary(UnaryOp::Exp, Box::new(x))
}
fn head() -> ScalarExpr {
    ScalarExpr::Exec(ExecId::ThreadIdx(0))
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
fn counted(local_id: u32, end: usize, body: Vec<Stmt>) -> Stmt {
    Stmt::For {
        unroll: Unroll::None,
        local: local_id,
        start: cu(0),
        end: cu(end),
        step: cu(1),
        body: Block(body),
    }
}
fn let_(local: u32, init: ScalarExpr) -> Stmt {
    Stmt::Let { local, init }
}
fn assign(local: u32, value: ScalarExpr) -> Stmt {
    Stmt::Assign { local, value }
}

// Locals: 0 dot, 1 j, 2 column, 3 x, 4 y, 5 m, 6 sum, 7 score, 8 value, 9 m', 10 alpha, 11 p, 12 acc, 13 e, 14 kv head (GQA) or
// output column d (MLA), 15 r (MLA), 16 latent (MLA).

/// The online update for one cached token, given the score in local 7 and the value in local 8.
fn online_tail() -> Vec<Stmt> {
    vec![
        let_(
            9,
            ScalarExpr::Binary(BinaryOp::Max, Box::new(l(5)), Box::new(l(7))),
        ),
        let_(10, exp(sub(l(5), l(9)))),
        let_(11, exp(sub(l(7), l(9)))),
        assign(6, add(mul(l(6), l(10)), l(11))),
        assign(12, add(mul(l(12), l(10)), mul(l(11), l(8)))),
        assign(5, l(9)),
    ]
}

/// `dot += x[xi(e)] * y[yi(e)]` over `e in 0..n`, into local 0 (zeroed first).
fn dot(
    fresh: bool,
    n: usize,
    xb: u32,
    xi: impl Fn(ScalarExpr) -> ScalarExpr,
    yb: u32,
    yi: impl Fn(ScalarExpr) -> ScalarExpr,
) -> Vec<Stmt> {
    vec![
        if fresh {
            let_(0, ScalarExpr::Const(0.0))
        } else {
            assign(0, ScalarExpr::Const(0.0))
        },
        counted(
            13,
            n,
            vec![
                load(3, xb, xi(l(13))),
                load(4, yb, yi(l(13))),
                assign(0, add(l(0), mul(l(3), l(4)))),
            ],
        ),
    ]
}

fn start() -> Vec<Stmt> {
    vec![
        let_(5, ScalarExpr::Const(NO_SCORE)),
        let_(6, ScalarExpr::Const(0.0)),
        let_(12, ScalarExpr::Const(0.0)),
    ]
}

fn locals() -> Vec<LocalDecl> {
    let mut v = vec![LocalDecl::scalar(F32); 17];
    for i in [1, 2, 13, 14, 15] {
        v[i] = LocalDecl::scalar(U32);
    }
    v
}

/// GQA decode. Buffers: 0 `q[HQ*D]`, 1 `k[L*HKV*D]`, 2 `v[L*HKV*DV]` (read), 3 `out[HQ*DV]`; all `F32`. Thread `h` is query head `h`.
pub(crate) fn gqa_decode(mutant: Mutant) -> KernelFn {
    let kv_head = match mutant {
        Mutant::InterleavedKvHead => sub(head(), mul(div(head(), cu(HKV)), cu(HKV))),
        Mutant::SingleKvHead => cu(0),
        _ => div(head(), cu(HQ / HKV)),
    };
    let kv_row = |j: ScalarExpr| add(mul(j, cu(HKV)), l(14));
    let mut step = dot(
        true,
        D,
        0,
        |e| add(mul(head(), cu(D)), e),
        1,
        |e| add(mul(kv_row(l(1)), cu(D)), e),
    );
    step.push(let_(7, mul(l(0), ScalarExpr::Const(f64::from(SCALE)))));
    step.push(load(8, 2, add(mul(kv_row(l(1)), cu(DV)), l(2))));
    step.extend(online_tail());
    let mut col = start();
    col.push(counted(1, L, step));
    col.push(Stmt::Store {
        place: flat(3, add(mul(head(), cu(DV)), l(2))),
        value: div(l(12), l(6)),
    });
    KernelFn::new(
        "attention_gqa_decode",
        F32,
        vec![
            Param::buffer("q", F32, BufferRole::Read, Some(1)),
            Param::buffer("k", F32, BufferRole::Read, Some(1)),
            Param::buffer("v", F32, BufferRole::Read, Some(1)),
            Param::buffer("out", F32, BufferRole::ReadWrite, Some(1)),
        ],
        locals(),
        Block(vec![let_(14, kv_head), counted(2, DV, col)]),
    )
    .with_launch(LaunchShape::new([HQ as u32, 1, 1], [1, 1, 1]))
}

/// The per-token step of the MLA latent pass: the score (latent dot plus rope dot, scaled) into local 7, the latent value of the
/// current column into local 8, then the online update.
fn mla_token_step(mutant: Mutant) -> Vec<Stmt> {
    let mut step = dot(
        true,
        R,
        0,
        |e| add(mul(head(), cu(R)), e),
        2,
        |e| add(mul(l(1), cu(R)), e),
    );
    // the rope term accumulates into the same score: save the latent dot in 7, then dot the rope parts
    step.push(let_(7, l(0)));
    if mutant != Mutant::NoRope {
        step.extend(dot(
            false,
            RP,
            1,
            |e| add(mul(head(), cu(RP)), e),
            3,
            |e| add(mul(l(1), cu(RP)), e),
        ));
        step.push(assign(7, add(l(7), l(0))));
    }
    step.push(assign(7, mul(l(7), ScalarExpr::Const(f64::from(SCALE)))));
    step.push(load(8, 2, add(mul(l(1), cu(R)), l(2))));
    step.extend(online_tail());
    step
}

/// The latent pass for one latent column (local 2): a streamed softmax over the tokens, `lat[h][r] = acc / sum`.
fn mla_latent_column(mutant: Mutant) -> Vec<Stmt> {
    let mut latent = start();
    latent.push(counted(1, L, mla_token_step(mutant)));
    latent.push(Stmt::Store {
        place: flat(5, add(mul(head(), cu(R)), l(2))),
        value: div(l(12), l(6)),
    });
    latent
}

/// The up-projection `out[h][d] = sum_r W_UV[h][d][r] lat[h][r]` (column `d` is local 14, `r` is local 15).
fn mla_up_projection(mutant: Mutant) -> Stmt {
    let w_at = if mutant == Mutant::TransposedUp {
        add(mul(add(mul(head(), cu(R)), l(15)), cu(DV)), l(14))
    } else {
        add(mul(add(mul(head(), cu(DV)), l(14)), cu(R)), l(15))
    };
    counted(
        14,
        DV,
        vec![
            let_(0, ScalarExpr::Const(0.0)),
            counted(
                15,
                R,
                vec![
                    load(3, 4, w_at),
                    load(16, 5, add(mul(head(), cu(R)), l(15))),
                    assign(0, add(l(0), mul(l(3), l(16)))),
                ],
            ),
            Stmt::Store {
                place: flat(6, add(mul(head(), cu(DV)), l(14))),
                value: l(0),
            },
        ],
    )
}

/// MLA absorbed decode. Buffers: 0 `q_abs[H*R]`, 1 `q_rope[H*RP]`, 2 `c[L*R]`, 3 `k_rope[L*RP]`, 4 `w_uv[H*DV*R]` (read),
/// 5 `lat[H*R]` scratch (each thread writes and reads only its own row), 6 `out[H*DV]`; all `F32`. Thread `h` is head `h`.
pub(crate) fn mla_decode(mutant: Mutant) -> KernelFn {
    KernelFn::new(
        "attention_mla_decode",
        F32,
        vec![
            Param::buffer("q_abs", F32, BufferRole::Read, Some(1)),
            Param::buffer("q_rope", F32, BufferRole::Read, Some(1)),
            Param::buffer("c", F32, BufferRole::Read, Some(1)),
            Param::buffer("k_rope", F32, BufferRole::Read, Some(1)),
            Param::buffer("w_uv", F32, BufferRole::Read, Some(1)),
            Param::buffer("lat", F32, BufferRole::ReadWrite, Some(1)),
            Param::buffer("out", F32, BufferRole::ReadWrite, Some(1)),
        ],
        locals(),
        Block(vec![
            counted(2, R, mla_latent_column(mutant)),
            mla_up_projection(mutant),
        ]),
    )
    .with_launch(LaunchShape::new([H as u32, 1, 1], [1, 1, 1]))
}

/// Deterministic, non-degenerate data: `n` values from a small-integer pattern, scaled to quarters.
pub(crate) fn pattern(n: usize, mul_a: usize, modulus: usize, offset: usize) -> Vec<f32> {
    (0..n)
        .map(|i| {
            (((i * mul_a + i / 3 * 2 + offset) % modulus) as f32 - (modulus / 2) as f32) * 0.25
        })
        .collect()
}
