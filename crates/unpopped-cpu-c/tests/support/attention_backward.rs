// SPDX-License-Identifier: MIT OR Apache-2.0
//! The attention backward conformance kernels of ROADMAP U10 A5 (`docs/options-u10-attention.md`), written with the kernel
//! layer alone. Given the forward inputs `q`, `k`, `v`, the forward output `o`, the upstream gradient `dout` and the saved
//! row log-sum-exp `lse`, each thread recomputes `P[i][j] = exp(S[i][j] - lse[i])` (no score matrix is stored), the row
//! constant `delta[i] = sum_d dout[i][d] * o[i][d]`, `dP[i][j] = dout[i] . v[j]` and `dS = P * (dP - delta)`. One kernel owns a
//! query row (`dQ`), the other a key row (`dK`, `dV`), so neither needs an atomic. Shared by the oracle test
//! (`kernel_attention_backward.rs`) and the C test (`unpopped-cpu-c/tests/kernel_attention_backward_emit_agrees.rs`), which
//! includes this very file. It reads `attention` (the forward's shapes and data) as a sibling module.

use super::attention::{D, DV, F32, LK, LQ, SCALE, U32, attention_data, attention_reference};
use unpopped::exec::{ExecId, LaunchShape};
use unpopped::ir::{ScalarExpr, UnaryOp};
use unpopped::kernel::{Block, BufferRole, KernelFn, LocalDecl, Param, Place, Stmt};

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
fn row() -> ScalarExpr {
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
        local: local_id,
        start: cu(0),
        end: cu(end),
        step: cu(1),
        body: Block(body),
    }
}
fn zero(local: u32) -> Stmt {
    Stmt::Let {
        local,
        init: ScalarExpr::Const(0.0),
    }
}

// Buffers: 0 q `[LQ*D]`, 1 k `[LK*D]`, 2 v `[LK*DV]`, 3 o `[LQ*DV]`, 4 dout `[LQ*DV]`, 5 lse `[LQ]` (all read); then the outputs.
// Locals: 0 acc, 1 outer index, 2 output column, 3 score, 4 inner index, 7 p, 8 delta, 9 dP, 10 dS, 11 and 12 loaded
// operands, 13 lse.

/// `p` (local 7) and `dS` (local 10) for the pair (query row `i`, key row `j`), by the recompute formulas above.
fn pair_gradients(i: ScalarExpr, j: ScalarExpr) -> Vec<Stmt> {
    let dot = |acc: u32, n: usize, a: (u32, ScalarExpr, usize), b: (u32, ScalarExpr, usize)| {
        vec![
            zero(acc),
            counted(
                4,
                n,
                vec![
                    load(11, a.0, add(mul(a.1, cu(a.2)), local(4))),
                    load(12, b.0, add(mul(b.1, cu(b.2)), local(4))),
                    Stmt::Assign {
                        local: acc,
                        value: add(local(acc), mul(local(11), local(12))),
                    },
                ],
            ),
        ]
    };
    let mut b = dot(3, D, (0, i.clone(), D), (1, j.clone(), D));
    b.push(Stmt::Assign {
        local: 3,
        value: mul(local(3), ScalarExpr::Const(SCALE)),
    });
    b.push(load(13, 5, i.clone()));
    b.push(Stmt::Let {
        local: 7,
        init: ScalarExpr::Unary(UnaryOp::Exp, Box::new(sub(local(3), local(13)))),
    });
    b.extend(dot(9, DV, (4, i.clone(), DV), (2, j, DV)));
    b.extend(dot(8, DV, (4, i.clone(), DV), (3, i, DV)));
    b.push(Stmt::Let {
        local: 10,
        init: mul(local(7), sub(local(9), local(8))),
    });
    b
}

fn locals() -> Vec<LocalDecl> {
    let mut l = vec![LocalDecl::scalar(F32); 14];
    for u in [1, 2, 4] {
        l[u] = LocalDecl::scalar(U32);
    }
    l
}

fn reads() -> Vec<Param> {
    ["q", "k", "v", "o", "dout", "lse"]
        .into_iter()
        .map(|n| Param::buffer(n, F32, BufferRole::Read, Some(1)))
        .collect()
}

/// `dQ[i] = SCALE * sum_j dS[i][j] * K[j]`, one thread per query row. Output buffer 6 `dq` `[LQ*D]`.
pub(crate) fn backward_dq() -> KernelFn {
    let column = counted(
        2,
        D,
        vec![
            zero(0),
            counted(1, LK, {
                let mut s = pair_gradients(row(), local(1));
                s.push(load(12, 1, add(mul(local(1), cu(D)), local(2))));
                s.push(Stmt::Assign {
                    local: 0,
                    value: add(local(0), mul(local(10), local(12))),
                });
                s
            }),
            Stmt::Store {
                place: flat(6, add(mul(row(), cu(D)), local(2))),
                value: mul(local(0), ScalarExpr::Const(SCALE)),
            },
        ],
    );
    let mut params = reads();
    params.push(Param::buffer("dq", F32, BufferRole::ReadWrite, Some(1)));
    KernelFn::new(
        "attention_backward_dq",
        F32,
        params,
        locals(),
        Block(vec![column]),
    )
    .with_launch(LaunchShape::new([LQ as u32, 1, 1], [1, 1, 1]))
}

/// `dK[j] = SCALE * sum_i dS[i][j] * Q[i]` and `dV[j] = sum_i P[i][j] * dout[i]`, one thread per key row.
/// Output buffers 6 `dk` `[LK*D]` and 7 `dv` `[LK*DV]`.
pub(crate) fn backward_dkv() -> KernelFn {
    let mut params = reads();
    params.push(Param::buffer("dk", F32, BufferRole::ReadWrite, Some(1)));
    params.push(Param::buffer("dv", F32, BufferRole::ReadWrite, Some(1)));
    KernelFn::new(
        "attention_backward_dkv",
        F32,
        params,
        locals(),
        Block(vec![dk_columns(), dv_columns()]),
    )
    .with_launch(LaunchShape::new([LK as u32, 1, 1], [1, 1, 1]))
}

/// The pair gradients for (query row `local 1`, key row `row()`), then `tail`.
fn key_pair(tail: Vec<Stmt>) -> Vec<Stmt> {
    let mut s = pair_gradients(local(1), row());
    s.extend(tail);
    s
}

/// `dK[row][e] = SCALE * sum_i dS[i][row] * Q[i][e]` for every `e`.
fn dk_columns() -> Stmt {
    counted(
        2,
        D,
        vec![
            zero(0),
            counted(
                1,
                LQ,
                key_pair(vec![
                    load(12, 0, add(mul(local(1), cu(D)), local(2))),
                    Stmt::Assign {
                        local: 0,
                        value: add(local(0), mul(local(10), local(12))),
                    },
                ]),
            ),
            Stmt::Store {
                place: flat(6, add(mul(row(), cu(D)), local(2))),
                value: mul(local(0), ScalarExpr::Const(SCALE)),
            },
        ],
    )
}

/// `dV[row][c] = sum_i P[i][row] * dout[i][c]` for every `c`.
fn dv_columns() -> Stmt {
    counted(
        2,
        DV,
        vec![
            zero(0),
            counted(
                1,
                LQ,
                key_pair(vec![
                    load(12, 4, add(mul(local(1), cu(DV)), local(2))),
                    Stmt::Assign {
                        local: 0,
                        value: add(local(0), mul(local(7), local(12))),
                    },
                ]),
            ),
            Stmt::Store {
                place: flat(7, add(mul(row(), cu(DV)), local(2))),
                value: local(0),
            },
        ],
    )
}

/// The upstream gradient: small integers, different per row and column, so no gradient is symmetric by accident.
pub(crate) fn dout_data() -> Vec<f32> {
    (0..LQ * DV)
        .map(|i| ((i / DV * 5 + i % DV * 3) % 7) as f32 - 3.0)
        .collect()
}

/// `(q, k, v, o, dout, lse)`: the forward result `o` and the row log-sum-exp from the host forward.
pub(crate) fn backward_inputs() -> [Vec<f32>; 6] {
    let (q, k, v) = attention_data();
    let (s, o) = attention_reference();
    let lse = (0..LQ)
        .map(|i| {
            let r = &s[i * LK..(i + 1) * LK];
            let m = r.iter().cloned().fold(r[0], f32::max);
            m + r.iter().map(|x| (x - m).exp()).sum::<f32>().ln()
        })
        .collect();
    [q, k, v, o, dout_data(), lse]
}

/// The loss `L = sum(dout * attention(q, k, v))` in `f64`, for the finite-difference control.
pub(crate) fn loss_f64(q: &[f64], k: &[f64], v: &[f64], dout: &[f64]) -> f64 {
    let mut l = 0.0;
    for i in 0..LQ {
        let s: Vec<f64> = (0..LK)
            .map(|j| SCALE * (0..D).map(|e| q[i * D + e] * k[j * D + e]).sum::<f64>())
            .collect();
        let m = s.iter().cloned().fold(f64::MIN, f64::max);
        let z: f64 = s.iter().map(|x| (x - m).exp()).sum();
        for d in 0..DV {
            let o: f64 = (0..LK).map(|j| (s[j] - m).exp() / z * v[j * DV + d]).sum();
            l += dout[i * DV + d] * o;
        }
    }
    l
}

/// `(P, O)` in `f64`: the row softmax of the scaled scores and its product with `V`.
fn softmax_and_output(q: &[f64], k: &[f64], v: &[f64]) -> ([f64; LQ * LK], [f64; LQ * DV]) {
    let mut p = [0f64; LQ * LK];
    let mut o = [0f64; LQ * DV];
    for i in 0..LQ {
        let s: Vec<f64> = (0..LK)
            .map(|j| SCALE * (0..D).map(|e| q[i * D + e] * k[j * D + e]).sum::<f64>())
            .collect();
        let m = s.iter().cloned().fold(f64::MIN, f64::max);
        let z: f64 = s.iter().map(|x| (x - m).exp()).sum();
        for j in 0..LK {
            p[i * LK + j] = (s[j] - m).exp() / z;
        }
        for d in 0..DV {
            o[i * DV + d] = (0..LK).map(|j| p[i * LK + j] * v[j * DV + d]).sum();
        }
    }
    (p, o)
}

/// `(dQ, dK, dV)` in `f64`, written as whole matrices (`P`, `dP = dout V^T`, `dS`, then three products), a different
/// organisation from the kernels' per-pair recompute.
pub(crate) fn backward_reference() -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let (q, k, v) = attention_data();
    let dout = dout_data();
    let (q, k, v, dout): (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) = (
        q.iter().map(|&x| x as f64).collect(),
        k.iter().map(|&x| x as f64).collect(),
        v.iter().map(|&x| x as f64).collect(),
        dout.iter().map(|&x| x as f64).collect(),
    );
    let (p, o) = softmax_and_output(&q, &k, &v);
    let mut ds = [0f64; LQ * LK];
    for i in 0..LQ {
        let delta: f64 = (0..DV).map(|d| dout[i * DV + d] * o[i * DV + d]).sum();
        for j in 0..LK {
            let dp: f64 = (0..DV).map(|d| dout[i * DV + d] * v[j * DV + d]).sum();
            ds[i * LK + j] = p[i * LK + j] * (dp - delta);
        }
    }
    let dq = (0..LQ * D)
        .map(|x| {
            SCALE
                * (0..LK)
                    .map(|j| ds[x / D * LK + j] * k[j * D + x % D])
                    .sum::<f64>()
        })
        .collect();
    let dk = (0..LK * D)
        .map(|x| {
            SCALE
                * (0..LQ)
                    .map(|i| ds[i * LK + x / D] * q[i * D + x % D])
                    .sum::<f64>()
        })
        .collect();
    let dv = (0..LK * DV)
        .map(|x| {
            (0..LQ)
                .map(|i| p[i * LK + x / DV] * dout[i * DV + x % DV])
                .sum::<f64>()
        })
        .collect();
    (dq, dk, dv)
}
