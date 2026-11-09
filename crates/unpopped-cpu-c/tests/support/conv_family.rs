// COPY of `crates/unpopped/tests/support/conv_family.rs`: a test may not `#[path]` into another crate's directory (sweep SUP-004).
// Change both files together.
// SPDX-License-Identifier: MIT OR Apache-2.0
//! The implicit-GEMM convolution family of ROADMAP R5.7 F7, written with the kernel layer alone: the forward convolution
//! (`fprop`), the gradient with respect to the input (`dgrad`) and the gradient with respect to the filter (`wgrad`), all
//! NHWC with a `[R, S, C, K]` filter, **stride, dilation and padding on both axes**, and no staged im2col matrix (the index
//! math is in the loops). `conv.rs` is the stride-1, dilation-1 forward kernel of U9 K7; this file generalises it and adds
//! the two backward convolutions that CUTLASS names beside it.
//!
//! The shapes make an index mix-up loud: a non-square image and filter, stride `(3, 1)`, dilation `(2, 3)`, padding `(1, 1)`,
//! several channels, and taps that fall outside the image at both ends of both axes. Integer data, so every comparison is
//! exact. Shared by the oracle test (`kernel_conv_family.rs`) and the C test
//! (`unpopped-cpu-c/tests/kernel_conv_family_emit_agrees.rs`, which keeps a copy: change both).

use unpopped::exec::{ExecId, LaunchShape};
use unpopped::ir::{BinaryOp, CastSpec, RoundMode, Saturation, ScalarExpr};
use unpopped::kernel::{Block, BufferRole, KernelFn, LocalDecl, Param, Place, Stmt};
use unpopped_vocab::ElementKind;

pub(crate) const I32: ElementKind = ElementKind::I32;
pub(crate) const N: i64 = 2;
pub(crate) const H: i64 = 6;
pub(crate) const W: i64 = 6;
pub(crate) const C: i64 = 3;
pub(crate) const K: i64 = 2;
pub(crate) const R: i64 = 3;
pub(crate) const S: i64 = 2;
pub(crate) const SH: i64 = 3;
pub(crate) const SW: i64 = 1;
pub(crate) const DH: i64 = 2;
pub(crate) const DW: i64 = 3;
pub(crate) const PH: i64 = 1;
pub(crate) const PW: i64 = 1;
pub(crate) const P: i64 = (H + 2 * PH - DH * (R - 1) - 1) / SH + 1;
pub(crate) const Q: i64 = (W + 2 * PW - DW * (S - 1) - 1) / SW + 1;

/// The geometry of one spatial axis: the input coordinate of output `o` and filter tap `t` is `o * stride + t * dil - pad`,
/// valid inside `0..limit`.
#[derive(Clone, Copy)]
struct Axis {
    stride: i64,
    dil: i64,
    pad: i64,
    limit: i64,
}

const ROWS: Axis = Axis {
    stride: SH,
    dil: DH,
    pad: PH,
    limit: H,
};
const COLS: Axis = Axis {
    stride: SW,
    dil: DW,
    pad: PW,
    limit: W,
};

fn ci(v: i64) -> ScalarExpr {
    ScalarExpr::ConstInt(v, I32)
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

fn cmp(op: BinaryOp, a: ScalarExpr, b: ScalarExpr) -> ScalarExpr {
    ScalarExpr::Binary(op, Box::new(a), Box::new(b))
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

fn at4(buffer: u32, c: [ScalarExpr; 4]) -> Place {
    Place::Logical {
        buffer,
        coords: c.to_vec(),
    }
}

fn counted(local: u32, end: i64, body: Vec<Stmt>) -> Stmt {
    Stmt::For {
        local,
        start: ci(0),
        end: ci(end),
        step: ci(1),
        body: Block(body),
    }
}

fn when(cond: ScalarExpr, then: Vec<Stmt>) -> Stmt {
    Stmt::If {
        cond,
        then: Block(then),
        els: Block(vec![]),
    }
}

/// `if 0 <= x < limit { then }`.
fn within(x: ScalarExpr, limit: i64, then: Vec<Stmt>) -> Stmt {
    when(
        cmp(BinaryOp::CmpLt, ci(-1), x.clone()),
        vec![when(cmp(BinaryOp::CmpLt, x, ci(limit)), then)],
    )
}

fn let_(local: u32, init: ScalarExpr) -> Stmt {
    Stmt::Let { local, init }
}

fn load(local: u32, place: Place) -> Stmt {
    Stmt::Load { local, place }
}

/// `acc += a * b`.
fn mac(acc: u32, a: u32, b: u32) -> Stmt {
    Stmt::Assign {
        local: acc,
        value: add(local(acc), mul(local(a), local(b))),
    }
}

fn params(names: [(&str, BufferRole); 3]) -> Vec<Param> {
    names
        .into_iter()
        .map(|(n, role)| Param::buffer(n, I32, role, Some(4)))
        .collect()
}

fn locals(n: usize) -> Vec<LocalDecl> {
    (0..n).map(|_| LocalDecl::scalar(I32)).collect()
}

/// The input coordinate of output position `o` and filter tap `t`: `o * stride + t * dilation - pad`.
fn coord(o: ScalarExpr, t: ScalarExpr, stride: i64, dil: i64, pad: i64) -> ScalarExpr {
    sub(add(mul(o, ci(stride)), mul(t, ci(dil))), ci(pad))
}

/// Forward: buffers 0 `input` `[N, H, W, C]`, 1 `filter` `[R, S, C, K]`, 2 `out` `[N, P, Q, K]`. Block `n`; thread
/// `(x, y, z)` is output `(q, p, k)`. Locals: 0 `r`, 1 `s`, 2 `ih`, 3 `iw`, 4 `c`, 5 input, 6 filter, 7 acc.
pub(crate) fn fprop() -> KernelFn {
    let n = exec_i32(ExecId::BlockIdx(0));
    let (q, p, k) = (
        exec_i32(ExecId::ThreadIdx(0)),
        exec_i32(ExecId::ThreadIdx(1)),
        exec_i32(ExecId::ThreadIdx(2)),
    );
    let tap = vec![counted(
        4,
        C,
        vec![
            load(5, at4(0, [n.clone(), local(2), local(3), local(4)])),
            load(6, at4(1, [local(0), local(1), local(4), k.clone()])),
            mac(7, 5, 6),
        ],
    )];
    let s_body = vec![
        let_(3, coord(q.clone(), local(1), SW, DW, PW)),
        within(local(3), W, tap),
    ];
    let r_body = vec![
        let_(2, coord(p.clone(), local(0), SH, DH, PH)),
        within(local(2), H, vec![counted(1, S, s_body)]),
    ];
    let body = vec![
        let_(7, ci(0)),
        counted(0, R, r_body),
        Stmt::Store {
            place: at4(2, [n, p, q, k]),
            value: local(7),
        },
    ];
    KernelFn::new(
        "fprop",
        I32,
        params([
            ("input", BufferRole::Read),
            ("filter", BufferRole::Read),
            ("out", BufferRole::ReadWrite),
        ]),
        locals(8),
        Block(body),
    )
    .with_launch(LaunchShape::new(
        [Q as u32, P as u32, K as u32],
        [N as u32, 1, 1],
    ))
}

/// The output position `o` that a filter tap `t` reads input coordinate `i` from, if any: `(i + pad - t * dilation)` must be
/// non-negative and a multiple of the stride, and the quotient inside `0..limit`. Sets local `num` and `o`, then runs `then`.
fn source(
    i: ScalarExpr,
    t: ScalarExpr,
    axis: Axis,
    (num, o): (u32, u32),
    then: Vec<Stmt>,
) -> Vec<Stmt> {
    let Axis {
        stride,
        dil,
        pad,
        limit,
    } = axis;
    vec![
        let_(num, sub(add(i, ci(pad)), mul(t, ci(dil)))),
        when(
            cmp(BinaryOp::CmpLt, ci(-1), local(num)),
            vec![
                let_(o, div(local(num), ci(stride))),
                when(
                    cmp(BinaryOp::CmpEq, mul(local(o), ci(stride)), local(num)),
                    vec![when(cmp(BinaryOp::CmpLt, local(o), ci(limit)), then)],
                ),
            ],
        ),
    ]
}

/// The sum over `k` of `d_out[n][p][q][k] * filter[r][s][c][k]`, for the `(p, q)` that `source` set in locals 5 and 6.
fn reduce_k(n: ScalarExpr, c: ScalarExpr) -> Vec<Stmt> {
    vec![counted(
        2,
        K,
        vec![
            load(7, at4(0, [n, local(5), local(6), local(2)])),
            load(8, at4(1, [local(0), local(1), c, local(2)])),
            mac(9, 7, 8),
        ],
    )]
}

/// Gradient with respect to the input: buffers 0 `d_out` `[N, P, Q, K]`, 1 `filter` `[R, S, C, K]`, 2 `d_in`
/// `[N, H, W, C]`. Block `n`; thread `(x, y, z)` is input `(w, h, c)`. Each input element sums, over the taps `(r, s)` that
/// reach it, `d_out[n][p][q][k] * filter[r][s][c][k]` where `p * SH + r * DH - PH == h` (and likewise for `q`). Locals: 0 `r`,
/// 1 `s`, 2 `k`, 3 `pn`, 4 `qn`, 5 `p`, 6 `q`, 7 `d_out`, 8 filter, 9 acc.
pub(crate) fn dgrad() -> KernelFn {
    let n = exec_i32(ExecId::BlockIdx(0));
    let (w, h, c) = (
        exec_i32(ExecId::ThreadIdx(0)),
        exec_i32(ExecId::ThreadIdx(1)),
        exec_i32(ExecId::ThreadIdx(2)),
    );
    let s_body = source(
        w.clone(),
        local(1),
        Axis { limit: Q, ..COLS },
        (4, 6),
        reduce_k(n.clone(), c.clone()),
    );
    let r_body = source(
        h.clone(),
        local(0),
        Axis { limit: P, ..ROWS },
        (3, 5),
        vec![counted(1, S, s_body)],
    );
    let body = vec![
        let_(9, ci(0)),
        counted(0, R, r_body),
        Stmt::Store {
            place: at4(2, [n, h, w, c]),
            value: local(9),
        },
    ];
    KernelFn::new(
        "dgrad",
        I32,
        params([
            ("d_out", BufferRole::Read),
            ("filter", BufferRole::Read),
            ("d_in", BufferRole::ReadWrite),
        ]),
        locals(10),
        Block(body),
    )
    .with_launch(LaunchShape::new(
        [W as u32, H as u32, C as u32],
        [N as u32, 1, 1],
    ))
}

/// Gradient with respect to the filter: buffers 0 `input` `[N, H, W, C]`, 1 `d_out` `[N, P, Q, K]`, 2 `d_filter`
/// `[R, S, C, K]`. Block `r`; thread `(x, y, z)` is filter `(s, c, k)`; the loops run over `n`, `p` and `q`. Locals: 0 `n`,
/// 1 `p`, 2 `q`, 3 `ih`, 4 `iw`, 5 input, 6 `d_out`, 7 acc.
pub(crate) fn wgrad() -> KernelFn {
    let r = exec_i32(ExecId::BlockIdx(0));
    let (s, c, k) = (
        exec_i32(ExecId::ThreadIdx(0)),
        exec_i32(ExecId::ThreadIdx(1)),
        exec_i32(ExecId::ThreadIdx(2)),
    );
    let tap = vec![
        load(5, at4(0, [local(0), local(3), local(4), c.clone()])),
        load(6, at4(1, [local(0), local(1), local(2), k.clone()])),
        mac(7, 5, 6),
    ];
    let q_body = vec![
        let_(4, coord(local(2), s.clone(), SW, DW, PW)),
        within(local(4), W, tap),
    ];
    let p_body = vec![
        let_(3, coord(local(1), r.clone(), SH, DH, PH)),
        within(local(3), H, vec![counted(2, Q, q_body)]),
    ];
    let body = vec![
        let_(7, ci(0)),
        counted(0, N, vec![counted(1, P, p_body)]),
        Stmt::Store {
            place: at4(2, [r, s, c, k]),
            value: local(7),
        },
    ];
    KernelFn::new(
        "wgrad",
        I32,
        params([
            ("input", BufferRole::Read),
            ("d_out", BufferRole::Read),
            ("d_filter", BufferRole::ReadWrite),
        ]),
        locals(8),
        Block(body),
    )
    .with_launch(LaunchShape::new(
        [S as u32, C as u32, K as u32],
        [R as u32, 1, 1],
    ))
}

/// Deterministic mixed-sign data: `(input, filter, d_out)`.
pub(crate) fn data() -> (Vec<i32>, Vec<i32>, Vec<i32>) {
    let input = (0..N * H * W * C)
        .map(|i| ((i * 7 + 3) % 11) as i32 - 5)
        .collect();
    let filter = (0..R * S * C * K)
        .map(|i| ((i * 5 + 1) % 7) as i32 - 3)
        .collect();
    let d_out = (0..N * P * Q * K)
        .map(|i| ((i * 3 + 2) % 9) as i32 - 4)
        .collect();
    (input, filter, d_out)
}

fn idx4(d: [i64; 4], i: [i64; 4]) -> usize {
    (((i[0] * d[1] + i[1]) * d[2] + i[2]) * d[3] + i[3]) as usize
}

/// The input position of output `o`, tap `t`, if it lies inside `0..limit`.
fn src(o: i64, t: i64, axis: Axis) -> Option<i64> {
    let i = o * axis.stride + t * axis.dil - axis.pad;
    (0..axis.limit).contains(&i).then_some(i)
}

/// The host references, each a **scatter** over every `(n, p, q, r, s, c, k)` (the kernels gather), so the loop structure
/// differs from the kernel's: `(fprop, dgrad, wgrad)` outputs for `data()`.
pub(crate) fn reference(
    input: &[i32],
    filter: &[i32],
    d_out: &[i32],
) -> (Vec<i32>, Vec<i32>, Vec<i32>) {
    let mut out = vec![0; (N * P * Q * K) as usize];
    let mut d_in = vec![0; (N * H * W * C) as usize];
    let mut d_filter = vec![0; (R * S * C * K) as usize];
    for n in 0..N {
        for p in 0..P {
            for q in 0..Q {
                for r in 0..R {
                    for s in 0..S {
                        let (Some(ih), Some(iw)) = (src(p, r, ROWS), src(q, s, COLS)) else {
                            continue;
                        };
                        for c in 0..C {
                            for k in 0..K {
                                let x = input[idx4([N, H, W, C], [n, ih, iw, c])];
                                let f = filter[idx4([R, S, C, K], [r, s, c, k])];
                                let g = d_out[idx4([N, P, Q, K], [n, p, q, k])];
                                out[idx4([N, P, Q, K], [n, p, q, k])] += x * f;
                                d_in[idx4([N, H, W, C], [n, ih, iw, c])] += g * f;
                                d_filter[idx4([R, S, C, K], [r, s, c, k])] += x * g;
                            }
                        }
                    }
                }
            }
        }
    }
    (out, d_in, d_filter)
}
