// SPDX-License-Identifier: MIT OR Apache-2.0
//! Grouped, depthwise and transposed convolution at the kernel layer (CUTLASS survey rows 46 and 48; `support/conv_family.rs` is
//! the dense 2D family and is not edited). NHWC activations, integer data (exact), index math in the loops, no staged matrix.
//!
//! * `grouped_fprop(geo, mutant)`: forward, filter `[K, R, S, C/G]` (the grouped KRSC layout), stride `(2, 1)`, dilation
//!   `(1, 2)`, asymmetric padding top 1 / bottom 0 and left 2 / right 1. Output channel `k` belongs to group `k / (K/G)` and
//!   reads input channels `group * (C/G) .. (group + 1) * (C/G)`. `G = C = K` is the depthwise convolution.
//! * `transposed_fprop(mutant)`: the transposed convolution as a gather, stride `(2, 1)`, dilation `(1, 2)`, padding 1, with
//!   an output padding of `(1, 1)` that makes the output larger than the smallest one the stride allows.

use unpopped::exec::{ExecId, LaunchShape};
use unpopped::ir::{BinaryOp, CastSpec, RoundMode, Saturation, ScalarExpr};
use unpopped::kernel::{Block, BufferRole, KernelFn, LocalDecl, Param, Place, Stmt, Unroll};
use unpopped_vocab::ElementKind;

pub(crate) const I32: ElementKind = ElementKind::I32;
pub(crate) const N: i64 = 2;
pub(crate) const H: i64 = 5;
pub(crate) const W: i64 = 5;
pub(crate) const R: i64 = 3;
pub(crate) const S: i64 = 2;
pub(crate) const SH: i64 = 2;
pub(crate) const SW: i64 = 1;
pub(crate) const DH: i64 = 1;
pub(crate) const DW: i64 = 2;
/// Asymmetric padding of the forward convolution: top, bottom, left, right.
pub(crate) const PT: i64 = 1;
pub(crate) const PB: i64 = 0;
pub(crate) const PL: i64 = 2;
pub(crate) const PR: i64 = 1;
pub(crate) const P: i64 = (H + PT + PB - DH * (R - 1) - 1) / SH + 1;
pub(crate) const Q: i64 = (W + PL + PR - DW * (S - 1) - 1) / SW + 1;

/// A forward-convolution channel geometry: input channels, output channels, groups.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Geo {
    pub(crate) c: i64,
    pub(crate) k: i64,
    pub(crate) g: i64,
}

pub(crate) const GROUPED: Geo = Geo { c: 4, k: 6, g: 2 };
pub(crate) const DEPTHWISE: Geo = Geo { c: 4, k: 4, g: 4 };

/// Transposed convolution: input extent, channels in and out, padding, output padding.
pub(crate) const TH: i64 = 3;
pub(crate) const TW: i64 = 4;
pub(crate) const TC: i64 = 2;
pub(crate) const TK: i64 = 3;
pub(crate) const TPAD: i64 = 1;
pub(crate) const OPH: i64 = 1;
pub(crate) const OPW: i64 = 1;
pub(crate) const HO: i64 = (TH - 1) * SH - 2 * TPAD + DH * (R - 1) + OPH + 1;
pub(crate) const WO: i64 = (TW - 1) * SW - 2 * TPAD + DW * (S - 1) + OPW + 1;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum Mutant {
    None,
    /// The group of output channel `k` is `k % G` (interleaved) rather than `k / (K/G)`.
    InterleavedGroups,
    /// Every output channel reads the channels of group 0.
    FirstGroupOnly,
    /// The transposed kernel's tap offset subtracts the padding where it should add it.
    PadSign,
}

fn ci(v: i64) -> ScalarExpr {
    ScalarExpr::ConstInt(v, I32)
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
        unroll: Unroll::None,
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
fn mac(acc: u32, a: u32, b: u32) -> Stmt {
    Stmt::Assign {
        local: acc,
        value: add(l(acc), mul(l(a), l(b))),
    }
}
fn params(names: [&str; 3]) -> Vec<Param> {
    let roles = [BufferRole::Read, BufferRole::Read, BufferRole::ReadWrite];
    names
        .into_iter()
        .zip(roles)
        .map(|(n, role)| Param::buffer(n, I32, role, Some(4)))
        .collect()
}
fn locals(n: usize) -> Vec<LocalDecl> {
    (0..n).map(|_| LocalDecl::scalar(I32)).collect()
}

/// Forward grouped convolution. Buffers: 0 `input` `[N, H, W, C]`, 1 `filter` `[K, R, S, C/G]`, 2 `out` `[N, P, Q, K]`. Block
/// `n`; thread `(x, y, z)` is output `(q, p, k)`. Locals: 0 `r`, 1 `s`, 2 `ih`, 3 `iw`, 4 `cg`, 5 input, 6 filter, 7 acc,
/// 8 input channel.
pub(crate) fn grouped_fprop(geo: Geo, mutant: Mutant) -> KernelFn {
    let n = exec_i32(ExecId::BlockIdx(0));
    let (q, p, k) = (
        exec_i32(ExecId::ThreadIdx(0)),
        exec_i32(ExecId::ThreadIdx(1)),
        exec_i32(ExecId::ThreadIdx(2)),
    );
    let (cg, kg) = (geo.c / geo.g, geo.k / geo.g);
    let group = match mutant {
        Mutant::InterleavedGroups => sub(k.clone(), mul(div(k.clone(), ci(geo.g)), ci(geo.g))),
        Mutant::FirstGroupOnly => ci(0),
        _ => div(k.clone(), ci(kg)),
    };
    let tap = vec![counted(
        4,
        cg,
        vec![
            let_(8, add(mul(group, ci(cg)), l(4))),
            load(5, at4(0, [n.clone(), l(2), l(3), l(8)])),
            load(6, at4(1, [k.clone(), l(0), l(1), l(4)])),
            mac(7, 5, 6),
        ],
    )];
    let s_body = vec![
        let_(
            3,
            sub(add(mul(q.clone(), ci(SW)), mul(l(1), ci(DW))), ci(PL)),
        ),
        within(l(3), W, tap),
    ];
    let r_body = vec![
        let_(
            2,
            sub(add(mul(p.clone(), ci(SH)), mul(l(0), ci(DH))), ci(PT)),
        ),
        within(l(2), H, vec![counted(1, S, s_body)]),
    ];
    let body = vec![
        let_(7, ci(0)),
        counted(0, R, r_body),
        Stmt::Store {
            place: at4(2, [n, p, q, k]),
            value: l(7),
        },
    ];
    KernelFn::new(
        "grouped_fprop",
        I32,
        params(["input", "filter", "out"]),
        locals(9),
        Block(body),
    )
    .with_launch(LaunchShape::new(
        [Q as u32, P as u32, geo.k as u32],
        [N as u32, 1, 1],
    ))
}

/// Transposed convolution as a gather. Buffers: 0 `input` `[N, TH, TW, TC]`, 1 `filter` `[R, S, TC, TK]`, 2 `out`
/// `[N, HO, WO, TK]`. Block `n`; thread `(x, y, z)` is output `(wo, ho, k)`. Output `(ho, wo)` takes input `(ih, iw)` through
/// tap `(r, s)` when `ih * SH + r * DH - TPAD == ho` (and likewise for the columns). Locals: 0 `r`, 1 `s`, 2 `c`, 3 `nh`,
/// 4 `nw`, 5 `ih`, 6 `iw`, 7 input, 8 filter, 9 acc.
pub(crate) fn transposed_fprop(mutant: Mutant) -> KernelFn {
    let n = exec_i32(ExecId::BlockIdx(0));
    let (wo, ho, k) = (
        exec_i32(ExecId::ThreadIdx(0)),
        exec_i32(ExecId::ThreadIdx(1)),
        exec_i32(ExecId::ThreadIdx(2)),
    );
    let pad = if mutant == Mutant::PadSign {
        -TPAD
    } else {
        TPAD
    };
    // the input index of output position `o` through tap `t`: `(o + pad - t * dil) / stride`, when exact and inside the extent
    let source = |o: ScalarExpr,
                  t: u32,
                  stride: i64,
                  dil: i64,
                  num: u32,
                  i: u32,
                  extent: i64,
                  then: Vec<Stmt>| {
        vec![
            let_(num, sub(add(o, ci(pad)), mul(l(t), ci(dil)))),
            when(
                cmp(BinaryOp::CmpLt, ci(-1), l(num)),
                vec![
                    let_(i, div(l(num), ci(stride))),
                    when(
                        cmp(BinaryOp::CmpEq, mul(l(i), ci(stride)), l(num)),
                        vec![when(cmp(BinaryOp::CmpLt, l(i), ci(extent)), then)],
                    ),
                ],
            ),
        ]
    };
    let channels = vec![counted(
        2,
        TC,
        vec![
            load(7, at4(0, [n.clone(), l(5), l(6), l(2)])),
            load(8, at4(1, [l(0), l(1), l(2), k.clone()])),
            mac(9, 7, 8),
        ],
    )];
    let cols = source(wo.clone(), 1, SW, DW, 4, 6, TW, channels);
    let rows = source(ho.clone(), 0, SH, DH, 3, 5, TH, vec![counted(1, S, cols)]);
    let body = vec![
        let_(9, ci(0)),
        counted(0, R, rows),
        Stmt::Store {
            place: at4(2, [n, ho, wo, k]),
            value: l(9),
        },
    ];
    KernelFn::new(
        "transposed_fprop",
        I32,
        params(["input", "filter", "out"]),
        locals(10),
        Block(body),
    )
    .with_launch(LaunchShape::new(
        [WO as u32, HO as u32, TK as u32],
        [N as u32, 1, 1],
    ))
}
