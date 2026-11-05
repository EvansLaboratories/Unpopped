// COPY of crates/unpopped/tests/support/conv.rs: a test may not #[path] into another crate's directory (sweep SUP-004).
// Change both files together.
// SPDX-License-Identifier: MIT OR Apache-2.0
//! The implicit-GEMM convolution conformance kernel of ROADMAP U9 K7 (`docs/options-u9-gemm-family.md`): NHWC forward
//! convolution computed as a GEMM over `(r, s, c)` with the im2col index math inside the loop, no staged matrix. The
//! shapes are chosen so that an index mix-up cannot hide: a non-square image (`H != W`), a non-square filter
//! (`R != S`), padding on height only, and several channels. Shared by the oracle test (`kernel_conv_implicit_gemm.rs`)
//! and the C test (`unpopped-cpu-c/tests/kernel_conv_implicit_gemm_emit_agrees.rs`, which keeps a copy: change both).

use unpopped::exec::{ExecId, LaunchShape};
use unpopped::ir::{BinaryOp, CastSpec, RoundMode, Saturation, ScalarExpr};
use unpopped::kernel::{Block, BufferRole, KernelFn, LocalDecl, Param, Place, Stmt};
use unpopped_vocab::ElementKind;

pub(crate) const I32: ElementKind = ElementKind::I32;
pub(crate) const BATCH: i64 = 2;
pub(crate) const H: i64 = 4;
pub(crate) const W: i64 = 5;
pub(crate) const C: i64 = 3;
pub(crate) const KOUT: i64 = 2;
pub(crate) const R: i64 = 3;
pub(crate) const S: i64 = 2;
pub(crate) const PAD_H: i64 = 1;
pub(crate) const PAD_W: i64 = 0;
pub(crate) const P: i64 = H + 2 * PAD_H - R + 1;
pub(crate) const Q: i64 = W + 2 * PAD_W - S + 1;

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

fn lt(a: ScalarExpr, b: ScalarExpr) -> ScalarExpr {
    ScalarExpr::Binary(BinaryOp::CmpLt, Box::new(a), Box::new(b))
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

/// `if 0 <= x < limit { then }`.
fn within(x: ScalarExpr, limit: i64, then: Vec<Stmt>) -> Stmt {
    Stmt::If {
        cond: lt(ci(-1), x.clone()),
        then: Block(vec![Stmt::If {
            cond: lt(x, ci(limit)),
            then: Block(then),
            els: Block(vec![]),
        }]),
        els: Block(vec![]),
    }
}

/// `acc += in[n][ih][iw][c] * f[r][s][c][k]` (locals: 1 `r`, 2 `s`, 3 `ih`, 4 `iw`, 5 input, 6 filter, 7 `c`, 8 acc).
fn tap(n: &ScalarExpr, k: &ScalarExpr) -> Vec<Stmt> {
    vec![counted(
        7,
        C,
        vec![
            Stmt::Load {
                local: 5,
                place: at4(0, [n.clone(), local(3), local(4), local(7)]),
            },
            Stmt::Load {
                local: 6,
                place: at4(1, [local(1), local(2), local(7), k.clone()]),
            },
            Stmt::Assign {
                local: 8,
                value: add(local(8), mul(local(5), local(6))),
            },
        ],
    )]
}

/// The `(r, s)` loops: the input coordinate is `out + filter_index - pad`, and a tap outside the image is skipped (zero
/// padding).
fn taps(n: &ScalarExpr, p: &ScalarExpr, q: &ScalarExpr, k: &ScalarExpr) -> Vec<Stmt> {
    let s_body = vec![
        Stmt::Let {
            local: 4,
            init: sub(add(q.clone(), local(2)), ci(PAD_W)),
        },
        within(local(4), W, tap(n, k)),
    ];
    let r_body = vec![
        Stmt::Let {
            local: 3,
            init: sub(add(p.clone(), local(1)), ci(PAD_H)),
        },
        within(local(3), H, vec![counted(2, S, s_body)]),
    ];
    vec![counted(1, R, r_body)]
}

/// Buffers: 0 `input` `[N, H, W, C]`, 1 `filter` `[R, S, C, K]` (read), 2 `out` `[N, P, Q, K]`. Block `n`; thread
/// `(x, y, z)` is output `(q, p, k)`.
pub(crate) fn conv_implicit_gemm() -> KernelFn {
    let n = exec_i32(ExecId::BlockIdx(0));
    let (q, p, k) = (
        exec_i32(ExecId::ThreadIdx(0)),
        exec_i32(ExecId::ThreadIdx(1)),
        exec_i32(ExecId::ThreadIdx(2)),
    );
    let mut body = vec![Stmt::Let {
        local: 8,
        init: ci(0),
    }];
    body.extend(taps(&n, &p, &q, &k));
    body.push(Stmt::Store {
        place: at4(2, [n, p, q, k]),
        value: local(8),
    });
    let locals = (0..9).map(|_| LocalDecl::scalar(I32)).collect();
    let b = |name: &str, role: BufferRole| Param::buffer(name, I32, role, Some(4));
    KernelFn::new(
        "conv_igemm",
        I32,
        vec![
            b("input", BufferRole::Read),
            b("filter", BufferRole::Read),
            b("out", BufferRole::ReadWrite),
        ],
        locals,
        Block(body),
    )
    .with_launch(LaunchShape::new(
        [Q as u32, P as u32, KOUT as u32],
        [BATCH as u32, 1, 1],
    ))
}

/// Deterministic mixed-sign data: `(input, filter)`.
pub(crate) fn conv_data() -> (Vec<i32>, Vec<i32>) {
    let input = (0..BATCH * H * W * C)
        .map(|i| ((i * 7 + 3) % 11) as i32 - 5)
        .collect();
    let filter = (0..R * S * C * KOUT)
        .map(|i| ((i * 5 + 1) % 7) as i32 - 3)
        .collect();
    (input, filter)
}

/// The host reference: direct convolution, zero padding, stride 1, `[N, P, Q, K]` row-major.
pub(crate) fn conv_ref(input: &[i32], filter: &[i32]) -> Vec<i32> {
    let mut out = Vec::new();
    for n in 0..BATCH {
        for p in 0..P {
            for q in 0..Q {
                for k in 0..KOUT {
                    let mut acc = 0;
                    for r in 0..R {
                        for s in 0..S {
                            let (ih, iw) = (p + r - PAD_H, q + s - PAD_W);
                            if !(0..H).contains(&ih) || !(0..W).contains(&iw) {
                                continue;
                            }
                            for c in 0..C {
                                let i = ((n * H + ih) * W + iw) * C + c;
                                let f = ((r * S + s) * C + c) * KOUT + k;
                                acc += input[i as usize] * filter[f as usize];
                            }
                        }
                    }
                    out.push(acc);
                }
            }
        }
    }
    out
}
