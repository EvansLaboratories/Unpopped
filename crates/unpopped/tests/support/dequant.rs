// SPDX-License-Identifier: MIT OR Apache-2.0
//! The dequantizing GEMM of ROADMAP U9b S3 (`docs/options-u9b-subbyte-dequant.md`): int4 weights times wider activations,
//! `out[r][c] = sum over k of (w[r][k] - zero[r][k / GROUP]) * scale[r][k / GROUP] * x[k][c]`, with `w` an `I4` or `U4` buffer
//! (two elements to a byte), the zero points and group scales in `I32` buffers and the activations `I32`, so every product is exact.
//! Shared by the oracle test (`kernel_dequant_gemm.rs`) and the C test
//! (`unpopped-cpu-c/tests/kernel_dequant_gemm_emit_agrees.rs`, which keeps a copy: change both).

use unpopped::exec::{ExecId, LaunchShape};
use unpopped::ir::{CastSpec, RoundMode, Saturation, ScalarExpr};
use unpopped::kernel::{Block, BufferRole, KernelFn, LocalDecl, Param, Place, Stmt};
use unpopped_vocab::ElementKind;

pub(crate) const I32: ElementKind = ElementKind::I32;
pub(crate) const U32: ElementKind = ElementKind::U32;
/// Output rows, reduction length, output columns, and the number of `k` values that share a scale and a zero point.
pub(crate) const R: usize = 4;
pub(crate) const K: usize = 8;
pub(crate) const C: usize = 4;
pub(crate) const GROUP: usize = 4;
pub(crate) const G: usize = K / GROUP;

fn cu(v: usize) -> ScalarExpr {
    ScalarExpr::ConstInt(v as i64, U32)
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

fn widen(x: ScalarExpr) -> ScalarExpr {
    ScalarExpr::Cast(
        Box::new(x),
        CastSpec {
            to: I32,
            round: RoundMode::NearestEven,
            sat: Saturation::Wrap,
        },
    )
}

/// Locals: 0 `acc`, 1 `g`, 2 `kk`, 3 `w` (sub-byte), 4 `zero`, 5 `scale`, 6 `x`.
/// The loop body for group `g`, in-group index `kk`: `acc += (widen(w) - zero) * scale * x`.
fn step(r: &ScalarExpr, c: &ScalarExpr) -> Vec<Stmt> {
    let k = add(mul(local(1), cu(GROUP)), local(2));
    let gi = add(mul(r.clone(), cu(G)), local(1));
    vec![
        Stmt::Load {
            local: 3,
            place: flat(0, add(mul(r.clone(), cu(K)), k.clone())),
        },
        Stmt::Load {
            local: 4,
            place: flat(1, gi.clone()),
        },
        Stmt::Load {
            local: 5,
            place: flat(2, gi),
        },
        Stmt::Load {
            local: 6,
            place: flat(3, add(mul(k, cu(C)), c.clone())),
        },
        Stmt::Assign {
            local: 0,
            value: add(
                local(0),
                mul(
                    mul(
                        ScalarExpr::Sub(Box::new(widen(local(3))), Box::new(local(4))),
                        local(5),
                    ),
                    local(6),
                ),
            ),
        },
    ]
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

/// Buffers: 0 `w` `[R*K]` `sub` (read), 1 `zero` `[R*G]` `I32`, 2 `scale` `[R*G]` `I32`, 3 `x` `[K*C]` `I32`, 4 `out` `[R*C]` `I32`.
/// One thread per output element: `ThreadIdx(0)` is the column, `ThreadIdx(1)` the row.
pub(crate) fn dequant_gemm(sub: ElementKind) -> KernelFn {
    let r = ScalarExpr::Exec(ExecId::ThreadIdx(1));
    let c = ScalarExpr::Exec(ExecId::ThreadIdx(0));
    let body = vec![
        Stmt::Let {
            local: 0,
            init: ScalarExpr::ConstInt(0, I32),
        },
        counted(1, G, vec![counted(2, GROUP, step(&r, &c))]),
        Stmt::Store {
            place: flat(4, add(mul(r, cu(C)), c)),
            value: local(0),
        },
    ];
    KernelFn::new(
        "dequant_gemm",
        I32,
        vec![
            Param::buffer("w", sub, BufferRole::Read, Some(1)),
            Param::buffer("zero", I32, BufferRole::Read, Some(1)),
            Param::buffer("scale", I32, BufferRole::Read, Some(1)),
            Param::buffer("x", I32, BufferRole::Read, Some(1)),
            Param::buffer("out", I32, BufferRole::ReadWrite, Some(1)),
        ],
        vec![
            LocalDecl::scalar(I32),
            LocalDecl::scalar(U32),
            LocalDecl::scalar(U32),
            LocalDecl::scalar(sub),
            LocalDecl::scalar(I32),
            LocalDecl::scalar(I32),
            LocalDecl::scalar(I32),
        ],
        Block(body),
    )
    .with_launch(LaunchShape::new([C as u32, R as u32, 1], [1, 1, 1]))
}

/// The nibble of weight `(r, k)`: every value 0..=15 appears and neighbours differ.
pub(crate) fn nibble(r: usize, k: usize) -> u8 {
    ((r * 5 + k * 3 + 1) % 16) as u8
}

/// The packed weights (two to a byte, low nibble first), the zero points, the scales and the activations.
pub(crate) fn dequant_data() -> (Vec<u8>, Vec<i32>, Vec<i32>, Vec<i32>) {
    let nibbles: Vec<u8> = (0..R)
        .flat_map(|r| (0..K).map(move |k| nibble(r, k)))
        .collect();
    let packed = nibbles.chunks(2).map(|p| p[0] | p[1] << 4).collect();
    let zero = (0..R * G).map(|i| (i % 3) as i32 - 1).collect();
    let scale = (0..R * G).map(|i| 1 + (i * 2 % 3) as i32).collect();
    let x = (0..K * C).map(|i| (i * 7 % 9) as i32 - 4).collect();
    (packed, zero, scale, x)
}

/// The host-computed result for a weight of the given signedness.
pub(crate) fn dequant_reference(signed: bool) -> Vec<i32> {
    let (_, zero, scale, x) = dequant_data();
    let w = |r: usize, k: usize| {
        let n = i32::from(nibble(r, k));
        if signed && n >= 8 { n - 16 } else { n }
    };
    let mut out = vec![0; R * C];
    for r in 0..R {
        for c in 0..C {
            out[r * C + c] = (0..K)
                .map(|k| {
                    let g = r * G + k / GROUP;
                    (w(r, k) - zero[g]) * scale[g] * x[k * C + c]
                })
                .sum();
        }
    }
    out
}
