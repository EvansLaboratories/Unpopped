// SPDX-License-Identifier: MIT OR Apache-2.0
//! The mixed-input GEMM with floating-point scales at the kernel layer (CUTLASS survey rows 16 and 17; `support/dequant.rs` is the
//! I32-scale kernel of U9b S3 and is not edited). Integer weights (`I8`, or `I4` packed two to a byte) are widened to `F32`, a
//! per-group scale (and, in the scale-plus-zero-point mode, a zero point) is applied, and the product is accumulated against
//! `F16`, `Bf16` or `F32` activations. Scales and zero points are `F16`, `Bf16` or `F32`. Group size 4 along `K`. `F32`
//! accumulation.
//!
//! The modes follow the CUTLASS mixed-input collective's scale modes (`cutlass/gemm/collective/builders/` `KernelScheduleAuto`
//! `ConversionMode::{ConvertAndScale, ConvertAndScaleWithZero}`): UNVERIFIED against a checkout; only the arithmetic
//! `(w - zero) * scale` is used.

use unpopped::exec::{ExecId, LaunchShape};
use unpopped::ir::{CastSpec, RoundMode, Saturation, ScalarExpr};
use unpopped::kernel::{Block, BufferRole, KernelFn, LocalDecl, Param, Place, Stmt, Unroll};
use unpopped_vocab::ElementKind;

pub(crate) const F32: ElementKind = ElementKind::F32;
pub(crate) const U32: ElementKind = ElementKind::U32;
pub(crate) const R: usize = 4;
pub(crate) const K: usize = 8;
pub(crate) const C: usize = 4;
pub(crate) const GROUP: usize = 4;
pub(crate) const G: usize = K / GROUP;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum Mode {
    /// `w * scale`.
    Scale,
    /// `(w - zero) * scale`.
    ScaleZero,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum Mutant {
    None,
    /// The scale is looked up per row, not per group.
    ScalePerRow,
    /// `w * scale - zero`: the zero point applied after the scale.
    ZeroAfterScale,
    /// `(w + zero) * scale`.
    ZeroSign,
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
fn flat(buffer: u32, i: ScalarExpr) -> Place {
    Place::Logical {
        buffer,
        coords: vec![i],
    }
}
fn to_f32(x: ScalarExpr) -> ScalarExpr {
    ScalarExpr::Cast(
        Box::new(x),
        CastSpec {
            to: F32,
            round: RoundMode::NearestEven,
            sat: Saturation::Wrap,
        },
    )
}
/// A weight widened to `F32`; a sub-byte weight goes through `I32` first (the checker's rule for a sub-byte `Cast`).
fn weight_f32(w: ScalarExpr, wk: ElementKind) -> ScalarExpr {
    if wk == ElementKind::I4 {
        to_f32(ScalarExpr::Cast(
            Box::new(w),
            CastSpec {
                to: ElementKind::I32,
                round: RoundMode::NearestEven,
                sat: Saturation::Wrap,
            },
        ))
    } else {
        to_f32(w)
    }
}
fn counted(local: u32, end: usize, body: Vec<Stmt>) -> Stmt {
    Stmt::For {
        unroll: Unroll::None,
        local,
        start: cu(0),
        end: cu(end),
        step: cu(1),
        body: Block(body),
    }
}

/// The index of the scale (and zero point) of row `r` at group local 1: per group, or per row for the mutant.
fn scale_index(r: &ScalarExpr, mutant: Mutant) -> ScalarExpr {
    if mutant == Mutant::ScalePerRow {
        r.clone()
    } else {
        add(mul(r.clone(), cu(G)), l(1))
    }
}

/// The dequantized weight (locals 3, 4, 5 are the weight, the zero point and the scale) for the mode and mutant.
fn dequantized(wk: ElementKind, mode: Mode, mutant: Mutant) -> ScalarExpr {
    let (w, z, s) = (weight_f32(l(3), wk), to_f32(l(4)), to_f32(l(5)));
    match (mode, mutant) {
        (Mode::Scale, _) => mul(w, s),
        (Mode::ScaleZero, Mutant::ZeroAfterScale) => sub(mul(w, s), z),
        (Mode::ScaleZero, Mutant::ZeroSign) => mul(add(w, z), s),
        (Mode::ScaleZero, _) => mul(sub(w, z), s),
    }
}

/// The buffer parameters: weights, scales, activations, output, and the zero points in the `ScaleZero` mode.
fn gemm_params(wk: ElementKind, sk: ElementKind, ak: ElementKind, mode: Mode) -> Vec<Param> {
    let mut params = vec![
        Param::buffer("w", wk, BufferRole::Read, Some(1)),
        Param::buffer("scale", sk, BufferRole::Read, Some(1)),
        Param::buffer("x", ak, BufferRole::Read, Some(1)),
        Param::buffer("out", F32, BufferRole::ReadWrite, Some(1)),
    ];
    if mode == Mode::ScaleZero {
        params.push(Param::buffer("zero", sk, BufferRole::Read, Some(1)));
    }
    params
}

/// The locals: `acc`, the two loop indices, the weight, the zero point, the scale and the activation.
fn gemm_locals(wk: ElementKind, sk: ElementKind, ak: ElementKind) -> Vec<LocalDecl> {
    [F32, U32, U32, wk, sk, sk, ak]
        .map(LocalDecl::scalar)
        .to_vec()
}

/// Buffers: 0 `w` `[R*K]` of `wk` (read), 1 `scale` `[R*G]` of `sk`, 2 `x` `[K*C]` of `ak`, 3 `out` `[R*C]` `F32`, and in the
/// `ScaleZero` mode 4 `zero` `[R*G]` of `sk`. One thread per output element: `ThreadIdx(0)` is the column, `ThreadIdx(1)` the row.
/// Locals: 0 `acc`, 1 `g`, 2 `kk`, 3 `w`, 4 `zero`, 5 `scale`, 6 `x`.
pub(crate) fn mixed_input_gemm(
    wk: ElementKind,
    sk: ElementKind,
    ak: ElementKind,
    mode: Mode,
    mutant: Mutant,
) -> KernelFn {
    let r = ScalarExpr::Exec(ExecId::ThreadIdx(1));
    let c = ScalarExpr::Exec(ExecId::ThreadIdx(0));
    let k = add(mul(l(1), cu(GROUP)), l(2));
    let group_index = scale_index(&r, mutant);
    let load = |local, buffer, i| Stmt::Load {
        local,
        place: flat(buffer, i),
    };
    let mut step = vec![
        load(3, 0, add(mul(r.clone(), cu(K)), k.clone())),
        load(5, 1, group_index.clone()),
        load(6, 2, add(mul(k, cu(C)), c.clone())),
    ];
    if mode == Mode::ScaleZero {
        step.push(load(4, 4, group_index));
    }
    step.push(Stmt::Assign {
        local: 0,
        value: add(l(0), mul(dequantized(wk, mode, mutant), to_f32(l(6)))),
    });
    let body = vec![
        Stmt::Let {
            local: 0,
            init: ScalarExpr::Const(0.0),
        },
        counted(1, G, vec![counted(2, GROUP, step)]),
        Stmt::Store {
            place: flat(3, add(mul(r, cu(C)), c)),
            value: l(0),
        },
    ];
    KernelFn::new(
        "mixed_input_gemm",
        F32,
        gemm_params(wk, sk, ak, mode),
        gemm_locals(wk, sk, ak),
        Block(body),
    )
    .with_launch(LaunchShape::new([C as u32, R as u32, 1], [1, 1, 1]))
}
