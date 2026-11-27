// SPDX-License-Identifier: MIT OR Apache-2.0
//! Complex GEMM at the kernel layer (CUTLASS survey rows 26 and 82). Complex `f32` values are interleaved `(re, im)` pairs in
//! `F32` buffers; there is no complex element kind, so a complex product is real arithmetic over the pair. Two forms: the
//! four-multiply product, and the three-multiply Gaussian form (`k1 = br (ar + ai)`, `k2 = ar (bi - br)`, `k3 = ai (br + bi)`,
//! `re = k1 - k3`, `im = k1 + k2`). One thread per output element; `ThreadIdx(0)` is the column, `ThreadIdx(1)` the row.

use unpopped::exec::{ExecId, LaunchShape};
use unpopped::ir::ScalarExpr;
use unpopped::kernel::{Block, BufferRole, KernelFn, LocalDecl, Param, Place, Stmt, Unroll};
use unpopped_vocab::ElementKind;

pub(crate) const F32: ElementKind = ElementKind::F32;
pub(crate) const U32: ElementKind = ElementKind::U32;
pub(crate) const M: usize = 3;
pub(crate) const N: usize = 4;
pub(crate) const K: usize = 5;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum Form {
    /// Four real multiplies.
    Four,
    /// Three real multiplies (Gaussian).
    Three,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum Mutant {
    None,
    /// `B` is conjugated: its imaginary part is negated.
    ConjugateB,
    /// The imaginary part is `ar*bi - ai*br` (a sign slip).
    ImagSign,
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
fn at(buffer: u32, i: ScalarExpr) -> Place {
    Place::Logical {
        buffer,
        coords: vec![i],
    }
}

/// The element `(row, col)` part `p` of a `[rows][cols]` interleaved complex buffer.
fn part(row: &ScalarExpr, col: &ScalarExpr, cols: usize, p: usize) -> ScalarExpr {
    add(
        mul(add(mul(row.clone(), cu(cols)), col.clone()), cu(2)),
        cu(p),
    )
}

/// The per-step increments `(re, im)` from locals 3 `ar`, 4 `ai`, 5 `br`, 6 `bi`.
fn product(form: Form, mutant: Mutant) -> (ScalarExpr, ScalarExpr) {
    let bi = if mutant == Mutant::ConjugateB {
        sub(ScalarExpr::Const(0.0), l(6))
    } else {
        l(6)
    };
    let (ar, ai, br) = (l(3), l(4), l(5));
    match form {
        Form::Four => {
            let im = if mutant == Mutant::ImagSign {
                sub(mul(ar.clone(), bi.clone()), mul(ai.clone(), br.clone()))
            } else {
                add(mul(ar.clone(), bi.clone()), mul(ai.clone(), br.clone()))
            };
            (sub(mul(ar, br), mul(ai, bi)), im)
        }
        Form::Three => {
            let k1 = mul(br.clone(), add(ar.clone(), ai.clone()));
            let k2 = mul(ar, sub(bi.clone(), br.clone()));
            let k3 = mul(ai, add(br, bi));
            let im = if mutant == Mutant::ImagSign {
                sub(k1.clone(), k2)
            } else {
                add(k1.clone(), k2)
            };
            (sub(k1, k3), im)
        }
    }
}

/// The body of one `kk` step: the four loads and the two accumulations.
fn k_step(form: Form, mutant: Mutant, r: &ScalarExpr, c: &ScalarExpr) -> Vec<Stmt> {
    let kk = l(2);
    let load = |local, buffer, i| Stmt::Load {
        local,
        place: at(buffer, i),
    };
    let (dre, dim) = product(form, mutant);
    vec![
        load(3, 0, part(r, &kk, K, 0)),
        load(4, 0, part(r, &kk, K, 1)),
        load(5, 1, part(&kk, c, N, 0)),
        load(6, 1, part(&kk, c, N, 1)),
        Stmt::Assign {
            local: 0,
            value: add(l(0), dre),
        },
        Stmt::Assign {
            local: 1,
            value: add(l(1), dim),
        },
    ]
}

/// The buffer parameters and the locals.
fn signature() -> (Vec<Param>, Vec<LocalDecl>) {
    let params = vec![
        Param::buffer("a", F32, BufferRole::Read, Some(1)),
        Param::buffer("b", F32, BufferRole::Read, Some(1)),
        Param::buffer("out", F32, BufferRole::ReadWrite, Some(1)),
    ];
    let locals = [F32, F32, U32, F32, F32, F32, F32]
        .map(LocalDecl::scalar)
        .to_vec();
    (params, locals)
}

/// Buffers: 0 `a` `[M*K*2]`, 1 `b` `[K*N*2]`, 2 `out` `[M*N*2]`. Locals: 0 `re`, 1 `im`, 2 `kk`, 3 `ar`, 4 `ai`, 5 `br`, 6 `bi`.
pub(crate) fn complex_gemm(form: Form, mutant: Mutant) -> KernelFn {
    let r = ScalarExpr::Exec(ExecId::ThreadIdx(1));
    let c = ScalarExpr::Exec(ExecId::ThreadIdx(0));
    let step = k_step(form, mutant, &r, &c);
    let zero = |local| Stmt::Let {
        local,
        init: ScalarExpr::Const(0.0),
    };
    let body = vec![
        zero(0),
        zero(1),
        Stmt::For {
            unroll: Unroll::None,
            local: 2,
            start: cu(0),
            end: cu(K),
            step: cu(1),
            body: Block(step),
        },
        Stmt::Store {
            place: at(2, part(&r, &c, N, 0)),
            value: l(0),
        },
        Stmt::Store {
            place: at(2, part(&r, &c, N, 1)),
            value: l(1),
        },
    ];
    let (params, locals) = signature();
    KernelFn::new("complex_gemm", F32, params, locals, Block(body))
        .with_launch(LaunchShape::new([N as u32, M as u32, 1], [1, 1, 1]))
}
