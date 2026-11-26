// SPDX-License-Identifier: MIT OR Apache-2.0
//! The modules of `kernel_ssd_scan.rs`: a matrix-state linear recurrence, the scan inside a Mamba2 SSD block
//! (state-space duality; Dao and Gu, "Transformers are SSMs", 2024, arXiv:2405.21060, section 6 chunked algorithm), written
//! as one `Stmt::CollectiveTuple { kind: ScanWith }` over a five-component state `(A, H00, H01, H10, H11)`.
//!
//! One lane is one time step `t` of a head with a scalar decay `a_t`, an input `x_t` in R^2, a state input `B_t` in R^2 and
//! a readout `C_t` in R^2. The recurrence is `H_t = a_t * H_{t-1} + B_t (outer) x_t` (a 2 x 2 state) and `y_t = C_t^T H_t`.
//! The lane contributes the element `(a_t, B_t[i] * x_t[j])`. The combine is the associative operator on affine maps,
//! `(A1, H1) o (A2, H2) = (A1 * A2, A2 * H1 + H2)`, folded left to right in lane order; `A` is the product of the decays
//! so far (the carried decay of a chunk), `H` the state. Each lane then reads out `y`. The 2 x 2 state is the smallest that
//! is a matrix: the vector-state case is the `A`-and-one-component slice of it.
//!
//! UNVERIFIED against the Mamba2 reference kernels (`state-spaces/mamba`, `ssd_combined.py`): only the paper's algebra is
//! used here, and the reference in the test file is a plain loop, not a port of that code.

use unpopped::exec::{ExecId, LaunchShape, Scope};
use unpopped::ir::ScalarExpr;
use unpopped::kernel::{
    Block, BufferRole, CollectiveKind, KernelFn, LocalDecl, Module, Param, Place, Stmt,
};
use unpopped_vocab::ElementKind;

pub(crate) const F32: ElementKind = ElementKind::F32;
pub(crate) const I32: ElementKind = ElementKind::I32;
/// Lanes (time steps).
pub(crate) const N: usize = 8;
/// Lanes per subgroup.
pub(crate) const SG: usize = 4;

/// What the module is built wrong, to show the test can fail.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum Mutant {
    /// The kernel as intended.
    None,
    /// The combine scales the earlier state by its own decay `A1` instead of the later decay `A2`.
    WrongScale,
    /// The combine drops the decay product: `A = A2`.
    DropDecayProduct,
    /// A reduce rather than a scan: every lane receives the fold of all lanes.
    Reduce,
}

fn l(i: u32) -> ScalarExpr {
    ScalarExpr::Local(i)
}

fn tid() -> ScalarExpr {
    ScalarExpr::Exec(ExecId::ThreadIdx(0))
}

fn cj(v: i64) -> ScalarExpr {
    ScalarExpr::ConstInt(v, I32)
}

fn add(a: ScalarExpr, b: ScalarExpr) -> ScalarExpr {
    ScalarExpr::Add(Box::new(a), Box::new(b))
}

fn mul(a: ScalarExpr, b: ScalarExpr) -> ScalarExpr {
    ScalarExpr::Mul(Box::new(a), Box::new(b))
}

fn at(buffer: u32) -> Place {
    Place::Logical {
        buffer,
        coords: vec![tid()],
    }
}

fn at2(buffer: u32, col: i64) -> Place {
    Place::Logical {
        buffer,
        coords: vec![tid(), cj(col)],
    }
}

fn load(local: u32, place: Place) -> Stmt {
    Stmt::Load { local, place }
}

fn store(place: Place, value: ScalarExpr) -> Stmt {
    Stmt::Store { place, value }
}

/// `ssd_combine(A1, H1 (4), A2, H2 (4)) -> (A, H (4))`. Locals: 0 `A1`, 1..=4 `H1`, 5 `A2`, 6..=9 `H2` (the parameters),
/// 10 `A` (the result), 11..=14 `H`.
pub(crate) fn ssd_combine(mutant: Mutant) -> KernelFn {
    let scale = if mutant == Mutant::WrongScale { 0 } else { 5 };
    let mut body = vec![Stmt::Let {
        local: 10,
        init: if mutant == Mutant::DropDecayProduct {
            l(5)
        } else {
            mul(l(0), l(5))
        },
    }];
    for k in 0..4u32 {
        body.push(Stmt::Let {
            local: 11 + k,
            init: add(mul(l(scale), l(1 + k)), l(6 + k)),
        });
    }
    let mut params = vec![Param::scalar("a1", F32, 0)];
    params.extend((0..4).map(|k| Param::scalar(format!("h1_{k}"), F32, 1 + k)));
    params.push(Param::scalar("a2", F32, 5));
    params.extend((0..4).map(|k| Param::scalar(format!("h2_{k}"), F32, 6 + k)));
    KernelFn::new(
        "ssd_combine",
        F32,
        params,
        vec![LocalDecl::scalar(F32); 15],
        Block(body),
    )
    .with_result(10)
    .with_results([11, 12, 13, 14])
}

/// The launched entry (function 0) and the combine (function 1). Buffers: 0 `a[N]`, 1 `b[N, 2]`, 2 `x[N, 2]`, 3 `c[N, 2]`
/// (read), 4 `out_a[N]`, 5 `out_h[N, 4]`, 6 `out_y[N, 2]` (write). Locals: 0 `a`, 1..=2 `b`, 3..=4 `x`, 5..=6 `c`, 7 the
/// carried decay, 8..=11 the state, 12..=13 `y`.
pub(crate) fn ssd_module(scope: Scope, mutant: Mutant) -> Module {
    let mut body = vec![load(0, at(0))];
    body.extend((0..2u32).map(|j| load(1 + j, at2(1, i64::from(j)))));
    body.extend((0..2u32).map(|j| load(3 + j, at2(2, i64::from(j)))));
    body.extend((0..2u32).map(|j| load(5 + j, at2(3, i64::from(j)))));
    // the lane's element: its decay and the outer product B (outer) x, row-major
    let values = vec![
        l(0),
        mul(l(1), l(3)),
        mul(l(1), l(4)),
        mul(l(2), l(3)),
        mul(l(2), l(4)),
    ];
    body.push(Stmt::CollectiveTuple {
        kind: if mutant == Mutant::Reduce {
            CollectiveKind::ReduceWith { combine: 1 }
        } else {
            CollectiveKind::ScanWith { combine: 1 }
        },
        values,
        results: vec![7, 8, 9, 10, 11],
        scope,
    });
    body.push(store(at(4), l(7)));
    body.extend((0..4u32).map(|k| store(at2(5, i64::from(k)), l(8 + k))));
    // y[j] = sum_i C[i] * H[i][j]
    for j in 0..2u32 {
        body.push(Stmt::Let {
            local: 12 + j,
            init: add(mul(l(5), l(8 + j)), mul(l(6), l(10 + j))),
        });
        body.push(store(at2(6, i64::from(j)), l(12 + j)));
    }
    let params = vec![
        Param::buffer("a", F32, BufferRole::Read, Some(1)),
        Param::buffer("b", F32, BufferRole::Read, Some(2)),
        Param::buffer("x", F32, BufferRole::Read, Some(2)),
        Param::buffer("c", F32, BufferRole::Read, Some(2)),
        Param::buffer("out_a", F32, BufferRole::Write, Some(1)),
        Param::buffer("out_h", F32, BufferRole::Write, Some(2)),
        Param::buffer("out_y", F32, BufferRole::Write, Some(2)),
    ];
    let main = KernelFn::new(
        "main",
        F32,
        params,
        vec![LocalDecl::scalar(F32); 14],
        Block(body),
    )
    .with_launch(LaunchShape::new([N as u32, 1, 1], [1, 1, 1]).with_subgroup(SG as u32, SG as u32));
    Module::new(vec![main, ssd_combine(mutant)])
}
