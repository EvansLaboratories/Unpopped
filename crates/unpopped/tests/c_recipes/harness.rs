// SPDX-License-Identifier: MIT OR Apache-2.0
//! Shared harness of `kernel_c_recipes` and `kernel_c_recipes_mma`: layout-as-index-arithmetic and a one-block-array kernel
//! builder. Each test file uses part of it, hence the `dead_code` allowance (this module is not a test target).
#![allow(dead_code)]

use unpopped::exec::{AllocDecl, ExecId, Extent, LaunchShape, MemorySpace, Scope};
use unpopped::ir::ScalarExpr;
use unpopped::kernel::{
    BarrierKind, Block, BufferRole, KernelFn, LocalDecl, Param, Place, Stmt, Unroll, check_kernel,
};
use unpopped::oracle::TypedBuffer;
use unpopped::oracle::kernel_eval::KernelArg;
use unpopped::oracle::kernel_launch::run_launch;
use unpopped_vocab::ElementKind::{self, U32};
use unpopped_vocab::layout_algebra::Layout;

// ---- index arithmetic ------------------------------------------------------------------------------------------------

pub(crate) fn u(v: i64) -> ScalarExpr {
    ScalarExpr::ConstInt(v, U32)
}
pub(crate) fn l(i: u32) -> ScalarExpr {
    ScalarExpr::Local(i)
}
pub(crate) fn tid() -> ScalarExpr {
    ScalarExpr::Exec(ExecId::ThreadIdx(0))
}
pub(crate) fn blk() -> ScalarExpr {
    ScalarExpr::Exec(ExecId::BlockIdx(0))
}
pub(crate) fn add(a: ScalarExpr, b: ScalarExpr) -> ScalarExpr {
    ScalarExpr::Add(Box::new(a), Box::new(b))
}
pub(crate) fn mul(a: ScalarExpr, b: ScalarExpr) -> ScalarExpr {
    ScalarExpr::Mul(Box::new(a), Box::new(b))
}
pub(crate) fn div(a: ScalarExpr, b: i64) -> ScalarExpr {
    ScalarExpr::Div(Box::new(a), Box::new(u(b)))
}
/// `a % b` for integers, as `a - (a / b) * b` (the integer `Div` is the only integer quotient the IR has).
pub(crate) fn rem(a: ScalarExpr, b: i64) -> ScalarExpr {
    ScalarExpr::Sub(Box::new(a.clone()), Box::new(mul(div(a, b), u(b))))
}

pub(crate) fn lay(t: &str) -> Layout {
    t.parse().unwrap_or_else(|e| panic!("{t}: {e}"))
}

/// The layout `l` evaluated at the (runtime) linear index `idx`, colexicographic over the flattened shape: the C-recipe for
/// "apply this layout" (`Layout::call` as arithmetic). The last mode takes the remaining quotient, as `Layout::call` does.
pub(crate) fn lay_expr(layout: &Layout, idx: ScalarExpr) -> ScalarExpr {
    let modes = layout.flat_modes();
    let (mut prod, mut acc) = (1i64, u(0));
    for (k, (s, d)) in modes.iter().enumerate() {
        let mut c = if prod == 1 {
            idx.clone()
        } else {
            div(idx.clone(), prod)
        };
        if k + 1 != modes.len() {
            c = rem(c, *s);
        }
        acc = add(acc, mul(c, u(*d)));
        prod *= s;
    }
    acc
}

/// Mode `k` of `zipped_divide(tilers)`: 0 is the tile (thread) mode, 1 the rest mode.
pub(crate) fn zmode(layout: &Layout, tilers: &[Layout], k: usize) -> Layout {
    layout.zipped_divide(tilers).expect("divisible").modes()[k].clone()
}

// ---- the harness ---------------------------------------------------------------------------------------------------

/// The value the run puts at logical element `i` of the shared array.
pub(crate) fn val(i: i64) -> u32 {
    (i * 7 + 3) as u32
}

pub(crate) fn at(buffer: u32, i: ScalarExpr) -> Place {
    Place::Logical {
        buffer,
        coords: vec![i],
    }
}

pub(crate) fn sh(alloc: u32, index: ScalarExpr) -> Place {
    Place::Alloc { alloc, index }
}

/// `total` elements of `in` copied to allocation 0 by the block's threads (locals 0 and 1 are the loop counter and the value),
/// then a block barrier.
pub(crate) fn fill(total: i64, threads: u32) -> Vec<Stmt> {
    let copy = Block(vec![
        Stmt::Load {
            local: 1,
            place: at(0, l(0)),
        },
        Stmt::Store {
            place: sh(0, l(0)),
            value: l(1),
        },
    ]);
    vec![
        Stmt::For {
            local: 0,
            start: tid(),
            end: u(total),
            step: u(i64::from(threads)),
            unroll: Unroll::None,
            body: copy,
        },
        Stmt::Barrier {
            scope: Scope::Block,
            kind: BarrierKind::Full,
        },
    ]
}

pub(crate) struct Shape {
    pub(crate) threads: u32,
    pub(crate) blocks: u32,
    pub(crate) total: i64,
    pub(crate) locals: Vec<ElementKind>,
    pub(crate) allocs: Vec<AllocDecl>,
}

pub(crate) fn shared(n: u32) -> AllocDecl {
    AllocDecl::new("sh", MemorySpace::Shared, U32, Extent::Static(n))
}

/// A kernel `(in, out, ..extra)`: fill allocation 0 from `in`, then `mid`.
pub(crate) fn kernel(s: Shape, extra: Vec<Param>, mid: Vec<Stmt>) -> KernelFn {
    let mut params = vec![
        Param::buffer("in", U32, BufferRole::Read, Some(1)),
        Param::buffer("out", U32, BufferRole::Write, Some(1)),
    ];
    params.extend(extra);
    let mut body = fill(s.total, s.threads);
    body.extend(mid);
    let locals = s.locals.iter().map(|t| LocalDecl::scalar(*t)).collect();
    KernelFn::new("c_recipe", ElementKind::I32, params, locals, Block(body))
        .with_launch(LaunchShape::new([s.threads, 1, 1], [s.blocks, 1, 1]))
        .with_allocs(s.allocs)
}

pub(crate) fn buf(data: Vec<u32>) -> KernelArg {
    let n = data.len() as i64;
    let bytes = data.iter().flat_map(|v| v.to_le_bytes()).collect();
    KernelArg::Buffer(TypedBuffer::new(U32, vec![n], vec![1], bytes))
}

/// Run on `in[i] = val(i)` (`in_len` of them), `extra` arguments after `out` (`out_len` zeros).
pub(crate) fn run(k: &KernelFn, in_len: i64, out_len: usize, extra: Vec<KernelArg>) -> Vec<u32> {
    assert_eq!(check_kernel(k), Ok(()), "well formed");
    let mut args = vec![buf((0..in_len).map(val).collect()), buf(vec![0; out_len])];
    args.extend(extra);
    run_launch(k, &mut args).unwrap_or_else(|e| panic!("{e:?}"));
    let KernelArg::Buffer(out) = &args[1] else {
        unreachable!()
    };
    out.bytes
        .chunks(4)
        .map(|c| u32::from_le_bytes(c.try_into().unwrap()))
        .collect()
}

/// `m` loads of allocation 0 at `idx(j)` into locals `2..2+m`, stored at `out[(block * threads + tid) * m + j]`.
pub(crate) fn gather(threads: u32, m: u32, idx: &dyn Fn(u32) -> ScalarExpr) -> Vec<Stmt> {
    let mut v = Vec::new();
    for j in 0..m {
        v.push(Stmt::Load {
            local: 2 + j,
            place: sh(0, idx(j)),
        });
    }
    let base = mul(
        add(mul(blk(), u(i64::from(threads))), tid()),
        u(i64::from(m)),
    );
    for j in 0..m {
        v.push(Stmt::Store {
            place: at(1, add(base.clone(), u(i64::from(j)))),
            value: l(2 + j),
        });
    }
    v
}

/// The common one-array recipe: `threads` x `blocks`, `m` values per thread at `idx(j)`.
pub(crate) fn recipe(
    threads: u32,
    blocks: u32,
    m: u32,
    idx: &dyn Fn(u32) -> ScalarExpr,
) -> KernelFn {
    let s = Shape {
        threads,
        blocks,
        total: 64,
        locals: vec![U32; 2 + m as usize],
        allocs: vec![shared(64).with_layout("(4,16):(16,1)".parse().unwrap())],
    };
    kernel(s, vec![], gather(threads, m, idx))
}

pub(crate) fn run64(k: &KernelFn, out_len: usize) -> Vec<u32> {
    run(k, 64, out_len, vec![])
}
