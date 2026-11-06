// COPY of `crates/unpopped/tests/support/attention_mma.rs`: a test may not `#[path]` into another crate's directory (sweep SUP-004).
// Change both files together.
// SPDX-License-Identifier: MIT OR Apache-2.0
//! The tensor-core attention kernel of ROADMAP U10 A6 (`docs/options-u10-attention.md`): both GEMMs of attention as `Stmt::Mma`
//! on a toy `m4n4k8` atom, with the row softmax between them in scalar code. One subgroup of `LANES` lanes handles `M = 4`
//! queries of head dimension `K = 8` against `K = 8` keys: `S = scale * Q K^T` is two MMAs (keys 0..4 and 4..8, `N = 4` each),
//! the lanes exchange `S` through global scratch (a block barrier), lane `t` turns row `t` into `P = softmax(S)`, and `O = P V`
//! is one more MMA over the 8 keys. Shared by the oracle test (`kernel_attention_mma.rs`) and the C test
//! (`unpopped-cpu-c/tests/kernel_attention_mma_emit_agrees.rs`, which keeps a copy: change both).

use unpopped::exec::{AllocDecl, ExecId, Extent, LaunchShape, MemorySpace, Scope};
use unpopped::ir::{BinaryOp, ScalarExpr, UnaryOp};
use unpopped::kernel::{BarrierKind, Block, BufferRole, KernelFn, LocalDecl, Param, Place, Stmt};
use unpopped::mma::{AtomSpec, MmaAtom};
use unpopped_vocab::ElementKind;
use unpopped_vocab::layout_algebra::Layout;

pub(crate) const F32: ElementKind = ElementKind::F32;
pub(crate) const U32: ElementKind = ElementKind::U32;
pub(crate) const LANES: usize = 4;
/// Queries, keys per S tile (and the value width), head dimension (and the number of keys).
pub(crate) const M: usize = 4;
pub(crate) const N: usize = 4;
pub(crate) const K: usize = 8;
/// The score scale; with these inputs scores pass 89, where an `exp` without the max subtraction overflows an `f32`.
pub(crate) const SCALE: f64 = 5.0;

fn leaf(size: i64) -> Layout {
    Layout::leaf(size, 1).unwrap()
}

/// The `F32` toy atom (the shipped half-precision atoms have no C element type in cpu-c).
pub(crate) fn atom() -> MmaAtom {
    MmaAtom::new(
        AtomSpec {
            name: "toy.m4n4k8",
            mnk: (M as u32, N as u32, K as u32),
            dtypes: (F32, F32, F32),
            scope: Scope::Subgroup,
            lanes: LANES as u32,
        },
        [&leaf(32), &leaf(32), &leaf(16)],
    )
    .unwrap()
}

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

fn div(a: ScalarExpr, b: ScalarExpr) -> ScalarExpr {
    ScalarExpr::Div(Box::new(a), Box::new(b))
}

fn exp(x: ScalarExpr) -> ScalarExpr {
    ScalarExpr::Unary(UnaryOp::Exp, Box::new(x))
}

fn flat(buffer: u32, i: ScalarExpr) -> Place {
    Place::Logical {
        buffer,
        coords: vec![i],
    }
}

fn frag(alloc: u32, i: ScalarExpr) -> Place {
    Place::Alloc { alloc, index: i }
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

fn load(local: u32, place: Place) -> Stmt {
    Stmt::Load { local, place }
}

// Locals: 0 lane, 1 v, 2 id, 3 x, 4 j, 5 max, 6 sum, 7 score, 8 weight.
// Buffers: 0 q (A of S), 1 k0, 2 k1 (B of S, two key tiles), 3 v (B of O), 4 ia, 5 ib, 6 ic (id tables),
// 7 s0, 8 s1 (score scratch), 9 p (A of O), 10 out.
// Fragments (Private): 0 fq, 1 fk0, 2 fk1, 3 fp, 4 fv, 5 c0, 6 d0, 7 c1, 8 d1, 9 c2, 10 d2.

fn lane() -> ScalarExpr {
    ScalarExpr::Exec(ExecId::ThreadIdx(0))
}

/// Fragment `alloc` of operand `table` (0 = A, 1 = B, 2 = C) from logical buffer `buf`: `frag[v] = buf[table[lane * len + v]]`.
fn gather(alloc: u32, buf: u32, table: u32, len: usize) -> Stmt {
    counted(
        1,
        len,
        vec![
            load(2, flat(table, add(mul(lane(), cu(len)), local(1)))),
            load(3, flat(buf, local(2))),
            Stmt::Store {
                place: frag(alloc, local(1)),
                value: local(3),
            },
        ],
    )
}

fn zero_fragment(alloc: u32, len: usize) -> Stmt {
    counted(
        1,
        len,
        vec![Stmt::Store {
            place: frag(alloc, local(1)),
            value: ScalarExpr::Const(0.0),
        }],
    )
}

/// `scratch[table[lane * len + v]] = scale * frag[v]` (the S tile out to global scratch).
fn scatter_scaled(alloc: u32, scratch: u32, len: usize) -> Stmt {
    counted(
        1,
        len,
        vec![
            load(2, flat(6, add(mul(lane(), cu(len)), local(1)))),
            load(3, frag(alloc, local(1))),
            Stmt::Store {
                place: flat(scratch, local(2)),
                value: mul(local(3), ScalarExpr::Const(SCALE)),
            },
        ],
    )
}

fn mma(a: u32, b: u32, c: u32, d: u32) -> Stmt {
    Stmt::Mma {
        atom: atom(),
        a,
        b,
        c,
        d,
        extra: vec![],
    }
}

fn barrier() -> Stmt {
    Stmt::Barrier {
        scope: Scope::Block,
        kind: BarrierKind::Full,
    }
}

/// The score of this lane's row at key `j` of half `half` (0 or 1): scratch `s{half}` is `[row + M * key_in_half]`.
fn score(half: u32, j: u32) -> Stmt {
    load(7, flat(7 + half, add(lane(), mul(local(j), cu(M)))))
}

/// Row softmax of this lane's row: max over the 8 scores, sum of `exp(s - max)`, then `p[row + M * key] = exp(s - max) / sum`.
fn softmax_row() -> Vec<Stmt> {
    let halves = |body: &dyn Fn(u32) -> Vec<Stmt>| -> Vec<Stmt> {
        (0..2u32)
            .map(|h| {
                let mut b = vec![score(h, 4)];
                b.extend(body(h));
                counted(4, N, b)
            })
            .collect()
    };
    let mut out = vec![load(5, flat(7, lane()))];
    out.extend(halves(&|_| {
        vec![Stmt::Assign {
            local: 5,
            value: ScalarExpr::Binary(BinaryOp::Max, Box::new(local(5)), Box::new(local(7))),
        }]
    }));
    out.push(Stmt::Let {
        local: 6,
        init: ScalarExpr::Const(0.0),
    });
    out.extend(halves(&|_| {
        vec![Stmt::Assign {
            local: 6,
            value: add(local(6), exp(sub(local(7), local(5)))),
        }]
    }));
    out.extend(halves(&|h| {
        vec![
            Stmt::Let {
                local: 8,
                init: div(exp(sub(local(7), local(5))), local(6)),
            },
            Stmt::Store {
                place: flat(
                    9,
                    add(lane(), mul(add(local(4), cu(h as usize * N)), cu(M))),
                ),
                value: local(8),
            },
        ]
    }));
    out
}

fn params() -> Vec<Param> {
    let read = |n: &str, ty| Param::buffer(n, ty, BufferRole::Read, Some(1));
    let rw = |n: &str| Param::buffer(n, F32, BufferRole::ReadWrite, Some(1));
    vec![
        read("q", F32),
        read("k0", F32),
        read("k1", F32),
        read("v", F32),
        read("ia", U32),
        read("ib", U32),
        read("ic", U32),
        rw("s0"),
        rw("s1"),
        rw("p"),
        rw("out"),
    ]
}

fn allocs(a: &MmaAtom) -> Vec<AllocDecl> {
    let (va, vb, vc) = (a.values(0), a.values(1), a.values(2));
    [
        ("fq", va),
        ("fk0", vb),
        ("fk1", vb),
        ("fp", va),
        ("fv", vb),
        ("c0", vc),
        ("d0", vc),
        ("c1", vc),
        ("d1", vc),
        ("c2", vc),
        ("d2", vc),
    ]
    .into_iter()
    .map(|(n, len)| AllocDecl::new(n, MemorySpace::Private, F32, Extent::Static(len)))
    .collect()
}

/// Attention for `M` queries over 8 keys with both GEMMs on `Stmt::Mma`.
pub(crate) fn mma_attention() -> KernelFn {
    let a = atom();
    let (va, vb, vc) = (
        a.values(0) as usize,
        a.values(1) as usize,
        a.values(2) as usize,
    );
    let mut body = vec![
        Stmt::Let {
            local: 0,
            init: lane(),
        },
        gather(0, 0, 4, va),
        gather(1, 1, 5, vb),
        gather(2, 2, 5, vb),
        zero_fragment(5, vc),
        zero_fragment(7, vc),
        mma(0, 1, 5, 6),
        mma(0, 2, 7, 8),
        scatter_scaled(6, 7, vc),
        scatter_scaled(8, 8, vc),
        barrier(),
    ];
    body.extend(softmax_row());
    body.extend([
        barrier(),
        gather(3, 9, 4, va),
        gather(4, 3, 5, vb),
        zero_fragment(9, vc),
        mma(3, 4, 9, 10),
        counted(
            1,
            vc,
            vec![
                load(2, flat(6, add(mul(lane(), cu(vc)), local(1)))),
                load(3, frag(10, local(1))),
                Stmt::Store {
                    place: flat(10, local(2)),
                    value: local(3),
                },
            ],
        ),
    ]);
    let mut locals = vec![LocalDecl::scalar(F32); 9];
    for u in [0, 1, 2, 4] {
        locals[u] = LocalDecl::scalar(U32);
    }
    KernelFn::new("attention_mma", F32, params(), locals, Block(body))
        .with_launch(
            LaunchShape::new([LANES as u32, 1, 1], [1, 1, 1])
                .with_subgroup(LANES as u32, LANES as u32),
        )
        .with_allocs(allocs(&a))
}

/// The id table of operand `which` (0 = A, 1 = B, 2 = C): `table[lane * values + v]` is the logical column-major id.
pub(crate) fn id_table(which: usize) -> Vec<u32> {
    let a = atom();
    (0..LANES as u32)
        .flat_map(|lane| (0..a.values(which)).map(move |v| (lane, v)))
        .map(|(lane, v)| a.id(which, lane, v).unwrap())
        .collect()
}

/// `(Q, K, V)` as plain row-major host matrices: `Q` is `M x K`, `K` is `8 x K` (key, head), `V` is `8 x N` (key, column).
/// Small integers, so every dot product is exact before the scale.
pub(crate) fn attention_inputs() -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let q = (0..M * K)
        .map(|i| ((i / K * 3 + i % K * 5) % 7) as f32 - 3.0)
        .collect();
    let k = (0..K * K)
        .map(|i| ((i / K * 5 + i % K * 3 + i / 5) % 9) as f32 - 4.0)
        .collect();
    let v = (0..K * N)
        .map(|i| ((i / N * 7 + i % N) % 5) as f32 - 2.0)
        .collect();
    (q, k, v)
}

/// The logical column-major operand buffers `(q, k0, k1, v)` the kernel reads.
pub(crate) fn logical_buffers() -> [Vec<f32>; 4] {
    let (q, k, v) = attention_inputs();
    // A of S: M x K column-major, id = row + M * e.
    let qm = (0..M * K).map(|id| q[(id % M) * K + id / M]).collect();
    // B of S, tile h: K x N column-major, id = e + K * c, element K[key = 4h + c][e].
    let tile = |h: usize| -> Vec<f32> {
        (0..K * N)
            .map(|id| k[(N * h + id / K) * K + id % K])
            .collect()
    };
    // B of O: 8 keys x N column-major, id = key + 8 * column.
    let vm = (0..K * N).map(|id| v[(id % K) * N + id / K]).collect();
    [qm, tile(0), tile(1), vm]
}

/// The host result `(scores, out)`, row-major `M x 8` and `M x N`.
pub(crate) fn attention_reference() -> (Vec<f32>, Vec<f32>) {
    let (q, k, v) = attention_inputs();
    let mut s = vec![0f32; M * K];
    let mut out = vec![0f32; M * N];
    for i in 0..M {
        for j in 0..K {
            let mut acc = 0f32;
            for e in 0..K {
                acc += q[i * K + e] * k[j * K + e];
            }
            s[i * K + j] = acc * SCALE as f32;
        }
        let m = s[i * K..(i + 1) * K]
            .iter()
            .fold(s[i * K], |a, &b| a.max(b));
        let l: f32 = s[i * K..(i + 1) * K].iter().map(|x| (x - m).exp()).sum();
        for c in 0..N {
            let mut acc = 0f32;
            for j in 0..K {
                acc += ((s[i * K + j] - m).exp() / l) * v[j * N + c];
            }
            out[i * N + c] = acc;
        }
    }
    (s, out)
}
