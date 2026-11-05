// SPDX-License-Identifier: MIT OR Apache-2.0
//! **A K loop of `Stmt::Mma` accumulating in the fragment** (ROADMAP U9 K6, `docs/options-u9-gemm-family.md`): the
//! GEMM-over-MMA shape every tensor-core kernel has. Two `Stmt::Mma` run back to back over two K slices, the second
//! reading the first's result as its `C` (`d` is `c`), for a dense atom, a 2:4 sparse atom, a block-scaled atom and
//! both. Two things are checked for each: the compiled C agrees with the launch interpreter byte for byte, and the
//! interpreter's chained result equals the sum of the two slices run as separate single-slice MMAs on a zero `C` (so
//! the accumulation itself, not only the C emitter, is pinned). The data are small integers and the scales powers of
//! two, so every sum is exact.

use unpopped::exec::{AllocDecl, ExecId, Extent, LaunchShape, MemorySpace, Scope};
use unpopped::ir::ScalarExpr;
use unpopped::kernel::{Block, BufferRole, KernelFn, LocalDecl, Param, Place, Stmt};
use unpopped::mma::{AtomSpec, ExtraRole, MmaAtom};
use unpopped::oracle::TypedBuffer;
use unpopped::oracle::kernel_eval::KernelArg;
use unpopped::oracle::kernel_launch::run_launch;
use unpopped_cpu_c::CpuC;
use unpopped_vocab::ElementKind as E;
use unpopped_vocab::layout_algebra::Layout;

#[path = "support/cc.rs"]
mod cc;
#[expect(dead_code, reason = "the shared drivers; this file uses a subset")]
#[path = "support/kernel_run.rs"]
mod kernel_run;
#[path = "support/launch_run.rs"]
mod launch_run;
use launch_run::agree_launch;

const LANES: u32 = 4;
const M: u32 = 4;
const N: u32 = 4;
const K: u32 = 8;

fn leaf(size: i64) -> Layout {
    Layout::leaf(size, 1).unwrap()
}

fn toy(input: E, acc: E) -> MmaAtom {
    MmaAtom::new(
        AtomSpec {
            name: "toy.m4n4k8",
            mnk: (M, N, K),
            dtypes: (input, input, acc),
            scope: Scope::Subgroup,
            lanes: LANES,
        },
        [&leaf(32), &leaf(32), &leaf(16)],
    )
    .unwrap()
}

fn sparse(atom: MmaAtom) -> MmaAtom {
    atom.with_sparse_24(&leaf(16), (E::U8, &leaf(8))).unwrap()
}

fn scaled(atom: MmaAtom) -> MmaAtom {
    atom.with_block_scale(4, E::F32, (&leaf(8), &leaf(8)))
        .unwrap()
}

fn u(v: i64) -> ScalarExpr {
    ScalarExpr::ConstInt(v, E::U32)
}

fn slot(buffer: u32, vals: u32, v: u32) -> Place {
    let index = ScalarExpr::Add(
        Box::new(ScalarExpr::Mul(
            Box::new(ScalarExpr::Local(0)),
            Box::new(u(i64::from(vals))),
        )),
        Box::new(u(i64::from(v))),
    );
    Place::Logical {
        buffer,
        coords: vec![index],
    }
}

fn frag(alloc: u32, v: u32) -> Place {
    Place::Alloc {
        alloc,
        index: u(i64::from(v)),
    }
}

/// `(element type, fragment length)` of the A, B and extra operands of one slice, in `Stmt::Mma` order.
fn slice_ops(atom: &MmaAtom) -> Vec<(E, u32)> {
    let (ta, tb, _) = atom.dtypes();
    let mut ops = vec![(ta, atom.values(0)), (tb, atom.values(1))];
    for (i, (_, ty)) in atom.extras().enumerate() {
        ops.push((ty, atom.values(3 + i)));
    }
    ops
}

/// The operand number (in `atom.dims` / `atom.values` terms) of entry `j` of [`slice_ops`].
fn which_of(j: usize) -> usize {
    if j < 2 { j } else { j + 1 }
}

/// Each lane copies its slot of buffer `which` into (or out of) the fragment `alloc`.
fn copy(
    from_buffer: bool,
    ty: E,
    (buf, alloc, n): (u32, u32, u32),
    locals: &mut Vec<LocalDecl>,
) -> Vec<Stmt> {
    let mut out = Vec::new();
    for v in 0..n {
        let l = locals.len() as u32;
        locals.push(LocalDecl::scalar(ty));
        let (src, dst) = (slot(buf, n, v), frag(alloc, v));
        let (src, dst) = if from_buffer { (src, dst) } else { (dst, src) };
        out.push(Stmt::Load {
            local: l,
            place: src,
        });
        out.push(Stmt::Store {
            place: dst,
            value: ScalarExpr::Local(l),
        });
    }
    out
}

/// Buffers and fragments: slice 0's operands (`0..n`), slice 1's (`n..2n`), the zero `C` (`2n`), then `D`/`out` (`2n+1`).
fn chain_kernel(atom: &MmaAtom) -> KernelFn {
    let ops = slice_ops(atom);
    let n = ops.len() as u32;
    let acc = atom.dtypes().2;
    let mut locals = vec![LocalDecl::scalar(E::U32)];
    let mut body = vec![Stmt::Let {
        local: 0,
        init: ScalarExpr::Exec(ExecId::ThreadIdx(0)),
    }];
    for s in 0..2u32 {
        for (j, (ty, len)) in ops.iter().enumerate() {
            let idx = s * n + j as u32;
            body.extend(copy(true, *ty, (idx, idx, *len), &mut locals));
        }
    }
    body.extend(copy(true, acc, (2 * n, 2 * n, atom.values(2)), &mut locals));
    let (c0, d) = (2 * n, 2 * n + 1);
    for s in 0..2u32 {
        body.push(Stmt::Mma {
            atom: atom.clone(),
            a: s * n,
            b: s * n + 1,
            c: if s == 0 { c0 } else { d },
            d,
            extra: (2..n).map(|j| s * n + j).collect(),
        });
    }
    body.extend(copy(false, acc, (d, d, atom.values(2)), &mut locals));
    let mut params: Vec<Param> = (0..2)
        .flat_map(|s| ops.iter().enumerate().map(move |(j, (ty, _))| (s, j, *ty)))
        .map(|(s, j, ty)| Param::buffer(format!("s{s}_{j}"), ty, BufferRole::Read, Some(1)))
        .collect();
    params.push(Param::buffer("c0", acc, BufferRole::Read, Some(1)));
    params.push(Param::buffer("out", acc, BufferRole::ReadWrite, Some(1)));
    let mut allocs: Vec<AllocDecl> = (0..2)
        .flat_map(|s| {
            ops.iter()
                .enumerate()
                .map(move |(j, (ty, len))| (s, j, *ty, *len))
        })
        .map(|(s, j, ty, len)| {
            AllocDecl::new(
                format!("f{s}_{j}"),
                MemorySpace::Private,
                ty,
                Extent::Static(len),
            )
        })
        .collect();
    for name in ["c", "d"] {
        allocs.push(AllocDecl::new(
            name,
            MemorySpace::Private,
            acc,
            Extent::Static(atom.values(2)),
        ));
    }
    KernelFn::new("mma_chain", acc, params, locals, Block(body))
        .with_launch(LaunchShape::new([LANES, 1, 1], [1, 1, 1]).with_subgroup(LANES, LANES))
        .with_allocs(allocs)
}

/// Small deterministic values in `-3..=3`.
fn val(seed: u32, i: usize) -> f64 {
    f64::from(((seed * 7919 + i as u32 * 104_729) >> 3) % 7) - 3.0
}

fn pair(r: usize, g: usize) -> (usize, usize) {
    const PAIRS: [(usize, usize); 6] = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)];
    PAIRS[(r + 2 * g) % 6]
}

fn encode(ty: E, vals: &[f64]) -> KernelArg {
    let bytes: Vec<u8> = vals
        .iter()
        .flat_map(|&v| match ty {
            E::I8 => (v as i8).to_le_bytes().to_vec(),
            E::U8 => (v as u8).to_le_bytes().to_vec(),
            E::I32 => (v as i32).to_le_bytes().to_vec(),
            E::F32 => (v as f32).to_le_bytes().to_vec(),
            other => panic!("no host encoder for {other:?}"),
        })
        .collect();
    KernelArg::Buffer(TypedBuffer::new(
        ty,
        vec![vals.len() as i64],
        vec![1],
        bytes,
    ))
}

fn decode(ty: E, a: &KernelArg) -> Vec<f64> {
    let KernelArg::Buffer(b) = a else {
        unreachable!("a buffer")
    };
    match ty {
        E::I32 => b
            .bytes
            .chunks(4)
            .map(|c| f64::from(i32::from_le_bytes(c.try_into().unwrap())))
            .collect(),
        E::F32 => b
            .bytes
            .chunks(4)
            .map(|c| f64::from(f32::from_le_bytes(c.try_into().unwrap())))
            .collect(),
        other => panic!("no host decoder for {other:?}"),
    }
}

/// The logical data of operand `which` for K slice `slice` (column-major, `atom.dims(which)`).
fn logical(atom: &MmaAtom, which: usize, slice: u32) -> Vec<f64> {
    let (rows, cols) = atom.dims(which);
    let len = (rows * cols) as usize;
    let role = atom
        .extras()
        .nth(which.wrapping_sub(3))
        .map(|(role, _)| role);
    match role {
        Some(ExtraRole::Metadata) => (0..len)
            .map(|id| {
                let (i0, i1) = pair((id % M as usize + slice as usize) % 4, id / M as usize);
                (i0 | i1 << 2) as f64
            })
            .collect(),
        Some(_) => (0..len)
            .map(|i| f64::from(1u32 << ((val(4 + which as u32 + 31 * slice, i) as i32 + 3) % 4)))
            .collect(),
        None => (0..len)
            .map(|i| val(which as u32 + 1 + 17 * slice, i))
            .collect(),
    }
}

/// Every lane's slots of operand `which`, fed through the atom's own layout.
fn fragments(atom: &MmaAtom, which: usize, data: &[f64]) -> Vec<f64> {
    (0..LANES)
        .flat_map(|lane| (0..atom.values(which)).map(move |v| (lane, v)))
        .map(|(lane, v)| data[atom.id(which, lane, v).unwrap() as usize])
        .collect()
}

fn slice_args(atom: &MmaAtom, slice: u32) -> Vec<KernelArg> {
    slice_ops(atom)
        .iter()
        .enumerate()
        .map(|(j, (ty, _))| {
            let w = which_of(j);
            encode(*ty, &fragments(atom, w, &logical(atom, w, slice)))
        })
        .collect()
}

fn zeros(atom: &MmaAtom) -> KernelArg {
    encode(
        atom.dtypes().2,
        &vec![0.0; (LANES * atom.values(2)) as usize],
    )
}

/// The single-slice kernel's output on a zero `C`: the chained kernel with one slice's data, run twice.
fn single(atom: &MmaAtom, slice: u32) -> Vec<f64> {
    // slice 0 data in both slots would double-count, so feed the wanted slice to slot 0 and zero data to slot 1
    let mut args = slice_args(atom, slice);
    let ops = slice_ops(atom);
    for (j, (ty, len)) in ops.iter().enumerate() {
        // an all-zero A and B contribute nothing; extras keep their (valid) metadata and scales
        let zero = if j < 2 {
            vec![0.0; (LANES * len) as usize]
        } else {
            let w = which_of(j);
            fragments(atom, w, &logical(atom, w, slice))
        };
        args.push(encode(*ty, &zero));
    }
    args.push(zeros(atom));
    args.push(zeros(atom));
    run_launch(&chain_kernel(atom), &mut args).expect("the oracle runs it");
    decode(atom.dtypes().2, args.last().unwrap())
}

fn chain_args(atom: &MmaAtom) -> Vec<KernelArg> {
    let mut args = slice_args(atom, 0);
    args.extend(slice_args(atom, 1));
    args.push(zeros(atom));
    args.push(zeros(atom));
    args
}

fn check(tag: &str, atom: &MmaAtom) {
    let k = chain_kernel(atom);
    agree_launch(tag, &k, chain_args(atom));
    let mut run = chain_args(atom);
    run_launch(&k, &mut run).expect("the oracle runs it");
    let chained = decode(atom.dtypes().2, run.last().unwrap());
    let (s0, s1) = (single(atom, 0), single(atom, 1));
    let sum: Vec<f64> = s0.iter().zip(&s1).map(|(a, b)| a + b).collect();
    assert_eq!(chained, sum, "{tag}: chained == slice 0 + slice 1");
    assert!(
        chained.iter().any(|v| *v != 0.0) && s0 != s1,
        "{tag}: the slices differ and the sum is not all zero"
    );
}

#[test]
fn a_dense_atom_accumulates_across_k_slices() {
    check("mma_chain_f32", &toy(E::F32, E::F32));
}

#[test]
fn a_2_4_sparse_atom_accumulates_across_k_slices() {
    check("mma_chain_sp_s8", &sparse(toy(E::I8, E::I32)));
}

#[test]
fn a_block_scaled_atom_accumulates_across_k_slices() {
    check("mma_chain_bs_f32", &scaled(toy(E::F32, E::F32)));
}

#[test]
fn a_sparse_and_scaled_atom_accumulates_across_k_slices() {
    check("mma_chain_sp_bs_f32", &scaled(sparse(toy(E::F32, E::F32))));
}

#[test]
fn the_c_holds_two_mma_calls_with_the_accumulator_reused() {
    use unpopped::backend::Backend;
    let atom = toy(E::F32, E::F32);
    let g = CpuC
        .lower_kernel(&chain_kernel(&atom))
        .expect("a chained MMA lowers");
    assert!(g.source.matches("mma").count() >= 2, "{}", g.source);
}
