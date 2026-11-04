// SPDX-License-Identifier: MIT OR Apache-2.0
//! **The serialized C of a `Stmt::Mma` with extra operands agrees with the launch interpreter** (ROADMAP U7 S3b). A toy
//! 4-lane atom (`M = N = 4`, `K = 8`) carries 2:4 metadata and/or block scales; every lane stores its fragments in
//! `Private` arrays, the MMA runs, each lane writes D out, and the compiled C is compared byte for byte with
//! `run_launch`. Fragments are placed through `MmaAtom::id`, so a broadcast scale (a layout that holds one element
//! in several slots) is fed the same value in each. Scales are powers of two and values small integers, so every
//! float sum is exact and the comparison is by bytes.

use unpopped::exec::{AllocDecl, ExecId, Extent, LaunchShape, MemorySpace, Scope};
use unpopped::ir::ScalarExpr;
use unpopped::kernel::{Block, BufferRole, KernelFn, LocalDecl, Param, Place, Stmt};
use unpopped::mma::{AtomSpec, ExtraRole, MmaAtom};
use unpopped::oracle::TypedBuffer;
use unpopped::oracle::kernel_eval::KernelArg;
use unpopped_vocab::ElementKind as E;
use unpopped_vocab::layout_algebra::Layout;

#[path = "support/cc.rs"]
mod cc;
#[allow(dead_code)]
#[path = "support/kernel_run.rs"]
mod kernel_run;
#[allow(dead_code)]
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

fn scaled(atom: MmaAtom, sfb: &Layout) -> MmaAtom {
    atom.with_block_scale(4, E::F32, (&leaf(8), sfb)).unwrap()
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

/// Element types and fragment lengths of A, B, C, then the extras.
fn operands(atom: &MmaAtom) -> Vec<(E, u32)> {
    let (ta, tb, tacc) = atom.dtypes();
    let mut ops = vec![
        (ta, atom.values(0)),
        (tb, atom.values(1)),
        (tacc, atom.values(2)),
    ];
    for (i, (_, ty)) in atom.extras().enumerate() {
        ops.push((ty, atom.values(3 + i)));
    }
    ops
}

fn kernel(atom: &MmaAtom) -> KernelFn {
    let ops = operands(atom);
    let nin = ops.len() as u32;
    let (_, _, acc) = atom.dtypes();
    let mut allocs: Vec<AllocDecl> = ops
        .iter()
        .enumerate()
        .map(|(i, (ty, n))| {
            AllocDecl::new(
                format!("f{i}"),
                MemorySpace::Private,
                *ty,
                Extent::Static(*n),
            )
        })
        .collect();
    allocs.push(AllocDecl::new(
        "d",
        MemorySpace::Private,
        acc,
        Extent::Static(atom.values(2)),
    ));
    let mut locals = vec![LocalDecl::scalar(E::U32)];
    let mut body = vec![Stmt::Let {
        local: 0,
        init: ScalarExpr::Exec(ExecId::ThreadIdx(0)),
    }];
    for (i, (ty, n)) in ops.iter().enumerate() {
        for v in 0..*n {
            let l = locals.len() as u32;
            locals.push(LocalDecl::scalar(*ty));
            body.push(Stmt::Load {
                local: l,
                place: slot(i as u32, *n, v),
            });
            body.push(Stmt::Store {
                place: frag(i as u32, v),
                value: ScalarExpr::Local(l),
            });
        }
    }
    body.push(Stmt::Mma {
        atom: atom.clone(),
        a: 0,
        b: 1,
        c: 2,
        d: nin,
        extra: (3..nin).collect(),
    });
    for v in 0..atom.values(2) {
        let l = locals.len() as u32;
        locals.push(LocalDecl::scalar(acc));
        body.push(Stmt::Load {
            local: l,
            place: frag(nin, v),
        });
        body.push(Stmt::Store {
            place: slot(nin, atom.values(2), v),
            value: ScalarExpr::Local(l),
        });
    }
    let mut params: Vec<Param> = ops
        .iter()
        .enumerate()
        .map(|(i, (ty, _))| Param::buffer(format!("in{i}"), *ty, BufferRole::Read, Some(1)))
        .collect();
    params.push(Param::buffer("out", acc, BufferRole::ReadWrite, Some(1)));
    KernelFn::new("mma_extras", acc, params, locals, Block(body))
        .with_launch(LaunchShape::new([LANES, 1, 1], [1, 1, 1]).with_subgroup(LANES, LANES))
        .with_allocs(allocs)
}

/// Small deterministic values in `-3..=3`.
fn val(seed: u32, i: usize) -> f64 {
    f64::from(((seed * 7919 + i as u32 * 104_729) >> 3) % 7) - 3.0
}

/// The positions of the two kept values of group `g` of row `r`: every one of the six pairs occurs.
fn pair(r: usize, g: usize) -> (usize, usize) {
    const PAIRS: [(usize, usize); 6] = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)];
    PAIRS[(r + 2 * g) % 6]
}

fn encode(ty: E, vals: &[f64]) -> TypedBuffer {
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
    TypedBuffer::new(ty, vec![vals.len() as i64], vec![1], bytes)
}

/// The logical data of operand `which` (column-major, `atom.dims(which)`).
fn logical(atom: &MmaAtom, which: usize) -> Vec<f64> {
    let (rows, cols) = atom.dims(which);
    let len = (rows * cols) as usize;
    let role = atom
        .extras()
        .nth(which.wrapping_sub(3))
        .map(|(role, _)| role);
    match role {
        Some(ExtraRole::Metadata) => (0..len)
            .map(|id| {
                let (i0, i1) = pair(id % M as usize, id / M as usize);
                (i0 | i1 << 2) as f64
            })
            .collect(),
        // powers of two, so every scaled product stays exact
        Some(_) => (0..len)
            .map(|i| f64::from(1u32 << ((val(4 + which as u32, i) as i32 + 3) % 4)))
            .collect(),
        None => (0..len).map(|i| val(which as u32 + 1, i)).collect(),
    }
}

/// Every lane's slots of operand `which`, fed through the atom's own layout.
fn fragments(atom: &MmaAtom, which: usize, data: &[f64]) -> Vec<f64> {
    (0..LANES)
        .flat_map(|lane| (0..atom.values(which)).map(move |v| (lane, v)))
        .map(|(lane, v)| data[atom.id(which, lane, v).unwrap() as usize])
        .collect()
}

fn args(atom: &MmaAtom) -> Vec<KernelArg> {
    let ops = operands(atom);
    let mut out: Vec<KernelArg> = ops
        .iter()
        .enumerate()
        .map(|(w, (ty, _))| KernelArg::Buffer(encode(*ty, &fragments(atom, w, &logical(atom, w)))))
        .collect();
    let acc = atom.dtypes().2;
    out.push(KernelArg::Buffer(encode(
        acc,
        &vec![0.0; (LANES * atom.values(2)) as usize],
    )));
    out
}

#[test]
fn a_2_4_sparse_atom_agrees_with_the_oracle() {
    let atom = sparse(toy(E::I8, E::I32));
    agree_launch("mma_sp_s8", &kernel(&atom), args(&atom));
}

#[test]
fn a_block_scaled_atom_agrees_with_the_oracle() {
    let atom = scaled(toy(E::F32, E::F32), &leaf(8));
    agree_launch("mma_bs_f32", &kernel(&atom), args(&atom));
}

#[test]
fn a_sparse_and_scaled_atom_agrees_with_the_oracle() {
    let atom = scaled(sparse(toy(E::F32, E::F32)), &leaf(8));
    agree_launch("mma_sp_bs_f32", &kernel(&atom), args(&atom));
}

#[test]
fn a_broadcast_scale_agrees_with_the_oracle() {
    // SFB held twice per lane: slot v and v + 2 are one logical element
    let twice =
        Layout::from_modes(&[Layout::leaf(8, 1).unwrap(), Layout::leaf(2, 0).unwrap()]).unwrap();
    let atom = scaled(toy(E::F32, E::F32), &twice);
    assert_eq!(atom.values(4), 4);
    agree_launch("mma_bs_bcast", &kernel(&atom), args(&atom));
}

#[test]
fn an_extra_operand_missing_from_the_statement_is_refused() {
    use unpopped::backend::Backend;
    let atom = sparse(toy(E::I8, E::I32));
    let mut k = kernel(&atom);
    for st in &mut k.body.0 {
        if let Stmt::Mma { extra, .. } = st {
            extra.clear();
        }
    }
    assert!(unpopped_cpu_c::CpuC.lower_kernel(&k).is_err());
}
