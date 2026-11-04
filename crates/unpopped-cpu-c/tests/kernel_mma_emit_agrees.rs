// SPDX-License-Identifier: MIT OR Apache-2.0
//! **The serialized thread-loop C of a `Stmt::Mma` agrees with the launch interpreter** (ROADMAP U7 S2). A warp of 32
//! lanes stores A, B and C fragments in `Private` arrays, the MMA runs, and each lane writes its D fragment out. The
//! C gathers and scatters through id tables baked from the atom (`MmaAtom::id`), forms `D = A.B + C` in `double` (or
//! `long long`), and is compiled with a real host C compiler and compared byte for byte with `run_launch`.
//!
//! Atoms are data, so besides the shipped s8 atom a **toy `F32` atom with the PTX m16n8k16 layouts** exercises the
//! float path (the shipped f16 and bf16 atoms have no C element type here and are refused with a typed error).

use unpopped::exec::{AllocDecl, ExecId, Extent, LaunchShape, MemorySpace, Scope};
use unpopped::ir::ScalarExpr;
use unpopped::kernel::{Block, BufferRole, KernelFn, LocalDecl, Param, Place, Stmt};
use unpopped::mma::MmaAtom;
use unpopped::oracle::TypedBuffer;
use unpopped::oracle::kernel_eval::KernelArg;
use unpopped_vocab::ElementKind;
use unpopped_vocab::layout_algebra::Layout;

#[path = "support/cc.rs"]
mod cc;
// The shared compile-and-run drivers; this file uses a subset of them.
#[allow(dead_code)]
#[path = "support/kernel_run.rs"]
mod kernel_run;
#[allow(dead_code)]
#[path = "support/launch_run.rs"]
mod launch_run;
use launch_run::agree_launch;

const U32: ElementKind = ElementKind::U32;

fn u(v: i64) -> ScalarExpr {
    ScalarExpr::ConstInt(v, U32)
}

fn at(lane: &ScalarExpr, vals: u32, v: u32) -> ScalarExpr {
    ScalarExpr::Add(
        Box::new(ScalarExpr::Mul(
            Box::new(lane.clone()),
            Box::new(u(i64::from(vals))),
        )),
        Box::new(u(i64::from(v))),
    )
}

fn buf(buffer: u32, index: ScalarExpr) -> Place {
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

/// Copy `vals` slots of buffer `which` into `Private` array `which`; return the statements.
fn copy_in(which: u32, ty: ElementKind, vals: u32, locals: &mut Vec<LocalDecl>) -> Vec<Stmt> {
    let lane = ScalarExpr::Local(0);
    let mut out = Vec::new();
    for v in 0..vals {
        let l = locals.len() as u32;
        locals.push(LocalDecl::scalar(ty));
        out.push(Stmt::Load {
            local: l,
            place: buf(which, at(&lane, vals, v)),
        });
        out.push(Stmt::Store {
            place: frag(which, v),
            value: ScalarExpr::Local(l),
        });
    }
    out
}

fn kernel(atom: &MmaAtom) -> KernelFn {
    let (ta, tb, tacc) = atom.dtypes();
    let vals = [atom.values(0), atom.values(1), atom.values(2)];
    let tys = [ta, tb, tacc, tacc];
    let lens = [vals[0], vals[1], vals[2], vals[2]];
    let allocs = (0..4)
        .map(|i| {
            AllocDecl::new(
                ["a", "b", "c", "d"][i],
                MemorySpace::Private,
                tys[i],
                Extent::Static(lens[i]),
            )
        })
        .collect();
    let mut locals = vec![LocalDecl::scalar(U32)];
    let mut body = vec![Stmt::Let {
        local: 0,
        init: ScalarExpr::Exec(ExecId::ThreadIdx(0)),
    }];
    for w in 0..3 {
        body.extend(copy_in(w as u32, tys[w], vals[w], &mut locals));
    }
    body.push(Stmt::Mma {
        atom: atom.clone(),
        a: 0,
        b: 1,
        c: 2,
        d: 3,
    });
    for v in 0..vals[2] {
        let l = locals.len() as u32;
        locals.push(LocalDecl::scalar(tacc));
        body.push(Stmt::Load {
            local: l,
            place: frag(3, v),
        });
        body.push(Stmt::Store {
            place: buf(3, at(&ScalarExpr::Local(0), vals[2], v)),
            value: ScalarExpr::Local(l),
        });
    }
    let mut k = KernelFn::new(
        "mma",
        tacc,
        vec![
            Param::buffer("fa", ta, BufferRole::Read, Some(1)),
            Param::buffer("fb", tb, BufferRole::Read, Some(1)),
            Param::buffer("fc", tacc, BufferRole::Read, Some(1)),
            Param::buffer("fd", tacc, BufferRole::ReadWrite, Some(1)),
        ],
        locals,
        Block(body),
    )
    .with_allocs(allocs);
    k.launch = Some(LaunchShape::new([32, 1, 1], [1, 1, 1]).with_subgroup(32, 32));
    k
}

/// A small deterministic value in `-4..=4`, different per `(seed, i)`.
fn val(seed: u32, i: u32) -> i32 {
    ((seed
        .wrapping_mul(2_654_435_761)
        .wrapping_add(i.wrapping_mul(40_503)))
        >> 7) as i32
        % 9
        - 4
}

fn buffer(ty: ElementKind, seed: u32, len: u32) -> TypedBuffer {
    let bytes: Vec<u8> = (0..len)
        .flat_map(|i| {
            let v = val(seed, i);
            match ty {
                ElementKind::I8 => (v as i8).to_le_bytes().to_vec(),
                ElementKind::I32 => v.to_le_bytes().to_vec(),
                ElementKind::F32 => (v as f32 * 0.125).to_le_bytes().to_vec(),
                other => panic!("no host encoder for {other:?}"),
            }
        })
        .collect();
    TypedBuffer::new(ty, vec![i64::from(len)], vec![1], bytes)
}

fn args(atom: &MmaAtom) -> Vec<KernelArg> {
    let (ta, tb, tacc) = atom.dtypes();
    let n = |w| 32 * atom.values(w);
    vec![
        KernelArg::Buffer(buffer(ta, 1, n(0))),
        KernelArg::Buffer(buffer(tb, 2, n(1))),
        KernelArg::Buffer(buffer(tacc, 3, n(2))),
        KernelArg::Buffer(buffer(tacc, 0, n(2))),
    ]
}

fn leaves(modes: &[(i64, i64)]) -> Layout {
    let l: Vec<Layout> = modes
        .iter()
        .map(|&(s, d)| Layout::leaf(s, d).unwrap())
        .collect();
    Layout::from_modes(&l).unwrap()
}

/// The PTX m16n8k16 fragment layouts with `F32` operands and accumulator: atoms are data.
fn toy_f32_atom() -> MmaAtom {
    let a = leaves(&[(4, 32), (8, 1), (2, 16), (2, 8), (2, 128)]);
    let b = leaves(&[(4, 2), (8, 16), (2, 1), (2, 8)]);
    let c = leaves(&[(4, 32), (8, 1), (2, 16), (2, 8)]);
    MmaAtom::new(
        "toy.m16n8k16.f32.f32",
        (16, 8, 16),
        (ElementKind::F32, ElementKind::F32, ElementKind::F32),
        (Scope::Subgroup, 32),
        [&a, &b, &c],
    )
    .unwrap()
}

#[test]
fn the_s8_atom_agrees_with_the_oracle() {
    let atom = MmaAtom::nvidia_m16n8k32_s8_s32();
    agree_launch("mma_s8", &kernel(&atom), args(&atom));
}

#[test]
fn a_float_atom_agrees_with_the_oracle() {
    let atom = toy_f32_atom();
    agree_launch("mma_f32", &kernel(&atom), args(&atom));
}

#[test]
fn a_float_atom_that_rounds_agrees_with_the_oracle() {
    // C = 2^24 + products of 0.125 multiples: the f32 sum is inexact, so the order and the single rounding matter
    let atom = toy_f32_atom();
    let mut a = args(&atom);
    let c: Vec<u8> = (0..32 * atom.values(2))
        .flat_map(|_| 16_777_216.0_f32.to_le_bytes())
        .collect();
    a[2] = KernelArg::Buffer(TypedBuffer::new(
        ElementKind::F32,
        vec![i64::from(32 * atom.values(2))],
        vec![1],
        c,
    ));
    agree_launch("mma_f32_round", &kernel(&atom), a);
}

#[test]
fn the_f16_atom_is_refused_with_a_typed_error() {
    use unpopped::backend::{Backend, LowerError};
    let atom = MmaAtom::nvidia_m16n8k16_f16_f32();
    match unpopped_cpu_c::CpuC.lower_kernel(&kernel(&atom)) {
        Err(LowerError::UnsupportedKernel { detail }) => {
            assert!(detail.contains("F16"), "{detail}");
        }
        other => panic!("expected a typed refusal, got {other:?}"),
    }
}
