// SPDX-License-Identifier: MIT OR Apache-2.0
//! Shared harness for the NVIDIA `mma.sync` atom run tests (ROADMAP U7 theme 3): one warp hands the oracle its A, B
//! and C fragments, runs `Stmt::Mma`, and the D fragments are compared with a plain triple loop over the logical
//! matrices placed through a `(row, col)` formula table written in the test, independently of `unpopped::mma`.

use unpopped::exec::{AllocDecl, ExecId, Extent, LaunchShape, MemorySpace};
use unpopped::ir::ScalarExpr;
use unpopped::kernel::{Block, BufferRole, KernelFn, LocalDecl, Param, Place, Stmt};
use unpopped::mma::MmaAtom;
use unpopped::oracle::TypedBuffer;
use unpopped::oracle::kernel_eval::{KernelArg, KernelTrap};
use unpopped::oracle::kernel_launch::run_launch;
use unpopped_vocab::ElementKind::{self, Bf16, F16, F32, F64, I8, I32, U8, U32};
use unpopped_vocab::{Fp8E4M3FN, Fp8E5M2};

/// `(row, col)` of value `i` of `lane` for operand 0/1/2 = A/B/C.
pub(crate) type Table = fn(usize, u32, u32) -> (u32, u32);

fn u(v: i64) -> ScalarExpr {
    ScalarExpr::ConstInt(v, U32)
}

fn slot(buffer: u32, vals: u32, v: u32) -> Place {
    let lane = ScalarExpr::Mul(Box::new(ScalarExpr::Local(0)), Box::new(u(vals.into())));
    Place::Logical {
        buffer,
        coords: vec![ScalarExpr::Add(Box::new(lane), Box::new(u(v.into())))],
    }
}

fn frag(alloc: u32, v: u32) -> Place {
    Place::Alloc {
        alloc,
        index: u(v.into()),
    }
}

/// The four `Private` fragment allocations (A, B, C, D) for `atom`.
fn fragment_allocs(tys: [ElementKind; 4], vals: [u32; 4]) -> Vec<AllocDecl> {
    ["a", "b", "c", "d"]
        .iter()
        .enumerate()
        .map(|(i, n)| AllocDecl::new(*n, MemorySpace::Private, tys[i], Extent::Static(vals[i])))
        .collect()
}

/// The four buffer parameters (A, B, C read-only, D written).
fn buffer_params(tys: [ElementKind; 4]) -> Vec<Param> {
    vec![
        Param::buffer("fa", tys[0], BufferRole::Read, Some(1)),
        Param::buffer("fb", tys[1], BufferRole::Read, Some(1)),
        Param::buffer("fc", tys[2], BufferRole::Read, Some(1)),
        Param::buffer("fd", tys[3], BufferRole::ReadWrite, Some(1)),
    ]
}

/// Each lane copies its A/B/C slots from buffers 0..3 into `Private` fragments, runs the MMA, writes D to buffer 3.
pub(crate) fn mma_kernel(atom: &MmaAtom) -> KernelFn {
    let (ta, tb, tacc) = atom.dtypes();
    let tys = [ta, tb, tacc, tacc];
    let vals = [
        atom.values(0),
        atom.values(1),
        atom.values(2),
        atom.values(2),
    ];
    let mut locals = vec![LocalDecl::scalar(U32)];
    let mut body = vec![Stmt::Let {
        local: 0,
        init: ScalarExpr::Exec(ExecId::ThreadIdx(0)),
    }];
    let mut hop = |from: Place, to: Place, ty: ElementKind, body: &mut Vec<Stmt>| {
        let local = locals.len() as u32;
        locals.push(LocalDecl::scalar(ty));
        body.push(Stmt::Load { local, place: from });
        body.push(Stmt::Store {
            place: to,
            value: ScalarExpr::Local(local),
        });
    };
    for w in 0..3 {
        for v in 0..vals[w] {
            hop(
                slot(w as u32, vals[w], v),
                frag(w as u32, v),
                tys[w],
                &mut body,
            );
        }
    }
    body.push(Stmt::Mma {
        atom: atom.clone(),
        a: 0,
        b: 1,
        c: 2,
        d: 3,
        extra: vec![],
    });
    for v in 0..vals[3] {
        hop(frag(3, v), slot(3, vals[3], v), tacc, &mut body);
    }
    KernelFn::new("mma", tacc, buffer_params(tys), locals, Block(body))
        .with_launch(LaunchShape::new([32, 1, 1], [1, 1, 1]).with_subgroup(32, 32))
        .with_allocs(fragment_allocs(tys, vals))
}

/// A deterministic value of element `i` of operand `seed`, exactly representable in `ty`: the full range for the
/// 8-bit integers (so a signed/unsigned mix-up changes the answer), small integers for the floats, lattice fractions for fp8.
pub(crate) fn value(ty: ElementKind, seed: u32, i: u32) -> f64 {
    let h = seed
        .wrapping_mul(2_654_435_761)
        .wrapping_add(i.wrapping_mul(40_503))
        >> 7;
    match ty {
        I8 => f64::from(h % 256) - 128.0,
        U8 => f64::from(h % 256),
        // fractions on each fp8 lattice (3 and 2 mantissa bits), so a decode of the wrong format changes the answer
        ElementKind::Fp8E4M3FN => f64::from(h % 31) / 8.0 - 1.875,
        ElementKind::Fp8E5M2 => f64::from(h % 17) / 4.0 - 2.0,
        _ => f64::from(h % 9) - 4.0,
    }
}

fn encode(ty: ElementKind, x: f64) -> Vec<u8> {
    match ty {
        I8 => (x as i8).to_le_bytes().to_vec(),
        U8 => (x as u8).to_le_bytes().to_vec(),
        I32 => (x as i32).to_le_bytes().to_vec(),
        F32 => (x as f32).to_le_bytes().to_vec(),
        F64 => x.to_le_bytes().to_vec(),
        F16 => half::f16::from_f64(x).to_le_bytes().to_vec(),
        Bf16 => half::bf16::from_f64(x).to_le_bytes().to_vec(),
        ElementKind::Fp8E4M3FN => vec![Fp8E4M3FN::from_f32(x as f32).0],
        ElementKind::Fp8E5M2 => vec![Fp8E5M2::from_f32(x as f32).0],
        other => panic!("no host encoder for {other:?}"),
    }
}

fn decode(ty: ElementKind, b: &[u8]) -> f64 {
    match ty {
        I32 => f64::from(i32::from_le_bytes(b.try_into().unwrap())),
        F32 => f64::from(f32::from_le_bytes(b.try_into().unwrap())),
        F64 => f64::from_le_bytes(b.try_into().unwrap()),
        other => panic!("no host decoder for {other:?}"),
    }
}

fn buffer(ty: ElementKind, xs: &[f64]) -> TypedBuffer {
    let bytes = xs.iter().flat_map(|&x| encode(ty, x)).collect();
    TypedBuffer::new(ty, vec![xs.len() as i64], vec![1], bytes)
}

/// The logical matrices of `atom` (column-major): A is `m x k`, B `k x n`, C `m x n`.
pub(crate) struct Logical {
    pub(crate) a: Vec<f64>,
    pub(crate) b: Vec<f64>,
    pub(crate) c: Vec<f64>,
}

pub(crate) fn logical(atom: &MmaAtom) -> Logical {
    let (m, n, k) = atom.mnk();
    let (ta, tb, tacc) = atom.dtypes();
    let make = |ty, seed, len: u32| (0..len).map(|i| value(ty, seed, i)).collect();
    Logical {
        a: make(ta, 1, m * k),
        b: make(tb, 2, k * n),
        c: make(if tacc == I32 { I8 } else { F32 }, 3, m * n),
    }
}

/// Lay a logical matrix into per-lane fragments through `table`.
pub(crate) fn fragments(table: Table, which: usize, rows: u32, vals: u32, x: &[f64]) -> Vec<f64> {
    (0..32)
        .flat_map(|lane| (0..vals).map(move |v| (lane, v)))
        .map(|(lane, v)| {
            let (r, c) = table(which, lane, v);
            x[(r + rows * c) as usize]
        })
        .collect()
}

/// `D = f(A).f(B) + C` as a plain triple loop in `f64`, `f` applied to each operand element (identity or tf32).
pub(crate) fn reference(atom: &MmaAtom, l: &Logical, f: fn(f64) -> f64) -> Vec<f64> {
    let (m, n, k) = atom.mnk();
    let (m, n, k) = (m as usize, n as usize, k as usize);
    let mut d = vec![0.0; m * n];
    for col in 0..n {
        for row in 0..m {
            d[row + m * col] = (0..k).fold(l.c[row + m * col], |s, i| {
                s + f(l.a[row + m * i]) * f(l.b[i + k * col])
            });
        }
    }
    d
}

/// Run the MMA over `l` through `table`; the D fragments each lane wrote.
pub(crate) fn run(atom: &MmaAtom, table: Table, l: &Logical) -> Result<Vec<f64>, KernelTrap> {
    let (m, _, k) = atom.mnk();
    let (ta, tb, tacc) = atom.dtypes();
    let fc = fragments(table, 2, m, atom.values(2), &l.c);
    let mut args = vec![
        KernelArg::Buffer(buffer(ta, &fragments(table, 0, m, atom.values(0), &l.a))),
        KernelArg::Buffer(buffer(tb, &fragments(table, 1, k, atom.values(1), &l.b))),
        KernelArg::Buffer(buffer(tacc, &fc)),
        KernelArg::Buffer(buffer(tacc, &vec![0.0; fc.len()])),
    ];
    run_launch(&mma_kernel(atom), &mut args)?;
    let KernelArg::Buffer(out) = &args[3] else {
        unreachable!("fd is a buffer")
    };
    Ok(out
        .bytes
        .chunks(out.bytes.len() / fc.len())
        .map(|b| decode(tacc, b))
        .collect())
}

/// Run and compare with the reference D placed through the same table.
pub(crate) fn agrees(atom: &MmaAtom, table: Table) {
    let l = logical(atom);
    let (m, ..) = atom.mnk();
    let want = fragments(table, 2, m, atom.values(2), &reference(atom, &l, |x| x));
    let got = run(atom, table, &l).unwrap_or_else(|e| panic!("{}: {e:?}", atom.name()));
    assert_eq!(got, want, "{}", atom.name());
}

/// Every fragment element of `atom` sits where `table` (the PTX text as `(row, col)` formulas) says, the table covers
/// each matrix once, and the atom states its shape and types.
pub(crate) fn places(
    atom: &MmaAtom,
    table: Table,
    mnk: (u32, u32, u32),
    tys: (ElementKind, ElementKind, ElementKind),
) {
    assert_eq!((atom.mnk(), atom.dtypes()), (mnk, tys), "{}", atom.name());
    let (m, n, k) = mnk;
    for (which, rows, cols) in [(0, m, k), (1, k, n), (2, m, n)] {
        let mut seen = vec![0u32; (rows * cols) as usize];
        for lane in 0..atom.lanes() {
            for v in 0..atom.values(which) {
                let (r, c) = table(which, lane, v);
                let id = r + rows * c;
                assert_eq!(
                    atom.id(which, lane, v),
                    Some(id),
                    "{} op {which} lane {lane} v {v}",
                    atom.name()
                );
                seen[id as usize] += 1;
            }
        }
        assert!(
            seen.iter().all(|&c| c == 1),
            "{} table {which} is not a bijection",
            atom.name()
        );
    }
}
