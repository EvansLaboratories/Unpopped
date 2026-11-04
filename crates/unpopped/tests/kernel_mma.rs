// SPDX-License-Identifier: MIT OR Apache-2.0
//! **`Stmt::Mma` in the launch interpreter** (ROADMAP U7 S2). A warp of 32 lanes hands the oracle its A, B and C
//! fragments, the oracle gathers them through the atom's layouts, multiplies, and scatters D back. The fragments
//! the host feeds in and reads out are placed by the PTX ISA fragment tables written here as plain `(row, col)`
//! formulas, independently of the strides in `unpopped::mma`, and the expected D is a plain triple loop over the
//! logical matrices. So a wrong gather, a wrong scatter, or a wrong atom layout all show as a wrong number.
//!
//! The checker's refusals (a subgroup of the wrong size, a fragment of the wrong type or length, an MMA under
//! divergent control) and the uninitialized-fragment trap each have a program that raises them and a control that
//! does not.

use unpopped::exec::{AllocDecl, ExecId, Extent, LaunchShape, MemorySpace};
use unpopped::ir::ScalarExpr;
use unpopped::kernel::{Block, BufferRole, KernelError, KernelFn, LocalDecl, Param, Place, Stmt};
use unpopped::mma::MmaAtom;
use unpopped::oracle::TypedBuffer;
use unpopped::oracle::kernel_eval::{KernelArg, KernelTrap};
use unpopped::oracle::kernel_launch::run_launch;
use unpopped_vocab::ElementKind;

const U32: ElementKind = ElementKind::U32;

fn u(v: i64) -> ScalarExpr {
    ScalarExpr::ConstInt(v, U32)
}

fn add(a: ScalarExpr, b: ScalarExpr) -> ScalarExpr {
    ScalarExpr::Add(Box::new(a), Box::new(b))
}

fn mul(a: ScalarExpr, b: ScalarExpr) -> ScalarExpr {
    ScalarExpr::Mul(Box::new(a), Box::new(b))
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

/// PTX ISA fragment tables: `(row, col)` of value `i` of `lane`, operand 0/1/2 = A/B/C.
fn ptx_k16(which: usize, lane: u32, i: u32) -> (u32, u32) {
    let (gid, tig) = (lane >> 2, lane % 4);
    match which {
        0 => (gid + 8 * ((i / 2) % 2), tig * 2 + (i % 2) + 8 * (i / 4)),
        1 => (tig * 2 + (i % 2) + 8 * (i / 2), gid),
        _ => (gid + 8 * (i / 2), tig * 2 + (i % 2)),
    }
}

fn ptx_k32(which: usize, lane: u32, i: u32) -> (u32, u32) {
    let (gid, tig) = (lane >> 2, lane % 4);
    match which {
        0 => (gid + 8 * ((i / 4) % 2), tig * 4 + (i % 4) + 16 * (i / 8)),
        1 => (tig * 4 + (i % 4) + 16 * (i / 4), gid),
        _ => (gid + 8 * (i / 2), tig * 2 + (i % 2)),
    }
}

type Table = fn(usize, u32, u32) -> (u32, u32);

/// The kernel: each lane copies its A/B/C fragment slots from the input buffers (buffers 0..3, `lane * values +
/// v`), runs the MMA, and writes its D fragment to buffer 3.
fn mma_kernel(atom: &MmaAtom, skip_c_slot: Option<u32>) -> KernelFn {
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
    for which in 0..3 {
        let skip = if which == 2 { skip_c_slot } else { None };
        body.extend(copy_in(which, tys[which], vals[which], skip, &mut locals));
    }
    body.push(Stmt::Mma {
        atom: atom.clone(),
        a: 0,
        b: 1,
        c: 2,
        d: 3,
        extra: vec![],
    });
    body.extend(copy_out(tacc, vals[2], &mut locals));
    KernelFn::new(
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
    .with_launch(LaunchShape::new([32, 1, 1], [1, 1, 1]).with_subgroup(32, 32))
    .with_allocs(allocs)
}

/// `buf[lane * vals + v]` for the lane held in local 0.
fn lane_slot(vals: u32, v: u32) -> ScalarExpr {
    add(
        mul(ScalarExpr::Local(0), u(i64::from(vals))),
        u(i64::from(v)),
    )
}

/// Copy the slots of buffer `which` into `Private` array `which` (`skip` leaves one slot unstored).
fn copy_in(
    which: usize,
    ty: ElementKind,
    vals: u32,
    skip: Option<u32>,
    locals: &mut Vec<LocalDecl>,
) -> Vec<Stmt> {
    let mut out = Vec::new();
    for v in (0..vals).filter(|v| Some(*v) != skip) {
        let l = locals.len() as u32;
        locals.push(LocalDecl::scalar(ty));
        out.push(Stmt::Load {
            local: l,
            place: buf(which as u32, lane_slot(vals, v)),
        });
        out.push(Stmt::Store {
            place: frag(which as u32, v),
            value: ScalarExpr::Local(l),
        });
    }
    out
}

/// Write the D fragment (`Private` array 3) to buffer 3.
fn copy_out(ty: ElementKind, vals: u32, locals: &mut Vec<LocalDecl>) -> Vec<Stmt> {
    let mut out = Vec::new();
    for v in 0..vals {
        let l = locals.len() as u32;
        locals.push(LocalDecl::scalar(ty));
        out.push(Stmt::Load {
            local: l,
            place: frag(3, v),
        });
        out.push(Stmt::Store {
            place: buf(3, lane_slot(vals, v)),
            value: ScalarExpr::Local(l),
        });
    }
    out
}

/// Deterministic small values: exactly representable in f16 and bf16, so every product and sum is exact.
fn val(seed: u32, i: u32) -> i32 {
    ((seed
        .wrapping_mul(2_654_435_761)
        .wrapping_add(i.wrapping_mul(40_503)))
        >> 7) as i32
        % 9
        - 4
}

/// The host-side logical matrices (column-major) of an atom, as integers.
struct Logical {
    a: Vec<i32>,
    b: Vec<i32>,
    c: Vec<i32>,
}

fn logical(atom: &MmaAtom) -> Logical {
    let (m, n, k) = atom.mnk();
    let make = |seed, len: u32| (0..len).map(|i| val(seed, i)).collect();
    Logical {
        a: make(1, m * k),
        b: make(2, k * n),
        c: make(3, m * n),
    }
}

/// Lay a logical matrix into per-lane fragments through the PTX `table`.
fn fragments(table: Table, which: usize, rows: u32, vals: u32, logical: &[i32]) -> Vec<i32> {
    let mut out = Vec::new();
    for lane in 0..32 {
        for v in 0..vals {
            let (r, c) = table(which, lane, v);
            out.push(logical[(r + rows * c) as usize]);
        }
    }
    out
}

/// The plain triple loop `D = A.B + C`, column-major.
fn reference(atom: &MmaAtom, l: &Logical) -> Vec<i32> {
    let (m, n, k) = atom.mnk();
    let mut d = vec![0; (m * n) as usize];
    for col in 0..n {
        for row in 0..m {
            let mut s = l.c[(row + m * col) as usize];
            for i in 0..k {
                s += l.a[(row + m * i) as usize] * l.b[(i + k * col) as usize];
            }
            d[(row + m * col) as usize] = s;
        }
    }
    d
}

fn buffer(ty: ElementKind, vals: &[i32]) -> TypedBuffer {
    let bytes: Vec<u8> = vals
        .iter()
        .flat_map(|&v| match ty {
            ElementKind::I8 => (v as i8).to_le_bytes().to_vec(),
            ElementKind::I32 => v.to_le_bytes().to_vec(),
            ElementKind::F32 => (v as f32).to_le_bytes().to_vec(),
            ElementKind::F16 => half::f16::from_f32(v as f32).to_le_bytes().to_vec(),
            ElementKind::Bf16 => half::bf16::from_f32(v as f32).to_le_bytes().to_vec(),
            other => panic!("no host encoder for {other:?}"),
        })
        .collect();
    TypedBuffer::new(ty, vec![vals.len() as i64], vec![1], bytes)
}

fn read_back(ty: ElementKind, b: &TypedBuffer) -> Vec<i32> {
    match ty {
        ElementKind::I32 => b
            .bytes
            .chunks(4)
            .map(|c| i32::from_le_bytes(c.try_into().unwrap()))
            .collect(),
        ElementKind::F32 => b
            .bytes
            .chunks(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()) as i32)
            .collect(),
        other => panic!("no host decoder for {other:?}"),
    }
}

fn args(atom: &MmaAtom, table: Table, l: &Logical) -> Vec<KernelArg> {
    let (m, n, k) = atom.mnk();
    let (ta, tb, tacc) = atom.dtypes();
    let fa = fragments(table, 0, m, atom.values(0), &l.a);
    let fb = fragments(table, 1, k, atom.values(1), &l.b);
    let fc = fragments(table, 2, m, atom.values(2), &l.c);
    let _ = n;
    vec![
        KernelArg::Buffer(buffer(ta, &fa)),
        KernelArg::Buffer(buffer(tb, &fb)),
        KernelArg::Buffer(buffer(tacc, &fc)),
        KernelArg::Buffer(buffer(tacc, &vec![0; fc.len()])),
    ]
}

/// Run the MMA and compare each lane's D fragment with the reference D placed through the PTX table.
fn agrees(atom: &MmaAtom, table: Table) {
    let l = logical(atom);
    let (m, ..) = atom.mnk();
    let want = fragments(table, 2, m, atom.values(2), &reference(atom, &l));
    let mut a = args(atom, table, &l);
    run_launch(&mma_kernel(atom, None), &mut a)
        .unwrap_or_else(|e| panic!("{}: {e:?}", atom.name()));
    let KernelArg::Buffer(out) = &a[3] else {
        panic!("fd is a buffer")
    };
    assert_eq!(read_back(atom.dtypes().2, out), want, "{}", atom.name());
}

#[test]
fn the_f16_atom_computes_d_through_the_ptx_fragment_tables() {
    agrees(&MmaAtom::nvidia_m16n8k16_f16_f32(), ptx_k16);
}

#[test]
fn the_bf16_atom_computes_d_through_the_ptx_fragment_tables() {
    agrees(&MmaAtom::nvidia_m16n8k16_bf16_f32(), ptx_k16);
}

#[test]
fn the_s8_atom_computes_d_through_the_ptx_fragment_tables() {
    agrees(&MmaAtom::nvidia_m16n8k32_s8_s32(), ptx_k32);
}

#[test]
fn an_accumulator_that_overflows_wraps_like_the_hardware_integer() {
    // all A = B = 127, K = 32: 32 * 127 * 127 = 516128; start C at i32::MAX - 100 and the sum wraps
    let atom = MmaAtom::nvidia_m16n8k32_s8_s32();
    let l = Logical {
        a: vec![127; 16 * 32],
        b: vec![127; 32 * 8],
        c: vec![i32::MAX - 100; 16 * 8],
    };
    let mut a = args(&atom, ptx_k32, &l);
    run_launch(&mma_kernel(&atom, None), &mut a).unwrap();
    let KernelArg::Buffer(out) = &a[3] else {
        panic!("fd is a buffer")
    };
    let want = (i32::MAX - 100).wrapping_add(516_128);
    assert!(want < 0, "control: the sum does overflow");
    assert!(read_back(ElementKind::I32, out).iter().all(|&x| x == want));
}

#[test]
fn a_fragment_slot_never_stored_traps() {
    let atom = MmaAtom::nvidia_m16n8k16_f16_f32();
    let l = logical(&atom);
    let mut a = args(&atom, ptx_k16, &l);
    assert_eq!(
        run_launch(&mma_kernel(&atom, Some(2)), &mut a),
        Err(KernelTrap::UninitializedPrivate { alloc: 2 })
    );
    // control: the same kernel with the slot stored runs
    run_launch(&mma_kernel(&atom, None), &mut args(&atom, ptx_k16, &l)).unwrap();
}

fn refused(k: &KernelFn) -> &'static str {
    let mut a = args(
        &MmaAtom::nvidia_m16n8k16_f16_f32(),
        ptx_k16,
        &logical(&MmaAtom::nvidia_m16n8k16_f16_f32()),
    );
    match run_launch(k, &mut a) {
        Err(KernelTrap::Malformed(KernelError::MmaUnsupported { what })) => what,
        other => panic!("expected an MMA refusal, got {other:?}"),
    }
}

#[test]
fn the_checker_refuses_a_wrong_subgroup_a_wrong_fragment_and_no_launch() {
    let atom = MmaAtom::nvidia_m16n8k16_f16_f32();
    let mut wrong_sg = mma_kernel(&atom, None);
    wrong_sg.launch = Some(LaunchShape::new([32, 1, 1], [1, 1, 1]).with_subgroup(16, 32));
    assert!(refused(&wrong_sg).contains("subgroup size"));

    let mut wrong_len = mma_kernel(&atom, None);
    wrong_len.allocs[0] = AllocDecl::new(
        "a",
        MemorySpace::Private,
        ElementKind::F16,
        Extent::Static(7),
    );
    assert!(refused(&wrong_len).contains("length"));

    // the statement names a bf16 atom but the fragments were declared f16
    let mut wrong_ty = mma_kernel(&atom, None);
    for st in &mut wrong_ty.body.0 {
        if let Stmt::Mma { atom, .. } = st {
            *atom = MmaAtom::nvidia_m16n8k16_bf16_f32();
        }
    }
    assert!(refused(&wrong_ty).contains("element type"));

    let mut shared = mma_kernel(&atom, None);
    shared.allocs[1] = AllocDecl::new(
        "b",
        MemorySpace::Shared,
        ElementKind::F16,
        Extent::Static(4),
    );
    assert!(refused(&shared).contains("Private"));
}

#[test]
fn an_mma_under_divergent_control_is_refused() {
    let atom = MmaAtom::nvidia_m16n8k16_f16_f32();
    let k = mma_kernel(&atom, None);
    let body = k.body.0.clone();
    let split = body
        .iter()
        .position(|s| matches!(s, Stmt::Mma { .. }))
        .unwrap();
    let mut guarded = body;
    let mma = guarded.remove(split);
    guarded.insert(
        split,
        Stmt::If {
            cond: ScalarExpr::Binary(
                unpopped::ir::BinaryOp::CmpLt,
                Box::new(ScalarExpr::Local(0)),
                Box::new(u(16)),
            ),
            then: Block(vec![mma]),
            els: Block(vec![]),
        },
    );
    let mut bad = k.clone();
    bad.body = Block(guarded);
    let mut a = args(&atom, ptx_k16, &logical(&atom));
    assert_eq!(
        run_launch(&bad, &mut a),
        Err(KernelTrap::Malformed(KernelError::BarrierDivergent))
    );
}
