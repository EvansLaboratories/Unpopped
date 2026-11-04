// SPDX-License-Identifier: MIT OR Apache-2.0
//! **2:4 sparse and block-scaled MMA atoms in the launch interpreter** (ROADMAP U7 S3). A toy 4-lane atom
//! (`M = N = 4`, `K = 8`) whose every layout is the identity (`lane + 4 * value` is the logical id), so the host
//! places fragments by a one-line formula and the expected D is a plain triple loop over the **dense** logical
//! matrices: the sparse `A` is expanded by the host from the metadata it also feeds the oracle, the scales are
//! applied element by element. A wrong expansion, a wrong scale index or a wrong operand order shows as a wrong number.

use unpopped::exec::{AllocDecl, ExecId, Extent, LaunchShape, MemorySpace, Scope};
use unpopped::ir::ScalarExpr;
use unpopped::kernel::{Block, BufferRole, KernelError, KernelFn, LocalDecl, Param, Place, Stmt};
use unpopped::mma::{AtomSpec, MmaAtom, MmaAtomError};
use unpopped::oracle::TypedBuffer;
use unpopped::oracle::kernel_eval::{KernelArg, KernelTrap};
use unpopped::oracle::kernel_launch::run_launch;
use unpopped_vocab::ElementKind as E;
use unpopped_vocab::layout_algebra::Layout;

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

/// The compressed, identity-laid `A` and the `M x K/4` metadata operand.
fn sparse(atom: MmaAtom) -> Result<MmaAtom, MmaAtomError> {
    atom.with_sparse_24(&leaf(16), (E::U8, &leaf(8)))
}

/// Scales per block of four `k`: `SFA` is `4 x 2`, `SFB` is `2 x 4`.
fn scaled(atom: MmaAtom) -> Result<MmaAtom, MmaAtomError> {
    atom.with_block_scale(4, E::F32, (&leaf(8), &leaf(8)))
}

fn u(v: i64) -> ScalarExpr {
    ScalarExpr::ConstInt(v, E::U32)
}

/// `buffer[lane * vals + v]`.
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

/// The operands of a statement: element types and fragment lengths of A, B, C, then the extras.
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

/// The `Private` fragment arrays: one per input operand, then D.
fn fragment_allocs(atom: &MmaAtom, ops: &[(E, u32)]) -> Vec<AllocDecl> {
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
        atom.dtypes().2,
        Extent::Static(atom.values(2)),
    ));
    allocs
}

/// Copy every slot of every input buffer into its `Private` fragment.
fn copy_in(ops: &[(E, u32)], locals: &mut Vec<LocalDecl>) -> Vec<Stmt> {
    let mut out = Vec::new();
    for (i, (ty, n)) in ops.iter().enumerate() {
        for v in 0..*n {
            let l = locals.len() as u32;
            locals.push(LocalDecl::scalar(*ty));
            out.push(Stmt::Load {
                local: l,
                place: slot(i as u32, *n, v),
            });
            out.push(Stmt::Store {
                place: frag(i as u32, v),
                value: ScalarExpr::Local(l),
            });
        }
    }
    out
}

/// Write the D fragment (`Private` array `alloc`) to the buffer of the same number.
fn copy_out(atom: &MmaAtom, alloc: u32, locals: &mut Vec<LocalDecl>) -> Vec<Stmt> {
    let mut out = Vec::new();
    for v in 0..atom.values(2) {
        let l = locals.len() as u32;
        locals.push(LocalDecl::scalar(atom.dtypes().2));
        out.push(Stmt::Load {
            local: l,
            place: frag(alloc, v),
        });
        out.push(Stmt::Store {
            place: slot(alloc, atom.values(2), v),
            value: ScalarExpr::Local(l),
        });
    }
    out
}

/// Each lane copies every input fragment from its buffer, runs the MMA, writes D to the last buffer.
fn kernel(atom: &MmaAtom) -> KernelFn {
    let ops = operands(atom);
    let nin = ops.len() as u32;
    let acc = atom.dtypes().2;
    let allocs = fragment_allocs(atom, &ops);
    let mut locals = vec![LocalDecl::scalar(E::U32)];
    let mut body = vec![Stmt::Let {
        local: 0,
        init: ScalarExpr::Exec(ExecId::ThreadIdx(0)),
    }];
    body.extend(copy_in(&ops, &mut locals));
    body.push(Stmt::Mma {
        atom: atom.clone(),
        a: 0,
        b: 1,
        c: 2,
        d: nin,
        extra: (3..nin).collect(),
    });
    body.extend(copy_out(atom, nin, &mut locals));
    let mut params: Vec<Param> = ops
        .iter()
        .enumerate()
        .map(|(i, (ty, _))| Param::buffer(format!("in{i}"), *ty, BufferRole::Read, Some(1)))
        .collect();
    params.push(Param::buffer("out", acc, BufferRole::ReadWrite, Some(1)));
    KernelFn::new("mma_flags", acc, params, locals, Block(body))
        .with_launch(LaunchShape::new([LANES, 1, 1], [1, 1, 1]).with_subgroup(LANES, LANES))
        .with_allocs(allocs)
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

fn decode(ty: E, b: &TypedBuffer) -> Vec<f64> {
    b.bytes
        .chunks(4)
        .map(|c| match ty {
            E::I32 => f64::from(i32::from_le_bytes(c.try_into().unwrap())),
            _ => f64::from(f32::from_le_bytes(c.try_into().unwrap())),
        })
        .collect()
}

/// Per-lane fragments of a logical matrix under the identity layout: slot `(lane, v)` holds `lane + 4 v`.
fn per_lane(logical: &[f64]) -> Vec<f64> {
    let vals = logical.len() / LANES as usize;
    (0..LANES as usize)
        .flat_map(|lane| (0..vals).map(move |v| logical[lane + LANES as usize * v]))
        .collect()
}

/// Small deterministic values in `-3..=3`.
fn val(seed: u32, i: usize) -> f64 {
    f64::from(((seed * 7919 + i as u32 * 104_729) >> 3) % 7) - 3.0
}

/// The positions `(i0, i1)` of the two kept values of group `g` of row `r`: every one of the six pairs occurs.
fn pair(r: usize, g: usize) -> (usize, usize) {
    const PAIRS: [(usize, usize); 6] = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)];
    PAIRS[(r + 2 * g) % 6]
}

struct Case {
    a: Vec<f64>,
    meta: Vec<f64>,
    sfa: Vec<f64>,
    sfb: Vec<f64>,
}

/// The logical host data: dense or compressed `A`, the metadata and the scales (powers of two, so exact).
fn case(atom: &MmaAtom) -> Case {
    let a: Vec<f64> = (0..atom.dims(0).0 * atom.dims(0).1)
        .map(|i| val(1, i as usize))
        .collect();
    let meta = (0..(M * K / 4) as usize)
        .map(|id| {
            let (r, g) = (id % M as usize, id / M as usize);
            let (i0, i1) = pair(r, g);
            (i0 | i1 << 2) as f64
        })
        .collect();
    let pow2 = |seed, len| {
        (0..len)
            .map(|i| f64::from(1u32 << ((val(seed, i) as i32 + 3) % 4)))
            .collect()
    };
    Case {
        a,
        meta,
        sfa: pow2(5, (M * K / 4) as usize),
        sfb: pow2(6, (K / 4 * N) as usize),
    }
}

/// The dense `M x K` `A`: expanded from the compressed one through the metadata when the atom is sparse.
fn dense_a(atom: &MmaAtom, c: &Case) -> Vec<f64> {
    if !atom.is_sparse() {
        return c.a.clone();
    }
    let mut full = vec![0.0; (M * K) as usize];
    for g in 0..(K / 4) as usize {
        for r in 0..M as usize {
            let (i0, i1) = pair(r, g);
            full[r + M as usize * (4 * g + i0)] = c.a[r + M as usize * (2 * g)];
            full[r + M as usize * (4 * g + i1)] = c.a[r + M as usize * (2 * g + 1)];
        }
    }
    full
}

/// The plain triple loop, applying the scales of the atom's `k` block to the factors.
fn reference(atom: &MmaAtom, c: &Case, b: &[f64], cm: &[f64]) -> Vec<f64> {
    let a = dense_a(atom, c);
    let (m, n, k) = (M as usize, N as usize, K as usize);
    let mut d = vec![0.0; m * n];
    for col in 0..n {
        for row in 0..m {
            let mut s = cm[row + m * col];
            for i in 0..k {
                let (mut x, mut y) = (a[row + m * i], b[i + k * col]);
                if let Some(block) = atom.block() {
                    let blk = i / block as usize;
                    x *= c.sfa[row + m * blk];
                    y *= c.sfb[blk + (k / block as usize) * col];
                }
                s += x * y;
            }
            d[row + m * col] = s;
        }
    }
    d
}

fn run(atom: &MmaAtom, c: &Case) -> Result<Vec<f64>, KernelTrap> {
    let (ta, tb, tacc) = atom.dtypes();
    let b: Vec<f64> = (0..(K * N) as usize).map(|i| val(2, i)).collect();
    let cm: Vec<f64> = (0..(M * N) as usize).map(|i| val(3, i)).collect();
    let mut args = vec![
        KernelArg::Buffer(encode(ta, &per_lane(&c.a))),
        KernelArg::Buffer(encode(tb, &per_lane(&b))),
        KernelArg::Buffer(encode(tacc, &per_lane(&cm))),
    ];
    for (role, ty) in atom.extras() {
        let data = match role {
            unpopped::mma::ExtraRole::Metadata => &c.meta,
            unpopped::mma::ExtraRole::ScaleA => &c.sfa,
            _ => &c.sfb,
        };
        args.push(KernelArg::Buffer(encode(ty, &per_lane(data))));
    }
    args.push(KernelArg::Buffer(encode(tacc, &[0.0; (M * N) as usize])));
    run_launch(&kernel(atom), &mut args)?;
    let KernelArg::Buffer(out) = args.last().unwrap() else {
        panic!("the output is a buffer")
    };
    // D comes back per lane; put it back into column-major order and compare with the reference
    let per = decode(tacc, out);
    let vals = per.len() / LANES as usize;
    let mut d = vec![0.0; per.len()];
    for lane in 0..LANES as usize {
        for v in 0..vals {
            d[lane + LANES as usize * v] = per[lane * vals + v];
        }
    }
    let want = reference(atom, c, &b, &cm);
    assert_eq!(d, want, "{}", atom.name());
    Ok(d)
}

#[test]
fn a_dense_toy_atom_agrees_with_the_triple_loop() {
    let atom = toy(E::I8, E::I32);
    run(&atom, &case(&atom)).unwrap();
}

#[test]
fn a_2_4_sparse_atom_expands_a_through_the_metadata() {
    let atom = sparse(toy(E::I8, E::I32)).unwrap();
    assert!(atom.is_sparse());
    let c = case(&atom);
    let d = run(&atom, &c).unwrap();
    // control: the compressed A read as if it were dense would give another answer
    let dense = toy(E::I8, E::I32);
    let as_dense = Case {
        a: (0..(M * K) as usize).map(|i| val(1, i)).collect(),
        ..case(&dense)
    };
    assert_ne!(d, run(&dense, &as_dense).unwrap());
}

#[test]
fn a_block_scaled_atom_applies_the_scale_of_each_k_block() {
    let atom = scaled(toy(E::F32, E::F32)).unwrap();
    assert_eq!(atom.block(), Some(4));
    let c = case(&atom);
    let d = run(&atom, &c).unwrap();
    // control: with every scale 1 the answer differs, so the scales were applied
    let ones = Case {
        sfa: vec![1.0; c.sfa.len()],
        sfb: vec![1.0; c.sfb.len()],
        ..case(&atom)
    };
    assert_ne!(d, run(&atom, &ones).unwrap());
}

#[test]
fn a_sparse_and_scaled_atom_composes_both() {
    let atom = scaled(sparse(toy(E::F32, E::F32)).unwrap()).unwrap();
    assert_eq!(atom.extras().count(), 3);
    run(&atom, &case(&atom)).unwrap();
}

#[test]
fn a_metadata_entry_with_unordered_positions_traps() {
    let atom = sparse(toy(E::I8, E::I32)).unwrap();
    let mut c = case(&atom);
    let original = c.meta[0];
    c.meta[0] = f64::from(2 | 1 << 2); // i0 = 2, i1 = 1
    assert!(matches!(
        run(&atom, &c),
        Err(KernelTrap::Unsupported(what)) if what.contains("ascending")
    ));
    c.meta[0] = f64::from(1 | 1 << 2); // i0 == i1 is no pair either
    assert!(matches!(run(&atom, &c), Err(KernelTrap::Unsupported(_))));
    c.meta[0] = original; // control: ordered again
    run(&atom, &c).unwrap();
}

#[test]
fn flags_the_shape_cannot_carry_are_refused() {
    let int = toy(E::I8, E::I32);
    assert!(matches!(scaled(int), Err(MmaAtomError::BadFlag(_))));
    let twice = sparse(toy(E::I8, E::I32)).unwrap();
    assert_eq!(sparse(twice).unwrap_err(), MmaAtomError::FlagTwice);
    let f = toy(E::F32, E::F32);
    let short = f.clone().with_block_scale(4, E::F32, (&leaf(4), &leaf(8)));
    assert!(matches!(short, Err(MmaAtomError::NotACover(_))));
    let past = f.with_block_scale(4, E::F32, (&leaf(16), &leaf(8)));
    assert!(matches!(past, Err(MmaAtomError::NotACover(_))));
}

#[test]
fn a_broadcast_scale_must_agree_across_the_lanes_that_hold_it() {
    // SFB (2 x 4 = 8 elements) held twice: slot v and v + 1 of a lane are the same logical element
    let twice =
        Layout::from_modes(&[Layout::leaf(8, 1).unwrap(), Layout::leaf(2, 0).unwrap()]).unwrap();
    let atom = toy(E::F32, E::F32)
        .with_block_scale(4, E::F32, (&leaf(8), &twice))
        .unwrap();
    assert_eq!(atom.values(4), 4);
    let c = case(&atom);
    // the host feeds each lane's four slots: logical ids lane, lane + 4, lane, lane + 4
    let (ta, tb, tacc) = atom.dtypes();
    let b: Vec<f64> = (0..(K * N) as usize).map(|i| val(2, i)).collect();
    let cm: Vec<f64> = (0..(M * N) as usize).map(|i| val(3, i)).collect();
    let feed = |sfb: &[f64]| {
        vec![
            KernelArg::Buffer(encode(ta, &per_lane(&c.a))),
            KernelArg::Buffer(encode(tb, &per_lane(&b))),
            KernelArg::Buffer(encode(tacc, &per_lane(&cm))),
            KernelArg::Buffer(encode(E::F32, &per_lane(&c.sfa))),
            KernelArg::Buffer(encode(E::F32, sfb)),
            KernelArg::Buffer(encode(tacc, &[0.0; 16])),
        ]
    };
    // per-lane layout of the slots is lane-major: lane l holds ids (l + 4 v) % 8 for v in 0..4
    let by_lane: Vec<f64> = (0..4usize)
        .flat_map(|l| (0..4usize).map(move |v| (l + 4 * v) % 8))
        .map(|id| c.sfb[id])
        .collect();
    run_launch(&kernel(&atom), &mut feed(&by_lane)).unwrap();
    let mut bad = by_lane;
    bad[2] += 1.0; // slot 2 of lane 0 is element 0 again, but now differs from slot 0
    assert!(matches!(
        run_launch(&kernel(&atom), &mut feed(&bad)),
        Err(KernelTrap::Unsupported(what)) if what.contains("disagree")
    ));
}

#[test]
fn the_checker_wants_the_atoms_extra_operands() {
    let atom = sparse(toy(E::I8, E::I32)).unwrap();
    let mut missing = kernel(&atom);
    for st in &mut missing.body.0 {
        if let Stmt::Mma { extra, .. } = st {
            extra.clear();
        }
    }
    let c = case(&atom);
    let mut args = vec![KernelArg::Buffer(encode(E::I8, &per_lane(&c.a)))];
    args.truncate(1);
    match run_launch(&missing, &mut args) {
        Err(KernelTrap::Malformed(KernelError::MmaUnsupported { what })) => {
            assert!(what.contains("extra operands"), "{what}");
        }
        other => panic!("expected an MMA refusal, got {other:?}"),
    }
    // control: with the extra operands the kernel is well formed
    run(&atom, &c).unwrap();
}
