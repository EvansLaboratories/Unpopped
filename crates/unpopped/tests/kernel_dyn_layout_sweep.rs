// SPDX-License-Identifier: MIT OR Apache-2.0
//! Conformance sweep of `AllocDecl::dyn_layout` (`docs/design-dynamic-layout.md`, after #536): over EVERY `(m, n, ld)` in a box,
//! the oracle's launch-time bind accepts exactly the argument sets that an independent, hand-written brute-force predicate accepts
//! (a valid shape, size equal to the extent, offsets distinct, cosize inside the extent), runs the accepted ones to the right
//! VALUE, and traps the rest with `KernelTrap::DynLayout`. A bind is invisible in a bijective layout's result, so acceptance is the
//! observable, which is why this is a sweep and not a spot check. Mutants are wrong predicates that must disagree somewhere.

use unpopped::exec::{AllocDecl, ExecId, Extent, LaunchShape, MemorySpace, Scope};
use unpopped::ir::ScalarExpr;
use unpopped::kernel::{
    BarrierKind, Block, BufferRole, KernelFn, LocalDecl, Param, Place, Stmt, check_kernel,
};
use unpopped::oracle::TypedBuffer;
use unpopped::oracle::kernel_eval::{KernelArg, KernelTrap};
use unpopped::oracle::kernel_launch::run_launch;
use unpopped_vocab::ElementKind::{I32, U32};
use unpopped_vocab::layout_algebra::DynLayout;

const EXTENT: i128 = 12;

fn tid() -> ScalarExpr {
    ScalarExpr::Exec(ExecId::ThreadIdx(0))
}

fn at(buffer: u32, i: ScalarExpr) -> Place {
    Place::Logical {
        buffer,
        coords: vec![i],
    }
}

fn alloc_at(i: ScalarExpr) -> Place {
    Place::Alloc { alloc: 0, index: i }
}

/// Thread `t` stores `in[t]` to `sh[t]`, then (after a barrier) `out[t] = sh[t]`: the identity, through the allocation's layout.
fn kernel(text: &str) -> KernelFn {
    let layout: DynLayout = text.parse().expect("layout text");
    let body = Block(vec![
        Stmt::Load {
            local: 0,
            place: at(0, tid()),
        },
        Stmt::Store {
            place: alloc_at(tid()),
            value: ScalarExpr::Local(0),
        },
        Stmt::Barrier {
            scope: Scope::Block,
            kind: BarrierKind::Full,
        },
        Stmt::Load {
            local: 1,
            place: alloc_at(tid()),
        },
        Stmt::Store {
            place: at(1, tid()),
            value: ScalarExpr::Local(1),
        },
    ]);
    let mut locals = vec![LocalDecl::scalar(U32), LocalDecl::scalar(U32)];
    locals.extend((0..3).map(|_| LocalDecl::scalar(I32)));
    let mut params = vec![
        Param::buffer("in", U32, BufferRole::Read, Some(1)),
        Param::buffer("out", U32, BufferRole::Write, Some(1)),
    ];
    params.extend(
        ["m", "n", "ld"]
            .iter()
            .zip(2u32..)
            .map(|(nm, loc)| Param::scalar(*nm, I32, loc)),
    );
    let alloc = AllocDecl::new(
        "sh",
        MemorySpace::Shared,
        U32,
        Extent::Static(EXTENT as u32),
    )
    .with_dyn_layout(layout);
    KernelFn::new("dynsweep", I32, params, locals, body)
        .with_launch(LaunchShape::new([EXTENT as u32, 1, 1], [1, 1, 1]))
        .with_allocs(vec![alloc])
}

fn launch(k: &KernelFn, (m, n, ld): (i128, i128, i128)) -> Result<Vec<u32>, KernelTrap> {
    let data: Vec<u8> = (0..EXTENT as u32)
        .flat_map(|i| (i * 5 + 1).to_le_bytes())
        .collect();
    let buf = |bytes| KernelArg::Buffer(TypedBuffer::new(U32, vec![EXTENT as i64], vec![1], bytes));
    let mut args = vec![
        buf(data),
        buf(vec![0u8; 4 * EXTENT as usize]),
        KernelArg::Int(m),
        KernelArg::Int(n),
        KernelArg::Int(ld),
    ];
    run_launch(k, &mut args)?;
    let KernelArg::Buffer(out) = &args[1] else {
        unreachable!()
    };
    Ok(out
        .bytes
        .chunks(4)
        .map(|c| u32::from_le_bytes(c.try_into().unwrap()))
        .collect())
}

/// Hand-written, no use of any layout code: row-major `(m,n):(ld,1)` offsets by double loop.
/// `skip` selects a mutant: 1 ignores injectivity, 2 ignores the extent bound, 3 ignores the size rule.
fn accepts((m, n, ld): (i128, i128, i128), skip: u8) -> bool {
    if m < 1 || n < 1 || ld < 0 {
        return false;
    }
    if skip != 3 && m * n != EXTENT {
        return false;
    }
    let mut seen = std::collections::BTreeSet::new();
    for r in 0..m {
        for c in 0..n {
            let off = r * ld + c;
            if skip != 2 && off >= EXTENT {
                return false;
            }
            if !seen.insert(off) && skip != 1 {
                return false;
            }
        }
    }
    true
}

fn sweep() -> impl Iterator<Item = (i128, i128, i128)> {
    (0..=13i128)
        .flat_map(|m| (0..=13i128).flat_map(move |n| (0..=13i128).map(move |ld| (m, n, ld))))
}

#[test]
fn the_launch_time_bind_accepts_exactly_the_argument_sets_a_brute_force_predicate_accepts() {
    let k = kernel("(?2,?3):(?4,1)");
    assert_eq!(check_kernel(&k), Ok(()), "well formed");
    let want: Vec<u32> = (0..EXTENT as u32).map(|i| i * 5 + 1).collect();
    let (mut ok, mut bad) = (0, 0);
    for args in sweep() {
        match (accepts(args, 0), launch(&k, args)) {
            (true, Ok(got)) => {
                assert_eq!(got, want, "{args:?}");
                ok += 1;
            }
            (false, Err(KernelTrap::DynLayout { .. })) => bad += 1,
            (expect, got) => panic!("{args:?}: predicate {expect}, oracle {got:?}"),
        }
    }
    assert_eq!(ok + bad, 14 * 14 * 14, "the whole box was classified");
    assert!(
        ok >= 15 && bad > ok,
        "both arms are populated: {ok} accepted, {bad} refused"
    );
}

#[test]
fn each_wrong_predicate_disagrees_with_the_oracle_somewhere() {
    // The mutants: ignoring injectivity, the extent bound or the size rule. Each must accept an argument set the oracle refuses,
    // else the sweep above could not tell the rule exists.
    let k = kernel("(?2,?3):(?4,1)");
    for skip in 1..=3u8 {
        let hit = sweep().find(|&a| accepts(a, skip) && launch(&k, a).is_err());
        assert!(
            hit.is_some(),
            "mutant {skip} agrees with the oracle everywhere"
        );
    }
}

#[test]
fn a_column_major_bind_swaps_which_leaf_is_the_leading_dimension() {
    // `(?2,?3):(1,?2)` is bijective for every m*n == 12 whatever `ld` is: the same box, a different predicate.
    let k = kernel("(?2,?3):(1,?2)");
    for args @ (m, n, _) in sweep() {
        let want_ok = m >= 1 && n >= 1 && m * n == EXTENT;
        assert_eq!(launch(&k, args).is_ok(), want_ok, "{args:?}");
    }
}
