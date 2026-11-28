// SPDX-License-Identifier: MIT OR Apache-2.0
//! **The serial C of an `AllocDecl::dyn_layout` agrees with the launch interpreter** (`docs/design-dynamic-layout.md` section 5): the
//! offset is C arithmetic over the scalar parameters, and the launch-time proof of the oracle (`bind_dyn_layouts`) is emitted at the
//! top of the kernel function. Over a sweep of argument sets the C returns status 4 exactly where the oracle traps with
//! `KernelTrap::DynLayout`, and writes the oracle's bytes where it finishes. The seeded mutants zero one offset term of the emitted
//! access (each term in turn) or skip the injectivity run, and each must change the status or the bytes.

use unpopped::exec::{AllocDecl, ExecId, Extent, LaunchShape, MemorySpace, Scope};
use unpopped::ir::ScalarExpr;
use unpopped::kernel::{
    BarrierKind, Block, BufferRole, KernelFn, LocalDecl, Param, Place, Stmt, check_kernel,
};
use unpopped::oracle::kernel_eval::{KernelArg, KernelTrap};
use unpopped::oracle::kernel_launch::run_launch;
use unpopped_vocab::ElementKind::{I32, U32};
use unpopped_vocab::layout_algebra::DynLayout;

#[path = "support/cc.rs"]
mod cc;
#[path = "support/kernel_data.rs"]
mod kernel_data;
#[allow(
    dead_code,
    reason = "the shared support module is included whole by #[path]; each test file uses a subset of it"
)]
#[path = "support/kernel_run.rs"]
mod kernel_run;
use kernel_data::{dense, zeroed};
use kernel_run::{run_c, run_c_mutated};

const EXTENT: u32 = 12;
/// The status the C returns for a launch whose `dyn_layout` does not bind to an accepted layout.
const STATUS: i32 = 4;

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

/// `sh[t] = in[t]`, a barrier, then `out[t] = sh[EXTENT - 1 - t]`: a reversal through the allocation's layout, so a layout whose
/// offset map is wrong (two indices on one offset) changes the bytes.
fn kernel(layout: &str) -> KernelFn {
    let rev = ScalarExpr::Sub(
        Box::new(ScalarExpr::ConstInt(i64::from(EXTENT) - 1, U32)),
        Box::new(tid()),
    );
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
            place: alloc_at(rev),
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
    let layout: DynLayout = layout.parse().expect("layout text");
    let alloc = AllocDecl::new("sh", MemorySpace::Shared, U32, Extent::Static(EXTENT))
        .with_dyn_layout(layout);
    KernelFn::new("dynemit", I32, params, locals, body)
        .with_launch(LaunchShape::new([EXTENT, 1, 1], [1, 1, 1]))
        .with_allocs(vec![alloc])
}

fn args((m, n, ld): (i128, i128, i128)) -> Vec<KernelArg> {
    let len = i64::from(EXTENT);
    vec![
        KernelArg::Buffer(dense(U32, &[len], 5)),
        KernelArg::Buffer(zeroed(U32, &[len])),
        KernelArg::Int(m),
        KernelArg::Int(n),
        KernelArg::Int(ld),
    ]
}

/// `None` when the oracle traps with `DynLayout`, else the bytes of `out`.
fn oracle(k: &KernelFn, a: (i128, i128, i128)) -> Option<Vec<u8>> {
    let mut v = args(a);
    match run_launch(k, &mut v) {
        Ok(()) => {
            let KernelArg::Buffer(b) = &v[1] else {
                unreachable!()
            };
            Some(b.bytes.clone())
        }
        Err(KernelTrap::DynLayout { .. }) => None,
        Err(e) => panic!("{a:?}: an unexpected trap {e:?}"),
    }
}

/// Run `k` as C and as the oracle on every argument set; the statuses and bytes must agree. Returns `(accepted, refused)`.
fn agree_all(tag: &str, k: &KernelFn, sets: &[(i128, i128, i128)]) -> (usize, usize) {
    assert_eq!(check_kernel(k), Ok(()), "well formed");
    let (mut ok, mut refused) = (0, 0);
    for (j, &a) in sets.iter().enumerate() {
        let Some(c) = run_c(&format!("{tag}_{j}"), k, &args(a)) else {
            return (0, 0);
        };
        match oracle(k, a) {
            Some(want) => {
                assert_eq!(
                    c.status, 0,
                    "{tag} {a:?}: the oracle finished, the C reported {}",
                    c.status
                );
                assert_eq!(c.outs[0], want, "{tag} {a:?}: bytes");
                ok += 1;
            }
            None => {
                assert_eq!(
                    c.status, STATUS,
                    "{tag} {a:?}: the oracle trapped DynLayout"
                );
                refused += 1;
            }
        }
    }
    (ok, refused)
}

fn row_major_sets() -> Vec<(i128, i128, i128)> {
    // accepted: (3,4,4) (4,3,3) (12,1,1); refused: non-injective, cosize past the extent, size not the extent, a zero shape leaf
    vec![
        (3, 4, 4),
        (4, 3, 3),
        (12, 1, 1),
        (3, 4, 3),
        (3, 4, 12),
        (3, 3, 4),
        (0, 12, 1),
    ]
}

fn nested_sets() -> Vec<(i128, i128, i128)> {
    // accepted when ld = 2m: (3,2,6) (2,3,4); refused: size not the extent, a collision, a cosize past the extent
    vec![(3, 2, 6), (2, 3, 4), (2, 2, 4), (3, 2, 4), (3, 2, 12)]
}

#[test]
fn a_row_major_dyn_layout_agrees_with_the_oracle_over_the_sweep() {
    let (ok, refused) = agree_all("dyn_row", &kernel("(?2,?3):(?4,1)"), &row_major_sets());
    // both arms are reached (a compiler-less host skips the whole sweep)
    assert!(
        (ok >= 2 && refused >= 2) || ok + refused == 0,
        "{ok} {refused}"
    );
}

#[test]
fn a_nested_three_term_dyn_layout_agrees_with_the_oracle_over_the_sweep() {
    // shape ((m,2),n), stride ((2,1),ld): offset = (i % m) * 2 + (i / m % 2) + (i / 2m) * ld, a bijection when ld = 2m
    let (ok, refused) = agree_all(
        "dyn_nested",
        &kernel("((?2,2),?3):((2,1),?4)"),
        &nested_sets(),
    );
    assert!(
        (ok >= 2 && refused >= 2) || ok + refused == 0,
        "{ok} {refused}"
    );
}

/// The offset lines of the access carry `lq<n>`, the launch proof's own carry `t<i>`: a mutant of the access only.
fn zero_access_term(k: usize) -> impl FnOnce(String) -> String {
    move |src| {
        let mut seen = 0;
        let mut out = Vec::new();
        for line in src.lines() {
            if line.contains("lq") && line.contains("+= ") {
                if seen == k {
                    out.push(line.replacen("+= ", "+= 0 * ", 1));
                } else {
                    out.push(line.to_string());
                }
                seen += 1;
            } else {
                out.push(line.to_string());
            }
        }
        assert!(
            seen > k,
            "the mutant's anchor was not found (term {k} of {seen})"
        );
        out.join("\n")
    }
}

#[test]
fn each_offset_term_of_the_access_and_the_injectivity_run_are_observable() {
    let k = kernel("(?2,?3):(?4,1)");
    let a = (3, 4, 4);
    let want = oracle(&k, a).expect("an accepted bind");
    let Some(plain) = run_c("dyn_mut_plain", &k, &args(a)) else {
        return;
    };
    assert_eq!(
        (plain.status, &plain.outs[0]),
        (0, &want),
        "the unmutated C"
    );
    // zeroing either term of the access collides two logical elements on one physical offset
    for term in 0..2 {
        let c = run_c_mutated(
            &format!("dyn_mut_term{term}"),
            &k,
            &args(a),
            zero_access_term(term),
        )
        .expect("compiler present");
        assert!(
            c.status != 0 || c.outs[0] != want,
            "term {term}: the mutant was not caught"
        );
    }
    // skipping the injectivity run: a non-injective bind (ld = 3 over (3,4)) must then run instead of returning status 4
    let bad = (3, 4, 3);
    assert!(oracle(&k, bad).is_none());
    let honest = run_c("dyn_mut_bad", &k, &args(bad)).expect("compiler present");
    assert_eq!(honest.status, STATUS);
    let skipped = run_c_mutated("dyn_mut_skip", &k, &args(bad), |src| {
        let anchor = "if (po0 >= ext0 || seen0[po0]) {";
        assert_eq!(src.matches(anchor).count(), 1, "anchor");
        src.replace(anchor, "if (po0 >= ext0 || 0) {")
    })
    .expect("compiler present");
    assert_ne!(
        skipped.status, STATUS,
        "skipping the injectivity run changed nothing"
    );
}
