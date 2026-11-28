// SPDX-License-Identifier: MIT OR Apache-2.0
//! **The serial C of an `AllocDecl::dyn_layout` agrees with the launch interpreter** (`docs/design-dynamic-layout.md` section 5): the
//! offset is C arithmetic over the scalar parameters, and the launch-time proof of the oracle (`bind_dyn_layouts`) is emitted at the
//! top of the kernel function. Over a sweep of argument sets the C returns status 4 exactly where the oracle traps with
//! `KernelTrap::DynLayout`, and writes the oracle's bytes where it finishes. The seeded mutants zero one offset term of the emitted
//! access (each term in turn) or skip the injectivity run, and each must change the status or the bytes.

use unpopped::backend::Backend;
use unpopped::exec::{AllocDecl, ExecId, Extent, LaunchShape, MemorySpace, Scope};
use unpopped::ir::ScalarExpr;
use unpopped::kernel::{
    BarrierKind, Block, BufferRole, KernelFn, LocalDecl, Param, Place, Stmt, check_kernel,
};
use unpopped::oracle::kernel_eval::{KernelArg, KernelTrap};
use unpopped::oracle::kernel_launch::run_launch;
use unpopped_cpu_c::CpuC;
use unpopped_vocab::ElementKind::{I32, U32};
use unpopped_vocab::layout_algebra::{DynLayout, Swizzle};

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
/// `swz(2,2,3)` on 4-byte elements: a 128-byte period of 32 elements, so the swizzled kernels use multiples of 32.
fn swz() -> Swizzle {
    Swizzle::new(2, 2, 3).unwrap()
}
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
    kernel_x(layout, EXTENT, false)
}

/// The parameters (two buffers and the scalars `m`, `n`, `ld` at locals 2 to 4) and the locals of the kernel.
fn signature() -> (Vec<Param>, Vec<LocalDecl>) {
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
    (params, locals)
}

/// [`kernel`] over `n` elements, optionally under the byte swizzle (which needs `align_bytes` of the period).
fn kernel_x(layout: &str, n: u32, swizzled: bool) -> KernelFn {
    let rev = ScalarExpr::Sub(
        Box::new(ScalarExpr::ConstInt(i64::from(n) - 1, U32)),
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
    let (params, locals) = signature();
    let layout: DynLayout = layout.parse().expect("layout text");
    let mut alloc = AllocDecl::new("sh", MemorySpace::Shared, U32, Extent::Static(n));
    alloc = if swizzled {
        alloc
            .with_dyn_layout(layout.with_swizzle(swz()))
            .with_align_bytes(128)
    } else {
        alloc.with_dyn_layout(layout)
    };
    KernelFn::new("dynemit", I32, params, locals, body)
        .with_launch(LaunchShape::new([n, 1, 1], [1, 1, 1]))
        .with_allocs(vec![alloc])
}

fn args(elements: u32, (m, n, ld): (i128, i128, i128)) -> Vec<KernelArg> {
    let len = i64::from(elements);
    vec![
        KernelArg::Buffer(dense(U32, &[len], 5)),
        KernelArg::Buffer(zeroed(U32, &[len])),
        KernelArg::Int(m),
        KernelArg::Int(n),
        KernelArg::Int(ld),
    ]
}

/// The elements the kernel was built over (its block size).
fn n_of(k: &KernelFn) -> u32 {
    k.launch.expect("a launched kernel").block[0]
}

/// `None` when the oracle traps with `DynLayout`, else the bytes of `out`.
fn oracle(k: &KernelFn, a: (i128, i128, i128)) -> Option<Vec<u8>> {
    let mut v = args(n_of(k), a);
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
    CpuC.lower_kernel(k)
        .expect("the emitter spells this dyn_layout");
    let (mut ok, mut refused) = (0, 0);
    for (j, &a) in sets.iter().enumerate() {
        let Some(c) = run_c(&format!("{tag}_{j}"), k, &args(n_of(k), a)) else {
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
    let Some(plain) = run_c("dyn_mut_plain", &k, &args(EXTENT, a)) else {
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
            &args(EXTENT, a),
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
    let honest = run_c("dyn_mut_bad", &k, &args(EXTENT, bad)).expect("compiler present");
    assert_eq!(honest.status, STATUS);
    let skipped = run_c_mutated("dyn_mut_skip", &k, &args(EXTENT, bad), |src| {
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

#[test]
fn a_swizzled_dyn_layout_agrees_with_the_oracle_and_refuses_a_period_that_does_not_divide() {
    // extent 64 (two 32-element periods) and 96 (three): accepted binds run to the oracle's bytes; a non-injective bind, a size that is
    // not the extent, and extent 48 (not a multiple of the period: refused by the swizzle rule alone) return status 4
    let (mut ok, mut refused) = (0, 0);
    for (n, sets) in [
        (64u32, vec![(8, 8, 8), (4, 16, 16), (8, 8, 3), (8, 4, 8)]),
        (96, vec![(8, 12, 12)]),
        (48, vec![(8, 6, 6)]),
    ] {
        let k = kernel_x("(?2,?3):(?4,1)", n, true);
        let (o, r) = agree_all(&format!("dyn_swz{n}"), &k, &sets);
        ok += o;
        refused += r;
    }
    assert!(
        (ok >= 3 && refused >= 3) || ok + refused == 0,
        "{ok} {refused}"
    );
}

/// The k-th occurrence in the source of the XOR that applies the swizzle (0 is the launch check's, 1 the access's) becomes an OR,
/// which is not a bijection: two logical elements then share a physical offset.
fn xor_to_or(k: usize) -> impl FnOnce(String) -> String {
    move |src| {
        let at: Vec<usize> = src.match_indices(" ^= ").map(|(i, _)| i).collect();
        assert!(
            at.len() > k,
            "the mutant's anchor was not found ({} xor lines)",
            at.len()
        );
        let i = at[k];
        format!("{} |= {}", &src[..i], &src[i + 4..])
    }
}

#[test]
fn the_swizzle_xor_of_the_check_and_of_the_access_are_each_observable() {
    let k = kernel_x("(?2,?3):(?4,1)", 64, true);
    let a = (8, 8, 8);
    let want = oracle(&k, a).expect("an accepted bind");
    let Some(plain) = run_c("dyn_swz_plain", &k, &args(64, a)) else {
        return;
    };
    assert_eq!(
        (plain.status, &plain.outs[0]),
        (0, &want),
        "the unmutated C"
    );
    // the check's XOR as an OR: the proof sees a collision and refuses a bind the oracle accepts
    let c =
        run_c_mutated("dyn_swz_mut0", &k, &args(64, a), xor_to_or(0)).expect("compiler present");
    assert_eq!(c.status, STATUS, "the check's XOR is observable");
    // the access's XOR as an OR: two elements share an offset, so the bytes change
    let c =
        run_c_mutated("dyn_swz_mut1", &k, &args(64, a), xor_to_or(1)).expect("compiler present");
    assert!(
        c.status != 0 || c.outs[0] != want,
        "the access's XOR is observable"
    );
}
