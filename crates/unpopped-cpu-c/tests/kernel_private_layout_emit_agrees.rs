// SPDX-License-Identifier: MIT OR Apache-2.0
//! **A layout, static or dynamic, on a `Private` allocation** (`docs/design-dynamic-layout.md` section 9): a `Private` array is one per
//! thread, so the layout maps the thread's own index to its own physical element. Each thread fills its eight-element array from `in`,
//! (across a barrier) then writes it back to `out` reversed. The oracle's bytes are checked against the Rust reference, the serial C against the oracle
//! (`sh<i>[t * extent + offset]`), the dynamic layout's launch-time refusal returns status 4 where the oracle traps `DynLayout`, and the
//! mutants that break the bijection (a zeroed offset term) change the bytes.

use unpopped::backend::Backend;
use unpopped::exec::{AllocDecl, AllocLayout, ExecId, Extent, LaunchShape, MemorySpace, Scope};
use unpopped::ir::ScalarExpr;
use unpopped::kernel::{
    BarrierKind, Block, BufferRole, KernelFn, LocalDecl, Param, Place, Stmt, Unroll, check_kernel,
};
use unpopped::oracle::kernel_eval::{KernelArg, KernelTrap};
use unpopped::oracle::kernel_launch::run_launch;
use unpopped_cpu_c::CpuC;
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

const THREADS: u32 = 8;
const PER: u32 = 8;

fn c(v: i64) -> ScalarExpr {
    ScalarExpr::ConstInt(v, U32)
}

fn tid() -> ScalarExpr {
    ScalarExpr::Exec(ExecId::ThreadIdx(0))
}

fn add(a: ScalarExpr, b: ScalarExpr) -> ScalarExpr {
    ScalarExpr::Add(Box::new(a), Box::new(b))
}

fn at(buffer: u32, i: ScalarExpr) -> Place {
    Place::Logical {
        buffer,
        coords: vec![i],
    }
}

fn counted(local: u32, body: Vec<Stmt>) -> Stmt {
    Stmt::For {
        local,
        start: c(0),
        end: c(i64::from(PER)),
        step: c(1),
        unroll: Unroll::default(),
        body: Block(body),
    }
}

/// Thread `t`: `p[i] = in[t*8 + i]` for `i` in `0..8`, a barrier, then `out[t*8 + j] = p[7 - j]`.
fn kernel(alloc: AllocDecl) -> KernelFn {
    let base = ScalarExpr::Mul(Box::new(tid()), Box::new(c(i64::from(PER))));
    let fill = counted(
        2,
        vec![
            Stmt::Load {
                local: 0,
                place: at(0, add(base.clone(), ScalarExpr::Local(2))),
            },
            Stmt::Store {
                place: Place::Alloc {
                    alloc: 0,
                    index: ScalarExpr::Local(2),
                },
                value: ScalarExpr::Local(0),
            },
        ],
    );
    // the barrier ends the thread loop, so a Private array that is not sliced per thread is overwritten by the next thread
    let barrier = Stmt::Barrier {
        scope: Scope::Block,
        kind: BarrierKind::Full,
    };
    let rev = ScalarExpr::Sub(
        Box::new(c(i64::from(PER) - 1)),
        Box::new(ScalarExpr::Local(3)),
    );
    let drain = counted(
        3,
        vec![
            Stmt::Load {
                local: 1,
                place: Place::Alloc {
                    alloc: 0,
                    index: rev,
                },
            },
            Stmt::Store {
                place: at(1, add(base, ScalarExpr::Local(3))),
                value: ScalarExpr::Local(1),
            },
        ],
    );
    let mut params = vec![
        Param::buffer("in", U32, BufferRole::Read, Some(1)),
        Param::buffer("out", U32, BufferRole::Write, Some(1)),
    ];
    params.extend(
        ["m", "n", "ld"]
            .iter()
            .zip(4u32..)
            .map(|(nm, loc)| Param::scalar(*nm, I32, loc)),
    );
    let mut locals = vec![LocalDecl::scalar(U32); 4];
    locals.extend((0..3).map(|_| LocalDecl::scalar(I32)));
    KernelFn::new(
        "privlay",
        I32,
        params,
        locals,
        Block(vec![fill, barrier, drain]),
    )
    .with_launch(LaunchShape::new([THREADS, 1, 1], [1, 1, 1]))
    .with_allocs(vec![alloc])
}

fn private() -> AllocDecl {
    AllocDecl::new("p", MemorySpace::Private, U32, Extent::Static(PER))
}

fn args(a: (i128, i128, i128)) -> Vec<KernelArg> {
    let len = i64::from(THREADS * PER);
    vec![
        KernelArg::Buffer(dense(U32, &[len], 5)),
        KernelArg::Buffer(zeroed(U32, &[len])),
        KernelArg::Int(a.0),
        KernelArg::Int(a.1),
        KernelArg::Int(a.2),
    ]
}

/// The bytes of `out` the oracle writes, or `None` for a `DynLayout` trap.
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

/// Each thread's slice of `in` reversed (the layout permutes the storage and changes no result).
fn reference(k: &KernelFn) -> Vec<u8> {
    let KernelArg::Buffer(inp) = &args((0, 0, 0))[0] else {
        unreachable!()
    };
    let words: Vec<u32> = inp
        .bytes
        .chunks(4)
        .map(|w| u32::from_le_bytes(w.try_into().unwrap()))
        .collect();
    assert_eq!(check_kernel(k), Ok(()), "well formed");
    words
        .chunks(PER as usize)
        .flat_map(|t| t.iter().rev().flat_map(|w| w.to_le_bytes()))
        .collect()
}

fn dyn_alloc(text: &str) -> AllocDecl {
    private().with_dyn_layout(text.parse::<DynLayout>().expect("layout text"))
}

#[test]
fn a_private_array_under_a_static_layout_runs_and_the_c_agrees() {
    for (j, text) in ["(2,4):(1,2)", "(2,4):(4,1)", "((2,2),2):((2,1),4)"]
        .iter()
        .enumerate()
    {
        let k = kernel(private().with_layout(text.parse::<AllocLayout>().unwrap()));
        let want = reference(&k);
        assert_eq!(
            oracle(&k, (0, 0, 0)).as_deref(),
            Some(&want[..]),
            "{text}: oracle"
        );
        CpuC.lower_kernel(&k)
            .expect("the emitter spells a Private layout");
        let Some(r) = run_c(&format!("priv_static_{j}"), &k, &args((0, 0, 0))) else {
            return;
        };
        assert_eq!((r.status, &r.outs[0]), (0, &want), "{text}: C");
    }
}

#[test]
fn a_private_dyn_layout_binds_at_launch_and_the_c_returns_the_oracles_status() {
    let k = kernel(dyn_alloc("(?4,?5):(?6,1)"));
    let want = reference(&k);
    // (2,4,4) is the identity, (4,2,2) another shape: accepted; ld = 3 collides, (2,3,3) is not the extent: refused by both
    for (j, a) in [(2, 4, 4), (4, 2, 2), (2, 4, 3), (2, 3, 3)]
        .into_iter()
        .enumerate()
    {
        let got = oracle(&k, a);
        let Some(r) = run_c(&format!("priv_dyn_{j}"), &k, &args(a)) else {
            return;
        };
        match got {
            Some(bytes) => {
                assert_eq!(bytes, want, "{a:?}: oracle");
                assert_eq!((r.status, &r.outs[0]), (0, &want), "{a:?}: C");
            }
            None => assert_eq!(r.status, 4, "{a:?}: the oracle trapped DynLayout"),
        }
    }
}

#[test]
fn a_zeroed_offset_term_of_a_private_access_changes_the_bytes() {
    let k = kernel(private().with_layout("((2,2),2):((2,1),4)".parse::<AllocLayout>().unwrap()));
    let want = reference(&k);
    for term in 0..3 {
        let r = run_c_mutated(
            &format!("priv_mut_{term}"),
            &k,
            &args((0, 0, 0)),
            move |src| {
                let mut seen = 0;
                let lines: Vec<String> = src
                    .lines()
                    .map(|l| {
                        if l.contains("lq") && l.contains("+= ") {
                            seen += 1;
                            if seen == term + 1 {
                                return l.replacen("+= ", "+= 0 * ", 1);
                            }
                        }
                        l.to_string()
                    })
                    .collect();
                assert!(seen > term, "the mutant's anchor was not found");
                lines.join("\n")
            },
        );
        let Some(r) = r else { return };
        assert!(r.status != 0 || r.outs[0] != want, "term {term} not caught");
    }
}
