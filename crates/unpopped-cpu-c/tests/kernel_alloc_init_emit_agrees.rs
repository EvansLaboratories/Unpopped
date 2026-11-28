// SPDX-License-Identifier: MIT OR Apache-2.0
//! **The serial C of an initial table (`AllocDecl::init`, ROADMAP U40 and U66) agrees with the launch interpreter**: a `Constant` or a
//! `Private` table of integers or of float bit patterns is declared as an unsigned array of the element's width and copied into the block's
//! (or every thread's) storage, so a negative zero and an integer's sign are the checker's. An initial table is declined by name where it
//! would be dropped (a callee, a Device barrier) and a `Constant` allocation with no values stays declined. Mutants: the copy dropped (a zeroed
//! table) and a `Private` table copied into the first thread's slice only both change the bytes.

use unpopped::backend::Backend;
use unpopped::exec::{AllocDecl, ExecId, Extent, LaunchShape, MemorySpace, Scope};
use unpopped::ir::ScalarExpr;
use unpopped::kernel::{Block, BufferRole, KernelFn, LocalDecl, Param, Place, Stmt, check_kernel};
use unpopped::oracle::kernel_eval::KernelArg;
use unpopped::oracle::kernel_launch::run_launch;
use unpopped_cpu_c::CpuC;
use unpopped_vocab::ElementKind::{self, F32, U32};

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
use kernel_data::zeroed;
use kernel_run::{run_c, run_c_mutated};

const N: u32 = 8;

fn tid() -> ScalarExpr {
    ScalarExpr::Exec(ExecId::ThreadIdx(0))
}

fn gid() -> ScalarExpr {
    ScalarExpr::Add(
        Box::new(ScalarExpr::Mul(
            Box::new(ScalarExpr::Exec(ExecId::BlockIdx(0))),
            Box::new(ScalarExpr::ConstInt(i64::from(N), U32)),
        )),
        Box::new(tid()),
    )
}

fn at(buffer: u32, i: ScalarExpr) -> Place {
    Place::Logical {
        buffer,
        coords: vec![i],
    }
}

/// `out[g] = table[idx[g]]`, two blocks of `N` threads.
fn kernel(table: AllocDecl) -> KernelFn {
    let elem = table.elem;
    let body = Block(vec![
        Stmt::Load {
            local: 0,
            place: at(0, gid()),
        },
        Stmt::Load {
            local: 1,
            place: Place::Alloc {
                alloc: 0,
                index: ScalarExpr::Local(0),
            },
        },
        Stmt::Store {
            place: at(1, gid()),
            value: ScalarExpr::Local(1),
        },
    ]);
    KernelFn::new(
        "tabemit",
        elem,
        vec![
            Param::buffer("idx", U32, BufferRole::Read, Some(1)),
            Param::buffer("out", elem, BufferRole::Write, Some(1)),
        ],
        vec![LocalDecl::scalar(U32), LocalDecl::scalar(elem)],
        body,
    )
    .with_launch(LaunchShape::new([N, 1, 1], [2, 1, 1]))
    .with_allocs(vec![table])
}

fn table(space: MemorySpace, elem: ElementKind, values: Vec<i128>) -> AllocDecl {
    AllocDecl::new("t", space, elem, Extent::Static(N)).with_init(values)
}

fn args(elem: ElementKind) -> Vec<KernelArg> {
    let idx: Vec<u32> = [7, 6, 5, 4, 3, 2, 1, 0, 0, 1, 2, 3, 4, 5, 6, 7].to_vec();
    let bytes: Vec<u8> = idx.iter().flat_map(|v| v.to_le_bytes()).collect();
    vec![
        KernelArg::Buffer(unpopped::oracle::TypedBuffer::new(
            U32,
            vec![16],
            vec![1],
            bytes,
        )),
        KernelArg::Buffer(zeroed(elem, &[16])),
    ]
}

fn oracle(k: &KernelFn) -> Vec<u8> {
    let mut v = args(k.allocs[0].elem);
    run_launch(k, &mut v).expect("the oracle finishes");
    let KernelArg::Buffer(b) = &v[1] else {
        unreachable!()
    };
    b.bytes.clone()
}

fn agree(tag: &str, k: &KernelFn) {
    assert_eq!(check_kernel(k), Ok(()), "{tag}: well formed");
    CpuC.lower_kernel(k).expect("the emitter spells this table");
    let Some(c) = run_c(tag, k, &args(k.allocs[0].elem)) else {
        return;
    };
    assert_eq!(c.status, 0, "{tag}: status");
    assert_eq!(c.outs[0], oracle(k), "{tag}: bytes");
}

fn ints() -> Vec<i128> {
    vec![10, 20, 30, 40, 50, 60, 70, 4_000_000_000]
}

fn floats() -> Vec<i128> {
    [1.5f32, -0.0, 2.5, f32::INFINITY, 0.25, -7.0, 1e-3, 3.0]
        .iter()
        .map(|f| i128::from(f.to_bits()))
        .collect()
}

#[test]
fn constant_and_private_tables_of_integers_and_floats_agree_with_the_oracle() {
    for (j, (space, elem, vals)) in [
        (MemorySpace::Constant, U32, ints()),
        (MemorySpace::Private, U32, ints()),
        (MemorySpace::Constant, F32, floats()),
        (MemorySpace::Private, F32, floats()),
    ]
    .into_iter()
    .enumerate()
    {
        agree(&format!("tab_{j}"), &kernel(table(space, elem, vals)));
    }
}

#[test]
fn the_copy_of_a_table_and_the_copy_into_every_threads_slice_are_observable() {
    let k = kernel(table(MemorySpace::Private, U32, ints()));
    let want = oracle(&k);
    let Some(plain) = run_c("tab_mut_plain", &k, &args(U32)) else {
        return;
    };
    assert_eq!((plain.status, &plain.outs[0]), (0, &want));
    // the copy dropped: a zeroed table
    let c = run_c_mutated("tab_mut_drop", &k, &args(U32), |src| {
        assert_eq!(src.matches("memcpy(&sh0").count(), 1, "anchor");
        src.replace("memcpy(&sh0", "(void)(&sh0")
    })
    .expect("compiler present");
    assert!(
        c.status != 0 || c.outs[0] != want,
        "a zeroed table not caught"
    );
    // only the first thread's slice initialised
    let c = run_c_mutated("tab_mut_first", &k, &args(U32), |src| {
        let a = "for (int z = 0; z < 8; ++z) memcpy";
        assert_eq!(src.matches(a).count(), 1, "anchor");
        src.replace(a, "for (int z = 0; z < 1; ++z) memcpy")
    })
    .expect("compiler present");
    assert!(
        c.status != 0 || c.outs[0] != want,
        "a first-slice-only table not caught"
    );
}

#[test]
fn a_constant_without_values_and_a_table_where_it_would_be_dropped_decline_by_name() {
    let bare = AllocDecl::new("t", MemorySpace::Constant, U32, Extent::Static(N));
    let err = CpuC.lower_kernel(&kernel(bare)).expect_err("no values");
    assert!(format!("{err:?}").contains("Constant"), "{err:?}");
    // a Device barrier makes the body multi-phase: declined by name, not a zeroed table
    let mut phased = kernel(table(MemorySpace::Constant, U32, ints()));
    phased.body.0.insert(1, Stmt::barrier(Scope::Device));
    assert_eq!(check_kernel(&phased), Ok(()), "well formed");
    let err = CpuC.lower_kernel(&phased).expect_err("declined");
    assert!(format!("{err:?}").contains("initial table"), "{err:?}");
}
