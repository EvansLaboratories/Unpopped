// SPDX-License-Identifier: MIT OR Apache-2.0
//! **The serial C of a static `AllocDecl::layout` (and its swizzle) agrees with the launch interpreter** (ROADMAP T1, `docs/design-swizzled-allocation.md`;
//! the same pattern as `kernel_dyn_layout_emit_agrees.rs`): the physical offset is C arithmetic over constants, with the swizzle's XOR on the element
//! offset, and `check_kernel` has already proved the layout injective and inside the extent, so no launch check is emitted. A layout permutes the
//! storage and changes no result, so the mutants are the ones that break the bijection: each offset term of the access zeroed, and the swizzle's
//! XOR turned into an OR, must change the bytes. (The XOR's exact shift constant is not observable through logical accesses.)

use unpopped::backend::Backend;
use unpopped::exec::{AllocDecl, AllocLayout, ExecId, Extent, LaunchShape, MemorySpace, Scope};
use unpopped::ir::ScalarExpr;
use unpopped::kernel::{
    BarrierKind, Block, BufferRole, KernelFn, LocalDecl, Param, Place, Stmt, check_kernel,
};
use unpopped::oracle::kernel_eval::KernelArg;
use unpopped::oracle::kernel_launch::run_launch;
use unpopped_cpu_c::CpuC;
use unpopped_vocab::ElementKind::U32;
use unpopped_vocab::layout_algebra::Swizzle;

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

const N: u32 = 64;

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

/// `sh[t] = in[t]`, a barrier, then `out[t] = sh[63 - t]`, through the allocation's layout.
fn kernel(layout: &str, swizzled: bool) -> KernelFn {
    let rev = ScalarExpr::Sub(
        Box::new(ScalarExpr::ConstInt(i64::from(N) - 1, U32)),
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
    let mut l: AllocLayout = layout.parse().expect("layout text");
    let mut alloc = AllocDecl::new("sh", MemorySpace::Shared, U32, Extent::Static(N));
    if swizzled {
        l = l.with_swizzle(Swizzle::new(2, 2, 3).unwrap());
        alloc = alloc.with_align_bytes(128);
    }
    KernelFn::new(
        "layemit",
        U32,
        vec![
            Param::buffer("in", U32, BufferRole::Read, Some(1)),
            Param::buffer("out", U32, BufferRole::Write, Some(1)),
        ],
        vec![LocalDecl::scalar(U32), LocalDecl::scalar(U32)],
        body,
    )
    .with_launch(LaunchShape::new([N, 1, 1], [1, 1, 1]))
    .with_allocs(vec![alloc.with_layout(l)])
}

fn args() -> Vec<KernelArg> {
    let len = i64::from(N);
    vec![
        KernelArg::Buffer(dense(U32, &[len], 5)),
        KernelArg::Buffer(zeroed(U32, &[len])),
    ]
}

fn oracle(k: &KernelFn) -> Vec<u8> {
    let mut v = args();
    run_launch(k, &mut v).expect("the oracle finishes");
    let KernelArg::Buffer(b) = &v[1] else {
        unreachable!()
    };
    b.bytes.clone()
}

/// The serial C and the oracle write the same bytes for `k`.
fn agree(tag: &str, k: &KernelFn) {
    assert_eq!(check_kernel(k), Ok(()), "{tag}: well formed");
    CpuC.lower_kernel(k)
        .expect("the emitter spells this layout");
    let Some(c) = run_c(tag, k, &args()) else {
        return;
    };
    assert_eq!(c.status, 0, "{tag}: status");
    assert_eq!(c.outs[0], oracle(k), "{tag}: bytes");
}

const LAYOUTS: [&str; 2] = ["(8,8):(1,8)", "((4,2),8):((2,1),8)"];

#[test]
fn a_static_layout_and_a_swizzled_one_agree_with_the_oracle() {
    for (j, text) in LAYOUTS.iter().enumerate() {
        agree(&format!("lay_{j}"), &kernel(text, false));
        agree(&format!("lay_swz_{j}"), &kernel(text, true));
    }
}

/// The `k`-th offset line of the access (they carry `lq`) has its term zeroed.
fn zero_term(k: usize) -> impl FnOnce(String) -> String {
    move |src| {
        let mut seen = 0;
        let mut out = Vec::new();
        for line in src.lines() {
            if line.contains("lq") && line.contains("+= ") {
                out.push(if seen == k {
                    line.replacen("+= ", "+= 0 * ", 1)
                } else {
                    line.to_string()
                });
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

fn xor_to_or(src: String) -> String {
    let i = src.find(" ^= ").expect("the swizzle's XOR is emitted");
    format!("{} |= {}", &src[..i], &src[i + 4..])
}

#[test]
fn each_offset_term_and_the_swizzle_xor_are_observable() {
    // three terms: `(i % 4) * 2`, `(i / 4 % 2) * 1`, `(i / 8) * 8`
    let k = kernel(LAYOUTS[1], false);
    let want = oracle(&k);
    let Some(plain) = run_c("lay_mut_plain", &k, &args()) else {
        return;
    };
    assert_eq!(
        (plain.status, &plain.outs[0]),
        (0, &want),
        "the unmutated C"
    );
    for term in 0..3 {
        let c = run_c_mutated(&format!("lay_mut_t{term}"), &k, &args(), zero_term(term))
            .expect("compiler present");
        assert!(c.status != 0 || c.outs[0] != want, "term {term} not caught");
    }
    let k = kernel(LAYOUTS[0], true);
    let want = oracle(&k);
    let c = run_c_mutated("lay_mut_xor", &k, &args(), xor_to_or).expect("compiler present");
    assert!(c.status != 0 || c.outs[0] != want, "the XOR is not caught");
}
