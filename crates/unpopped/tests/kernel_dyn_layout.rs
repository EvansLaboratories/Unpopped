// SPDX-License-Identifier: MIT OR Apache-2.0
//! A layout whose leaves are scalar parameters on an allocation (`AllocDecl::dyn_layout`, `docs/design-dynamic-layout.md`): the
//! checker's static rules with an accepted control beside each refusal, the oracle's launch-time bind (the same kernel under
//! several argument sets, and a trap by name for a bound layout the enumerating checker refuses), the serde default, and the
//! red-first flip of conformance's "a dynamic mode is declined by name" finding.

use unpopped::exec::{AllocDecl, AllocLayout, ExecId, Extent, LaunchShape, MemorySpace, Scope};
use unpopped::ir::ScalarExpr;
use unpopped::kernel::{
    BarrierKind, Block, BufferRole, KernelError, KernelFn, LocalDecl, Param, Place, Stmt,
    check_kernel,
};
use unpopped::oracle::TypedBuffer;
use unpopped::oracle::kernel_eval::{KernelArg, KernelTrap};
use unpopped::oracle::kernel_launch::run_launch;
use unpopped_vocab::ElementKind::{I32, U32};
use unpopped_vocab::layout_algebra::DynLayout;

fn tid() -> ScalarExpr {
    ScalarExpr::Exec(ExecId::ThreadIdx(0))
}

fn at(buffer: u32, i: ScalarExpr) -> Place {
    Place::Logical {
        buffer,
        coords: vec![i],
    }
}

fn dynl(text: &str) -> DynLayout {
    text.parse().unwrap_or_else(|e| panic!("{text}: {e:?}"))
}

/// `out[t] = sh[63 - t]` after every thread stored `in[t]` to `sh[t]`: 64 threads, one shared array of 64, three integer scalar
/// parameters (locals 2, 3, 4 = `m`, `n`, `ld`) that a `dyn_layout` may name.
fn body() -> Block {
    let rev = ScalarExpr::Sub(Box::new(ScalarExpr::ConstInt(63, U32)), Box::new(tid()));
    Block(vec![
        Stmt::Load {
            local: 0,
            place: at(0, tid()),
        },
        Stmt::Store {
            place: Place::Alloc {
                alloc: 0,
                index: tid(),
            },
            value: ScalarExpr::Local(0),
        },
        Stmt::Barrier {
            scope: Scope::Block,
            kind: BarrierKind::Full,
        },
        Stmt::Load {
            local: 1,
            place: Place::Alloc {
                alloc: 0,
                index: rev,
            },
        },
        Stmt::Store {
            place: at(1, tid()),
            value: ScalarExpr::Local(1),
        },
    ])
}

fn kernel(alloc: AllocDecl) -> KernelFn {
    KernelFn::new(
        "dynlay",
        I32,
        vec![
            Param::buffer("in", U32, BufferRole::Read, Some(1)),
            Param::buffer("out", U32, BufferRole::Write, Some(1)),
            Param::scalar("m", I32, 2),
            Param::scalar("n", I32, 3),
            Param::scalar("ld", I32, 4),
        ],
        vec![
            LocalDecl::scalar(U32),
            LocalDecl::scalar(U32),
            LocalDecl::scalar(I32),
            LocalDecl::scalar(I32),
            LocalDecl::scalar(I32),
        ],
        body(),
    )
    .with_launch(LaunchShape::new([64, 1, 1], [1, 1, 1]))
    .with_allocs(vec![alloc])
}

fn shared(extent: Extent) -> AllocDecl {
    AllocDecl::new("sh", MemorySpace::Shared, U32, extent)
}

fn launch(k: &KernelFn, (m, n, ld): (i128, i128, i128)) -> Result<Vec<u32>, KernelTrap> {
    assert_eq!(check_kernel(k), Ok(()), "well formed");
    let data: Vec<u8> = (0..64u32).flat_map(|i| (i * 7 + 3).to_le_bytes()).collect();
    let buf = |bytes| KernelArg::Buffer(TypedBuffer::new(U32, vec![64], vec![1], bytes));
    let mut args = vec![
        buf(data),
        buf(vec![0u8; 256]),
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

fn want() -> Vec<u32> {
    (0..64u32).map(|t| (63 - t) * 7 + 3).collect()
}

fn refused(k: &KernelFn) -> &'static str {
    match check_kernel(k) {
        Err(KernelError::AllocLayout { why, .. }) => why,
        other => panic!("expected an AllocLayout refusal, got {other:?}"),
    }
}

fn why_of(r: Result<Vec<u32>, KernelTrap>) -> &'static str {
    match r {
        Err(KernelTrap::DynLayout { why, .. }) => why,
        other => panic!("expected a DynLayout trap, got {other:?}"),
    }
}

#[test]
fn one_kernel_runs_under_every_argument_set_that_binds_to_an_accepted_layout() {
    // row-major `(m,n):(ld,1)`: (8,8,8) identity, (4,16,16) another shape, and a padded leading dimension is refused below
    let k = kernel(shared(Extent::Static(64)).with_dyn_layout(dynl("(?2,?3):(?4,1)")));
    for args in [(8, 8, 8), (4, 16, 16), (16, 4, 4), (2, 32, 32)] {
        assert_eq!(launch(&k, args), Ok(want()), "{args:?}");
    }
    // column-major over the same parameters: `(m,n):(1,m)`
    let k = kernel(shared(Extent::Static(64)).with_dyn_layout(dynl("(?2,?3):(1,?2)")));
    for args in [(8, 8, 0), (4, 16, 0), (16, 4, 0)] {
        assert_eq!(launch(&k, args), Ok(want()), "{args:?}");
    }
}

#[test]
fn a_bound_layout_the_enumerating_checker_refuses_traps_by_name_before_the_first_statement() {
    let k = kernel(shared(Extent::Static(64)).with_dyn_layout(dynl("(?2,?3):(?4,1)")));
    // lda = 3 over (8,8): size 64 but two indices share an offset (the finding's non-injective bind)
    assert!(why_of(launch(&k, (8, 8, 3))).contains("not injective"));
    // lda = 16 over (8,8): injective but the cosize leaves the 64-element extent
    assert!(why_of(launch(&k, (8, 8, 16))).contains("exceeds the allocation's extent"));
    // a shape whose size is not the extent
    assert!(why_of(launch(&k, (8, 4, 8))).contains("size must equal"));
    // a bound leaf that is not a valid shape
    assert!(why_of(launch(&k, (0, 8, 8))).contains("valid shape"));
    // control: the accepted bind of the same kernel runs
    assert_eq!(launch(&k, (8, 8, 8)), Ok(want()));
}

#[test]
fn a_dynamic_extent_is_the_size_the_bound_layout_must_have() {
    // the extent is the value of `m` (local 2): the bound layout `(?2):(1)` is its size
    let mut k = kernel(shared(Extent::Param(2)).with_dyn_layout(dynl("?2:1")));
    assert_eq!(check_kernel(&k), Ok(()));
    assert_eq!(launch(&k, (64, 0, 0)), Ok(want()));
    k.allocs[0] = shared(Extent::Param(2)).with_dyn_layout(dynl("(?2,2):(1,?2)"));
    assert!(why_of(launch(&k, (64, 0, 0))).contains("size must equal"));
}

#[test]
fn check_kernel_refuses_each_static_violation_and_accepts_its_control() {
    let ok = shared(Extent::Static(64)).with_dyn_layout(dynl("(?2,?3):(?4,1)"));
    assert_eq!(check_kernel(&kernel(ok.clone())), Ok(()));
    // a parameter leaf that names no parameter of the kernel (local 9)
    let k = kernel(shared(Extent::Static(64)).with_dyn_layout(dynl("(?9,?3):(?4,1)")));
    assert!(refused(&k).contains("no integer scalar parameter"));
    // a parameter leaf that names a buffer-bound local, not a scalar parameter (local 0 is a temporary)
    let k = kernel(shared(Extent::Static(64)).with_dyn_layout(dynl("(?0,?3):(?4,1)")));
    assert!(refused(&k).contains("no integer scalar parameter"));
    // both a layout and a dyn_layout
    let both = ok
        .clone()
        .with_layout("(8,8):(8,1)".parse::<AllocLayout>().unwrap());
    assert!(refused(&kernel(both)).contains("not both"));
    // not on a Shared allocation
    let mut private = ok;
    private.space = MemorySpace::Private;
    assert!(refused(&kernel(private)).contains("only on a Shared"));
}

#[test]
fn a_dyn_layout_is_serialized_as_text_and_absent_when_there_is_none() {
    let plain = kernel(shared(Extent::Static(64)));
    let text = serde_json::to_string(&plain).unwrap();
    assert!(
        !text.contains("dyn_layout"),
        "a kernel without one keeps its text (and revision hash)"
    );
    let with = kernel(shared(Extent::Static(64)).with_dyn_layout(dynl("(?2,?3):(?4,1)")));
    let text = serde_json::to_string(&with).unwrap();
    assert!(text.contains(r#""dyn_layout":"(?2,?3):(?4,1)""#), "{text}");
    assert_eq!(serde_json::from_str::<KernelFn>(&text).unwrap(), with);
    // the document of the plain kernel loads with no dyn_layout
    let back: KernelFn = serde_json::from_str(&serde_json::to_string(&plain).unwrap()).unwrap();
    assert!(back.allocs[0].dyn_layout.is_none());
}
