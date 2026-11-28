// SPDX-License-Identifier: MIT OR Apache-2.0
//! A layout on an allocation that is passed to a callee (`CallArg::Alloc`; `docs/design-dynamic-layout.md` section 10): does the callee's
//! logical access go through the entry's layout? The entry fills the array through a callee (logical index `t` gets `t + 1`) and reads
//! it back through `Place::Alloc` (which applies the layout): the two agree only if the callee's access applied the same layout.

use unpopped::exec::{AllocDecl, AllocLayout, ExecId, Extent, LaunchShape, MemorySpace, Scope};
use unpopped::ir::ScalarExpr;
use unpopped::kernel::{
    BarrierKind, Block, BufferRole, CallArg, KernelFn, LocalDecl, Module, Param, Place, Stmt,
};
use unpopped::oracle::TypedBuffer;
use unpopped::oracle::kernel_eval::{KernelArg, KernelTrap};
use unpopped::oracle::kernel_launch::run_launch_module;
use unpopped_vocab::ElementKind::{F32, I32};
use unpopped_vocab::layout_algebra::DynLayout;

const N: u32 = 8;

fn tid() -> ScalarExpr {
    ScalarExpr::Exec(ExecId::ThreadIdx(0))
}

fn at(buffer: u32, i: ScalarExpr) -> Place {
    Place::Logical {
        buffer,
        coords: vec![i],
    }
}

fn rw(name: &str) -> Param {
    Param::buffer(name, F32, BufferRole::ReadWrite, Some(1))
}

/// `fill(t)`: `t[ThreadIdx.x] = ThreadIdx.x + 1` (as `f32`).
fn fill() -> KernelFn {
    let spec = unpopped::ir::CastSpec {
        to: F32,
        round: unpopped::ir::RoundMode::NearestEven,
        sat: unpopped::ir::Saturation::Wrap,
    };
    let v = ScalarExpr::Add(
        Box::new(ScalarExpr::Cast(Box::new(tid()), spec)),
        Box::new(ScalarExpr::Const(1.0)),
    );
    let body = vec![Stmt::Store {
        place: at(0, tid()),
        value: v,
    }];
    KernelFn::new("fill", F32, vec![rw("t")], vec![], Block(body))
}

/// The entry: `fill` through a call, a barrier, then `out[t] = sh[t]` through `Place::Alloc` (the layout applied).
fn entry(alloc: AllocDecl) -> KernelFn {
    let body = vec![
        Stmt::Call {
            callee: 1,
            args: vec![CallArg::Alloc(0)],
            result: None,
        },
        Stmt::Barrier {
            scope: Scope::Block,
            kind: BarrierKind::Full,
        },
        Stmt::Load {
            local: 0,
            place: Place::Alloc {
                alloc: 0,
                index: tid(),
            },
        },
        Stmt::Store {
            place: at(0, tid()),
            value: ScalarExpr::Local(0),
        },
    ];
    let mut params = vec![rw("out")];
    params.extend(
        ["m", "n", "ld"]
            .iter()
            .zip(1u32..)
            .map(|(nm, loc)| Param::scalar(*nm, I32, loc)),
    );
    let mut locals = vec![LocalDecl::scalar(F32)];
    locals.extend((0..3).map(|_| LocalDecl::scalar(I32)));
    // the scalar parameters take locals 1..=3; local 0 is the temporary
    KernelFn::new("main", F32, params, locals, Block(body))
        .with_launch(LaunchShape::new([N, 1, 1], [1, 1, 1]))
        .with_allocs(vec![alloc])
}

fn run(alloc: AllocDecl, a: (i128, i128, i128)) -> Result<Vec<f32>, KernelTrap> {
    let m = Module::new(vec![entry(alloc), fill()]);
    let out = KernelArg::Buffer(TypedBuffer::new(
        F32,
        vec![i64::from(N)],
        vec![1],
        vec![0; 4 * N as usize],
    ));
    let mut args = vec![
        out,
        KernelArg::Int(a.0),
        KernelArg::Int(a.1),
        KernelArg::Int(a.2),
    ];
    run_launch_module(&m, 0, &mut args)?;
    let KernelArg::Buffer(b) = &args[0] else {
        unreachable!()
    };
    Ok(b.bytes
        .chunks(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect())
}

fn want() -> Vec<f32> {
    (0..N).map(|i| (i + 1) as f32).collect()
}

fn shared() -> AllocDecl {
    AllocDecl::new("sh", MemorySpace::Shared, F32, Extent::Static(N))
}

#[test]
fn a_static_layout_applies_to_the_callees_access_too() {
    let l = "(2,4):(4,1)".parse::<AllocLayout>().unwrap();
    assert_eq!(run(shared().with_layout(l), (0, 0, 0)), Ok(want()));
}

#[test]
fn a_dyn_layout_applies_to_the_callees_access_too() {
    let d: DynLayout = "(?1,?2):(?2,1)".parse().unwrap();
    assert_eq!(
        run(shared().with_dyn_layout(d.clone()), (2, 4, 0)),
        Ok(want())
    );
    // and a bind the launch refuses is refused before the call
    assert!(matches!(
        run(shared().with_dyn_layout(d), (2, 3, 0)),
        Err(KernelTrap::DynLayout { .. })
    ));
}
