// SPDX-License-Identifier: MIT OR Apache-2.0
//! Allocations through a call (ROADMAP U3 R1, slice M4c of `docs/options-u3-calls.md`): the entry's `Shared`
//! or `Private` array is passed by alias to a callee's one-dimensional buffer parameter (`CallArg::Alloc`),
//! which the callee reads and writes through `Place::Logical`. The access is the allocation's: logged for
//! races, ordered by the entry's or the callee's barriers, bounds-checked against the allocation, and a
//! callee may forward its parameter to a nested callee (`CallArg::Buffer`).

use unpopped::exec::{AllocDecl, ExecId, Extent, LaunchShape, MemorySpace, Scope};
use unpopped::ir::ScalarExpr;
use unpopped::kernel::{
    BarrierKind, Block, BufferRole, CallArg, KernelError, KernelFn, LocalDecl, Module, Param,
    Place, Stmt,
};
use unpopped::oracle::TypedBuffer;
use unpopped::oracle::kernel_eval::{KernelArg, KernelTrap};
use unpopped::oracle::kernel_launch::run_launch_module;
use unpopped_vocab::ElementKind;

const F32: ElementKind = ElementKind::F32;
const U32: ElementKind = ElementKind::U32;
const N: u32 = 8;

fn tid() -> ScalarExpr {
    ScalarExpr::Exec(ExecId::ThreadIdx(0))
}

fn u(v: i64) -> ScalarExpr {
    ScalarExpr::ConstInt(v, U32)
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

fn barrier() -> Stmt {
    Stmt::Barrier {
        scope: Scope::Block,
        kind: BarrierKind::Full,
    }
}

fn call(callee: u32, args: Vec<CallArg>) -> Stmt {
    Stmt::Call {
        callee,
        args,
        result: None,
    }
}

fn f32_of(e: ScalarExpr) -> ScalarExpr {
    let spec = unpopped::ir::CastSpec {
        to: F32,
        round: unpopped::ir::RoundMode::NearestEven,
        sat: unpopped::ir::Saturation::Wrap,
    };
    ScalarExpr::Cast(Box::new(e), spec)
}

fn out_buf() -> KernelArg {
    KernelArg::Buffer(TypedBuffer::new(
        F32,
        vec![i64::from(N)],
        vec![1],
        vec![0; 4 * N as usize],
    ))
}

fn read(a: &KernelArg) -> Vec<f32> {
    let KernelArg::Buffer(b) = a else {
        unreachable!()
    };
    b.bytes
        .chunks(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect()
}

fn shared(n: u32) -> AllocDecl {
    AllocDecl::new("sh", MemorySpace::Shared, F32, Extent::Static(n))
}

/// The entry: allocation 0 and buffer `out` (param 0), one block of `N` threads.
fn entry(alloc: AllocDecl, body: Vec<Stmt>) -> KernelFn {
    KernelFn::new("main", F32, vec![rw("out")], vec![], Block(body))
        .with_launch(LaunchShape::new([N, 1, 1], [1, 1, 1]))
        .with_allocs(vec![alloc])
}

/// `fill(t)`: `t[ThreadIdx.x] = ThreadIdx.x + 1`.
fn fill() -> KernelFn {
    let v = ScalarExpr::Add(Box::new(f32_of(tid())), Box::new(ScalarExpr::Const(1.0)));
    let body = vec![Stmt::Store {
        place: at(0, tid()),
        value: v,
    }];
    KernelFn::new("fill", F32, vec![rw("t")], vec![], Block(body))
}

/// `mirror(t, out)`: `out[ThreadIdx.x] = t[N - 1 - ThreadIdx.x]`.
fn mirror() -> KernelFn {
    let rev = ScalarExpr::Sub(Box::new(u(i64::from(N) - 1)), Box::new(tid()));
    let body = vec![
        Stmt::Load {
            local: 0,
            place: at(0, rev),
        },
        Stmt::Store {
            place: at(1, tid()),
            value: ScalarExpr::Local(0),
        },
    ];
    KernelFn::new(
        "mirror",
        F32,
        vec![rw("t"), rw("out")],
        vec![LocalDecl::scalar(F32)],
        Block(body),
    )
}

fn run(m: &Module) -> Result<Vec<f32>, KernelTrap> {
    let mut args = vec![out_buf()];
    run_launch_module(m, 0, &mut args)?;
    Ok(read(&args[0]))
}

fn mirrored() -> Vec<f32> {
    (0..N).map(|i| (N - i) as f32).collect()
}

#[test]
fn a_shared_array_filled_in_one_call_and_read_in_another() {
    let body = vec![
        call(1, vec![CallArg::Alloc(0)]),
        barrier(),
        call(2, vec![CallArg::Alloc(0), CallArg::Buffer(0)]),
    ];
    let m = Module::new(vec![entry(shared(N), body), fill(), mirror()]);
    assert_eq!(run(&m).expect("runs"), mirrored());
}

#[test]
fn without_the_barrier_the_shared_array_is_caught() {
    // the oracle runs the threads of a phase one after another: the first thread to read the mirror slot
    // finds it unwritten (`UninitializedShared`), and a later one would race with the writer (`DataRace`)
    let body = vec![
        call(1, vec![CallArg::Alloc(0)]),
        call(2, vec![CallArg::Alloc(0), CallArg::Buffer(0)]),
    ];
    let m = Module::new(vec![entry(shared(N), body), fill(), mirror()]);
    assert!(
        matches!(
            run(&m),
            Err(KernelTrap::DataRace | KernelTrap::UninitializedShared { .. })
        ),
        "{:?}",
        run(&m)
    );
}

#[test]
fn a_barrier_inside_the_callee_orders_the_shared_array() {
    // exchange(t, out): fill; barrier; mirror, all in one callee that reaches a barrier
    let mut ex = mirror();
    ex.name = "exchange".into();
    ex.body.0.insert(
        0,
        Stmt::Store {
            place: at(0, tid()),
            value: ScalarExpr::Add(Box::new(f32_of(tid())), Box::new(ScalarExpr::Const(1.0))),
        },
    );
    ex.body.0.insert(1, barrier());
    let body = vec![call(1, vec![CallArg::Alloc(0), CallArg::Buffer(0)])];
    let m = Module::new(vec![entry(shared(N), body), ex]);
    assert_eq!(run(&m).expect("runs"), mirrored());
}

#[test]
fn a_callee_forwards_its_array_parameter_to_a_nested_callee() {
    // mid(t, out): fill(t) is called by the entry; mid forwards (t, out) to mirror
    let mid = KernelFn::new(
        "mid",
        F32,
        vec![rw("t"), rw("out")],
        vec![],
        Block(vec![call(3, vec![CallArg::Buffer(0), CallArg::Buffer(1)])]),
    );
    let body = vec![
        call(1, vec![CallArg::Alloc(0)]),
        barrier(),
        call(2, vec![CallArg::Alloc(0), CallArg::Buffer(0)]),
    ];
    let m = Module::new(vec![entry(shared(N), body), fill(), mid, mirror()]);
    assert_eq!(run(&m).expect("runs"), mirrored());
}

/// `fill2(p)`: `p[0] = ThreadIdx.x; p[1] = 100.0`.
fn fill2() -> KernelFn {
    let body = vec![
        Stmt::Store {
            place: at(0, u(0)),
            value: f32_of(tid()),
        },
        Stmt::Store {
            place: at(0, u(1)),
            value: ScalarExpr::Const(100.0),
        },
    ];
    KernelFn::new("fill2", F32, vec![rw("p")], vec![], Block(body))
}

/// The entry: a private array of 2, filled by `fill2`, read back and summed into `out[ThreadIdx.x]`.
fn private_entry() -> KernelFn {
    let load = |local, index| Stmt::Load {
        local,
        place: Place::Alloc {
            alloc: 0,
            index: u(index),
        },
    };
    let sum = ScalarExpr::Add(
        Box::new(ScalarExpr::Local(0)),
        Box::new(ScalarExpr::Local(1)),
    );
    let store = Stmt::Store {
        place: at(0, tid()),
        value: sum,
    };
    let body = vec![
        call(1, vec![CallArg::Alloc(0)]),
        load(0, 0),
        load(1, 1),
        store,
    ];
    let private = AllocDecl::new("p", MemorySpace::Private, F32, Extent::Static(2));
    KernelFn::new(
        "main",
        F32,
        vec![rw("out")],
        vec![LocalDecl::scalar(F32), LocalDecl::scalar(F32)],
        Block(body),
    )
    .with_launch(LaunchShape::new([N, 1, 1], [1, 1, 1]))
    .with_allocs(vec![private])
}

#[test]
fn a_private_array_is_the_calling_threads_own() {
    // each thread fills its private array through a callee and reads it back: no race, its own slot
    let out = run(&Module::new(vec![private_entry(), fill2()])).expect("runs");
    let want: Vec<f32> = (0..N).map(|i| i as f32 + 100.0).collect();
    assert_eq!(out, want);
}

#[test]
fn an_access_past_the_allocation_is_the_allocations_trap() {
    // the callee writes t[N] of an array of N elements
    let far = KernelFn::new(
        "far",
        F32,
        vec![rw("t")],
        vec![],
        Block(vec![Stmt::Store {
            place: at(0, u(i64::from(N))),
            value: ScalarExpr::Const(1.0),
        }]),
    );
    let body = vec![call(1, vec![CallArg::Alloc(0)])];
    let m = Module::new(vec![entry(shared(N), body), far]);
    assert!(
        matches!(run(&m), Err(KernelTrap::AllocOutOfBounds { .. })),
        "{:?}",
        run(&m)
    );
}

#[test]
fn the_checker_matches_an_allocation_to_its_parameter() {
    let call_fill = |alloc: AllocDecl, callee: KernelFn| {
        let body = vec![call(1, vec![CallArg::Alloc(0)])];
        Module::new(vec![entry(alloc, body), callee])
    };
    let check = |m: &Module| unpopped::kernel::check_module(m, 0);
    assert_eq!(check(&call_fill(shared(N), fill())), Ok(()));
    // a different element type
    let ints = AllocDecl::new(
        "sh",
        MemorySpace::Shared,
        ElementKind::I32,
        Extent::Static(N),
    );
    assert!(matches!(
        check(&call_fill(ints, fill())),
        Err(KernelError::CallArgMismatch { arg: 0, .. })
    ));
    // a two-dimensional parameter cannot take a one-dimensional allocation
    let mut two_d = fill();
    two_d.params[0] = Param::buffer("t", F32, BufferRole::ReadWrite, Some(2));
    assert!(matches!(
        check(&call_fill(shared(N), two_d)),
        Err(KernelError::CallArgMismatch { arg: 0, .. })
    ));
    // a parameter of unknown rank is not enough either: the callee's index count must be known
    let mut any = fill();
    any.params[0] = Param::buffer("t", F32, BufferRole::ReadWrite, None);
    assert!(matches!(
        check(&call_fill(shared(N), any)),
        Err(KernelError::CallArgMismatch { arg: 0, .. })
    ));
}
