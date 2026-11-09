// SPDX-License-Identifier: MIT OR Apache-2.0
//! **A collective with a user combine function** (ROADMAP R5.7 D4; PM ruling 2026-11-09: a custom combine stays at the
//! kernel layer, as a callee): `CollectiveKind::{ReduceWith, ScanWith} { combine }` names a plain function
//! `(a: T, b: T) -> T` of the module and folds the scope's lane values with it, left to right in lane order.
//!
//! The lanes of an 8-thread block contribute `VALS = [-2, 3, -1, 4, 0, -4, 1, -3]`. The combines are chosen so that
//! the order of the fold and the tie rule are visible: "first" and "last" are associative but not commutative, and
//! "max by absolute value, the first of equal magnitudes" decides the `4` against the `-4` by position. The expected
//! arrays are worked out by hand.

use unpopped::exec::{ExecId, LaunchShape, Scope};
use unpopped::ir::{BinaryOp, ReduceOp, ScalarExpr};
use unpopped::kernel::{
    Block, BufferRole, CollectiveKind, KernelError, KernelFn, LocalDecl, Module, Param, Place,
    Stmt, check_kernel, check_module,
};
use unpopped::oracle::TypedBuffer;
use unpopped::oracle::kernel_eval::{KernelArg, KernelTrap};
use unpopped::oracle::kernel_launch::run_launch_module;
use unpopped_vocab::ElementKind;

const I32: ElementKind = ElementKind::I32;
const F32: ElementKind = ElementKind::F32;
const N: usize = 8;
const VALS: [i32; N] = [-2, 3, -1, 4, 0, -4, 1, -3];

fn tid() -> ScalarExpr {
    ScalarExpr::Exec(ExecId::ThreadIdx(0))
}

fn at(buffer: u32) -> Place {
    Place::Logical {
        buffer,
        coords: vec![tid()],
    }
}

fn bin(op: BinaryOp, a: ScalarExpr, b: ScalarExpr) -> ScalarExpr {
    ScalarExpr::Binary(op, Box::new(a), Box::new(b))
}

fn lt0(e: ScalarExpr) -> ScalarExpr {
    bin(BinaryOp::CmpLt, e, ScalarExpr::ConstInt(0, I32))
}

/// `|e|` for an `I32`.
fn abs(e: ScalarExpr) -> ScalarExpr {
    let neg = ScalarExpr::Sub(Box::new(ScalarExpr::ConstInt(0, I32)), Box::new(e.clone()));
    ScalarExpr::Select(Box::new(lt0(e.clone())), Box::new(neg), Box::new(e))
}

/// A combine function over `ty`: parameters `a` (local 0) and `b` (local 1), result local 2 set to `body`.
fn combine_fn(name: &str, ty: ElementKind, body: ScalarExpr) -> KernelFn {
    let mut f = KernelFn::new(
        name,
        ty,
        vec![Param::scalar("a", ty, 0), Param::scalar("b", ty, 1)],
        vec![
            LocalDecl::scalar(ty),
            LocalDecl::scalar(ty),
            LocalDecl::scalar(ty),
        ],
        Block(vec![Stmt::Let {
            local: 2,
            init: body,
        }]),
    );
    f.result = Some(2);
    f
}

fn a() -> ScalarExpr {
    ScalarExpr::Local(0)
}

fn b() -> ScalarExpr {
    ScalarExpr::Local(1)
}

fn first() -> KernelFn {
    combine_fn("first", I32, a())
}

fn last() -> KernelFn {
    combine_fn("last", I32, b())
}

/// The larger magnitude, the first of two of equal magnitude (`|b| > |a|` replaces, a tie keeps `a`).
fn max_abs() -> KernelFn {
    let better = bin(BinaryOp::CmpGt, abs(b()), abs(a()));
    combine_fn(
        "max_abs",
        I32,
        ScalarExpr::Select(Box::new(better), Box::new(b()), Box::new(a())),
    )
}

fn add() -> KernelFn {
    combine_fn("add", I32, ScalarExpr::Add(Box::new(a()), Box::new(b())))
}

/// An 8-thread launched entry (function 0): load `vals[tid]`, run the collective, store `out[tid]`.
fn entry(kind: CollectiveKind, scope: Scope) -> KernelFn {
    KernelFn::new(
        "main",
        I32,
        vec![
            Param::buffer("vals", I32, BufferRole::Read, Some(1)),
            Param::buffer("out", I32, BufferRole::Write, Some(1)),
        ],
        vec![LocalDecl::scalar(I32), LocalDecl::scalar(I32)],
        Block(vec![
            Stmt::Load {
                local: 0,
                place: at(0),
            },
            Stmt::Collective {
                kind,
                value: ScalarExpr::Local(0),
                result: 1,
                scope,
            },
            Stmt::Store {
                place: at(1),
                value: ScalarExpr::Local(1),
            },
        ]),
    )
    .with_launch(LaunchShape::new([N as u32, 1, 1], [1, 1, 1]).with_subgroup(4, 4))
}

fn module(kind: CollectiveKind, scope: Scope, f: KernelFn) -> Module {
    Module::new(vec![entry(kind, scope), f])
}

fn run(m: &Module) -> Result<Vec<i32>, KernelTrap> {
    let vals: Vec<u8> = VALS.iter().flat_map(|v| v.to_le_bytes()).collect();
    let mut args = vec![
        KernelArg::Buffer(TypedBuffer::new(I32, vec![N as i64], vec![1], vals)),
        KernelArg::Buffer(TypedBuffer::new(
            I32,
            vec![N as i64],
            vec![1],
            vec![0; 4 * N],
        )),
    ];
    run_launch_module(m, 0, &mut args)?;
    let KernelArg::Buffer(out) = &args[1] else {
        unreachable!()
    };
    Ok(out
        .bytes
        .chunks(4)
        .map(|c| i32::from_le_bytes(c.try_into().unwrap()))
        .collect())
}

fn reduce(f: KernelFn, scope: Scope) -> Vec<i32> {
    run(&module(CollectiveKind::ReduceWith { combine: 1 }, scope, f)).expect("runs")
}

fn scan(f: KernelFn, scope: Scope) -> Vec<i32> {
    run(&module(CollectiveKind::ScanWith { combine: 1 }, scope, f)).expect("runs")
}

#[test]
fn the_fold_runs_left_to_right_in_lane_order() {
    // "first" keeps lane 0's value, "last" keeps the last lane's: an order that is not a commutative one.
    assert_eq!(reduce(first(), Scope::Block), vec![-2; N]);
    assert_eq!(reduce(last(), Scope::Block), vec![-3; N]);
    // within two subgroups of four: lanes 0..4 and 4..8
    assert_eq!(
        reduce(first(), Scope::Subgroup),
        vec![-2, -2, -2, -2, 0, 0, 0, 0]
    );
    assert_eq!(
        reduce(last(), Scope::Subgroup),
        vec![4, 4, 4, 4, -3, -3, -3, -3]
    );
}

#[test]
fn a_tie_between_equal_magnitudes_goes_to_the_earlier_lane() {
    // 4 (lane 3) and -4 (lane 5) tie: the first of them wins
    assert_eq!(reduce(max_abs(), Scope::Block), vec![4; N]);
    // inside the second subgroup (0, -4, 1, -3) the winner is -4
    assert_eq!(
        reduce(max_abs(), Scope::Subgroup),
        vec![4, 4, 4, 4, -4, -4, -4, -4]
    );
}

#[test]
fn a_scan_gives_each_lane_the_inclusive_prefix_fold() {
    assert_eq!(scan(max_abs(), Scope::Block), vec![-2, 3, 3, 4, 4, 4, 4, 4]);
    assert_eq!(scan(first(), Scope::Block), vec![-2; N]);
    assert_eq!(scan(last(), Scope::Block), VALS.to_vec());
    // a subgroup restarts the prefix at its first lane
    assert_eq!(
        scan(max_abs(), Scope::Subgroup),
        vec![-2, 3, 3, 4, 0, -4, -4, -4]
    );
}

#[test]
fn an_addition_combine_agrees_with_the_built_in_sum() {
    let built_in = run(&module(
        CollectiveKind::Reduce(ReduceOp::Sum),
        Scope::Block,
        add(),
    ))
    .expect("runs");
    assert_eq!(built_in, vec![-2; N], "the sum of VALS is -2 (the control)");
    assert_eq!(reduce(add(), Scope::Block), built_in);
    let scan_built_in = run(&module(
        CollectiveKind::Scan {
            op: ReduceOp::Sum,
            exclusive: false,
        },
        Scope::Block,
        add(),
    ))
    .expect("runs");
    assert_eq!(scan(add(), Scope::Block), scan_built_in);
}

#[test]
fn the_checker_resolves_the_function_in_the_module_and_refuses_alone() {
    let m = module(
        CollectiveKind::ReduceWith { combine: 1 },
        Scope::Block,
        add(),
    );
    assert_eq!(check_module(&m, 0), Ok(()));
    // alone there is no module to resolve it in
    assert!(matches!(
        check_kernel(&m.fns[0]),
        Err(KernelError::CallUnsupported { .. })
    ));
}

#[test]
fn a_combine_that_is_not_a_t_t_to_t_function_is_refused() {
    let refused = |m: Module| {
        matches!(
            check_module(&m, 0),
            Err(KernelError::CollectiveUnsupported { .. })
        )
    };
    // wrong arity
    let mut one = add();
    one.params.pop();
    assert!(refused(module(
        CollectiveKind::ReduceWith { combine: 1 },
        Scope::Block,
        one
    )));
    // over a different type than the collective's value
    let f32_add = combine_fn(
        "f32_add",
        F32,
        ScalarExpr::Add(Box::new(a()), Box::new(b())),
    );
    assert!(refused(module(
        CollectiveKind::ReduceWith { combine: 1 },
        Scope::Block,
        f32_add
    )));
    // no result
    let mut none = add();
    none.result = None;
    assert!(refused(module(
        CollectiveKind::ReduceWith { combine: 1 },
        Scope::Block,
        none
    )));
}

#[test]
fn a_combine_with_a_wrong_parameter_or_result_type_is_refused() {
    let refused = |m: Module| {
        matches!(
            check_module(&m, 0),
            Err(KernelError::CollectiveUnsupported { .. })
        )
    };
    // a third parameter
    let mut three = add();
    three.params.push(Param::scalar("c", I32, 3));
    three.locals.push(LocalDecl::scalar(I32));
    assert!(refused(module(
        CollectiveKind::ReduceWith { combine: 1 },
        Scope::Block,
        three
    )));
    // a first parameter of another type
    let mut mixed = add();
    mixed.params[0] = Param::scalar("a", F32, 0);
    mixed.locals[0] = LocalDecl::scalar(F32);
    assert!(refused(module(
        CollectiveKind::ReduceWith { combine: 1 },
        Scope::Block,
        mixed
    )));
    // a result of another type
    let mut wide = add();
    wide.locals[2] = LocalDecl::scalar(F32);
    assert!(refused(module(
        CollectiveKind::ReduceWith { combine: 1 },
        Scope::Block,
        wide
    )));
    // control: the same module with the right function checks
    assert_eq!(
        check_module(
            &module(
                CollectiveKind::ReduceWith { combine: 1 },
                Scope::Block,
                add()
            ),
            0
        ),
        Ok(())
    );
}

#[test]
fn a_combine_that_reaches_a_barrier_is_refused() {
    let mut f = add();
    f.body.0.push(Stmt::Barrier {
        scope: Scope::Block,
        kind: unpopped::kernel::BarrierKind::Full,
    });
    let m = module(CollectiveKind::ReduceWith { combine: 1 }, Scope::Block, f);
    assert!(matches!(
        check_module(&m, 0),
        Err(KernelError::CollectiveUnsupported { .. })
    ));
}

#[test]
fn a_combine_outside_the_module_or_naming_the_entry_is_a_typed_error() {
    let outside = module(
        CollectiveKind::ReduceWith { combine: 7 },
        Scope::Block,
        add(),
    );
    assert!(matches!(
        check_module(&outside, 0),
        Err(KernelError::BadCallee { callee: 7 })
    ));
    // the entry cannot be its own combine function: the collective is a call edge, so the graph has a cycle
    let own = module(
        CollectiveKind::ReduceWith { combine: 0 },
        Scope::Block,
        add(),
    );
    assert!(matches!(
        check_module(&own, 0),
        Err(KernelError::CallCycle { .. })
    ));
}

#[test]
fn running_without_a_module_is_refused_before_it_runs() {
    let k = entry(CollectiveKind::ReduceWith { combine: 1 }, Scope::Block);
    let mut args = vec![
        KernelArg::Buffer(TypedBuffer::new(
            I32,
            vec![N as i64],
            vec![1],
            vec![0; 4 * N],
        )),
        KernelArg::Buffer(TypedBuffer::new(
            I32,
            vec![N as i64],
            vec![1],
            vec![0; 4 * N],
        )),
    ];
    assert!(matches!(
        unpopped::oracle::kernel_launch::run_launch(&k, &mut args),
        Err(KernelTrap::Malformed(KernelError::CallUnsupported { .. }))
    ));
}
