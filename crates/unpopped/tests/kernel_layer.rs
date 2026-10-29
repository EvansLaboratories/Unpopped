// SPDX-License-Identifier: MIT OR Apache-2.0
//! The kernel layer's scope and type checker (`docs/ir-kernel-layer.md` section 4).
//!
//! Every rule has a **known-bad mutation**: the baseline program is valid, and each test changes
//! exactly one thing, so the test goes red if the check it targets is deleted, not merely if
//! checking is off. The unmutated baseline passing is the positive control that every mutation
//! shares.

use unpopped::ir::{ExprDag, OpDef, ScalarExpr, input};
use unpopped::kernel::{
    Block, BufferRole, KernelError, KernelFn, LocalDecl, Param, Place, Stmt, check_kernel,
};
use unpopped::plan::try_build_plan;
use unpopped::types::{Ty, TypeError, infer, infer_ty, infer_ty_in};
use unpopped::{op_from_text, op_to_text};
use unpopped_vocab::{ArchSku, ElementKind, OpCategory, OperandDesc, structure_key};

const F32: ElementKind = ElementKind::F32;
const I32: ElementKind = ElementKind::I32;

// Local ids of the baseline.
const N: u32 = 0; // scalar param n: i32
const I: u32 = 1; // loop counter: i32
const X: u32 = 2; // loaded element: f32

fn bx(e: ScalarExpr) -> Box<ScalarExpr> {
    Box::new(e)
}

fn ci(v: i64) -> ScalarExpr {
    ScalarExpr::ConstInt(v, I32)
}

fn local(i: u32) -> ScalarExpr {
    ScalarExpr::Local(i)
}

fn at(buffer: u32, c: ScalarExpr) -> Place {
    Place::Logical {
        buffer,
        coords: vec![c],
    }
}

/// `for i in 0..n { x = in[i]; out[i] = x + 1.0 }`
fn baseline_body() -> Vec<Stmt> {
    vec![Stmt::For {
        local: I,
        start: ci(0),
        end: local(N),
        step: ci(1),
        body: Block(vec![
            Stmt::Load {
                local: X,
                place: at(0, local(I)),
            },
            Stmt::Store {
                place: at(1, local(I)),
                value: ScalarExpr::Add(bx(local(X)), bx(ScalarExpr::Const(1.0))),
            },
        ]),
    }]
}

fn kernel(body: Vec<Stmt>) -> KernelFn {
    KernelFn::new(
        "baseline",
        F32,
        vec![
            Param::buffer("in", F32, BufferRole::Read, Some(1)),
            Param::buffer("out", F32, BufferRole::Write, Some(1)),
            Param::scalar("n", I32, N),
        ],
        vec![
            LocalDecl::scalar(I32),
            LocalDecl::scalar(I32),
            LocalDecl::scalar(F32),
        ],
        Block(body),
    )
}

/// The loop body of the baseline, for mutations that edit inside it.
fn loop_with(inner: Vec<Stmt>) -> Vec<Stmt> {
    vec![Stmt::For {
        local: I,
        start: ci(0),
        end: local(N),
        step: ci(1),
        body: Block(inner),
    }]
}

fn load_x() -> Stmt {
    Stmt::Load {
        local: X,
        place: at(0, local(I)),
    }
}

fn store_out(value: ScalarExpr) -> Stmt {
    Stmt::Store {
        place: at(1, local(I)),
        value,
    }
}

#[test]
fn the_baseline_is_valid() {
    assert_eq!(check_kernel(&kernel(baseline_body())), Ok(()));
}

#[test]
fn a_use_before_the_declaration_is_refused() {
    // Mutation: delete the Load that declares x.
    let k = kernel(loop_with(vec![store_out(local(X))]));
    assert_eq!(
        check_kernel(&k),
        Err(KernelError::UseBeforeLet { local: X })
    );
}

#[test]
fn a_use_after_the_declaring_block_closed_is_refused() {
    // Mutation: read x after the loop that declared it.
    let mut body = baseline_body();
    body.push(Stmt::Store {
        place: at(1, ci(0)),
        value: local(X),
    });
    assert_eq!(
        check_kernel(&kernel(body)),
        Err(KernelError::OutOfScope { local: X })
    );
}

#[test]
fn declaring_an_id_twice_in_scope_is_refused() {
    // Mutation: load into x a second time in the same block.
    let k = kernel(loop_with(vec![load_x(), load_x(), store_out(local(X))]));
    assert_eq!(check_kernel(&k), Err(KernelError::Redeclared { local: X }));
}

#[test]
fn an_id_outside_the_locals_table_is_refused() {
    let k = kernel(loop_with(vec![load_x(), store_out(local(9))]));
    assert_eq!(
        check_kernel(&k),
        Err(KernelError::UnknownLocal { local: 9 })
    );
}

#[test]
fn assigning_a_value_of_another_type_is_refused() {
    // Mutation: x is f32; assign it an i32.
    let k = kernel(loop_with(vec![
        load_x(),
        Stmt::Assign {
            local: X,
            value: ci(1),
        },
        store_out(local(X)),
    ]));
    assert_eq!(
        check_kernel(&k),
        Err(KernelError::TypeMismatch { what: "Assign" })
    );
    // Control: the same assignment at the right type passes.
    let ok = kernel(loop_with(vec![
        load_x(),
        Stmt::Assign {
            local: X,
            value: ScalarExpr::Const(2.0),
        },
        store_out(local(X)),
    ]));
    assert_eq!(check_kernel(&ok), Ok(()));
}

#[test]
fn a_load_into_a_local_of_the_wrong_type_is_refused() {
    // Mutation: the local that receives the f32 element is declared i32.
    let mut k = kernel(loop_with(vec![load_x(), store_out(local(X))]));
    k.locals[X as usize] = LocalDecl::scalar(I32);
    assert_eq!(
        check_kernel(&k),
        Err(KernelError::TypeMismatch {
            what: "declaration and its local"
        })
    );
}

#[test]
fn an_intent_layer_leaf_inside_a_kernel_is_refused() {
    // Mutation: Input(0), which has no meaning in a statement program.
    let k = kernel(loop_with(vec![store_out(input(0).0)]));
    assert_eq!(
        check_kernel(&k),
        Err(KernelError::IntentLeaf { node: "Input" })
    );
}

#[test]
fn branching_on_a_packed_value_is_refused() {
    // Mutation: the condition is a Pack of two f32 constants.
    let packed = ScalarExpr::Pack(vec![ScalarExpr::Const(1.0), ScalarExpr::Const(2.0)]);
    let k = kernel(vec![Stmt::If {
        cond: packed,
        then: Block(vec![]),
        els: Block(vec![]),
    }]);
    assert!(matches!(
        check_kernel(&k),
        Err(KernelError::TypeMismatch { .. })
    ));
    // Control: a scalar condition passes.
    let ok = kernel(vec![Stmt::If {
        cond: ScalarExpr::Const(1.0),
        then: Block(vec![]),
        els: Block(vec![]),
    }]);
    assert_eq!(check_kernel(&ok), Ok(()));
}

#[test]
fn break_and_continue_outside_a_loop_are_refused() {
    for st in [Stmt::Break, Stmt::Continue] {
        assert_eq!(
            check_kernel(&kernel(vec![st])),
            Err(KernelError::BreakOutsideLoop)
        );
    }
    // Control: inside the loop both are fine.
    let ok = kernel(loop_with(vec![Stmt::If {
        cond: ScalarExpr::Const(1.0),
        then: Block(vec![Stmt::Break]),
        els: Block(vec![Stmt::Continue]),
    }]));
    assert_eq!(check_kernel(&ok), Ok(()));
}

#[test]
fn a_statement_after_an_unconditional_exit_is_refused() {
    for exit in [Stmt::Return, Stmt::Break, Stmt::Continue] {
        let k = kernel(loop_with(vec![exit, store_out(ScalarExpr::Const(0.0))]));
        assert_eq!(check_kernel(&k), Err(KernelError::Unreachable));
    }
    // Control: an exit as the LAST statement is fine.
    let ok = kernel(loop_with(vec![
        store_out(ScalarExpr::Const(0.0)),
        Stmt::Return,
    ]));
    assert_eq!(check_kernel(&ok), Ok(()));
}

#[test]
fn buffer_roles_are_enforced_in_both_directions() {
    // Mutation: store into the Read buffer.
    let w = kernel(loop_with(vec![Stmt::Store {
        place: at(0, local(I)),
        value: ScalarExpr::Const(0.0),
    }]));
    assert_eq!(
        check_kernel(&w),
        Err(KernelError::WriteToReadOnly { buffer: 0 })
    );
    // Mutation: load from the Write buffer.
    let r = kernel(loop_with(vec![Stmt::Load {
        local: X,
        place: at(1, local(I)),
    }]));
    assert_eq!(
        check_kernel(&r),
        Err(KernelError::ReadFromWriteOnly { buffer: 1 })
    );
}

#[test]
fn a_place_must_name_a_buffer_with_integer_coordinates_of_the_right_arity() {
    // Wrong arity (rank 1 buffer, two coordinates).
    let two = kernel(loop_with(vec![Stmt::Load {
        local: X,
        place: Place::Logical {
            buffer: 0,
            coords: vec![local(I), local(I)],
        },
    }]));
    assert_eq!(check_kernel(&two), Err(KernelError::BadPlace { buffer: 0 }));
    // A float coordinate.
    let fl = kernel(loop_with(vec![Stmt::Load {
        local: X,
        place: at(0, ScalarExpr::Const(1.0)),
    }]));
    assert_eq!(check_kernel(&fl), Err(KernelError::BadPlace { buffer: 0 }));
    // The scalar parameter is not a buffer.
    let sc = kernel(loop_with(vec![Stmt::Load {
        local: X,
        place: at(2, local(I)),
    }]));
    assert_eq!(check_kernel(&sc), Err(KernelError::BadPlace { buffer: 2 }));
}

#[test]
fn the_loop_counter_and_scalar_params_cannot_be_assigned() {
    let counter = kernel(loop_with(vec![Stmt::Assign {
        local: I,
        value: ci(5),
    }]));
    assert_eq!(
        check_kernel(&counter),
        Err(KernelError::CounterAssigned { local: I })
    );
    let param = kernel(vec![Stmt::Assign {
        local: N,
        value: ci(5),
    }]);
    assert_eq!(
        check_kernel(&param),
        Err(KernelError::ParamAssigned { local: N })
    );
}

#[test]
fn for_bounds_must_share_one_integer_type() {
    // Mutation: the end bound is an f32.
    let mixed = kernel(vec![Stmt::For {
        local: I,
        start: ci(0),
        end: ScalarExpr::Const(4.0),
        step: ci(1),
        body: Block(vec![]),
    }]);
    assert_eq!(
        check_kernel(&mixed),
        Err(KernelError::TypeMismatch { what: "For bounds" })
    );
}

#[test]
fn a_counter_id_is_free_again_after_its_loop() {
    // Two sibling loops may reuse the counter id: scope ended with the first.
    let mut body = baseline_body();
    body.extend(baseline_body());
    assert_eq!(check_kernel(&kernel(body)), Ok(()));
}

#[test]
fn a_local_has_no_type_outside_a_kernel() {
    assert_eq!(infer_ty(&local(0), F32), Err(TypeError::LocalOutsideKernel));
    assert_eq!(infer(&local(0), F32), Err(TypeError::LocalOutsideKernel));
    // Inside a kernel it has the declared type, and an operator over it types normally.
    let tys = [Ty::Scalar(I32), Ty::Scalar(F32)];
    assert_eq!(infer_ty_in(&local(1), F32, &tys), Ok(Ty::Scalar(F32)));
    let sum = ScalarExpr::Add(bx(local(1)), bx(ScalarExpr::Const(1.0)));
    assert_eq!(infer_ty_in(&sum, F32, &tys), Ok(Ty::Scalar(F32)));
    // An id outside the table is still LocalOutsideKernel.
    assert_eq!(
        infer_ty_in(&local(5), F32, &tys),
        Err(TypeError::LocalOutsideKernel)
    );
}

#[test]
fn local_round_trips_the_dag_and_the_text_format() {
    let e = ScalarExpr::Add(bx(local(3)), bx(local(3)));
    let dag = ExprDag::from_expr(&e);
    assert_eq!(dag.to_expr(), e);
    // Two reads of one local are one DAG node.
    let locals = (0..dag.len() as u32)
        .filter(|&i| matches!(dag.node(i), unpopped::ir::DagNode::Local(3)))
        .count();
    assert_eq!(locals, 1);
    let op = OpDef::elementwise("l", 1, &[F32], unpopped::ir::Expr(e));
    assert_eq!(op_from_text(&op_to_text(&op)).expect("parses"), op);
}

#[test]
fn the_plan_gate_declines_an_op_body_containing_a_local() {
    let a = OperandDesc::new(1, &[7], &[1], F32, 256);
    let o = OperandDesc::new(1, &[7], &[1], F32, 256);
    let key = structure_key(OpCategory::UnaryElementwise, &[a, o], ArchSku::Sm89);
    let op = OpDef::elementwise("l", 1, &[F32], unpopped::ir::Expr(local(0)));
    let planned = std::panic::catch_unwind(|| try_build_plan(&op, &key).map(|_| ()));
    assert!(
        matches!(planned, Ok(Err(_))),
        "an OpDef body must not contain a kernel-layer local: {planned:?}"
    );
    // Control: the same op without the local plans.
    let plain = OpDef::elementwise("p", 1, &[F32], input(0) * unpopped::ir::konst(2.0));
    assert!(try_build_plan(&plain, &key).is_ok());
}

#[test]
fn a_kernel_round_trips_through_json() {
    let k = kernel(baseline_body());
    let text = serde_json::to_string(&k).expect("serializes");
    let back: KernelFn = serde_json::from_str(&text).expect("parses");
    assert_eq!(back, k);
    assert_eq!(check_kernel(&back), Ok(()));
}
