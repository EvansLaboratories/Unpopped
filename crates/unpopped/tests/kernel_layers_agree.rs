// SPDX-License-Identifier: MIT OR Apache-2.0
//! **The layers agree** (`docs/ir-kernel-layer.md` sections 5 and 7): an Elementwise `OpDef`
//! lifted into the kernel layer and run by the kernel interpreter produces, byte for byte, what
//! `oracle::evaluate` produces for the `OpDef`.
//!
//! Special values (signed zero, NaN, infinities, subnormals, extremes) are in every corpus: an
//! agreement over friendly numbers would pass a lift that dropped them. A broadcast input
//! (row stride 0) proves the view survives behind `Place::Logical`.

use unpopped::ir::{BinaryOp, Expr, OpDef, ScalarExpr, UnaryOp, input};
use unpopped::kernel::{BufferRole, KernelFn, Param, check_kernel};
use unpopped::kernel_lift::{LiftDecline, lift_elementwise_opdef};
use unpopped::oracle::kernel_eval::{KernelArg, KernelTrap, run_kernel};
use unpopped::oracle::{TypedBuffer, evaluate};
use unpopped::plan::try_build_plan;
use unpopped_vocab::{ArchSku, ElementKind, OpCategory, OperandDesc, structure_key};

const ROWS: i64 = 3;
const COLS: i64 = 5;

fn bx(e: ScalarExpr) -> Box<ScalarExpr> {
    Box::new(e)
}

fn un(op: UnaryOp) -> ScalarExpr {
    ScalarExpr::Unary(op, bx(input(0).0))
}

fn bin(op: BinaryOp) -> ScalarExpr {
    ScalarExpr::Binary(op, bx(input(0).0), bx(input(1).0))
}

fn f32_bytes(vs: &[f32]) -> Vec<u8> {
    vs.iter().flat_map(|v| v.to_le_bytes()).collect()
}

fn f64_bytes(vs: &[f64]) -> Vec<u8> {
    vs.iter().flat_map(|v| v.to_le_bytes()).collect()
}

fn i32_bytes(vs: &[i32]) -> Vec<u8> {
    vs.iter().flat_map(|v| v.to_le_bytes()).collect()
}

/// 15 f32 values with the awkward ones in.
fn f32_corpus(seed: usize) -> Vec<f32> {
    let base = [
        0.0,
        -0.0,
        1.0,
        -1.0,
        0.5,
        2.0,
        -2.5,
        f32::MIN_POSITIVE,
        1.0e-40,
        3.0e38,
        f32::INFINITY,
        f32::NEG_INFINITY,
        f32::NAN,
        0.1,
        -7.25,
    ];
    (0..15).map(|i| base[(i + seed) % base.len()]).collect()
}

/// Run `op` through the oracle and through the lifted kernel; return both output byte images.
fn both(
    op: &OpDef,
    cat: OpCategory,
    dtype: ElementKind,
    inputs: &[TypedBuffer],
) -> (Vec<u8>, Vec<u8>) {
    let rank = 2usize;
    let desc = |tb: &TypedBuffer| OperandDesc::new(rank, &tb.shape, &tb.strides, dtype, 256);
    let out_shape = [ROWS, COLS];
    let out_strides = [COLS, 1];
    let out_desc = OperandDesc::new(rank, &out_shape, &out_strides, dtype, 256);
    let mut descs: Vec<OperandDesc> = inputs.iter().map(desc).collect();
    descs.push(out_desc);
    let key = structure_key(cat, &descs, ArchSku::Sm89);
    let plan = try_build_plan(op, &key).expect("the op plans");
    let want = evaluate(&plan, &descs, inputs, &[]).remove(0).bytes;

    let kernel = lift_elementwise_opdef(op, dtype, 2).expect("the op lifts");
    assert_eq!(
        check_kernel(&kernel),
        Ok(()),
        "a lifted kernel is well formed"
    );
    let n_bytes = (ROWS * COLS) as usize * 16;
    let mut args: Vec<KernelArg> = inputs.iter().cloned().map(KernelArg::Buffer).collect();
    args.push(KernelArg::Buffer(TypedBuffer::new(
        dtype,
        out_shape.to_vec(),
        out_strides.to_vec(),
        vec![0u8; n_bytes],
    )));
    args.push(KernelArg::Int(i128::from(ROWS)));
    args.push(KernelArg::Int(i128::from(COLS)));
    run_kernel(&kernel, &mut args).expect("the kernel runs");
    let KernelArg::Buffer(out) = &args[inputs.len()] else {
        unreachable!()
    };
    let size = want.len();
    (want, out.bytes[..size].to_vec())
}

fn dense_f32(seed: usize) -> TypedBuffer {
    TypedBuffer::new(
        ElementKind::F32,
        vec![ROWS, COLS],
        vec![COLS, 1],
        f32_bytes(&f32_corpus(seed)),
    )
}

/// A row vector broadcast down the rows (row stride 0).
fn broadcast_f32(seed: usize) -> TypedBuffer {
    TypedBuffer::new(
        ElementKind::F32,
        vec![ROWS, COLS],
        vec![0, 1],
        f32_bytes(&f32_corpus(seed)[..COLS as usize]),
    )
}

#[test]
fn every_unary_op_agrees_on_f32_special_values() {
    let ops = [
        UnaryOp::Neg,
        UnaryOp::Abs,
        UnaryOp::Sqr,
        UnaryOp::Sqrt,
        UnaryOp::Rsqrt,
        UnaryOp::Recip,
        UnaryOp::Exp,
        UnaryOp::Log,
        UnaryOp::Tanh,
        UnaryOp::Sigmoid,
        UnaryOp::Relu,
        UnaryOp::Erf,
    ];
    for u in ops {
        let op = OpDef::elementwise("u", 1, &[ElementKind::F32], Expr(un(u)));
        let (want, got) = both(
            &op,
            OpCategory::UnaryElementwise,
            ElementKind::F32,
            &[dense_f32(0)],
        );
        assert_eq!(got, want, "{u:?}");
    }
}

#[test]
fn every_binary_op_agrees_on_f32_special_values_with_a_broadcast_input() {
    let ops = [
        BinaryOp::Max,
        BinaryOp::Min,
        BinaryOp::Pow,
        BinaryOp::Rem,
        BinaryOp::Atan2,
        BinaryOp::Copysign,
        BinaryOp::Nextafter,
        BinaryOp::FmaxIeee,
        BinaryOp::FminIeee,
        BinaryOp::CmpEq,
        BinaryOp::CmpNe,
        BinaryOp::CmpLt,
        BinaryOp::CmpLe,
    ];
    for b in ops {
        let op = OpDef::elementwise("b", 2, &[ElementKind::F32], Expr(bin(b)));
        let (want, got) = both(
            &op,
            OpCategory::BinaryElementwise,
            ElementKind::F32,
            &[dense_f32(3), broadcast_f32(7)],
        );
        assert_eq!(got, want, "{b:?}");
    }
}

#[test]
fn arithmetic_select_fma_and_a_cast_agree() {
    let a = || input(0).0;
    let b = || input(1).0;
    let bodies: Vec<(&str, ScalarExpr)> = vec![
        ("add", ScalarExpr::Add(bx(a()), bx(b()))),
        ("sub", ScalarExpr::Sub(bx(a()), bx(b()))),
        ("mul", ScalarExpr::Mul(bx(a()), bx(b()))),
        ("div", ScalarExpr::Div(bx(a()), bx(b()))),
        (
            "fma",
            ScalarExpr::Fma(bx(a()), bx(b()), bx(ScalarExpr::Const(0.1))),
        ),
        (
            "select",
            ScalarExpr::Select(
                bx(ScalarExpr::Binary(BinaryOp::CmpLt, bx(a()), bx(b()))),
                bx(a()),
                bx(b()),
            ),
        ),
        (
            "const mix",
            ScalarExpr::Add(
                bx(ScalarExpr::Const(1.5)),
                bx(ScalarExpr::Mul(bx(a()), bx(ScalarExpr::Const(-0.0)))),
            ),
        ),
    ];
    for (name, body) in bodies {
        let op = OpDef::elementwise(name, 2, &[ElementKind::F32], Expr(body));
        let (want, got) = both(
            &op,
            OpCategory::BinaryElementwise,
            ElementKind::F32,
            &[dense_f32(1), broadcast_f32(4)],
        );
        assert_eq!(got, want, "{name}");
    }
}

#[test]
fn f64_compute_agrees() {
    let vals: Vec<f64> = vec![
        0.0,
        -0.0,
        1.0,
        -1.0,
        1.0e-310,
        1.7e308,
        f64::INFINITY,
        f64::NAN,
        0.1,
        -7.25,
        3.0,
        1.0e-5,
        2.5,
        -2.5,
        123456.789,
    ];
    let tb = TypedBuffer::new(
        ElementKind::F64,
        vec![ROWS, COLS],
        vec![COLS, 1],
        f64_bytes(&vals),
    );
    let op = OpDef::elementwise(
        "f64",
        1,
        &[ElementKind::F64],
        Expr(ScalarExpr::Mul(
            bx(un(UnaryOp::Sqr)),
            bx(ScalarExpr::Const(3.0)),
        )),
    );
    let (want, got) = both(&op, OpCategory::UnaryElementwise, ElementKind::F64, &[tb]);
    assert_eq!(got, want);
}

#[test]
fn integer_compute_wraps_exactly_as_the_oracle_does() {
    let vals = [
        i32::MAX,
        i32::MIN,
        -1,
        0,
        1,
        2,
        -2,
        65_536,
        -65_536,
        i32::MAX - 1,
        7,
        -7,
        1_000_000,
        -1_000_000,
        46_341,
    ];
    let tb = |s: usize| {
        let v: Vec<i32> = (0..15).map(|i| vals[(i + s) % vals.len()]).collect();
        TypedBuffer::new(
            ElementKind::I32,
            vec![ROWS, COLS],
            vec![COLS, 1],
            i32_bytes(&v),
        )
    };
    for (name, body) in [
        ("add", ScalarExpr::Add(bx(input(0).0), bx(input(1).0))),
        ("sub", ScalarExpr::Sub(bx(input(0).0), bx(input(1).0))),
        ("mul", ScalarExpr::Mul(bx(input(0).0), bx(input(1).0))),
    ] {
        let op = OpDef::elementwise(name, 2, &[ElementKind::I32], Expr(body));
        let (want, got) = both(
            &op,
            OpCategory::BinaryElementwise,
            ElementKind::I32,
            &[tb(0), tb(5)],
        );
        assert_eq!(got, want, "{name}");
    }
}

#[test]
fn forms_the_lift_does_not_cover_decline_with_a_reason() {
    let f32k = [ElementKind::F32];
    // Control: the plain op lifts.
    let plain = OpDef::elementwise("p", 1, &f32k, Expr(input(0).0));
    assert!(lift_elementwise_opdef(&plain, ElementKind::F32, 2).is_ok());
    // A body that reads Coord.
    let coord = OpDef::elementwise("c", 1, &f32k, Expr(ScalarExpr::Coord(0)));
    assert_eq!(
        lift_elementwise_opdef(&coord, ElementKind::F32, 2),
        Err(LiftDecline::UnsupportedLeaf("Coord"))
    );
    // A body that reads a runtime Param.
    let param = OpDef::elementwise("q", 1, &f32k, Expr(ScalarExpr::Param(0)));
    assert_eq!(
        lift_elementwise_opdef(&param, ElementKind::F32, 2),
        Err(LiftDecline::UnsupportedLeaf("Param"))
    );
    // Rank 0.
    assert_eq!(
        lift_elementwise_opdef(&plain, ElementKind::F32, 0),
        Err(LiftDecline::BadRank)
    );
    // Dtype mismatch between the op's declared dtypes and the compute dtype.
    assert_eq!(
        lift_elementwise_opdef(&plain, ElementKind::F64, 2),
        Err(LiftDecline::MixedDtypes)
    );
}

// --------------------------------------------------------------------------------------
// The interpreter's own semantics, on hand-built kernels.

use unpopped::kernel::{Block, LocalDecl, Place, Stmt};

const I32: ElementKind = ElementKind::I32;

fn ci(v: i64) -> ScalarExpr {
    ScalarExpr::ConstInt(v, I32)
}

fn local(i: u32) -> ScalarExpr {
    ScalarExpr::Local(i)
}

fn out_buf(n: usize) -> KernelArg {
    KernelArg::Buffer(TypedBuffer::new(
        I32,
        vec![n as i64],
        vec![1],
        vec![0u8; n * 4],
    ))
}

fn read_i32(a: &KernelArg) -> Vec<i32> {
    let KernelArg::Buffer(b) = a else { panic!() };
    b.bytes
        .chunks(4)
        .map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

fn store_at(idx: ScalarExpr, v: ScalarExpr) -> Stmt {
    Stmt::Store {
        place: Place::Logical {
            buffer: 0,
            coords: vec![idx],
        },
        value: v,
    }
}

/// `out` (i32, rank 1) plus counter local 0 and an accumulator local 1.
fn k(body: Vec<Stmt>) -> KernelFn {
    KernelFn::new(
        "t",
        ElementKind::F32,
        vec![Param::buffer("out", I32, BufferRole::Write, Some(1))],
        vec![
            LocalDecl::scalar(I32),
            LocalDecl::scalar(I32),
            LocalDecl::scalar(I32),
        ],
        Block(body),
    )
}

fn for_loop(start: i64, end: i64, step: i64, body: Vec<Stmt>) -> Stmt {
    Stmt::For {
        local: 0,
        start: ci(start),
        end: ci(end),
        step: ci(step),
        body: Block(body),
    }
}

#[test]
fn a_counted_loop_visits_start_to_end_exclusive_by_step() {
    let kernel = k(vec![for_loop(1, 8, 3, vec![store_at(local(0), ci(9))])]);
    let mut args = vec![out_buf(8)];
    run_kernel(&kernel, &mut args).expect("runs");
    assert_eq!(read_i32(&args[0]), [0, 9, 0, 0, 9, 0, 0, 9]);
}

#[test]
fn a_negative_step_counts_down_and_stops_above_end() {
    let kernel = k(vec![for_loop(6, 1, -2, vec![store_at(local(0), ci(5))])]);
    let mut args = vec![out_buf(8)];
    run_kernel(&kernel, &mut args).expect("runs");
    assert_eq!(read_i32(&args[0]), [0, 0, 5, 0, 5, 0, 5, 0]);
}

#[test]
fn break_continue_and_return_do_what_they_say() {
    // i = 0..8: skip odd, stop after storing at 4.
    let body = vec![
        Stmt::If {
            cond: ScalarExpr::Sub(Box::new(local(0)), Box::new(ci(4))),
            then: Block(vec![Stmt::If {
                // odd? (i * 1) - 2*(i/2) is not expressible without division: use a table of
                // known indices instead.
                cond: ScalarExpr::Sub(Box::new(local(0)), Box::new(ci(1))),
                then: Block(vec![]),
                els: Block(vec![Stmt::Continue]),
            }]),
            els: Block(vec![store_at(local(0), ci(7)), Stmt::Break]),
        },
        store_at(local(0), ci(3)),
    ];
    let kernel = k(vec![for_loop(0, 8, 1, body)]);
    let mut args = vec![out_buf(8)];
    run_kernel(&kernel, &mut args).expect("runs");
    // i=0: cond != 0 -> inner cond (0-1) != 0 -> then (empty) -> store 3; i=1: inner cond 0 -> Continue
    // (no store); i=2,3: store 3; i=4: outer cond 0 -> else: store 7, Break.
    assert_eq!(read_i32(&args[0]), [3, 0, 3, 3, 7, 0, 0, 0]);

    // Return leaves the whole kernel, even from inside the loop.
    let early = k(vec![
        for_loop(0, 8, 1, vec![store_at(local(0), ci(1)), Stmt::Return]),
        store_at(ci(7), ci(9)),
    ]);
    let mut args = vec![out_buf(8)];
    run_kernel(&early, &mut args).expect("runs");
    assert_eq!(read_i32(&args[0]), [1, 0, 0, 0, 0, 0, 0, 0]);
}

#[test]
fn bounds_are_read_once_before_the_first_pass() {
    // The loop's end is local 1; the body assigns local 1 = 0. Reading it each pass would stop
    // after one iteration; reading once runs all four.
    let body = vec![
        Stmt::Assign {
            local: 1,
            value: ci(0),
        },
        store_at(local(0), ci(1)),
    ];
    let kernel = k(vec![
        Stmt::Let {
            local: 1,
            init: ci(4),
        },
        Stmt::For {
            local: 0,
            start: ci(0),
            end: local(1),
            step: ci(1),
            body: Block(body),
        },
    ]);
    let mut args = vec![out_buf(8)];
    run_kernel(&kernel, &mut args).expect("runs");
    assert_eq!(read_i32(&args[0]), [1, 1, 1, 1, 0, 0, 0, 0]);
}

#[test]
fn a_loop_ending_at_the_top_of_the_type_range_terminates() {
    let kernel = k(vec![Stmt::For {
        local: 0,
        start: ci(i64::from(i32::MAX) - 1),
        end: ci(i64::from(i32::MAX)),
        step: ci(3),
        body: Block(vec![Stmt::Let {
            local: 2,
            init: ci(1),
        }]),
    }]);
    // One iteration at MAX-1 (< MAX); MAX-1+3 leaves i32 and the loop ends. A counter that wrapped
    // to a negative value would be below the end again and spin until the step limit.
    let mut args = vec![out_buf(1)];
    assert_eq!(run_kernel(&kernel, &mut args), Ok(()));
}

#[test]
fn traps_are_typed_not_panics() {
    // Zero step.
    let zero = k(vec![for_loop(0, 4, 0, vec![])]);
    let mut args = vec![out_buf(4)];
    assert_eq!(run_kernel(&zero, &mut args), Err(KernelTrap::ZeroStep));
    // Out of bounds.
    let oob = k(vec![store_at(ci(9), ci(1))]);
    let mut args = vec![out_buf(4)];
    assert_eq!(
        run_kernel(&oob, &mut args),
        Err(KernelTrap::OutOfBounds { buffer: 0 })
    );
    // Control: the in-bounds store passes.
    let ok = k(vec![store_at(ci(3), ci(1))]);
    let mut args = vec![out_buf(4)];
    assert_eq!(run_kernel(&ok, &mut args), Ok(()));
    // A runaway While is stopped by the step limit.
    let spin = k(vec![Stmt::While {
        cond: ci(1),
        body: Block(vec![]),
    }]);
    let mut args = vec![out_buf(4)];
    assert_eq!(run_kernel(&spin, &mut args), Err(KernelTrap::StepLimit));
    // A malformed kernel is refused before it runs.
    let bad = k(vec![Stmt::Break]);
    let mut args = vec![out_buf(4)];
    assert!(matches!(
        run_kernel(&bad, &mut args),
        Err(KernelTrap::Malformed(_))
    ));
    // Wrong argument count.
    let mut none: Vec<KernelArg> = vec![];
    assert!(matches!(
        run_kernel(&ok, &mut none),
        Err(KernelTrap::ArgMismatch(_))
    ));
}
