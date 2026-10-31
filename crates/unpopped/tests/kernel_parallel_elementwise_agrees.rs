// SPDX-License-Identifier: MIT OR Apache-2.0
//! **E1d-0: a grid-stride Elementwise launch computes what the serial kernel computes**
//! (`docs/e1d-parallel-mapping.md`).
//!
//! `lift_elementwise_opdef_launched` turns the outermost output axis into a grid-stride loop with a
//! caller-chosen geometry. Whatever the grid and block sizes, `run_launch` must give the serial
//! kernel's bytes (itself proven equal to `oracle::evaluate` in `kernel_layers_agree`). The
//! geometry below includes launches that cover far fewer than `n` elements per pass (the stride
//! has to loop) and launches with more threads than elements (the surplus must do nothing).
//!
//! Two kernels built here from the launched one, each by breaking one thing, must **disagree** with
//! the serial result: an agreement test that cannot fail proves nothing.

use unpopped::exec::ExecId;
use unpopped::ir::{BinaryOp, Expr, OpDef, ScalarExpr, UnaryOp, input};
use unpopped::kernel::{KernelFn, Stmt, check_kernel};
use unpopped::kernel_lift::{LiftDecline, lift_elementwise_opdef, lift_elementwise_opdef_launched};
use unpopped::oracle::TypedBuffer;
use unpopped::oracle::kernel_eval::{KernelArg, KernelTrap, run_kernel};
use unpopped::oracle::kernel_launch::run_launch;
use unpopped_vocab::ElementKind;

const N: usize = 70;
const F32: ElementKind = ElementKind::F32;

fn bx(e: ScalarExpr) -> Box<ScalarExpr> {
    Box::new(e)
}

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
    (0..N).map(|i| base[(i * 7 + seed) % base.len()]).collect()
}

fn buffer(seed: usize) -> TypedBuffer {
    let bytes: Vec<u8> = f32_corpus(seed)
        .iter()
        .flat_map(|v| v.to_le_bytes())
        .collect();
    TypedBuffer::new(F32, vec![N as i64], vec![1], bytes)
}

fn args(n_inputs: usize) -> Vec<KernelArg> {
    let mut a: Vec<KernelArg> = (0..n_inputs)
        .map(|k| KernelArg::Buffer(buffer(k * 3)))
        .collect();
    a.push(KernelArg::Buffer(TypedBuffer::new(
        F32,
        vec![N as i64],
        vec![1],
        vec![0u8; N * 4],
    )));
    a.push(KernelArg::Int(N as i128));
    a
}

fn output(a: &[KernelArg], n_inputs: usize) -> Vec<u8> {
    let KernelArg::Buffer(out) = &a[n_inputs] else {
        unreachable!("the output buffer follows the inputs")
    };
    out.bytes.clone()
}

fn serial(k: &KernelFn, n_inputs: usize) -> Vec<u8> {
    let mut a = args(n_inputs);
    run_kernel(k, &mut a).expect("the serial kernel runs");
    output(&a, n_inputs)
}

fn try_launch(k: &KernelFn, n_inputs: usize) -> Result<Vec<u8>, KernelTrap> {
    let mut a = args(n_inputs);
    run_launch(k, &mut a)?;
    Ok(output(&a, n_inputs))
}

fn launched(k: &KernelFn, n_inputs: usize) -> Vec<u8> {
    try_launch(k, n_inputs).expect("the launch runs")
}

fn ops() -> Vec<(OpDef, usize)> {
    let add = ScalarExpr::Binary(BinaryOp::Max, bx(input(0).0), bx(input(1).0));
    let exp = ScalarExpr::Unary(UnaryOp::Exp, bx(input(0).0));
    let chain = ScalarExpr::Add(
        bx(ScalarExpr::Mul(bx(input(0).0), bx(input(1).0))),
        bx(ScalarExpr::Unary(UnaryOp::Neg, bx(input(0).0))),
    );
    vec![
        (OpDef::elementwise("max", 2, &[F32, F32], Expr(add)), 2),
        (OpDef::elementwise("exp", 1, &[F32], Expr(exp)), 1),
        (OpDef::elementwise("chain", 2, &[F32, F32], Expr(chain)), 2),
    ]
}

#[test]
fn every_geometry_gives_the_serial_bytes() {
    for (op, n_inputs) in ops() {
        let want = serial(
            &lift_elementwise_opdef(&op, F32, 1).expect("the op lifts"),
            n_inputs,
        );
        // grid x block: 1x1 (70 passes), 1x2, 3x8 (24 per pass), 7x32 (224 threads > 70), 1x256.
        for (grid, block) in [(1, 1), (1, 2), (3, 8), (7, 32), (1, 256), (7, 1)] {
            let k = lift_elementwise_opdef_launched(&op, F32, 1, block, grid).expect("lifts");
            assert_eq!(check_kernel(&k), Ok(()), "{} well formed", op.name);
            assert_eq!(
                launched(&k, n_inputs),
                want,
                "{} on grid {grid} x block {block}",
                op.name
            );
        }
    }
}

/// The two controls: break the stride (the launch gives a wrong answer), then break the thread's
/// start (the launch is refused as a data race); neither may still agree. If either still agreed, the test above could not tell a working transform from a
/// broken one.
#[test]
fn a_launch_with_its_stride_or_its_start_broken_disagrees() {
    let (op, n_inputs) = ops().remove(0);
    let want = serial(
        &lift_elementwise_opdef(&op, F32, 1).expect("the op lifts"),
        n_inputs,
    );
    let (grid, block) = (3, 8); // 24 threads, 70 elements: needs the stride, needs distinct starts

    let mut no_stride = lift_elementwise_opdef_launched(&op, F32, 1, block, grid).expect("lifts");
    let Some(Stmt::For { step, .. }) = no_stride.body.0.first_mut() else {
        panic!("the outer statement is a loop")
    };
    *step = ScalarExpr::ConstInt(1 << 20, ElementKind::I32);
    assert_ne!(
        launched(&no_stride, n_inputs),
        want,
        "one element per thread"
    );

    let mut same_start = lift_elementwise_opdef_launched(&op, F32, 1, block, grid).expect("lifts");
    let Some(Stmt::For { start, .. }) = same_start.body.0.first_mut() else {
        panic!("the outer statement is a loop")
    };
    *start = ScalarExpr::Cast(
        bx(ScalarExpr::Exec(ExecId::ThreadIdx(0))),
        unpopped::ir::CastSpec {
            to: ElementKind::I32,
            round: unpopped::ir::RoundMode::NearestEven,
            sat: unpopped::ir::Saturation::Wrap,
        },
    );
    // Every block repeating block 0's work writes the same elements from several blocks: the
    // interpreter's race check refuses the launch, which is a stronger failure than a wrong answer.
    assert_eq!(
        try_launch(&same_start, n_inputs),
        Err(KernelTrap::DataRace),
        "every block repeats block 0's work"
    );
}

#[test]
fn a_zero_geometry_is_a_typed_decline_and_a_valid_one_is_not() {
    let (op, _) = ops().remove(1);
    assert!(matches!(
        lift_elementwise_opdef_launched(&op, F32, 1, 0, 4),
        Err(LiftDecline::BadLaunch)
    ));
    assert!(matches!(
        lift_elementwise_opdef_launched(&op, F32, 1, 32, 0),
        Err(LiftDecline::BadLaunch)
    ));
    assert!(lift_elementwise_opdef_launched(&op, F32, 1, 32, 4).is_ok());
}

/// The interpreter's integer `Cast` (which the grid-stride start and step use to turn `U32` ids into
/// the `I32` counter): a wrapping cast reinterprets, so `u32::MAX` becomes `-1`; a saturating one is
/// still outside the subset and refuses rather than guessing.
#[test]
fn an_integer_cast_wraps_and_a_saturating_one_refuses() {
    use unpopped::ir::{CastSpec, RoundMode, Saturation};
    use unpopped::kernel::{Block, BufferRole, LocalDecl, Param, Place};
    let spec = |sat| CastSpec {
        to: ElementKind::I32,
        round: RoundMode::NearestEven,
        sat,
    };
    let kernel = |sat| {
        KernelFn::new(
            "cast",
            ElementKind::I32,
            vec![Param::buffer(
                "out",
                ElementKind::I32,
                BufferRole::Write,
                Some(1),
            )],
            vec![
                LocalDecl::scalar(ElementKind::U32),
                LocalDecl::scalar(ElementKind::I32),
                LocalDecl::scalar(ElementKind::I32),
            ],
            Block(vec![
                Stmt::Let {
                    local: 0,
                    init: ScalarExpr::ConstInt(i64::from(u32::MAX), ElementKind::U32),
                },
                Stmt::Let {
                    local: 1,
                    init: ScalarExpr::Cast(bx(ScalarExpr::Local(0)), spec(sat)),
                },
                Stmt::Store {
                    place: Place::Logical {
                        buffer: 0,
                        coords: vec![ScalarExpr::ConstInt(0, ElementKind::I32)],
                    },
                    value: ScalarExpr::Local(1),
                },
                // A comparison reads the cast's value itself (a store would truncate an unwrapped
                // one to the same four bytes), so it is what tells a wrapped cast from a raw one.
                Stmt::Let {
                    local: 2,
                    init: ScalarExpr::Binary(
                        BinaryOp::CmpLt,
                        bx(ScalarExpr::Cast(bx(ScalarExpr::Local(0)), spec(sat))),
                        bx(ScalarExpr::ConstInt(0, ElementKind::I32)),
                    ),
                },
                Stmt::Store {
                    place: Place::Logical {
                        buffer: 0,
                        coords: vec![ScalarExpr::ConstInt(1, ElementKind::I32)],
                    },
                    value: ScalarExpr::Local(2),
                },
            ]),
        )
    };
    let run = |k: &KernelFn| {
        let mut a = vec![KernelArg::Buffer(TypedBuffer::new(
            ElementKind::I32,
            vec![2],
            vec![1],
            vec![0u8; 8],
        ))];
        run_kernel(k, &mut a).map(|()| {
            let KernelArg::Buffer(b) = &a[0] else {
                unreachable!()
            };
            let word = |i: usize| i32::from_le_bytes(b.bytes[i * 4..i * 4 + 4].try_into().unwrap());
            (word(0), word(1))
        })
    };
    assert_eq!(run(&kernel(Saturation::Wrap)), Ok((-1, 1)));
    assert!(matches!(
        run(&kernel(Saturation::Clamp)),
        Err(KernelTrap::Unsupported(_))
    ));
}
