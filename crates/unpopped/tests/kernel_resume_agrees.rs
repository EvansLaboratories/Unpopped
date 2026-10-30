// SPDX-License-Identifier: MIT OR Apache-2.0
//! U3 row 4 PR 3: the resumable evaluator behind `run_launch` agrees with the recursive one behind
//! `run_kernel` on every control construct (`docs/u3-row4-scopes-and-memory-spaces.md` section 5,
//! step 2). Two implementations of loop semantics exist on purpose (the lifting differential keeps
//! the recursive one); these tests are what keeps them from drifting.

use unpopped::exec::LaunchShape;
use unpopped::ir::{BinaryOp, ScalarExpr};
use unpopped::kernel::{Block, BufferRole, KernelFn, LocalDecl, Param, Place, Stmt};
use unpopped::oracle::TypedBuffer;
use unpopped::oracle::kernel_eval::{KernelArg, KernelTrap, run_kernel};
use unpopped::oracle::kernel_launch::run_launch;
use unpopped_vocab::ElementKind;

const F32: ElementKind = ElementKind::F32;
const I32: ElementKind = ElementKind::I32;

const N: u32 = 0; // scalar param n: i32

fn local(i: u32) -> ScalarExpr {
    ScalarExpr::Local(i)
}

fn ci(v: i64) -> ScalarExpr {
    ScalarExpr::ConstInt(v, I32)
}

fn add(a: ScalarExpr, b: ScalarExpr) -> ScalarExpr {
    ScalarExpr::Add(Box::new(a), Box::new(b))
}

fn buf(buffer: u32, c: ScalarExpr) -> Place {
    Place::Logical {
        buffer,
        coords: vec![c],
    }
}

fn args(n: usize) -> Vec<KernelArg> {
    let input: Vec<f32> = (0..n).map(|i| i as f32 + 1.0).collect();
    let zeros = vec![0.0; n];
    vec![
        KernelArg::Buffer(TypedBuffer::from_f32(&[n as i64], &input)),
        KernelArg::Buffer(TypedBuffer::from_f32(&[n as i64], &zeros)),
        KernelArg::Buffer(TypedBuffer::from_f32(&[n as i64], &zeros)),
        KernelArg::Int(n as i128),
    ]
}

fn read(a: &[KernelArg], which: usize) -> Vec<f32> {
    let KernelArg::Buffer(b) = &a[which] else {
        panic!("a buffer")
    };
    b.bytes
        .chunks(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect()
}

// ---- the two evaluators agree -----------------------------------------------------------------

fn cmp(op: BinaryOp, a: ScalarExpr, b: ScalarExpr) -> ScalarExpr {
    ScalarExpr::Binary(op, Box::new(a), Box::new(b))
}

const I: u32 = 1;
const J: u32 = 2;
const ACC: u32 = 3;
const K: u32 = 4;

fn add_f(v: f64) -> ScalarExpr {
    ScalarExpr::Add(Box::new(local(ACC)), Box::new(ScalarExpr::Const(v)))
}

fn inner_loop() -> Stmt {
    Stmt::For {
        local: J,
        start: ci(0),
        end: ci(6),
        step: ci(1),
        body: Block(vec![
            Stmt::If {
                cond: cmp(BinaryOp::CmpEq, local(J), ci(1)),
                then: Block(vec![Stmt::Continue]),
                els: Block(vec![]),
            },
            Stmt::If {
                cond: cmp(BinaryOp::CmpGt, local(J), ci(3)),
                then: Block(vec![Stmt::Break]),
                els: Block(vec![]),
            },
            Stmt::Assign {
                local: ACC,
                value: add_f(1.0),
            },
        ]),
    }
}

fn outer_loop() -> Stmt {
    Stmt::For {
        local: I,
        start: ci(0),
        end: local(N),
        step: ci(1),
        body: Block(vec![
            inner_loop(),
            Stmt::If {
                cond: cmp(BinaryOp::CmpEq, local(I), ci(3)),
                then: Block(vec![Stmt::Break]),
                els: Block(vec![]),
            },
            Stmt::Assign {
                local: ACC,
                value: add_f(10.0),
            },
        ]),
    }
}

/// The `While` loop, the stores and the early `Return`.
fn tail_stmts() -> Vec<Stmt> {
    vec![
        Stmt::While {
            cond: cmp(BinaryOp::CmpLt, local(K), ci(3)),
            body: Block(vec![
                Stmt::Assign {
                    local: K,
                    value: add(local(K), ci(1)),
                },
                Stmt::Assign {
                    local: ACC,
                    value: add_f(100.0),
                },
            ]),
        },
        Stmt::Store {
            place: buf(1, ci(0)),
            value: local(ACC),
        },
        Stmt::If {
            cond: cmp(BinaryOp::CmpEq, local(N), ci(7)),
            then: Block(vec![Stmt::Return]),
            els: Block(vec![]),
        },
        Stmt::Store {
            place: buf(1, ci(1)),
            value: local(ACC),
        },
    ]
}

/// Nested loops with `Continue`, `Break` and an early `Return`, plus a `While`: every control
/// construct the resumable evaluator re-implements.
fn control_flow() -> KernelFn {
    let mut body = vec![
        Stmt::Let {
            local: ACC,
            init: ScalarExpr::Const(0.0),
        },
        Stmt::Let {
            local: K,
            init: ci(0),
        },
        outer_loop(),
    ];
    body.extend(tail_stmts());
    KernelFn::new(
        "control",
        F32,
        vec![
            Param::buffer("in", F32, BufferRole::Read, Some(1)),
            Param::buffer("out", F32, BufferRole::ReadWrite, Some(1)),
            Param::buffer("aux", F32, BufferRole::Write, Some(1)),
            Param::scalar("n", I32, N),
        ],
        vec![
            LocalDecl::scalar(I32),
            LocalDecl::scalar(I32),
            LocalDecl::scalar(I32),
            LocalDecl::scalar(F32),
            LocalDecl::scalar(I32),
        ],
        Block(body),
    )
}

#[test]
fn the_two_evaluators_agree() {
    let plain = control_flow();
    let launched = plain
        .clone()
        .with_launch(LaunchShape::new([1, 1, 1], [1, 1, 1]));
    let mut distinct = std::collections::BTreeSet::new();
    for n in [0, 1, 3, 4, 5, 7] {
        let (mut a, mut b) = (args(2), args(2));
        a[3] = KernelArg::Int(n);
        b[3] = KernelArg::Int(n);
        run_kernel(&plain, &mut a).expect("recursive evaluator");
        run_launch(&launched, &mut b).expect("resumable evaluator");
        assert_eq!(read(&a, 1), read(&b, 1), "n = {n}");
        distinct.insert(read(&a, 1).iter().map(|v| v.to_bits()).collect::<Vec<_>>());
    }
    // The inputs reach different branches: break at i == 3, an early return at n == 7.
    assert!(distinct.len() >= 5, "{} distinct outcomes", distinct.len());
}

#[test]
fn a_zero_step_traps_in_both_evaluators() {
    let looped = |step: i64| {
        KernelFn::new(
            "zero",
            F32,
            vec![],
            vec![LocalDecl::scalar(I32)],
            Block(vec![Stmt::For {
                local: 0,
                start: ci(0),
                end: ci(2),
                step: ci(step),
                body: Block(vec![]),
            }]),
        )
    };
    let launched = |k: KernelFn| k.with_launch(LaunchShape::new([1, 1, 1], [1, 1, 1]));
    run_kernel(&looped(1), &mut []).expect("control: a step of 1");
    run_launch(&launched(looped(1)), &mut []).expect("control: a step of 1");
    assert_eq!(run_kernel(&looped(0), &mut []), Err(KernelTrap::ZeroStep));
    assert_eq!(
        run_launch(&launched(looped(0)), &mut []),
        Err(KernelTrap::ZeroStep)
    );
}
