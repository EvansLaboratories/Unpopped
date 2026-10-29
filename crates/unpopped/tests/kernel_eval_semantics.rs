// SPDX-License-Identifier: MIT OR Apache-2.0
//! The kernel interpreter's own semantics (`docs/ir-kernel-layer.md` section 5), on hand-built
//! kernels: loop bounds, `Break` / `Continue` / `Return` in both loop forms, integer wrap in
//! control flow, and typed traps. The differential against `oracle::evaluate` is in
//! `kernel_layers_agree.rs`.

use unpopped::ir::ScalarExpr;
use unpopped::kernel::{Block, BufferRole, KernelFn, LocalDecl, Param, Place, Stmt};
use unpopped::oracle::TypedBuffer;
use unpopped::oracle::kernel_eval::{KernelArg, KernelTrap, run_kernel};
use unpopped_vocab::ElementKind;

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

#[test]
fn integer_bounds_wrap_to_the_type_before_the_loop_runs() {
    // end = i32::MAX + 1 wraps to i32::MIN, so `0 .. MIN` runs nothing. Unwrapped it would be a
    // 2^31-pass loop and hit the step limit. (Stores mask to the width, so a wrap error is
    // invisible in an output element; it is observable only through control flow.)
    let wrapped_end = ScalarExpr::Add(Box::new(ci(i64::from(i32::MAX))), Box::new(ci(1)));
    let kernel = k(vec![Stmt::For {
        local: 0,
        start: ci(0),
        end: wrapped_end,
        step: ci(1),
        body: Block(vec![store_at(ci(0), ci(5))]),
    }]);
    let mut args = vec![out_buf(2)];
    assert_eq!(run_kernel(&kernel, &mut args), Ok(()));
    assert_eq!(read_i32(&args[0]), [0, 0], "the loop must not have run");
    // Control: the same loop with an end of 1 runs once.
    let once = k(vec![for_loop(0, 1, 1, vec![store_at(ci(0), ci(5))])]);
    let mut args = vec![out_buf(2)];
    run_kernel(&once, &mut args).expect("runs");
    assert_eq!(read_i32(&args[0]), [5, 0]);
}

#[test]
fn break_and_continue_work_inside_a_while_loop() {
    // A While that stores and breaks on its first pass terminates; without the Break it would hit
    // the step limit (the condition is constant true).
    let kernel = k(vec![Stmt::While {
        cond: ci(1),
        body: Block(vec![store_at(ci(0), ci(1)), Stmt::Break]),
    }]);
    let mut args = vec![out_buf(2)];
    assert_eq!(run_kernel(&kernel, &mut args), Ok(()));
    assert_eq!(read_i32(&args[0]), [1, 0]);

    // Continue skips the rest of the pass: count 0..3 in local 1, store only when it is not 1.
    let counted = k(vec![
        Stmt::Let {
            local: 1,
            init: ci(0),
        },
        Stmt::While {
            cond: ScalarExpr::Sub(Box::new(ci(3)), Box::new(local(1))),
            body: Block(vec![
                Stmt::Assign {
                    local: 1,
                    value: ScalarExpr::Add(Box::new(local(1)), Box::new(ci(1))),
                },
                Stmt::If {
                    cond: ScalarExpr::Sub(Box::new(local(1)), Box::new(ci(2))),
                    then: Block(vec![]),
                    els: Block(vec![Stmt::Continue]),
                },
                store_at(ci(0), local(1)),
            ]),
        },
    ]);
    let mut args = vec![out_buf(2)];
    assert_eq!(run_kernel(&counted, &mut args), Ok(()));
    // Passes see local 1 = 1, 2 (skipped by Continue), 3: the last store wins with 3 (never 2).
    assert_eq!(read_i32(&args[0]), [3, 0]);
}
