// SPDX-License-Identifier: MIT OR Apache-2.0
//! **The emitted C agrees with the kernel interpreter** (`docs/e1-emit-from-the-kernel-layer.md`,
//! stage E1a): a kernel is lowered by `CpuC::lower_kernel`, compiled with a real host C compiler,
//! run, and its output buffers compared **byte for byte** (bit-exact except NaN sign/payload, see `same_bytes`) with `oracle::kernel_eval::run_kernel`
//! on the same arguments. The traps (`OutOfBounds`, `ZeroStep`) must agree too.
//!
//! The serial C is a correctness reference. Nothing here times anything.

use unpopped::backend::{Backend, LowerError};
use unpopped::ir::{BinaryOp, Expr, OpDef, ScalarExpr, UnaryOp, View, input};
use unpopped::kernel::{Block, BufferRole, KernelFn, LocalDecl, Param, Place, Stmt};
use unpopped::kernel_lift::lift_elementwise_opdef;
use unpopped::oracle::TypedBuffer;
use unpopped::oracle::kernel_eval::KernelArg;
use unpopped_cpu_c::CpuC;
use unpopped_vocab::ElementKind;

#[path = "support/cc.rs"]
mod cc;
#[path = "support/kernel_data.rs"]
mod kernel_data;
#[path = "support/kernel_run.rs"]
mod kernel_run;
use kernel_data::{dense, zeroed};
use kernel_run::{F32, F64, agree, agree_within, run_c};

const I32: ElementKind = ElementKind::I32;

fn elementwise(name: &str, n_in: u8, dtype: ElementKind, body: ScalarExpr) -> OpDef {
    OpDef::elementwise(name, n_in, &[dtype], Expr(body))
}

fn bx(e: ScalarExpr) -> Box<ScalarExpr> {
    Box::new(e)
}

fn c(v: f64) -> ScalarExpr {
    ScalarExpr::Const(v)
}

/// The lifted kernel's arguments: inputs, a zeroed output, one `I32` extent per axis.
fn ew_args(dtype: ElementKind, n_in: usize, shape: &[i64]) -> Vec<KernelArg> {
    let mut a: Vec<KernelArg> = (0..n_in)
        .map(|i| KernelArg::Buffer(dense(dtype, shape, i as u32 + 1)))
        .collect();
    a.push(KernelArg::Buffer(zeroed(dtype, shape)));
    a.extend(shape.iter().map(|&e| KernelArg::Int(i128::from(e))));
    a
}

#[test]
fn elementwise_binary_with_constants_agrees_f32_and_f64() {
    // `x * 0.1 + y * 0.1`: three roundings. F64 is byte-exact with the interpreter; F32 is within
    // 3 ulps of it (single rounding at f64 versus one rounding per operation).
    for dtype in [F32, F64] {
        // Constants reach an emitter already rounded to the compute precision (the optimizer's
        // contract), so the F32 literal is the f32 nearest 0.1, not the double.
        let tenth = if dtype == F32 {
            f64::from(0.1_f32)
        } else {
            0.1
        };
        // `Abs` keeps both addends positive: with a cancelling sum a ulp-of-the-result bound
        // would be unbounded and the comparison would measure the data, not the emitter.
        let abs = |e| ScalarExpr::Unary(UnaryOp::Abs, bx(e));
        let body = ScalarExpr::Add(
            bx(ScalarExpr::Mul(bx(abs(input(0).0)), bx(c(tenth)))),
            bx(ScalarExpr::Mul(bx(abs(input(1).0)), bx(c(tenth)))),
        );
        let k = lift_elementwise_opdef(&elementwise("consts", 2, dtype, body), dtype, 2).unwrap();
        agree_within(
            &format!("consts_{dtype:?}"),
            &k,
            ew_args(dtype, 2, &[5, 7]),
            3,
        );
    }
}

#[test]
fn f32_arithmetic_rounds_every_operation_like_float() {
    // The interpreter cannot see this: a C literal left a `double` evaluates the whole
    // expression at double and rounds once, which sits inside the interpreter's tolerance.
    // So the emitted C is also held to Rust's own per-operation `f32` arithmetic, exactly.
    let tenth = 0.1_f32;
    let body = ScalarExpr::Add(
        bx(ScalarExpr::Mul(bx(input(0).0), bx(c(f64::from(tenth))))),
        bx(ScalarExpr::Mul(bx(input(1).0), bx(c(f64::from(tenth))))),
    );
    let k = lift_elementwise_opdef(&elementwise("rne", 2, F32, body), F32, 1).unwrap();
    let args = ew_args(F32, 2, &[64]);
    let Some(ran) = run_c("per_op_rounding", &k, &args) else {
        return;
    };
    let f = |i: usize| match &args[i] {
        KernelArg::Buffer(b) => b
            .bytes
            .chunks(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
            .collect::<Vec<_>>(),
        _ => unreachable!(),
    };
    let want: Vec<u8> = f(0)
        .iter()
        .zip(f(1))
        .flat_map(|(x, y)| ((x * tenth) + (y * tenth)).to_le_bytes())
        .collect();
    assert_eq!(
        ran.outs[0], want,
        "the emitted C does not round per operation"
    );
}

#[test]
fn elementwise_unary_and_select_ops_agree() {
    let body = ScalarExpr::Binary(
        BinaryOp::Max,
        bx(ScalarExpr::Unary(
            UnaryOp::Sqrt,
            bx(ScalarExpr::Unary(UnaryOp::Abs, bx(input(0).0))),
        )),
        bx(ScalarExpr::Unary(UnaryOp::Neg, bx(input(1).0))),
    );
    let k = lift_elementwise_opdef(&elementwise("un", 2, F32, body), F32, 1).unwrap();
    agree("unary", &k, ew_args(F32, 2, &[33]));
}

#[test]
fn a_runtime_base_offset_and_a_flipped_stride_agree() {
    // Input 0 sits in junk padding (offset 3) and input 1 is read backwards (stride -1): both are
    // the buffer's own layout, which the emitted address arithmetic must honour.
    let k = lift_elementwise_opdef(
        &elementwise(
            "off",
            2,
            F32,
            ScalarExpr::Sub(
                bx(input(0).0),
                bx(ScalarExpr::Mul(bx(input(1).0), bx(c(2.0)))),
            ),
        ),
        F32,
        1,
    )
    .unwrap();
    let mut args = ew_args(F32, 2, &[5]);
    let padded = |mut b: TypedBuffer, pad: usize| {
        b.bytes = std::iter::repeat_n(0x7f_u8, pad * 4)
            .chain(b.bytes)
            .collect();
        b
    };
    let KernelArg::Buffer(a0) = args[0].clone() else {
        unreachable!()
    };
    let mut a0 = padded(a0, 3);
    a0.base_offset = 3;
    args[0] = KernelArg::Buffer(a0);
    let KernelArg::Buffer(a1) = args[1].clone() else {
        unreachable!()
    };
    let mut a1 = a1;
    a1.strides = vec![-1];
    a1.base_offset = 4;
    args[1] = KernelArg::Buffer(a1);
    agree("offset_and_flip", &k, args);
}

#[test]
fn a_permuted_view_agrees() {
    // Input 1 is read through `Permute`: the producer coordinates swap, so a wrong axis order
    // in the emitted address arithmetic reads another element.
    let mut op = elementwise(
        "perm",
        2,
        F32,
        ScalarExpr::Sub(
            bx(input(0).0),
            bx(ScalarExpr::Mul(bx(input(1).0), bx(c(2.0)))),
        ),
    );
    op.views = vec![View::Identity, View::Permute { perm: vec![1, 0] }];
    let k = lift_elementwise_opdef(&op, F32, 2).unwrap();
    let mut args = ew_args(F32, 2, &[4, 4]);
    args[1] = KernelArg::Buffer(dense(F32, &[4, 4], 9));
    agree("permute", &k, args);
}

/// A hand-written kernel with every control construct: `For` with a negative step, nested loops
/// with `Break` and `Continue`, a `While`, an `If` and an early `Return`.
fn l(i: u32) -> ScalarExpr {
    ScalarExpr::Local(i)
}

fn ci(v: i64) -> ScalarExpr {
    ScalarExpr::ConstInt(v, I32)
}

fn cmp(op: BinaryOp, a: ScalarExpr, b: ScalarExpr) -> ScalarExpr {
    ScalarExpr::Binary(op, bx(a), bx(b))
}

/// `acc += v` on the float accumulator, local 2.
fn acc(v: f64) -> Stmt {
    Stmt::Assign {
        local: 2,
        value: ScalarExpr::Add(bx(l(2)), bx(c(v))),
    }
}

/// The nested loops: a descending outer `For` over a `Continue`/`Break` inner `For`.
fn nested_loops() -> Stmt {
    let inner = Stmt::For {
        local: 3,
        start: ci(0),
        end: ci(6),
        step: ci(1),
        body: Block(vec![
            Stmt::If {
                cond: cmp(BinaryOp::CmpEq, l(3), ci(1)),
                then: Block(vec![Stmt::Continue]),
                els: Block(vec![]),
            },
            Stmt::If {
                cond: cmp(BinaryOp::CmpEq, l(3), ci(3)),
                then: Block(vec![Stmt::Break]),
                els: Block(vec![]),
            },
            acc(1.5),
        ]),
    };
    Stmt::For {
        local: 4,
        start: l(0),
        end: ci(-1),
        step: ci(-1),
        body: Block(vec![inner, acc(10.25)]),
    }
}

/// The `If`/`else`, the `While`, the stores and the early `Return`.
fn tail_stmts() -> Vec<Stmt> {
    vec![
        Stmt::If {
            cond: cmp(BinaryOp::CmpLt, l(0), ci(3)),
            then: Block(vec![acc(1000.0)]),
            els: Block(vec![acc(2000.0)]),
        },
        Stmt::While {
            cond: cmp(BinaryOp::CmpLt, l(5), ci(3)),
            body: Block(vec![
                Stmt::Assign {
                    local: 5,
                    value: ScalarExpr::Add(bx(l(5)), bx(ci(1))),
                },
                acc(100.0),
            ]),
        },
        Stmt::Store {
            place: Place::Logical {
                buffer: 0,
                coords: vec![ci(0)],
            },
            value: l(2),
        },
        Stmt::If {
            cond: cmp(BinaryOp::CmpEq, l(0), ci(2)),
            then: Block(vec![Stmt::Return]),
            els: Block(vec![]),
        },
        Stmt::Store {
            place: Place::Logical {
                buffer: 0,
                coords: vec![ci(1)],
            },
            value: l(2),
        },
    ]
}

fn control_flow() -> KernelFn {
    let mut body = vec![
        Stmt::Let {
            local: 2,
            init: c(0.0),
        },
        Stmt::Let {
            local: 5,
            init: ci(0),
        },
        nested_loops(),
    ];
    body.extend(tail_stmts());
    KernelFn::new(
        "control",
        F32,
        vec![
            Param::buffer("out", F32, BufferRole::ReadWrite, Some(1)),
            Param::scalar("n", I32, 0),
        ],
        vec![
            LocalDecl::scalar(I32),
            LocalDecl::scalar(I32),
            LocalDecl::scalar(F32),
            LocalDecl::scalar(I32),
            LocalDecl::scalar(I32),
            LocalDecl::scalar(I32),
        ],
        Block(body),
    )
}

#[test]
fn control_flow_agrees_over_many_extents() {
    for n in [0, 1, 2, 3, 4, 7] {
        let args = vec![KernelArg::Buffer(zeroed(F32, &[2])), KernelArg::Int(n)];
        agree(&format!("control_{n}"), &control_flow(), args);
    }
}

#[test]
fn an_out_of_bounds_access_traps_the_same_way() {
    let oob = KernelFn::new(
        "oob",
        F32,
        vec![
            Param::buffer("out", F32, BufferRole::Write, Some(1)),
            Param::scalar("i", I32, 0),
        ],
        vec![LocalDecl::scalar(I32)],
        Block(vec![Stmt::Store {
            place: Place::Logical {
                buffer: 0,
                coords: vec![ScalarExpr::Local(0)],
            },
            value: c(1.0),
        }]),
    );
    for i in [-1, 0, 1, 2, 3] {
        agree(
            &format!("oob_{i}"),
            &oob,
            vec![KernelArg::Buffer(zeroed(F32, &[3])), KernelArg::Int(i)],
        );
    }
}

#[test]
fn a_zero_step_traps_the_same_way() {
    let zero = KernelFn::new(
        "zero",
        F32,
        vec![
            Param::buffer("out", F32, BufferRole::Write, Some(1)),
            Param::scalar("s", I32, 0),
        ],
        vec![LocalDecl::scalar(I32), LocalDecl::scalar(I32)],
        Block(vec![Stmt::For {
            local: 1,
            start: ScalarExpr::ConstInt(0, I32),
            end: ScalarExpr::ConstInt(3, I32),
            step: ScalarExpr::Local(0),
            body: Block(vec![]),
        }]),
    );
    agree(
        "zero_step",
        &zero,
        vec![KernelArg::Buffer(zeroed(F32, &[1])), KernelArg::Int(0)],
    );
    agree(
        "one_step",
        &zero,
        vec![KernelArg::Buffer(zeroed(F32, &[1])), KernelArg::Int(1)],
    );
}

#[test]
fn a_kernel_outside_the_emitters_coverage_is_the_typed_kernel_decline() {
    use unpopped::exec::{LaunchShape, Scope};
    let base = control_flow();
    let launched = base
        .clone()
        .with_launch(LaunchShape::new([1, 1, 1], [1, 1, 1]));
    let barrier = {
        let mut k = launched.clone();
        k.body.0.insert(
            0,
            Stmt::Barrier {
                scope: Scope::Block,
            },
        );
        k
    };
    let f16 = KernelFn::new("h", ElementKind::F16, vec![], vec![], Block(vec![]));
    let ill_formed = KernelFn::new("bad", F32, vec![], vec![], Block(vec![Stmt::Break]));
    for (what, k) in [
        ("launch", launched),
        ("barrier", barrier),
        ("f16", f16),
        ("ill-formed", ill_formed),
    ] {
        let e = CpuC.lower_kernel(&k).expect_err(what);
        assert!(
            matches!(e, LowerError::UnsupportedKernel { .. }),
            "{what}: {e:?}"
        );
    }
}
