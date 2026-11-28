// SPDX-License-Identifier: MIT OR Apache-2.0
//! What `check_kernel` does with an `F128` (ROADMAP E3): every node and statement the oracle does not compute over a binary128 is
//! refused by name, so the oracle never meets one (the F6 lesson: an oracle must not panic on a kernel the checker accepted); every
//! one it does compute runs. Each refusal sits beside the accepted control.

use unpopped::exec::{ExecId, LaunchShape, Scope};
use unpopped::ir::{BinaryOp, CastSpec, RoundMode, Saturation, ScalarExpr, UnaryOp};
use unpopped::kernel::{
    AtomicOp, Block, BufferRole, CollectiveKind, KernelError, KernelFn, LocalDecl, MemoryOrder,
    Param, Place, ShuffleMode, Stmt, check_kernel,
};
use unpopped::oracle::TypedBuffer;
use unpopped::oracle::kernel_eval::KernelArg;
use unpopped::oracle::kernel_launch::run_launch;
use unpopped::types::TypeError;
use unpopped_vocab::ElementKind::{self, F16, F32, F64, F128, I32, U32, U128};

const ONE: u128 = 0x3fff << 112;

fn at(buffer: u32) -> Place {
    Place::Logical {
        buffer,
        coords: vec![ScalarExpr::Exec(ExecId::ThreadIdx(0))],
    }
}

fn bx(e: ScalarExpr) -> Box<ScalarExpr> {
    Box::new(e)
}

fn x() -> ScalarExpr {
    ScalarExpr::Local(0)
}

fn cast(e: ScalarExpr, to: ElementKind, round: RoundMode, sat: Saturation) -> ScalarExpr {
    ScalarExpr::Cast(bx(e), CastSpec { to, round, sat })
}

/// A 4-lane kernel (subgroup of 4) holding `x = a[t]` (`F128`) in local 0 and running `body`; buffer 1 is an `F128` read-write scratch.
fn kernel(body: Vec<Stmt>, locals: Vec<LocalDecl>) -> KernelFn {
    let mut stmts = vec![Stmt::Load {
        local: 0,
        place: at(0),
    }];
    stmts.extend(body);
    let mut ls = vec![LocalDecl::scalar(F128)];
    ls.extend(locals);
    KernelFn::new(
        "f128",
        F32,
        vec![
            Param::buffer("a", F128, BufferRole::ReadWrite, Some(1)),
            Param::buffer("out", F128, BufferRole::Write, Some(1)),
        ],
        ls,
        Block(stmts),
    )
    .with_launch(LaunchShape::new([4, 1, 1], [1, 1, 1]).with_subgroup(4, 4))
}

fn store_out(value: ScalarExpr) -> Stmt {
    Stmt::Store {
        place: at(1),
        value,
    }
}

fn refused_as_f128_operation(k: &KernelFn, node: &str) {
    match check_kernel(k) {
        Err(KernelError::Type(TypeError::F128Operation { node: n })) => assert_eq!(n, node),
        other => panic!("expected F128Operation({node}), got {other:?}"),
    }
}

#[test]
fn the_nodes_the_oracle_does_not_compute_over_an_f128_are_refused_by_name() {
    let control = kernel(vec![store_out(ScalarExpr::Add(bx(x()), bx(x())))], vec![]);
    assert_eq!(check_kernel(&control), Ok(()), "the control: an addition");
    let refuse = |value: ScalarExpr, node: &str| {
        refused_as_f128_operation(&kernel(vec![store_out(value)], vec![]), node);
    };
    refuse(ScalarExpr::Fma(bx(x()), bx(x()), bx(x())), "Fma");
    refuse(ScalarExpr::Unary(UnaryOp::Sqrt, bx(x())), "Unary");
    refuse(
        ScalarExpr::Binary(BinaryOp::Max, bx(x()), bx(x())),
        "Binary",
    );
    refuse(
        ScalarExpr::Binary(BinaryOp::Pow, bx(x()), bx(x())),
        "Binary",
    );
    refuse(
        ScalarExpr::Qualified(bx(ScalarExpr::Add(bx(x()), bx(x()))), Default::default()),
        "Qualified",
    );
    refuse(ScalarExpr::Bitcast(bx(x()), U128), "Bitcast");
    // a Select over F32 arms with an F128 condition
    let sel = kernel(
        vec![Stmt::Let {
            local: 1,
            init: ScalarExpr::Select(
                bx(x()),
                bx(ScalarExpr::Const(1.0)),
                bx(ScalarExpr::Const(2.0)),
            ),
        }],
        vec![LocalDecl::scalar(F32)],
    );
    refused_as_f128_operation(&sel, "Select");
}

#[test]
fn a_cast_of_an_f128_outside_f32_f64_and_the_integers_is_refused_by_name() {
    let to = |t: ElementKind, round: RoundMode, sat: Saturation| {
        kernel(
            vec![Stmt::Let {
                local: 1,
                init: cast(x(), t, round, sat),
            }],
            vec![LocalDecl::scalar(t)],
        )
    };
    let n = RoundMode::NearestEven;
    for (t, round, sat, ok) in [
        (F32, n, Saturation::Wrap, true),
        (F32, n, Saturation::ClampFinite, true),
        (F64, n, Saturation::Wrap, true),
        (I32, RoundMode::TowardZero, Saturation::Clamp, true),
        (F32, RoundMode::TowardZero, Saturation::Wrap, false),
        (F64, RoundMode::TowardPosInf, Saturation::Wrap, false),
        (I32, n, Saturation::Wrap, false),
        (F16, n, Saturation::Wrap, false),
        (U128, n, Saturation::Clamp, false),
    ] {
        let r = check_kernel(&to(t, round, sat));
        if ok {
            assert_eq!(r, Ok(()), "{t:?} {round:?} {sat:?}");
        } else {
            assert!(
                matches!(r, Err(KernelError::Type(TypeError::IllegalCast { .. }))),
                "{t:?} {round:?} {sat:?}: {r:?}"
            );
        }
    }
}

#[test]
fn a_collective_or_an_atomic_over_an_f128_is_refused_never_computed_in_f64() {
    let collective = |kind: CollectiveKind| {
        kernel(
            vec![Stmt::Collective {
                kind,
                value: x(),
                result: 1,
                scope: Scope::Subgroup,
            }],
            vec![LocalDecl::scalar(F128)],
        )
    };
    for kind in [
        CollectiveKind::Reduce(unpopped::ir::ReduceOp::Sum),
        CollectiveKind::Reduce(unpopped::ir::ReduceOp::Max),
        CollectiveKind::Shuffle {
            mode: ShuffleMode::Idx,
            operand: ScalarExpr::ConstInt(0, U32),
        },
    ] {
        let r = check_kernel(&collective(kind.clone()));
        assert!(r.is_err(), "{kind:?}: {r:?}");
    }
    let atomic = kernel(
        vec![Stmt::AtomicRmw {
            op: AtomicOp::Add,
            place: at(0),
            value: x(),
            old: None,
            order: MemoryOrder::Relaxed,
            scope: Scope::Block,
        }],
        vec![],
    );
    assert!(
        check_kernel(&atomic).is_err(),
        "{:?}",
        check_kernel(&atomic)
    );
}

#[test]
fn an_f128_condition_drives_an_if_and_a_masked_store_and_runs() {
    // `if x { out[t] = one }` and `out[t] = x != 0 ? ...`: a nonzero F128 (a NaN included) is taken, +0 and -0 are not
    let one = || {
        cast(
            ScalarExpr::Const(1.0),
            F128,
            RoundMode::NearestEven,
            Saturation::Wrap,
        )
    };
    let k = kernel(
        vec![
            store_out(one().clone()),
            Stmt::If {
                cond: x(),
                then: Block(vec![store_out(ScalarExpr::Sub(bx(one()), bx(one())))]),
                els: Block(vec![]),
            },
        ],
        vec![],
    );
    assert_eq!(check_kernel(&k), Ok(()));
    let a: [u128; 4] = [0, 1 << 127, ONE, 0x7fff_8000_0000_0000_0000_0000_0000_0000];
    let bytes: Vec<u8> = a.iter().flat_map(|v| v.to_le_bytes()).collect();
    let mut args = vec![
        KernelArg::Buffer(TypedBuffer::new(F128, vec![4], vec![1], bytes)),
        KernelArg::Buffer(TypedBuffer::new(F128, vec![4], vec![1], vec![0; 64])),
    ];
    run_launch(&k, &mut args).expect("runs");
    let KernelArg::Buffer(out) = &args[1] else {
        unreachable!()
    };
    let got: Vec<u128> = out
        .bytes
        .chunks(16)
        .map(|c| u128::from_le_bytes(c.try_into().unwrap()))
        .collect();
    // taken (x != 0): out = 1 - 1 = +0; not taken (x is +0 or -0): out stays 1
    assert_eq!(got, vec![ONE, ONE, 0, 0]);
}
