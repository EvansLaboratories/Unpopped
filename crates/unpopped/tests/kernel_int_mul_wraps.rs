// SPDX-License-Identifier: MIT OR Apache-2.0
//! Integer `Mul` wraps at the type's width, including where the exact product of two 64-bit operands exceeds `i128`
//! (`u64::MAX * u64::MAX` overflowed the oracle's own `i128` and panicked in a debug build; found by unpopped-conformance).

use unpopped::exec::{ExecId, LaunchShape, Scope};
use unpopped::ir::{CastSpec, ReduceOp, RoundMode, Saturation, ScalarExpr};
use unpopped::kernel::{
    Block, BufferRole, CollectiveKind, KernelFn, LocalDecl, Param, Place, Stmt, check_kernel,
};
use unpopped::oracle::TypedBuffer;
use unpopped::oracle::kernel_eval::KernelArg;
use unpopped::oracle::kernel_launch::{run_launch, run_launch_with_subgroup};
use unpopped_vocab::ElementKind::{self, I64, U64};

fn mul(t: ElementKind, a: &[u64], b: &[u64]) -> Vec<u64> {
    let at = |buffer| Place::Logical {
        buffer,
        coords: vec![ScalarExpr::Exec(ExecId::ThreadIdx(0))],
    };
    let k = KernelFn::new(
        "mul",
        ElementKind::I32,
        vec![
            Param::buffer("a", t, BufferRole::Read, Some(1)),
            Param::buffer("b", t, BufferRole::Read, Some(1)),
            Param::buffer("out", t, BufferRole::Write, Some(1)),
        ],
        vec![LocalDecl::scalar(t), LocalDecl::scalar(t)],
        Block(vec![
            Stmt::Load {
                local: 0,
                place: at(0),
            },
            Stmt::Load {
                local: 1,
                place: at(1),
            },
            Stmt::Store {
                place: at(2),
                value: ScalarExpr::Mul(
                    Box::new(ScalarExpr::Local(0)),
                    Box::new(ScalarExpr::Local(1)),
                ),
            },
        ]),
    )
    .with_launch(LaunchShape::new([a.len() as u32, 1, 1], [1, 1, 1]));
    assert_eq!(check_kernel(&k), Ok(()));
    let n = a.len() as i64;
    let buf = |v: &[u64]| {
        KernelArg::Buffer(TypedBuffer::new(
            t,
            vec![n],
            vec![1],
            v.iter().flat_map(|x| x.to_le_bytes()).collect(),
        ))
    };
    let mut args = vec![buf(a), buf(b), buf(&vec![0; a.len()])];
    run_launch(&k, &mut args).unwrap();
    let KernelArg::Buffer(out) = &args[2] else {
        unreachable!()
    };
    out.bytes
        .chunks(8)
        .map(|c| u64::from_le_bytes(c.try_into().unwrap()))
        .collect()
}

#[test]
fn a_64_bit_product_beyond_i128_wraps() {
    let (a, b) = (
        [u64::MAX, u64::MAX, 1 << 63, 3, 1 << 63],
        [u64::MAX, 2, 2, u64::MAX, 1 << 63],
    );
    let want: Vec<u64> = a.iter().zip(&b).map(|(x, y)| x.wrapping_mul(*y)).collect();
    assert_eq!(mul(U64, &a, &b), want, "U64");
    assert_eq!(mul(I64, &a, &b), want, "I64 bit patterns");
    assert_eq!(want[0], 1, "(2^64 - 1)^2 is 1 modulo 2^64");
}

#[test]
fn a_64_bit_collective_product_wraps() {
    // four lanes each hold u64::MAX: the running product passes (2^64 - 1)^2, beyond i128, and the result is (-1)^4 = 1
    let tid = || ScalarExpr::Exec(ExecId::ThreadIdx(0));
    let max = ScalarExpr::Cast(
        Box::new(ScalarExpr::ConstInt(-1, I64)),
        CastSpec {
            to: U64,
            round: RoundMode::NearestEven,
            sat: Saturation::Wrap,
        },
    );
    let k = KernelFn::new(
        "prod",
        ElementKind::I32,
        vec![Param::buffer("out", U64, BufferRole::Write, Some(1))],
        vec![LocalDecl::scalar(U64), LocalDecl::scalar(U64)],
        Block(vec![
            Stmt::Let {
                local: 0,
                init: max,
            },
            Stmt::Collective {
                kind: CollectiveKind::Reduce(ReduceOp::Prod),
                value: ScalarExpr::Local(0),
                result: 1,
                scope: Scope::Subgroup,
            },
            Stmt::Store {
                place: Place::Logical {
                    buffer: 0,
                    coords: vec![tid()],
                },
                value: ScalarExpr::Local(1),
            },
        ]),
    )
    .with_launch(LaunchShape::new([4, 1, 1], [1, 1, 1]).with_subgroup(1, 32));
    assert_eq!(check_kernel(&k), Ok(()));
    let mut args = vec![KernelArg::Buffer(TypedBuffer::new(
        U64,
        vec![4],
        vec![1],
        vec![0u8; 32],
    ))];
    run_launch_with_subgroup(&k, &mut args, Some(4)).unwrap();
    let KernelArg::Buffer(out) = &args[0] else {
        unreachable!()
    };
    let got: Vec<u64> = out
        .bytes
        .chunks(8)
        .map(|c| u64::from_le_bytes(c.try_into().unwrap()))
        .collect();
    assert_eq!(got, vec![1; 4]);
}
