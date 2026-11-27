// SPDX-License-Identifier: MIT OR Apache-2.0
//! **`tl.where` on integers** (Triton survey, "Select on integer dtypes"): an integer `Select` between two loaded
//! integers, with the condition a comparison and with the condition an integer MASK loaded from a `U8` buffer (tested
//! `!= 0`), at `I32`, `I64`, `U8` and `U32`. This is the index-masking idiom (`where(i < n, i, 0)`). Checked against a
//! host loop; the swapped-arms mutant must change the value.

use unpopped::exec::{ExecId, LaunchShape};
use unpopped::ir::{BinaryOp, ScalarExpr};
use unpopped::kernel::{Block, BufferRole, KernelFn, LocalDecl, Param, Place, Stmt, check_kernel};
use unpopped::oracle::TypedBuffer;
use unpopped::oracle::kernel_eval::KernelArg;
use unpopped::oracle::kernel_launch::run_launch;
use unpopped_vocab::ElementKind::{self, I32, I64, U8, U32};

const N: usize = 6;

fn l(i: u32) -> ScalarExpr {
    ScalarExpr::Local(i)
}

fn at(buffer: u32) -> Place {
    Place::Logical {
        buffer,
        coords: vec![ScalarExpr::Exec(ExecId::ThreadIdx(0))],
    }
}

/// Buffers 0 `a`, 1 `b` (type `t`), 2 `m` (`U8` mask), 3 `out` (`t`). `by_cmp`: the condition is `a < b` instead of `m`.
fn kernel(t: ElementKind, by_cmp: bool, swap: bool) -> KernelFn {
    let cond = if by_cmp {
        ScalarExpr::Binary(BinaryOp::CmpLt, Box::new(l(0)), Box::new(l(1)))
    } else {
        l(2)
    };
    let (x, y) = if swap { (l(1), l(0)) } else { (l(0), l(1)) };
    let mut k = KernelFn::new(
        "int_where",
        U32,
        vec![
            Param::buffer("a", t, BufferRole::Read, Some(1)),
            Param::buffer("b", t, BufferRole::Read, Some(1)),
            Param::buffer("m", U8, BufferRole::Read, Some(1)),
            Param::buffer("out", t, BufferRole::Write, Some(1)),
        ],
        vec![
            LocalDecl::scalar(t),
            LocalDecl::scalar(t),
            LocalDecl::scalar(U8),
        ],
        Block(vec![
            Stmt::Load {
                local: 0,
                place: at(0),
            },
            Stmt::Load {
                local: 1,
                place: at(1),
            },
            Stmt::Load {
                local: 2,
                place: at(2),
            },
            Stmt::Store {
                place: at(3),
                value: ScalarExpr::Select(Box::new(cond), Box::new(x), Box::new(y)),
            },
        ]),
    );
    k.launch = Some(LaunchShape::new([N as u32, 1, 1], [1, 1, 1]));
    k
}

const MASK: [u8; N] = [1, 0, 2, 0, 255, 1];

/// Two operand sets per type as `i64`s that fit every tested width (`U8` included).
const A: [i64; N] = [3, 9, 0, 200, 17, 100];
const B: [i64; N] = [5, 2, 0, 100, 18, 99];

fn bytes(t: ElementKind, v: &[i64]) -> Vec<u8> {
    v.iter()
        .flat_map(|x| match t {
            U8 => vec![*x as u8],
            I32 | U32 => (*x as u32).to_le_bytes().to_vec(),
            _ => x.to_le_bytes().to_vec(),
        })
        .collect()
}

fn run(t: ElementKind, by_cmp: bool, swap: bool) -> Vec<u8> {
    let k = kernel(t, by_cmp, swap);
    assert_eq!(check_kernel(&k), Ok(()), "{t:?}");
    let buf = |e, b| KernelArg::Buffer(TypedBuffer::new(e, vec![N as i64], vec![1], b));
    let mut args = [
        buf(t, bytes(t, &A)),
        buf(t, bytes(t, &B)),
        buf(U8, MASK.to_vec()),
        buf(t, vec![0; bytes(t, &A).len()]),
    ];
    run_launch(&k, &mut args).unwrap_or_else(|e| panic!("{t:?}: {e:?}"));
    let KernelArg::Buffer(o) = &args[3] else {
        unreachable!("a buffer")
    };
    o.bytes.clone()
}

fn want(t: ElementKind, by_cmp: bool) -> Vec<u8> {
    let v: Vec<i64> = (0..N)
        .map(|i| {
            let take_a = if by_cmp { A[i] < B[i] } else { MASK[i] != 0 };
            if take_a { A[i] } else { B[i] }
        })
        .collect();
    bytes(t, &v)
}

#[test]
fn integer_where_matches_the_host_loop_by_mask_and_by_comparison() {
    for t in [I32, I64, U8, U32] {
        for by_cmp in [false, true] {
            assert_eq!(run(t, by_cmp, false), want(t, by_cmp), "{t:?} cmp={by_cmp}");
        }
    }
}

#[test]
fn swapped_arms_change_the_value() {
    for t in [I32, I64, U8, U32] {
        for by_cmp in [false, true] {
            assert_ne!(run(t, by_cmp, true), want(t, by_cmp), "{t:?} cmp={by_cmp}");
        }
    }
}
