// SPDX-License-Identifier: MIT OR Apache-2.0
//! Shared helpers of the F128 kernel tests (`kernel_f128.rs`, `kernel_f128_casts.rs`): the reference vectors, buffers, a binary kernel.

use unpopped::exec::{ExecId, LaunchShape};
use unpopped::ir::{CastSpec, RoundMode, Saturation, ScalarExpr};
use unpopped::kernel::{Block, BufferRole, KernelFn, LocalDecl, Param, Place, Stmt, check_kernel};
use unpopped::oracle::TypedBuffer;
use unpopped::oracle::kernel_eval::KernelArg;
use unpopped::oracle::kernel_launch::run_launch;
use unpopped_vocab::ElementKind::{self, F32, F128};

pub(crate) const DATA: &str = include_str!("../data/soft_f128_vectors.txt");
pub(crate) const ONE: u128 = 0x3fff << 112;

pub(crate) fn hex(s: &str) -> u128 {
    u128::from_str_radix(s, 16).expect("hex")
}

/// The `(a, b, expected)` columns of the rows of `op`.
pub(crate) fn rows(op: &str) -> Vec<(u128, u128, String)> {
    DATA.lines()
        .filter(|l| l.starts_with(op) && l[op.len()..].starts_with(' '))
        .map(|l| {
            let mut it = l.split_whitespace().skip(1);
            let (a, b) = (hex(it.next().unwrap()), hex(it.next().unwrap()));
            (a, b, it.next().unwrap().to_string())
        })
        .collect()
}

pub(crate) fn at(buffer: u32) -> Place {
    Place::Logical {
        buffer,
        coords: vec![ScalarExpr::Exec(ExecId::ThreadIdx(0))],
    }
}

pub(crate) fn buf(dt: ElementKind, bytes: Vec<u8>, n: usize) -> KernelArg {
    KernelArg::Buffer(TypedBuffer::new(dt, vec![n as i64], vec![1], bytes))
}

pub(crate) fn le16(v: &[u128]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

pub(crate) fn out_bytes(args: &[KernelArg], i: usize) -> Vec<u8> {
    let KernelArg::Buffer(b) = &args[i] else {
        unreachable!()
    };
    b.bytes.clone()
}

pub(crate) fn words(bytes: &[u8]) -> Vec<u128> {
    bytes
        .chunks(16)
        .map(|c| u128::from_le_bytes(c.try_into().unwrap()))
        .collect()
}

/// `out[t] = f(a[t], b[t])` over `F128` buffers, `n` threads.
pub(crate) fn binary(n: usize, f: impl Fn(ScalarExpr, ScalarExpr) -> ScalarExpr) -> KernelFn {
    KernelFn::new(
        "f128",
        F32,
        vec![
            Param::buffer("a", F128, BufferRole::Read, Some(1)),
            Param::buffer("b", F128, BufferRole::Read, Some(1)),
            Param::buffer("out", F128, BufferRole::Write, Some(1)),
        ],
        vec![LocalDecl::scalar(F128), LocalDecl::scalar(F128)],
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
                value: f(ScalarExpr::Local(0), ScalarExpr::Local(1)),
            },
        ]),
    )
    .with_launch(LaunchShape::new([n as u32, 1, 1], [1, 1, 1]))
}

pub(crate) fn run_binary(k: &KernelFn, a: &[u128], b: &[u128]) -> Vec<u128> {
    assert_eq!(check_kernel(k), Ok(()), "well formed");
    let n = a.len();
    let mut args = vec![
        buf(F128, le16(a), n),
        buf(F128, le16(b), n),
        buf(F128, vec![0; 16 * n], n),
    ];
    run_launch(k, &mut args).unwrap_or_else(|e| panic!("{e:?}"));
    words(&out_bytes(&args, 2))
}

/// `7.0` as an `F128`: a literal has the compute type (`F32` here), so it is cast.
pub(crate) fn seven_f128() -> ScalarExpr {
    ScalarExpr::Cast(
        Box::new(ScalarExpr::Const(7.0)),
        CastSpec {
            to: F128,
            round: RoundMode::NearestEven,
            sat: Saturation::Wrap,
        },
    )
}

pub(crate) fn bx(e: ScalarExpr) -> Box<ScalarExpr> {
    Box::new(e)
}
