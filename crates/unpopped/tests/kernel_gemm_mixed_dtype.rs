// SPDX-License-Identifier: MIT OR Apache-2.0
//! **A GEMM whose addend `C` has another dtype than `D`, and whose compute dtype differs from its accumulator**
//! (CUTLASS 31; KISS 771 Q3: the addend is an ordinary `[M, N]` extra input, no `c` role hint). Built from ordinary ops
//! only: `I8` operands are widened with a `Cast` to `I16` (the compute type, where one product fits), multiplied there,
//! widened again to the `I32` accumulator, and the `I16` addend is a `Cast` of an ordinary input. Each mutant changes one
//! of those types and must change the VALUE: the data hold `-128 * -128`, which wraps in `I8`, and a K-sum past `I16`.

use unpopped::exec::{ExecId, LaunchShape};
use unpopped::ir::{CastSpec, RoundMode, Saturation, ScalarExpr};
use unpopped::kernel::{
    Block, BufferRole, KernelFn, LocalDecl, Param, Place, Stmt, Unroll, check_kernel,
};
use unpopped::oracle::TypedBuffer;
use unpopped::oracle::kernel_eval::KernelArg;
use unpopped::oracle::kernel_launch::run_launch;
use unpopped_vocab::ElementKind as E;

const M: usize = 2;
const N: usize = 2;
const K: usize = 4;

#[derive(Clone, Copy, PartialEq, Debug)]
enum Mutant {
    None,
    /// the product is narrowed to `I8` before widening (compute dtype too small)
    ProductI8,
    /// the running sum is kept in `I16` (accumulator too small)
    AccI16,
    /// the addend is dropped
    DropC,
}

fn cast(x: ScalarExpr, to: E) -> ScalarExpr {
    ScalarExpr::Cast(
        Box::new(x),
        CastSpec {
            to,
            round: RoundMode::NearestEven,
            sat: Saturation::Wrap,
        },
    )
}

fn at(buffer: u32, r: ScalarExpr, c: ScalarExpr) -> Place {
    Place::Logical {
        buffer,
        coords: vec![r, c],
    }
}

fn u(v: i64) -> ScalarExpr {
    ScalarExpr::ConstInt(v, E::U32)
}

fn l(i: u32) -> ScalarExpr {
    ScalarExpr::Local(i)
}

/// Locals: 0 `acc` I32, 1 `a` I8, 2 `b` I8, 3 `c` I16, 4 `k` U32.
fn mainloop(m: Mutant) -> Vec<Stmt> {
    let mut prod = ScalarExpr::Mul(Box::new(cast(l(1), E::I16)), Box::new(cast(l(2), E::I16)));
    if m == Mutant::ProductI8 {
        prod = cast(prod, E::I8);
    }
    let mut sum = ScalarExpr::Add(Box::new(l(0)), Box::new(cast(prod, E::I32)));
    if m == Mutant::AccI16 {
        sum = cast(cast(sum, E::I16), E::I32);
    }
    let row = ScalarExpr::Exec(ExecId::ThreadIdx(1));
    let col = ScalarExpr::Exec(ExecId::ThreadIdx(0));
    vec![
        Stmt::Let {
            local: 0,
            init: ScalarExpr::ConstInt(0, E::I32),
        },
        Stmt::For {
            unroll: Unroll::None,
            local: 4,
            start: u(0),
            end: u(K as i64),
            step: u(1),
            body: Block(vec![
                Stmt::Load {
                    local: 1,
                    place: at(0, row, l(4)),
                },
                Stmt::Load {
                    local: 2,
                    place: at(1, l(4), col),
                },
                Stmt::Assign {
                    local: 0,
                    value: sum,
                },
            ]),
        },
    ]
}

/// Buffers: 0 `a` I8 `[M,K]`, 1 `b` I8 `[K,N]`, 2 `c` I16 `[M,N]` (read), 3 `d` I32 `[M,N]`: `d = acc + widen(c)`.
fn kernel(m: Mutant) -> KernelFn {
    let row = ScalarExpr::Exec(ExecId::ThreadIdx(1));
    let col = ScalarExpr::Exec(ExecId::ThreadIdx(0));
    let mut body = mainloop(m);
    body.push(Stmt::Load {
        local: 3,
        place: at(2, row.clone(), col.clone()),
    });
    let addend = if m == Mutant::DropC {
        ScalarExpr::ConstInt(0, E::I32)
    } else {
        cast(l(3), E::I32)
    };
    body.push(Stmt::Store {
        place: at(3, row, col),
        value: ScalarExpr::Add(Box::new(l(0)), Box::new(addend)),
    });
    let p = |n: &str, e: E, r: BufferRole| Param::buffer(n, e, r, Some(2));
    let locals = vec![
        LocalDecl::scalar(E::I32),
        LocalDecl::scalar(E::I8),
        LocalDecl::scalar(E::I8),
        LocalDecl::scalar(E::I16),
        LocalDecl::scalar(E::U32),
    ];
    KernelFn::new(
        "gemm_mixed_dtype",
        E::I32,
        vec![
            p("a", E::I8, BufferRole::Read),
            p("b", E::I8, BufferRole::Read),
            p("c", E::I16, BufferRole::Read),
            p("d", E::I32, BufferRole::ReadWrite),
        ],
        locals,
        Block(body),
    )
    .with_launch(LaunchShape::new([N as u32, M as u32, 1], [1, 1, 1]))
}

fn buf(e: E, shape: [i64; 2], bytes: Vec<u8>) -> KernelArg {
    KernelArg::Buffer(TypedBuffer::new(
        e,
        shape.to_vec(),
        vec![shape[1], 1],
        bytes,
    ))
}

fn data() -> (Vec<i8>, Vec<i8>, Vec<i16>) {
    (
        vec![-128, -128, -128, -128, 127, -128, 5, -7],
        vec![-128, 127, -128, -128, -128, 3, -128, 100],
        vec![-300, 1000, 7, -32768],
    )
}

fn reference() -> Vec<i32> {
    let (a, b, c) = data();
    let mut d = vec![0i32; M * N];
    for r in 0..M {
        for col in 0..N {
            let acc: i32 = (0..K)
                .map(|k| i32::from(a[r * K + k]) * i32::from(b[k * N + col]))
                .sum();
            d[r * N + col] = acc + i32::from(c[r * N + col]);
        }
    }
    d
}

fn run(m: Mutant) -> Vec<i32> {
    let k = kernel(m);
    assert_eq!(check_kernel(&k), Ok(()), "{m:?}");
    let (a, b, c) = data();
    let mut args = [
        buf(
            E::I8,
            [M as i64, K as i64],
            a.iter().map(|x| *x as u8).collect(),
        ),
        buf(
            E::I8,
            [K as i64, N as i64],
            b.iter().map(|x| *x as u8).collect(),
        ),
        buf(
            E::I16,
            [M as i64, N as i64],
            c.iter().flat_map(|x| x.to_le_bytes()).collect(),
        ),
        buf(E::I32, [M as i64, N as i64], vec![0; M * N * 4]),
    ];
    run_launch(&k, &mut args).unwrap();
    let KernelArg::Buffer(d) = &args[3] else {
        unreachable!("a buffer")
    };
    d.bytes
        .chunks(4)
        .map(|w| i32::from_le_bytes(w.try_into().unwrap()))
        .collect()
}

#[test]
fn the_mixed_dtype_gemm_matches_the_host_loop() {
    assert_eq!(run(Mutant::None), reference());
}

#[test]
fn each_type_mutant_changes_the_value() {
    let want = reference();
    for m in [Mutant::ProductI8, Mutant::AccI16, Mutant::DropC] {
        assert_ne!(run(m), want, "{m:?} must be caught by value");
    }
}
