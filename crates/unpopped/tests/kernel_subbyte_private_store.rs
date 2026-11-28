// SPDX-License-Identifier: MIT OR Apache-2.0
//! A `Store` of a sub-byte element (`I4`, `U4`) into a `Private` allocation (ROADMAP U9b, the narrow rule for MMA fragments of 4-bit
//! elements). The U9b ruling of 2026-11-05 refuses a sub-byte `Store` because it is a read-modify-write of a byte two threads
//! share; a `Private` allocation is one thread's own, so no byte is shared. A `Shared` or global sub-byte buffer stays read-only.

use unpopped::exec::{AllocDecl, ExecId, Extent, LaunchShape, MemorySpace};
use unpopped::ir::{CastSpec, RoundMode, Saturation, ScalarExpr};
use unpopped::kernel::{
    Block, BufferRole, KernelError, KernelFn, LocalDecl, Param, Place, Stmt, check_kernel,
};
use unpopped::oracle::TypedBuffer;
use unpopped::oracle::kernel_eval::KernelArg;
use unpopped::oracle::kernel_launch::run_launch;
use unpopped_vocab::ElementKind::{self, I4, I32, U4};

fn tid() -> ScalarExpr {
    ScalarExpr::Exec(ExecId::ThreadIdx(0))
}

fn idx(i: i64) -> ScalarExpr {
    ScalarExpr::ConstInt(i, ElementKind::U32)
}

fn narrow(x: ScalarExpr, to: ElementKind) -> ScalarExpr {
    ScalarExpr::Cast(
        Box::new(x),
        CastSpec {
            to,
            round: RoundMode::NearestEven,
            sat: Saturation::Wrap,
        },
    )
}

fn widen(x: ScalarExpr) -> ScalarExpr {
    narrow(x, I32)
}

fn slot(space: MemorySpace, sub: ElementKind, i: i64) -> (AllocDecl, Place) {
    (
        AllocDecl::new("frag", space, sub, Extent::Static(4)),
        Place::Alloc {
            alloc: 0,
            index: idx(i),
        },
    )
}

/// `out[t] = widen(frag[0]) * 100 + widen(frag[1])` after `frag[1] = (sub) x[t]` then `frag[0] = (sub) (x[t] + 1)`: the two
/// elements share one byte, so the second store must leave the first alone.
fn kernel(space: MemorySpace, sub: ElementKind) -> KernelFn {
    let at = |buffer| Place::Logical {
        buffer,
        coords: vec![tid()],
    };
    let (decl, first) = slot(space, sub, 0);
    let (_, second) = slot(space, sub, 1);
    let plus_one = ScalarExpr::Add(
        Box::new(ScalarExpr::Local(0)),
        Box::new(ScalarExpr::ConstInt(1, I32)),
    );
    KernelFn::new(
        "frag",
        I32,
        vec![
            Param::buffer("x", I32, BufferRole::Read, Some(1)),
            Param::buffer("out", I32, BufferRole::Write, Some(1)),
        ],
        vec![
            LocalDecl::scalar(I32),
            LocalDecl::scalar(sub),
            LocalDecl::scalar(sub),
        ],
        Block(vec![
            Stmt::Load {
                local: 0,
                place: at(0),
            },
            Stmt::Store {
                place: second.clone(),
                value: narrow(ScalarExpr::Local(0), sub),
            },
            Stmt::Store {
                place: first.clone(),
                value: narrow(plus_one, sub),
            },
            Stmt::Load {
                local: 1,
                place: first,
            },
            Stmt::Load {
                local: 2,
                place: second,
            },
            Stmt::Store {
                place: at(1),
                value: ScalarExpr::Add(
                    Box::new(ScalarExpr::Mul(
                        Box::new(widen(ScalarExpr::Local(1))),
                        Box::new(ScalarExpr::ConstInt(100, I32)),
                    )),
                    Box::new(widen(ScalarExpr::Local(2))),
                ),
            },
        ]),
    )
    .with_launch(LaunchShape::new([8, 1, 1], [1, 1, 1]))
    .with_allocs(vec![decl])
}

fn run(k: &KernelFn) -> Vec<i32> {
    assert_eq!(check_kernel(k), Ok(()), "well formed");
    let x: Vec<u8> = (0..8i32).flat_map(|v| (v * 3 - 7).to_le_bytes()).collect();
    let buf = |bytes, n: i64| KernelArg::Buffer(TypedBuffer::new(I32, vec![n], vec![1], bytes));
    let mut args = vec![buf(x, 8), buf(vec![0u8; 32], 8)];
    run_launch(k, &mut args).unwrap_or_else(|e| panic!("{e:?}"));
    let KernelArg::Buffer(out) = &args[1] else {
        unreachable!()
    };
    out.bytes
        .chunks(4)
        .map(|c| i32::from_le_bytes(c.try_into().unwrap()))
        .collect()
}

/// The wrapped value of `v` in a 4-bit element: `I4` sign-extends, `U4` zero-extends.
fn wrap4(v: i32, signed: bool) -> i32 {
    let n = v & 0xf;
    if signed && n >= 8 { n - 16 } else { n }
}

#[test]
fn private_sub_byte_stores_wrap_and_do_not_disturb_the_neighbour() {
    for (sub, signed) in [(I4, true), (U4, false)] {
        let want: Vec<i32> = (0..8i32)
            .map(|t| {
                let x = t * 3 - 7;
                wrap4(x + 1, signed) * 100 + wrap4(x, signed)
            })
            .collect();
        assert_eq!(run(&kernel(MemorySpace::Private, sub)), want, "{sub:?}");
    }
}

#[test]
fn a_shared_sub_byte_store_is_still_refused() {
    for sub in [I4, U4] {
        match check_kernel(&kernel(MemorySpace::Shared, sub)) {
            Err(KernelError::TypeMismatch { what }) => {
                assert!(what.contains("outside a Private allocation"), "{what}");
            }
            other => panic!("{sub:?}: expected the refusal, got {other:?}"),
        }
    }
}
