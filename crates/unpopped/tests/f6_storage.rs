// SPDX-License-Identifier: MIT OR Apache-2.0
//! **The oracle holds an `F6E2M3` / `F6E3M2` element in an 8-bit container** (ROADMAP row f6-oracle-storage; design
//! `docs/design-f6-storage.md`): one element per byte, the six value bits low, the two high bits zero, as PTX ISA 9.5 section 5.2.3
//! (p. 43) holds them. A `Private` fragment of the type, a buffer of it, a `ClampFinite` store into it and a load out of it run and
//! round-trip over every code of both formats; a set padding bit is ignored on read and every store writes zero there. `F4E2M1`
//! (the 4-bit float the oracle already holds) is the control. The expected values come from the OCP MX definition written
//! here, not from `cast_ref`.

use unpopped::exec::{AllocDecl, ExecId, Extent, LaunchShape, MemorySpace};
use unpopped::ir::{CastSpec, RoundMode, Saturation, ScalarExpr};
use unpopped::kernel::{Block, BufferRole, KernelFn, LocalDecl, Param, Place, Stmt, check_kernel};
use unpopped::oracle::TypedBuffer;
use unpopped::oracle::kernel_eval::KernelArg;
use unpopped::oracle::kernel_launch::run_launch;
use unpopped_vocab::ElementKind::{self, F4E2M1, F6E2M3, F6E3M2, F32};

/// The value of the OCP MX element code `bits` (sign, `e` exponent bits, `m` mantissa bits; no infinity, no NaN).
fn mx(bits: u32, e: u32, m: u32) -> f64 {
    let (sign, exp, man) = (
        bits >> (e + m) & 1,
        bits >> m & ((1 << e) - 1),
        bits & ((1 << m) - 1),
    );
    let bias = (1i32 << (e - 1)) - 1;
    let mag = if exp == 0 {
        f64::from(man) * 2f64.powi(1 - bias - m as i32)
    } else {
        (1.0 + f64::from(man) / f64::from(1u32 << m)) * 2f64.powi(exp as i32 - bias)
    };
    if sign == 1 { -mag } else { mag }
}

fn fmt(ty: ElementKind) -> (u32, u32) {
    match ty {
        F6E2M3 => (2, 3),
        F6E3M2 => (3, 2),
        _ => (2, 1),
    }
}

fn cast(e: ScalarExpr, to: ElementKind) -> ScalarExpr {
    ScalarExpr::Cast(
        Box::new(e),
        CastSpec {
            to,
            round: RoundMode::NearestEven,
            sat: Saturation::ClampFinite,
        },
    )
}

fn tid() -> ScalarExpr {
    ScalarExpr::Exec(ExecId::ThreadIdx(0))
}

fn at(buffer: u32) -> Place {
    Place::Logical {
        buffer,
        coords: vec![tid()],
    }
}

/// `out[tid] = (f32) frag[0]` where `frag` is a `Private` allocation of `ty` that `tid` filled from `in[tid]` through a
/// `ClampFinite` cast: the store, the load and the decode all go through the oracle's container.
fn through_private(ty: ElementKind, n: u32) -> KernelFn {
    KernelFn::new(
        "f6_private",
        F32,
        vec![
            Param::buffer("in", F32, BufferRole::Read, Some(1)),
            Param::buffer("out", F32, BufferRole::ReadWrite, Some(1)),
        ],
        vec![LocalDecl::scalar(F32), LocalDecl::scalar(ty)],
        Block(vec![
            Stmt::Load {
                local: 0,
                place: at(0),
            },
            Stmt::Store {
                place: Place::Alloc {
                    alloc: 0,
                    index: ScalarExpr::ConstInt(0, ElementKind::U32),
                },
                value: cast(ScalarExpr::Local(0), ty),
            },
            Stmt::Load {
                local: 1,
                place: Place::Alloc {
                    alloc: 0,
                    index: ScalarExpr::ConstInt(0, ElementKind::U32),
                },
            },
            Stmt::Store {
                place: at(1),
                value: cast(ScalarExpr::Local(1), F32),
            },
        ]),
    )
    .with_launch(LaunchShape::new([n, 1, 1], [1, 1, 1]))
    .with_allocs(vec![AllocDecl::new(
        "frag",
        MemorySpace::Private,
        ty,
        Extent::Static(1),
    )])
}

fn f32s(out: &TypedBuffer) -> Vec<f32> {
    out.bytes
        .chunks(4)
        .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
        .collect()
}

fn run(k: &KernelFn, mut args: Vec<KernelArg>) -> Vec<KernelArg> {
    assert_eq!(check_kernel(k), Ok(()), "{}", k.name);
    run_launch(k, &mut args).expect("the oracle runs it");
    args
}

#[test]
fn every_code_of_each_6_bit_format_round_trips_through_a_private_fragment() {
    for ty in [F6E2M3, F6E3M2, F4E2M1] {
        let (e, m) = fmt(ty);
        let codes = 1u32 << (1 + e + m);
        let values: Vec<f32> = (0..codes).map(|c| mx(c, e, m) as f32).collect();
        let k = through_private(ty, codes);
        let args = run(
            &k,
            vec![
                KernelArg::Buffer(TypedBuffer::from_f32(&[i64::from(codes)], &values)),
                KernelArg::Buffer(TypedBuffer::from_f32(
                    &[i64::from(codes)],
                    &vec![9.0; codes as usize],
                )),
            ],
        );
        let KernelArg::Buffer(out) = &args[1] else {
            unreachable!()
        };
        assert_eq!(
            f32s(out),
            values,
            "{ty:?}: every lattice value survives the store, the load and the decode"
        );
    }
}

#[test]
fn a_value_past_the_largest_finite_clamps_in_the_container() {
    // E2M3 tops out at 7.5 and E3M2 at 28 (OCP MX); `ClampFinite` saturates instead of overflowing
    for (ty, top) in [(F6E2M3, 7.5f32), (F6E3M2, 28.0)] {
        let k = through_private(ty, 2);
        let args = run(
            &k,
            vec![
                KernelArg::Buffer(TypedBuffer::from_f32(&[2], &[1000.0, -1000.0])),
                KernelArg::Buffer(TypedBuffer::from_f32(&[2], &[0.0, 0.0])),
            ],
        );
        let KernelArg::Buffer(out) = &args[1] else {
            unreachable!()
        };
        assert_eq!(f32s(out), vec![top, -top], "{ty:?}");
    }
}

#[test]
fn a_buffer_of_6_bit_elements_is_one_byte_each_with_the_padding_ignored_on_read_and_zero_on_write()
{
    for ty in [F6E2M3, F6E3M2] {
        let (e, m) = fmt(ty);
        // read: every code with the two padding bits SET reads as the code without them
        let padded: Vec<u8> = (0..64u8).map(|c| c | 0xC0).collect();
        let k = KernelFn::new(
            "f6_read",
            F32,
            vec![
                Param::buffer("in", ty, BufferRole::Read, Some(1)),
                Param::buffer("out", F32, BufferRole::ReadWrite, Some(1)),
            ],
            vec![LocalDecl::scalar(ty)],
            Block(vec![
                Stmt::Load {
                    local: 0,
                    place: at(0),
                },
                Stmt::Store {
                    place: at(1),
                    value: cast(ScalarExpr::Local(0), F32),
                },
            ]),
        )
        .with_launch(LaunchShape::new([64, 1, 1], [1, 1, 1]));
        let args = run(
            &k,
            vec![
                KernelArg::Buffer(TypedBuffer::new(ty, vec![64], vec![1], padded)),
                KernelArg::Buffer(TypedBuffer::from_f32(&[64], &[0.0; 64])),
            ],
        );
        let KernelArg::Buffer(out) = &args[1] else {
            unreachable!()
        };
        let want: Vec<f32> = (0..64).map(|c| mx(c, e, m) as f32).collect();
        assert_eq!(f32s(out), want, "{ty:?} read with padding set");
        // write: a store into a buffer of the type writes zero in bits 6 and 7
        let w = KernelFn::new(
            "f6_write",
            F32,
            vec![
                Param::buffer("in", F32, BufferRole::Read, Some(1)),
                Param::buffer("out", ty, BufferRole::ReadWrite, Some(1)),
            ],
            vec![LocalDecl::scalar(F32)],
            Block(vec![
                Stmt::Load {
                    local: 0,
                    place: at(0),
                },
                Stmt::Store {
                    place: at(1),
                    value: cast(ScalarExpr::Local(0), ty),
                },
            ]),
        )
        .with_launch(LaunchShape::new([64, 1, 1], [1, 1, 1]));
        let args = run(
            &w,
            vec![
                KernelArg::Buffer(TypedBuffer::from_f32(&[64], &want)),
                KernelArg::Buffer(TypedBuffer::new(ty, vec![64], vec![1], vec![0xFF; 64])),
            ],
        );
        let KernelArg::Buffer(out) = &args[1] else {
            unreachable!()
        };
        assert_eq!(out.bytes.len(), 64, "{ty:?}: one byte per element");
        assert!(
            out.bytes.iter().all(|b| b >> 6 == 0),
            "{ty:?}: the padding bits are written as zero"
        );
        assert_eq!(
            out.bytes,
            (0..64u8).map(|c| c).collect::<Vec<_>>(),
            "{ty:?}: the byte is the code"
        );
    }
}
