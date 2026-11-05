// SPDX-License-Identifier: MIT OR Apache-2.0
//! **The serial C of a sub-byte kernel agrees with the launch interpreter** (ROADMAP U9b S2, `docs/options-u9b-subbyte-dequant.md`).
//! cpu-c spells an `I4`/`U4` buffer as packed bytes (two elements to a byte, low nibble first), a load as a shift and mask
//! (sign-extended for `I4`), a widening `Cast` as the plain C conversion and a narrowing `Cast` as a wrap modulo 16. The C,
//! compiled with a host compiler and run, must write the bytes `run_launch` writes, over every nibble value.

use unpopped::backend::Backend;
use unpopped::exec::{ExecId, LaunchShape};
use unpopped::ir::{CastSpec, RoundMode, Saturation, ScalarExpr};
use unpopped::kernel::{Block, BufferRole, KernelFn, LocalDecl, Param, Place, Stmt};
use unpopped::oracle::TypedBuffer;
use unpopped::oracle::kernel_eval::KernelArg;
use unpopped_cpu_c::CpuC;
use unpopped_vocab::ElementKind;

#[path = "support/cc.rs"]
mod cc;
#[expect(
    dead_code,
    reason = "the shared driver; this file uses the part that runs a launched kernel"
)]
#[path = "support/kernel_run.rs"]
mod kernel_run;
#[path = "support/launch_run.rs"]
mod launch_run;
use launch_run::agree_launch;

const I32: ElementKind = ElementKind::I32;
const I4: ElementKind = ElementKind::I4;
const U4: ElementKind = ElementKind::U4;
const N: u32 = 16;

fn at(buffer: u32) -> Place {
    Place::Logical {
        buffer,
        coords: vec![ScalarExpr::Exec(ExecId::ThreadIdx(0))],
    }
}

fn cast(x: ScalarExpr, to: ElementKind) -> ScalarExpr {
    ScalarExpr::Cast(
        Box::new(x),
        CastSpec {
            to,
            round: RoundMode::NearestEven,
            sat: Saturation::Wrap,
        },
    )
}

fn sixteen_threads(k: KernelFn) -> KernelFn {
    k.with_launch(LaunchShape::new([N, 1, 1], [1, 1, 1]))
}

/// `out[t] = (i32) w[t]`.
fn widen(sub: ElementKind) -> KernelFn {
    sixteen_threads(KernelFn::new(
        "widen",
        I32,
        vec![
            Param::buffer("w", sub, BufferRole::Read, Some(1)),
            Param::buffer("out", I32, BufferRole::ReadWrite, Some(1)),
        ],
        vec![LocalDecl::scalar(sub)],
        Block(vec![
            Stmt::Load {
                local: 0,
                place: at(0),
            },
            Stmt::Store {
                place: at(1),
                value: cast(ScalarExpr::Local(0), I32),
            },
        ]),
    ))
}

/// `out[t] = (i32)(sub) x[t]`.
fn narrow(sub: ElementKind) -> KernelFn {
    sixteen_threads(KernelFn::new(
        "narrow",
        I32,
        vec![
            Param::buffer("x", I32, BufferRole::Read, Some(1)),
            Param::buffer("out", I32, BufferRole::ReadWrite, Some(1)),
        ],
        vec![LocalDecl::scalar(I32)],
        Block(vec![
            Stmt::Load {
                local: 0,
                place: at(0),
            },
            Stmt::Store {
                place: at(1),
                value: cast(cast(ScalarExpr::Local(0), sub), I32),
            },
        ]),
    ))
}

fn i32_buf(v: &[i32]) -> KernelArg {
    KernelArg::Buffer(TypedBuffer::new(
        I32,
        vec![v.len() as i64],
        vec![1],
        v.iter().flat_map(|x| x.to_le_bytes()).collect(),
    ))
}

#[test]
fn every_nibble_widens_the_same_in_c_and_in_the_oracle() {
    // the bytes 0x10, 0x32, ... 0xFE: each of the 16 nibble values appears once, in order
    let bytes: Vec<u8> = (0..8u8).map(|i| (2 * i) | ((2 * i + 1) << 4)).collect();
    for sub in [I4, U4] {
        let w = KernelArg::Buffer(TypedBuffer::new(sub, vec![16], vec![1], bytes.clone()));
        assert!(CpuC.lower_kernel(&widen(sub)).is_ok(), "{sub:?} lowers");
        agree_launch(
            &format!("widen {sub:?}"),
            &widen(sub),
            vec![w, i32_buf(&[0; 16])],
        );
    }
}

#[test]
fn narrowing_wraps_the_same_in_c_and_in_the_oracle() {
    let x = [
        9,
        -9,
        16,
        7,
        8,
        -8,
        100,
        -1,
        0,
        15,
        -16,
        17,
        -17,
        255,
        -256,
        i32::MAX,
    ];
    for sub in [I4, U4] {
        agree_launch(
            &format!("narrow {sub:?}"),
            &narrow(sub),
            vec![i32_buf(&x), i32_buf(&[0; 16])],
        );
    }
}
