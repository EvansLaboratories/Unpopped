// SPDX-License-Identifier: MIT OR Apache-2.0
//! **A `Cast` of a wide source to `F16`/`Bf16` rounds once, in C as in the oracle** (ROADMAP U26): an `F64` or a 32/64-bit
//! integer cannot reach a `float` without a first rounding, so `unpopped_cpu_c::CpuC` spells the cast through a `double`
//! encoder (`unp_d_to_f16`, `unp_d_to_bf16`). The data sits a hair above and below every tie, where rounding through
//! `float` first lands on the tie and goes to even, and at the overflow edges; the 16-bit outputs compare byte for byte.

use unpopped::backend::Backend;
use unpopped::ir::{CastSpec, RoundMode, Saturation, ScalarExpr};
use unpopped::kernel::{Block, BufferRole, KernelFn, LocalDecl, Param, Place, Stmt};
use unpopped::oracle::TypedBuffer;
use unpopped::oracle::kernel_eval::KernelArg;
use unpopped_cpu_c::CpuC;
use unpopped_vocab::ElementKind;

#[path = "support/cc.rs"]
mod cc;
#[path = "support/kernel_run.rs"]
mod kernel_run;
use kernel_run::agree;

const I32: ElementKind = ElementKind::I32;

fn at(i: ScalarExpr, buffer: u32) -> Place {
    Place::Logical {
        buffer,
        coords: vec![i],
    }
}

/// `out[i] = (to) in[i]` for `i` in `0..n`.
fn convert(tag: &str, (from, to): (ElementKind, ElementKind), n: usize) -> KernelFn {
    let i = || ScalarExpr::Local(0);
    let spec = CastSpec {
        to,
        round: RoundMode::NearestEven,
        sat: Saturation::Wrap,
    };
    let body = vec![
        Stmt::Load {
            local: 1,
            place: at(i(), 0),
        },
        Stmt::Store {
            place: at(i(), 1),
            value: ScalarExpr::Cast(Box::new(ScalarExpr::Local(1)), spec),
        },
    ];
    KernelFn::new(
        tag,
        ElementKind::F32,
        vec![
            Param::buffer("in", from, BufferRole::Read, Some(1)),
            Param::buffer("out", to, BufferRole::Write, Some(1)),
        ],
        vec![LocalDecl::scalar(I32), LocalDecl::scalar(from)],
        Block(vec![Stmt::For {
            local: 0,
            start: ScalarExpr::ConstInt(0, I32),
            end: ScalarExpr::ConstInt(n as i64, I32),
            step: ScalarExpr::ConstInt(1, I32),
            body: Block(body),
        }]),
    )
}

fn buffer(t: ElementKind, n: usize, bytes: Vec<u8>) -> KernelArg {
    KernelArg::Buffer(TypedBuffer::new(t, vec![n as i64], vec![1], bytes))
}

/// The value of the 16-bit pattern `p` of `to`, exactly.
fn value(to: ElementKind, p: u16) -> f64 {
    if to == ElementKind::Bf16 {
        return f64::from(f32::from_bits(u32::from(p) << 16));
    }
    let (e, m) = (i32::from((p >> 10) & 0x1F), f64::from(p & 0x3FF));
    if e == 0 {
        m * 2f64.powi(-24)
    } else {
        (m + 1024.0) * 2f64.powi(e - 25)
    }
}

/// Every `step`-th finite pattern's midpoint with its next neighbour, and a hair above and below it.
fn near_ties(to: ElementKind, step: usize) -> Vec<f64> {
    let top = if to == ElementKind::Bf16 {
        0x7F7F
    } else {
        0x7BFF
    };
    let mut out = vec![
        0.0,
        -0.0,
        1e300,
        -1e300,
        1e-300,
        65504.0,
        65519.999,
        65520.0,
        f64::INFINITY,
        f64::NAN,
    ];
    for p in (0..top).step_by(step) {
        let mid = (value(to, p) + value(to, p + 1)) / 2.0;
        out.extend([
            mid,
            mid * (1.0 + 1e-13),
            mid * (1.0 - 1e-13),
            -mid * (1.0 + 1e-13),
        ]);
    }
    out
}

fn check(tag: &str, (from, to): (ElementKind, ElementKind), bytes: Vec<u8>, n: usize) {
    let k = convert(tag, (from, to), n);
    agree(
        tag,
        &k,
        vec![buffer(from, n, bytes), buffer(to, n, vec![0; n * 2])],
    );
}

#[test]
fn an_f64_source_rounds_once_near_every_tie() {
    for (tag, to) in [("d_f16", ElementKind::F16), ("d_bf16", ElementKind::Bf16)] {
        let v = near_ties(to, 11);
        check(
            tag,
            (ElementKind::F64, to),
            v.iter().flat_map(|x| x.to_le_bytes()).collect(),
            v.len(),
        );
    }
}

#[test]
fn a_32_or_64_bit_integer_source_rounds_once() {
    let ints: Vec<i64> = vec![
        0,
        1,
        -1,
        2047,
        2049,
        65504,
        65519,
        65520,
        65535,
        65536,
        -65520,
        16_777_217,
        2_147_483_647,
        -2_147_483_648,
        4_294_967_295,
        9_007_199_254_740_993,
        i64::MAX,
        i64::MIN,
        33_554_433,
        8_388_609,
    ];
    for to in [ElementKind::F16, ElementKind::Bf16] {
        let tag = |s: &str| format!("int_{s}_{to:?}");
        let wide = |t, f: &dyn Fn(i64) -> Vec<u8>, keep: &dyn Fn(i64) -> bool| {
            let v: Vec<i64> = ints.iter().copied().filter(|x| keep(*x)).collect();
            (
                t,
                v.iter().flat_map(|x| f(*x)).collect::<Vec<u8>>(),
                v.len(),
            )
        };
        for (t, bytes, n) in [
            wide(I32, &|x| (x as i32).to_le_bytes().to_vec(), &|x| {
                i32::try_from(x).is_ok()
            }),
            wide(
                ElementKind::U32,
                &|x| (x as u32).to_le_bytes().to_vec(),
                &|x| u32::try_from(x).is_ok(),
            ),
            wide(ElementKind::I64, &|x| x.to_le_bytes().to_vec(), &|_| true),
            wide(
                ElementKind::U64,
                &|x| (x as u64).to_le_bytes().to_vec(),
                &|x| x >= 0,
            ),
        ] {
            check(&tag(&format!("{t:?}")), (t, to), bytes, n);
        }
    }
}

#[test]
fn the_exact_encoder_is_in_the_emitted_c_only_when_a_wide_cast_needs_it() {
    let text = |from, to| {
        CpuC.lower_kernel(&convert("t", (from, to), 1))
            .expect("lowers")
            .source
    };
    assert!(text(ElementKind::F64, ElementKind::F16).contains("unp_d_to_f16("));
    assert!(text(I32, ElementKind::Bf16).contains("unp_d_to_bf16("));
    assert!(!text(ElementKind::F32, ElementKind::F16).contains("unp_d_to_f16"));
}
