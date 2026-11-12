// SPDX-License-Identifier: MIT OR Apache-2.0
//! **The C of `F16` and `Bf16` agrees with the oracle** (ROADMAP U14): buffers, locals and casts of the two narrow floats
//! lowered by `unpopped_cpu_c::CpuC::lower_kernel` (the 16-bit pattern in an `unsigned short`, the codec of
//! `cfamily::half_helpers`), compiled with a host C compiler, run, and compared byte for byte with `run_kernel` /
//! `run_launch`. A decode of every one of the 65536 patterns, an encode of the values on and between the patterns (the
//! round-to-nearest-even ties and the overflow edge), a move of the bits through `Load`/`Store`/`Select`/`CopyAsync` (a
//! NaN payload survives), and the refusals: arithmetic or a comparison on a narrow value, a `Cast` that would round
//! twice. The seeded mutants of the emitted C break the codec and the carrier, and each must change the bytes.

use unpopped::backend::Backend;
use unpopped::exec::{AllocDecl, ExecId, Extent, LaunchShape, MemorySpace, Scope};
use unpopped::ir::{BinaryOp, CastSpec, RoundMode, Saturation, ScalarExpr};
use unpopped::kernel::{
    BarrierKind, Block, BufferRole, KernelFn, LocalDecl, Param, Place, Stmt, check_kernel,
};
use unpopped::oracle::TypedBuffer;
use unpopped::oracle::kernel_eval::KernelArg;
use unpopped_cpu_c::CpuC;
use unpopped_vocab::ElementKind;

#[path = "support/cc.rs"]
mod cc;
#[allow(dead_code)]
#[path = "support/kernel_run.rs"]
mod kernel_run;
#[allow(dead_code)]
#[path = "support/launch_run.rs"]
mod launch_run;
use kernel_run::{agree, run_c_mutated};
use launch_run::agree_launch;

const F32: ElementKind = ElementKind::F32;
const I32: ElementKind = ElementKind::I32;
const U32: ElementKind = ElementKind::U32;

fn ci(v: i64) -> ScalarExpr {
    ScalarExpr::ConstInt(v, I32)
}

fn at(buffer: u32, i: ScalarExpr) -> Place {
    Place::Logical {
        buffer,
        coords: vec![i],
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

fn narrow(t: ElementKind, patterns: &[u16]) -> KernelArg {
    let bytes = patterns.iter().flat_map(|p| p.to_le_bytes()).collect();
    KernelArg::Buffer(TypedBuffer::new(
        t,
        vec![patterns.len() as i64],
        vec![1],
        bytes,
    ))
}

fn floats(v: &[f32]) -> KernelArg {
    let bytes = v.iter().flat_map(|x| x.to_le_bytes()).collect();
    KernelArg::Buffer(TypedBuffer::new(F32, vec![v.len() as i64], vec![1], bytes))
}

/// `for i in 0..n { body(i) }` over locals `0` (the index, `I32`) and the given extras.
fn looped(
    name: &str,
    params: Vec<Param>,
    extra: Vec<ElementKind>,
    n: usize,
    body: Vec<Stmt>,
) -> KernelFn {
    let mut locals = vec![LocalDecl::scalar(I32)];
    locals.extend(extra.into_iter().map(LocalDecl::scalar));
    KernelFn::new(
        name,
        F32,
        params,
        locals,
        Block(vec![Stmt::For {
            local: 0,
            start: ci(0),
            end: ci(n as i64),
            step: ci(1),
            body: Block(body),
        }]),
    )
}

fn load(local: u32, buffer: u32) -> Stmt {
    Stmt::Load {
        local,
        place: at(buffer, ScalarExpr::Local(0)),
    }
}

fn store(buffer: u32, value: ScalarExpr) -> Stmt {
    Stmt::Store {
        place: at(buffer, ScalarExpr::Local(0)),
        value,
    }
}

/// `out32[i] = (f32) in16[i]`.
fn decode(t: ElementKind, n: usize) -> KernelFn {
    looped(
        "decode",
        vec![
            Param::buffer("in", t, BufferRole::Read, Some(1)),
            Param::buffer("out", F32, BufferRole::Write, Some(1)),
        ],
        vec![t],
        n,
        vec![load(1, 0), store(1, cast(ScalarExpr::Local(1), F32))],
    )
}

/// `out16[i] = (narrow) in32[i]`.
fn encode(t: ElementKind, n: usize) -> KernelFn {
    looped(
        "encode",
        vec![
            Param::buffer("in", F32, BufferRole::Read, Some(1)),
            Param::buffer("out", t, BufferRole::Write, Some(1)),
        ],
        vec![F32, t],
        n,
        vec![
            load(1, 0),
            Stmt::Let {
                local: 2,
                init: cast(ScalarExpr::Local(1), t),
            },
            store(1, ScalarExpr::Local(2)),
        ],
    )
}

/// The `f32` a binary16 pattern denotes (`h`'s sign, 5-bit exponent, 10-bit mantissa).
fn f16_value(h: u16) -> f32 {
    let (sign, exp, mant) = (h >> 15, i32::from((h >> 10) & 0x1F), f32::from(h & 0x3FF));
    let mag = match exp {
        0 => mant * 2f32.powi(-24),
        0x1F if mant == 0.0 => f32::INFINITY,
        0x1F => f32::NAN,
        _ => (mant + 1024.0) * 2f32.powi(exp - 25),
    };
    if sign == 1 { -mag } else { mag }
}

/// Values on, between and around every `step`-th pattern's neighbours: the exact patterns, the exact midpoints (the ties),
/// and the `f32` neighbours of both. `value` is the pattern's `f32`.
fn around(patterns: impl Iterator<Item = u16>, value: impl Fn(u16) -> f32) -> Vec<f32> {
    let mut out = Vec::new();
    for p in patterns {
        let (a, b) = (value(p), value(p.wrapping_add(1)));
        let mid = if a.is_finite() && b.is_finite() {
            (a + b) / 2.0
        } else {
            a
        };
        for x in [a, mid] {
            let bits = x.to_bits();
            out.extend([
                x,
                f32::from_bits(bits.wrapping_add(1)),
                f32::from_bits(bits.wrapping_sub(1)),
            ]);
        }
    }
    out
}

/// Pseudo-random `f32` bit patterns (a wide exponent range, so overflow and underflow both occur).
fn scatter(n: usize) -> Vec<f32> {
    let mut s = 0x2545_F491_4F6C_DD1Du64;
    (0..n)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            f32::from_bits((s >> 16) as u32)
        })
        .collect()
}

fn every_pattern() -> Vec<u16> {
    (0..=u16::MAX).collect()
}

#[test]
fn a_decode_of_every_pattern_agrees() {
    for (tag, t) in [
        ("dec_f16", ElementKind::F16),
        ("dec_bf16", ElementKind::Bf16),
    ] {
        let all = every_pattern();
        agree(
            tag,
            &decode(t, all.len()),
            vec![narrow(t, &all), floats(&vec![0.0; all.len()])],
        );
    }
}

#[test]
fn an_encode_agrees_on_and_between_the_patterns_and_at_the_overflow_edge() {
    let mut f16 = around((0..=u16::MAX).step_by(5), f16_value);
    f16.extend([
        65504.0,
        65519.99,
        65520.0,
        65536.0,
        -65520.0,
        1e-8,
        5.960_464_5e-8,
        2.980_232_2e-8,
        f32::NAN,
        f32::INFINITY,
    ]);
    f16.extend(scatter(4000));
    let mut bf16 = around((0..=u16::MAX).step_by(5), |p| {
        f32::from_bits(u32::from(p) << 16)
    });
    bf16.extend([
        f32::MAX,
        f32::from_bits(0x7F7F_8000),
        f32::from_bits(0x7F7F_7FFF),
        f32::NAN,
        f32::NEG_INFINITY,
        1e-40,
    ]);
    bf16.extend(scatter(4000));
    for (tag, t, xs) in [
        ("enc_f16", ElementKind::F16, f16),
        ("enc_bf16", ElementKind::Bf16, bf16),
    ] {
        let n = xs.len();
        agree(
            tag,
            &encode(t, n),
            vec![floats(&xs), narrow(t, &vec![0; n])],
        );
    }
}

#[test]
fn a_narrow_to_narrow_cast_agrees() {
    // F16 -> Bf16 (loses precision), Bf16 -> F16 (loses range), F16 -> F16 (identity), I16 -> F16 (exact int, one rounding)
    let pats: Vec<u16> = (0..=u16::MAX).step_by(3).collect();
    for (tag, from, to) in [
        ("f16_to_bf16", ElementKind::F16, ElementKind::Bf16),
        ("bf16_to_f16", ElementKind::Bf16, ElementKind::F16),
        ("f16_to_f16", ElementKind::F16, ElementKind::F16),
    ] {
        let k = looped(
            tag,
            vec![
                Param::buffer("in", from, BufferRole::Read, Some(1)),
                Param::buffer("out", to, BufferRole::Write, Some(1)),
            ],
            vec![from],
            pats.len(),
            vec![load(1, 0), store(1, cast(ScalarExpr::Local(1), to))],
        );
        agree(
            tag,
            &k,
            vec![narrow(from, &pats), narrow(to, &vec![0; pats.len()])],
        );
    }
}

#[test]
fn an_integer_source_of_16_bits_agrees() {
    let ints: Vec<i16> = (-32768i32..=32767).step_by(7).map(|v| v as i16).collect();
    let bytes = ints.iter().flat_map(|v| v.to_le_bytes()).collect();
    let k = looped(
        "i16_to_f16",
        vec![
            Param::buffer("in", ElementKind::I16, BufferRole::Read, Some(1)),
            Param::buffer("out", ElementKind::F16, BufferRole::Write, Some(1)),
        ],
        vec![ElementKind::I16],
        ints.len(),
        vec![
            load(1, 0),
            store(1, cast(ScalarExpr::Local(1), ElementKind::F16)),
        ],
    );
    agree(
        "i16_to_f16",
        &k,
        vec![
            KernelArg::Buffer(TypedBuffer::new(
                ElementKind::I16,
                vec![ints.len() as i64],
                vec![1],
                bytes,
            )),
            narrow(ElementKind::F16, &vec![0; ints.len()]),
        ],
    );
}

#[test]
fn a_narrow_float_to_an_integer_agrees() {
    let pats: Vec<u16> = (0..=u16::MAX).step_by(3).collect();
    let k = looped(
        "f16_to_i32",
        vec![
            Param::buffer("in", ElementKind::F16, BufferRole::Read, Some(1)),
            Param::buffer("out", I32, BufferRole::Write, Some(1)),
        ],
        vec![ElementKind::F16, F32],
        pats.len(),
        vec![
            load(1, 0),
            // (widen to F32 first, into a local: the oracle's integer casts read an F32 local)
            Stmt::Let {
                local: 2,
                init: cast(ScalarExpr::Local(1), F32),
            },
            store(
                1,
                ScalarExpr::Cast(
                    Box::new(ScalarExpr::Local(2)),
                    CastSpec {
                        to: I32,
                        round: RoundMode::TowardZero,
                        sat: Saturation::Clamp,
                    },
                ),
            ),
        ],
    );
    let out = KernelArg::Buffer(TypedBuffer::new(
        I32,
        vec![pats.len() as i64],
        vec![1],
        vec![0; pats.len() * 4],
    ));
    agree("f16_to_i32", &k, vec![narrow(ElementKind::F16, &pats), out]);
}

#[test]
fn the_bits_move_unchanged_through_load_store_select_and_masked_access() {
    // NaN payloads and both zeros survive a move: no decode, no encode.
    let pats: Vec<u16> = vec![
        0x7C01, 0x7E00, 0xFE55, 0x8000, 0x0000, 0x7C00, 0xFC00, 0x3C00, 0x0001,
    ];
    let n = pats.len();
    let pick = ScalarExpr::Select(
        Box::new(ScalarExpr::Binary(
            BinaryOp::CmpLt,
            Box::new(ScalarExpr::Local(0)),
            Box::new(ci(4)),
        )),
        Box::new(ScalarExpr::Local(1)),
        Box::new(ScalarExpr::Local(2)),
    );
    for (tag, t) in [
        ("move_f16", ElementKind::F16),
        ("move_bf16", ElementKind::Bf16),
    ] {
        let k = looped(
            tag,
            vec![
                Param::buffer("a", t, BufferRole::Read, Some(1)),
                Param::buffer("b", t, BufferRole::Read, Some(1)),
                Param::buffer("out", t, BufferRole::Write, Some(1)),
            ],
            vec![t, t, t],
            n,
            vec![
                load(1, 0),
                load(2, 1),
                Stmt::Let {
                    local: 3,
                    init: pick.clone(),
                },
                store(2, ScalarExpr::Local(3)),
            ],
        );
        let rev: Vec<u16> = pats.iter().rev().copied().collect();
        agree(
            tag,
            &k,
            vec![narrow(t, &pats), narrow(t, &rev), narrow(t, &vec![0; n])],
        );
    }
}

#[test]
fn a_launched_copy_through_shared_memory_agrees() {
    let pats: Vec<u16> = vec![
        0x3C00, 0x4000, 0x7E00, 0x8000, 0x0001, 0xFC00, 0x7BFF, 0x1234,
    ];
    for (tag, t) in [
        ("shared_f16", ElementKind::F16),
        ("shared_bf16", ElementKind::Bf16),
    ] {
        agree_launch(
            tag,
            &mirrored_copy(tag, t),
            vec![narrow(t, &pats), narrow(t, &[0; 8])],
        );
    }
}

/// `2 * v + k`, in `U32`.
fn pair_index(v: ScalarExpr, k: i64) -> ScalarExpr {
    ScalarExpr::Add(
        Box::new(ScalarExpr::Mul(
            Box::new(v),
            Box::new(ScalarExpr::ConstInt(2, U32)),
        )),
        Box::new(ScalarExpr::ConstInt(k, U32)),
    )
}

/// Thread `t` of 4 copies `in[2t..2t+2]` (4 bytes) into shared memory, waits, and after a barrier writes the mirror
/// thread's pair to `out[2t..2t+2]`.
fn mirrored_copy(tag: &str, t: ElementKind) -> KernelFn {
    let tid = ScalarExpr::Exec(ExecId::ThreadIdx(0));
    let mirror = ScalarExpr::Sub(
        Box::new(ScalarExpr::ConstInt(3, U32)),
        Box::new(tid.clone()),
    );
    let mut body = vec![
        Stmt::CopyAsync {
            dst: Place::Alloc {
                alloc: 0,
                index: pair_index(tid.clone(), 0),
            },
            src: at(0, pair_index(tid.clone(), 0)),
            elems: 2,
        },
        Stmt::AsyncCommit,
        Stmt::AsyncWait { pending: 0 },
        Stmt::Barrier {
            scope: Scope::Block,
            kind: BarrierKind::Full,
        },
    ];
    for k in 0..2 {
        body.push(Stmt::Load {
            local: k as u32,
            place: Place::Alloc {
                alloc: 0,
                index: pair_index(mirror.clone(), k),
            },
        });
        body.push(Stmt::Store {
            place: at(1, pair_index(tid.clone(), k)),
            value: ScalarExpr::Local(k as u32),
        });
    }
    let params = vec![
        Param::buffer("in", t, BufferRole::Read, Some(1)),
        Param::buffer("out", t, BufferRole::Write, Some(1)),
    ];
    KernelFn::new(
        tag,
        F32,
        params,
        vec![LocalDecl::scalar(t), LocalDecl::scalar(t)],
        Block(body),
    )
    .with_allocs(vec![AllocDecl::new(
        "tile",
        MemorySpace::Shared,
        t,
        Extent::Static(8),
    )])
    .with_launch(LaunchShape::new([4, 1, 1], [1, 1, 1]))
}

#[test]
fn a_narrow_scalar_parameter_is_moved_and_cast() {
    let k = KernelFn::new(
        "scalar_param",
        F32,
        vec![
            Param::scalar("s", ElementKind::F16, 0),
            Param::buffer("out", F32, BufferRole::Write, Some(1)),
        ],
        vec![LocalDecl::scalar(ElementKind::F16)],
        Block(vec![Stmt::Store {
            place: at(1, ci(0)),
            value: cast(ScalarExpr::Local(0), F32),
        }]),
    );
    // the driver binds only int and float scalars, so this kernel is checked for its spelling, not run
    let g = CpuC.lower_kernel(&k);
    assert!(g.is_ok(), "{g:?}");
    assert!(g.unwrap().source.contains("unp_f16_dec(l0)"));
}

fn refusal(k: &KernelFn) -> String {
    match CpuC.lower_kernel(k) {
        Err(e) => e.to_string(),
        Ok(_) => panic!("the kernel was lowered, expected a refusal"),
    }
}

#[test]
fn arithmetic_or_a_comparison_on_a_narrow_value_is_refused_not_miscomputed() {
    let t = ElementKind::F16;
    for (tag, e) in [
        (
            "add",
            ScalarExpr::Add(
                Box::new(ScalarExpr::Local(1)),
                Box::new(ScalarExpr::Local(1)),
            ),
        ),
        (
            "cmp",
            ScalarExpr::Binary(
                BinaryOp::CmpLt,
                Box::new(ScalarExpr::Local(1)),
                Box::new(ScalarExpr::Local(1)),
            ),
        ),
        (
            "cast_then_add",
            ScalarExpr::Add(
                Box::new(cast(ScalarExpr::Local(1), t)),
                Box::new(cast(ScalarExpr::Const(1.0), t)),
            ),
        ),
    ] {
        let k = looped(
            tag,
            vec![
                Param::buffer("in", t, BufferRole::Read, Some(1)),
                Param::buffer("out", F32, BufferRole::Write, Some(1)),
            ],
            vec![t],
            4,
            vec![load(1, 0), store(1, cast(e, F32))],
        );
        // the checker admits each of these kernels, so the refusal is the emitter's own, and it says what to do
        assert_eq!(check_kernel(&k), Ok(()), "{tag}: well formed");
        let msg = refusal(&k);
        assert!(msg.contains("Cast it to F32 first"), "{tag}: {msg}");
    }
}

#[test]
fn a_cast_that_would_round_twice_is_a_typed_decline() {
    for (tag, from) in [("i32", I32), ("f64", ElementKind::F64), ("u32", U32)] {
        let k = looped(
            tag,
            vec![
                Param::buffer("in", from, BufferRole::Read, Some(1)),
                Param::buffer("out", ElementKind::F16, BufferRole::Write, Some(1)),
            ],
            vec![from],
            2,
            vec![
                load(1, 0),
                store(1, cast(ScalarExpr::Local(1), ElementKind::F16)),
            ],
        );
        let msg = refusal(&k);
        assert!(msg.contains("would round twice"), "{tag}: {msg}");
    }
}

#[test]
fn the_corpus_catches_seeded_mutants_of_the_emitted_codec_and_carrier() {
    let mut xs: Vec<f32> = around((0..=u16::MAX).step_by(331), f16_value);
    xs.extend([
        65504.0,
        65519.0,
        65520.0,
        65530.0,
        65536.0,
        69000.0,
        70000.0,
        1.0e9,
        -65520.0,
        6.0e-8,
        3.0e-8,
        2.0e-8,
        f32::NAN,
        -f32::NAN,
    ]);
    let xs = &xs[..];
    let k = encode(ElementKind::F16, xs.len());
    let args = || vec![floats(xs), narrow(ElementKind::F16, &vec![0; xs.len()])];
    let base = run_c_mutated("narrow_base", &k, &args(), |s| s).expect("a C compiler");
    for (what, from, to) in [
        (
            "tie-to-even flipped",
            "q - f == 0.5f && ((long)f & 1)",
            "q - f == 0.5f && !((long)f & 1)",
        ),
        ("overflow edge moved", "a >= 65520.0f", "a >= 70000.0f"),
        ("subnormal scale", "ldexpf(a, 24)", "ldexpf(a, 23)"),
        (
            "sign dropped",
            "sign = (x < 0.0f ||",
            "sign = (0 && x < 0.0f ||",
        ),
    ] {
        let got = run_c_mutated("narrow_mut", &k, &args(), |s| {
            assert_eq!(s.matches(from).count(), 1, "{what}: anchor");
            s.replacen(from, to, 1)
        })
        .expect("a C compiler");
        assert_ne!(got.outs, base.outs, "mutant survived: {what}");
    }
}

#[test]
fn a_cast_to_a_narrow_float_declines_a_mode_the_codec_cannot_honour() {
    let spec = |round, sat| CastSpec {
        to: ElementKind::F16,
        round,
        sat,
    };
    for (tag, s) in [
        ("toward_zero", spec(RoundMode::TowardZero, Saturation::Wrap)),
        (
            "clamp_finite",
            spec(RoundMode::NearestEven, Saturation::ClampFinite),
        ),
    ] {
        let k = looped(
            tag,
            vec![
                Param::buffer("in", F32, BufferRole::Read, Some(1)),
                Param::buffer("out", ElementKind::F16, BufferRole::Write, Some(1)),
            ],
            vec![F32],
            2,
            vec![
                load(1, 0),
                store(1, ScalarExpr::Cast(Box::new(ScalarExpr::Local(1)), s)),
            ],
        );
        let msg = refusal(&k);
        assert!(
            msg.contains("NearestEven") || msg.contains("Saturation::Wrap"),
            "{tag}: {msg}"
        );
    }
}
