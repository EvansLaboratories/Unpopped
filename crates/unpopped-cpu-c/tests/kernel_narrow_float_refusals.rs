// SPDX-License-Identifier: MIT OR Apache-2.0
//! **What the C reference refuses of `F16` and `Bf16`** (ROADMAP U14; the spelling is `kernel_narrow_float_emit_agrees`): a
//! narrow float is a bit pattern in C, so arithmetic or a comparison on one, and a mode the
//! codec cannot honour are typed refusals that say what to do instead, never a miscomputed number. The checker admits
//! every kernel here, so each refusal is the emitter's own.

use unpopped::backend::Backend;
use unpopped::ir::{CastSpec, RoundMode, Saturation, ScalarExpr};
use unpopped::kernel::{Block, BufferRole, KernelFn, LocalDecl, Param, Place, Stmt};
use unpopped_cpu_c::CpuC;
use unpopped_vocab::ElementKind;

const F32: ElementKind = ElementKind::F32;
const I32: ElementKind = ElementKind::I32;

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
    assert!(g.unwrap().source.contains("unpopped_f16_load(l0)"));
}

fn refusal(k: &KernelFn) -> String {
    match CpuC.lower_kernel(k) {
        Err(e) => e.to_string(),
        Ok(_) => panic!("the kernel was lowered, expected a refusal"),
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
