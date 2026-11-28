// SPDX-License-Identifier: MIT OR Apache-2.0
//! **The shared subnormal probes** (ROADMAP U59): each probe's expected bits are checked against the **host's own
//! arithmetic** (native `f32`, the `half` crate for `F16`/`Bf16`), a method independent of the `f64` scaling the probes
//! are built with; a simulated flush-to-zero device must disagree with the subnormal probes and with none of the
//! controls, so the set can see the failure it exists for.

use half::{bf16, f16};
use unpopped::exactness::probe::{Expected, ProbeOp, SubnormalProbe, subnormal_probes};
use unpopped_vocab::ElementKind;

const DTYPES: [ElementKind; 3] = [ElementKind::F32, ElementKind::F16, ElementKind::Bf16];

fn decode(dtype: ElementKind, bits: u64) -> f32 {
    match dtype {
        ElementKind::F32 => f32::from_bits(bits as u32),
        ElementKind::F16 => f16::from_bits(bits as u16).to_f32(),
        ElementKind::Bf16 => bf16::from_bits(bits as u16).to_f32(),
        _ => unreachable!("a probed dtype"),
    }
}

/// The bits of `v` rounded once to `dtype` (the values here are exact in `f32`, so the `f32` hop loses nothing).
fn encode(dtype: ElementKind, v: f32) -> u64 {
    match dtype {
        ElementKind::F32 => u64::from(v.to_bits()),
        ElementKind::F16 => u64::from(f16::from_f32(v).to_bits()),
        ElementKind::Bf16 => u64::from(bf16::from_f32(v).to_bits()),
        _ => unreachable!("a probed dtype"),
    }
}

fn is_subnormal(dtype: ElementKind, v: f32) -> bool {
    let min_normal = match dtype {
        ElementKind::F16 => f32::powi(2.0, -14),
        _ => f32::MIN_POSITIVE,
    };
    v != 0.0 && v.abs() < min_normal
}

/// What a device would give: `flush` turns every subnormal operand and a subnormal result to zero.
fn run(p: &SubnormalProbe, flush: bool) -> u64 {
    let ops: Vec<f32> = p
        .operands
        .iter()
        .map(|b| {
            let v = decode(p.dtype, *b);
            if flush && is_subnormal(p.dtype, v) {
                0.0
            } else {
                v
            }
        })
        .collect();
    let exact = match p.op {
        ProbeOp::Mul => ops[0] * ops[1],
        ProbeOp::Add => ops[0] + ops[1],
        // two roundings: the product rounded to the dtype first (the host's `f32` hop is exact for these sets)
        ProbeOp::MulAdd => decode(p.dtype, encode(p.dtype, ops[0] * ops[1])) + ops[2],
        // one rounding: the host's fused multiply-add (exact in `f32` for these sets at the narrow dtypes)
        ProbeOp::Fma => ops[0].mul_add(ops[1], ops[2]),
        // `ProbeOp` is `non_exhaustive`: a probe kind this host model does not know is a test to extend, not to skip
        other => panic!("the host model has no arithmetic for {other:?}"),
    };
    // the intermediate a flushing device would also flush (a product that underflows before the add)
    let exact = if flush
        && matches!(p.op, ProbeOp::MulAdd | ProbeOp::Fma)
        && is_subnormal(p.dtype, ops[0] * ops[1])
    {
        ops[2]
    } else {
        exact
    };
    let result = encode(p.dtype, exact);
    if flush && is_subnormal(p.dtype, decode(p.dtype, result)) {
        encode(p.dtype, 0.0)
    } else {
        result
    }
}

fn agrees(p: &SubnormalProbe, got: u64) -> bool {
    match p.expected {
        Expected::Bits(b) => got == b,
        Expected::Nan => decode(p.dtype, got).is_nan(),
        other => panic!("no agreement rule for {other:?}"),
    }
}

#[test]
fn every_probe_matches_the_hosts_own_arithmetic() {
    for dtype in DTYPES {
        let probes = subnormal_probes(dtype).expect("probes");
        assert_eq!(probes.len(), 10, "{dtype:?}");
        for p in &probes {
            let got = if p.name == "control_nan" {
                encode(dtype, f32::NAN)
            } else {
                run(p, false)
            };
            assert!(
                agrees(p, got),
                "{dtype:?} {}: host gave {got:#x}, probe expects {:?}",
                p.name,
                p.expected
            );
        }
    }
}

#[test]
fn a_flushing_device_disagrees_with_the_subnormal_probes_and_with_no_control() {
    for dtype in DTYPES {
        let probes = subnormal_probes(dtype).unwrap();
        for p in &probes {
            let touches = p.subnormal_operand || p.subnormal_result;
            let flushed_ok = agrees(p, run(p, true));
            if p.name.starts_with("control") {
                assert!(
                    flushed_ok,
                    "{dtype:?} {}: a flush must not move a control",
                    p.name
                );
                assert!(!touches);
            } else {
                assert!(touches, "{dtype:?} {} touches a subnormal", p.name);
                assert!(
                    !flushed_ok,
                    "{dtype:?} {}: a flushing device must disagree",
                    p.name
                );
            }
        }
    }
}

#[test]
fn the_probes_name_each_way_to_flush_and_a_dtype_without_probes_gives_none() {
    for dtype in DTYPES {
        let probes = subnormal_probes(dtype).unwrap();
        let has = |f: &dyn Fn(&SubnormalProbe) -> bool| probes.iter().any(f);
        assert!(
            has(&|p| p.subnormal_operand && !p.subnormal_result),
            "{dtype:?}: operand only"
        );
        assert!(
            has(&|p| !p.subnormal_operand && p.subnormal_result),
            "{dtype:?}: result only"
        );
        assert!(has(&|p| p.name == "mul_add_underflow_then_add_back"));
        let mut names: Vec<_> = probes.iter().map(|p| p.name).collect();
        names.sort_unstable();
        names.dedup();
        assert_eq!(names.len(), probes.len(), "names are unique");
    }
    assert!(subnormal_probes(ElementKind::F64).is_none());
    assert!(subnormal_probes(ElementKind::I32).is_none());
}
