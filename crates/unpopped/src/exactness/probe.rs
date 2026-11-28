// SPDX-License-Identifier: MIT OR Apache-2.0
//! **Shared subnormal probes** for the measured-exactness ledger (ROADMAP U59, `docs/design-kernel-fidelity.md`).
//!
//! Unpopped never probes a device; an emitter lane's harness does. These are the **operand sets** that harness should
//! run, with the expected result each must give on a device that keeps IEEE subnormals, so the lanes do not each
//! invent a set that never reaches a subnormal (vulkane's Vulkan pairs did not, so a flush hid until a dedicated case:
//! Metal f32 and bf16 flush, f16 preserves).
//!
//! A probe is data: an operation (`Mul`, `Add`, `MulAdd` = `a * b + c`), raw operand bits at the dtype, and the bits
//! the result must have. The expectation is computed here by **exact arithmetic in `f64`** (every operand set is chosen
//! so the product and the sum are exact in `f64`, and a final single rounding to the dtype follows), so no probe depends
//! on a device or on the oracle's evaluation order. Each set exercises one way a device can flush: a subnormal
//! **operand**, a subnormal **result**, and a product that **underflows to a subnormal and is added back**. Infinity
//! and NaN controls are included so a mismatch on them is told apart from a flush.
//!
//! The dtypes are `F32`, `F16` and `Bf16`; another dtype has no probes (`None`).

use unpopped_vocab::ElementKind;

/// The operation a probe runs.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum ProbeOp {
    /// `a * b` (two operands).
    Mul,
    /// `a + b` (two operands).
    Add,
    /// `a * b + c` as **two operations, each rounded** (three operands): the product is rounded to the dtype, then the sum.
    /// What a compiler that does not contract (`contract(off)`, `-ffp-contract=off`) must give.
    MulAdd,
    /// `fma(a, b, c)`: **one** rounding of the exact `a * b + c` (three operands). Differs from [`ProbeOp::MulAdd`] where
    /// the rounded product loses what the fused one keeps (`s * 0.5 + s` below), so a harness must run the one it means.
    Fma,
}

/// What a probe's result must be on a device that preserves subnormals.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum Expected {
    /// These exact bits.
    Bits(u64),
    /// Any NaN (the payload is not specified).
    Nan,
}

/// One operand set and its expected result.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SubnormalProbe {
    /// A stable name for records and messages.
    pub name: &'static str,
    /// The element type of every operand and of the result.
    pub dtype: ElementKind,
    /// The operation.
    pub op: ProbeOp,
    /// Raw operand bits at the dtype (low bits of each `u64`): two for `Mul`/`Add`, three for `MulAdd`.
    pub operands: Vec<u64>,
    /// The bits the result must have where subnormals are preserved.
    pub expected: Expected,
    /// A subnormal reaches the device as an **operand**.
    pub subnormal_operand: bool,
    /// The correct result is itself subnormal (nonzero).
    pub subnormal_result: bool,
}

/// `(mantissa bits, minimum normal exponent, exponent bits)` of a probed dtype.
const fn format(dtype: ElementKind) -> Option<(i32, i32, u32)> {
    match dtype {
        ElementKind::F32 => Some((23, -126, 8)),
        ElementKind::F16 => Some((10, -14, 5)),
        ElementKind::Bf16 => Some((7, -126, 8)),
        _ => None,
    }
}

/// The bits of the finite, exactly representable value `v` (positive or zero) at `dtype`.
fn encode(dtype: ElementKind, v: f64) -> u64 {
    let (mbits, emin, ebits) = format(dtype).expect("a probed dtype");
    if v == 0.0 {
        return 0;
    }
    let e = v.log2().floor() as i32;
    let bias = (1 << (ebits - 1)) - 1;
    if e < emin {
        // subnormal: v = m * 2^(emin - mbits)
        let m = v / f64::powi(2.0, emin - mbits);
        assert!(
            m.fract() == 0.0 && m >= 1.0,
            "{v} is not a subnormal of {dtype:?}"
        );
        return m as u64;
    }
    let frac = v / f64::powi(2.0, e) - 1.0;
    let m = frac * f64::powi(2.0, mbits);
    assert!(m.fract() == 0.0, "{v} is not representable at {dtype:?}");
    (((e + bias) as u64) << mbits) | m as u64
}

/// `v` rounded once to `dtype` (ties to even), by exact scaling.
fn round_to(dtype: ElementKind, v: f64) -> f64 {
    let (mbits, emin, _) = format(dtype).expect("a probed dtype");
    if v == 0.0 {
        return 0.0;
    }
    let e = (v.abs().log2().floor() as i32).max(emin);
    let quantum = f64::powi(2.0, e - mbits);
    (v / quantum).round_ties_even() * quantum
}

fn probe(name: &'static str, dtype: ElementKind, op: ProbeOp, operands: &[f64]) -> SubnormalProbe {
    let (mbits, emin, _) = format(dtype).expect("a probed dtype");
    let smallest_normal = f64::powi(2.0, emin);
    let is_sub = |v: f64| v != 0.0 && v.abs() < smallest_normal;
    let exact = match op {
        ProbeOp::Mul => operands[0] * operands[1],
        ProbeOp::Add => operands[0] + operands[1],
        ProbeOp::MulAdd => round_to(dtype, operands[0] * operands[1]) + operands[2],
        ProbeOp::Fma => operands[0] * operands[1] + operands[2],
    };
    let result = round_to(dtype, exact);
    // the operand sets are built so the `f64` arithmetic above is exact before the single rounding
    let _ = mbits;
    SubnormalProbe {
        name,
        dtype,
        op,
        operands: operands.iter().map(|v| encode(dtype, *v)).collect(),
        expected: Expected::Bits(encode(dtype, result)),
        subnormal_operand: operands.iter().any(|v| is_sub(*v)),
        subnormal_result: is_sub(result),
    }
}

/// The two controls with a special operand: `inf + 1 = inf` and `NaN * 2 = NaN` (neither involves a subnormal).
fn special_controls(dtype: ElementKind, (mbits, ebits): (i32, u32)) -> [SubnormalProbe; 2] {
    let inf_bits = ((1u64 << ebits) - 1) << mbits;
    let nan_bits = inf_bits | (1 << (mbits - 1));
    let control = |name, op, operands, expected| SubnormalProbe {
        name,
        dtype,
        op,
        operands,
        expected,
        subnormal_operand: false,
        subnormal_result: false,
    };
    [
        control(
            "control_infinity",
            ProbeOp::Add,
            vec![inf_bits, encode(dtype, 1.0)],
            Expected::Bits(inf_bits),
        ),
        control(
            "control_nan",
            ProbeOp::Mul,
            vec![nan_bits, encode(dtype, 2.0)],
            Expected::Nan,
        ),
    ]
}

/// The probes for `dtype`, or `None` for a dtype that has none.
///
/// With `s` the smallest subnormal, `n` the smallest normal and `big` a larger subnormal (`3 s`):
///
/// * `mul_add_subnormal_operand`: `s * 0.5 + s` as two roundings (the product ties to zero, so the sum is `s`; a device
///   that flushes the operand gives zero), the case that exposed Metal f32 and bf16;
/// * `fma_subnormal_operand`: the same operands fused (the exact `1.5 s` rounds to `2 s`; a flush gives zero);
/// * `mul_subnormal_operand_exact`: `big * 2` (a subnormal operand, a subnormal result `6 s`);
/// * `mul_subnormal_operand_normal_result`: `big * 2^(m+1)` (a subnormal operand and a **normal** result: only an
///   operand flush can change it);
/// * `mul_normal_to_subnormal_result`: `n * 0.5` (a normal operand, a subnormal result: flush-to-zero of the result);
/// * `add_subnormal_operands`: `s + big` (subnormal operands and result);
/// * `mul_add_underflow_then_add_back`: `n * 0.25 + big` (the product underflows to a subnormal and is added back;
///   a device that flushes the intermediate gives `big`);
/// * controls: `inf + 1 = inf`, `NaN * 2 = NaN`, and a plain normal `1.5 * 2 + 1 = 4`.
#[must_use]
pub fn subnormal_probes(dtype: ElementKind) -> Option<Vec<SubnormalProbe>> {
    let (mbits, emin, ebits) = format(dtype)?;
    let s = f64::powi(2.0, emin - mbits);
    let n = f64::powi(2.0, emin);
    let big = 3.0 * s;
    let mut v = vec![
        probe(
            "mul_add_subnormal_operand",
            dtype,
            ProbeOp::MulAdd,
            &[s, 0.5, s],
        ),
        probe("fma_subnormal_operand", dtype, ProbeOp::Fma, &[s, 0.5, s]),
        probe(
            "mul_subnormal_operand_exact",
            dtype,
            ProbeOp::Mul,
            &[big, 2.0],
        ),
        probe(
            "mul_subnormal_operand_normal_result",
            dtype,
            ProbeOp::Mul,
            &[big, f64::powi(2.0, mbits + 1)],
        ),
        probe(
            "mul_normal_to_subnormal_result",
            dtype,
            ProbeOp::Mul,
            &[n, 0.5],
        ),
        probe("add_subnormal_operands", dtype, ProbeOp::Add, &[s, big]),
        probe(
            "mul_add_underflow_then_add_back",
            dtype,
            ProbeOp::MulAdd,
            &[n, 0.25, big],
        ),
        probe("control_normal", dtype, ProbeOp::MulAdd, &[1.5, 2.0, 1.0]),
    ];
    v.extend(special_controls(dtype, (mbits, ebits)));
    Some(v)
}
