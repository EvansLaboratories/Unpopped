//! Measured exactness: what an exhaustive run on one device said about one body.
//!
//! # Why this exists
//!
//! Exact (`correctly_rounded`) is a claim about **arithmetic**, and arithmetic is done by a
//! driver, not by a chip. The `unpopped-harness` run of 2026-10-21 compared the Slang `relu`
//! against the cpu-c reference over all 2^32 f32 inputs: by default both an AMD Radeon 610M and an
//! RTX 4070 Laptop differ on negative subnormals (the device flushed them), and with
//! `-denorm-mode-fp32 preserve` both are exact. NVIDIA reports `shaderDenormPreserveFloat32 =
//! false` yet honours the mode, so the Vulkan property bits are not a usable gate: **only the
//! measurement is**.
//!
//! This module is the place a measurement is *recorded*. [`crate::contract::AccuracyKey`] is the
//! place it is *read*. Unpopped never probes a device itself: a record is produced by a harness
//! that ran the kernel, and handed in.
//!
//! # What a record is a measurement of
//!
//! A [`DeviceStamp`]: the device, the runtime, **the driver version and the compiler arguments**.
//! A record is a measurement of a driver and a flag set, not of a chip, so a changed stamp is a
//! different device and no record applies: Exact declines again until it is re-measured.
//!
//! # Coverage is exactly what was run, with no extrapolation
//!
//! A record names one body at one dtype. A `relu` record says nothing about `add`, and an f32
//! record says nothing about f16. Exhaustive 2^32 testing only exists for **unary f32** bodies:
//! a binary or f64 body has a record only if someone ran a measurement for it (a sampled run can
//! never accept, so a sampled run produces a [`ExactnessVerdict::Flushes`] or nothing, never an
//! [`ExactnessVerdict::Exact`]). With no record the answer is the one the contract gives without
//! this module.
//!
//! # The body key
//!
//! A body is keyed by [`body_key`]: the serde-JSON text the IR already round-trips through
//! (`NaN` and the infinities are tagged strings, so they key by identity). That text is a
//! contract of this module: a test pins it, because **a key that changes silently drops every
//! record** and nothing would say so.

use crate::ir::ScalarExpr;
use unpopped_vocab::{ElementKind, HwStamp};

/// What a measurement of one body on one device concluded.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum ExactnessVerdict {
    /// Every input the run covered matched the reference bit for bit. Only an exhaustive run may
    /// say this; the harness is the producer and the one that knows whether it was exhaustive.
    Exact,
    /// At least one input differed from the reference (a flushed subnormal, in the case that
    /// motivated this module). The body is **not** exact on this device with this stamp.
    Flushes,
}

/// The identity of what was measured: a device, a runtime, a driver and the compiler arguments.
///
/// Equality ignores [`HwStamp::captured_unix_s`]: when a measurement was taken says nothing
/// about what it measured. Every other field must match for a record to apply.
#[derive(Clone, Debug)]
pub struct DeviceStamp {
    /// Target, device name, runtime version (and the capture time, which is ignored).
    pub hw: HwStamp,
    /// The driver / ICD version as the platform spells it. Opaque: compared, never parsed.
    pub driver_version: String,
    /// The compiler arguments the kernel was built with (for Slang, the `slangc` arguments such
    /// as `-denorm-mode-fp32 preserve`). Order is part of the identity.
    pub compiler_args: Vec<String>,
}

impl DeviceStamp {
    /// Build a stamp.
    #[must_use]
    pub fn new(hw: HwStamp, driver_version: impl Into<String>, compiler_args: Vec<String>) -> Self {
        Self {
            hw,
            driver_version: driver_version.into(),
            compiler_args,
        }
    }
}

impl PartialEq for DeviceStamp {
    fn eq(&self, other: &Self) -> bool {
        self.hw.target == other.hw.target
            && self.hw.device_name == other.hw.device_name
            && self.hw.runtime_version == other.hw.runtime_version
            && self.driver_version == other.driver_version
            && self.compiler_args == other.compiler_args
    }
}

impl Eq for DeviceStamp {}

/// The canonical key of a body at a dtype: `"{dtype}:{serde-JSON of the body}"`.
///
/// See the module docs: this text is pinned by a test, and a change to it is a breaking change to
/// every stored record.
#[must_use]
pub fn body_key(body: &ScalarExpr, dtype: ElementKind) -> String {
    let json = serde_json::to_string(body).expect("a ScalarExpr always serialises");
    format!("{}:{json}", unpopped_vocab::dtype_token(dtype))
}

/// One measurement: this body, at this dtype, on this stamp, concluded `verdict`.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ExactnessRecord {
    /// What was measured on.
    pub stamp: DeviceStamp,
    /// [`body_key`] of what was measured.
    pub key: String,
    /// What the measurement concluded.
    pub verdict: ExactnessVerdict,
}

impl ExactnessRecord {
    /// Record that `body` at `dtype` on `stamp` concluded `verdict`.
    #[must_use]
    pub fn new(
        stamp: DeviceStamp,
        body: &ScalarExpr,
        dtype: ElementKind,
        verdict: ExactnessVerdict,
    ) -> Self {
        Self {
            stamp,
            key: body_key(body, dtype),
            verdict,
        }
    }
}

/// A set of [`ExactnessRecord`]s.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct ExactnessLedger {
    records: Vec<ExactnessRecord>,
}

impl ExactnessLedger {
    /// An empty ledger: nothing measured, so nothing is claimed.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a record. Records are never merged or replaced: see [`Self::lookup`].
    pub fn push(&mut self, record: ExactnessRecord) {
        self.records.push(record);
    }

    /// How many records the ledger holds.
    #[must_use]
    pub fn len(&self) -> usize {
        self.records.len()
    }

    /// Whether the ledger holds no record.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.records.is_empty()
    }

    /// What measurement says about `body` at `dtype` on exactly `stamp`, or `None` if nothing
    /// was measured on that stamp.
    ///
    /// **Disagreement resolves to [`ExactnessVerdict::Flushes`]**: if two records for the same
    /// stamp and body disagree, one of the runs saw a mismatch, and a run that saw none does not
    /// cancel it. A record for any other stamp (another device, driver, runtime or argument set)
    /// is not consulted.
    #[must_use]
    pub fn lookup(
        &self,
        stamp: &DeviceStamp,
        body: &ScalarExpr,
        dtype: ElementKind,
    ) -> Option<ExactnessVerdict> {
        let key = body_key(body, dtype);
        let mut found = None;
        for r in self
            .records
            .iter()
            .filter(|r| r.key == key && r.stamp == *stamp)
        {
            if r.verdict == ExactnessVerdict::Flushes {
                return Some(ExactnessVerdict::Flushes);
            }
            found = Some(r.verdict);
        }
        found
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::UnaryOp;
    use unpopped_vocab::TargetId;

    fn stamp(driver: &str, args: &[&str], captured: u64) -> DeviceStamp {
        DeviceStamp::new(
            HwStamp {
                target: TargetId::parse("vulkan:sg64.ops-abr.arith-f16.cm-none").unwrap(),
                device_name: "AMD Radeon 610M".into(),
                runtime_version: "1.3.280".into(),
                captured_unix_s: captured,
            },
            driver,
            args.iter().map(|s| (*s).to_string()).collect(),
        )
    }

    fn relu() -> ScalarExpr {
        ScalarExpr::Unary(UnaryOp::Relu, Box::new(ScalarExpr::Input(0)))
    }

    /// The key text is a contract: a change here silently drops every stored record.
    #[test]
    fn the_body_key_text_is_pinned() {
        assert_eq!(
            body_key(&relu(), ElementKind::F32),
            r#"f32:{"Unary":["Relu",{"Input":0}]}"#
        );
    }

    #[test]
    fn a_body_and_a_dtype_are_each_part_of_the_key() {
        let neg = ScalarExpr::Unary(UnaryOp::Neg, Box::new(ScalarExpr::Input(0)));
        assert_ne!(
            body_key(&relu(), ElementKind::F32),
            body_key(&neg, ElementKind::F32)
        );
        assert_ne!(
            body_key(&relu(), ElementKind::F32),
            body_key(&relu(), ElementKind::F16)
        );
    }

    #[test]
    fn nan_constants_key_by_identity() {
        let a = ScalarExpr::Const(f64::NAN);
        assert_eq!(
            body_key(&a, ElementKind::F32),
            body_key(&a.clone(), ElementKind::F32),
            "NaN != NaN under PartialEq, so the key must not route through it"
        );
    }

    #[test]
    fn an_empty_ledger_says_nothing() {
        let l = ExactnessLedger::new();
        assert!(l.is_empty());
        assert_eq!(
            l.lookup(&stamp("24.1", &[], 1), &relu(), ElementKind::F32),
            None
        );
    }

    #[test]
    fn a_record_applies_to_exactly_its_stamp_body_and_dtype() {
        let s = stamp("24.1", &["-denorm-mode-fp32", "preserve"], 1);
        let mut l = ExactnessLedger::new();
        l.push(ExactnessRecord::new(
            s.clone(),
            &relu(),
            ElementKind::F32,
            ExactnessVerdict::Exact,
        ));
        assert_eq!(l.len(), 1);
        assert_eq!(
            l.lookup(&s, &relu(), ElementKind::F32),
            Some(ExactnessVerdict::Exact)
        );
        // The capture time is not part of what was measured.
        let later = stamp("24.1", &["-denorm-mode-fp32", "preserve"], 99);
        assert_eq!(
            l.lookup(&later, &relu(), ElementKind::F32),
            Some(ExactnessVerdict::Exact)
        );
        let neg = ScalarExpr::Unary(UnaryOp::Neg, Box::new(ScalarExpr::Input(0)));
        assert_eq!(
            l.lookup(&s, &neg, ElementKind::F32),
            None,
            "a relu run says nothing of neg"
        );
        assert_eq!(
            l.lookup(&s, &relu(), ElementKind::F16),
            None,
            "an f32 run says nothing of f16"
        );
    }

    /// The known-bad control for the stamp: a measurement of one driver is not a measurement of
    /// the next, and a different flag set is a different kernel.
    #[test]
    fn a_changed_driver_or_flag_set_is_a_different_device() {
        let s = stamp("24.1", &["-denorm-mode-fp32", "preserve"], 1);
        let mut l = ExactnessLedger::new();
        l.push(ExactnessRecord::new(
            s,
            &relu(),
            ElementKind::F32,
            ExactnessVerdict::Exact,
        ));
        let new_driver = stamp("24.2", &["-denorm-mode-fp32", "preserve"], 1);
        let no_flag = stamp("24.1", &[], 1);
        assert_eq!(l.lookup(&new_driver, &relu(), ElementKind::F32), None);
        assert_eq!(l.lookup(&no_flag, &relu(), ElementKind::F32), None);
    }

    #[test]
    fn disagreeing_records_resolve_to_flushes_in_either_order() {
        let s = stamp("24.1", &[], 1);
        for flip in [false, true] {
            let mut l = ExactnessLedger::new();
            let exact = ExactnessRecord::new(
                s.clone(),
                &relu(),
                ElementKind::F32,
                ExactnessVerdict::Exact,
            );
            let flush = ExactnessRecord::new(
                s.clone(),
                &relu(),
                ElementKind::F32,
                ExactnessVerdict::Flushes,
            );
            if flip {
                l.push(flush);
                l.push(exact);
            } else {
                l.push(exact);
                l.push(flush);
            }
            assert_eq!(
                l.lookup(&s, &relu(), ElementKind::F32),
                Some(ExactnessVerdict::Flushes)
            );
        }
    }
}
