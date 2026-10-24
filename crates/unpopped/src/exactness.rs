// SPDX-License-Identifier: MIT OR Apache-2.0
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
//! A [`DeviceStamp`]: the device, the runtime, **the driver version, the compiler version and the
//! compiler arguments**. A record is a measurement of a driver, a compiler and a flag set, not of
//! a chip, so a changed stamp is a different device and no record applies: Exact declines again
//! until it is re-measured.
//!
//! And of **one kernel as compiled**: [`ExactnessRecord::kernel_hash`] names what actually ran
//! (SPIR-V for Vulkan, source for cpu-c). A record applies only to a kernel with the same hash, so
//! re-emitting a body differently can never inherit the old `Exact`.
//!
//! # Coverage is exactly what was run, with no extrapolation
//!
//! A record names one body at one dtype. A `relu` record says nothing about `add`, and an f32
//! record says nothing about f16. A record also says how much it covered
//! ([`ExactnessRecord::coverage`]): **only an [`CoverageKind::Exhaustive`] run can admit**. A
//! [`CoverageKind::Sampled`] `Exact` is kept (it is evidence) but admits nothing, and is reported
//! as [`Withheld::SampledOnly`]. A sampled `Flushes` still declines, because one mismatch is a
//! mismatch. With no record the answer is the one the contract gives without this module.
//!
//! # The body key
//!
//! A body is keyed by [`body_key`]: the serde-JSON text the IR already round-trips through
//! (`NaN` and the infinities are tagged strings, so they key by identity). That text is a
//! contract of this module: a test pins it, because **a key that changes silently drops every
//! record** and nothing would say so.
//!
//! # Admission, and the file
//!
//! [`ExactnessLedger::admit`] turns a ledger, a stamp, the kernel's hash and the caller's
//! [`Requirement`] into an [`Admission`], a typed value the adopter acts on: Unpopped decides and
//! names, it never launches a kernel. [`AdmissionTally`] counts the safe failures, because a
//! decline on "no record" is correct and otherwise invisible. The ledger reads and writes as JSON
//! Lines ([`ExactnessLedger::parse_jsonl`], [`ExactnessLedger::to_jsonl`]); the schema is owned
//! here and nowhere else, and an unreadable line is a counted [`LineError`], never a panic and
//! never silently dropped. Design: `docs/design-per-device-decline.md`.

use std::collections::BTreeMap;

use serde_json::{Map, Value, json};

use crate::ir::ScalarExpr;
use unpopped_vocab::{ElementKind, HwStamp, TargetId};

/// What a measurement of one body on one device concluded.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum ExactnessVerdict {
    /// Every input the run covered matched the reference bit for bit. Only an exhaustive run may
    /// admit (see [`CoverageKind`]); the harness is the producer and the one that knows whether
    /// it was exhaustive.
    Exact,
    /// At least one input differed from the reference (a flushed subnormal, in the case that
    /// motivated this module). The body is **not** exact on this device with this stamp.
    Flushes,
}

/// The identity of what was measured: a device, a runtime, a driver, a compiler and its
/// arguments.
///
/// Equality ignores [`HwStamp::captured_unix_s`]: when a measurement was taken says nothing
/// about what it measured. Every other field must match for a record to apply.
#[derive(Clone, Debug)]
pub struct DeviceStamp {
    /// Target, device name, runtime version (and the capture time, which is ignored).
    pub hw: HwStamp,
    /// The driver / ICD version as the platform spells it. Opaque: compared, never parsed.
    pub driver_version: String,
    /// The compiler's own version (for Slang, what `slangc -v` prints). Opaque. The arguments do
    /// not carry it, and a compiler upgrade changes the code it emits, so it is part of the
    /// identity.
    pub compiler_version: String,
    /// The compiler arguments the kernel was built with (for Slang, the `slangc` arguments such
    /// as `-denorm-mode-fp32 preserve`). Order is part of the identity.
    pub compiler_args: Vec<String>,
}

impl DeviceStamp {
    /// Build a stamp.
    #[must_use]
    pub fn new(
        hw: HwStamp,
        driver_version: impl Into<String>,
        compiler_version: impl Into<String>,
        compiler_args: Vec<String>,
    ) -> Self {
        Self {
            hw,
            driver_version: driver_version.into(),
            compiler_version: compiler_version.into(),
            compiler_args,
        }
    }

    /// Whether `other` is the same device, runtime, driver and compiler, differing at most in the
    /// compiler arguments: the relation under which one argument set can be offered for another.
    fn same_device_and_compiler(&self, other: &Self) -> bool {
        self.hw.target == other.hw.target
            && self.hw.device_name == other.hw.device_name
            && self.hw.runtime_version == other.hw.runtime_version
            && self.driver_version == other.driver_version
            && self.compiler_version == other.compiler_version
    }
}

impl PartialEq for DeviceStamp {
    fn eq(&self, other: &Self) -> bool {
        self.same_device_and_compiler(other) && self.compiler_args == other.compiler_args
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

/// How much of the input space a run covered.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum CoverageKind {
    /// Every input the body has at this dtype. The only coverage that can admit.
    Exhaustive,
    /// A subset (a stride screen, a seeded sample). Evidence of a mismatch, never of exactness.
    Sampled,
}

/// What a run covered: how, how many inputs (or input tuples), and the sampler's seed if it had
/// one.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct Coverage {
    /// Exhaustive or sampled.
    pub kind: CoverageKind,
    /// Inputs (or tuples) compared.
    pub tuples: u64,
    /// The sampler's seed, for a sampled run that had one.
    pub seed: Option<u64>,
}

impl Coverage {
    /// An exhaustive run over `tuples` inputs.
    #[must_use]
    pub fn exhaustive(tuples: u64) -> Self {
        Self {
            kind: CoverageKind::Exhaustive,
            tuples,
            seed: None,
        }
    }

    /// A sampled run over `tuples` inputs.
    #[must_use]
    pub fn sampled(tuples: u64, seed: Option<u64>) -> Self {
        Self {
            kind: CoverageKind::Sampled,
            tuples,
            seed,
        }
    }
}

/// One measurement: this body, at this dtype, on this stamp, concluded `verdict`, for the kernel
/// with this hash.
///
/// `#[non_exhaustive]`: build one with [`ExactnessRecord::new`], so a future field is not a
/// breaking change.
#[derive(Clone, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub struct ExactnessRecord {
    /// What was measured on.
    pub stamp: DeviceStamp,
    /// [`body_key`] of what was measured.
    pub key: String,
    /// What the measurement concluded.
    pub verdict: ExactnessVerdict,
    /// Hash of what actually ran (`sha256:<hex>` of the SPIR-V for Vulkan, of the source for
    /// cpu-c). Opaque here: compared, never computed. A record applies only to a kernel with this
    /// hash.
    pub kernel_hash: String,
    /// Which emitter produced the kernel (`"{crate} {version}"`). Audit only: it is not part of
    /// the lookup, because the kernel hash already names what ran.
    pub emitter_version: String,
    /// How much of the input space the run covered.
    pub coverage: Coverage,
    /// Fields a newer writer added that this version does not know, kept so they are written back.
    pub extra: BTreeMap<String, Value>,
}

impl ExactnessRecord {
    /// Record that `body` at `dtype` on `stamp` concluded `verdict`, for the kernel hashed
    /// `kernel_hash` that `emitter_version` produced, over `coverage`.
    #[must_use]
    pub fn new(
        stamp: DeviceStamp,
        body: &ScalarExpr,
        dtype: ElementKind,
        verdict: ExactnessVerdict,
        kernel_hash: impl Into<String>,
        emitter_version: impl Into<String>,
        coverage: Coverage,
    ) -> Self {
        Self {
            stamp,
            key: body_key(body, dtype),
            verdict,
            kernel_hash: kernel_hash.into(),
            emitter_version: emitter_version.into(),
            coverage,
            extra: BTreeMap::new(),
        }
    }
}

/// Why a claim is withheld: the four ways a ledger fails to admit, kept apart because they have
/// different remedies (measure, change flags, re-measure, run the full space).
#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
#[non_exhaustive]
pub enum Withheld {
    /// No record for this body, dtype and stamp. **Distinct from [`Withheld::Flushes`]**:
    /// collapsing them hides that a device was never measured.
    Unmeasured,
    /// A record for this stamp and kernel saw a mismatch.
    Flushes,
    /// Records exist for this body and stamp, but for a kernel with another hash: the kernel was
    /// re-emitted or rebuilt since it was measured.
    StaleKernel,
    /// The only `Exact` for this kernel is from a sampled run, which cannot admit.
    SampledOnly,
}

/// What a ledger says about one body, kernel and stamp.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum Finding {
    /// An exhaustive run on exactly this stamp and kernel matched bit for bit.
    Exact,
    /// Nothing admits; see [`Withheld`].
    Withheld(Withheld),
}

/// What the caller needs, so that "no" can be either quiet or loud.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum Requirement {
    /// Take the kernel regardless; only the claim in its contract depends on the ledger.
    BestEffort,
    /// The caller needs Exact: a kernel that cannot be shown Exact is declined, and a variant that
    /// can be shown Exact is offered.
    Exact,
}

/// A different argument set that a ledger says is Exact on the same device and compiler.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ArgsOffer {
    /// The compiler arguments to build with.
    pub compiler_args: Vec<String>,
    /// The hash the rebuilt kernel must have for the record to apply. Unpopped never builds the
    /// kernel, so the adopter checks this after building; a different hash means the record does
    /// not cover what was built.
    pub kernel_hash: String,
}

/// The outcome of [`ExactnessLedger::admit`], a value the adopter acts on. Unpopped never
/// turns a decline into an error.
#[derive(Clone, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum Admission {
    /// An exhaustive measurement on this stamp and kernel says Exact.
    Admitted,
    /// [`Requirement::BestEffort`]: the kernel stays and runs, but its contract must not claim
    /// Exact.
    ClaimWithheld(Withheld),
    /// [`Requirement::Exact`] and nothing shows this kernel Exact on this stamp, and no variant
    /// does either.
    Declined(Withheld),
    /// [`Requirement::Exact`], this stamp does not admit, but the same device and compiler is
    /// recorded Exact with these arguments.
    UseArgs(ArgsOffer),
}

impl Admission {
    /// The reason this was not [`Admission::Admitted`], for counting. `None` when admitted.
    #[must_use]
    pub fn withheld(&self) -> Option<Withheld> {
        match self {
            Admission::Admitted => None,
            Admission::ClaimWithheld(w) | Admission::Declined(w) => Some(*w),
            // A variant is offered because the stamp as asked did not admit.
            Admission::UseArgs(_) => None,
        }
    }
}

/// What the `*_with_exactness` entry points read: the stamp of the device the kernel will run on,
/// the hash of the kernel **as built**, and the ledger. The hash is the adopter's to supply because
/// only the adopter has the compiled artifact; a record applies only to a kernel with the same hash.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct Measured<'a> {
    /// The device, driver, compiler and arguments the kernel will run with.
    pub stamp: &'a DeviceStamp,
    /// Hash of the kernel as built (the same spelling the ledger's records use).
    pub kernel_hash: &'a str,
    /// The measurements to read.
    pub ledger: &'a ExactnessLedger,
}

impl<'a> Measured<'a> {
    /// Name what is about to run and what has been measured.
    #[must_use]
    pub fn new(stamp: &'a DeviceStamp, kernel_hash: &'a str, ledger: &'a ExactnessLedger) -> Self {
        Self {
            stamp,
            kernel_hash,
            ledger,
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

    /// Add a record. Records are never merged or replaced: see [`Self::find`].
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

    /// The records, in the order they were pushed.
    #[must_use]
    pub fn records(&self) -> &[ExactnessRecord] {
        &self.records
    }

    /// What measurement says about `body` at `dtype` for the kernel hashed `kernel_hash` on
    /// exactly `stamp`.
    ///
    /// A record counts only when its stamp **and** kernel hash match. Among those:
    ///
    /// - **A `Flushes` dominates**: if records disagree, one run saw a mismatch, and a run that
    ///   saw none does not cancel it. This holds for a sampled run too.
    /// - An `Exact` admits only from an exhaustive run; a sampled one is [`Withheld::SampledOnly`].
    /// - Records for the body and stamp under another hash are [`Withheld::StaleKernel`].
    ///
    /// A record for any other stamp (another device, driver, runtime, compiler or argument set)
    /// is not consulted.
    #[must_use]
    pub fn find(
        &self,
        stamp: &DeviceStamp,
        kernel_hash: &str,
        body: &ScalarExpr,
        dtype: ElementKind,
    ) -> Finding {
        let key = body_key(body, dtype);
        let (mut exact, mut sampled, mut stale) = (false, false, false);
        for r in self
            .records
            .iter()
            .filter(|r| r.key == key && r.stamp == *stamp)
        {
            if r.kernel_hash != kernel_hash {
                stale = true;
                continue;
            }
            match (r.verdict, r.coverage.kind) {
                (ExactnessVerdict::Flushes, _) => return Finding::Withheld(Withheld::Flushes),
                (ExactnessVerdict::Exact, CoverageKind::Exhaustive) => exact = true,
                (ExactnessVerdict::Exact, CoverageKind::Sampled) => sampled = true,
            }
        }
        if exact {
            Finding::Exact
        } else if sampled {
            Finding::Withheld(Withheld::SampledOnly)
        } else if stale {
            Finding::Withheld(Withheld::StaleKernel)
        } else {
            Finding::Withheld(Withheld::Unmeasured)
        }
    }

    /// [`Self::find`] as the pair the contract reads: `Some(Exact)` only for [`Finding::Exact`],
    /// `Some(Flushes)` only for a measured mismatch, and `None` for everything else. An unmeasured,
    /// stale or sampled-only kernel therefore claims nothing, exactly as an empty ledger does.
    #[must_use]
    pub fn lookup(
        &self,
        stamp: &DeviceStamp,
        kernel_hash: &str,
        body: &ScalarExpr,
        dtype: ElementKind,
    ) -> Option<ExactnessVerdict> {
        match self.find(stamp, kernel_hash, body, dtype) {
            Finding::Exact => Some(ExactnessVerdict::Exact),
            Finding::Withheld(Withheld::Flushes) => Some(ExactnessVerdict::Flushes),
            Finding::Withheld(_) => None,
        }
    }

    /// Decide what to do with this kernel on this stamp. See [`Admission`] and [`Requirement`].
    ///
    /// A variant is offered only under [`Requirement::Exact`], only from a record on the same
    /// device, runtime, driver and compiler with different arguments, and only if that record
    /// would itself admit (its own stamp and hash, exhaustive, not outweighed by a `Flushes`).
    /// The first such record in ledger order wins, so the offer is deterministic.
    #[must_use]
    pub fn admit(
        &self,
        stamp: &DeviceStamp,
        kernel_hash: &str,
        body: &ScalarExpr,
        dtype: ElementKind,
        requirement: Requirement,
    ) -> Admission {
        let why = match self.find(stamp, kernel_hash, body, dtype) {
            Finding::Exact => return Admission::Admitted,
            Finding::Withheld(w) => w,
        };
        if requirement != Requirement::Exact {
            return Admission::ClaimWithheld(why);
        }
        let key = body_key(body, dtype);
        let offer = self.records.iter().find(|r| {
            r.key == key
                && r.verdict == ExactnessVerdict::Exact
                && r.stamp.same_device_and_compiler(stamp)
                && r.stamp.compiler_args != stamp.compiler_args
                && self.find(&r.stamp, &r.kernel_hash, body, dtype) == Finding::Exact
        });
        match offer {
            Some(r) => Admission::UseArgs(ArgsOffer {
                compiler_args: r.stamp.compiler_args.clone(),
                kernel_hash: r.kernel_hash.clone(),
            }),
            None => Admission::Declined(why),
        }
    }
}

/// Counts of the safe failures, per device stamp and reason. A kernel declined for want of a
/// record looks identical to one declined after a failed measurement unless something counts the
/// difference.
#[derive(Clone, Debug, Default)]
pub struct AdmissionTally {
    counts: Vec<(DeviceStamp, Withheld, u64)>,
}

impl AdmissionTally {
    /// An empty tally.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Count one decision on `stamp`. Admissions and offered variants count nothing: the tally is
    /// of withheld claims, and an offer is reported by the value itself.
    pub fn observe(&mut self, stamp: &DeviceStamp, admission: &Admission) {
        if let Some(w) = admission.withheld() {
            self.add(stamp, w);
        }
    }

    /// Count one withheld claim on `stamp` for `why`. Use this for a decision that did not go
    /// through [`ExactnessLedger::admit`] (the contract entry points withhold without an
    /// [`Admission`]).
    pub fn add(&mut self, stamp: &DeviceStamp, why: Withheld) {
        match self
            .counts
            .iter_mut()
            .find(|(s, w, _)| s == stamp && *w == why)
        {
            Some((_, _, n)) => *n += 1,
            None => self.counts.push((stamp.clone(), why, 1)),
        }
    }

    /// How many times `why` was counted on `stamp`.
    #[must_use]
    pub fn count(&self, stamp: &DeviceStamp, why: Withheld) -> u64 {
        self.counts
            .iter()
            .find(|(s, w, _)| s == stamp && *w == why)
            .map_or(0, |(_, _, n)| *n)
    }

    /// Every `(stamp, reason, count)`, in first-seen order.
    #[must_use]
    pub fn entries(&self) -> &[(DeviceStamp, Withheld, u64)] {
        &self.counts
    }

    /// The total across every stamp and reason.
    #[must_use]
    pub fn total(&self) -> u64 {
        self.counts.iter().map(|(_, _, n)| *n).sum()
    }
}

// ---------------------------------------------------------------------------------------------
// The file: JSON Lines, one record per line, `schema` first.
// ---------------------------------------------------------------------------------------------

/// The `schema` value this version writes and reads.
pub const SCHEMA: &str = "unpopped-exactness-ledger/1";

/// Why a line was not read.
#[derive(Clone, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum LineError {
    /// The line is not a JSON object.
    NotJson(String),
    /// A well-formed line from a schema this version does not know.
    UnknownSchema(String),
    /// A required field is missing or has the wrong type.
    Field(&'static str),
}

/// A ledger read from text, and the lines that were not read.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct LedgerRead {
    /// The records that parsed, in file order.
    pub ledger: ExactnessLedger,
    /// `(1-based line number, why)` for every non-blank line that was not read. Never empty by
    /// omission: a line is either in `ledger` or here.
    pub rejected: Vec<(usize, LineError)>,
}

impl ExactnessRecord {
    /// The record as one JSON line, without its newline. `schema` is first and the order is
    /// fixed; fields in [`Self::extra`] follow.
    #[must_use]
    pub fn to_line(&self) -> String {
        let st = &self.stamp;
        let verdict = match self.verdict {
            ExactnessVerdict::Exact => "exact",
            ExactnessVerdict::Flushes => "flushes",
        };
        let kind = match self.coverage.kind {
            CoverageKind::Exhaustive => "exhaustive",
            CoverageKind::Sampled => "sampled",
        };
        // serde_json's map is sorted, so the line is assembled field by field to keep `schema`
        // first and the order stable.
        let mut fields: Vec<(&str, Value)> = vec![
            ("schema", json!(SCHEMA)),
            ("target", json!(st.hw.target.as_str())),
            ("device_name", json!(st.hw.device_name)),
            ("runtime_version", json!(st.hw.runtime_version)),
            ("driver_version", json!(st.driver_version)),
            ("compiler_args", json!(st.compiler_args)),
            ("compiler_version", json!(st.compiler_version)),
            ("key", json!(self.key)),
            ("verdict", json!(verdict)),
            ("kernel_hash", json!(self.kernel_hash)),
            ("emitter_version", json!(self.emitter_version)),
            (
                "coverage",
                json!({
                    "kind": kind,
                    "tuples": self.coverage.tuples,
                    "seed": self.coverage.seed,
                }),
            ),
            ("captured_unix_s", json!(st.hw.captured_unix_s)),
        ];
        fields.extend(self.extra.iter().map(|(k, v)| (k.as_str(), v.clone())));
        let body: Vec<String> = fields
            .into_iter()
            .map(|(k, v)| format!("{}:{v}", Value::from(k)))
            .collect();
        format!("{{{}}}", body.join(","))
    }

    /// Read one line written by [`Self::to_line`].
    ///
    /// # Errors
    /// A [`LineError`] naming why: not JSON, a schema this version does not know, or the first
    /// required field that is missing or mistyped.
    pub fn parse_line(line: &str) -> Result<Self, LineError> {
        let v: Value = serde_json::from_str(line).map_err(|e| LineError::NotJson(e.to_string()))?;
        let Value::Object(mut m) = v else {
            return Err(LineError::NotJson("not an object".into()));
        };
        match m.remove("schema") {
            Some(Value::String(s)) if s == SCHEMA => {}
            Some(Value::String(s)) => return Err(LineError::UnknownSchema(s)),
            _ => return Err(LineError::Field("schema")),
        }
        let stamp = take_stamp(&mut m)?;
        let key = take_text(&mut m, "key")?;
        let verdict = match take_text(&mut m, "verdict")?.as_str() {
            "exact" => ExactnessVerdict::Exact,
            "flushes" => ExactnessVerdict::Flushes,
            _ => return Err(LineError::Field("verdict")),
        };
        let kernel_hash = take_text(&mut m, "kernel_hash")?;
        let emitter_version = take_text(&mut m, "emitter_version")?;
        let coverage = take_coverage(&mut m)?;
        Ok(Self {
            stamp,
            key,
            verdict,
            kernel_hash,
            emitter_version,
            coverage,
            extra: m.into_iter().collect(),
        })
    }
}

fn take_text(m: &mut Map<String, Value>, f: &'static str) -> Result<String, LineError> {
    match m.remove(f) {
        Some(Value::String(s)) => Ok(s),
        _ => Err(LineError::Field(f)),
    }
}

fn take_u64(m: &mut Map<String, Value>, f: &'static str) -> Result<u64, LineError> {
    m.remove(f)
        .and_then(|t| t.as_u64())
        .ok_or(LineError::Field(f))
}

fn take_args(m: &mut Map<String, Value>) -> Result<Vec<String>, LineError> {
    let Some(Value::Array(a)) = m.remove("compiler_args") else {
        return Err(LineError::Field("compiler_args"));
    };
    a.into_iter()
        .map(|x| match x {
            Value::String(s) => Ok(s),
            _ => Err(LineError::Field("compiler_args")),
        })
        .collect()
}

/// The device, runtime, driver, compiler and arguments fields, and the capture time.
fn take_stamp(m: &mut Map<String, Value>) -> Result<DeviceStamp, LineError> {
    let target =
        TargetId::parse(&take_text(m, "target")?).map_err(|_| LineError::Field("target"))?;
    let device_name = take_text(m, "device_name")?;
    let runtime_version = take_text(m, "runtime_version")?;
    let driver_version = take_text(m, "driver_version")?;
    let compiler_args = take_args(m)?;
    let compiler_version = take_text(m, "compiler_version")?;
    let captured_unix_s = take_u64(m, "captured_unix_s")?;
    let hw = HwStamp {
        target,
        device_name,
        runtime_version,
        captured_unix_s,
    };
    Ok(DeviceStamp::new(
        hw,
        driver_version,
        compiler_version,
        compiler_args,
    ))
}

fn take_coverage(m: &mut Map<String, Value>) -> Result<Coverage, LineError> {
    let Some(Value::Object(mut c)) = m.remove("coverage") else {
        return Err(LineError::Field("coverage"));
    };
    let kind = match c.remove("kind") {
        Some(Value::String(s)) if s == "exhaustive" => CoverageKind::Exhaustive,
        Some(Value::String(s)) if s == "sampled" => CoverageKind::Sampled,
        _ => return Err(LineError::Field("coverage.kind")),
    };
    let tuples = take_u64(&mut c, "tuples").map_err(|_| LineError::Field("coverage.tuples"))?;
    let seed = match c.remove("seed") {
        None | Some(Value::Null) => None,
        Some(s) => Some(s.as_u64().ok_or(LineError::Field("coverage.seed"))?),
    };
    Ok(Coverage { kind, tuples, seed })
}

impl ExactnessLedger {
    /// Read JSON Lines. Blank lines are skipped; every other unreadable line is in
    /// [`LedgerRead::rejected`] with its line number.
    #[must_use]
    pub fn parse_jsonl(text: &str) -> LedgerRead {
        let mut read = LedgerRead::default();
        for (i, line) in text.lines().enumerate() {
            if line.trim().is_empty() {
                continue;
            }
            match ExactnessRecord::parse_line(line) {
                Ok(r) => read.ledger.push(r),
                Err(e) => read.rejected.push((i + 1, e)),
            }
        }
        read
    }

    /// The ledger as JSON Lines, one record per line, each ending in a newline.
    #[must_use]
    pub fn to_jsonl(&self) -> String {
        self.records.iter().fold(String::new(), |mut s, r| {
            s.push_str(&r.to_line());
            s.push('\n');
            s
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::UnaryOp;

    fn stamp(driver: &str, args: &[&str], captured: u64) -> DeviceStamp {
        DeviceStamp::new(
            HwStamp {
                target: TargetId::parse("vulkan:sg64.ops-abr.arith-f16.cm-none").unwrap(),
                device_name: "AMD Radeon 610M".into(),
                runtime_version: "1.3.280".into(),
                captured_unix_s: captured,
            },
            driver,
            "slangc 2026.4",
            args.iter().map(|s| (*s).to_string()).collect(),
        )
    }

    fn relu() -> ScalarExpr {
        ScalarExpr::Unary(UnaryOp::Relu, Box::new(ScalarExpr::Input(0)))
    }

    const H: &str = "sha256:aa";

    fn rec(
        s: &DeviceStamp,
        verdict: ExactnessVerdict,
        hash: &str,
        cov: Coverage,
    ) -> ExactnessRecord {
        ExactnessRecord::new(
            s.clone(),
            &relu(),
            ElementKind::F32,
            verdict,
            hash,
            "unpopped-slang 0.21.0",
            cov,
        )
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
            l.find(&stamp("24.1", &[], 1), H, &relu(), ElementKind::F32),
            Finding::Withheld(Withheld::Unmeasured)
        );
    }

    #[test]
    fn the_capture_time_is_not_part_of_what_was_measured() {
        let s = stamp("24.1", &["-denorm-mode-fp32", "preserve"], 1);
        let mut l = ExactnessLedger::new();
        l.push(rec(
            &s,
            ExactnessVerdict::Exact,
            H,
            Coverage::exhaustive(1 << 32),
        ));
        let later = stamp("24.1", &["-denorm-mode-fp32", "preserve"], 99);
        assert_eq!(l.find(&later, H, &relu(), ElementKind::F32), Finding::Exact);
    }
}
