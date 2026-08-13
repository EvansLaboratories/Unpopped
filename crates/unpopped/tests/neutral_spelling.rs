//! The neutral C-family speller vocabulary must not name a vendor's types.
//!
//! `cfamily` is documented as "deliberately backend-neutral" and is the module
//! the generator keeps when each emitter is carved into its own crate. Two of its
//! arms are not neutral: `scalar_ctype` spells `F16`/`Bf16` as NVIDIA's `__half`
//! / `__nv_bfloat16`, and the half load/store tail emits `__half2float`-class
//! CUDA intrinsics.
//!
//! # Why this needs a test rather than a comment
//!
//! The module header already warns about it. A comment cannot fail. The spellings
//! are *correct output for CUDA*, so nothing in the CUDA backend's goldens objects
//! to them, and no in-crate backend reaches them at all — `CpuC` declines f16/bf16
//! via `supports_dtype` and Slang never calls these. So the hazard is not that the
//! code is wrong today; it is that it is **wrong in a direction no existing test
//! can point at**. The first non-CUDA backend that supports f16 (Vulkane's
//! SPIR-V backend is the live case) calls a neutral-looking API and gets `__half`
//! spelled into its output.
//!
//! # The reachability claim, stated precisely
//!
//! The module header says these arms are "reachable ONLY from the CUDA backend".
//! That is true of the **plan-driven** paths and not of the API as a whole:
//!
//! - Plan-driven (`out_ctype_of`, `store_expr_of`, `param_ctype`) really is gated.
//!   `supports_dtype` rejects f16/bf16 for `plan.dtype`, and `plan.out_dtype_of(j)`
//!   cannot smuggle one in — `plan::assert_valid_out_dtype` admits only `U8`, `I32`
//!   and `I64` as hetero output dtypes, so a divergent output dtype is never a half.
//! - The **free functions are ungated public API**. `scalar_ctype`, `cast_scalar`,
//!   `promote_load_f32`, `demote_store_f32` and both `half_*_intrinsic` take a bare
//!   `ElementKind` with no plan and no backend in sight. Since `unpopped` publishes
//!   to crates.io, "reachable" means reachable by any third party who reads the
//!   module doc's word "neutral" and believes it.
//!
//! So the gap tests below pin a **known-wrong** spelling and are expected to fail
//! when the seam lands — that failure is the point, it forces the fix to announce
//! itself. `cast_scalar_over_neutral_dtypes_names_no_vendor` is the opposite: a
//! live guard that must stay green, and that catches a *new* vendor spelling
//! entering the neutral core.
//!
//! # Trap for whoever implements the seam: decline, do not fall through
//!
//! Found by running the seam as a mutation against these tests. The obvious first
//! move — make `scalar_ctype` and `half_load_intrinsic` return `None` for the
//! halves — does **not** make `cast_scalar` decline. It selects its branch on
//! `half_load_intrinsic(from).is_some()`, so a `None` drops it into the
//! arithmetic→arithmetic arm and it emits a plain C cast:
//!
//! ```text
//!   before:  cast_scalar(F16, F32, "v")  ==  "(float)__half2float(v)"
//!   after:   cast_scalar(F16, F32, "v")  ==  "(float)v"     // <-- silently wrong
//! ```
//!
//! `(float)v` on a `__half`-typed value is not a widening conversion; it is a
//! different, wrong program that still compiles. That trades a *visible* vendor
//! leak for a *silent* numerical bug — strictly worse than the state this test
//! was written to flag. The neutral core must therefore **refuse** an unspellable
//! dtype (a typed decline, or a panic naming the missing seam, matching the
//! `backend::default_seam` convention already used for `REDUCED`/`COORD`/
//! `SELECT`), never fall through to a default arm.

use unpopped::cfamily::{
    cast_scalar, demote_store_f32, half_load_intrinsic, half_store_intrinsic, promote_load_f32,
    scalar_ctype,
};
use unpopped_vocab::ElementKind;

/// How a dtype's scalar type name is spelled by the *neutral* module.
#[derive(Debug, PartialEq, Eq)]
enum Spelling {
    /// A type name in standard C — portable to every C-family target.
    PortableC,
    /// A vendor's type name. Must not come from a neutral module.
    Vendor,
    /// The neutral module declines to spell it (`scalar_ctype` returns `None`).
    Declined,
}

/// Every dtype, classified.
///
/// The `match` is **exhaustive on purpose**. `ElementKind` is deliberately not
/// `#[non_exhaustive]` (see its doc: a new dtype is meant to surface as a build
/// break at every match site), so adding one breaks this test and forces whoever
/// adds it to state whether its spelling is portable or a vendor's. A
/// hand-maintained list would silently miss it.
fn expected(dt: ElementKind) -> Spelling {
    use ElementKind::*;
    match dt {
        // `short` / `unsigned short` are exact, portable C spellings with no
        // vendor intrinsic and no packing — genuinely neutral, unlike the halves.
        F32 | F32Strict | F64 | I32 | I64 | I8 | U8 | U32 | U64 | I16 | U16 => Spelling::PortableC,

        // THE GAP. Correct for CUDA, wrong for a module that calls itself neutral.
        // When the spelling seam lands these become `Declined` and the backend
        // supplies the name.
        F16 | Bf16 => Spelling::Vendor,

        // Declined for three different reasons, worth keeping distinct:
        //   * `Fp8E4M3FNUZ`/`Fp8E5M2FNUZ` are RESERVED by KISS-Classify
        //     §6.1-0001 — recognized, distinguished from unknown, and never
        //     computed with at this schema version. Declining is REQUIRED here,
        //     not a gap.
        //   * `I4`/`U4`/`B1` are sub-byte packed; `Fp8E4M3FN`/`Fp8E5M2` need a
        //     software codec; `Complex64`/`Complex128` need a struct ABI. All
        //     are unimplemented rather than impossible.
        //   * `F8E8M0`/`F8E6M2` are the MX shared block SCALES — active §6.1
        //     dtypes at sk4, but 8-bit floats with no portable C type, so the
        //     neutral module declines them like the other FP8 rows.
        Bool | Fp8E4M3FN | Fp8E5M2 | Fp8E4M3FNUZ | Fp8E5M2FNUZ | F8E8M0 | F8E6M2 | I4 | U4 | B1
        | Complex64 | Complex128 => Spelling::Declined,
    }
}

/// Type names that are plain C and carry no vendor identity.
const PORTABLE_C_TYPES: &[&str] = &[
    "float",
    "double",
    "int",
    "long long",
    "signed char",
    "unsigned char",
    "unsigned int",
    // `U64` was held back here until the audit its old note demanded: the wrap
    // was two's-complement SIGNED and the comparison projected through f64,
    // which cannot represent every u64. Both are fixed — `is_unsigned_arith`
    // covers it and the tolerant comparator routes integers through `i128` — so
    // the spelling is claimed rather than declined. The note is deleted rather
    // than amended, because a hold-back whose reason has lapsed is exactly the
    // stale marker this suite keeps finding.
    "unsigned long long",
    "short",
    "unsigned short",
];

/// Substrings that identify a spelling as CUDA's.
const VENDOR_MARKERS: &[&str] = &[
    "__half",
    "__nv_",
    "__bfloat16",
    "__float2half",
    "__float2bfloat16",
    "__bfloat162float",
];

fn names_a_vendor(s: &str) -> bool {
    VENDOR_MARKERS.iter().any(|m| s.contains(m))
}

/// The dtypes a neutral C-family backend actually lowers today (CpuC's set:
/// everything with a portable ctype, minus `U32`, which is an index/address
/// dtype rather than a compute dtype).
const NEUTRAL_COMPUTE_DTYPES: &[ElementKind] = &[
    ElementKind::I16,
    ElementKind::U16,
    ElementKind::F32,
    ElementKind::F32Strict,
    ElementKind::F64,
    ElementKind::I32,
    ElementKind::I64,
    ElementKind::I8,
    ElementKind::U8,
];

/// Every dtype spells the way `expected` says, and the vendor set is *exactly*
/// the two halves.
///
/// Pinning the vendor set exactly is what makes this a tripwire in both
/// directions: it fails if the gap closes (the fix must acknowledge itself) and
/// it fails if the gap widens (a third vendor-spelled dtype appears).
#[test]
fn scalar_ctype_spells_portable_c_except_the_two_known_half_arms() {
    let all = [
        ElementKind::F32,
        ElementKind::F32Strict,
        ElementKind::F64,
        ElementKind::F16,
        ElementKind::Bf16,
        ElementKind::I32,
        ElementKind::I64,
        ElementKind::I8,
        ElementKind::I16,
        ElementKind::U8,
        ElementKind::U16,
        ElementKind::U32,
        ElementKind::U64,
        ElementKind::Fp8E4M3FNUZ,
        ElementKind::Fp8E5M2FNUZ,
        ElementKind::F8E8M0,
        ElementKind::F8E6M2,
        ElementKind::Bool,
        ElementKind::Fp8E4M3FN,
        ElementKind::Fp8E5M2,
        ElementKind::I4,
        ElementKind::U4,
        ElementKind::B1,
        ElementKind::Complex64,
        ElementKind::Complex128,
    ];

    let mut vendor_spelled = Vec::new();
    for dt in all {
        let got = scalar_ctype(dt);
        match expected(dt) {
            Spelling::PortableC => {
                let ct = got.unwrap_or_else(|| panic!("{dt:?} should have a portable ctype"));
                assert!(
                    PORTABLE_C_TYPES.contains(&ct),
                    "{dt:?} spells {ct:?}, which is not in the portable-C set — a neutral \
                     module must not invent a type name"
                );
                assert!(!names_a_vendor(ct), "{dt:?} spells the vendor name {ct:?}");
            }
            Spelling::Vendor => {
                let ct = got.unwrap_or_else(|| panic!("{dt:?} is expected to spell (wrongly)"));
                assert!(
                    names_a_vendor(ct),
                    "{dt:?} spells {ct:?}, which no longer names a vendor. If the spelling \
                     seam landed, move {dt:?} to `Spelling::Declined` and delete it from the \
                     expected-vendor set below — this test is the fix's acknowledgement."
                );
                vendor_spelled.push(dt);
            }
            Spelling::Declined => assert_eq!(
                got, None,
                "{dt:?} gained a spelling in the neutral module; classify it"
            ),
        }
    }

    assert_eq!(
        vendor_spelled,
        vec![ElementKind::F16, ElementKind::Bf16],
        "the vendor-spelled set must be exactly the two known half arms — it grew or shrank"
    );
}

/// The positive control: prove `names_a_vendor` can actually say no.
///
/// Without this, a detector that returned `true` for everything would satisfy
/// every vendor assertion above and the test would certify nothing.
#[test]
fn the_vendor_detector_discriminates() {
    for portable in PORTABLE_C_TYPES {
        assert!(
            !names_a_vendor(portable),
            "detector flagged the portable C type {portable:?} — it cannot tell \
             portable from vendor, so every assertion using it is vacuous"
        );
    }
    for marker in [
        "__half",
        "__nv_bfloat16",
        "__half2float",
        "__float2bfloat16",
    ] {
        assert!(
            names_a_vendor(marker),
            "detector missed the CUDA spelling {marker:?}"
        );
    }
}

/// The half intrinsics are CUDA's, pinned so the seam has to move them.
#[test]
fn the_half_intrinsics_are_the_known_cuda_gap() {
    assert_eq!(half_load_intrinsic(ElementKind::F16), Some("__half2float"));
    assert_eq!(
        half_load_intrinsic(ElementKind::Bf16),
        Some("__bfloat162float")
    );
    assert_eq!(half_store_intrinsic(ElementKind::F16), Some("__float2half"));
    assert_eq!(
        half_store_intrinsic(ElementKind::Bf16),
        Some("__float2bfloat16")
    );

    // The negative control for the intrinsic pair: a non-half dtype must have no
    // intrinsic at all, and must pass through promote/demote untouched. Without
    // this, a function returning `Some(...)` unconditionally would pass above.
    for dt in NEUTRAL_COMPUTE_DTYPES {
        assert_eq!(half_load_intrinsic(*dt), None, "{dt:?} is not a half");
        assert_eq!(half_store_intrinsic(*dt), None, "{dt:?} is not a half");
        assert_eq!(
            promote_load_f32(*dt, "x"),
            "x",
            "{dt:?} must widen to nothing"
        );
        assert_eq!(
            demote_store_f32(*dt, "x"),
            "x",
            "{dt:?} must narrow to nothing"
        );
    }
}

/// **The live guard.** Every cast between dtypes a neutral backend actually
/// lowers is free of vendor identity.
///
/// Unlike the gap pins above, this one must stay green forever. It is what
/// catches a *new* vendor spelling entering the neutral core — the failure mode
/// the module header warns about but nothing currently detects.
#[test]
fn cast_scalar_over_neutral_dtypes_names_no_vendor() {
    let mut checked = 0;
    for from in NEUTRAL_COMPUTE_DTYPES {
        for to in NEUTRAL_COMPUTE_DTYPES {
            let out = cast_scalar(*from, *to, "v");
            assert!(
                !names_a_vendor(&out),
                "cast_scalar({from:?} -> {to:?}) emitted {out:?}, which names a vendor \
                 from the neutral module"
            );
            assert!(
                out.contains('v'),
                "cast_scalar({from:?} -> {to:?}) dropped its operand: {out:?}"
            );
            checked += 1;
        }
    }
    assert_eq!(
        checked,
        NEUTRAL_COMPUTE_DTYPES.len() * NEUTRAL_COMPUTE_DTYPES.len(),
        "no pairs were checked — vacuous pass"
    );
}

/// And the counterpart that shows the guard above is not green by accident: the
/// same call with a half **does** produce vendor text today.
///
/// This is the whole bug in one assertion. `cast_scalar` is public, takes a bare
/// `ElementKind`, and has no backend or plan to gate it — so a third party gets
/// CUDA text out of a module documented as neutral.
#[test]
fn cast_scalar_leaks_vendor_text_for_halves_known_gap() {
    let out = cast_scalar(ElementKind::F16, ElementKind::F32, "v");
    assert!(
        names_a_vendor(&out),
        "cast_scalar(F16 -> F32) = {out:?} no longer names a vendor. If the spelling \
         seam landed, this gap test should be deleted and \
         `cast_scalar_over_neutral_dtypes_names_no_vendor` widened to cover the halves."
    );
    assert_eq!(out, "(float)__half2float(v)");
}
