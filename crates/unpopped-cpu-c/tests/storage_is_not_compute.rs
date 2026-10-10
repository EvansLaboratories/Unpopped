//! **A storage spelling must never imply compute support.**
//!
//! # Where this came from
//!
//! Vulkane, who owns the `vulkan:` capability vocabulary, flagged the trap in
//! their namespace: `st16` is `storageBuffer16BitAccess` — a **storage**
//! capability — and a device may accept 16-bit data in a buffer while performing
//! the arithmetic in `f32`. Reading it as permission to emit 16-bit integer math
//! is a silently wrong lowering on *conformant* hardware.
//!
//! Baracuda then gave the CUDA mirror: fp8 lives in memory on any arch but
//! computes only on Ada+; bf16 loads broadly while tensor-core math is
//! arch-gated. Two namespaces, same shape — so it is a property of dtypes, not
//! of one vendor's spelling.
//!
//! # And it was live here, as a reasoning shape
//!
//! `CpuC::supports_dtype` used to be
//! `!matches!(F16 | Bf16) && scalar_ctype(dtype).is_some()` — "has a C type
//! spelling, minus two". `scalar_ctype` returns the **carrier**: `unsigned char`
//! for FP8, bool and the sub-byte dtypes, `__half` for `f16`. So a compute
//! question was being answered with a storage fact.
//!
//! It gave correct answers, for a reason the predicate never stated — CpuC also
//! emits software codecs, so carrier-plus-codec really is compute support here.
//! The `F16`/`Bf16` exclusion was the patch over the gap, which is evidence the
//! inference had already broken once and been special-cased rather than fixed.
//!
//! The live hazard was that it was a **denylist over a growing set**: give any
//! new dtype a `scalar_ctype` spelling and CpuC would silently claim to compute
//! it, with nobody having written the codec.
//!
//! # What these tests DO and DO NOT prove — measured, not assumed
//!
//! Seeded the obvious mutation — revert `supports_dtype` to the old
//! `!matches!(F16 | Bf16) && scalar_ctype(dtype).is_some()` — and **every test
//! in this file still passed.** That is not a gap to paper over; it is the
//! honest shape of the change:
//!
//! **The two predicates are extensionally equal today.** They agree on all 24
//! dtypes. The old one gave right answers for a reason it never stated; the new
//! one gives the same answers for a reason it does state. No runtime test can
//! separate them, and one that claimed to would be lying.
//!
//! **What the change actually buys is a COMPILE-TIME property.** The allowlist
//! is an exhaustive `match` over `ElementKind`, which is deliberately not
//! `#[non_exhaustive]` — so adding a dtype is a build error *here* and someone
//! must say whether CpuC computes it. Under the old predicate a new dtype with a
//! carrier was silently admitted. That guarantee is the compiler's, not this
//! file's, and it cannot be tested from inside the same crate.
//!
//! **So what these tests are for** is the other half: pinning the admission set
//! so an accidental *widening* is loud — someone adding a dtype to the allowlist
//! without writing its codec — and recording the distinction where the next
//! reader will meet it.

use unpopped::backend::Backend;
use unpopped::cfamily::scalar_ctype;
use unpopped_cpu_c::CpuC;
use unpopped_vocab::{ArchSku, ElementKind, TargetId};

fn target() -> TargetId {
    ArchSku::Sm89.into()
}

/// Carrier plus an emitted codec is compute support: the dtypes CpuC computes
/// on although C has no native type for them.
///
/// The halves joined in 0.15.0. Until then they were this file's witness that a
/// storage spelling does not imply compute support: they had a carrier
/// (`__half`, `__nv_bfloat16`) and no CpuC codec. Now they spell `unsigned short`
/// and carry `cfamily::half_helpers`, so they are admitted for the same reason
/// FP8 is.
#[test]
fn a_carrier_plus_an_emitted_codec_is_compute_support() {
    for (dt, carrier) in [
        (ElementKind::Fp8E4M3FN, "unsigned char"),
        (ElementKind::Fp8E5M2, "unsigned char"),
        (ElementKind::I4, "unsigned char"),
        (ElementKind::U4, "unsigned char"),
        (ElementKind::B1, "unsigned char"),
        (ElementKind::F16, "unsigned short"),
        (ElementKind::Bf16, "unsigned short"),
    ] {
        assert_eq!(scalar_ctype(dt), Some(carrier), "{dt:?}'s carrier");
        assert!(
            CpuC.supports_dtype(dt, target()),
            "{dt:?} has a carrier AND an emitted codec, so CpuC computes it"
        );
    }
}

/// The MX shared block scales are declined **by design**, not for want of a
/// codec — they are sibling operands (§6.1-0013), never an element compute
/// dtype. Pinned separately from the halves so a future reader does not
/// "helpfully" add a codec for them.
#[test]
fn the_mx_scales_are_declined_by_design_not_for_want_of_a_codec() {
    for dt in [ElementKind::F8E8M0, ElementKind::F8E6M2] {
        assert!(!CpuC.supports_dtype(dt, target()));
    }
    // The reserved fnuz pair must never lower at this schema version either.
    for dt in [ElementKind::Fp8E4M3FNUZ, ElementKind::Fp8E5M2FNUZ] {
        assert!(!CpuC.supports_dtype(dt, target()));
    }
}

/// **The admission set is pinned exactly**, so a widening is loud.
///
/// Since 0.15.0 the carrier set and the compute set are EQUAL: the halves were
/// the only dtypes with a carrier and no CpuC codec, and they now have one. That
/// does not make `supports_dtype` a storage check. It is still an exhaustive
/// allowlist, and that compile-time property is what keeps a new carrier from
/// being admitted silently (see the module header: no runtime test can separate
/// the two predicates). What this test adds is the set itself, written out, so a
/// dtype added to the allowlist without its codec, or one dropped, fails here.
#[test]
fn the_admission_set_is_pinned_so_a_widening_is_loud() {
    let has_carrier: Vec<ElementKind> = ElementKind::ALL
        .iter()
        .copied()
        .filter(|&d| scalar_ctype(d).is_some())
        .collect();
    let computes: Vec<ElementKind> = ElementKind::ALL
        .iter()
        .copied()
        .filter(|&d| CpuC.supports_dtype(d, target()))
        .collect();

    let mut want = vec![
        ElementKind::F32,
        ElementKind::F32Strict,
        ElementKind::F64,
        ElementKind::F16,
        ElementKind::Bf16,
        ElementKind::I8,
        ElementKind::U8,
        ElementKind::I16,
        ElementKind::U16,
        ElementKind::I32,
        ElementKind::U32,
        ElementKind::I64,
        ElementKind::U64,
        ElementKind::Bool,
        ElementKind::Fp8E4M3FN,
        ElementKind::Fp8E5M2,
        ElementKind::I4,
        ElementKind::U4,
        ElementKind::B1,
        ElementKind::Complex64,
        ElementKind::Complex128,
    ];
    let mut got = computes.clone();
    want.sort_by_key(|d| format!("{d:?}"));
    got.sort_by_key(|d| format!("{d:?}"));
    assert_eq!(got, want, "CpuC's admission set changed");

    let carrier_only: Vec<ElementKind> = has_carrier
        .iter()
        .copied()
        .filter(|d| !computes.contains(d))
        .collect();
    assert!(
        carrier_only.is_empty(),
        "a dtype has a carrier and no CpuC compute: {carrier_only:?}. Either write its          codec and admit it, or record here why it is storage-only"
    );
}
