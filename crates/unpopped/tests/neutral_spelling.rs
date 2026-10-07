//! The neutral C-family speller vocabulary must not name a vendor's types.
//!
//! `cfamily` is documented as "deliberately backend-neutral" and is the module
//! the generator keeps when each emitter is carved into its own crate. Until
//! 0.15.0 two of its arms were not neutral: `scalar_ctype` spelled `F16`/`Bf16`
//! as NVIDIA's `__half` / `__nv_bfloat16`, and the half load/store tail emitted
//! `__half2float`-class CUDA intrinsics.
//!
//! # How the seam closed
//!
//! The halves now have FP8's shape (`docs/deferred.md`, Section B): `scalar_ctype`
//! answers the STORAGE question only (`unsigned short`, the 16-bit carrier), and
//! the conversion is a software codec emitted into the kernel
//! (`cfamily::half_helpers`), reached through the same `narrow_load_fn` /
//! `narrow_store_fn` names FP8 uses. Nothing in the neutral module names a vendor.
//!
//! A backend with native halves (CUDA's `__half`) spells them itself: baracuda
//! keeps a closed local shadow of every `cfamily` function that used to reach the
//! half arms (baracuda#154), so this change moved none of its bytes.
//!
//! # Why this needs a test rather than a comment
//!
//! The old spellings were *correct output for CUDA*, so no CUDA golden objected
//! to them, and no in-tree backend reached them. The hazard was a neutral-looking
//! public API that put `__half` into a non-CUDA backend's output. The tests below
//! pin the closed state in both directions: no dtype spells a vendor name, and the
//! codec names the halves use are names the emitted helpers actually define.
//!
//! # The trap the old version of this file recorded, kept because it still holds
//!
//! Making `scalar_ctype` return `None` for the halves does **not** make
//! `cast_scalar` decline: it selects its branch on `narrow_load_fn(from).is_some()`,
//! so a `None` drops into the arithmetic arm and emits a plain C cast — a silent
//! numerical bug instead of a visible leak. That is why the seam closed by
//! *spelling* the halves (carrier plus codec) rather than by declining them, and
//! why `the_half_codec_is_emitted_not_named` checks that a half reaches its codec.

use unpopped::cfamily::{
    cast_scalar, complex_helpers, demote_store_f32, half_helpers, narrow_load_fn,
    narrow_store_fn, promote_load_f32, scalar_ctype,
};
use unpopped_vocab::ElementKind;

/// How a dtype's scalar type name is spelled by the *neutral* module.
#[derive(Debug, PartialEq, Eq)]
enum Spelling {
    /// A type name in standard C — portable to every C-family target.
    PortableC,
    /// A name this crate DEFINES and emits into the same translation unit.
    ///
    /// Neutral, but for a different reason than [`Spelling::PortableC`]: not
    /// because every C compiler already knows the name, but because the kernel
    /// carries its own definition and so depends on no header at all. The test
    /// for this tier is therefore stronger — the emitted helper text must
    /// actually contain the typedef, or the "neutral" spelling is simply an
    /// undefined type that fails to compile.
    SelfDefined,
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
        // vendor intrinsic and no packing.
        F32 | F32Strict | F64 | I32 | I64 | I8 | U8 | U32 | U64 | I16 | U16 => Spelling::PortableC,

        // `f8e4m3fn`/`f8e5m2` spell `unsigned char` — the STORAGE type — and their
        // conversions are software helpers emitted into the kernel
        // (`cfamily::fp8_helpers`), not vendor intrinsics.
        Fp8E4M3FN | Fp8E5M2 => Spelling::PortableC,

        // The same shape, one width up: `f16`/`bf16` spell `unsigned short`, and
        // their codec is `cfamily::half_helpers`. Until 0.15.0 these were the two
        // vendor-spelled arms (`__half` / `__nv_bfloat16`).
        F16 | Bf16 => Spelling::PortableC,

        // `bool` spells `unsigned char` — its storage width equals `u8`'s (§6.1).
        // The normalization that makes it a different DTYPE lives in the logical
        // spellers, not in the type name.
        Bool => Spelling::PortableC,

        // Sub-byte dtypes spell their CONTAINER (`unsigned char`); the packing is
        // emitted helpers (`cfamily::sub_byte_helpers`), portable and vendor-free.
        I4 | U4 | B1 => Spelling::PortableC,

        // Complex spells a STRUCT this crate defines and emits. C99 `_Complex`
        // is not portable — MSVC does not implement it (`error C2440`) — so the
        // struct is the neutral answer rather than a fallback.
        Complex64 | Complex128 => Spelling::SelfDefined,

        // Declined for two different reasons, worth keeping distinct:
        //   * `Fp8E4M3FNUZ`/`Fp8E5M2FNUZ` are RESERVED by KISS-Classify
        //     §6.1-0001 — recognized, distinguished from unknown, and never
        //     computed with at this schema version. Declining is REQUIRED here.
        //   * `F8E8M0`/`F8E6M2` are the MX shared block SCALES — active §6.1
        //     dtypes at sk4, but never an element compute dtype.
        Fp8E4M3FNUZ | Fp8E5M2FNUZ | F8E8M0 | F8E6M2 => Spelling::Declined,
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

/// The halves, which reach the neutral module's own emitted codec.
const HALVES: &[(ElementKind, &str, &str)] = &[
    (ElementKind::F16, "unpopped_f16_load", "unpopped_f16_store"),
    (ElementKind::Bf16, "unpopped_bf16_load", "unpopped_bf16_store"),
];

/// Dtypes a neutral C-family backend computes on with no codec at all.
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

/// Every dtype spells the way `expected` says, and no dtype names a vendor.
#[test]
fn scalar_ctype_spells_no_vendor_type_for_any_dtype() {
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

    let mut self_defined = Vec::new();
    for dt in all {
        let got = scalar_ctype(dt);
        if let Some(ct) = got {
            assert!(!names_a_vendor(ct), "{dt:?} spells the vendor name {ct:?}");
        }
        match expected(dt) {
            Spelling::PortableC => {
                let ct = got.unwrap_or_else(|| panic!("{dt:?} should have a portable ctype"));
                assert!(
                    PORTABLE_C_TYPES.contains(&ct),
                    "{dt:?} spells {ct:?}, which is not in the portable-C set — a neutral \
                     module must not invent a type name"
                );
            }
            Spelling::SelfDefined => {
                let ct = got.unwrap_or_else(|| panic!("{dt:?} should have a ctype"));
                assert!(
                    !PORTABLE_C_TYPES.contains(&ct),
                    "{dt:?} spells the builtin {ct:?} — classify it PortableC, not SelfDefined"
                );
                // The load-bearing half: the kernel must DEFINE what it names.
                let helpers = complex_helpers(dt)
                    .unwrap_or_else(|| panic!("{dt:?} names {ct:?} but emits no definition"));
                assert!(
                    helpers.contains(&format!("}} {ct};")),
                    "{dt:?} names {ct:?}, but the emitted helpers do not typedef it:
{helpers}"
                );
                self_defined.push(ct);
            }
            Spelling::Declined => assert_eq!(
                got, None,
                "{dt:?} gained a spelling in the neutral module; classify it"
            ),
        }
    }

    // A self-defined name is the weakest neutrality claim of the three, so the
    // set that gets to make it does not grow without someone saying so.
    assert_eq!(
        self_defined,
        vec!["unpopped_c64", "unpopped_c128"],
        "the self-defined set must be exactly the two complex types"
    );
}

/// The positive control: prove `names_a_vendor` can actually say no, and yes.
///
/// Without this, a detector that returned `false` for everything would satisfy
/// every "names no vendor" assertion in this file and certify nothing.
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

/// The halves load and store through a codec the kernel DEFINES, by the same
/// `narrow_*_fn` names FP8 uses, and the codec names no vendor.
///
/// "Defines" is the load-bearing word, as for complex: a codec name the helpers
/// don't define is an undeclared function, which C99 rejects and C89 silently
/// treats as returning `int`.
#[test]
fn the_half_codec_is_emitted_not_named() {
    for &(dt, load, store) in HALVES {
        assert_eq!(narrow_load_fn(dt), Some(load), "{dt:?} load");
        assert_eq!(narrow_store_fn(dt), Some(store), "{dt:?} store");
        assert_eq!(promote_load_f32(dt, "x"), format!("{load}(x)"));
        assert_eq!(demote_store_f32(dt, "x"), format!("{store}(x)"));

        let helpers = half_helpers(dt).unwrap_or_else(|| panic!("{dt:?} emits no codec"));
        assert!(!names_a_vendor(helpers), "{dt:?} codec names a vendor:\n{helpers}");
        for name in [load, store] {
            assert!(
                helpers.contains(&format!(" {name}(")),
                "{dt:?}: the emitted helpers do not define `{name}`:\n{helpers}"
            );
        }
    }

    // The negative control: a dtype with no codec gets no helpers and passes
    // through promote/demote untouched. Without this, functions returning
    // `Some(..)` unconditionally would pass above.
    for dt in NEUTRAL_COMPUTE_DTYPES {
        assert_eq!(half_helpers(*dt), None, "{dt:?} is not a half");
        assert_eq!(narrow_load_fn(*dt), None, "{dt:?} needs no load codec");
        assert_eq!(promote_load_f32(*dt, "x"), "x", "{dt:?} must widen to nothing");
        assert_eq!(demote_store_f32(*dt, "x"), "x", "{dt:?} must narrow to nothing");
    }
}

/// **The live guard.** Every cast between dtypes a neutral backend lowers,
/// halves included, is free of vendor identity, and a cast from or to a half
/// goes through that half's codec rather than a plain C cast.
#[test]
fn cast_scalar_names_no_vendor_and_routes_halves_through_their_codec() {
    let mut set: Vec<ElementKind> = NEUTRAL_COMPUTE_DTYPES.to_vec();
    set.extend(HALVES.iter().map(|h| h.0));
    let codec = |dt: ElementKind| HALVES.iter().find(|h| h.0 == dt);

    let mut checked = 0;
    for &from in &set {
        for &to in &set {
            let out = cast_scalar(from, to, "v");
            assert!(
                !names_a_vendor(&out),
                "cast_scalar({from:?} -> {to:?}) emitted {out:?}, which names a vendor \
                 from the neutral module"
            );
            assert!(out.contains('v'), "cast_scalar({from:?} -> {to:?}) dropped its operand");
            if from == to {
                // A same-dtype cast moves the carrier unchanged, which is exact.
                assert_eq!(out, "v", "cast_scalar({from:?} -> {to:?}) must be the identity");
                checked += 1;
                continue;
            }
            if let Some(&(_, load, _)) = codec(from) {
                assert!(
                    out.contains(load),
                    "cast_scalar({from:?} -> {to:?}) = {out:?} does not decode the half — \
                     a plain C cast of a 16-bit carrier is the silent bug in this file's header"
                );
            }
            if let Some(&(_, _, store)) = codec(to) {
                assert!(
                    out.contains(store),
                    "cast_scalar({from:?} -> {to:?}) = {out:?} does not encode to the half"
                );
            }
            checked += 1;
        }
    }
    assert_eq!(checked, set.len() * set.len(), "vacuous pass");
}
