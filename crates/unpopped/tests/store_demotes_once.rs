//! **A uniform store must demote exactly once — and this path has no in-tree
//! emitter to catch it.**
//!
//! # The regression this pins
//!
//! `313a798` changed `store_expr_of`'s uniform branch from `return root` to
//! `return demote_store_f32(d, &root)`, justified as: *"`demote_store_f32` is the
//! identity for every dtype that needs no detour, so this stays byte-identical
//! for f32/f64/integer cells."*
//!
//! That sentence is **true as written and enumerates only the cases where the
//! change does nothing.** It never asks what happens to the four dtypes where
//! `demote_store_f32` is *not* the identity: the FP8 pair, which needed it, and
//! **f16/bf16, which did not** — those arrive already demoted, because the house
//! promote-demote convention lowers an f16 body root as `__float2half(<f32>)`.
//!
//! The result shipped in `0.2.0`:
//!
//! ```text
//! 0.1.0   out[i] = __float2half(atan2f(__half2float(a), __half2float(b)));
//! 0.2.0   out[i] = __float2half(__float2half(atan2f(__half2float(a), ...)));
//! ```
//!
//! Numerically the identity for representable values, so there is **no
//! result-level symptom** — only byte-identity goldens catch it, and ten of
//! Baracuda's f16 ones did.
//!
//! # Why nothing here caught it, which is the structural part
//!
//! `store_expr_of` lives in neutral core, and in-tree its only caller is
//! `unpopped-cpu-c` — **which declines f16/bf16** (no CPU half codec). So the
//! f16 store path is reachable only from `baracuda-cuda-emit`, outside this
//! repo. A neutral code path whose sole exerciser is out-of-tree gets no
//! coverage from any suite that runs here, and this is what that costs.
//!
//! So this test calls `store_expr_of` **directly** rather than through a
//! backend. It is the only way to reach the path from inside this crate.

use unpopped::cfamily::store_expr_of;
use unpopped::ir::{OpDef, input};
use unpopped::plan::KernelPlan;
use unpopped::{build_plan, ir};
use unpopped_vocab::{ArchSku, ElementKind, OpCategory, OperandDesc, structure_key};

fn plan_for(dt: ElementKind) -> (OpDef, unpopped_vocab::StructureKey) {
    let op = OpDef::elementwise("add", 2, &[dt], input(0) + input(1));
    let d = OperandDesc::new(1, &[8], &[1], dt, 256);
    let key = structure_key(OpCategory::BinaryElementwise, &[d, d, d], ArchSku::Sm89);
    (op, key)
}

fn store_for(dt: ElementKind, root: &str) -> String {
    let (op, key) = plan_for(dt);
    let plan: KernelPlan<'_> = build_plan(&op, &key);
    store_expr_of(&plan, 0, root.to_string())
}

/// **f16/bf16 arrive already demoted, so the store must not demote again.**
///
/// Fails in BOTH directions by counting: a doubled intrinsic is the shipped
/// regression, and zero intrinsics would mean the root's own demotion had been
/// dropped — the truncation bug the original change was written to prevent.
#[test]
fn a_half_store_demotes_exactly_once() {
    for (dt, intrinsic) in [
        (ElementKind::F16, "__float2half"),
        (ElementKind::Bf16, "__float2bfloat16"),
    ] {
        // The root as the body lowering actually produces it: already demoted.
        let root = format!("{intrinsic}(fmaf(a, b, c))");
        let out = store_for(dt, &root);
        let n = out.matches(intrinsic).count();
        assert_eq!(
            n, 1,
            "{dt:?}: expected exactly one `{intrinsic}` in the store, got {n}.\n\
             2 = the 313a798 double-demote that shipped in 0.2.0 (numerically the \
             identity, so ONLY a byte golden sees it).\n\
             0 = the root's own demotion was dropped, which is the silent \
             float->storage truncation the demote exists to prevent.\n\
             got: {out}"
        );
    }
}

/// The other half of the partition, and the reason the fix is a match rather
/// than a deletion: **FP8 genuinely needs the store-site codec**, because its
/// call is applied here rather than by the body lowering.
///
/// Without this, "fix the double-demote" reads as "drop the demote" and
/// reintroduces the truncation `313a798` was written to close.
#[test]
fn an_fp8_store_still_gets_its_codec() {
    for (dt, codec) in [
        (ElementKind::Fp8E4M3FN, "unpopped_f8e4m3fn_store"),
        (ElementKind::Fp8E5M2, "unpopped_f8e5m2_store"),
    ] {
        let out = store_for(dt, "fmaf(a, b, c)");
        assert!(
            out.contains(codec),
            "{dt:?}: the store must apply `{codec}` — its body root is a plain f32 \
             expression, so without the codec C truncates float->unsigned char \
             silently (1.5f stored as 1).\ngot: {out}"
        );
    }
}

/// Control: the dtypes that need no detour are still returned untouched, so the
/// fix cannot have widened into a dtype that was always fine.
#[test]
fn a_wide_or_integer_store_is_returned_unchanged() {
    for dt in [
        ElementKind::F32,
        ElementKind::F64,
        ElementKind::I32,
        ElementKind::U32,
    ] {
        let root = "fmaf(a, b, c)";
        assert_eq!(
            store_for(dt, root),
            root,
            "{dt:?}: a uniform store at a dtype needing no conversion must be the \
             root verbatim"
        );
    }
}

/// Guards the assumption the other three tests rest on: that `ir` is reachable
/// and the probe op really builds a uniform cell. Without it, a plan-shape change
/// could make every assertion above vacuous by routing to the hetero branch.
#[test]
fn the_probe_really_builds_a_uniform_cell() {
    let _ = ir::BinaryOp::Max; // module reachable
    let (op, key) = plan_for(ElementKind::F32);
    let plan = build_plan(&op, &key);
    assert_eq!(
        plan.out_dtype_of(0),
        plan.dtype,
        "the probe must be a UNIFORM cell (out dtype == compute dtype), or these \
         tests exercise the hetero branch and prove nothing about the uniform one"
    );
}
