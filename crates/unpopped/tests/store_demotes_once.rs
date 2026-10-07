//! **A uniform narrow-float store must encode exactly once — and this path has
//! no in-tree emitter that would show a double encode.**
//!
//! # The regression this pins
//!
//! `313a798` changed `store_expr_of`'s uniform branch from `return root` to
//! `return demote_store_f32(d, &root)`. Under the convention of the time, f16/bf16
//! body roots arrived ALREADY demoted (`__float2half(<f32>)`), so the change
//! shipped a double demote in `0.2.0`:
//!
//! ```text
//! 0.1.0   out[i] = __float2half(atan2f(__half2float(a), __half2float(b)));
//! 0.2.0   out[i] = __float2half(__float2half(atan2f(__half2float(a), ...)));
//! ```
//!
//! Numerically the identity for representable values, so only byte goldens saw
//! it, and ten of Baracuda's f16 ones did. The fix routed only the FP8 pair to the
//! store-site codec.
//!
//! # What changed in 0.15.0
//!
//! f16/bf16 now have FP8's shape: the body computes at f32 and the root is a plain
//! f32 expression, so the store-site codec is what encodes it, for all four narrow
//! floats alike. (Baracuda keeps the old CUDA convention in its own shadow of
//! `store_expr_of`, baracuda#154, so this moved none of its bytes.)
//!
//! The test counts the codec, so it fails in both directions: 2 is the shipped
//! double encode, and 0 is a float stored into an integer carrier by a plain C
//! conversion, which truncates silently (1.5f stored as 1).
//!
//! # Why this calls `store_expr_of` directly
//!
//! In-tree, `unpopped-cpu-c` is now a caller at every narrow float, but a double
//! encode is numerically the identity for representable values, so its end-to-end
//! tests cannot see one. Only counting the text can.

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

/// **Every narrow float encodes exactly once at the store.**
///
/// The body root is a plain f32 expression for FP8 and, since 0.15.0, for the
/// halves too, so the store-site codec is the one and only encode.
#[test]
fn a_narrow_float_store_encodes_exactly_once() {
    for (dt, codec) in [
        (ElementKind::Fp8E4M3FN, "unpopped_f8e4m3fn_store"),
        (ElementKind::Fp8E5M2, "unpopped_f8e5m2_store"),
        (ElementKind::F16, "unpopped_f16_store"),
        (ElementKind::Bf16, "unpopped_bf16_store"),
    ] {
        let out = store_for(dt, "fmaf(a, b, c)");
        let n = out.matches(codec).count();
        assert_eq!(
            n, 1,
            "{dt:?}: expected exactly one `{codec}` in the store, got {n}.
             2 = a double encode (the 313a798 class; numerically the identity, so              ONLY this count sees it).
             0 = the f32 root stored into the integer carrier by a plain C              conversion, which truncates silently.
             got: {out}"
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

/// Guards the assumption the other two tests rest on: that `ir` is reachable
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
