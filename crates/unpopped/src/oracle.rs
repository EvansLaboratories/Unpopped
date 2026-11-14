// SPDX-License-Identifier: MIT OR Apache-2.0
//! # CPU ORACLE (plan-interpreter) — v1
//!
//! An **independent** CPU interpreter that computes what a generated CUDA kernel
//! computes, from the SAME IR ([`crate::plan::KernelPlan`] + [`OperandDesc`]) but
//! via a **separate code path**. It is therefore a differential test of the
//! emission (`crate::cuda`), a universal numeric reference, and a GPU-free CI base.
//!
//! ## Independence (the review discipline)
//!
//! This module shares **zero** lowering code with the emitter. It does NOT call
//! `lower_expr` / `lower_dag` / the `Lowering` struct / `unary_f32` / `binary_f32`
//! / `binary_int` / `cuda_select` / `const_lit` / `offset_expr` /
//! `gathered_offset_expr` / any `emit_*`. Every `ScalarExpr` evaluation, every
//! scalar op, all index / fold / sort math, and the half codec are re-implemented
//! here from each op's DEFINITION (not its CUDA spelling), so a shared spelling
//! bug cannot hide in both the emitter and the oracle.
//!
//! It MAY reuse the pieces UPSTREAM of both lowerings: the IR types
//! ([`crate::ir`]), [`crate::plan::build_plan`]/[`KernelPlan`], and the operand
//! role classifier [`crate::plan::rr_role`] (a broadcast-mask predicate, not a
//! spelling).
//!
//! **Honest scope:** the oracle catches EMISSION bugs, not IR-construction bugs
//! (it reads the same `build_plan` output the emitter does). It complements — does
//! not replace — the hand/bespoke oracles and the `build_plan`-direct gate tests.
//!
//! ## Which of this module's rules a BACKEND must reproduce
//!
//! Much of what follows is phrased as "the emitter's inner loop order",
//! "mirroring the emitter", "the emitter tests `((float)(c)) != 0.0f`". That was
//! sufficient while CUDA was the only backend — "what it does" and "what is
//! required" were the same sentence. They are not any more, and the two kinds of
//! statement look identical in this source.
//!
//! `docs/conformance.md` separates them: which rules are NORMATIVE (a backend
//! that differs produces output the oracle rejects), which are INCIDENTAL
//! (mirrored from CUDA, absorbed by the tolerant comparator, free to differ), and
//! which are OPEN. Two worth knowing before writing a backend:
//!
//! - **Discontinuities decide in the COMPUTE dtype**, never in wider precision.
//!   A tolerance cannot absorb a discontinuity — deciding `x > 0` in f64 where
//!   the kernel decides in f32 selects the other branch, not a nearby value.
//! - **`Max`/`Min` propagate NaN**, which is *not* C `fmax` / IEEE `maxNum`
//!   semantics and *not* GLSL `max`. Every C-family target offers a convenient
//!   built-in with the opposite behavior, and the difference is invisible on all
//!   NaN-free input.
//!
//! ## Precision posture
//!
//! Correctness/precision over speed: arithmetic accumulates in `f64` (never the
//! emitter's "double-then-round-once f32" convention — that would couple the
//! oracle to the speller). Integer arithmetic/bitwise is exact-wrapping in the
//! two's-complement width the emitter uses. The [`compare`] helper encodes the
//! bit-exact-vs-tolerance dichotomy.
//!
//! ## v1 scope
//!
//! [`Access::Elementwise`], [`Access::Reduction`], [`Access::RowReduce`],
//! [`Access::Scan`], [`Access::Window`], [`Access::Im2Col`] + the full
//! `ScalarExpr` vocabulary + layout math (contiguous / strided / broadcast /
//! flipped / permuted / base-offset) + multi-output / hetero.
//!
//! ### Not covered
//!
//! Stated as scope rather than as a TODO list, because this list had gone stale
//! in the direction that matters: it claimed `Access::Contraction` was deferred
//! long after `eval_contraction` shipped and `evaluate` began dispatching it.
//! A reader trusting it would conclude matmul results were unvalidated and skip
//! writing the differential test that already works. An over-cautious scope note
//! is not harmless — it suppresses testing.
//!
//! - `Access::RowSort` — the NaN-greatest `key_lt`, stable index ties, and the
//!   TopK comparator. Genuinely not implemented.
//! - gather / scatter — `ReadIndex`/`WriteIndex` OOB policies plus FP-`atomicAdd`
//!   nondeterminism. Note this one is not merely unimplemented: under a
//!   nondeterministic accumulation order only an **order-independent invariant**
//!   is checkable at all, so it needs a different notion of "correct" rather
//!   than more of this module.
//!
//! **`Access::Contraction` IS covered** (`eval_contraction`, dispatched by
//! [`evaluate`]) — rank-2 and batched rank-3, transposed / permuted / broadcast
//! operands, the `Reduced(0)` epilogue and the fused per-column bias.
//!
//! **Device-side differential fuzzing is not deferred, it is elsewhere**: the
//! cross-backend IR fuzzer moved to `baracuda-cuda-emit`'s integration tests
//! during the carve, because it drives a device backend. It is not something
//! this crate can host.
//!
//! ### Transcendental accuracy
//!
//! `Erf`/`Erfc`/`Gelu` use libm (~1 ULP f64) — a pure-Rust implementation
//! INDEPENDENT of the emitter's device `erf`/`erff`, so the differential check
//! stays honest while being a genuinely tighter reference than the kernel it
//! validates. `Lgamma` uses an in-house Lanczos series (~1e-13). `exp`/`ln`/
//! `sqrt`/`tanh`/… are accurate std `f64`. None is used in a bit-exact path.
//!
//! ### Runtime base offsets
//!
//! A [`crate::ir::BaseOffset::Runtime`] value is threaded through
//! [`TypedBuffer::base_offset`] (an element offset applied to the operand's base
//! BEFORE all per-element address math), since the plan carries only the PRESENCE
//! mask, not the runtime value.

mod accum_policy;
mod kernel_async;
mod kernel_call;
mod kernel_collective;
pub mod kernel_eval;
mod kernel_mma;
pub use kernel_call::run_module;
pub mod kernel_launch;
mod kernel_mem;
mod kernel_resume;
mod kernel_sync;

mod buffers;
mod elem;
mod expr_eval;
mod fidelity;
mod float_codec;
mod plan_contract;
mod plan_eval;
mod plan_window;
mod scalar_ops;
mod value;

use crate::ir::{Access, BinaryOp, ReduceOp, ScalarExpr, SortOrder, SortOut, UnaryOp, View};
use crate::plan::{KernelPlan, RrRole, Schedule, rr_role};
use unpopped_vocab::{ElementKind, OperandDesc};

pub use self::buffers::TypedBuffer;
pub(crate) use self::elem::elem_bits;
pub use self::fidelity::Fidelity;
pub use self::fidelity::compare;
pub use self::fidelity::compare_for_plan;
pub use self::fidelity::compare_for_plan_with_exactness;
pub use self::fidelity::compare_for_plan_within;
pub use self::fidelity::required_fidelity;
pub use self::fidelity::required_fidelity_with_exactness;
pub use self::plan_eval::evaluate;

#[cfg(test)]
use self::buffers::*;
use self::elem::*;
use self::expr_eval::*;
#[cfg(test)]
use self::fidelity::*;
use self::float_codec::*;
use self::plan_contract::*;
use self::plan_eval::*;
use self::plan_window::*;
use self::scalar_ops::*;
use self::value::*;

// ===========================================================================
// §6 Bootstrap-trust tests (GPU-free): the oracle must prove ITSELF first.
// ===========================================================================
#[cfg(test)]
mod complex_arithmetic_tests {
    use super::*;
    use crate::ir::ScalarExpr;

    /// Evaluate `expr` at a complex dtype with two complex inputs.
    ///
    /// Goes through the real `eval` path rather than calling the arithmetic
    /// closures directly, so the routing (`is_complex` before the int/float
    /// split) is exercised too.
    fn eval_complex(expr: &ScalarExpr, a: (f64, f64), b: (f64, f64)) -> (f64, f64) {
        let leaf = |i: u8| {
            let (re, im) = if i == 0 { a } else { b };
            Val::Complex(re, im)
        };
        let never = |_: u8| -> Val { panic!("not used") };
        let ev = Eval {
            dtype: ElementKind::Complex128,
            params: &[],
            leaf: &leaf,
            reduced: &never,
            coord: &never,
        };
        eval(expr, &ev).complex()
    }

    /// **`Mul` is not component-wise, and this is the only test that can see it.**
    ///
    /// Complex multiplication is `(ac − bd, ad + bc)`. Applying the real rule
    /// component-wise gives `(ac, bd)` — which agrees with the correct answer
    /// whenever either operand is purely real or purely imaginary, so a
    /// carelessly-chosen input pair cannot tell them apart. `(1+2i)(3+4i)` is
    /// `-5 + 10i` correctly and `3 + 8i` component-wise: different in both
    /// components, so neither half can be right by accident.
    ///
    /// # Why this is a unit test
    ///
    /// Complex is not plan-admissible yet, so complex arithmetic is unreachable
    /// through `build_plan`/`evaluate` — the integration tests in
    /// `tests/complex_domain.rs` exercise buffers, sizes and comparison, and
    /// **cannot reach the arithmetic at all**. Seeding the component-wise rule
    /// left every one of them green. Tested here because here is where it is
    /// reachable; when the plan gate admits complex, this belongs end-to-end too.
    #[test]
    fn complex_mul_mixes_components() {
        let got = eval_complex(
            &ScalarExpr::Mul(
                Box::new(ScalarExpr::Input(0)),
                Box::new(ScalarExpr::Input(1)),
            ),
            (1.0, 2.0),
            (3.0, 4.0),
        );
        assert_eq!(got, (-5.0, 10.0), "(1+2i)(3+4i) = -5+10i");
        assert_ne!(
            got,
            (3.0, 8.0),
            "component-wise multiplication is the classic wrong rule"
        );

        // i * i = -1: the identity that defines the domain.
        let i_squared = eval_complex(
            &ScalarExpr::Mul(
                Box::new(ScalarExpr::Input(0)),
                Box::new(ScalarExpr::Input(1)),
            ),
            (0.0, 1.0),
            (0.0, 1.0),
        );
        assert_eq!(i_squared, (-1.0, 0.0), "i^2 must be -1");
    }

    /// `Add`/`Sub` ARE component-wise — the contrast that makes `Mul` a rule
    /// rather than an accident.
    #[test]
    fn complex_add_and_sub_are_component_wise() {
        assert_eq!(
            eval_complex(
                &ScalarExpr::Add(
                    Box::new(ScalarExpr::Input(0)),
                    Box::new(ScalarExpr::Input(1)),
                ),
                (1.0, 2.0),
                (3.0, 4.0)
            ),
            (4.0, 6.0)
        );
        assert_eq!(
            eval_complex(
                &ScalarExpr::Sub(
                    Box::new(ScalarExpr::Input(0)),
                    Box::new(ScalarExpr::Input(1)),
                ),
                (1.0, 2.0),
                (3.0, 4.0)
            ),
            (-2.0, -2.0)
        );
    }

    /// A real operand embeds as `(x, 0)`, so a real literal is usable in a
    /// complex expression — and `Val::f64` refuses the reverse rather than
    /// silently dropping the imaginary half.
    #[test]
    fn reals_widen_into_the_complex_domain_but_not_back() {
        let leaf = |_: u8| Val::Float(3.0);
        let never = |_: u8| -> Val { panic!("not used") };
        let ev = Eval {
            dtype: ElementKind::Complex128,
            params: &[],
            leaf: &leaf,
            reduced: &never,
            coord: &never,
        };
        assert_eq!(eval(&ScalarExpr::Input(0), &ev).complex(), (3.0, 0.0));

        let r = std::panic::catch_unwind(|| Val::Complex(1.0, 2.0).f64());
        assert!(
            r.is_err(),
            "projecting a complex into the float domain must be fatal, not the              real part — a vanished imaginary half is a wrong answer that looks right"
        );
    }
}

#[cfg(test)]
mod tests {

    /// `Extract(BitcastPacked(x, elem, n), i)` is lane `i` of `x`'s bits, lane 0 least
    /// significant, and `Bitcast(BitcastPacked(..))` reassembles the same bits. The plan gate
    /// declines packed bodies, so the public API cannot reach this arm; evaluate directly.
    #[test]
    fn bitcast_packed_slices_lane_zero_from_the_low_bits_and_reassembles() {
        let leaf = |_: u8| Val::Float(0.0);
        let none = |_: u8| -> Val { unreachable!() };
        let ev = Eval {
            dtype: ElementKind::F32,
            params: &[],
            leaf: &leaf,
            reduced: &none,
            coord: &none,
        };
        let word = ScalarExpr::ConstInt(0x0403_0201, ElementKind::U32);
        let packed = ScalarExpr::BitcastPacked(Box::new(word.clone()), ElementKind::I8, 4);
        for (i, want) in [(0u8, 1u128), (1, 2), (2, 3), (3, 4)] {
            let lane = ScalarExpr::Extract(Box::new(packed.clone()), i);
            match eval(&lane, &ev) {
                Val::Raw(b, ElementKind::I8) => assert_eq!(b, want, "lane {i}"),
                _ => panic!("lane {i}: not a raw I8"),
            }
        }
        let back = ScalarExpr::Bitcast(Box::new(packed), ElementKind::U32);
        match eval(&back, &ev) {
            Val::Raw(b, ElementKind::U32) => assert_eq!(b, 0x0403_0201),
            _ => panic!("not a raw U32"),
        }
    }

    /// A packed `Select` mask picks per lane: lane `i` of the result is the first arm's lane `i`
    /// when the condition's lane `i` is non-zero, else the second's. Integer and float mask
    /// lanes are both tested `!= 0` (a float `-0.0` is false, a NaN true), exactly as a scalar
    /// `Select` condition. Evaluated directly: the plan gate declines packed bodies.
    #[test]
    fn a_packed_select_mask_chooses_each_lane_independently() {
        let leaf = |_: u8| Val::Float(0.0);
        let none = |_: u8| -> Val { unreachable!() };
        let ev = Eval {
            dtype: ElementKind::F32,
            params: &[],
            leaf: &leaf,
            reduced: &none,
            coord: &none,
        };
        let c = |v: f64| ScalarExpr::Const(v);
        let pack = |xs: Vec<ScalarExpr>| ScalarExpr::Pack(xs);
        let a = pack(vec![c(10.0), c(20.0), c(30.0), c(40.0)]);
        let b = pack(vec![c(-1.0), c(-2.0), c(-3.0), c(-4.0)]);
        let ints = pack(
            [0i64, 5, 0, -7]
                .map(|v| ScalarExpr::ConstInt(v, ElementKind::I32))
                .to_vec(),
        );
        let floats = pack(vec![c(0.0), c(-0.0), c(f64::NAN), c(0.5)]);
        let lane = |mask: &ScalarExpr, i: u8| {
            let sel = ScalarExpr::Select(
                Box::new(mask.clone()),
                Box::new(a.clone()),
                Box::new(b.clone()),
            );
            eval(&ScalarExpr::Extract(Box::new(sel), i), &ev).f64()
        };
        let got: Vec<f64> = (0..4).map(|i| lane(&ints, i)).collect();
        assert_eq!(got, [-1.0, 20.0, -3.0, 40.0], "integer mask lanes");
        let got: Vec<f64> = (0..4).map(|i| lane(&floats, i)).collect();
        assert_eq!(
            got,
            [-1.0, -2.0, 30.0, 40.0],
            "float mask: -0.0 false, NaN true"
        );
    }

    /// The shared sign/magnitude masks, checked against an **independently
    /// authored** width table rather than against themselves.
    ///
    /// # Why this exists
    ///
    /// `unpopped-cpu-c`'s emitter and this oracle both delegate to
    /// `ir::narrow_sign_masks`. That deduplication is right — one normative fact,
    /// one definition — but it means the emitter-vs-oracle differential
    /// (`a_sign_edit_agrees_with_the_oracle_on_every_fp8_byte`) compares the
    /// constant against itself. **Measured 2026-09-06: mutating the fp8 pair to
    /// `(0x40, 0xBF)` leaves that differential GREEN.**
    ///
    /// The constant is in fact covered — by literal expected C text in
    /// `narrow_float_moves_do_not_round.rs`. ⚠️ **But that coverage is a string
    /// match in a downstream crate's test file, and nothing said it was
    /// load-bearing.** Deleting it as redundant would silently unguard the mask.
    ///
    /// This derives the same fact from `elem_bits`, which is authored separately
    /// and maintained for a different reason, so the two disagree if either
    /// moves. **KISS-CONSUME §8.2-0001: the question is provenance, not
    /// equality.**
    #[test]
    fn the_shared_sign_masks_agree_with_an_independent_width_derivation() {
        for dt in [
            ElementKind::Fp8E4M3FN,
            ElementKind::Fp8E5M2,
            ElementKind::F16,
            ElementKind::Bf16,
        ] {
            let (sign, mag) = crate::ir::narrow_sign_masks(dt)
                .unwrap_or_else(|| panic!("{dt:?} is a narrow float and must have masks"));

            // Independent derivation: an IEEE-shaped float's sign is the top bit
            // of its storage width, and its magnitude mask is everything else.
            let bits = elem_bits(dt);
            let want_sign = 1u128 << (bits - 1);
            let want_mag = want_sign - 1;

            assert_eq!(
                sign, want_sign,
                "{dt:?}: narrow_sign_masks says sign={sign:#X}, but elem_bits \
                 says {bits} bits so the top bit is {want_sign:#X}. One of the \
                 two tables is wrong -- this test cannot say which, only that \
                 they no longer agree"
            );
            assert_eq!(
                mag, want_mag,
                "{dt:?}: magnitude mask {mag:#X} is not the complement of sign \
                 {sign:#X} within {bits} bits ({want_mag:#X})"
            );
            assert_eq!(
                sign | mag,
                (1u128 << bits) - 1,
                "{dt:?}: sign and magnitude must partition the storage width \
                 exactly -- no bit unclaimed, none claimed twice"
            );
            assert_eq!(sign & mag, 0, "{dt:?}: the two masks must not overlap");
        }

        // Control: a dtype that is NOT a narrow float must decline, or the loop
        // above is vacuous for any type someone adds to the match later.
        assert!(
            crate::ir::narrow_sign_masks(ElementKind::F32).is_none(),
            "control: f32 is not a NARROW float and must decline, so this \
             family stays the set the emitter actually mask-edits"
        );
    }

    /// Pins the module doc's coverage list to the code.
    ///
    /// The doc previously claimed `Access::Contraction` was deferred for a long
    /// time after `evaluate` began dispatching it. A stale scope note is the one
    /// defect an ordinary test cannot catch: every implementation test passed,
    /// because the implementation was fine — it was the *claim about* the
    /// implementation that had rotted, and a reader trusting it would skip
    /// writing a differential test that already worked.
    ///
    /// This closes that by construction. The `match` is exhaustive over
    /// [`Access`], so a new access pattern breaks the build here and whoever adds
    /// it must classify it — next to the doc that has to agree. It needs no
    /// constructed values: exhaustiveness is checked at compile time.
    #[derive(Debug, PartialEq, Eq)]
    enum Coverage {
        /// `evaluate` dispatches it to a real evaluator.
        Evaluated,
        /// `evaluate` panics — genuinely not implemented.
        ///
        /// Unreachable since `eval_row_sort` shipped and every `Access` became
        /// `Evaluated`. Kept rather than deleted so the next deferred access has
        /// a home, and `#[expect]` rather than `#[allow]` deliberately: the
        /// moment something constructs this, the expectation goes unfulfilled and
        /// reds, which is the prompt to remove the attribute. An `#[allow]` would
        /// sit here silently forever.
        ///
        /// (`#[expect]` is safe here where it is not generally safe in this
        /// workspace: this lives inside `#[cfg(test)]`, so there is exactly one
        /// compilation of it and the dead-in-lib/live-in-test trap does not
        /// apply.)
        #[expect(dead_code, reason = "a home for the next deferred Access")]
        Deferred,
    }

    fn coverage(a: &Access) -> Coverage {
        match a {
            Access::Elementwise
            | Access::Reduction { .. }
            | Access::RowReduce { .. }
            | Access::Scan { .. }
            | Access::Window { .. }
            | Access::Im2Col { .. }
            // Covered since `eval_contraction` shipped. The module doc said
            // otherwise for long enough that it is worth naming here.
            | Access::Contraction { .. }
            // Covered since `eval_row_sort` shipped: NaN-greatest ordering in
            // both directions, stable index ties, and the TopK cap read off the
            // output operand's width.
            | Access::RowSort { .. } => Coverage::Evaluated,
        }
    }

    #[test]
    fn the_documented_coverage_matches_what_evaluate_dispatches() {
        // The real work is the exhaustive `match` above, enforced at compile
        // time. This asserts the one variant that is cheap to name, so the
        // classifier cannot be deleted without a test failing too.
        assert_eq!(coverage(&Access::Elementwise), Coverage::Evaluated);
        // Every variant is now Evaluated, which makes `Deferred` unreachable —
        // kept rather than deleted so the NEXT deferred access has somewhere to
        // go, and asserted here so "nothing is deferred" is a measured claim
        // rather than an empty enum nobody noticed.
        assert_eq!(
            coverage(&Access::RowSort {
                order: SortOrder::Asc,
                stable: true,
                out: SortOut::Values,
                limit: crate::ir::SortLimit::Full,
            }),
            Coverage::Evaluated,
            "RowSort is evaluated since eval_row_sort shipped"
        );
    }

    use super::*;
    use crate::ir::{BinaryOp, OobPolicy, OpDef, ReduceOp, input, konst};
    use crate::plan::build_plan;
    use unpopped_vocab::{ArchSku, OpCategory, StructureKey, structure_key};

    // --- plan/key builders --------------------------------------------------

    fn desc(shape: &[i64], strides: &[i64], dt: ElementKind) -> OperandDesc {
        OperandDesc::new(shape.len(), shape, strides, dt, 256)
    }

    fn key(cat: OpCategory, ops: &[OperandDesc]) -> StructureKey {
        structure_key(cat, ops, ArchSku::Sm89)
    }

    fn f32b(shape: &[i64], data: &[f32]) -> TypedBuffer {
        TypedBuffer::from_f32(shape, data)
    }

    // Decode helpers reused across the suite.
    fn i32s(b: &TypedBuffer) -> Vec<i32> {
        (0..b.len()).map(|i| b.bits_at(i) as u32 as i32).collect()
    }

    fn f32s(b: &TypedBuffer) -> Vec<f32> {
        (0..b.len())
            .map(|i| f32::from_bits(b.bits_at(i) as u32))
            .collect()
    }

    // The raw 32-bit storage pattern of element `i`.
    fn bit32(b: &TypedBuffer, i: usize) -> u32 {
        b.bits_at(i) as u32
    }

    // --- A. Elementwise -----------------------------------------------------

    // NOTE (oracle→kiss-ref consolidation, 2026-07-29): the pure float
    // VALUE-semantics elementwise self-tests that lived here —
    // `elementwise_add_contiguous`, `elementwise_relu_neg_zero_and_nan`,
    // `elementwise_maxmin_prop_signed_zero_ties_keep_a`,
    // `elementwise_affine_with_params` (and `probe_classes_add_bit_exact_zeros`
    // below) — were RETIRED. kiss-ref is now the single float value-semantics
    // reference; each edge is asserted against it via the in-tree converter in
    // `kiss_ref_diff::tests` (add_contiguous / relu signed-zero+NaN /
    // max_prop-min_prop a-on-ties / affine-via-runtime_scalar / signed-zero-add).
    // The `Access::Elementwise` arm and the PLUMBING tests below (raw-bit select,
    // int8 store-truncation, strided/broadcast/flipped/permuted views,
    // compute-dtype cmp/select) STAY — that is Baracuda emitter/layout territory
    // kiss-ref's f32/dense value-DAG model does not cover (the locked boundary).

    // NOTE (oracle→kiss-ref consolidation): `select_raw_bit_neg_zero_survives`
    // was RETIRED. kiss-ref's raw-bit select (resolve.rs:58, byte-identical
    // between 0.1.0 and the strengthened test at 5d0538b) is the equal-or-better
    // reference; Baracuda's select round-trips through the converter and
    // preserves -0.0 bit-for-bit, asserted in
    // `kiss_ref_diff::tests::select_raw_bit_neg_zero_through_kiss_ref` (which
    // also verified the previously-unproven select converter path). The
    // Access::Elementwise arm + `cmp_and_select_decide_in_compute_dtype` stay.

    // KILL-test: i8 (S8) add wraps at 127 (mod-2^8 store truncation).
    #[test]
    fn int8_add_wraps() {
        let op = OpDef::elementwise("add8", 2, &[ElementKind::I8], input(0) + input(1));
        let a = desc(&[2], &[1], ElementKind::I8);
        let k = key(OpCategory::BinaryElementwise, &[a, a, a]);
        let plan = build_plan(&op, &k);
        let ops = [a, a, a];
        let ins = [
            TypedBuffer::from_i8(&[2], &[100, 127]),
            TypedBuffer::from_i8(&[2], &[100, 1]),
        ];
        let out = evaluate(&plan, &ops, &ins, &[]);
        // 100+100=200 -> -56 (i8); 127+1=128 -> -128 (i8).
        assert_eq!(out[0].bytes[0] as i8, -56);
        assert_eq!(out[0].bytes[1] as i8, -128);
    }

    // Layout axes: broadcast (stride 0), flipped (neg stride), permute, base_offset.
    #[test]
    fn elementwise_broadcast_input() {
        let op = OpDef::elementwise("addb", 2, &[ElementKind::F32], input(0) + input(1));
        let a = desc(&[3], &[1], ElementKind::F32);
        let b = desc(&[3], &[0], ElementKind::F32); // broadcast scalar
        let k = key(OpCategory::BinaryElementwise, &[a, b, a]);
        let plan = build_plan(&op, &k);
        let ops = [a, b, a];
        let ins = [
            f32b(&[3], &[1.0, 2.0, 3.0]),
            TypedBuffer::new(
                ElementKind::F32,
                vec![3],
                vec![0],
                10.0f32.to_bits().to_le_bytes().to_vec(),
            ),
        ];
        let out = evaluate(&plan, &ops, &ins, &[]);
        assert_eq!(f32s(&out[0]), vec![11.0, 12.0, 13.0]);
    }

    #[test]
    fn elementwise_flipped_input() {
        // Input 0 read with a negative stride (reversed view); base_offset points
        // at the last element so reads walk backward in-bounds.
        let op = OpDef::elementwise("flip", 1, &[ElementKind::F32], input(0) + konst(0.0));
        let a = desc(&[3], &[-1], ElementKind::F32);
        let out_d = desc(&[3], &[1], ElementKind::F32);
        let k = key(OpCategory::UnaryElementwise, &[a, out_d]);
        let plan = build_plan(&op, &k);
        let ops = [a, out_d];
        let mut buf = f32b(&[3], &[1.0, 2.0, 3.0]);
        buf.strides = vec![-1];
        buf.base_offset = 2; // origin at element 2
        let ins = [buf];
        let out = evaluate(&plan, &ops, &ins, &[]);
        assert_eq!(f32s(&out[0]), vec![3.0, 2.0, 1.0]);
    }

    #[test]
    fn elementwise_base_offset_slice() {
        // A base_offset slice: read a 2-element window starting at element 1.
        let op = OpDef::elementwise("slice", 1, &[ElementKind::F32], input(0) + konst(0.0));
        let a = desc(&[2], &[1], ElementKind::F32);
        let k = key(OpCategory::UnaryElementwise, &[a, a]);
        let plan = build_plan(&op, &k);
        let ops = [a, a];
        let mut buf = f32b(&[4], &[10.0, 20.0, 30.0, 40.0]);
        buf.shape = vec![2];
        buf.strides = vec![1];
        buf.base_offset = 1;
        let ins = [buf];
        let out = evaluate(&plan, &ops, &ins, &[]);
        assert_eq!(f32s(&out[0]), vec![20.0, 30.0]);
    }

    // --- B. Reduction -------------------------------------------------------

    // NOTE (oracle→kiss-ref consolidation, Task 3): the pure float
    // VALUE-semantics reduction self-tests — `reduction_sum_last_axis`,
    // `reduction_max_nan_sticks`, `reduction_mean_divisor`, and
    // `reduction_outer_axis` (below) — were RETIRED. Each fold is asserted
    // against kiss-ref via the in-tree converter in `kiss_ref_diff::tests`
    // (reduce_sum_last / reduce_max_nan_sticks — pins kiss-ref's reduce-Max as
    // NaN-propagating / reduce_mean_divisor / reduce_outer_axis). The
    // `Access::Reduction` arm STAYS — exercised by shape.rs's oracle differential
    // and reserved for physical/int reduction plumbing (kiss-ref's f32/dense
    // value-DAG model does not cover those).

    // --- C. RowReduce (softmax, layernorm) ----------------------------------
    // NOTE (oracle→kiss-ref consolidation, Task 5 — prove-then-retire): the
    // rowreduce VALUE self-tests `rowreduce_softmax` and `rowreduce_layernorm`
    // were RETIRED into the `oracle_and_kiss_ref` DIFFERENTIALS in
    // `kiss_ref_diff::tests` (softmax_oracle_eq_kiss_ref /
    // layernorm_oracle_eq_kiss_ref). Each runs the op through BOTH the oracle
    // AND kiss-ref, asserts they agree within tolerance (exp/div/rsqrt are
    // transcendental, not bit-exact), AND asserts the ported independent
    // invariants on kiss-ref (softmax sums-to-1 + monotone; layernorm zero-mean
    // + hand-computed values). So the oracle RowReduce arm stays EXERCISED (leg 1)
    // and the value stays independently checked — equal-or-better than the
    // retired single-impl test.

    // --- D. Scan (cumsum, cummax; fwd/reverse/exclusive) --------------------

    // NOTE (oracle→kiss-ref consolidation, Task 4): the portable float scan
    // VALUE-semantics self-tests `scan_cumsum_forward` and
    // `scan_cumsum_exclusive_first_pos_identity` were RETIRED — asserted against
    // kiss-ref via the converter in `kiss_ref_diff::tests`
    // (scan_cumsum_forward / scan_cumsum_exclusive). The `Access::Scan` arm STAYS
    // (exercised by shape.rs + the cummax test below). `scan_cummax_*` is KEPT
    // WHOLE: it exercises REVERSE scan, an emission-level honest miss (flip
    // withdrawn — `semantics_dag` returns None) that kiss-ref cannot cover, so it
    // has no equal-or-better replacement to retire against.

    // --- I. Gather / scatter -----------------------------------------------

    fn i32b(shape: &[i64], data: &[i32]) -> TypedBuffer {
        let mut b = TypedBuffer::new(
            ElementKind::I32,
            shape.to_vec(),
            row_major(shape),
            vec![0u8; data.len() * 4],
        );
        for (i, &v) in data.iter().enumerate() {
            store_val(Val::Int(i128::from(v)), ElementKind::I32, &mut b.bytes, i);
        }
        b
    }

    fn row_major(shape: &[i64]) -> Vec<i64> {
        let mut st = vec![1i64; shape.len()];
        for d in (0..shape.len().saturating_sub(1)).rev() {
            st[d] = st[d + 1] * shape[d + 1];
        }
        st
    }

    /// A gather reads at a data-dependent coordinate.
    #[test]
    fn gather_permutes_by_its_index_operand() {
        let d = desc(&[1, 4], &[4, 1], ElementKind::F32);
        let i = desc(&[1, 4], &[4, 1], ElementKind::I32);
        let op = OpDef::gather(
            "g",
            &[ElementKind::F32],
            1,
            OobPolicy::Clamp,
            ElementKind::I32,
        );
        let k = key(OpCategory::UnaryElementwise, &[d, i, d]);
        let out = evaluate(
            &build_plan(&op, &k),
            &[d, i, d],
            &[
                f32b(&[1, 4], &[10.0, 20.0, 30.0, 40.0]),
                i32b(&[1, 4], &[3, 0, 2, 1]),
            ],
            &[],
        );
        assert_eq!(f32s(&out[0]), vec![40.0, 10.0, 30.0, 20.0]);
    }

    /// `Clamp` pins out-of-range indices to the ends, in BOTH directions.
    ///
    /// ⚠️ `Skip` and `ZeroFill` are NOT distinguishable here and that is a
    /// property of the oracle rather than of the semantics: `alloc_output`
    /// zeroes the destination, so "leave the cell alone" and "write zero"
    /// produce the same bytes. On device they differ whenever the caller
    /// pre-filled the buffer. **Recorded rather than tested, because a test
    /// asserting they agree would be asserting the oracle's limitation as if it
    /// were the contract.**
    #[test]
    fn gather_clamp_pins_both_ends_and_negative_counts_as_out_of_range() {
        let d = desc(&[1, 4], &[4, 1], ElementKind::F32);
        let i = desc(&[1, 4], &[4, 1], ElementKind::I32);
        let op = OpDef::gather(
            "gc",
            &[ElementKind::F32],
            1,
            OobPolicy::Clamp,
            ElementKind::I32,
        );
        let k = key(OpCategory::UnaryElementwise, &[d, i, d]);
        let out = evaluate(
            &build_plan(&op, &k),
            &[d, i, d],
            &[
                f32b(&[1, 4], &[10.0, 20.0, 30.0, 40.0]),
                i32b(&[1, 4], &[3, -1, 5, 1]),
            ],
            &[],
        );
        assert_eq!(
            f32s(&out[0]),
            vec![40.0, 10.0, 40.0, 20.0],
            "-1 clamps to index 0 and 5 clamps to index 3"
        );
    }

    /// ⚠️ A scatter iterates its SOURCE, not its destination.
    ///
    /// This is the bincount shape: 6 source elements into 2 bins. An
    /// implementation that walked the output shape would visit 2 positions and
    /// silently drop four fifths of the input — and every same-extent test above
    /// would still pass. That is why this one has unequal extents.
    #[test]
    fn scatter_add_iterates_the_source_not_the_destination() {
        let src = desc(&[1, 6], &[6, 1], ElementKind::I32);
        let idx = desc(&[1, 6], &[6, 1], ElementKind::I32);
        let dst = desc(&[1, 2], &[2, 1], ElementKind::I32);
        let op = OpDef::scatter_add("sa", &[ElementKind::I32], 1, ElementKind::I32);
        let k = key(OpCategory::UnaryElementwise, &[src, idx, dst]);
        let out = evaluate(
            &build_plan(&op, &k),
            &[src, idx, dst],
            &[
                i32b(&[1, 6], &[1, 2, 3, 4, 5, 6]),
                i32b(&[1, 6], &[0, 1, 0, 1, 0, 1]),
            ],
            &[],
        );
        assert_eq!(
            i32s(&out[0]),
            vec![9, 12],
            "1+3+5 into bin 0 and 2+4+6 into bin 1; a destination-shaped walk gives [1,2]"
        );
    }

    /// An out-of-range scatter target is dropped — v1 pins the policy to `Skip`.
    #[test]
    fn scatter_skips_out_of_range_targets() {
        let src = desc(&[1, 4], &[4, 1], ElementKind::I32);
        let idx = desc(&[1, 4], &[4, 1], ElementKind::I32);
        let dst = desc(&[1, 2], &[2, 1], ElementKind::I32);
        let op = OpDef::scatter_add("ss", &[ElementKind::I32], 1, ElementKind::I32);
        let k = key(OpCategory::UnaryElementwise, &[src, idx, dst]);
        let out = evaluate(
            &build_plan(&op, &k),
            &[src, idx, dst],
            &[
                i32b(&[1, 4], &[7, 100, 9, 200]),
                i32b(&[1, 4], &[0, 5, 1, -1]),
            ],
            &[],
        );
        assert_eq!(
            i32s(&out[0]),
            vec![7, 9],
            "the 5 and -1 targets are dropped"
        );
    }

    /// ⚠️ `bincount` has ONE input, and it is its own index, so the source of the walk
    /// is that input and not "the input that is not the index". Naming operand
    /// `usize::from(index_operand == 0)` named the OUTPUT instead, and 14 values into
    /// 6 bins counted only the first 6 of them. Every other scatter test here has a
    /// separate value operand, which is why nothing caught it; the data is longer than
    /// the bin count on purpose.
    #[test]
    fn bincount_counts_its_whole_input_not_the_first_bins_elements() {
        let x = desc(&[14], &[1], ElementKind::I32);
        let bins = desc(&[6], &[1], ElementKind::I32);
        let op = OpDef::bincount("bc", ElementKind::I32);
        let k = key(OpCategory::UnaryElementwise, &[x, bins]);
        let out = evaluate(
            &build_plan(&op, &k),
            &[x, bins],
            &[i32b(&[14], &[0, 2, 2, 0, 5, 1, 0, -1, 6, 9, 2, 1, -7, 0])],
            &[],
        );
        assert_eq!(
            i32s(&out[0]),
            vec![4, 2, 3, 0, 0, 1],
            "10 in-range values counted; -1, -7, 6 and 9 dropped; bin 4 never hit"
        );
    }

    /// Float `AtomicAdd` is bit-exact on the deterministic corpus.
    ///
    /// The values are integer-valued floats well inside f32's exact range
    /// (2^24), so every partial sum is exact and the result is independent of
    /// accumulation order — which is what makes an order-nondeterministic op
    /// checkable by VALUE rather than by a bound. See `docs/deferred.md`.
    #[test]
    fn scatter_add_f32_is_exact_on_the_integer_valued_corpus() {
        let src = desc(&[1, 6], &[6, 1], ElementKind::F32);
        let idx = desc(&[1, 6], &[6, 1], ElementKind::I32);
        let dst = desc(&[1, 2], &[2, 1], ElementKind::F32);
        let op = OpDef::scatter_add("saf", &[ElementKind::F32], 1, ElementKind::I32);
        let k = key(OpCategory::UnaryElementwise, &[src, idx, dst]);
        let out = evaluate(
            &build_plan(&op, &k),
            &[src, idx, dst],
            &[
                f32b(&[1, 6], &[1.0, 2.0, 4.0, 8.0, 16.0, 32.0]),
                i32b(&[1, 6], &[0, 1, 0, 1, 0, 1]),
            ],
            &[],
        );
        assert_eq!(
            f32s(&out[0]),
            vec![21.0, 42.0],
            "1+4+16 and 2+8+32, exact in f32 at any accumulation order"
        );
    }

    // --- H. RowSort ---------------------------------------------------------

    /// NaN orders GREATEST in BOTH directions — so ascending puts it last and
    /// **descending puts it FIRST**.
    ///
    /// The descending arm is the one a reader expects to be wrong, and a test
    /// that only checked ascending would pass against a comparator that treated
    /// NaN as smallest-under-reversal. Both arms, from one corpus.
    #[test]
    fn row_sort_orders_nan_greatest_in_both_directions() {
        let ind = desc(&[1, 4], &[4, 1], ElementKind::F32);
        let data = [3.0f32, f32::NAN, 1.0, 2.0];
        let k = key(OpCategory::Sorting, &[ind, ind]);

        let asc = OpDef::row_sort("s_asc", ElementKind::F32, SortOrder::Asc);
        let out = evaluate(
            &build_plan(&asc, &k),
            &[ind, ind],
            &[f32b(&[1, 4], &data)],
            &[],
        );
        let v = f32s(&out[0]);
        assert_eq!(&v[..3], &[1.0, 2.0, 3.0], "ascending values");
        assert!(v[3].is_nan(), "ascending puts NaN LAST, got {v:?}");

        let desc_op = OpDef::row_sort("s_desc", ElementKind::F32, SortOrder::Desc);
        let out = evaluate(
            &build_plan(&desc_op, &k),
            &[ind, ind],
            &[f32b(&[1, 4], &data)],
            &[],
        );
        let v = f32s(&out[0]);
        assert!(v[0].is_nan(), "descending puts NaN FIRST, got {v:?}");
        assert_eq!(&v[1..], &[3.0, 2.0, 1.0], "descending values after the NaN");
    }

    /// Ties keep their original index order, in both directions.
    ///
    /// `sort_by` is stable and reversing a comparator does not change which
    /// pairs compare Equal — so descending must NOT reverse ties. An
    /// implementation that sorted ascending then reversed the whole row would
    /// pass the ordering test above and fail this one.
    #[test]
    fn row_sort_ties_keep_their_original_index_in_both_directions() {
        // `row_sort_indices` builds `SortOut::Both`: input, VALUES out, INDICES
        // out. Three operands, and the indices are output 1.
        let ind = desc(&[1, 4], &[4, 1], ElementKind::F32);
        let iout = desc(&[1, 4], &[4, 1], ElementKind::I32);
        let data = [5.0f32, 1.0, 5.0, 1.0];
        let k = key(OpCategory::Sorting, &[ind, ind, iout]);
        let ops = [ind, ind, iout];

        let asc = OpDef::row_sort_indices("i_asc", ElementKind::F32, SortOrder::Asc);
        let out = evaluate(&build_plan(&asc, &k), &ops, &[f32b(&[1, 4], &data)], &[]);
        assert_eq!(
            i32s(&out[1]),
            vec![1, 3, 0, 2],
            "the two 1.0s keep 1<3, the two 5.0s keep 0<2"
        );
        assert_eq!(
            f32s(&out[0]),
            vec![1.0, 1.0, 5.0, 5.0],
            "Both's value output must agree with its own index output"
        );

        let dsc = OpDef::row_sort_indices("i_desc", ElementKind::F32, SortOrder::Desc);
        let out = evaluate(&build_plan(&dsc, &k), &ops, &[f32b(&[1, 4], &data)], &[]);
        assert_eq!(
            i32s(&out[1]),
            vec![0, 2, 1, 3],
            "descending reverses the VALUES but not the ties — a whole-row reverse would give [2,0,3,1]"
        );
    }

    /// A narrower output is a TopK cap, read off the operand rather than a flag.
    #[test]
    fn row_sort_truncates_to_the_output_width() {
        let ind = desc(&[1, 5], &[5, 1], ElementKind::F32);
        let outd = desc(&[1, 2], &[2, 1], ElementKind::F32);
        let data = [3.0f32, 1.0, 4.0, 1.5, 9.0];
        let k = key(OpCategory::Sorting, &[ind, outd]);
        let op = OpDef::row_sort("top2", ElementKind::F32, SortOrder::Desc);
        let out = evaluate(
            &build_plan(&op, &k),
            &[ind, outd],
            &[f32b(&[1, 5], &data)],
            &[],
        );
        assert_eq!(f32s(&out[0]), vec![9.0, 4.0], "top 2 descending");
    }

    /// Multiple rows are sorted independently.
    ///
    /// A row loop that computed `base` once, or shared the permutation, passes
    /// every single-row test above.
    #[test]
    fn row_sort_treats_rows_independently() {
        let ind = desc(&[2, 3], &[3, 1], ElementKind::F32);
        let data = [3.0f32, 1.0, 2.0, 9.0, 7.0, 8.0];
        let k = key(OpCategory::Sorting, &[ind, ind]);
        let op = OpDef::row_sort("rows", ElementKind::F32, SortOrder::Asc);
        let out = evaluate(
            &build_plan(&op, &k),
            &[ind, ind],
            &[f32b(&[2, 3], &data)],
            &[],
        );
        assert_eq!(f32s(&out[0]), vec![1.0, 2.0, 3.0, 7.0, 8.0, 9.0]);
    }

    #[test]
    fn scan_cummax_forward_and_reverse_and_exclusive() {
        let ind = desc(&[1, 4], &[4, 1], ElementKind::F32);
        let data = [3.0f32, 1.0, 4.0, 2.0];

        // forward inclusive
        let op = OpDef::scan_simple(
            "cummax",
            &[ElementKind::F32],
            ReduceOp::Max,
            1,
            false,
            false,
        );
        let k = key(OpCategory::Scan, &[ind, ind]);
        let plan = build_plan(&op, &k);
        let out = evaluate(&plan, &[ind, ind], &[f32b(&[1, 4], &data)], &[]);
        assert_eq!(f32s(&out[0]), vec![3.0, 3.0, 4.0, 4.0]);

        // reverse inclusive
        let op = OpDef::scan_simple(
            "cummax_r",
            &[ElementKind::F32],
            ReduceOp::Max,
            1,
            true,
            false,
        );
        let plan = build_plan(&op, &k);
        let out = evaluate(&plan, &[ind, ind], &[f32b(&[1, 4], &data)], &[]);
        assert_eq!(f32s(&out[0]), vec![4.0, 4.0, 4.0, 2.0]);

        // forward EXCLUSIVE: first pos = -inf (Max monoid identity).
        let op = OpDef::scan_simple(
            "cummax_e",
            &[ElementKind::F32],
            ReduceOp::Max,
            1,
            false,
            true,
        );
        let plan = build_plan(&op, &k);
        let out = evaluate(&plan, &[ind, ind], &[f32b(&[1, 4], &data)], &[]);
        let g = f32s(&out[0]);
        assert_eq!(g[0], f32::NEG_INFINITY, "excl cummax first pos = -inf");
        assert_eq!(&g[1..], &[3.0, 3.0, 4.0]);
    }

    // --- E. Window (maxpool, avgpool) ---------------------------------------

    #[test]
    fn window_maxpool() {
        // k_in=4, size=2, stride=2 -> k_out=2. max over [1,2],[3,4].
        let op = OpDef::window_simple(
            "maxpool",
            &[ElementKind::F32],
            ReduceOp::Max,
            1,
            2,
            2,
            1,
            0,
            0,
            false,
        );
        let ind = desc(&[1, 4], &[4, 1], ElementKind::F32);
        let outd = desc(&[1, 2], &[2, 1], ElementKind::F32);
        let k = key(OpCategory::Pooling, &[ind, outd]);
        let plan = build_plan(&op, &k);
        let ops = [ind, outd];
        let ins = [f32b(&[1, 4], &[1.0, 2.0, 3.0, 4.0])];
        let out = evaluate(&plan, &ops, &ins, &[]);
        assert_eq!(f32s(&out[0]), vec![2.0, 4.0]);
    }

    // KILL-test: avg_pool count_include_pad divisor.
    #[test]
    fn window_avgpool_count_include_pad_divisor() {
        // k_in=4, size=3, stride=2, pad_lo=1, pad_hi=1 -> k_out=2.
        // o=0: taps p=-1(pad),0,1 -> valid {x0,x1}. o=1: p=1,2,3 -> {x1,x2,x3}.
        let ind = desc(&[1, 4], &[4, 1], ElementKind::F32);
        let outd = desc(&[1, 2], &[2, 1], ElementKind::F32);
        let k = key(OpCategory::Pooling, &[ind, outd]);
        let data = [2.0f32, 4.0, 6.0, 8.0];

        // count_include_pad = false: divide by valid count.
        let op = OpDef::window_simple(
            "avg",
            &[ElementKind::F32],
            ReduceOp::Mean,
            1,
            3,
            2,
            1,
            1,
            1,
            false,
        );
        let plan = build_plan(&op, &k);
        let out = evaluate(&plan, &[ind, outd], &[f32b(&[1, 4], &data)], &[]);
        // o0 = (2+4)/2 = 3 ; o1 = (4+6+8)/3 = 6.
        assert_eq!(f32s(&out[0]), vec![3.0, 6.0]);

        // count_include_pad = true: divide by size (3).
        let op = OpDef::window_simple(
            "avg_cip",
            &[ElementKind::F32],
            ReduceOp::Mean,
            1,
            3,
            2,
            1,
            1,
            1,
            true,
        );
        let plan = build_plan(&op, &k);
        let out = evaluate(&plan, &[ind, outd], &[f32b(&[1, 4], &data)], &[]);
        // o0 = (2+4)/3 ; o1 = (4+6+8)/3 = 6.
        let g = f32s(&out[0]);
        assert!((g[0] - 6.0 / 3.0).abs() < 1e-6);
        assert!((g[1] - 6.0).abs() < 1e-6);
    }

    // --- G. Im2Col ----------------------------------------------------------

    #[test]
    fn im2col_3x3() {
        // N=1,C=1,H=3,W=3; 3x3 kernel, stride 1, pad 1, dil 1 -> oH=oW=3.
        // Output [1, 9, 9]. Verify the center tap (ki=1,kj=1) reproduces the input,
        // and a corner tap zero-pads out of bounds.
        let op = OpDef::im2col_2d("im2col", ElementKind::F32, (3, 3), (1, 1), (1, 1), (1, 1));
        let ind = desc(&[1, 1, 3, 3], &[9, 9, 3, 1], ElementKind::F32);
        let outd = desc(&[1, 9, 9], &[81, 9, 1], ElementKind::F32);
        let k = key(OpCategory::Convolution, &[ind, outd]);
        let plan = build_plan(&op, &k);
        let ops = [ind, outd];
        let data: Vec<f32> = (1..=9).map(|v| v as f32).collect();
        let ins = [f32b(&[1, 1, 3, 3], &data)];
        let out = evaluate(&plan, &ops, &ins, &[]);
        let g = f32s(&out[0]);
        // Center tap row = c*9 + ki*3 + kj = 0*9 + 1*3 + 1 = 4; each col=oh*3+ow.
        for oh in 0..3usize {
            for ow in 0..3usize {
                let col = oh * 3 + ow;
                let val = g[4 * 9 + col];
                assert_eq!(val, data[oh * 3 + ow], "center tap = input");
            }
        }
        // Top-left tap (ki=0,kj=0) at output (oh=0,ow=0) is t=0: in_h=-1,in_w=-1
        // is out of bounds -> zero-pad.
        assert_eq!(g[0], 0.0, "OOB tap zero-pads");
    }

    // --- half codec round-trips + probe classes -----------------------------

    #[test]
    fn half_codec_roundtrip() {
        for &v in &[0.0f32, -0.0, 1.0, -2.5, 0.5, 65504.0] {
            let b = f32_to_f16_bits(v);
            let back = f16_to_f64(b);
            assert!(
                (back - f64::from(v)).abs() <= f64::from(v).abs() * 1e-2 + 1e-3,
                "{v} -> {back}"
            );
        }
        // ±inf and -0.0 map exactly.
        assert_eq!(f16_to_f64(f32_to_f16_bits(f32::INFINITY)), f64::INFINITY);
        assert!(f16_to_f64(f32_to_f16_bits(-0.0)).is_sign_negative());
        // bf16 preserves the high mantissa; 1.0 exact.
        assert_eq!(bf16_to_f64(f32_to_bf16_bits(1.0)), 1.0);
        // NaN stays NaN in both codecs.
        assert!(f16_to_f64(f32_to_f16_bits(f32::NAN)).is_nan());
        assert!(bf16_to_f64(f32_to_bf16_bits(f32::NAN)).is_nan());
    }

    /// ROADMAP U26: an `f64` is rounded to a 16-bit float once. `1 + 2^-11` is the tie between `0x3C00` and `0x3C01`;
    /// a hair above it must round up, which a first rounding to `f32` (landing on the tie, then to even) got wrong.
    #[test]
    fn an_f64_rounds_once_to_a_16_bit_float() {
        let tie = 1.0 + 2f64.powi(-11);
        assert_eq!(encode_float(tie, ElementKind::F16), 0x3C00); // the tie itself goes to even
        assert_eq!(encode_float(tie + 1e-12, ElementKind::F16), 0x3C01);
        assert_eq!(encode_float(tie - 1e-12, ElementKind::F16), 0x3C00);
        assert_eq!(encode_float(-(tie + 1e-12), ElementKind::F16), 0xBC01);
        // Bf16: `1 + 2^-8` is the tie between `0x3F80` and `0x3F81`
        let bt = 1.0 + 2f64.powi(-8);
        assert_eq!(encode_float(bt, ElementKind::Bf16), 0x3F80);
        assert_eq!(encode_float(bt + 1e-15, ElementKind::Bf16), 0x3F81);
        // beyond `f32`'s range both ends still saturate / underflow with their sign
        assert_eq!(encode_float(1e300, ElementKind::F16), 0x7C00);
        assert_eq!(encode_float(-1e300, ElementKind::Bf16), 0xFF80);
        assert_eq!(encode_float(-1e-300, ElementKind::F16), 0x8000);
        assert_eq!(encode_float(f64::MAX, ElementKind::Bf16), 0x7F80);
        // a NaN keeps its sign
        assert_eq!(encode_float(-f64::NAN, ElementKind::F16) & 0x8000, 0x8000);
    }

    /// ROADMAP U25: a 16-bit float NaN keeps its sign bit through the decode and every narrow-to-narrow re-encode, as
    /// hardware does, so a negative NaN is not silently turned positive.
    #[test]
    fn a_narrow_float_nan_keeps_its_sign() {
        for (pos, neg) in [(0x7E00u16, 0xFE00u16), (0x7C01, 0xFC01), (0x7FFF, 0xFFFF)] {
            assert!(
                f16_to_f64(pos).is_nan() && f16_to_f64(pos).is_sign_positive(),
                "{pos:#06x}"
            );
            assert!(
                f16_to_f64(neg).is_nan() && f16_to_f64(neg).is_sign_negative(),
                "{neg:#06x}"
            );
            // F16 to F16 and F16 to Bf16 through the compute domain keep the sign bit
            let (h, b) = (
                f32_to_f16_bits(f16_to_f64(neg) as f32),
                f32_to_bf16_bits(f16_to_f64(neg) as f32),
            );
            assert_eq!(h & 0x8000, 0x8000, "{neg:#06x} -> f16");
            assert_eq!(b & 0x8000, 0x8000, "{neg:#06x} -> bf16");
            assert_eq!(
                f32_to_bf16_bits(f16_to_f64(pos) as f32) & 0x8000,
                0,
                "{pos:#06x} -> bf16"
            );
        }
    }

    /// The f32 inputs that decide a 16-bit encoder: every finite value of the
    /// format, every midpoint between neighbours (the ties), one f32 ulp either
    /// side of each midpoint, the overflow edges, and a pseudo-random sweep.
    ///
    /// `decode` turns a pattern into its exact value and `inf` is the format's
    /// +infinity pattern. Midpoints of a 16-bit format need at most 12
    /// significant bits, so every one is exact in f32 (asserted).
    fn half_encoder_probes(decode: impl Fn(u16) -> f64, inf: u16) -> Vec<f32> {
        let mut v = Vec::new();
        for p in 0u16..inf {
            let a = decode(p);
            let b = decode(p + 1);
            if !(a.is_finite() && b.is_finite()) {
                continue;
            }
            let mid = ((a + b) / 2.0) as f32;
            assert_eq!(f64::from(mid), (a + b) / 2.0, "midpoint not exact in f32");
            for x in [a as f32, mid, f32::from_bits(mid.to_bits() + 1)] {
                v.push(x);
                v.push(-x);
            }
            if mid.to_bits() > 0 {
                let below = f32::from_bits(mid.to_bits() - 1);
                v.push(below);
                v.push(-below);
            }
        }
        // The overflow edges, by bits: f16's 65520 (the first value that rounds
        // to infinity) and one ulp below it; bf16's equivalent, (2 - 2^-8) * 2^127.
        for x in [
            f32::from_bits(0x477F_E000),
            f32::from_bits(0x477F_DFFF),
            f32::from_bits(0x7F7F_8000),
            f32::from_bits(0x7F7F_7FFF),
            f32::MAX,
            f32::INFINITY,
            f32::MIN_POSITIVE,
            f32::from_bits(1),
            0.0,
        ] {
            v.push(x);
            v.push(-x);
        }
        // A fixed LCG, so a failure reproduces.
        let mut s: u32 = 0x2545_F491;
        for _ in 0..1_000_000 {
            s = s.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            v.push(f32::from_bits(s));
        }
        v
    }

    /// The oracle's f16 codec agrees with the `half` crate on every probe.
    ///
    /// This oracle is what the emitted C codec (`cfamily::half_helpers`) is held
    /// to bit for bit, and until now its only test was `half_codec_roundtrip`: a
    /// loose round-trip on six values, which an encoder that truncates instead of
    /// rounding also passes. Shown to discriminate: breaking ties-to-even
    /// (`rem >= half`) fails at `0x38801000`.
    #[test]
    fn the_oracle_f16_codec_matches_an_independent_one() {
        for p in 0u16..=0xFFFF {
            let want = f64::from(half::f16::from_bits(p).to_f32());
            let got = f16_to_f64(p);
            assert!(
                (want.is_nan() && got.is_nan()) || want.to_bits() == got.to_bits(),
                "decode {p:#06x}: oracle {got}, half {want}"
            );
        }
        let mut checked = 0usize;
        for x in half_encoder_probes(f16_to_f64, 0x7C00) {
            let want = half::f16::from_f32(x).to_bits();
            let got = f32_to_f16_bits(x);
            if x.is_nan() {
                assert!(f16_to_f64(got).is_nan(), "encode NaN {x:?} gave {got:#06x}");
            } else {
                assert_eq!(got, want, "encode {x:e} ({:#010x})", x.to_bits());
            }
            checked += 1;
        }
        assert!(checked > 1_000_000, "only {checked} probes");
    }

    /// The same for bf16, whose ties sit at a different bit. Shown to
    /// discriminate: rounding ties away from zero fails at `0x00008000`.
    #[test]
    fn the_oracle_bf16_codec_matches_an_independent_one() {
        for p in 0u16..=0xFFFF {
            let want = f64::from(half::bf16::from_bits(p).to_f32());
            let got = bf16_to_f64(p);
            assert!(
                (want.is_nan() && got.is_nan()) || want.to_bits() == got.to_bits(),
                "decode {p:#06x}: oracle {got}, half {want}"
            );
        }
        let mut checked = 0usize;
        for x in half_encoder_probes(bf16_to_f64, 0x7F80) {
            let want = half::bf16::from_f32(x).to_bits();
            let got = f32_to_bf16_bits(x);
            if x.is_nan() {
                assert!(
                    bf16_to_f64(got).is_nan(),
                    "encode NaN {x:?} gave {got:#06x}"
                );
            } else {
                assert_eq!(got, want, "encode {x:e} ({:#010x})", x.to_bits());
            }
            checked += 1;
        }
        assert!(checked > 1_000_000, "only {checked} probes");
    }

    // (`probe_classes_add_bit_exact_zeros` retired here — signed-zero add is now
    // referenced against kiss-ref in `kiss_ref_diff::tests::signed_zero_add_through_kiss_ref`.)

    // --- H. Contraction (matmul) --------------------------------------------
    // NOTE (oracle→kiss-ref consolidation, Task 6 — prove-then-retire): the
    // matmul VALUE self-tests (identity / relu-epilogue / bias-relu / batched)
    // and their gemm cell builders were RETIRED into the `oracle_and_kiss_ref`
    // DIFFERENTIALS in `kiss_ref_diff::tests` (matmul_identity /
    // matmul_relu_epilogue / matmul_bias_relu / batched_matmul, all suffixed
    // `_oracle_eq_kiss_ref`). Each runs the op through BOTH the oracle AND
    // kiss-ref over an exactly-representable integer cell, asserts they agree
    // BIT-for-bit, AND asserts kiss-ref equals the hand-computed value. So the
    // oracle Contraction arm stays EXERCISED (leg 1) and the value stays
    // independently checked — equal-or-better than the retired single-impl tests.

    // Task 6: `eval_contraction` reads operand layout by PHYSICAL strides (the
    // ground truth, matching the emitter's `operand_stride_binding`), not by
    // re-applying the op's `View::Permute` on top of already-transposed strides
    // (that would double-transpose). This is a VIEWED contraction — the op
    // carries `View::Permute` on rhs — so it exercises the `input_perm` path.
    #[test]
    fn oracle_matches_reference_transposed_rhs_f32() {
        use crate::ir::{ContractionAxes, View, reduced};
        let (m, k, n) = (2usize, 3usize, 2usize);
        // lhs [M,K] row-major; rhs LOGICAL [K,N] but stored transposed ([N,K], K unit).
        let lhs_d = desc(&[m as i64, k as i64], &[k as i64, 1], ElementKind::F32);
        let rhs_d = desc(&[k as i64, n as i64], &[1, k as i64], ElementKind::F32); // transposed
        let out_d = desc(&[m as i64, n as i64], &[n as i64, 1], ElementKind::F32);
        let op = OpDef::contraction(
            "matmul",
            &[ElementKind::F32],
            ContractionAxes::matmul(),
            reduced(0),
        )
        .with_views(vec![View::Identity, View::Permute { perm: vec![1, 0] }]);
        let kk = key(OpCategory::Gemm, &[lhs_d, rhs_d, out_d]);
        let plan = build_plan(&op, &kk);
        // lhs data: logical [M,K] row-major = [[1,2,3],[4,5,6]].
        let lhs = f32b(&[m as i64, k as i64], &[1., 2., 3., 4., 5., 6.]);
        // rhs LOGICAL [K,N]=[[10,11],[12,13],[14,15]] stored transposed [N,K] row-major
        // = flatten([[10,12,14],[11,13,15]]) so physical strides [1,k] read it back.
        let rhs = f32b(&[k as i64, n as i64], &[10., 12., 14., 11., 13., 15.]);
        let got = f32s(&evaluate(&plan, &[lhs_d, rhs_d, out_d], &[lhs, rhs], &[])[0]);
        // from-scratch [M,K]·[K,N], K ascending: want[m][n] = Σ_k lhsL[m][k]·rhsL[k][n]
        let lhs_l = [[1., 2., 3.], [4., 5., 6.]];
        let rhs_l = [[10., 11.], [12., 13.], [14., 15.]];
        let mut want = vec![0f32; m * n];
        for mi in 0..m {
            for ni in 0..n {
                let mut a = 0f32;
                for ki in 0..k {
                    a += lhs_l[mi][ki] * rhs_l[ki][ni];
                }
                want[mi * n + ni] = a;
            }
        }
        assert_eq!(
            got, want,
            "oracle transposed-rhs must equal the logical matmul reference"
        );
    }

    // Task 6 differential matrix: {lhs_order, rhs_order} x {rank-2, rank-3
    // batched}, each vs a from-scratch `[M,K]·[K,N]` (or batched) reference.
    //
    // Broadcast-batch rhs (stride 0) is deliberately NOT a matrix cell here:
    // at the time this matrix was written, `classify_mat_layout` (Task 1/2,
    // already shipped on this branch) declined ANY zero-stride axis as
    // non-packed regardless of position, so `derive_contraction` returned
    // `None` for a stride-0 batch axis and `build_plan` panicked
    // ("contraction cell must carry ContractionKey facts") before a plan
    // could even be built — verified directly with a throwaway probe during
    // development, not guessed. That gate sat upstream of Task 6 (in the
    // vocab crate) and was out of this task's surgical scope to relax.
    //
    // UPDATE (Task 8, commit 2203bcfb): `classify_mat_layout` now admits a
    // stride-0 axis (skips it in the packedness walk instead of declining),
    // so a broadcast-KV rhs is constructible. Task 9 adds the two
    // now-possible broadcast-KV (GQA) cells in a DEDICATED helper below
    // (`assert_broadcast_kv_matches_reference`) rather than folding them into
    // this matrix/helper, to keep this committed 8-cell matrix untouched.

    /// Realize a (possibly batched) operand's PHYSICAL storage for a chosen
    /// storage order: `logical[b]` is batch `b`'s row-major `rows*cols`
    /// logical matrix (`logical.len()==1` for the unbatched case). Batch, when
    /// present, is the OUTERMOST storage axis (v1 convention, per the Task 5
    /// emitter). `transposed=false` -> canonical row-major (`strides=
    /// [cols,1]`, physical == logical); `transposed=true` -> physically
    /// `[cols,rows]` row-major per batch (`strides=[1,rows]`), so reading
    /// logical `(r,c)` through the returned strides reproduces
    /// `logical[b][r*cols+c]`. Returns `(strides, physical_flat_data)`.
    fn layout_matrix(
        rows: i64,
        cols: i64,
        logical: &[Vec<f32>],
        transposed: bool,
    ) -> (Vec<i64>, Vec<f32>) {
        let (r, c) = (rows as usize, cols as usize);
        let mut strides = Vec::new();
        if logical.len() > 1 {
            strides.push(rows * cols); // batch stride: outermost, packed
        }
        if transposed {
            strides.extend_from_slice(&[1, rows]);
        } else {
            strides.extend_from_slice(&[cols, 1]);
        }
        let mut phys = Vec::with_capacity(logical.len() * r * c);
        for slice in logical {
            if transposed {
                let mut buf = vec![0f32; r * c];
                for ri in 0..r {
                    for ci in 0..c {
                        buf[ci * r + ri] = slice[ri * c + ci];
                    }
                }
                phys.extend(buf);
            } else {
                phys.extend_from_slice(slice);
            }
        }
        (strides, phys)
    }

    // Hand-verification of `layout_matrix`'s transposed branch: the rank-2
    // case reuses the EXACT rhs fixture from
    // `oracle_matches_reference_transposed_rhs_f32` above (hand-verified
    // against the brief: rhs LOGICAL [K,N]=[[10,11],[12,13],[14,15]] stored
    // transposed [N,K] row-major = [10,12,14,11,13,15], strides=[1,k]) as a
    // golden case, plus a from-scratch batched hand trace.
    #[test]
    fn layout_matrix_matches_hand_verified_fixtures() {
        // rank-2 transposed: matches the RED fixture above exactly.
        let (strides, phys) = layout_matrix(3, 2, &[vec![10., 11., 12., 13., 14., 15.]], true);
        assert_eq!(strides, vec![1, 3]);
        assert_eq!(phys, vec![10., 12., 14., 11., 13., 15.]);
        // canonical (identity) is a pass-through.
        let (strides, phys) = layout_matrix(3, 2, &[vec![10., 11., 12., 13., 14., 15.]], false);
        assert_eq!(strides, vec![2, 1]);
        assert_eq!(phys, vec![10., 11., 12., 13., 14., 15.]);
        // batched (B=2, rows=2, cols=2), transposed: batch stays outermost.
        // logical batch0=[[1,2],[3,4]], batch1=[[5,6],[7,8]]; transposed
        // per-batch physical [cols,rows] row-major = [[1,3],[2,4]] /
        // [[5,7],[6,8]] -> flat [1,3,2,4, 5,7,6,8]; strides=[rows*cols,1,rows]
        // = [4,1,2]. Hand-check logical(b=0,r=1,c=0)=3: offset =
        // 0*4 + 1*1 + 0*2 = 1 -> phys[1] = 3. ✓
        let (strides, phys) =
            layout_matrix(2, 2, &[vec![1., 2., 3., 4.], vec![5., 6., 7., 8.]], true);
        assert_eq!(strides, vec![4, 1, 2]);
        assert_eq!(phys, vec![1., 3., 2., 4., 5., 7., 6., 8.]);
    }

    /// One matrix cell: build a (possibly batched, possibly layout-permuted)
    /// contraction through the REAL `build_plan`/`evaluate` pipeline and
    /// assert it equals a from-scratch `[M,K]·[K,N]` reference (K ascending).
    fn assert_contraction_matches_reference(
        batched: bool,
        lhs_transposed: bool,
        rhs_transposed: bool,
    ) {
        use crate::ir::{ContractionAxes, View, reduced};
        let (m, k, n): (i64, i64, i64) = (2, 3, 2);
        let bdim: i64 = if batched { 2 } else { 1 };

        // Small distinct-per-batch integer data (exact in f32).
        let lhs_logical: Vec<Vec<f32>> = (0..bdim)
            .map(|bi| (0..m * k).map(|idx| (bi * 100 + idx + 1) as f32).collect())
            .collect();
        let rhs_logical: Vec<Vec<f32>> = (0..bdim)
            .map(|bi| (0..k * n).map(|idx| (bi * 100 + idx + 1) as f32).collect())
            .collect();

        let (lhs_strides, lhs_phys) = layout_matrix(m, k, &lhs_logical, lhs_transposed);
        let (rhs_strides, rhs_phys) = layout_matrix(k, n, &rhs_logical, rhs_transposed);

        let (lhs_shape, rhs_shape, out_shape): (Vec<i64>, Vec<i64>, Vec<i64>) = if batched {
            (vec![bdim, m, k], vec![bdim, k, n], vec![bdim, m, n])
        } else {
            (vec![m, k], vec![k, n], vec![m, n])
        };
        let out_strides = if batched {
            vec![m * n, n, 1]
        } else {
            vec![n, 1]
        };
        let rank = lhs_shape.len();

        let lhs_d = desc(&lhs_shape, &lhs_strides, ElementKind::F32);
        let rhs_d = desc(&rhs_shape, &rhs_strides, ElementKind::F32);
        let out_d = desc(&out_shape, &out_strides, ElementKind::F32);

        // The permutation `classify_mat_layout` itself derives for "swap the
        // last two axes, batch (if any) stays outermost": [1,0] at rank-2,
        // [0,2,1] at rank-3 (hand-derived and cross-checked against
        // `classify_mat_layout`'s algorithm for these exact strides). Not
        // cross-checked BY `build_plan` (a caller precondition per the
        // module docs — `read_strided` ignores it after Edit 1 regardless),
        // but kept truthful to what a real planner would emit.
        let perm_for = |transposed: bool| -> View {
            if !transposed {
                View::Identity
            } else if rank == 2 {
                View::Permute { perm: vec![1, 0] }
            } else {
                View::Permute {
                    perm: vec![0, 2, 1],
                }
            }
        };
        let views = vec![perm_for(lhs_transposed), perm_for(rhs_transposed)];

        let axes = if batched {
            ContractionAxes::batched_matmul()
        } else {
            ContractionAxes::matmul()
        };
        let op =
            OpDef::contraction("matmul", &[ElementKind::F32], axes, reduced(0)).with_views(views);
        let kk = key(OpCategory::Gemm, &[lhs_d, rhs_d, out_d]);
        let plan = build_plan(&op, &kk);

        let lhs_buf = f32b(&lhs_shape, &lhs_phys);
        let rhs_buf = f32b(&rhs_shape, &rhs_phys);
        let got = f32s(&evaluate(&plan, &[lhs_d, rhs_d, out_d], &[lhs_buf, rhs_buf], &[])[0]);

        // From-scratch reference, K ascending — independent of `eval_contraction`.
        let mut want = vec![0f32; (bdim * m * n) as usize];
        for bi in 0..bdim {
            for mi in 0..m {
                for ni in 0..n {
                    let mut acc = 0f32;
                    for ki in 0..k {
                        let a = lhs_logical[bi as usize][(mi * k + ki) as usize];
                        let b = rhs_logical[bi as usize][(ki * n + ni) as usize];
                        acc += a * b;
                    }
                    want[(bi * m * n + mi * n + ni) as usize] = acc;
                }
            }
        }
        assert_eq!(
            got, want,
            "contraction matrix cell mismatch: batched={batched} \
             lhs_transposed={lhs_transposed} rhs_transposed={rhs_transposed}"
        );
    }

    #[test]
    fn oracle_contraction_layout_batch_differential_matrix() {
        for &batched in &[false, true] {
            for &lhs_t in &[false, true] {
                for &rhs_t in &[false, true] {
                    assert_contraction_matches_reference(batched, lhs_t, rhs_t);
                }
            }
        }
    }

    // --- Task 9: broadcast-KV (GQA) differential cells ----------------------
    //
    // View-requirement probe (empirical, run before writing the fixtures
    // below): a throwaway `build_plan` call with the Cell-1 broadcast rhs
    // key/operands and a VIEW-FREE op (no `.with_views(...)`) succeeded —
    // `assert_valid_views` returns immediately when `op.views.is_empty()`
    // (plan.rs:1066), and `build_plan`'s `Access::Contraction` arm only
    // requires `key.contraction.is_some()` (plan.rs:369), which
    // `derive_contraction` now supplies via `classify_mat_layout`'s stride-0
    // admission (Task 8) — no view of any kind is consulted. Independently,
    // `eval_contraction` calls `read_strided(.., i, &coords, None)` for BOTH
    // lhs (index 0) and rhs (index 1) with a HARD-CODED `None` perm
    // (oracle.rs:1329-1330) — `input_perm` is only ever applied to a fused
    // BIAS operand (index >= 2) inside the epilogue, never to lhs/rhs — so a
    // `View::Permute`/`View::Broadcast` on lhs/rhs would not even be read by
    // the oracle. The same is true for the transposed-broadcast Cell 2: its
    // rhs strides `[0, 1, k]` are read directly off `OperandDesc::strides` by
    // logical coordinate, with no view involved. Conclusion: BOTH cells are
    // VIEW-FREE — no `.with_views(...)` call on either `OpDef`.

    /// One broadcast-KV (GQA) matrix cell: rhs is LOGICAL `[B,K,N]` with a
    /// stride-0 BATCH axis — the physical buffer holds exactly ONE `K·N`
    /// slice, read for every batch index (`bi*0 == 0`). lhs is a REAL batched
    /// `[B,M,K]` dense operand (GQA broadcasts KV=rhs; broadcast-LHS is not a
    /// real use case and is out of scope). Built through the REAL
    /// `build_plan`/`evaluate` pipeline — constructible since Task 8 made
    /// `classify_mat_layout` admit the stride-0 rhs batch axis — and checked
    /// against a from-scratch reference (K ascending) that reads the SAME
    /// rhs slice for every `bi`. `rhs_transposed=false` -> canonical `[K,N]`
    /// rhs slice (Cell 1, strides `[0,n,1]`); `rhs_transposed=true` ->
    /// physically `[N,K]` per slice (Cell 2, the real Kᵀ+GQA case, strides
    /// `[0,1,k]`). Reuses `layout_matrix` (already hand-verified in
    /// `layout_matrix_matches_hand_verified_fixtures` above) for the K,N
    /// slice's physical strides/data, then prepends the broadcast batch
    /// stride `0`.
    fn assert_broadcast_kv_matches_reference(rhs_transposed: bool) {
        use crate::ir::{ContractionAxes, reduced};
        let (b, m, k, n): (i64, i64, i64, i64) = (2, 2, 3, 2);

        // lhs: REAL batched [B,M,K] dense, distinct per-batch small integers.
        let lhs_logical: Vec<Vec<f32>> = (0..b)
            .map(|bi| (0..m * k).map(|idx| (bi * 100 + idx + 1) as f32).collect())
            .collect();
        let lhs_strides = vec![m * k, k, 1];
        let lhs_phys: Vec<f32> = lhs_logical.iter().flatten().copied().collect();

        // rhs: ONE [K,N] logical slice (not one per batch), broadcast via a
        // stride-0 batch axis prepended to the K,N strides `layout_matrix`
        // derives for the chosen orientation.
        let rhs_logical_slice: Vec<f32> = (0..k * n).map(|idx| (idx + 10) as f32).collect();
        let (kn_strides, rhs_phys) = layout_matrix(
            k,
            n,
            std::slice::from_ref(&rhs_logical_slice),
            rhs_transposed,
        );
        let mut rhs_strides = vec![0i64];
        rhs_strides.extend(kn_strides);

        let lhs_shape = vec![b, m, k];
        let rhs_shape = vec![b, k, n];
        let out_shape = vec![b, m, n];
        let out_strides = vec![m * n, n, 1];

        let lhs_d = desc(&lhs_shape, &lhs_strides, ElementKind::F32);
        let rhs_d = desc(&rhs_shape, &rhs_strides, ElementKind::F32);
        let out_d = desc(&out_shape, &out_strides, ElementKind::F32);

        // View-free (see the probe note above): no `.with_views(...)` call.
        let op = OpDef::contraction(
            "matmul",
            &[ElementKind::F32],
            ContractionAxes::batched_matmul(),
            reduced(0),
        );
        let kk_key = key(OpCategory::Gemm, &[lhs_d, rhs_d, out_d]);
        let plan = build_plan(&op, &kk_key);

        let lhs_buf = f32b(&lhs_shape, &lhs_phys);
        // Buffer shape is a storage convenience (only `.bytes`/`.base_offset`
        // are read by `read_strided`; the OperandDesc `rhs_d` above supplies
        // the strides that actually drive address math) — [k, n] holds
        // exactly the one slice's `k*n` elements, matching Task 6's
        // convention of sizing the buffer to the logical (here: per-slice)
        // element count.
        let rhs_buf = f32b(&[k, n], &rhs_phys);
        let got = f32s(&evaluate(&plan, &[lhs_d, rhs_d, out_d], &[lhs_buf, rhs_buf], &[])[0]);

        // From-scratch reference, independent of `eval_contraction`: the SAME
        // rhs slice for every batch, K ascending.
        let mut want = vec![0f32; (b * m * n) as usize];
        for bi in 0..b {
            for mi in 0..m {
                for ni in 0..n {
                    let mut acc = 0f32;
                    for ki in 0..k {
                        let a = lhs_logical[bi as usize][(mi * k + ki) as usize];
                        let rv = rhs_logical_slice[(ki * n + ni) as usize];
                        acc += a * rv;
                    }
                    want[(bi * m * n + mi * n + ni) as usize] = acc;
                }
            }
        }
        assert_eq!(
            got, want,
            "broadcast-KV cell mismatch: rhs_transposed={rhs_transposed} \
             (the SAME rhs slice must be read for every batch)"
        );
    }

    // Cell 1: broadcast-batch rhs, canonical [K,N] storage order (strides
    // [0,n,1]) — the plain GQA broadcast-KV case.
    #[test]
    fn oracle_broadcast_kv_canonical_rhs_matches_reference() {
        assert_broadcast_kv_matches_reference(false);
    }

    // Cell 2: broadcast-batch rhs, transposed [K,N] storage order (physically
    // [N,K] per slice, strides [0,1,k]) — the real Kᵀ+GQA cell.
    #[test]
    fn oracle_broadcast_kv_transposed_rhs_matches_reference() {
        assert_broadcast_kv_matches_reference(true);
    }

    // --- compare helper -----------------------------------------------------

    #[test]
    fn compare_bit_exact_and_tolerant() {
        let a = f32b(&[2], &[1.0, 2.0]);
        let b = f32b(&[2], &[1.0, 2.0]);
        assert!(compare(&a, &b, Fidelity::BitExact).is_ok());
        // A difference above the f32 ULP at 2.0 (~2.4e-7) so the bits truly differ.
        let c = f32b(&[2], &[1.0, 2.0 + 1e-6]);
        assert!(compare(&a, &c, Fidelity::BitExact).is_err());
        assert!(
            compare(
                &a,
                &c,
                Fidelity::Tolerant {
                    rel: 1e-5,
                    abs: 1e-5
                }
            )
            .is_ok()
        );
        // both-NaN equal under tolerance.
        let n1 = f32b(&[1], &[f32::NAN]);
        let n2 = f32b(&[1], &[f32::NAN]);
        assert!(compare(&n1, &n2, Fidelity::Tolerant { rel: 0.0, abs: 0.0 }).is_ok());
    }

    // --- extra layout: permuted (transpose) read ----------------------------

    #[test]
    fn elementwise_permute_transpose() {
        use crate::ir::View;
        // Read input 0 through a transpose view: iteration [2,3] reads producer [3,2].
        let mut op = OpDef::elementwise("t", 1, &[ElementKind::F32], input(0) + konst(0.0));
        op.views = vec![View::Permute { perm: vec![1, 0] }];
        // producer storage is [3,2] row-major, strides [2,1]; iteration shape [2,3].
        let a = desc(&[3, 2], &[2, 1], ElementKind::F32);
        let outd = desc(&[2, 3], &[3, 1], ElementKind::F32);
        let k = key(OpCategory::UnaryElementwise, &[a, outd]);
        let plan = build_plan(&op, &k);
        let ops = [a, outd];
        // producer[3][2] = [[1,2],[3,4],[5,6]] laid out row-major.
        let ins = [f32b(&[3, 2], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0])];
        let out = evaluate(&plan, &ops, &ins, &[]);
        // out[i][j] = producer[j][i]: row0 = [1,3,5], row1 = [2,4,6].
        assert_eq!(f32s(&out[0]), vec![1.0, 3.0, 5.0, 2.0, 4.0, 6.0]);
    }

    // (`reduction_outer_axis` retired here — outer-axis fold is now referenced
    // against kiss-ref in `kiss_ref_diff::tests::reduce_outer_axis_through_kiss_ref`.)

    // --- Adversarial-review regression kills (5 confirmed defects) -----------

    // NOTE (oracle→kiss-ref consolidation, closeout): the compute-dtype
    // cmp/select VALUE test `cmp_and_select_decide_in_compute_dtype` was RETIRED.
    // kiss-ref is monomorphic on the compute dtype (`eval_op::<f32>`,
    // resolve.rs:67), so it narrows the const into the f32 lane and decides the
    // compare in f32 — matching the kernel (and Baracuda's oracle fix). VERIFIED
    // + asserted against kiss-ref via the converter in
    // `kiss_ref_diff::tests::cmp_select_decide_in_compute_dtype_through_kiss_ref`
    // (the previously-unproven cmp/select-const converter path — no divergence).
    // The compute-dtype `sign` underflow + F32Strict `nextafter` tests below STAY:
    // the former is a compute-dtype-storage nuance kept on the oracle; the latter
    // rides `F32Strict`, a Baracuda compute-precision distinction kiss-ref (single
    // F32) does not model.

    // KILL (major): Sign/Step decide on the compute-dtype-rounded value. sign(a*b)
    // with a=b=1e-30f: the product underflows to 0.0f in f32 → sign 0; the pre-fix
    // oracle kept 1e-60 in f64 → sign 1.
    #[test]
    fn sign_of_underflowing_product_is_zero_in_f32() {
        let op = OpDef::elementwise(
            "sgn",
            2,
            &[ElementKind::F32],
            (input(0) * input(1)).unary(UnaryOp::Sign),
        );
        let a = desc(&[1], &[1], ElementKind::F32);
        let k = key(OpCategory::BinaryElementwise, &[a, a, a]);
        let plan = build_plan(&op, &k);
        let ops = [a, a, a];
        let ins = [f32b(&[1], &[1e-30]), f32b(&[1], &[1e-30])];
        let out = evaluate(&plan, &ops, &ins, &[]);
        assert_eq!(
            f32s(&out[0]),
            vec![0.0],
            "product underflows to 0.0f → sign 0"
        );
    }

    // KILL (major): F32Strict Nextafter steps the f32 lattice (not f64, which would
    // demote-round back to a silent no-op). next(1.0f → 2.0f) = the next f32.
    #[test]
    fn nextafter_f32strict_steps_the_f32_lattice() {
        let op = OpDef::elementwise(
            "naft",
            1,
            &[ElementKind::F32Strict],
            input(0).binary(BinaryOp::Nextafter, konst(2.0)),
        );
        let a = desc(&[1], &[1], ElementKind::F32Strict);
        let k = key(OpCategory::UnaryElementwise, &[a, a]);
        let plan = build_plan(&op, &k);
        let ops = [a, a];
        let ins = [f32b(&[1], &[1.0])];
        let out = evaluate(&plan, &ops, &ins, &[]);
        assert_eq!(
            bit32(&out[0], 0),
            1.0f32.to_bits() + 1,
            "F32Strict nextafter must step one f32 ULP, not no-op"
        );
    }

    // KILL (major): the Tolerant comparator rejects a non-equal infinity instead of
    // letting the infinite relative band swallow it. An output overflowing to +inf
    // must NOT silently match 0.0 / -inf; an equal +inf still passes.
    #[test]
    fn tolerant_compare_rejects_nonequal_infinity() {
        let tol = Fidelity::Tolerant {
            rel: 1e-5,
            abs: 1e-6,
        };
        let inf = f32b(&[1], &[f32::INFINITY]);
        assert!(
            compare(&inf, &f32b(&[1], &[0.0]), tol).is_err(),
            "+inf vs 0"
        );
        assert!(
            compare(&inf, &f32b(&[1], &[f32::NEG_INFINITY]), tol).is_err(),
            "+inf vs -inf (sign bug) must fail"
        );
        assert!(
            compare(&inf, &f32b(&[1], &[1e30]), tol).is_err(),
            "+inf vs finite must fail"
        );
        assert!(compare(&inf, &inf, tol).is_ok(), "equal +inf still passes");
    }

    // KILL (minor): integer Mean is out of scope — panic (mirroring the emitter),
    // not silently return the un-divided sum.
    #[test]
    #[should_panic(expected = "integer Mean")]
    fn integer_mean_reduction_panics() {
        let op = OpDef::reduction("imean", 1, &[ElementKind::I32], input(0), ReduceOp::Mean);
        let ind = desc(&[1, 4], &[4, 1], ElementKind::I32);
        let outd = desc(&[1], &[1], ElementKind::I32);
        let k = key(OpCategory::Reduction, &[ind, outd]);
        let plan = build_plan(&op, &k);
        let ops = [ind, outd];
        let ins = [TypedBuffer::from_i32(&[1, 4], &[2, 4, 6, 8])];
        let _ = evaluate(&plan, &ops, &ins, &[]); // must panic, not return sum=20
    }

    // KILL (minor): erf/erfc/gelu are accurate (libm ~1 ULP), tight enough to be a
    // real reference for an f32/f64 erf/gelu kernel. The v1 ~1e-7 rational fails
    // this ~1e-12 pin. gelu(1) = Φ(1); erf+erfc = 1.
    #[test]
    fn erf_family_is_accurate_f64() {
        assert!((erf(1.0) - 0.842_700_792_949_714_9).abs() < 1e-12, "erf(1)");
        assert!(
            (erfc(1.0) - 0.157_299_207_050_285_13).abs() < 1e-12,
            "erfc(1)"
        );
        assert!((erf(1.0) + erfc(1.0) - 1.0).abs() < 1e-15, "erf+erfc=1");
        assert!(
            (unary_op_f64(UnaryOp::Gelu, 1.0) - 0.841_344_746_068_542_9).abs() < 1e-12,
            "gelu(1) = Phi(1)"
        );
    }
}

#[cfg(test)]
mod codec_pins {
    //! Pins from the 2026-10-17 mutation audit of this file: the narrow-float encoders.
    //!
    //! The fp8 encoders are checked against the DECODERS, which the audit found already
    //! pinned: for every adjacent pair of representable magnitudes the exact midpoint
    //! ties to the even pattern, a hair above rounds up and a hair below rounds down.
    //! The f16 encoder is checked against the `half` crate, a third codec.
    use super::*;

    fn up(x: f64) -> f64 {
        f64::from_bits(x.to_bits() + 1)
    }
    fn down(x: f64) -> f64 {
        f64::from_bits(x.to_bits() - 1)
    }

    /// Walks adjacent magnitude patterns `0..=max_pat`, checking the three rounding
    /// cases at each midpoint, the exact value, and the sign.
    fn check_fp8(encode: fn(f64) -> u8, decode: fn(u8) -> f64, max_pat: u8) {
        for pat in 0..=max_pat {
            let v = decode(pat);
            assert_eq!(encode(v), pat, "exact {pat:#04x}");
            assert_eq!(encode(-v), 0x80 | pat, "negative {pat:#04x}");
            if pat == max_pat {
                continue;
            }
            let next = decode(pat + 1);
            let mid = (v + next) / 2.0;
            let even = if pat % 2 == 0 { pat } else { pat + 1 };
            assert_eq!(encode(mid), even, "tie at {pat:#04x}");
            assert_eq!(encode(-mid), 0x80 | even, "negative tie at {pat:#04x}");
            assert_eq!(encode(up(mid)), pat + 1, "above tie at {pat:#04x}");
            assert_eq!(encode(down(mid)), pat, "below tie at {pat:#04x}");
        }
    }

    #[test]
    fn e4m3fn_rounds_to_nearest_even_on_every_adjacent_pair() {
        check_fp8(f64_to_fp8_e4m3fn_bits, fp8_e4m3fn_to_f64, 0x7e);
    }

    #[test]
    fn e5m2_rounds_to_nearest_even_on_every_adjacent_pair() {
        check_fp8(f64_to_fp8_e5m2_bits, fp8_e5m2_to_f64, 0x7b);
    }

    #[test]
    fn e4m3fn_saturates_and_keeps_a_nan_sign() {
        assert_eq!(f64_to_fp8_e4m3fn_bits(448.0), 0x7e);
        assert_eq!(f64_to_fp8_e4m3fn_bits(449.0), 0x7e);
        assert_eq!(f64_to_fp8_e4m3fn_bits(1.0e9), 0x7e);
        assert_eq!(f64_to_fp8_e4m3fn_bits(f64::INFINITY), 0x7e);
        assert_eq!(f64_to_fp8_e4m3fn_bits(f64::NEG_INFINITY), 0xfe);
        assert_eq!(f64_to_fp8_e4m3fn_bits(f64::NAN), 0x7f);
        assert_eq!(f64_to_fp8_e4m3fn_bits(-f64::NAN), 0xff);
    }

    #[test]
    fn e5m2_overflows_to_infinity_at_the_halfway_point_and_keeps_a_nan_sign() {
        assert_eq!(f64_to_fp8_e5m2_bits(57344.0), 0x7b);
        assert_eq!(f64_to_fp8_e5m2_bits(down(61440.0)), 0x7b);
        assert_eq!(f64_to_fp8_e5m2_bits(61440.0), 0x7c);
        assert_eq!(f64_to_fp8_e5m2_bits(1.0e9), 0x7c);
        assert_eq!(f64_to_fp8_e5m2_bits(f64::INFINITY), 0x7c);
        assert_eq!(f64_to_fp8_e5m2_bits(f64::NEG_INFINITY), 0xfc);
        assert_eq!(f64_to_fp8_e5m2_bits(f64::NAN), 0x7f);
        assert_eq!(f64_to_fp8_e5m2_bits(-f64::NAN), 0xff);
    }

    #[test]
    fn f16_encoder_agrees_with_the_half_crate() {
        let mut probes: Vec<f32> = Vec::new();
        for bits in 0u16..0x7bff {
            let v = half::f16::from_bits(bits).to_f32();
            probes.push(v);
            probes.push(-v);
            let next = half::f16::from_bits(bits + 1).to_f32();
            let mid = ((f64::from(v) + f64::from(next)) / 2.0) as f32;
            for p in [
                mid,
                f32::from_bits(mid.to_bits() + 1),
                f32::from_bits(mid.to_bits() - 1),
            ] {
                probes.push(p);
                probes.push(-p);
            }
        }
        // Overflow edge and underflow edge, either side of each rounding boundary.
        for v in [
            65504.0f32,
            65519.0,
            65520.0,
            65536.0,
            1.0e10,
            2.0f32.powi(-24),
            2.0f32.powi(-25),
            2.0f32.powi(-26),
        ] {
            probes.push(v);
            probes.push(f32::from_bits(v.to_bits() + 1));
            probes.push(f32::from_bits(v.to_bits() - 1));
            probes.push(-v);
        }
        let mut x = 0x9e37_79b9u32;
        for _ in 0..200_000 {
            x ^= x << 13;
            x ^= x >> 17;
            x ^= x << 5;
            let f = f32::from_bits(x);
            if !f.is_nan() {
                probes.push(f);
            }
        }
        for f in probes {
            if f.is_nan() {
                continue;
            }
            assert_eq!(
                f32_to_f16_bits(f),
                half::f16::from_f32(f).to_bits(),
                "{f:e} ({:#010x})",
                f.to_bits()
            );
        }
        assert_eq!(f32_to_f16_bits(f32::INFINITY), 0x7c00);
        assert_eq!(f32_to_f16_bits(f32::NEG_INFINITY), 0xfc00);
        assert_eq!(f32_to_f16_bits(f32::NAN) & 0x7fff, 0x7e00);
        assert_eq!(f32_to_f16_bits(-f32::NAN), 0xfe00);
    }
}

#[cfg(test)]
mod scalar_math_pins {
    //! Pins from the 2026-10-17 mutation audit of this file: the scalar evaluators.
    //!
    //! Reference values for the float ops are generated from Python's `math` (the
    //! platform C library), which shares no code with the oracle's formulas.
    use super::*;

    fn close(got: f64, want: f64, tol: f64) -> bool {
        (got - want).abs() <= tol * want.abs().max(1.0)
    }

    /// `(op, x, expected)` from Python's `math`, an implementation independent of the oracle.
    // Reference outputs that happen to equal a named constant (sqrt(2), e, ln 2) are
    // still table data, not a use of the constant.
    #[allow(
        clippy::approx_constant,
        reason = "reference outputs that equal a named constant are table data, not a use of the constant"
    )]
    const UNARY: &[(UnaryOp, f64, f64)] = &[
        (UnaryOp::Neg, -3.5, 3.5),
        (UnaryOp::Neg, -2.0, 2.0),
        (UnaryOp::Neg, -1.0, 1.0),
        (UnaryOp::Neg, -0.5, 0.5),
        (UnaryOp::Neg, -0.25, 0.25),
        (UnaryOp::Neg, 0.25, -0.25),
        (UnaryOp::Neg, 0.5, -0.5),
        (UnaryOp::Neg, 1.0, -1.0),
        (UnaryOp::Neg, 1.5, -1.5),
        (UnaryOp::Neg, 2.0, -2.0),
        (UnaryOp::Neg, 3.5, -3.5),
        (UnaryOp::Abs, -3.5, 3.5),
        (UnaryOp::Abs, -2.0, 2.0),
        (UnaryOp::Abs, -1.0, 1.0),
        (UnaryOp::Abs, -0.5, 0.5),
        (UnaryOp::Abs, -0.25, 0.25),
        (UnaryOp::Abs, 0.25, 0.25),
        (UnaryOp::Abs, 0.5, 0.5),
        (UnaryOp::Abs, 1.0, 1.0),
        (UnaryOp::Abs, 1.5, 1.5),
        (UnaryOp::Abs, 2.0, 2.0),
        (UnaryOp::Abs, 3.5, 3.5),
        (UnaryOp::Sqr, -3.5, 12.25),
        (UnaryOp::Sqr, -2.0, 4.0),
        (UnaryOp::Sqr, -1.0, 1.0),
        (UnaryOp::Sqr, -0.5, 0.25),
        (UnaryOp::Sqr, -0.25, 0.0625),
        (UnaryOp::Sqr, 0.25, 0.0625),
        (UnaryOp::Sqr, 0.5, 0.25),
        (UnaryOp::Sqr, 1.0, 1.0),
        (UnaryOp::Sqr, 1.5, 2.25),
        (UnaryOp::Sqr, 2.0, 4.0),
        (UnaryOp::Sqr, 3.5, 12.25),
        (UnaryOp::Sqrt, 0.25, 0.5),
        (UnaryOp::Sqrt, 0.5, 0.7071067811865476),
        (UnaryOp::Sqrt, 1.0, 1.0),
        (UnaryOp::Sqrt, 1.5, 1.224744871391589),
        (UnaryOp::Sqrt, 2.0, 1.4142135623730951),
        (UnaryOp::Sqrt, 3.5, 1.8708286933869707),
        (UnaryOp::Sqrt, 10.0, 3.1622776601683795),
        (UnaryOp::Rsqrt, 0.25, 2.0),
        (UnaryOp::Rsqrt, 0.5, 1.414213562373095),
        (UnaryOp::Rsqrt, 1.0, 1.0),
        (UnaryOp::Rsqrt, 1.5, 0.8164965809277261),
        (UnaryOp::Rsqrt, 2.0, 0.7071067811865475),
        (UnaryOp::Rsqrt, 3.5, 0.5345224838248488),
        (UnaryOp::Rsqrt, 10.0, 0.31622776601683794),
        (UnaryOp::Recip, -3.5, -0.2857142857142857),
        (UnaryOp::Recip, -2.0, -0.5),
        (UnaryOp::Recip, -1.0, -1.0),
        (UnaryOp::Recip, -0.5, -2.0),
        (UnaryOp::Recip, -0.25, -4.0),
        (UnaryOp::Recip, 0.25, 4.0),
        (UnaryOp::Recip, 0.5, 2.0),
        (UnaryOp::Recip, 1.0, 1.0),
        (UnaryOp::Recip, 1.5, 0.6666666666666666),
        (UnaryOp::Recip, 2.0, 0.5),
        (UnaryOp::Recip, 3.5, 0.2857142857142857),
        (UnaryOp::Exp, -3.5, 0.0301973834223185),
        (UnaryOp::Exp, -2.0, 0.1353352832366127),
        (UnaryOp::Exp, -1.0, 0.36787944117144233),
        (UnaryOp::Exp, -0.5, 0.6065306597126334),
        (UnaryOp::Exp, -0.25, 0.7788007830714049),
        (UnaryOp::Exp, 0.25, 1.2840254166877414),
        (UnaryOp::Exp, 0.5, 1.6487212707001282),
        (UnaryOp::Exp, 1.0, 2.718281828459045),
        (UnaryOp::Exp, 1.5, 4.4816890703380645),
        (UnaryOp::Exp, 2.0, 7.38905609893065),
        (UnaryOp::Exp, 3.5, 33.11545195869231),
        (UnaryOp::Log, 0.25, -1.3862943611198906),
        (UnaryOp::Log, 0.5, -0.6931471805599453),
        (UnaryOp::Log, 1.0, 0.0),
        (UnaryOp::Log, 1.5, 0.4054651081081644),
        (UnaryOp::Log, 2.0, 0.6931471805599453),
        (UnaryOp::Log, 3.5, 1.252762968495368),
        (UnaryOp::Log, 10.0, 2.302585092994046),
        (UnaryOp::Tanh, -3.5, -0.9981778976111987),
        (UnaryOp::Tanh, -2.0, -0.9640275800758169),
        (UnaryOp::Tanh, -1.0, -0.7615941559557649),
        (UnaryOp::Tanh, -0.5, -0.46211715726000974),
        (UnaryOp::Tanh, -0.25, -0.24491866240370913),
        (UnaryOp::Tanh, 0.25, 0.24491866240370913),
        (UnaryOp::Tanh, 0.5, 0.46211715726000974),
        (UnaryOp::Tanh, 1.0, 0.7615941559557649),
        (UnaryOp::Tanh, 1.5, 0.9051482536448664),
        (UnaryOp::Tanh, 2.0, 0.9640275800758169),
        (UnaryOp::Tanh, 3.5, 0.9981778976111987),
        (UnaryOp::Sigmoid, -3.5, 0.029312230751356305),
        (UnaryOp::Sigmoid, -2.0, 0.11920292202211757),
        (UnaryOp::Sigmoid, -1.0, 0.2689414213699951),
        (UnaryOp::Sigmoid, -0.5, 0.3775406687981454),
        (UnaryOp::Sigmoid, -0.25, 0.4378234991142019),
        (UnaryOp::Sigmoid, 0.25, 0.5621765008857981),
        (UnaryOp::Sigmoid, 0.5, 0.6224593312018546),
        (UnaryOp::Sigmoid, 1.0, 0.7310585786300049),
        (UnaryOp::Sigmoid, 1.5, 0.8175744761936437),
        (UnaryOp::Sigmoid, 2.0, 0.8807970779778824),
        (UnaryOp::Sigmoid, 3.5, 0.9706877692486438),
        (UnaryOp::Relu, -3.5, 0.0),
        (UnaryOp::Relu, -2.0, 0.0),
        (UnaryOp::Relu, -1.0, 0.0),
        (UnaryOp::Relu, -0.5, 0.0),
        (UnaryOp::Relu, -0.25, 0.0),
        (UnaryOp::Relu, 0.25, 0.25),
        (UnaryOp::Relu, 0.5, 0.5),
        (UnaryOp::Relu, 1.0, 1.0),
        (UnaryOp::Relu, 1.5, 1.5),
        (UnaryOp::Relu, 2.0, 2.0),
        (UnaryOp::Relu, 3.5, 3.5),
        (UnaryOp::Erf, -3.5, -0.9999992569016276),
        (UnaryOp::Erf, -2.0, -0.9953222650189527),
        (UnaryOp::Erf, -1.0, -0.8427007929497149),
        (UnaryOp::Erf, -0.5, -0.5204998778130465),
        (UnaryOp::Erf, -0.25, -0.27632639016823696),
        (UnaryOp::Erf, 0.25, 0.27632639016823696),
        (UnaryOp::Erf, 0.5, 0.5204998778130465),
        (UnaryOp::Erf, 1.0, 0.8427007929497149),
        (UnaryOp::Erf, 1.5, 0.9661051464753108),
        (UnaryOp::Erf, 2.0, 0.9953222650189527),
        (UnaryOp::Erf, 3.5, 0.9999992569016276),
        (UnaryOp::Gelu, -3.5, -0.0008142017766243376),
        (UnaryOp::Gelu, -2.0, -0.04550026389635843),
        (UnaryOp::Gelu, -1.0, -0.15865525393145707),
        (UnaryOp::Gelu, -0.5, -0.15426876936299347),
        (UnaryOp::Gelu, -0.25, -0.10032341857926907),
        (UnaryOp::Gelu, 0.25, 0.14967658142073093),
        (UnaryOp::Gelu, 0.5, 0.3457312306370065),
        (UnaryOp::Gelu, 1.0, 0.8413447460685428),
        (UnaryOp::Gelu, 1.5, 1.399789198096713),
        (UnaryOp::Gelu, 2.0, 1.9544997361036418),
        (UnaryOp::Gelu, 3.5, 3.4991857982233756),
        (UnaryOp::Silu, -3.5, -0.10259280762974707),
        (UnaryOp::Silu, -2.0, -0.23840584404423515),
        (UnaryOp::Silu, -1.0, -0.2689414213699951),
        (UnaryOp::Silu, -0.5, -0.1887703343990727),
        (UnaryOp::Silu, -0.25, -0.10945587477855047),
        (UnaryOp::Silu, 0.25, 0.14054412522144952),
        (UnaryOp::Silu, 0.5, 0.3112296656009273),
        (UnaryOp::Silu, 1.0, 0.7310585786300049),
        (UnaryOp::Silu, 1.5, 1.2263617142904655),
        (UnaryOp::Silu, 2.0, 1.7615941559557649),
        (UnaryOp::Silu, 3.5, 3.3974071923702533),
        (UnaryOp::Sin, -3.5, 0.35078322768961984),
        (UnaryOp::Sin, -2.0, -0.9092974268256817),
        (UnaryOp::Sin, -1.0, -0.8414709848078965),
        (UnaryOp::Sin, -0.5, -0.479425538604203),
        (UnaryOp::Sin, -0.25, -0.24740395925452294),
        (UnaryOp::Sin, 0.25, 0.24740395925452294),
        (UnaryOp::Sin, 0.5, 0.479425538604203),
        (UnaryOp::Sin, 1.0, 0.8414709848078965),
        (UnaryOp::Sin, 1.5, 0.9974949866040544),
        (UnaryOp::Sin, 2.0, 0.9092974268256817),
        (UnaryOp::Sin, 3.5, -0.35078322768961984),
        (UnaryOp::Cos, -3.5, -0.9364566872907963),
        (UnaryOp::Cos, -2.0, -0.4161468365471424),
        (UnaryOp::Cos, -1.0, 0.5403023058681398),
        (UnaryOp::Cos, -0.5, 0.8775825618903728),
        (UnaryOp::Cos, -0.25, 0.9689124217106447),
        (UnaryOp::Cos, 0.25, 0.9689124217106447),
        (UnaryOp::Cos, 0.5, 0.8775825618903728),
        (UnaryOp::Cos, 1.0, 0.5403023058681398),
        (UnaryOp::Cos, 1.5, 0.0707372016677029),
        (UnaryOp::Cos, 2.0, -0.4161468365471424),
        (UnaryOp::Cos, 3.5, -0.9364566872907963),
        (UnaryOp::Floor, -3.5, -4.0),
        (UnaryOp::Floor, -2.0, -2.0),
        (UnaryOp::Floor, -1.0, -1.0),
        (UnaryOp::Floor, -0.5, -1.0),
        (UnaryOp::Floor, -0.25, -1.0),
        (UnaryOp::Floor, 0.25, 0.0),
        (UnaryOp::Floor, 0.5, 0.0),
        (UnaryOp::Floor, 1.0, 1.0),
        (UnaryOp::Floor, 1.5, 1.0),
        (UnaryOp::Floor, 2.0, 2.0),
        (UnaryOp::Floor, 3.5, 3.0),
        (UnaryOp::Ceil, -3.5, -3.0),
        (UnaryOp::Ceil, -2.0, -2.0),
        (UnaryOp::Ceil, -1.0, -1.0),
        (UnaryOp::Ceil, -0.5, 0.0),
        (UnaryOp::Ceil, -0.25, 0.0),
        (UnaryOp::Ceil, 0.25, 1.0),
        (UnaryOp::Ceil, 0.5, 1.0),
        (UnaryOp::Ceil, 1.0, 1.0),
        (UnaryOp::Ceil, 1.5, 2.0),
        (UnaryOp::Ceil, 2.0, 2.0),
        (UnaryOp::Ceil, 3.5, 4.0),
        (UnaryOp::Round, -2.5, -2.0),
        (UnaryOp::Round, -1.5, -2.0),
        (UnaryOp::Round, -0.5, 0.0),
        (UnaryOp::Round, 0.5, 0.0),
        (UnaryOp::Round, 1.5, 2.0),
        (UnaryOp::Round, 2.5, 2.0),
        (UnaryOp::Round, 3.2, 3.0),
        (UnaryOp::Round, -3.7, -4.0),
        (UnaryOp::Sign, -3.5, -1.0),
        (UnaryOp::Sign, -2.0, -1.0),
        (UnaryOp::Sign, -1.0, -1.0),
        (UnaryOp::Sign, -0.5, -1.0),
        (UnaryOp::Sign, -0.25, -1.0),
        (UnaryOp::Sign, 0.25, 1.0),
        (UnaryOp::Sign, 0.5, 1.0),
        (UnaryOp::Sign, 1.0, 1.0),
        (UnaryOp::Sign, 1.5, 1.0),
        (UnaryOp::Sign, 2.0, 1.0),
        (UnaryOp::Sign, 3.5, 1.0),
        (UnaryOp::Step, -3.5, 0.0),
        (UnaryOp::Step, -2.0, 0.0),
        (UnaryOp::Step, -1.0, 0.0),
        (UnaryOp::Step, -0.5, 0.0),
        (UnaryOp::Step, -0.25, 0.0),
        (UnaryOp::Step, 0.25, 1.0),
        (UnaryOp::Step, 0.5, 1.0),
        (UnaryOp::Step, 1.0, 1.0),
        (UnaryOp::Step, 1.5, 1.0),
        (UnaryOp::Step, 2.0, 1.0),
        (UnaryOp::Step, 3.5, 1.0),
        (UnaryOp::Erfc, -3.5, 1.9999992569016276),
        (UnaryOp::Erfc, -2.0, 1.9953222650189528),
        (UnaryOp::Erfc, -1.0, 1.842700792949715),
        (UnaryOp::Erfc, -0.5, 1.5204998778130465),
        (UnaryOp::Erfc, -0.25, 1.276326390168237),
        (UnaryOp::Erfc, 0.25, 0.7236736098317631),
        (UnaryOp::Erfc, 0.5, 0.4795001221869534),
        (UnaryOp::Erfc, 1.0, 0.1572992070502851),
        (UnaryOp::Erfc, 1.5, 0.033894853524689274),
        (UnaryOp::Erfc, 2.0, 0.004677734981047265),
        (UnaryOp::Erfc, 3.5, 7.430983723414126e-07),
        (UnaryOp::Trunc, -3.5, -3.0),
        (UnaryOp::Trunc, -2.0, -2.0),
        (UnaryOp::Trunc, -1.0, -1.0),
        (UnaryOp::Trunc, -0.5, 0.0),
        (UnaryOp::Trunc, -0.25, 0.0),
        (UnaryOp::Trunc, 0.25, 0.0),
        (UnaryOp::Trunc, 0.5, 0.0),
        (UnaryOp::Trunc, 1.0, 1.0),
        (UnaryOp::Trunc, 1.5, 1.0),
        (UnaryOp::Trunc, 2.0, 2.0),
        (UnaryOp::Trunc, 3.5, 3.0),
        (UnaryOp::Trunc, -2.7, -2.0),
        (UnaryOp::Trunc, 2.7, 2.0),
        (UnaryOp::Exp2, -3.5, 0.08838834764831845),
        (UnaryOp::Exp2, -2.0, 0.25),
        (UnaryOp::Exp2, -1.0, 0.5),
        (UnaryOp::Exp2, -0.5, 0.7071067811865476),
        (UnaryOp::Exp2, -0.25, 0.8408964152537145),
        (UnaryOp::Exp2, 0.25, 1.189207115002721),
        (UnaryOp::Exp2, 0.5, 1.4142135623730951),
        (UnaryOp::Exp2, 1.0, 2.0),
        (UnaryOp::Exp2, 1.5, 2.8284271247461903),
        (UnaryOp::Exp2, 2.0, 4.0),
        (UnaryOp::Exp2, 3.5, 11.313708498984761),
        (UnaryOp::Expm1, -3.5, -0.9698026165776815),
        (UnaryOp::Expm1, -2.0, -0.8646647167633873),
        (UnaryOp::Expm1, -1.0, -0.6321205588285577),
        (UnaryOp::Expm1, -0.5, -0.39346934028736663),
        (UnaryOp::Expm1, -0.25, -0.22119921692859512),
        (UnaryOp::Expm1, 0.25, 0.2840254166877415),
        (UnaryOp::Expm1, 0.5, 0.6487212707001282),
        (UnaryOp::Expm1, 1.0, 1.718281828459045),
        (UnaryOp::Expm1, 1.5, 3.4816890703380645),
        (UnaryOp::Expm1, 2.0, 6.38905609893065),
        (UnaryOp::Expm1, 3.5, 32.11545195869231),
        (UnaryOp::Log2, 0.25, -2.0),
        (UnaryOp::Log2, 0.5, -1.0),
        (UnaryOp::Log2, 1.0, 0.0),
        (UnaryOp::Log2, 1.5, 0.5849625007211562),
        (UnaryOp::Log2, 2.0, 1.0),
        (UnaryOp::Log2, 3.5, 1.8073549220576042),
        (UnaryOp::Log2, 10.0, 3.321928094887362),
        (UnaryOp::Log10, 0.25, -0.6020599913279624),
        (UnaryOp::Log10, 0.5, -0.3010299956639812),
        (UnaryOp::Log10, 1.0, 0.0),
        (UnaryOp::Log10, 1.5, 0.17609125905568124),
        (UnaryOp::Log10, 2.0, 0.3010299956639812),
        (UnaryOp::Log10, 3.5, 0.5440680443502757),
        (UnaryOp::Log10, 10.0, 1.0),
        (UnaryOp::Log1p, -0.5, -0.6931471805599453),
        (UnaryOp::Log1p, 0.25, 0.22314355131420976),
        (UnaryOp::Log1p, 1.0, 0.6931471805599453),
        (UnaryOp::Log1p, 3.5, 1.5040773967762742),
        (UnaryOp::Sinh, -3.5, -16.542627287635),
        (UnaryOp::Sinh, -2.0, -3.6268604078470186),
        (UnaryOp::Sinh, -1.0, -1.1752011936438014),
        (UnaryOp::Sinh, -0.5, -0.5210953054937474),
        (UnaryOp::Sinh, -0.25, -0.2526123168081683),
        (UnaryOp::Sinh, 0.25, 0.2526123168081683),
        (UnaryOp::Sinh, 0.5, 0.5210953054937474),
        (UnaryOp::Sinh, 1.0, 1.1752011936438014),
        (UnaryOp::Sinh, 1.5, 2.1292794550948173),
        (UnaryOp::Sinh, 2.0, 3.6268604078470186),
        (UnaryOp::Sinh, 3.5, 16.542627287635),
        (UnaryOp::Cosh, -3.5, 16.572824671057315),
        (UnaryOp::Cosh, -2.0, 3.7621956910836314),
        (UnaryOp::Cosh, -1.0, 1.5430806348152437),
        (UnaryOp::Cosh, -0.5, 1.1276259652063807),
        (UnaryOp::Cosh, -0.25, 1.0314130998795732),
        (UnaryOp::Cosh, 0.25, 1.0314130998795732),
        (UnaryOp::Cosh, 0.5, 1.1276259652063807),
        (UnaryOp::Cosh, 1.0, 1.5430806348152437),
        (UnaryOp::Cosh, 1.5, 2.352409615243247),
        (UnaryOp::Cosh, 2.0, 3.7621956910836314),
        (UnaryOp::Cosh, 3.5, 16.572824671057315),
        (UnaryOp::Tan, -3.5, -0.3745856401585947),
        (UnaryOp::Tan, -2.0, 2.185039863261519),
        (UnaryOp::Tan, -1.0, -1.5574077246549023),
        (UnaryOp::Tan, -0.5, -0.5463024898437905),
        (UnaryOp::Tan, -0.25, -0.25534192122103627),
        (UnaryOp::Tan, 0.25, 0.25534192122103627),
        (UnaryOp::Tan, 0.5, 0.5463024898437905),
        (UnaryOp::Tan, 1.0, 1.5574077246549023),
        (UnaryOp::Tan, 1.5, 14.101419947171719),
        (UnaryOp::Tan, 2.0, -2.185039863261519),
        (UnaryOp::Tan, 3.5, 0.3745856401585947),
        (UnaryOp::Asin, -0.9, -1.1197695149986342),
        (UnaryOp::Asin, -0.5, -0.5235987755982989),
        (UnaryOp::Asin, -0.25, -0.25268025514207865),
        (UnaryOp::Asin, 0.25, 0.25268025514207865),
        (UnaryOp::Asin, 0.5, 0.5235987755982989),
        (UnaryOp::Asin, 0.9, 1.1197695149986342),
        (UnaryOp::Acos, -0.9, 2.6905658417935308),
        (UnaryOp::Acos, -0.5, 2.0943951023931957),
        (UnaryOp::Acos, -0.25, 1.8234765819369754),
        (UnaryOp::Acos, 0.25, 1.318116071652818),
        (UnaryOp::Acos, 0.5, 1.0471975511965979),
        (UnaryOp::Acos, 0.9, 0.45102681179626236),
        (UnaryOp::Atan, -3.5, -1.2924966677897853),
        (UnaryOp::Atan, -2.0, -1.1071487177940904),
        (UnaryOp::Atan, -1.0, -0.7853981633974483),
        (UnaryOp::Atan, -0.5, -0.4636476090008061),
        (UnaryOp::Atan, -0.25, -0.24497866312686414),
        (UnaryOp::Atan, 0.25, 0.24497866312686414),
        (UnaryOp::Atan, 0.5, 0.4636476090008061),
        (UnaryOp::Atan, 1.0, 0.7853981633974483),
        (UnaryOp::Atan, 1.5, 0.982793723247329),
        (UnaryOp::Atan, 2.0, 1.1071487177940904),
        (UnaryOp::Atan, 3.5, 1.2924966677897853),
        (UnaryOp::Asinh, -3.5, -1.9657204716496515),
        (UnaryOp::Asinh, -2.0, -1.4436354751788103),
        (UnaryOp::Asinh, -1.0, -0.881373587019543),
        (UnaryOp::Asinh, -0.5, -0.48121182505960347),
        (UnaryOp::Asinh, -0.25, -0.24746646154726346),
        (UnaryOp::Asinh, 0.25, 0.24746646154726346),
        (UnaryOp::Asinh, 0.5, 0.48121182505960347),
        (UnaryOp::Asinh, 1.0, 0.881373587019543),
        (UnaryOp::Asinh, 1.5, 1.1947632172871094),
        (UnaryOp::Asinh, 2.0, 1.4436354751788103),
        (UnaryOp::Asinh, 3.5, 1.9657204716496515),
        (UnaryOp::Acosh, 1.0, 0.0),
        (UnaryOp::Acosh, 1.5, 0.9624236501192069),
        (UnaryOp::Acosh, 2.0, 1.3169578969248166),
        (UnaryOp::Acosh, 3.5, 1.9248473002384139),
        (UnaryOp::Acosh, 10.0, 2.993222846126381),
        (UnaryOp::Atanh, -0.9, -1.4722194895832204),
        (UnaryOp::Atanh, -0.5, -0.5493061443340549),
        (UnaryOp::Atanh, -0.25, -0.25541281188299536),
        (UnaryOp::Atanh, 0.25, 0.25541281188299536),
        (UnaryOp::Atanh, 0.5, 0.5493061443340549),
        (UnaryOp::Atanh, 0.9, 1.4722194895832204),
        (UnaryOp::Cbrt, -3.5, -1.5182944859378313),
        (UnaryOp::Cbrt, -2.0, -1.259921049894873),
        (UnaryOp::Cbrt, -1.0, -1.0),
        (UnaryOp::Cbrt, -0.5, -0.7937005259840997),
        (UnaryOp::Cbrt, -0.25, -0.6299605249474365),
        (UnaryOp::Cbrt, 0.25, 0.6299605249474365),
        (UnaryOp::Cbrt, 0.5, 0.7937005259840997),
        (UnaryOp::Cbrt, 1.0, 1.0),
        (UnaryOp::Cbrt, 1.5, 1.1447142425533319),
        (UnaryOp::Cbrt, 2.0, 1.259921049894873),
        (UnaryOp::Cbrt, 3.5, 1.5182944859378313),
        (UnaryOp::Cbrt, 8.0, 2.0),
        (UnaryOp::Cbrt, -27.0, -3.0),
        (UnaryOp::Lgamma, -3.5, -1.3090066849930417),
        (UnaryOp::Lgamma, -2.5, -0.05624371649767457),
        (UnaryOp::Lgamma, -1.5, 0.8600470153764812),
        (UnaryOp::Lgamma, -0.5, 1.265512123484645),
        (UnaryOp::Lgamma, 0.25, 1.2880225246980772),
        (UnaryOp::Lgamma, 0.5, 0.5723649429247004),
        (UnaryOp::Lgamma, 1.0, 0.0),
        (UnaryOp::Lgamma, 1.5, -0.12078223763524543),
        (UnaryOp::Lgamma, 2.0, 0.0),
        (UnaryOp::Lgamma, 3.5, 1.2009736023470738),
        (UnaryOp::Lgamma, 10.0, 12.801827480081467),
        (UnaryOp::Lgamma, 50.0, 144.5657439463449),
    ];

    /// `(op, a, b, expected)`, same provenance.
    // Reference outputs that happen to equal a named constant (sqrt(2), e, ln 2) are
    // still table data, not a use of the constant.
    #[allow(
        clippy::approx_constant,
        reason = "reference outputs that equal a named constant are table data, not a use of the constant"
    )]
    const BINARY: &[(BinaryOp, f64, f64, f64)] = &[
        (BinaryOp::Max, -3.5, -2.0, -2.0),
        (BinaryOp::Max, -3.5, -0.5, -0.5),
        (BinaryOp::Max, -3.5, 0.5, 0.5),
        (BinaryOp::Max, -3.5, 1.0, 1.0),
        (BinaryOp::Max, -3.5, 2.0, 2.0),
        (BinaryOp::Max, -3.5, 3.5, 3.5),
        (BinaryOp::Max, -1.5, -2.0, -1.5),
        (BinaryOp::Max, -1.5, -0.5, -0.5),
        (BinaryOp::Max, -1.5, 0.5, 0.5),
        (BinaryOp::Max, -1.5, 1.0, 1.0),
        (BinaryOp::Max, -1.5, 2.0, 2.0),
        (BinaryOp::Max, -1.5, 3.5, 3.5),
        (BinaryOp::Max, -0.5, -2.0, -0.5),
        (BinaryOp::Max, -0.5, -0.5, -0.5),
        (BinaryOp::Max, -0.5, 0.5, 0.5),
        (BinaryOp::Max, -0.5, 1.0, 1.0),
        (BinaryOp::Max, -0.5, 2.0, 2.0),
        (BinaryOp::Max, -0.5, 3.5, 3.5),
        (BinaryOp::Max, 0.5, -2.0, 0.5),
        (BinaryOp::Max, 0.5, -0.5, 0.5),
        (BinaryOp::Max, 0.5, 0.5, 0.5),
        (BinaryOp::Max, 0.5, 1.0, 1.0),
        (BinaryOp::Max, 0.5, 2.0, 2.0),
        (BinaryOp::Max, 0.5, 3.5, 3.5),
        (BinaryOp::Max, 1.0, -2.0, 1.0),
        (BinaryOp::Max, 1.0, -0.5, 1.0),
        (BinaryOp::Max, 1.0, 0.5, 1.0),
        (BinaryOp::Max, 1.0, 1.0, 1.0),
        (BinaryOp::Max, 1.0, 2.0, 2.0),
        (BinaryOp::Max, 1.0, 3.5, 3.5),
        (BinaryOp::Max, 2.5, -2.0, 2.5),
        (BinaryOp::Max, 2.5, -0.5, 2.5),
        (BinaryOp::Max, 2.5, 0.5, 2.5),
        (BinaryOp::Max, 2.5, 1.0, 2.5),
        (BinaryOp::Max, 2.5, 2.0, 2.5),
        (BinaryOp::Max, 2.5, 3.5, 3.5),
        (BinaryOp::Max, 1.0, 1.0, 1.0),
        (BinaryOp::Max, 2.0, 2.0, 2.0),
        (BinaryOp::Min, -3.5, -2.0, -3.5),
        (BinaryOp::Min, -3.5, -0.5, -3.5),
        (BinaryOp::Min, -3.5, 0.5, -3.5),
        (BinaryOp::Min, -3.5, 1.0, -3.5),
        (BinaryOp::Min, -3.5, 2.0, -3.5),
        (BinaryOp::Min, -3.5, 3.5, -3.5),
        (BinaryOp::Min, -1.5, -2.0, -2.0),
        (BinaryOp::Min, -1.5, -0.5, -1.5),
        (BinaryOp::Min, -1.5, 0.5, -1.5),
        (BinaryOp::Min, -1.5, 1.0, -1.5),
        (BinaryOp::Min, -1.5, 2.0, -1.5),
        (BinaryOp::Min, -1.5, 3.5, -1.5),
        (BinaryOp::Min, -0.5, -2.0, -2.0),
        (BinaryOp::Min, -0.5, -0.5, -0.5),
        (BinaryOp::Min, -0.5, 0.5, -0.5),
        (BinaryOp::Min, -0.5, 1.0, -0.5),
        (BinaryOp::Min, -0.5, 2.0, -0.5),
        (BinaryOp::Min, -0.5, 3.5, -0.5),
        (BinaryOp::Min, 0.5, -2.0, -2.0),
        (BinaryOp::Min, 0.5, -0.5, -0.5),
        (BinaryOp::Min, 0.5, 0.5, 0.5),
        (BinaryOp::Min, 0.5, 1.0, 0.5),
        (BinaryOp::Min, 0.5, 2.0, 0.5),
        (BinaryOp::Min, 0.5, 3.5, 0.5),
        (BinaryOp::Min, 1.0, -2.0, -2.0),
        (BinaryOp::Min, 1.0, -0.5, -0.5),
        (BinaryOp::Min, 1.0, 0.5, 0.5),
        (BinaryOp::Min, 1.0, 1.0, 1.0),
        (BinaryOp::Min, 1.0, 2.0, 1.0),
        (BinaryOp::Min, 1.0, 3.5, 1.0),
        (BinaryOp::Min, 2.5, -2.0, -2.0),
        (BinaryOp::Min, 2.5, -0.5, -0.5),
        (BinaryOp::Min, 2.5, 0.5, 0.5),
        (BinaryOp::Min, 2.5, 1.0, 1.0),
        (BinaryOp::Min, 2.5, 2.0, 2.0),
        (BinaryOp::Min, 2.5, 3.5, 2.5),
        (BinaryOp::Min, 1.0, 1.0, 1.0),
        (BinaryOp::Min, 2.0, 2.0, 2.0),
        (BinaryOp::Pow, 0.5, -2.0, 4.0),
        (BinaryOp::Pow, 0.5, -0.5, 1.4142135623730951),
        (BinaryOp::Pow, 0.5, 0.5, 0.7071067811865476),
        (BinaryOp::Pow, 0.5, 1.0, 0.5),
        (BinaryOp::Pow, 0.5, 2.0, 0.25),
        (BinaryOp::Pow, 0.5, 3.0, 0.125),
        (BinaryOp::Pow, 1.0, -2.0, 1.0),
        (BinaryOp::Pow, 1.0, -0.5, 1.0),
        (BinaryOp::Pow, 1.0, 0.5, 1.0),
        (BinaryOp::Pow, 1.0, 1.0, 1.0),
        (BinaryOp::Pow, 1.0, 2.0, 1.0),
        (BinaryOp::Pow, 1.0, 3.0, 1.0),
        (BinaryOp::Pow, 2.5, -2.0, 0.16),
        (BinaryOp::Pow, 2.5, -0.5, 0.6324555320336759),
        (BinaryOp::Pow, 2.5, 0.5, 1.5811388300841898),
        (BinaryOp::Pow, 2.5, 1.0, 2.5),
        (BinaryOp::Pow, 2.5, 2.0, 6.25),
        (BinaryOp::Pow, 2.5, 3.0, 15.625),
        (BinaryOp::Pow, 4.0, -2.0, 0.0625),
        (BinaryOp::Pow, 4.0, -0.5, 0.5),
        (BinaryOp::Pow, 4.0, 0.5, 2.0),
        (BinaryOp::Pow, 4.0, 1.0, 4.0),
        (BinaryOp::Pow, 4.0, 2.0, 16.0),
        (BinaryOp::Pow, 4.0, 3.0, 64.0),
        (BinaryOp::Rem, -3.5, -2.0, -1.5),
        (BinaryOp::Rem, -3.5, -0.5, -0.0),
        (BinaryOp::Rem, -3.5, 0.5, 0.0),
        (BinaryOp::Rem, -3.5, 1.0, 0.5),
        (BinaryOp::Rem, -3.5, 2.0, 0.5),
        (BinaryOp::Rem, -3.5, 3.5, 0.0),
        (BinaryOp::Rem, -1.5, -2.0, -1.5),
        (BinaryOp::Rem, -1.5, -0.5, -0.0),
        (BinaryOp::Rem, -1.5, 0.5, 0.0),
        (BinaryOp::Rem, -1.5, 1.0, 0.5),
        (BinaryOp::Rem, -1.5, 2.0, 0.5),
        (BinaryOp::Rem, -1.5, 3.5, 2.0),
        (BinaryOp::Rem, -0.5, -2.0, -0.5),
        (BinaryOp::Rem, -0.5, -0.5, -0.0),
        (BinaryOp::Rem, -0.5, 0.5, 0.0),
        (BinaryOp::Rem, -0.5, 1.0, 0.5),
        (BinaryOp::Rem, -0.5, 2.0, 1.5),
        (BinaryOp::Rem, -0.5, 3.5, 3.0),
        (BinaryOp::Rem, 0.5, -2.0, -1.5),
        (BinaryOp::Rem, 0.5, -0.5, -0.0),
        (BinaryOp::Rem, 0.5, 0.5, 0.0),
        (BinaryOp::Rem, 0.5, 1.0, 0.5),
        (BinaryOp::Rem, 0.5, 2.0, 0.5),
        (BinaryOp::Rem, 0.5, 3.5, 0.5),
        (BinaryOp::Rem, 1.0, -2.0, -1.0),
        (BinaryOp::Rem, 1.0, -0.5, -0.0),
        (BinaryOp::Rem, 1.0, 0.5, 0.0),
        (BinaryOp::Rem, 1.0, 1.0, 0.0),
        (BinaryOp::Rem, 1.0, 2.0, 1.0),
        (BinaryOp::Rem, 1.0, 3.5, 1.0),
        (BinaryOp::Rem, 2.5, -2.0, -1.5),
        (BinaryOp::Rem, 2.5, -0.5, -0.0),
        (BinaryOp::Rem, 2.5, 0.5, 0.0),
        (BinaryOp::Rem, 2.5, 1.0, 0.5),
        (BinaryOp::Rem, 2.5, 2.0, 0.5),
        (BinaryOp::Rem, 2.5, 3.5, 2.5),
        (BinaryOp::RemTrunc, -3.5, -2.0, -1.5),
        (BinaryOp::RemTrunc, -3.5, -0.5, -0.0),
        (BinaryOp::RemTrunc, -3.5, 0.5, -0.0),
        (BinaryOp::RemTrunc, -3.5, 1.0, -0.5),
        (BinaryOp::RemTrunc, -3.5, 2.0, -1.5),
        (BinaryOp::RemTrunc, -3.5, 3.5, -0.0),
        (BinaryOp::RemTrunc, -1.5, -2.0, -1.5),
        (BinaryOp::RemTrunc, -1.5, -0.5, -0.0),
        (BinaryOp::RemTrunc, -1.5, 0.5, -0.0),
        (BinaryOp::RemTrunc, -1.5, 1.0, -0.5),
        (BinaryOp::RemTrunc, -1.5, 2.0, -1.5),
        (BinaryOp::RemTrunc, -1.5, 3.5, -1.5),
        (BinaryOp::RemTrunc, -0.5, -2.0, -0.5),
        (BinaryOp::RemTrunc, -0.5, -0.5, -0.0),
        (BinaryOp::RemTrunc, -0.5, 0.5, -0.0),
        (BinaryOp::RemTrunc, -0.5, 1.0, -0.5),
        (BinaryOp::RemTrunc, -0.5, 2.0, -0.5),
        (BinaryOp::RemTrunc, -0.5, 3.5, -0.5),
        (BinaryOp::RemTrunc, 0.5, -2.0, 0.5),
        (BinaryOp::RemTrunc, 0.5, -0.5, 0.0),
        (BinaryOp::RemTrunc, 0.5, 0.5, 0.0),
        (BinaryOp::RemTrunc, 0.5, 1.0, 0.5),
        (BinaryOp::RemTrunc, 0.5, 2.0, 0.5),
        (BinaryOp::RemTrunc, 0.5, 3.5, 0.5),
        (BinaryOp::RemTrunc, 1.0, -2.0, 1.0),
        (BinaryOp::RemTrunc, 1.0, -0.5, 0.0),
        (BinaryOp::RemTrunc, 1.0, 0.5, 0.0),
        (BinaryOp::RemTrunc, 1.0, 1.0, 0.0),
        (BinaryOp::RemTrunc, 1.0, 2.0, 1.0),
        (BinaryOp::RemTrunc, 1.0, 3.5, 1.0),
        (BinaryOp::RemTrunc, 2.5, -2.0, 0.5),
        (BinaryOp::RemTrunc, 2.5, -0.5, 0.0),
        (BinaryOp::RemTrunc, 2.5, 0.5, 0.0),
        (BinaryOp::RemTrunc, 2.5, 1.0, 0.5),
        (BinaryOp::RemTrunc, 2.5, 2.0, 0.5),
        (BinaryOp::RemTrunc, 2.5, 3.5, 2.5),
        (BinaryOp::Atan2, -3.5, -2.0, -2.0899424410414196),
        (BinaryOp::Atan2, -3.5, -0.5, -1.7126933813990606),
        (BinaryOp::Atan2, -3.5, 0.5, -1.4288992721907328),
        (BinaryOp::Atan2, -3.5, 1.0, -1.2924966677897853),
        (BinaryOp::Atan2, -3.5, 2.0, -1.0516502125483738),
        (BinaryOp::Atan2, -3.5, 3.5, -0.7853981633974483),
        (BinaryOp::Atan2, -1.5, -2.0, -2.498091544796509),
        (BinaryOp::Atan2, -1.5, -0.5, -1.892546881191539),
        (BinaryOp::Atan2, -1.5, 0.5, -1.2490457723982544),
        (BinaryOp::Atan2, -1.5, 1.0, -0.982793723247329),
        (BinaryOp::Atan2, -1.5, 2.0, -0.6435011087932844),
        (BinaryOp::Atan2, -1.5, 3.5, -0.40489178628508343),
        (BinaryOp::Atan2, -0.5, -2.0, -2.896613990462929),
        (BinaryOp::Atan2, -0.5, -0.5, -2.356194490192345),
        (BinaryOp::Atan2, -0.5, 0.5, -0.7853981633974483),
        (BinaryOp::Atan2, -0.5, 1.0, -0.4636476090008061),
        (BinaryOp::Atan2, -0.5, 2.0, -0.24497866312686414),
        (BinaryOp::Atan2, -0.5, 3.5, -0.14189705460416394),
        (BinaryOp::Atan2, 0.5, -2.0, 2.896613990462929),
        (BinaryOp::Atan2, 0.5, -0.5, 2.356194490192345),
        (BinaryOp::Atan2, 0.5, 0.5, 0.7853981633974483),
        (BinaryOp::Atan2, 0.5, 1.0, 0.4636476090008061),
        (BinaryOp::Atan2, 0.5, 2.0, 0.24497866312686414),
        (BinaryOp::Atan2, 0.5, 3.5, 0.14189705460416394),
        (BinaryOp::Atan2, 1.0, -2.0, 2.677945044588987),
        (BinaryOp::Atan2, 1.0, -0.5, 2.0344439357957027),
        (BinaryOp::Atan2, 1.0, 0.5, 1.1071487177940904),
        (BinaryOp::Atan2, 1.0, 1.0, 0.7853981633974483),
        (BinaryOp::Atan2, 1.0, 2.0, 0.4636476090008061),
        (BinaryOp::Atan2, 1.0, 3.5, 0.27829965900511133),
        (BinaryOp::Atan2, 2.5, -2.0, 2.2455372690184494),
        (BinaryOp::Atan2, 2.5, -0.5, 1.7681918866447774),
        (BinaryOp::Atan2, 2.5, 0.5, 1.373400766945016),
        (BinaryOp::Atan2, 2.5, 1.0, 1.1902899496825317),
        (BinaryOp::Atan2, 2.5, 2.0, 0.8960553845713439),
        (BinaryOp::Atan2, 2.5, 3.5, 0.6202494859828215),
        (BinaryOp::Copysign, -3.5, -2.0, -3.5),
        (BinaryOp::Copysign, -3.5, -0.5, -3.5),
        (BinaryOp::Copysign, -3.5, 0.5, 3.5),
        (BinaryOp::Copysign, -3.5, 1.0, 3.5),
        (BinaryOp::Copysign, -3.5, 2.0, 3.5),
        (BinaryOp::Copysign, -3.5, 3.5, 3.5),
        (BinaryOp::Copysign, -1.5, -2.0, -1.5),
        (BinaryOp::Copysign, -1.5, -0.5, -1.5),
        (BinaryOp::Copysign, -1.5, 0.5, 1.5),
        (BinaryOp::Copysign, -1.5, 1.0, 1.5),
        (BinaryOp::Copysign, -1.5, 2.0, 1.5),
        (BinaryOp::Copysign, -1.5, 3.5, 1.5),
        (BinaryOp::Copysign, -0.5, -2.0, -0.5),
        (BinaryOp::Copysign, -0.5, -0.5, -0.5),
        (BinaryOp::Copysign, -0.5, 0.5, 0.5),
        (BinaryOp::Copysign, -0.5, 1.0, 0.5),
        (BinaryOp::Copysign, -0.5, 2.0, 0.5),
        (BinaryOp::Copysign, -0.5, 3.5, 0.5),
        (BinaryOp::Copysign, 0.5, -2.0, -0.5),
        (BinaryOp::Copysign, 0.5, -0.5, -0.5),
        (BinaryOp::Copysign, 0.5, 0.5, 0.5),
        (BinaryOp::Copysign, 0.5, 1.0, 0.5),
        (BinaryOp::Copysign, 0.5, 2.0, 0.5),
        (BinaryOp::Copysign, 0.5, 3.5, 0.5),
        (BinaryOp::Copysign, 1.0, -2.0, -1.0),
        (BinaryOp::Copysign, 1.0, -0.5, -1.0),
        (BinaryOp::Copysign, 1.0, 0.5, 1.0),
        (BinaryOp::Copysign, 1.0, 1.0, 1.0),
        (BinaryOp::Copysign, 1.0, 2.0, 1.0),
        (BinaryOp::Copysign, 1.0, 3.5, 1.0),
        (BinaryOp::Copysign, 2.5, -2.0, -2.5),
        (BinaryOp::Copysign, 2.5, -0.5, -2.5),
        (BinaryOp::Copysign, 2.5, 0.5, 2.5),
        (BinaryOp::Copysign, 2.5, 1.0, 2.5),
        (BinaryOp::Copysign, 2.5, 2.0, 2.5),
        (BinaryOp::Copysign, 2.5, 3.5, 2.5),
        (BinaryOp::FmaxIeee, -3.5, -2.0, -2.0),
        (BinaryOp::FmaxIeee, -3.5, -0.5, -0.5),
        (BinaryOp::FmaxIeee, -3.5, 0.5, 0.5),
        (BinaryOp::FmaxIeee, -3.5, 1.0, 1.0),
        (BinaryOp::FmaxIeee, -3.5, 2.0, 2.0),
        (BinaryOp::FmaxIeee, -3.5, 3.5, 3.5),
        (BinaryOp::FmaxIeee, -1.5, -2.0, -1.5),
        (BinaryOp::FmaxIeee, -1.5, -0.5, -0.5),
        (BinaryOp::FmaxIeee, -1.5, 0.5, 0.5),
        (BinaryOp::FmaxIeee, -1.5, 1.0, 1.0),
        (BinaryOp::FmaxIeee, -1.5, 2.0, 2.0),
        (BinaryOp::FmaxIeee, -1.5, 3.5, 3.5),
        (BinaryOp::FmaxIeee, -0.5, -2.0, -0.5),
        (BinaryOp::FmaxIeee, -0.5, -0.5, -0.5),
        (BinaryOp::FmaxIeee, -0.5, 0.5, 0.5),
        (BinaryOp::FmaxIeee, -0.5, 1.0, 1.0),
        (BinaryOp::FmaxIeee, -0.5, 2.0, 2.0),
        (BinaryOp::FmaxIeee, -0.5, 3.5, 3.5),
        (BinaryOp::FmaxIeee, 0.5, -2.0, 0.5),
        (BinaryOp::FmaxIeee, 0.5, -0.5, 0.5),
        (BinaryOp::FmaxIeee, 0.5, 0.5, 0.5),
        (BinaryOp::FmaxIeee, 0.5, 1.0, 1.0),
        (BinaryOp::FmaxIeee, 0.5, 2.0, 2.0),
        (BinaryOp::FmaxIeee, 0.5, 3.5, 3.5),
        (BinaryOp::FmaxIeee, 1.0, -2.0, 1.0),
        (BinaryOp::FmaxIeee, 1.0, -0.5, 1.0),
        (BinaryOp::FmaxIeee, 1.0, 0.5, 1.0),
        (BinaryOp::FmaxIeee, 1.0, 1.0, 1.0),
        (BinaryOp::FmaxIeee, 1.0, 2.0, 2.0),
        (BinaryOp::FmaxIeee, 1.0, 3.5, 3.5),
        (BinaryOp::FmaxIeee, 2.5, -2.0, 2.5),
        (BinaryOp::FmaxIeee, 2.5, -0.5, 2.5),
        (BinaryOp::FmaxIeee, 2.5, 0.5, 2.5),
        (BinaryOp::FmaxIeee, 2.5, 1.0, 2.5),
        (BinaryOp::FmaxIeee, 2.5, 2.0, 2.5),
        (BinaryOp::FmaxIeee, 2.5, 3.5, 3.5),
        (BinaryOp::FminIeee, -3.5, -2.0, -3.5),
        (BinaryOp::FminIeee, -3.5, -0.5, -3.5),
        (BinaryOp::FminIeee, -3.5, 0.5, -3.5),
        (BinaryOp::FminIeee, -3.5, 1.0, -3.5),
        (BinaryOp::FminIeee, -3.5, 2.0, -3.5),
        (BinaryOp::FminIeee, -3.5, 3.5, -3.5),
        (BinaryOp::FminIeee, -1.5, -2.0, -2.0),
        (BinaryOp::FminIeee, -1.5, -0.5, -1.5),
        (BinaryOp::FminIeee, -1.5, 0.5, -1.5),
        (BinaryOp::FminIeee, -1.5, 1.0, -1.5),
        (BinaryOp::FminIeee, -1.5, 2.0, -1.5),
        (BinaryOp::FminIeee, -1.5, 3.5, -1.5),
        (BinaryOp::FminIeee, -0.5, -2.0, -2.0),
        (BinaryOp::FminIeee, -0.5, -0.5, -0.5),
        (BinaryOp::FminIeee, -0.5, 0.5, -0.5),
        (BinaryOp::FminIeee, -0.5, 1.0, -0.5),
        (BinaryOp::FminIeee, -0.5, 2.0, -0.5),
        (BinaryOp::FminIeee, -0.5, 3.5, -0.5),
        (BinaryOp::FminIeee, 0.5, -2.0, -2.0),
        (BinaryOp::FminIeee, 0.5, -0.5, -0.5),
        (BinaryOp::FminIeee, 0.5, 0.5, 0.5),
        (BinaryOp::FminIeee, 0.5, 1.0, 0.5),
        (BinaryOp::FminIeee, 0.5, 2.0, 0.5),
        (BinaryOp::FminIeee, 0.5, 3.5, 0.5),
        (BinaryOp::FminIeee, 1.0, -2.0, -2.0),
        (BinaryOp::FminIeee, 1.0, -0.5, -0.5),
        (BinaryOp::FminIeee, 1.0, 0.5, 0.5),
        (BinaryOp::FminIeee, 1.0, 1.0, 1.0),
        (BinaryOp::FminIeee, 1.0, 2.0, 1.0),
        (BinaryOp::FminIeee, 1.0, 3.5, 1.0),
        (BinaryOp::FminIeee, 2.5, -2.0, -2.0),
        (BinaryOp::FminIeee, 2.5, -0.5, -0.5),
        (BinaryOp::FminIeee, 2.5, 0.5, 0.5),
        (BinaryOp::FminIeee, 2.5, 1.0, 1.0),
        (BinaryOp::FminIeee, 2.5, 2.0, 2.0),
        (BinaryOp::FminIeee, 2.5, 3.5, 2.5),
        (BinaryOp::CmpEq, -3.5, -2.0, 0.0),
        (BinaryOp::CmpEq, -3.5, -0.5, 0.0),
        (BinaryOp::CmpEq, -3.5, 0.5, 0.0),
        (BinaryOp::CmpEq, -3.5, 1.0, 0.0),
        (BinaryOp::CmpEq, -3.5, 2.0, 0.0),
        (BinaryOp::CmpEq, -3.5, 3.5, 0.0),
        (BinaryOp::CmpEq, -1.5, -2.0, 0.0),
        (BinaryOp::CmpEq, -1.5, -0.5, 0.0),
        (BinaryOp::CmpEq, -1.5, 0.5, 0.0),
        (BinaryOp::CmpEq, -1.5, 1.0, 0.0),
        (BinaryOp::CmpEq, -1.5, 2.0, 0.0),
        (BinaryOp::CmpEq, -1.5, 3.5, 0.0),
        (BinaryOp::CmpEq, -0.5, -2.0, 0.0),
        (BinaryOp::CmpEq, -0.5, -0.5, 1.0),
        (BinaryOp::CmpEq, -0.5, 0.5, 0.0),
        (BinaryOp::CmpEq, -0.5, 1.0, 0.0),
        (BinaryOp::CmpEq, -0.5, 2.0, 0.0),
        (BinaryOp::CmpEq, -0.5, 3.5, 0.0),
        (BinaryOp::CmpEq, 0.5, -2.0, 0.0),
        (BinaryOp::CmpEq, 0.5, -0.5, 0.0),
        (BinaryOp::CmpEq, 0.5, 0.5, 1.0),
        (BinaryOp::CmpEq, 0.5, 1.0, 0.0),
        (BinaryOp::CmpEq, 0.5, 2.0, 0.0),
        (BinaryOp::CmpEq, 0.5, 3.5, 0.0),
        (BinaryOp::CmpEq, 1.0, -2.0, 0.0),
        (BinaryOp::CmpEq, 1.0, -0.5, 0.0),
        (BinaryOp::CmpEq, 1.0, 0.5, 0.0),
        (BinaryOp::CmpEq, 1.0, 1.0, 1.0),
        (BinaryOp::CmpEq, 1.0, 2.0, 0.0),
        (BinaryOp::CmpEq, 1.0, 3.5, 0.0),
        (BinaryOp::CmpEq, 2.5, -2.0, 0.0),
        (BinaryOp::CmpEq, 2.5, -0.5, 0.0),
        (BinaryOp::CmpEq, 2.5, 0.5, 0.0),
        (BinaryOp::CmpEq, 2.5, 1.0, 0.0),
        (BinaryOp::CmpEq, 2.5, 2.0, 0.0),
        (BinaryOp::CmpEq, 2.5, 3.5, 0.0),
        (BinaryOp::CmpEq, 1.0, 1.0, 1.0),
        (BinaryOp::CmpEq, 2.0, 2.0, 1.0),
        (BinaryOp::CmpNe, -3.5, -2.0, 1.0),
        (BinaryOp::CmpNe, -3.5, -0.5, 1.0),
        (BinaryOp::CmpNe, -3.5, 0.5, 1.0),
        (BinaryOp::CmpNe, -3.5, 1.0, 1.0),
        (BinaryOp::CmpNe, -3.5, 2.0, 1.0),
        (BinaryOp::CmpNe, -3.5, 3.5, 1.0),
        (BinaryOp::CmpNe, -1.5, -2.0, 1.0),
        (BinaryOp::CmpNe, -1.5, -0.5, 1.0),
        (BinaryOp::CmpNe, -1.5, 0.5, 1.0),
        (BinaryOp::CmpNe, -1.5, 1.0, 1.0),
        (BinaryOp::CmpNe, -1.5, 2.0, 1.0),
        (BinaryOp::CmpNe, -1.5, 3.5, 1.0),
        (BinaryOp::CmpNe, -0.5, -2.0, 1.0),
        (BinaryOp::CmpNe, -0.5, -0.5, 0.0),
        (BinaryOp::CmpNe, -0.5, 0.5, 1.0),
        (BinaryOp::CmpNe, -0.5, 1.0, 1.0),
        (BinaryOp::CmpNe, -0.5, 2.0, 1.0),
        (BinaryOp::CmpNe, -0.5, 3.5, 1.0),
        (BinaryOp::CmpNe, 0.5, -2.0, 1.0),
        (BinaryOp::CmpNe, 0.5, -0.5, 1.0),
        (BinaryOp::CmpNe, 0.5, 0.5, 0.0),
        (BinaryOp::CmpNe, 0.5, 1.0, 1.0),
        (BinaryOp::CmpNe, 0.5, 2.0, 1.0),
        (BinaryOp::CmpNe, 0.5, 3.5, 1.0),
        (BinaryOp::CmpNe, 1.0, -2.0, 1.0),
        (BinaryOp::CmpNe, 1.0, -0.5, 1.0),
        (BinaryOp::CmpNe, 1.0, 0.5, 1.0),
        (BinaryOp::CmpNe, 1.0, 1.0, 0.0),
        (BinaryOp::CmpNe, 1.0, 2.0, 1.0),
        (BinaryOp::CmpNe, 1.0, 3.5, 1.0),
        (BinaryOp::CmpNe, 2.5, -2.0, 1.0),
        (BinaryOp::CmpNe, 2.5, -0.5, 1.0),
        (BinaryOp::CmpNe, 2.5, 0.5, 1.0),
        (BinaryOp::CmpNe, 2.5, 1.0, 1.0),
        (BinaryOp::CmpNe, 2.5, 2.0, 1.0),
        (BinaryOp::CmpNe, 2.5, 3.5, 1.0),
        (BinaryOp::CmpNe, 1.0, 1.0, 0.0),
        (BinaryOp::CmpNe, 2.0, 2.0, 0.0),
        (BinaryOp::CmpLt, -3.5, -2.0, 1.0),
        (BinaryOp::CmpLt, -3.5, -0.5, 1.0),
        (BinaryOp::CmpLt, -3.5, 0.5, 1.0),
        (BinaryOp::CmpLt, -3.5, 1.0, 1.0),
        (BinaryOp::CmpLt, -3.5, 2.0, 1.0),
        (BinaryOp::CmpLt, -3.5, 3.5, 1.0),
        (BinaryOp::CmpLt, -1.5, -2.0, 0.0),
        (BinaryOp::CmpLt, -1.5, -0.5, 1.0),
        (BinaryOp::CmpLt, -1.5, 0.5, 1.0),
        (BinaryOp::CmpLt, -1.5, 1.0, 1.0),
        (BinaryOp::CmpLt, -1.5, 2.0, 1.0),
        (BinaryOp::CmpLt, -1.5, 3.5, 1.0),
        (BinaryOp::CmpLt, -0.5, -2.0, 0.0),
        (BinaryOp::CmpLt, -0.5, -0.5, 0.0),
        (BinaryOp::CmpLt, -0.5, 0.5, 1.0),
        (BinaryOp::CmpLt, -0.5, 1.0, 1.0),
        (BinaryOp::CmpLt, -0.5, 2.0, 1.0),
        (BinaryOp::CmpLt, -0.5, 3.5, 1.0),
        (BinaryOp::CmpLt, 0.5, -2.0, 0.0),
        (BinaryOp::CmpLt, 0.5, -0.5, 0.0),
        (BinaryOp::CmpLt, 0.5, 0.5, 0.0),
        (BinaryOp::CmpLt, 0.5, 1.0, 1.0),
        (BinaryOp::CmpLt, 0.5, 2.0, 1.0),
        (BinaryOp::CmpLt, 0.5, 3.5, 1.0),
        (BinaryOp::CmpLt, 1.0, -2.0, 0.0),
        (BinaryOp::CmpLt, 1.0, -0.5, 0.0),
        (BinaryOp::CmpLt, 1.0, 0.5, 0.0),
        (BinaryOp::CmpLt, 1.0, 1.0, 0.0),
        (BinaryOp::CmpLt, 1.0, 2.0, 1.0),
        (BinaryOp::CmpLt, 1.0, 3.5, 1.0),
        (BinaryOp::CmpLt, 2.5, -2.0, 0.0),
        (BinaryOp::CmpLt, 2.5, -0.5, 0.0),
        (BinaryOp::CmpLt, 2.5, 0.5, 0.0),
        (BinaryOp::CmpLt, 2.5, 1.0, 0.0),
        (BinaryOp::CmpLt, 2.5, 2.0, 0.0),
        (BinaryOp::CmpLt, 2.5, 3.5, 1.0),
        (BinaryOp::CmpLt, 1.0, 1.0, 0.0),
        (BinaryOp::CmpLt, 2.0, 2.0, 0.0),
        (BinaryOp::CmpLe, -3.5, -2.0, 1.0),
        (BinaryOp::CmpLe, -3.5, -0.5, 1.0),
        (BinaryOp::CmpLe, -3.5, 0.5, 1.0),
        (BinaryOp::CmpLe, -3.5, 1.0, 1.0),
        (BinaryOp::CmpLe, -3.5, 2.0, 1.0),
        (BinaryOp::CmpLe, -3.5, 3.5, 1.0),
        (BinaryOp::CmpLe, -1.5, -2.0, 0.0),
        (BinaryOp::CmpLe, -1.5, -0.5, 1.0),
        (BinaryOp::CmpLe, -1.5, 0.5, 1.0),
        (BinaryOp::CmpLe, -1.5, 1.0, 1.0),
        (BinaryOp::CmpLe, -1.5, 2.0, 1.0),
        (BinaryOp::CmpLe, -1.5, 3.5, 1.0),
        (BinaryOp::CmpLe, -0.5, -2.0, 0.0),
        (BinaryOp::CmpLe, -0.5, -0.5, 1.0),
        (BinaryOp::CmpLe, -0.5, 0.5, 1.0),
        (BinaryOp::CmpLe, -0.5, 1.0, 1.0),
        (BinaryOp::CmpLe, -0.5, 2.0, 1.0),
        (BinaryOp::CmpLe, -0.5, 3.5, 1.0),
        (BinaryOp::CmpLe, 0.5, -2.0, 0.0),
        (BinaryOp::CmpLe, 0.5, -0.5, 0.0),
        (BinaryOp::CmpLe, 0.5, 0.5, 1.0),
        (BinaryOp::CmpLe, 0.5, 1.0, 1.0),
        (BinaryOp::CmpLe, 0.5, 2.0, 1.0),
        (BinaryOp::CmpLe, 0.5, 3.5, 1.0),
        (BinaryOp::CmpLe, 1.0, -2.0, 0.0),
        (BinaryOp::CmpLe, 1.0, -0.5, 0.0),
        (BinaryOp::CmpLe, 1.0, 0.5, 0.0),
        (BinaryOp::CmpLe, 1.0, 1.0, 1.0),
        (BinaryOp::CmpLe, 1.0, 2.0, 1.0),
        (BinaryOp::CmpLe, 1.0, 3.5, 1.0),
        (BinaryOp::CmpLe, 2.5, -2.0, 0.0),
        (BinaryOp::CmpLe, 2.5, -0.5, 0.0),
        (BinaryOp::CmpLe, 2.5, 0.5, 0.0),
        (BinaryOp::CmpLe, 2.5, 1.0, 0.0),
        (BinaryOp::CmpLe, 2.5, 2.0, 0.0),
        (BinaryOp::CmpLe, 2.5, 3.5, 1.0),
        (BinaryOp::CmpLe, 1.0, 1.0, 1.0),
        (BinaryOp::CmpLe, 2.0, 2.0, 1.0),
        (BinaryOp::CmpGt, -3.5, -2.0, 0.0),
        (BinaryOp::CmpGt, -3.5, -0.5, 0.0),
        (BinaryOp::CmpGt, -3.5, 0.5, 0.0),
        (BinaryOp::CmpGt, -3.5, 1.0, 0.0),
        (BinaryOp::CmpGt, -3.5, 2.0, 0.0),
        (BinaryOp::CmpGt, -3.5, 3.5, 0.0),
        (BinaryOp::CmpGt, -1.5, -2.0, 1.0),
        (BinaryOp::CmpGt, -1.5, -0.5, 0.0),
        (BinaryOp::CmpGt, -1.5, 0.5, 0.0),
        (BinaryOp::CmpGt, -1.5, 1.0, 0.0),
        (BinaryOp::CmpGt, -1.5, 2.0, 0.0),
        (BinaryOp::CmpGt, -1.5, 3.5, 0.0),
        (BinaryOp::CmpGt, -0.5, -2.0, 1.0),
        (BinaryOp::CmpGt, -0.5, -0.5, 0.0),
        (BinaryOp::CmpGt, -0.5, 0.5, 0.0),
        (BinaryOp::CmpGt, -0.5, 1.0, 0.0),
        (BinaryOp::CmpGt, -0.5, 2.0, 0.0),
        (BinaryOp::CmpGt, -0.5, 3.5, 0.0),
        (BinaryOp::CmpGt, 0.5, -2.0, 1.0),
        (BinaryOp::CmpGt, 0.5, -0.5, 1.0),
        (BinaryOp::CmpGt, 0.5, 0.5, 0.0),
        (BinaryOp::CmpGt, 0.5, 1.0, 0.0),
        (BinaryOp::CmpGt, 0.5, 2.0, 0.0),
        (BinaryOp::CmpGt, 0.5, 3.5, 0.0),
        (BinaryOp::CmpGt, 1.0, -2.0, 1.0),
        (BinaryOp::CmpGt, 1.0, -0.5, 1.0),
        (BinaryOp::CmpGt, 1.0, 0.5, 1.0),
        (BinaryOp::CmpGt, 1.0, 1.0, 0.0),
        (BinaryOp::CmpGt, 1.0, 2.0, 0.0),
        (BinaryOp::CmpGt, 1.0, 3.5, 0.0),
        (BinaryOp::CmpGt, 2.5, -2.0, 1.0),
        (BinaryOp::CmpGt, 2.5, -0.5, 1.0),
        (BinaryOp::CmpGt, 2.5, 0.5, 1.0),
        (BinaryOp::CmpGt, 2.5, 1.0, 1.0),
        (BinaryOp::CmpGt, 2.5, 2.0, 1.0),
        (BinaryOp::CmpGt, 2.5, 3.5, 0.0),
        (BinaryOp::CmpGt, 1.0, 1.0, 0.0),
        (BinaryOp::CmpGt, 2.0, 2.0, 0.0),
        (BinaryOp::CmpGe, -3.5, -2.0, 0.0),
        (BinaryOp::CmpGe, -3.5, -0.5, 0.0),
        (BinaryOp::CmpGe, -3.5, 0.5, 0.0),
        (BinaryOp::CmpGe, -3.5, 1.0, 0.0),
        (BinaryOp::CmpGe, -3.5, 2.0, 0.0),
        (BinaryOp::CmpGe, -3.5, 3.5, 0.0),
        (BinaryOp::CmpGe, -1.5, -2.0, 1.0),
        (BinaryOp::CmpGe, -1.5, -0.5, 0.0),
        (BinaryOp::CmpGe, -1.5, 0.5, 0.0),
        (BinaryOp::CmpGe, -1.5, 1.0, 0.0),
        (BinaryOp::CmpGe, -1.5, 2.0, 0.0),
        (BinaryOp::CmpGe, -1.5, 3.5, 0.0),
        (BinaryOp::CmpGe, -0.5, -2.0, 1.0),
        (BinaryOp::CmpGe, -0.5, -0.5, 1.0),
        (BinaryOp::CmpGe, -0.5, 0.5, 0.0),
        (BinaryOp::CmpGe, -0.5, 1.0, 0.0),
        (BinaryOp::CmpGe, -0.5, 2.0, 0.0),
        (BinaryOp::CmpGe, -0.5, 3.5, 0.0),
        (BinaryOp::CmpGe, 0.5, -2.0, 1.0),
        (BinaryOp::CmpGe, 0.5, -0.5, 1.0),
        (BinaryOp::CmpGe, 0.5, 0.5, 1.0),
        (BinaryOp::CmpGe, 0.5, 1.0, 0.0),
        (BinaryOp::CmpGe, 0.5, 2.0, 0.0),
        (BinaryOp::CmpGe, 0.5, 3.5, 0.0),
        (BinaryOp::CmpGe, 1.0, -2.0, 1.0),
        (BinaryOp::CmpGe, 1.0, -0.5, 1.0),
        (BinaryOp::CmpGe, 1.0, 0.5, 1.0),
        (BinaryOp::CmpGe, 1.0, 1.0, 1.0),
        (BinaryOp::CmpGe, 1.0, 2.0, 0.0),
        (BinaryOp::CmpGe, 1.0, 3.5, 0.0),
        (BinaryOp::CmpGe, 2.5, -2.0, 1.0),
        (BinaryOp::CmpGe, 2.5, -0.5, 1.0),
        (BinaryOp::CmpGe, 2.5, 0.5, 1.0),
        (BinaryOp::CmpGe, 2.5, 1.0, 1.0),
        (BinaryOp::CmpGe, 2.5, 2.0, 1.0),
        (BinaryOp::CmpGe, 2.5, 3.5, 0.0),
        (BinaryOp::CmpGe, 1.0, 1.0, 1.0),
        (BinaryOp::CmpGe, 2.0, 2.0, 1.0),
    ];

    #[test]
    fn every_unary_op_matches_the_reference_values() {
        for &(op, x, want) in UNARY {
            let tol = if op == UnaryOp::Lgamma { 1e-10 } else { 1e-12 };
            let got = unary_op_f64(op, x);
            assert!(close(got, want, tol), "{op:?}({x}) = {got}, want {want}");
        }
    }

    #[test]
    fn every_binary_op_matches_the_reference_values() {
        for &(op, a, b, want) in BINARY {
            let got = binary_op_f64(op, a, b);
            assert!(
                close(got, want, 1e-12),
                "{op:?}({a}, {b}) = {got}, want {want}"
            );
        }
    }

    #[test]
    fn unary_special_values_keep_their_semantics() {
        // Relu keeps -0.0 and propagates NaN; Sign and Step send NaN to 0.
        assert_eq!(
            unary_op_f64(UnaryOp::Relu, -0.0).to_bits(),
            (-0.0f64).to_bits()
        );
        assert!(unary_op_f64(UnaryOp::Relu, f64::NAN).is_nan());
        assert_eq!(unary_op_f64(UnaryOp::Sign, f64::NAN), 0.0);
        assert_eq!(unary_op_f64(UnaryOp::Sign, 0.0), 0.0);
        assert_eq!(unary_op_f64(UnaryOp::Step, 0.0), 0.0);
        assert_eq!(unary_op_f64(UnaryOp::Step, f64::NAN), 0.0);
        assert_eq!(unary_op_f64(UnaryOp::Round, 2.5), 2.0);
        assert_eq!(unary_op_f64(UnaryOp::Round, 3.5), 4.0);
    }

    #[test]
    fn max_and_min_propagate_nan_and_keep_a_on_signed_zero_ties() {
        let neg = -0.0f64;
        let pos = 0.0f64;
        assert_eq!(
            binary_op_f64(BinaryOp::Max, neg, pos).to_bits(),
            neg.to_bits()
        );
        assert_eq!(
            binary_op_f64(BinaryOp::Max, pos, neg).to_bits(),
            pos.to_bits()
        );
        assert_eq!(
            binary_op_f64(BinaryOp::Min, neg, pos).to_bits(),
            neg.to_bits()
        );
        assert_eq!(
            binary_op_f64(BinaryOp::Min, pos, neg).to_bits(),
            pos.to_bits()
        );
        for op in [BinaryOp::Max, BinaryOp::Min] {
            assert!(binary_op_f64(op, f64::NAN, 1.0).is_nan(), "{op:?} a");
            assert!(binary_op_f64(op, 1.0, f64::NAN).is_nan(), "{op:?} b");
        }
        assert_eq!(binary_op_f64(BinaryOp::FmaxIeee, f64::NAN, 1.0), 1.0);
        assert_eq!(binary_op_f64(BinaryOp::FminIeee, 1.0, f64::NAN), 1.0);
        for op in [
            BinaryOp::CmpEq,
            BinaryOp::CmpLt,
            BinaryOp::CmpLe,
            BinaryOp::CmpGt,
            BinaryOp::CmpGe,
        ] {
            assert_eq!(binary_op_f64(op, f64::NAN, 1.0), 0.0, "{op:?}");
        }
        assert_eq!(binary_op_f64(BinaryOp::CmpNe, f64::NAN, 1.0), 1.0);
    }

    #[test]
    fn next_after_steps_one_ulp_toward_the_target() {
        let one = 1.0f64;
        let eps = f64::EPSILON;
        assert_eq!(next_after_f64(one, 2.0), one + eps);
        assert_eq!(next_after_f64(one, 0.0), one - eps / 2.0);
        assert_eq!(next_after_f64(-one, -2.0), -(one + eps));
        assert_eq!(next_after_f64(-one, 0.0), -(one - eps / 2.0));
        assert_eq!(next_after_f64(0.0, 1.0).to_bits(), 1);
        assert_eq!(next_after_f64(0.0, -1.0).to_bits(), 1 | (1u64 << 63));
        assert_eq!(next_after_f64(-0.0, 1.0).to_bits(), 1);
        assert_eq!(next_after_f64(1.0, 1.0), 1.0);
        assert_eq!(next_after_f64(0.0, -0.0).to_bits(), (-0.0f64).to_bits());
        assert!(next_after_f64(f64::NAN, 1.0).is_nan());
        assert!(next_after_f64(1.0, f64::NAN).is_nan());

        let one = 1.0f32;
        let eps = f32::EPSILON;
        assert_eq!(next_after_f32(one, 2.0), one + eps);
        assert_eq!(next_after_f32(one, 0.0), one - eps / 2.0);
        assert_eq!(next_after_f32(-one, -2.0), -(one + eps));
        assert_eq!(next_after_f32(-one, 0.0), -(one - eps / 2.0));
        assert_eq!(next_after_f32(0.0, 1.0).to_bits(), 1);
        assert_eq!(next_after_f32(0.0, -1.0).to_bits(), 1 | (1u32 << 31));
        assert_eq!(next_after_f32(1.0, 1.0), 1.0);
        assert!(next_after_f32(f32::NAN, 1.0).is_nan());
        assert!(next_after_f32(1.0, f32::NAN).is_nan());
    }

    #[test]
    fn integer_bitwise_and_shift_ops_wrap_at_the_promotion_width() {
        use BinaryOp::*;
        let i = ElementKind::I32;
        assert_eq!(binary_op_int(BitAnd, 0b1100, 0b1010, i), 0b1000);
        assert_eq!(binary_op_int(BitOr, 0b1100, 0b1010, i), 0b1110);
        assert_eq!(binary_op_int(BitXor, 0b1100, 0b1010, i), 0b0110);
        assert_eq!(binary_op_int(Shl, 1, 0, i), 1);
        assert_eq!(binary_op_int(Shl, 1, 31, i), i128::from(i32::MIN));
        assert_eq!(
            binary_op_int(Shl, 1, 32, i),
            1,
            "a count of the width reads as 0 (modulo the width)"
        );
        assert_eq!(
            binary_op_int(Shl, 1, -1, i),
            i128::from(i32::MIN),
            "-1 reads as 31"
        );
        assert_eq!(
            binary_op_int(Shr, -8, 1, i),
            -4,
            "arithmetic on a signed operand"
        );
        assert_eq!(binary_op_int(Shr, -8, 32, i), -8, "32 reads as 0");
        assert_eq!(binary_op_int(Shr, 8, 32, i), 8);
        assert_eq!(binary_op_int(Shr, -8, 33, i), -4, "33 reads as 1");
        assert_eq!(binary_op_int(Shr, 0, 32, i), 0);
        assert_eq!(binary_op_int(Shr, 0, -1, i), 0);
        assert_eq!(binary_op_int(Shr, -1, 31, i), -1);
        assert_eq!(binary_op_int(Shr, 8, 3, i), 1);
        let u = ElementKind::U32;
        assert_eq!(binary_op_int(Shl, 1, 31, u), 1 << 31, "zero-extended");
        assert_eq!(
            binary_op_int(Shr, 1 << 31, 31, u),
            1,
            "logical on an unsigned operand"
        );
        assert_eq!(binary_op_int(Shl, 1, 32, u), 1);
        let l = ElementKind::I64;
        assert_eq!(binary_op_int(Shl, 1, 32, l), 1 << 32);
        assert_eq!(binary_op_int(Shl, 1, 63, l), i128::from(i64::MIN));
        assert_eq!(binary_op_int(Shl, 1, 64, l), 1);
        assert_eq!(binary_op_int(Shl, 1, 63, ElementKind::U64), 1 << 63);
        // 8-bit dtypes promote to int: no wrap at 8 bits during the op.
        assert_eq!(binary_op_int(Shl, 200, 1, ElementKind::U8), 400);
        for (a, b) in [(0, 0), (0, 5), (5, 0), (5, 7)] {
            assert_eq!(
                binary_op_int(LogicalAnd, a, b, i),
                i128::from(a != 0 && b != 0)
            );
            assert_eq!(
                binary_op_int(LogicalOr, a, b, i),
                i128::from(a != 0 || b != 0)
            );
            assert_eq!(
                binary_op_int(LogicalXor, a, b, i),
                i128::from((a != 0) != (b != 0))
            );
        }
    }

    #[test]
    fn wrapping_extends_per_signedness() {
        assert_eq!(wrap_bits(255, 8), -1);
        assert_eq!(wrap_bits(127, 8), 127);
        assert_eq!(wrap_bits(128, 8), -128);
        assert_eq!(wrap_bits(i128::MAX, 128), i128::MAX);
        assert_eq!(wrap_bits(-5, 200), -5);
        assert_eq!(wrap_for(0xFFFF_FFFF, 32, ElementKind::I32), -1);
        assert_eq!(wrap_for(-1, 32, ElementKind::U32), 0xFFFF_FFFF);
        assert_eq!(wrap_for(-1, 64, ElementKind::U64), i128::from(u64::MAX));
        assert_eq!(wrap_for(-1, 128, ElementKind::U64), -1);
    }

    #[test]
    fn integer_extremes_are_the_dtype_limits() {
        use ElementKind::*;
        let table: &[(ElementKind, i128, i128)] = &[
            (I8, i128::from(i8::MIN), i128::from(i8::MAX)),
            (I16, i128::from(i16::MIN), i128::from(i16::MAX)),
            (I32, i128::from(i32::MIN), i128::from(i32::MAX)),
            (I64, i128::from(i64::MIN), i128::from(i64::MAX)),
            (U8, 0, 255),
            (Bool, 0, 255),
            (U16, 0, i128::from(u16::MAX)),
            (U32, 0, i128::from(u32::MAX)),
            (U64, 0, i128::from(u64::MAX)),
        ];
        for &(dt, lo, hi) in table {
            assert_eq!(int_extreme(dt, true), lo, "{dt:?} most negative");
            assert_eq!(int_extreme(dt, false), hi, "{dt:?} most positive");
        }
    }

    #[test]
    fn dense_strides_are_row_major() {
        assert_eq!(dense_strides(&[2, 3, 4]), vec![12, 4, 1]);
        assert_eq!(dense_strides(&[5]), vec![1]);
        assert_eq!(dense_strides(&[]), Vec::<i64>::new());
    }

    #[test]
    fn a_buffer_reports_emptiness_and_sub_byte_packing() {
        assert!(TypedBuffer::from_f64_as(ElementKind::F32, &[0], &[]).is_empty());
        assert!(!TypedBuffer::from_f64_as(ElementKind::F32, &[1], &[1.0]).is_empty());
        assert!(is_sub_byte(ElementKind::I4));
        assert!(is_sub_byte(ElementKind::B1));
        assert!(!is_sub_byte(ElementKind::U8));
        assert!(!is_sub_byte(ElementKind::F16));
    }

    #[test]
    fn elements_are_stored_at_their_own_offset() {
        // `i * sz` -> `i / sz` in `from_f64_as`: the second f32 lands at byte 4.
        let b = TypedBuffer::from_f64_as(ElementKind::F32, &[3], &[1.0, 2.0, 3.0]);
        let mut want = Vec::new();
        for v in [1.0f32, 2.0, 3.0] {
            want.extend_from_slice(&v.to_le_bytes());
        }
        assert_eq!(b.bytes, want);
    }

    #[test]
    fn sub_byte_writes_touch_only_their_own_bits() {
        // 4-bit: low nibble first, a write keeps the neighbour.
        let mut b = [0u8; 2];
        write_elem(&mut b, 0, ElementKind::U4, 0xF);
        write_elem(&mut b, 1, ElementKind::U4, 0x3);
        assert_eq!(b, [0x3F, 0]);
        write_elem(&mut b, 0, ElementKind::U4, 0x5);
        assert_eq!(b, [0x35, 0]);
        write_elem(&mut b, 1, ElementKind::U4, 0xA);
        assert_eq!(b, [0xA5, 0]);
        write_elem(&mut b, 2, ElementKind::U4, 0x7);
        write_elem(&mut b, 3, ElementKind::U4, 0x1);
        assert_eq!(b, [0xA5, 0x17]);
        // The value is masked to the nibble.
        let mut m = [0u8; 1];
        write_elem(&mut m, 0, ElementKind::U4, 0x1F);
        assert_eq!(m, [0x0F]);
        // 1-bit: LSB first, a clear leaves the others.
        let mut bits = [0u8; 2];
        for i in [0usize, 2, 3, 8, 9] {
            write_elem(&mut bits, i, ElementKind::B1, 1);
        }
        assert_eq!(bits, [0b0000_1101, 0b0000_0011]);
        write_elem(&mut bits, 2, ElementKind::B1, 0);
        assert_eq!(bits, [0b0000_1001, 0b0000_0011]);
        let mut hi = [0u8; 1];
        write_elem(&mut hi, 7, ElementKind::B1, 0b11);
        assert_eq!(hi, [0x80], "masked to one bit");
        for i in 0..16 {
            let want = u128::from(i == 0 || i == 3 || i == 8 || i == 9);
            assert_eq!(read_elem(&bits, i, ElementKind::B1), want, "bit {i}");
        }
        assert_eq!(read_elem(&[0xA5], 0, ElementKind::U4), 0x5);
        assert_eq!(read_elem(&[0xA5], 1, ElementKind::U4), 0xA);
    }
}

#[cfg(test)]
mod fold_pins {
    //! Pins from the 2026-10-17 mutation audit of this file: the fold-shaped
    //! evaluators (`fold_step`, RowReduce, Scan, Window).
    //!
    //! Every case uses TWO rows, and the max/min cases use rows that are all negative
    //! (max) or all positive (min): the accumulators start at zero, so a single
    //! row or a mixed-sign row cannot tell "seed with the first element" from
    //! "compare against zero".
    use super::*;
    use crate::ir::{OpDef, ReduceOp, ReduceStage, input, reduced};
    use crate::plan::build_plan;
    use unpopped_vocab::{ArchSku, OpCategory, StructureKey, structure_key};

    fn desc(shape: &[i64], strides: &[i64], dt: ElementKind) -> OperandDesc {
        OperandDesc::new(shape.len(), shape, strides, dt, 256)
    }

    fn key(cat: OpCategory, ops: &[OperandDesc]) -> StructureKey {
        structure_key(cat, ops, ArchSku::Sm89)
    }

    fn f32s(b: &TypedBuffer) -> Vec<f32> {
        (0..b.len())
            .map(|i| f32::from_bits(b.bits_at(i) as u32))
            .collect()
    }

    fn i32s(b: &TypedBuffer) -> Vec<i32> {
        (0..b.len()).map(|i| b.bits_at(i) as u32 as i32).collect()
    }

    fn same_f32(got: &[f32], want: &[f32]) -> bool {
        got.len() == want.len()
            && got
                .iter()
                .zip(want)
                .all(|(g, w)| (g.is_nan() && w.is_nan()) || g.to_bits() == w.to_bits())
    }

    const NAN: f32 = f32::NAN;
    const INF: f32 = f32::INFINITY;

    /// `[rows, k]` reduced over the last axis to `[rows]`.
    fn reduce(
        op: ReduceOp,
        dt: ElementKind,
        input_buf: TypedBuffer,
        rows: i64,
        k: i64,
    ) -> TypedBuffer {
        let o = OpDef::reduction("r", 1, &[dt], input(0), op);
        let ind = desc(&[rows, k], &[k, 1], dt);
        let outd = desc(&[rows], &[1], dt);
        let k = key(OpCategory::Reduction, &[ind, outd]);
        let plan = build_plan(&o, &k);
        evaluate(&plan, &[ind, outd], &[input_buf], &[]).remove(0)
    }

    fn f32in(rows: usize, data: &[f32]) -> TypedBuffer {
        TypedBuffer::from_f32(&[rows as i64, (data.len() / rows) as i64], data)
    }

    fn i32in(rows: usize, data: &[i32]) -> TypedBuffer {
        TypedBuffer::from_i32(&[rows as i64, (data.len() / rows) as i64], data)
    }

    #[test]
    fn float_reductions_fold_each_row_independently() {
        let f = ElementKind::F32;
        let pos = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let neg = [-3.0, -1.0, -2.0, -5.0, -4.0, -9.0];
        assert_eq!(
            f32s(&reduce(ReduceOp::Sum, f, f32in(2, &pos), 2, 3)),
            [6.0, 15.0]
        );
        assert_eq!(
            f32s(&reduce(ReduceOp::Mean, f, f32in(2, &pos), 2, 3)),
            [2.0, 5.0]
        );
        assert_eq!(
            f32s(&reduce(ReduceOp::Prod, f, f32in(2, &pos), 2, 3)),
            [6.0, 120.0]
        );
        assert_eq!(
            f32s(&reduce(ReduceOp::Max, f, f32in(2, &neg), 2, 3)),
            [-1.0, -4.0]
        );
        assert_eq!(
            f32s(&reduce(ReduceOp::Min, f, f32in(2, &pos), 2, 3)),
            [1.0, 4.0]
        );
        assert_eq!(
            f32s(&reduce(ReduceOp::Max, f, f32in(2, &pos), 2, 3)),
            [3.0, 6.0]
        );
        assert_eq!(
            f32s(&reduce(ReduceOp::Min, f, f32in(2, &neg), 2, 3)),
            [-3.0, -9.0]
        );
        // A NaN sticks in a max/min run (and only in its own row).
        let with_nan = [1.0, NAN, 3.0, 4.0, 5.0, 6.0];
        assert!(same_f32(
            &f32s(&reduce(ReduceOp::Max, f, f32in(2, &with_nan), 2, 3)),
            &[NAN, 6.0]
        ));
        assert!(same_f32(
            &f32s(&reduce(ReduceOp::Min, f, f32in(2, &with_nan), 2, 3)),
            &[NAN, 4.0]
        ));
        // Only the NaN is the LAST element: it must still win.
        let nan_last = [1.0, 2.0, NAN, 4.0, 5.0, 6.0];
        assert!(same_f32(
            &f32s(&reduce(ReduceOp::Max, f, f32in(2, &nan_last), 2, 3)),
            &[NAN, 6.0]
        ));
    }

    #[test]
    fn integer_reductions_fold_each_row_independently() {
        let i = ElementKind::I32;
        let pos = [1, 2, 3, 4, 5, 6];
        let neg = [-3, -1, -2, -5, -4, -9];
        assert_eq!(
            i32s(&reduce(ReduceOp::Sum, i, i32in(2, &pos), 2, 3)),
            [6, 15]
        );
        assert_eq!(
            i32s(&reduce(ReduceOp::Prod, i, i32in(2, &pos), 2, 3)),
            [6, 120]
        );
        assert_eq!(
            i32s(&reduce(ReduceOp::Max, i, i32in(2, &neg), 2, 3)),
            [-1, -4]
        );
        assert_eq!(
            i32s(&reduce(ReduceOp::Min, i, i32in(2, &pos), 2, 3)),
            [1, 4]
        );
        assert_eq!(
            i32s(&reduce(ReduceOp::Max, i, i32in(2, &pos), 2, 3)),
            [3, 6]
        );
        assert_eq!(
            i32s(&reduce(ReduceOp::Min, i, i32in(2, &neg), 2, 3)),
            [-3, -9]
        );
        assert_eq!(
            i32s(&reduce(ReduceOp::Sum, i, i32in(2, &neg), 2, 3)),
            [-6, -18]
        );
    }

    /// A one-stage row reduction broadcast back over the row.
    fn row_reduce_one(op: ReduceOp, input_buf: TypedBuffer, rows: i64, k: i64) -> Vec<f32> {
        let stages = vec![ReduceStage {
            pre: input(0).0,
            op,
        }];
        let o = OpDef::row_reduce("rr", 1, &[ElementKind::F32], stages, reduced(0));
        let ind = desc(&[rows, k], &[k, 1], ElementKind::F32);
        let k = key(OpCategory::Softmax, &[ind, ind]);
        let plan = build_plan(&o, &k);
        f32s(&evaluate(&plan, &[ind, ind], &[input_buf], &[]).remove(0))
    }

    #[test]
    fn row_reduce_stages_fold_each_row_over_its_own_elements() {
        let pos = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let neg = [-3.0, -1.0, -2.0, -5.0, -4.0, -9.0];
        let rep = |a: f32, b: f32| vec![a, a, a, b, b, b];
        assert_eq!(
            row_reduce_one(ReduceOp::Sum, f32in(2, &pos), 2, 3),
            rep(6.0, 15.0)
        );
        assert_eq!(
            row_reduce_one(ReduceOp::Mean, f32in(2, &pos), 2, 3),
            rep(2.0, 5.0)
        );
        assert_eq!(
            row_reduce_one(ReduceOp::Max, f32in(2, &neg), 2, 3),
            rep(-1.0, -4.0)
        );
        assert_eq!(
            row_reduce_one(ReduceOp::Min, f32in(2, &pos), 2, 3),
            rep(1.0, 4.0)
        );
        assert_eq!(
            row_reduce_one(ReduceOp::Max, f32in(2, &pos), 2, 3),
            rep(3.0, 6.0)
        );
        assert_eq!(
            row_reduce_one(ReduceOp::Min, f32in(2, &neg), 2, 3),
            rep(-3.0, -9.0)
        );
        let with_nan = [1.0, NAN, 3.0, 4.0, 5.0, 6.0];
        assert!(same_f32(
            &row_reduce_one(ReduceOp::Max, f32in(2, &with_nan), 2, 3),
            &[NAN, NAN, NAN, 6.0, 6.0, 6.0]
        ));
        assert!(same_f32(
            &row_reduce_one(ReduceOp::Min, f32in(2, &with_nan), 2, 3),
            &[NAN, NAN, NAN, 4.0, 4.0, 4.0]
        ));
    }

    #[test]
    fn row_reduce_epilogue_reads_both_the_element_and_the_stage_results() {
        // x - mean(x), then x / max(x): two stages, epilogue mixes input and Reduced.
        let stages = vec![
            ReduceStage {
                pre: input(0).0,
                op: ReduceOp::Mean,
            },
            ReduceStage {
                pre: (input(0) - reduced(0)).0,
                op: ReduceOp::Max,
            },
        ];
        let o = OpDef::row_reduce(
            "rr2",
            1,
            &[ElementKind::F32],
            stages,
            (input(0) - reduced(0)) / reduced(1),
        );
        let ind = desc(&[2, 4], &[4, 1], ElementKind::F32);
        let k = key(OpCategory::Softmax, &[ind, ind]);
        let plan = build_plan(&o, &k);
        let data = [1.0, 2.0, 3.0, 6.0, 10.0, 10.0, 10.0, 14.0];
        let out = f32s(&evaluate(&plan, &[ind, ind], &[f32in(2, &data)], &[]).remove(0));
        // Row 0: mean 3, deviations [-2,-1,0,3], max dev 3. Row 1: mean 11, devs [-1,-1,-1,3].
        let want = [
            -2.0 / 3.0,
            -1.0 / 3.0,
            0.0,
            1.0,
            -1.0 / 3.0,
            -1.0 / 3.0,
            -1.0 / 3.0,
            1.0,
        ];
        for (g, w) in out.iter().zip(want) {
            assert!((g - w).abs() < 1e-6, "{out:?} vs {want:?}");
        }
    }

    fn scan(
        op: ReduceOp,
        dt: ElementKind,
        reverse: bool,
        exclusive: bool,
        input_buf: TypedBuffer,
    ) -> TypedBuffer {
        let o = OpDef::scan_simple("s", &[dt], op, 1, reverse, exclusive);
        let ind = desc(&[2, 4], &[4, 1], dt);
        let k = key(OpCategory::Scan, &[ind, ind]);
        let plan = build_plan(&o, &k);
        evaluate(&plan, &[ind, ind], &[input_buf], &[]).remove(0)
    }

    #[test]
    fn float_scans_run_per_row_in_every_direction() {
        let f = ElementKind::F32;
        let d = [1.0, 2.0, 3.0, 4.0, 10.0, 20.0, 30.0, 40.0];
        let run = |op, rev, excl, data: &[f32]| f32s(&scan(op, f, rev, excl, f32in(2, data)));
        assert_eq!(
            run(ReduceOp::Sum, false, false, &d),
            [1.0, 3.0, 6.0, 10.0, 10.0, 30.0, 60.0, 100.0]
        );
        assert_eq!(
            run(ReduceOp::Sum, false, true, &d),
            [0.0, 1.0, 3.0, 6.0, 0.0, 10.0, 30.0, 60.0]
        );
        assert_eq!(
            run(ReduceOp::Sum, true, false, &d),
            [10.0, 9.0, 7.0, 4.0, 100.0, 90.0, 70.0, 40.0]
        );
        assert_eq!(
            run(ReduceOp::Sum, true, true, &d),
            [9.0, 7.0, 4.0, 0.0, 90.0, 70.0, 40.0, 0.0]
        );
        let p = [1.0, 2.0, 3.0, 4.0, 2.0, 2.0, 2.0, 2.0];
        assert_eq!(
            run(ReduceOp::Prod, false, false, &p),
            [1.0, 2.0, 6.0, 24.0, 2.0, 4.0, 8.0, 16.0]
        );
        assert_eq!(
            run(ReduceOp::Prod, false, true, &p),
            [1.0, 1.0, 2.0, 6.0, 1.0, 2.0, 4.0, 8.0]
        );
        // All-negative rows for Max, all-positive for Min: the identity is +-inf.
        let neg = [-3.0, -1.0, -4.0, -2.0, -9.0, -8.0, -7.0, -6.0];
        assert_eq!(
            run(ReduceOp::Max, false, false, &neg),
            [-3.0, -1.0, -1.0, -1.0, -9.0, -8.0, -7.0, -6.0]
        );
        assert_eq!(
            run(ReduceOp::Max, false, true, &neg),
            [-INF, -3.0, -1.0, -1.0, -INF, -9.0, -8.0, -7.0]
        );
        assert_eq!(
            run(ReduceOp::Max, true, false, &neg),
            [-1.0, -1.0, -2.0, -2.0, -6.0, -6.0, -6.0, -6.0]
        );
        let pos = [3.0, 1.0, 4.0, 2.0, 9.0, 8.0, 7.0, 6.0];
        assert_eq!(
            run(ReduceOp::Min, false, false, &pos),
            [3.0, 1.0, 1.0, 1.0, 9.0, 8.0, 7.0, 6.0]
        );
        assert_eq!(
            run(ReduceOp::Min, false, true, &pos),
            [INF, 3.0, 1.0, 1.0, INF, 9.0, 8.0, 7.0]
        );
        assert_eq!(
            run(ReduceOp::Min, true, false, &pos),
            [1.0, 1.0, 2.0, 2.0, 6.0, 6.0, 6.0, 6.0]
        );
        // A NaN sticks for the rest of its row's prefix, and only that row's.
        let n = [1.0, NAN, 3.0, 2.0, 5.0, 6.0, 7.0, 8.0];
        assert!(same_f32(
            &run(ReduceOp::Max, false, false, &n),
            &[1.0, NAN, NAN, NAN, 5.0, 6.0, 7.0, 8.0]
        ));
        assert!(same_f32(
            &run(ReduceOp::Min, false, false, &n),
            &[1.0, NAN, NAN, NAN, 5.0, 5.0, 5.0, 5.0]
        ));
    }

    #[test]
    fn integer_scans_run_per_row_in_every_direction() {
        let t = ElementKind::I32;
        let run = |op, rev, excl, data: &[i32]| i32s(&scan(op, t, rev, excl, i32in(2, data)));
        let d = [1, 2, 3, 4, 10, 20, 30, 40];
        assert_eq!(
            run(ReduceOp::Sum, false, false, &d),
            [1, 3, 6, 10, 10, 30, 60, 100]
        );
        assert_eq!(
            run(ReduceOp::Sum, false, true, &d),
            [0, 1, 3, 6, 0, 10, 30, 60]
        );
        assert_eq!(
            run(ReduceOp::Sum, true, false, &d),
            [10, 9, 7, 4, 100, 90, 70, 40]
        );
        let p = [1, 2, 3, 4, 2, 2, 2, 2];
        assert_eq!(
            run(ReduceOp::Prod, false, false, &p),
            [1, 2, 6, 24, 2, 4, 8, 16]
        );
        let neg = [-3, -1, -4, -2, -9, -8, -7, -6];
        assert_eq!(
            run(ReduceOp::Max, false, false, &neg),
            [-3, -1, -1, -1, -9, -8, -7, -6]
        );
        assert_eq!(
            run(ReduceOp::Max, false, true, &neg),
            [i32::MIN, -3, -1, -1, i32::MIN, -9, -8, -7]
        );
        let pos = [3, 1, 4, 2, 9, 8, 7, 6];
        assert_eq!(
            run(ReduceOp::Min, false, false, &pos),
            [3, 1, 1, 1, 9, 8, 7, 6]
        );
        assert_eq!(
            run(ReduceOp::Min, false, true, &pos),
            [i32::MAX, 3, 1, 1, i32::MAX, 9, 8, 7]
        );
        assert_eq!(
            run(ReduceOp::Min, true, false, &pos),
            [1, 1, 2, 2, 6, 6, 6, 6]
        );
    }

    #[allow(
        clippy::too_many_arguments,
        reason = "test helper: one argument per window parameter"
    )]
    fn window(
        op: ReduceOp,
        dt: ElementKind,
        size: u8,
        stride: u8,
        dilation: u8,
        pad_lo: u8,
        pad_hi: u8,
        count_include_pad: bool,
        k_in: i64,
        k_out: i64,
        input_buf: TypedBuffer,
    ) -> TypedBuffer {
        let o = OpDef::window_simple(
            "w",
            &[dt],
            op,
            1,
            size,
            stride,
            dilation,
            pad_lo,
            pad_hi,
            count_include_pad,
        );
        let ind = desc(&[2, k_in], &[k_in, 1], dt);
        let outd = desc(&[2, k_out], &[k_out, 1], dt);
        let k = key(OpCategory::Pooling, &[ind, outd]);
        let plan = build_plan(&o, &k);
        evaluate(&plan, &[ind, outd], &[input_buf], &[]).remove(0)
    }

    #[test]
    fn windows_pool_each_row_over_its_own_taps() {
        let f = ElementKind::F32;
        // Max/min, size 2 stride 2: all-negative second row for max, all-positive for min.
        let d = [
            1.0, 5.0, 2.0, 2.0, 9.0, 3.0, -1.0, -5.0, -2.0, -7.0, -9.0, -3.0,
        ];
        let mx = window(ReduceOp::Max, f, 2, 2, 1, 0, 0, false, 6, 3, f32in(2, &d));
        assert_eq!(f32s(&mx), [5.0, 2.0, 9.0, -1.0, -2.0, -3.0]);
        let p = [4.0, 5.0, 2.0, 2.0, 9.0, 3.0, 8.0, 5.0, 7.0, 7.0, 9.0, 3.0];
        let mn = window(ReduceOp::Min, f, 2, 2, 1, 0, 0, false, 6, 3, f32in(2, &p));
        assert_eq!(f32s(&mn), [4.0, 2.0, 3.0, 5.0, 7.0, 3.0]);
        // Sum with padding on both sides: taps outside the row are skipped.
        let s = [1.0, 2.0, 3.0, 4.0, 10.0, 20.0, 30.0, 40.0];
        let sum = window(ReduceOp::Sum, f, 3, 1, 1, 1, 1, false, 4, 4, f32in(2, &s));
        assert_eq!(f32s(&sum), [3.0, 6.0, 9.0, 7.0, 30.0, 60.0, 90.0, 70.0]);
        // Dilation 2: taps o and o + 2.
        let dl = [1.0, 2.0, 3.0, 4.0, 5.0, 10.0, 20.0, 30.0, 40.0, 50.0];
        let dil = window(ReduceOp::Sum, f, 2, 1, 2, 0, 0, false, 5, 3, f32in(2, &dl));
        assert_eq!(f32s(&dil), [4.0, 6.0, 8.0, 40.0, 60.0, 80.0]);
        // Stride 3, size 2: windows start at 0 and 3.
        let st = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0];
        let strided = window(ReduceOp::Sum, f, 2, 3, 1, 0, 0, false, 6, 2, f32in(2, &st));
        assert_eq!(f32s(&strided), [3.0, 9.0, 2.0, 2.0]);
        // A NaN in a window sticks for max.
        let n = [1.0, NAN, 3.0, 4.0, 5.0, 6.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let nan = window(ReduceOp::Max, f, 2, 2, 1, 0, 0, false, 6, 3, f32in(2, &n));
        assert!(same_f32(&f32s(&nan), &[NAN, 4.0, 6.0, 2.0, 4.0, 6.0]));
        let nan_min = window(ReduceOp::Min, f, 2, 2, 1, 0, 0, false, 6, 3, f32in(2, &n));
        assert!(same_f32(&f32s(&nan_min), &[NAN, 3.0, 5.0, 1.0, 3.0, 5.0]));
    }

    #[test]
    fn an_average_window_divides_by_the_size_or_the_valid_taps() {
        let f = ElementKind::F32;
        let d = [3.0, 6.0, 9.0, 30.0, 60.0, 90.0];
        let with_pad = window(ReduceOp::Mean, f, 3, 1, 1, 1, 1, true, 3, 3, f32in(2, &d));
        assert_eq!(f32s(&with_pad), [3.0, 6.0, 5.0, 30.0, 60.0, 50.0]);
        let no_pad = window(ReduceOp::Mean, f, 3, 1, 1, 1, 1, false, 3, 3, f32in(2, &d));
        assert_eq!(f32s(&no_pad), [4.5, 6.0, 7.5, 45.0, 60.0, 75.0]);
    }

    #[test]
    fn integer_windows_pool_each_row_over_its_own_taps() {
        let t = ElementKind::I32;
        let neg = [-1, -5, -2, -2, -9, -3, -4, -5, -7, -7, -9, -3];
        let mx = window(ReduceOp::Max, t, 2, 2, 1, 0, 0, false, 6, 3, i32in(2, &neg));
        assert_eq!(i32s(&mx), [-1, -2, -3, -4, -7, -3]);
        let pos = [4, 5, 2, 2, 9, 3, 8, 5, 7, 7, 9, 3];
        let mn = window(ReduceOp::Min, t, 2, 2, 1, 0, 0, false, 6, 3, i32in(2, &pos));
        assert_eq!(i32s(&mn), [4, 2, 3, 5, 7, 3]);
        let s = [1, 2, 3, 4, 10, 20, 30, 40];
        let sum = window(ReduceOp::Sum, t, 3, 1, 1, 1, 1, false, 4, 4, i32in(2, &s));
        assert_eq!(i32s(&sum), [3, 6, 9, 7, 30, 60, 90, 70]);
    }

    fn bits(v: &[f32]) -> Vec<u32> {
        v.iter().map(|x| x.to_bits()).collect()
    }

    #[test]
    fn float_max_and_min_keep_the_first_of_two_equal_zeros() {
        // `-0.0 == 0.0`, so a strict compare keeps the first one seen; `>=`/`<=` would
        // overwrite it and flip the sign of the result.
        let f = ElementKind::F32;
        let neg_first = [-0.0f32, 0.0];
        let pos_first = [0.0f32, -0.0];
        let red = |op, d: &[f32]| f32s(&reduce(op, f, f32in(1, d), 1, 2));
        assert_eq!(bits(&red(ReduceOp::Max, &neg_first)), [(-0.0f32).to_bits()]);
        assert_eq!(bits(&red(ReduceOp::Min, &pos_first)), [0.0f32.to_bits()]);

        let row = |op, d: &[f32]| row_reduce_one(op, f32in(1, d), 1, 2);
        assert_eq!(
            bits(&row(ReduceOp::Max, &neg_first))[0],
            (-0.0f32).to_bits()
        );
        assert_eq!(bits(&row(ReduceOp::Min, &pos_first))[0], 0.0f32.to_bits());

        let sc = |op, d: &[f32]| {
            let o = OpDef::scan_simple("s", &[f], op, 1, false, false);
            let ind = desc(&[1, 2], &[2, 1], f);
            let k = key(OpCategory::Scan, &[ind, ind]);
            let plan = build_plan(&o, &k);
            f32s(&evaluate(&plan, &[ind, ind], &[f32in(1, d)], &[]).remove(0))
        };
        assert_eq!(
            bits(&sc(ReduceOp::Max, &neg_first)),
            [(-0.0f32).to_bits(); 2]
        );
        assert_eq!(bits(&sc(ReduceOp::Min, &pos_first)), [0.0f32.to_bits(); 2]);

        // Two taps per window, stride one: windows (x0,x1) and (x1,x2) of each row.
        let win = |op, d: &[f32]| {
            let two = [d, d].concat();
            f32s(&window(
                op,
                f,
                2,
                1,
                1,
                0,
                0,
                false,
                3,
                2,
                TypedBuffer::from_f32(&[2, 3], &two),
            ))
        };
        let neg_pos_neg = [-0.0f32, 0.0, -0.0];
        let pos_neg_pos = [0.0f32, -0.0, 0.0];
        let first_of_each = [-0.0f32, 0.0, -0.0, 0.0];
        assert_eq!(
            bits(&win(ReduceOp::Max, &neg_pos_neg)),
            bits(&first_of_each)
        );
        let first_of_each = [0.0f32, -0.0, 0.0, -0.0];
        assert_eq!(
            bits(&win(ReduceOp::Min, &pos_neg_pos)),
            bits(&first_of_each)
        );
    }

    #[test]
    #[should_panic(expected = "has no in-bounds tap")]
    fn a_window_over_only_padding_is_declined_not_answered_with_zero() {
        // Two taps three apart with two cells of padding each side: the middle output's
        // taps (-1 and 2) both miss the two-element row. Before 0.39.0 the oracle answered 0
        // here (the expected values were [7, 0, 5, 3, 0, 1]); KISS-OPS-6.13-0011 makes a reader
        // decline such a window, so the oracle now panics naming it.
        let f = ElementKind::F32;
        let got = f32s(&window(
            ReduceOp::Mean,
            f,
            2,
            1,
            3,
            2,
            2,
            false,
            2,
            3,
            TypedBuffer::from_f32(&[2, 2], &[5.0, 7.0, 1.0, 3.0]),
        ));
        panic!("the oracle must have declined, got {got:?}");
    }

    #[test]
    fn a_per_row_scalar_operand_is_read_once_per_row_in_every_role_based_op() {
        // `last` (the feature axis) is what tells a per-row scalar ([rows,k] with strides
        // [1,0]) from a per-column weight; the scalar must be indexed by row.
        let f = ElementKind::F32;
        let ind = desc(&[2, 4], &[4, 1], f);
        let rs = desc(&[2, 4], &[1, 0], f);
        let k = key(OpCategory::Softmax, &[ind, rs, ind]);
        let per_row = TypedBuffer::from_f32(&[2], &[100.0, 200.0]);
        let x = TypedBuffer::from_f32(&[2, 4], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
        let want = [101.0, 102.0, 103.0, 104.0, 205.0, 206.0, 207.0, 208.0];

        let rr = OpDef::row_reduce(
            "rr",
            2,
            &[f, f],
            vec![ReduceStage {
                pre: input(0).0,
                op: ReduceOp::Sum,
            }],
            input(0) + input(1),
        );
        let plan = build_plan(&rr, &k);
        let got = evaluate(&plan, &[ind, rs, ind], &[x.clone(), per_row.clone()], &[]);
        assert_eq!(f32s(&got[0]), want);

        let k = key(OpCategory::Scan, &[ind, rs, ind]);
        let sc = OpDef::scan(
            "s",
            2,
            &[f, f],
            ReduceOp::Sum,
            1,
            false,
            false,
            input(0) + input(1),
            reduced(0),
        );
        let plan = build_plan(&sc, &k);
        let got = evaluate(&plan, &[ind, rs, ind], &[x.clone(), per_row.clone()], &[]);
        assert_eq!(
            f32s(&got[0]),
            [101.0, 203.0, 306.0, 410.0, 205.0, 411.0, 618.0, 826.0]
        );

        let k = key(OpCategory::Pooling, &[ind, rs, ind]);
        let wi = OpDef::window(
            "w",
            2,
            &[f, f],
            ReduceOp::Sum,
            1,
            1,
            1,
            1,
            0,
            0,
            false,
            input(0),
            reduced(0) + input(1),
        );
        let plan = build_plan(&wi, &k);
        let got = evaluate(&plan, &[ind, rs, ind], &[x, per_row], &[]);
        assert_eq!(f32s(&got[0]), want);
    }
}

#[cfg(test)]
mod im2col_pins {
    //! Pins from the 2026-10-17 mutation audit of this file: `eval_im2col`.
    //!
    //! The reference is a direct six-deep loop over (batch, channel, kernel tap, output
    //! position) that never divides or takes a remainder, unlike the flat-index
    //! decomposition in the oracle.
    use super::*;
    use crate::ir::OpDef;
    use crate::plan::build_plan;
    use unpopped_vocab::{ArchSku, OpCategory, structure_key};

    fn desc(shape: &[i64], strides: &[i64], dt: ElementKind) -> OperandDesc {
        OperandDesc::new(shape.len(), shape, strides, dt, 256)
    }

    fn run(
        shape: [i64; 4],
        kernel: (u8, u8),
        stride: (u8, u8),
        pad: (u8, u8),
        dilation: (u8, u8),
        base_offset: i64,
    ) {
        let [n, c, h, w] = shape;
        let (kh, kw) = (i64::from(kernel.0), i64::from(kernel.1));
        let o_h = (h + 2 * i64::from(pad.0) - i64::from(dilation.0) * (kh - 1) - 1)
            / i64::from(stride.0)
            + 1;
        let o_w = (w + 2 * i64::from(pad.1) - i64::from(dilation.1) * (kw - 1) - 1)
            / i64::from(stride.1)
            + 1;
        let dt = ElementKind::F32;
        let ind = desc(&shape, &[c * h * w, h * w, w, 1], dt);
        let outd = desc(
            &[n, c * kh * kw, o_h * o_w],
            &[c * kh * kw * o_h * o_w, o_h * o_w, 1],
            dt,
        );
        let o = OpDef::im2col_2d("im2col", dt, kernel, stride, pad, dilation);
        let k = structure_key(OpCategory::Convolution, &[ind, outd], ArchSku::Sm89);
        let plan = build_plan(&o, &k);

        // Element (nn, cc, y, x) holds a distinct value; the buffer starts `base_offset`
        // elements before it.
        let value =
            |nn: i64, cc: i64, y: i64, x: i64| (((nn * c + cc) * h + y) * w + x) as f32 + 1.0;
        let mut data = vec![-7.0f32; base_offset as usize];
        for nn in 0..n {
            for cc in 0..c {
                for y in 0..h {
                    for x in 0..w {
                        data.push(value(nn, cc, y, x));
                    }
                }
            }
        }
        let mut input = TypedBuffer::from_f32(&[data.len() as i64], &data);
        input.shape = shape.to_vec();
        input.base_offset = base_offset;

        let got = evaluate(&plan, &[ind, outd], &[input], &[]).remove(0);
        let got: Vec<f32> = (0..got.len())
            .map(|i| f32::from_bits(got.bits_at(i) as u32))
            .collect();

        let mut want = Vec::new();
        for nn in 0..n {
            for cc in 0..c {
                for ki in 0..kh {
                    for kj in 0..kw {
                        for oh in 0..o_h {
                            for ow in 0..o_w {
                                let y = oh * i64::from(stride.0) - i64::from(pad.0)
                                    + ki * i64::from(dilation.0);
                                let x = ow * i64::from(stride.1) - i64::from(pad.1)
                                    + kj * i64::from(dilation.1);
                                let inside = (0..h).contains(&y) && (0..w).contains(&x);
                                want.push(if inside { value(nn, cc, y, x) } else { 0.0 });
                            }
                        }
                    }
                }
            }
        }
        assert_eq!(
            got, want,
            "{shape:?} k{kernel:?} s{stride:?} p{pad:?} d{dilation:?}"
        );
    }

    #[test]
    fn im2col_unrolls_every_batch_channel_tap_and_position() {
        // Every extent differs, so a swapped height/width or batch/channel stride is visible.
        run([2, 3, 5, 4], (2, 3), (1, 1), (0, 0), (1, 1), 0);
        run([2, 3, 5, 4], (3, 2), (2, 1), (1, 2), (1, 1), 0);
        run([2, 2, 6, 5], (2, 2), (1, 2), (2, 1), (2, 3), 0);
        run([2, 2, 7, 6], (3, 3), (3, 2), (1, 1), (2, 1), 0);
        // Size-1 batch or channel axes (rejected by the plan gate before 0.20.6).
        run([1, 2, 5, 4], (2, 2), (1, 1), (1, 0), (1, 1), 0);
        run([2, 1, 5, 4], (3, 2), (2, 1), (0, 1), (1, 2), 0);
    }

    #[test]
    fn im2col_reads_from_the_buffers_base_offset() {
        run([2, 2, 4, 3], (2, 2), (1, 1), (1, 1), (1, 1), 5);
    }
}

#[cfg(test)]
mod compare_pins {
    //! Pins from the 2026-10-18 mutation audit of this file: `compare`.
    //!
    //! Every case states which message it expects, because `compare` returns the SAME `Err`
    //! type for every rejection and a mutant that swaps one rejection for another still
    //! returns `Err`.
    use super::*;

    fn tol(rel: f64, abs: f64) -> Fidelity {
        Fidelity::Tolerant { rel, abs }
    }

    fn err(e: &TypedBuffer, a: &TypedBuffer, f: Fidelity) -> String {
        compare(e, a, f).expect_err("expected a mismatch")
    }

    #[test]
    fn a_bit_exact_mismatch_names_the_first_differing_element() {
        let e = TypedBuffer::from_f32(&[3], &[1.0, 2.0, 3.0]);
        let a = TypedBuffer::from_f32(&[3], &[1.0, 9.0, 3.0]);
        let msg = err(&e, &a, Fidelity::BitExact);
        assert!(msg.starts_with("bit-exact mismatch at element 1:"), "{msg}");
        assert!(
            msg.contains(&format!(
                "0x{:x} vs 0x{:x}",
                2.0f32.to_bits(),
                9.0f32.to_bits()
            )),
            "{msg}"
        );
    }

    #[test]
    fn equal_elements_with_unequal_bytes_report_a_byte_length_mismatch() {
        let mut long = 1.0f32.to_le_bytes().to_vec();
        long.extend_from_slice(&2.0f32.to_le_bytes());
        let mut longer = long.clone();
        longer.extend_from_slice(&[0, 0, 0, 0]);
        let mk = |bytes: Vec<u8>| TypedBuffer::new(ElementKind::F32, vec![2], vec![1], bytes);
        assert_eq!(
            err(&mk(long), &mk(longer), Fidelity::BitExact),
            "byte-length mismatch"
        );
    }

    #[test]
    fn complex_equal_infinities_match_and_unequal_ones_do_not() {
        let inf = f32::INFINITY;
        let a = TypedBuffer::from_complex64(&[1], &[(inf, 1.0)]);
        assert!(compare(&a, &a, tol(0.0, 0.0)).is_ok());
        let b = TypedBuffer::from_complex64(&[1], &[(1.0, 1.0)]);
        // With a relative band an infinite operand would swallow any mismatch, so the
        // rejection has to name the infinity, not fall through to the band test.
        let msg = err(&a, &b, tol(0.5, 0.0));
        assert!(msg.contains("infinity mismatch at element 0 (re)"), "{msg}");
        let msg = err(&b, &a, tol(0.5, 0.0));
        assert!(msg.contains("infinity mismatch at element 0 (re)"), "{msg}");
        let c = TypedBuffer::from_complex64(&[1], &[(1.0, f32::NEG_INFINITY)]);
        let msg = err(&b, &c, tol(0.5, 0.0));
        assert!(msg.contains("infinity mismatch at element 0 (im)"), "{msg}");
    }

    #[test]
    fn complex_nan_matches_only_nan() {
        let nan = f32::NAN;
        let a = TypedBuffer::from_complex64(&[1], &[(nan, 0.0)]);
        let b = TypedBuffer::from_complex64(&[1], &[(1.0, 0.0)]);
        assert!(compare(&a, &a, tol(0.0, 0.0)).is_ok());
        assert!(compare(&a, &b, tol(1.0, 1.0)).is_err());
        assert!(compare(&b, &a, tol(1.0, 1.0)).is_err());
    }

    #[test]
    fn the_complex_band_is_abs_plus_rel_times_the_larger_magnitude_per_component() {
        // |d| = 0.5, band = 0.1 + 0.01 * 100.5 = 1.105 for the real part.
        let e = TypedBuffer::from_complex64(&[1], &[(100.0, 100.0)]);
        let near = TypedBuffer::from_complex64(&[1], &[(100.5, 100.0)]);
        assert!(compare(&e, &near, tol(0.01, 0.1)).is_ok());
        let near_im = TypedBuffer::from_complex64(&[1], &[(100.0, 100.5)]);
        assert!(compare(&e, &near_im, tol(0.01, 0.1)).is_ok());
        // |d| = 2 is outside it, in either component, and the message names the part.
        let far = TypedBuffer::from_complex64(&[1], &[(102.0, 100.0)]);
        assert!(err(&e, &far, tol(0.01, 0.1)).contains("tolerance mismatch at element 0 (re)"));
        let far_im = TypedBuffer::from_complex64(&[1], &[(100.0, 102.0)]);
        assert!(err(&e, &far_im, tol(0.01, 0.1)).contains("(im)"));
    }

    #[test]
    fn the_integer_band_is_abs_plus_rel_times_the_larger_magnitude() {
        // |d| = 10, band = 1 + 0.01 * 1010 = 11.1.
        let e = TypedBuffer::from_i32(&[1], &[1000]);
        let a = TypedBuffer::from_i32(&[1], &[1010]);
        assert!(compare(&e, &a, tol(0.01, 1.0)).is_ok());
        assert!(compare(&a, &e, tol(0.01, 1.0)).is_ok());
        // |d| = 12 is outside it.
        let far = TypedBuffer::from_i32(&[1], &[1012]);
        assert!(err(&e, &far, tol(0.01, 1.0)).contains("tolerance mismatch at element 0"));
        assert!(compare(&e, &far, tol(0.0, 0.0)).is_err());
    }

    #[test]
    fn mixed_int_and_float_buffers_compare_through_the_float_projection() {
        // Both sides must be integer to take the exact path; one integer side is a float
        // comparison, not a panic in `to_i128_vec`.
        let i = TypedBuffer::from_i32(&[2], &[1, 2]);
        let f = TypedBuffer::from_f32(&[2], &[1.0, 2.0]);
        assert!(compare(&i, &f, tol(0.0, 0.0)).is_ok());
        assert!(compare(&f, &i, tol(0.0, 0.0)).is_ok());
    }

    // A complex buffer against a real one is a misuse the comparator does not support. It must
    // still go down the real projection (which refuses complex), not the component-wise path
    // (whose own assertion would blame the real buffer instead).
    #[test]
    #[should_panic(expected = "raw_to_f64 on unsupported dtype")]
    fn a_complex_expected_against_a_real_actual_takes_the_real_projection() {
        let c = TypedBuffer::from_complex64(&[1], &[(1.0, 0.0)]);
        let f = TypedBuffer::from_f32(&[1], &[1.0]);
        let _ = compare(&c, &f, tol(0.0, 0.0));
    }

    #[test]
    #[should_panic(expected = "raw_to_f64 on unsupported dtype")]
    fn a_real_expected_against_a_complex_actual_takes_the_real_projection() {
        let c = TypedBuffer::from_complex64(&[1], &[(1.0, 0.0)]);
        let f = TypedBuffer::from_f32(&[1], &[1.0]);
        let _ = compare(&f, &c, tol(0.0, 0.0));
    }

    #[test]
    fn float_nan_matches_only_nan() {
        let n = TypedBuffer::from_f32(&[1], &[f32::NAN]);
        let one = TypedBuffer::from_f32(&[1], &[1.0]);
        assert!(compare(&n, &n, tol(0.0, 0.0)).is_ok());
        assert!(compare(&n, &one, tol(1.0, 1.0)).is_err());
        assert!(compare(&one, &n, tol(1.0, 1.0)).is_err());
    }

    #[test]
    fn the_float_band_is_abs_plus_rel_times_the_larger_magnitude() {
        let e = TypedBuffer::from_f64(&[1], &[100.0]);
        let near = TypedBuffer::from_f64(&[1], &[100.5]);
        assert!(compare(&e, &near, tol(0.01, 0.1)).is_ok());
        assert!(compare(&near, &e, tol(0.01, 0.1)).is_ok());
        let far = TypedBuffer::from_f64(&[1], &[102.0]);
        assert!(err(&e, &far, tol(0.01, 0.1)).contains("tolerance mismatch at element 0"));
    }
}

#[cfg(test)]
mod binary_pins {
    //! Pins from the 2026-10-18 mutation audit of this file: `eval_binary`.
    //!
    //! The bit-preserving `Copysign` / `Max` / `Min` path on a narrow float moves the winner's
    //! RAW bits, so every case here is read back as storage bits, not as a value: the `f64`
    //! detour agrees with it on every ordinary number and loses only a NaN's payload and
    //! (for a tie) which zero was kept.
    use super::*;
    use crate::ir::{OpDef, input};
    use crate::plan::build_plan;
    use unpopped_vocab::{ArchSku, OpCategory, structure_key};

    // (dtype, 1.0, 2.0, -2.0, +0.0, -0.0, a NaN payload that `f64` cannot carry)
    struct Fmt {
        dt: ElementKind,
        one: u16,
        two: u16,
        neg_two: u16,
        pos_zero: u16,
        neg_zero: u16,
        nan: u16,
        neg_nan: u16,
    }

    const FORMATS: [Fmt; 4] = [
        // e5m2: s eeeee mm; NaN is exp 31, mantissa != 0 (0x7D, 0x7E, 0x7F).
        Fmt {
            dt: ElementKind::Fp8E5M2,
            one: 0x3C,
            two: 0x40,
            neg_two: 0xC0,
            pos_zero: 0x00,
            neg_zero: 0x80,
            nan: 0x7D,
            neg_nan: 0xFD,
        },
        // e4m3fn: s eeee mmm; NaN is 0x7F only, so the NaN case is the single encoding.
        Fmt {
            dt: ElementKind::Fp8E4M3FN,
            one: 0x38,
            two: 0x40,
            neg_two: 0xC0,
            pos_zero: 0x00,
            neg_zero: 0x80,
            nan: 0x7F,
            neg_nan: 0xFF,
        },
        // f16: a signaling NaN with a payload.
        Fmt {
            dt: ElementKind::F16,
            one: 0x3C00,
            two: 0x4000,
            neg_two: 0xC000,
            pos_zero: 0x0000,
            neg_zero: 0x8000,
            nan: 0x7C01,
            neg_nan: 0xFC01,
        },
        // bf16: a signaling NaN with a payload.
        Fmt {
            dt: ElementKind::Bf16,
            one: 0x3F80,
            two: 0x4000,
            neg_two: 0xC000,
            pos_zero: 0x0000,
            neg_zero: 0x8000,
            nan: 0x7F81,
            neg_nan: 0xFF81,
        },
    ];

    fn buf(dt: ElementKind, bits: &[u16]) -> TypedBuffer {
        let bytes: Vec<u8> = if elem_size(dt) == 1 {
            bits.iter().map(|&b| b as u8).collect()
        } else {
            bits.iter().flat_map(|b| b.to_le_bytes()).collect()
        };
        TypedBuffer::new(dt, vec![bits.len() as i64], vec![1], bytes)
    }

    /// Run `build(input(0), input(1))` over the pairs and return the output storage bits.
    fn run(
        dt: ElementKind,
        build: fn(crate::ir::Expr, crate::ir::Expr) -> crate::ir::Expr,
        pairs: &[(u16, u16)],
    ) -> Vec<u16> {
        let op = OpDef::elementwise("b", 2, &[dt], build(input(0), input(1)));
        let n = pairs.len() as i64;
        let d = OperandDesc::new(1, &[n], &[1], dt, 256);
        let k = structure_key(OpCategory::BinaryElementwise, &[d, d, d], ArchSku::Sm89);
        let plan = build_plan(&op, &k);
        let a: Vec<u16> = pairs.iter().map(|p| p.0).collect();
        let b: Vec<u16> = pairs.iter().map(|p| p.1).collect();
        let out = evaluate(&plan, &[d, d, d], &[buf(dt, &a), buf(dt, &b)], &[]);
        (0..pairs.len()).map(|i| out[0].bits_at(i) as u16).collect()
    }

    fn copysign(a: crate::ir::Expr, b: crate::ir::Expr) -> crate::ir::Expr {
        a.binary(BinaryOp::Copysign, b)
    }

    #[test]
    fn copysign_takes_the_magnitude_of_a_and_the_sign_of_b_bit_for_bit() {
        for f in &FORMATS {
            let got = run(
                f.dt,
                copysign,
                &[
                    (f.one, f.neg_two),
                    (f.neg_two, f.two),
                    (f.two, f.two),
                    (f.neg_two, f.neg_two),
                    // The NaN's payload survives; only its sign comes from b.
                    (f.nan, f.neg_two),
                    (f.neg_nan, f.two),
                ],
            );
            let neg_one = f.one | (f.neg_zero);
            let pos_two = f.two;
            assert_eq!(
                got,
                vec![neg_one, pos_two, f.two, f.neg_two, f.neg_nan, f.nan],
                "{:?}",
                f.dt
            );
        }
    }

    #[test]
    fn max_picks_the_larger_operand_in_either_position() {
        for f in &FORMATS {
            let got = run(
                f.dt,
                |a, b| a.max(b),
                &[
                    (f.one, f.two),
                    (f.two, f.one),
                    (f.neg_two, f.one),
                    (f.one, f.neg_two),
                ],
            );
            assert_eq!(got, vec![f.two, f.two, f.one, f.one], "{:?}", f.dt);
        }
    }

    #[test]
    fn min_picks_the_smaller_operand_in_either_position() {
        for f in &FORMATS {
            let got = run(
                f.dt,
                |a, b| a.min(b),
                &[
                    (f.one, f.two),
                    (f.two, f.one),
                    (f.neg_two, f.one),
                    (f.one, f.neg_two),
                ],
            );
            assert_eq!(got, vec![f.one, f.one, f.neg_two, f.neg_two], "{:?}", f.dt);
        }
    }

    #[test]
    fn max_and_min_return_the_nan_operand_with_its_payload_in_either_position() {
        for f in &FORMATS {
            let pairs = [
                (f.nan, f.one),
                (f.one, f.nan),
                (f.neg_nan, f.one),
                (f.one, f.neg_nan),
                // Two NaNs: the first one wins.
                (f.nan, f.neg_nan),
                (f.neg_nan, f.nan),
            ];
            let want = vec![f.nan, f.nan, f.neg_nan, f.neg_nan, f.nan, f.neg_nan];
            assert_eq!(run(f.dt, |a, b| a.max(b), &pairs), want, "max {:?}", f.dt);
            assert_eq!(run(f.dt, |a, b| a.min(b), &pairs), want, "min {:?}", f.dt);
        }
    }

    #[test]
    fn max_and_min_of_equal_values_keep_the_first_operand_bits() {
        for f in &FORMATS {
            // +0 and -0 compare equal, so the tie is decided by position and the bits show it.
            let pairs = [(f.pos_zero, f.neg_zero), (f.neg_zero, f.pos_zero)];
            let want = vec![f.pos_zero, f.neg_zero];
            assert_eq!(run(f.dt, |a, b| a.max(b), &pairs), want, "max {:?}", f.dt);
            assert_eq!(run(f.dt, |a, b| a.min(b), &pairs), want, "min {:?}", f.dt);
        }
    }
}

#[cfg(test)]
mod arith_pins {
    //! Pins from the 2026-10-18 mutation audit of this file: `arith_steps`.
    //!
    //! Every operand count is distinct (1, 2, 3, 5), so a sum that became a difference or a
    //! product cannot land on the right total by coincidence.
    use super::*;
    use crate::ir::{BinaryOp, Expr, konst, param, reduced};
    use crate::ir::{coord, input};

    fn steps(e: &Expr) -> u32 {
        arith_steps(&e.0)
    }

    /// An expression of exactly `n` arithmetic steps: a chain of square roots over a leaf.
    fn chain(n: u32) -> Expr {
        (0..n).fold(input(0), |e, _| e.sqrt())
    }

    #[test]
    fn leaves_cost_nothing() {
        for e in [input(0), konst(1.0), param(0), reduced(0), coord(0)] {
            assert_eq!(steps(&e), 0);
        }
    }

    #[test]
    fn an_infix_node_costs_one_plus_both_operands() {
        // 1 + 2 + 3 = 6 for each of the four infix ops, with the larger chain on either side.
        for build in [
            |a: Expr, b: Expr| a + b,
            |a: Expr, b: Expr| a - b,
            |a: Expr, b: Expr| a * b,
            |a: Expr, b: Expr| a / b,
        ] {
            assert_eq!(steps(&build(chain(2), chain(3))), 6);
            assert_eq!(steps(&build(chain(3), chain(2))), 6);
        }
    }

    #[test]
    fn a_named_binary_node_costs_one_plus_both_operands() {
        for op in [BinaryOp::Max, BinaryOp::Pow, BinaryOp::Atan2] {
            assert_eq!(steps(&chain(2).binary(op, chain(3))), 6, "{op:?}");
            assert_eq!(steps(&chain(3).binary(op, chain(2))), 6, "{op:?}");
        }
    }

    #[test]
    fn a_unary_node_costs_one_plus_its_operand() {
        assert_eq!(steps(&chain(1)), 1);
        assert_eq!(steps(&chain(5)), 5);
        assert_eq!(steps(&chain(3).sqrt()), 4);
    }

    #[test]
    fn a_select_costs_only_its_three_subexpressions() {
        // 1 + 2 + 3 = 6 with no step of its own.
        assert_eq!(steps(&chain(1).select(chain(2), chain(3))), 6);
        assert_eq!(steps(&chain(3).select(chain(1), chain(2))), 6);
        assert_eq!(steps(&chain(2).select(chain(3), chain(1))), 6);
    }
}

#[cfg(test)]
mod encode_pins {
    //! Pins from the 2026-10-18 mutation audit of this file: the storage encoders
    //! (`encode_int`, `encode_float`, `encode_complex`, `encode_int_as_complex`), the
    //! `Val::complex` projection and `unit_roundoff`.
    use super::*;

    #[test]
    fn unit_roundoff_is_half_an_epsilon_of_the_compute_precision() {
        let f64_u = 2f64.powi(-53);
        let f32_u = 2f64.powi(-24);
        assert_eq!(unit_roundoff(ElementKind::F64), f64_u);
        assert_eq!(unit_roundoff(ElementKind::Complex128), f64_u);
        // Every narrow float computes in f32, so it shares f32's roundoff.
        for dt in [
            ElementKind::F32,
            ElementKind::Complex64,
            ElementKind::F16,
            ElementKind::Bf16,
            ElementKind::Fp8E4M3FN,
            ElementKind::Fp8E5M2,
        ] {
            assert_eq!(unit_roundoff(dt), f32_u, "{dt:?}");
        }
    }

    #[test]
    fn a_complex_pattern_has_the_real_part_low_and_the_imaginary_part_high() {
        let c64 = encode_complex(1.5, -2.5, ElementKind::Complex64);
        assert_eq!(c64 & 0xffff_ffff, u128::from(1.5f32.to_bits()));
        assert_eq!(c64 >> 32, u128::from((-2.5f32).to_bits()));
        let c128 = encode_complex(1.5, -2.5, ElementKind::Complex128);
        assert_eq!(c128 & u128::from(u64::MAX), u128::from(1.5f64.to_bits()));
        assert_eq!(c128 >> 64, u128::from((-2.5f64).to_bits()));
        // A one-sided value shows each half is kept, not merged or swapped.
        assert_eq!(
            encode_complex(0.0, 1.0, ElementKind::Complex64),
            u128::from(1.0f32.to_bits()) << 32
        );
        assert_eq!(
            encode_complex(1.0, 0.0, ElementKind::Complex128),
            u128::from(1.0f64.to_bits())
        );
    }

    #[test]
    fn an_integer_widens_to_a_complex_with_a_zero_imaginary_part() {
        assert_eq!(
            encode_int_as_complex(3, ElementKind::Complex64),
            u128::from(3.0f32.to_bits())
        );
        assert_eq!(
            encode_int_as_complex(-2, ElementKind::Complex128),
            u128::from((-2.0f64).to_bits())
        );
    }

    #[test]
    fn val_complex_projects_each_raw_domain_to_a_pair() {
        assert_eq!(
            Val::Raw(u128::from(1.5f32.to_bits()), ElementKind::F32).complex(),
            (1.5, 0.0)
        );
        assert_eq!(
            Val::Raw(u128::from((-5i32) as u32), ElementKind::I32).complex(),
            (-5.0, 0.0)
        );
        assert_eq!(
            Val::Raw(
                encode_complex(1.5, -2.5, ElementKind::Complex64),
                ElementKind::Complex64
            )
            .complex(),
            (1.5, -2.5)
        );
        assert_eq!(Val::Float(2.5).complex(), (2.5, 0.0));
        assert_eq!(Val::Int(-7).complex(), (-7.0, 0.0));
    }

    #[test]
    fn four_bit_and_one_bit_encoders_keep_only_their_own_bits() {
        for dt in [ElementKind::I4, ElementKind::U4] {
            assert_eq!(encode_int(-3, dt), 0xd, "{dt:?}");
            assert_eq!(encode_int(0x1f, dt), 0xf, "{dt:?}");
            assert_eq!(encode_int(0x10, dt), 0, "{dt:?}");
            assert_eq!(encode_float(-3.0, dt), 0xd, "{dt:?}");
            assert_eq!(encode_float(18.0, dt), 2, "{dt:?}");
            assert_eq!(encode_float(3.0, dt), 3, "{dt:?}");
        }
        assert_eq!(encode_int(3, ElementKind::B1), 1);
        assert_eq!(encode_int(2, ElementKind::B1), 0);
        assert_eq!(encode_int(-1, ElementKind::B1), 1);
    }

    #[test]
    fn a_one_bit_float_encoder_is_nonzero_ness() {
        assert_eq!(encode_float(2.0, ElementKind::B1), 1);
        assert_eq!(encode_float(-0.5, ElementKind::B1), 1);
        assert_eq!(encode_float(0.0, ElementKind::B1), 0);
        assert_eq!(encode_float(-0.0, ElementKind::B1), 0);
    }
}

#[cfg(test)]
mod plumbing_pins {
    //! Pins from the 2026-10-18 mutation audit of this file: the value plumbing around
    //! the evaluators (`raw_sign_masks`, `round_to_compute`, `unravel`, `store_val`,
    //! `alloc_output`) and the `Neg` / `Abs` / `Select` arms of `eval`.
    use super::*;
    use crate::ir::{Expr, OpDef, UnaryOp, input, konst};
    use crate::plan::build_plan;
    use unpopped_vocab::{ArchSku, OpCategory, structure_key};

    #[test]
    fn only_the_narrow_floats_have_sign_masks() {
        for dt in [ElementKind::Fp8E4M3FN, ElementKind::Fp8E5M2] {
            assert_eq!(raw_sign_masks(dt), Some((0x80, 0x7f)), "{dt:?}");
        }
        for dt in [ElementKind::F16, ElementKind::Bf16] {
            assert_eq!(raw_sign_masks(dt), Some((0x8000, 0x7fff)), "{dt:?}");
        }
        for dt in [ElementKind::F32, ElementKind::F64, ElementKind::I32] {
            assert_eq!(raw_sign_masks(dt), None, "{dt:?}");
        }
    }

    #[test]
    fn round_to_compute_rounds_through_the_compute_dtype_and_leaves_f64_alone() {
        // 0.1 is not an f32; the compute dtype's nearest value is what comes back.
        assert_eq!(round_to_compute(0.1, ElementKind::F32), f64::from(0.1f32));
        assert_eq!(
            round_to_compute(0.1, ElementKind::F32Strict),
            f64::from(0.1f32)
        );
        // Below f32's range the value underflows to zero; above f16's it overflows.
        assert_eq!(round_to_compute(1e-60, ElementKind::F32), 0.0);
        assert_eq!(round_to_compute(1e6, ElementKind::F16), f64::INFINITY);
        assert_eq!(round_to_compute(1.00390625, ElementKind::Bf16), 1.0);
        assert_eq!(round_to_compute(0.1, ElementKind::F64), 0.1);
        assert_eq!(round_to_compute(1e-60, ElementKind::F64), 1e-60);
    }

    #[test]
    fn unravel_is_last_axis_fastest() {
        assert_eq!(unravel(0, &[2, 3]), vec![0, 0]);
        assert_eq!(unravel(4, &[2, 3]), vec![1, 1]);
        assert_eq!(unravel(5, &[2, 3]), vec![1, 2]);
        assert_eq!(unravel(23, &[2, 3, 4]), vec![1, 2, 3]);
        assert_eq!(unravel(17, &[2, 3, 4]), vec![1, 1, 1]);
        assert_eq!(unravel(7, &[8]), vec![7]);
    }

    fn stored(v: Val, out: ElementKind) -> u128 {
        let mut buf = vec![0u8; elem_size(out)];
        store_val(v, out, &mut buf, 0);
        read_elem(&buf, 0, out)
    }

    #[test]
    fn a_raw_value_of_the_output_dtype_is_stored_verbatim() {
        // A signaling NaN's payload cannot survive the f64 detour, only a verbatim move.
        assert_eq!(
            stored(Val::Raw(0x7c01, ElementKind::F16), ElementKind::F16),
            0x7c01
        );
        assert_eq!(
            stored(Val::Raw(0xfd, ElementKind::Fp8E5M2), ElementKind::Fp8E5M2),
            0xfd
        );
    }

    #[test]
    fn a_raw_value_of_another_dtype_is_converted_not_moved() {
        // f32 1.0 into an f16 slot is 0x3c00, not the f32 pattern truncated.
        assert_eq!(
            stored(
                Val::Raw(u128::from(1.0f32.to_bits()), ElementKind::F32),
                ElementKind::F16
            ),
            0x3c00
        );
        // An integer raw value into an integer slot wraps to the slot's width.
        assert_eq!(stored(Val::Raw(300, ElementKind::I32), ElementKind::I8), 44);
        // A float raw value into an integer slot goes through the float->integer path.
        assert_eq!(
            stored(
                Val::Raw(u128::from(7.0f32.to_bits()), ElementKind::F32),
                ElementKind::I32
            ),
            7
        );
    }

    #[test]
    fn a_real_value_stored_to_a_complex_dtype_gets_a_zero_imaginary_part() {
        assert_eq!(
            stored(Val::Float(2.5), ElementKind::Complex64),
            u128::from(2.5f32.to_bits())
        );
        assert_eq!(
            stored(Val::Int(3), ElementKind::Complex128),
            u128::from(3.0f64.to_bits())
        );
        assert_eq!(
            stored(Val::Complex(1.0, -2.0), ElementKind::Complex64),
            encode_complex(1.0, -2.0, ElementKind::Complex64)
        );
    }

    fn buf(dt: ElementKind, bits: &[u16]) -> TypedBuffer {
        let bytes: Vec<u8> = if elem_size(dt) == 1 {
            bits.iter().map(|&b| b as u8).collect()
        } else {
            bits.iter().flat_map(|b| b.to_le_bytes()).collect()
        };
        TypedBuffer::new(dt, vec![bits.len() as i64], vec![1], bytes)
    }

    fn run_unary(dt: ElementKind, op: UnaryOp, bits: &[u16]) -> Vec<u16> {
        let e: Expr = input(0).unary(op);
        let o = OpDef::elementwise("u", 1, &[dt], e);
        let n = bits.len() as i64;
        let d = OperandDesc::new(1, &[n], &[1], dt, 256);
        let k = structure_key(OpCategory::UnaryElementwise, &[d, d], ArchSku::Sm89);
        let plan = build_plan(&o, &k);
        let out = evaluate(&plan, &[d, d], &[buf(dt, bits)], &[]);
        (0..bits.len()).map(|i| out[0].bits_at(i) as u16).collect()
    }

    #[test]
    fn neg_and_abs_edit_only_the_sign_bit_of_a_narrow_float() {
        // (dtype, 1.0, -1.0, NaN with a payload, the same NaN negated)
        let cases = [
            (ElementKind::Fp8E5M2, 0x3c_u16, 0xbc_u16, 0x7d_u16, 0xfd_u16),
            (ElementKind::F16, 0x3c00, 0xbc00, 0x7c01, 0xfc01),
            (ElementKind::Bf16, 0x3f80, 0xbf80, 0x7f81, 0xff81),
        ];
        for (dt, one, neg_one, nan, neg_nan) in cases {
            let ins = [one, neg_one, nan, neg_nan];
            assert_eq!(
                run_unary(dt, UnaryOp::Neg, &ins),
                vec![neg_one, one, neg_nan, nan],
                "neg {dt:?}"
            );
            assert_eq!(
                run_unary(dt, UnaryOp::Abs, &ins),
                vec![one, one, nan, nan],
                "abs {dt:?}"
            );
        }
    }

    #[test]
    fn select_tests_the_condition_in_the_compute_dtype() {
        // cond = in2 * in3; in f64 the product 1e-60 is nonzero, but in the compute dtype
        // (f32) it underflows to zero, so the emitter's cast-then-compare picks the else arm.
        let body = (input(2) * input(3)).select(input(0), input(1));
        let o = OpDef::elementwise("sel", 4, &[ElementKind::F32], body);
        let d = OperandDesc::new(1, &[5], &[1], ElementKind::F32, 256);
        let k = structure_key(OpCategory::BinaryElementwise, &[d; 5], ArchSku::Sm89);
        let plan = build_plan(&o, &k);
        let f = |v: &[f32]| TypedBuffer::from_f32(&[5], v);
        // conds: 1*1 (true), 0*1 (false), -0*1 (false), NaN*1 (true), 1e-30*1e-30 (false)
        let ins = [
            f(&[10.0; 5]),
            f(&[20.0; 5]),
            f(&[1.0, 0.0, -0.0, f32::NAN, 1e-30]),
            f(&[1.0, 1.0, 1.0, 1.0, 1e-30]),
        ];
        let out = evaluate(&plan, &[d; 5], &ins, &[]);
        let got: Vec<f32> = (0..5)
            .map(|i| f32::from_bits(out[0].bits_at(i) as u32))
            .collect();
        assert_eq!(got, vec![10.0, 20.0, 20.0, 10.0, 20.0]);
    }

    #[test]
    fn each_output_of_a_multi_output_op_gets_its_own_operand_layout() {
        // One broadcast (stride-0) input beside two dense outputs: output 1's layout must
        // come from its own descriptor, not from the input's.
        let o = OpDef::elementwise_multi(
            "two",
            1,
            &[ElementKind::F32],
            vec![input(0) + konst(1.0), input(0) * konst(2.0)],
        );
        let i = OperandDesc::new(1, &[3], &[0], ElementKind::F32, 256);
        let d = OperandDesc::new(1, &[3], &[1], ElementKind::F32, 256);
        let k = structure_key(OpCategory::BinaryElementwise, &[i, d, d], ArchSku::Sm89);
        let plan = build_plan(&o, &k);
        let ins = [TypedBuffer::new(
            ElementKind::F32,
            vec![3],
            vec![0],
            4.0f32.to_le_bytes().to_vec(),
        )];
        let out = evaluate(&plan, &[i, d, d], &ins, &[]);
        assert_eq!(out.len(), 2);
        for b in &out {
            assert_eq!(b.strides, vec![1]);
            assert_eq!(b.shape, vec![3]);
        }
        let f = |b: &TypedBuffer| -> Vec<f32> {
            (0..3)
                .map(|i| f32::from_bits(b.bits_at(i) as u32))
                .collect()
        };
        assert_eq!(f(&out[0]), vec![5.0; 3]);
        assert_eq!(f(&out[1]), vec![8.0; 3]);
    }
}

#[cfg(test)]
mod walk_pins {
    //! Pins from the 2026-10-18 mutation audit of this file: the index and layout
    //! arithmetic of the walkers (`eval_row_sort`, `eval_elementwise` gather/scatter,
    //! `eval_reduction`) and the dtype test in `required_fidelity`.
    use super::*;
    use crate::ir::{BinaryOp, OobPolicy, OpDef, ReduceOp, SortOrder, input};
    use crate::plan::build_plan;
    use unpopped_vocab::{ArchSku, AxisMask, OpCategory, structure_key};

    fn desc(shape: &[i64], strides: &[i64], dt: ElementKind) -> OperandDesc {
        OperandDesc::new(shape.len(), shape, strides, dt, 256)
    }

    fn f32s(b: &TypedBuffer) -> Vec<f32> {
        (0..b.len())
            .map(|i| f32::from_bits(b.bits_at(i) as u32))
            .collect()
    }

    fn i32s(b: &TypedBuffer) -> Vec<i32> {
        (0..b.len()).map(|i| b.bits_at(i) as u32 as i32).collect()
    }

    #[test]
    fn a_top_k_sort_with_both_outputs_sizes_each_output_from_its_own_operand() {
        // Two rows of five, keeping two: each output row is two wide, so the second row
        // starts at element 2 of BOTH outputs. Sizing the index output from the input
        // operand (width 5) would put row 1 at element 5.
        let ind = desc(&[2, 5], &[5, 1], ElementKind::F32);
        let vals = desc(&[2, 2], &[2, 1], ElementKind::F32);
        let idxs = desc(&[2, 2], &[2, 1], ElementKind::I32);
        let op = OpDef::row_sort_indices("both", ElementKind::F32, SortOrder::Desc);
        let k = structure_key(OpCategory::Sorting, &[ind, vals, idxs], ArchSku::Sm89);
        let out = evaluate(
            &build_plan(&op, &k),
            &[ind, vals, idxs],
            &[TypedBuffer::from_f32(
                &[2, 5],
                &[3.0, 1.0, 4.0, 1.5, 9.0, 2.0, 8.0, 6.0, 7.0, 5.0],
            )],
            &[],
        );
        assert_eq!(out.len(), 2);
        assert_eq!(f32s(&out[0]), vec![9.0, 4.0, 8.0, 7.0]);
        assert_eq!(i32s(&out[1]), vec![4, 2, 1, 3]);
    }

    #[test]
    fn a_skipping_gather_writes_nothing_for_an_index_outside_the_axis() {
        // Extent 4: 3 and 0 are the in-range ends, -1 and 4 are the first indices outside.
        let d = desc(&[1, 4], &[4, 1], ElementKind::F32);
        let i = desc(&[1, 4], &[4, 1], ElementKind::I32);
        let op = OpDef::gather(
            "gs",
            &[ElementKind::F32],
            1,
            OobPolicy::Skip,
            ElementKind::I32,
        );
        let k = structure_key(OpCategory::UnaryElementwise, &[d, i, d], ArchSku::Sm89);
        let out = evaluate(
            &build_plan(&op, &k),
            &[d, i, d],
            &[
                TypedBuffer::from_f32(&[1, 4], &[10.0, 20.0, 30.0, 40.0]),
                TypedBuffer::new(
                    ElementKind::I32,
                    vec![1, 4],
                    vec![4, 1],
                    [3i32, -1, 4, 0]
                        .iter()
                        .flat_map(|v| v.to_le_bytes())
                        .collect(),
                ),
            ],
            &[],
        );
        assert_eq!(f32s(&out[0]), vec![40.0, 0.0, 0.0, 10.0]);
    }

    #[test]
    fn a_scatter_walks_the_value_operand_even_when_the_index_is_longer() {
        // The value operand (slot 0) has three elements; the index operand has four. The walk
        // is over the value operand's extent, so the fourth index is never read.
        let src = desc(&[1, 3], &[3, 1], ElementKind::I32);
        let idx = desc(&[1, 4], &[4, 1], ElementKind::I32);
        let dst = desc(&[1, 2], &[2, 1], ElementKind::I32);
        let op = OpDef::scatter_add("sl", &[ElementKind::I32], 1, ElementKind::I32);
        let k = structure_key(
            OpCategory::UnaryElementwise,
            &[src, idx, dst],
            ArchSku::Sm89,
        );
        let ib = |shape: &[i64], v: &[i32]| {
            TypedBuffer::new(
                ElementKind::I32,
                shape.to_vec(),
                vec![shape[1], 1],
                v.iter().flat_map(|x| x.to_le_bytes()).collect(),
            )
        };
        let out = evaluate(
            &build_plan(&op, &k),
            &[src, idx, dst],
            &[ib(&[1, 3], &[5, 6, 7]), ib(&[1, 4], &[0, 1, 0, 1])],
            &[],
        );
        assert_eq!(i32s(&out[0]), vec![12, 6]);
    }

    fn sum_over_middle(keepdim: bool) -> Vec<f32> {
        // [2,3,2] summed over axis 1: out[a][c] = sum_b data[a][b][c].
        let ind = desc(&[2, 3, 2], &[6, 2, 1], ElementKind::F32);
        let outd = if keepdim {
            desc(&[2, 1, 2], &[2, 2, 1], ElementKind::F32)
        } else {
            desc(&[2, 2], &[2, 1], ElementKind::F32)
        };
        let op = OpDef::reduction_axes(
            "mid",
            1,
            &[ElementKind::F32],
            input(0),
            ReduceOp::Sum,
            AxisMask(0b010),
            keepdim,
        );
        let k = structure_key(OpCategory::Reduction, &[ind, outd], ArchSku::Sm89);
        let data: Vec<f32> = (1..=12).map(|v| v as f32).collect();
        let out = evaluate(
            &build_plan(&op, &k),
            &[ind, outd],
            &[TypedBuffer::from_f32(&[2, 3, 2], &data)],
            &[],
        );
        f32s(&out[0])
    }

    #[test]
    fn a_middle_axis_reduction_places_each_result_by_the_output_strides() {
        let want = vec![9.0, 12.0, 27.0, 30.0];
        assert_eq!(sum_over_middle(false), want, "collapsed");
        assert_eq!(sum_over_middle(true), want, "keepdim");
    }

    #[test]
    fn a_mask_output_beside_a_float_value_is_not_an_integer_cell() {
        // The key dtype is f32 and output 0 is a u8 predicate mask: only one side is
        // integer, so the cell is tolerance-compared, not bit-exact.
        let op = OpDef::elementwise_multi_hetero(
            "mask",
            2,
            &[ElementKind::F32],
            vec![
                (
                    input(0).binary(BinaryOp::CmpLt, input(1)),
                    Some(ElementKind::U8),
                ),
                (input(0) + input(1), None),
            ],
        );
        let f = desc(&[4], &[1], ElementKind::F32);
        let u = desc(&[4], &[1], ElementKind::U8);
        let ops = [f, f, u, f];
        let k = structure_key(OpCategory::BinaryElementwise, &ops, ArchSku::Sm89);
        let plan = build_plan(&op, &k);
        assert!(
            matches!(
                required_fidelity(&plan, &ops),
                Some(Fidelity::Tolerant { .. })
            ),
            "{:?}",
            required_fidelity(&plan, &ops)
        );
    }
}

/// **FIX (0.40.10).** `Max`, `Min` and the comparisons at an INTEGER compute dtype went through `f64`,
/// which holds an integer exactly only up to 2^53. At `I64`/`U64`, `max(2^53 + 1, 0)` came out `2^53`
/// and `i64::MAX == i64::MAX - 1` came out true. The reference here is the host's own integer
/// operators, which share no code with the oracle.
#[cfg(test)]
mod int64_exactness_tests {
    use super::{Eval, Val, eval, panic_reduced};
    use crate::ir::{BinaryOp, ScalarExpr};
    use unpopped_vocab::ElementKind;

    const P: i128 = (1 << 53) + 1;

    /// The oracle's value of `a <op> b` at `dtype`, as an integer.
    fn oracle(dtype: ElementKind, op: BinaryOp, a: i128, b: i128) -> i128 {
        let leaf = move |k: u8| Val::Int(if k == 0 { a } else { b });
        let coord = |_: u8| Val::Float(0.0);
        let ev = Eval {
            dtype,
            params: &[],
            leaf: &leaf,
            reduced: &panic_reduced,
            coord: &coord,
        };
        let e = ScalarExpr::Binary(
            op,
            Box::new(ScalarExpr::Input(0)),
            Box::new(ScalarExpr::Input(1)),
        );
        match eval(&e, &ev) {
            Val::Int(v) => v,
            Val::Float(f) => f as i128,
            other => panic!("unexpected {other:?}"),
        }
    }

    fn host(op: BinaryOp, a: i128, b: i128) -> i128 {
        match op {
            BinaryOp::Max => a.max(b),
            BinaryOp::Min => a.min(b),
            BinaryOp::CmpEq => i128::from(a == b),
            BinaryOp::CmpNe => i128::from(a != b),
            BinaryOp::CmpLt => i128::from(a < b),
            BinaryOp::CmpLe => i128::from(a <= b),
            BinaryOp::CmpGt => i128::from(a > b),
            _ => i128::from(a >= b),
        }
    }

    #[test]
    fn integer_max_min_and_comparisons_are_exact_above_2_pow_53() {
        let i64s = [P, i128::from(i64::MAX), i128::from(i64::MAX - 1), -P, 0];
        let u64s = [P, i128::from(u64::MAX), i128::from(u64::MAX - 1), 0];
        let ops = [
            BinaryOp::Max,
            BinaryOp::Min,
            BinaryOp::CmpEq,
            BinaryOp::CmpNe,
            BinaryOp::CmpLt,
            BinaryOp::CmpLe,
            BinaryOp::CmpGt,
            BinaryOp::CmpGe,
        ];
        for (dt, vals) in [(ElementKind::I64, &i64s[..]), (ElementKind::U64, &u64s[..])] {
            for op in ops {
                for &a in vals {
                    for &b in vals {
                        let (got, want) = (oracle(dt, op, a, b), host(op, a, b));
                        assert_eq!(got, want, "{op:?} at {dt:?}: a = {a}, b = {b}");
                    }
                }
            }
        }
    }
}
