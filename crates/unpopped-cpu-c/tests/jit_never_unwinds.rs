//! The JIT trust boundary must **decline**, never unwind.
//!
//! `jit.rs` states the rule outright: *"the Synthesizer trait must never unwind
//! across the boundary"*. Fuel calls `synthesize` with a region it chose, so a
//! request the generator cannot serve has to come back as a typed `JitError` —
//! a panic crossing an FFI-shaped seam is not a decline, it is a crash in the
//! caller's process.
//!
//! # The bug this pins, and why nothing caught it
//!
//! The guarantee was enforced by a **second legality table**: `dtype_compatible`,
//! which pre-screened the request before `build_plan`. It covered *dtype and op*
//! legality — is this op legal at this dtype — and **not schedule legality**.
//!
//! Nothing in the pipeline asked whether the backend could emit the *cell the
//! planner chose*. So an entirely ordinary request — a `u8` `Add` over 256
//! contiguous, 256-byte-aligned elements — planned to `Vectorized { width: 8 }`,
//! which the CpuC v1 emitter does not serve, and the refusal panicked straight
//! through `synthesize` into the caller.
//!
//! It was reachable with the simplest region anyone would write. It survived
//! because the pre-screen and the emitter's limits were two different tables and
//! nothing compared them.
//!
//! # A note on how this test was arrived at
//!
//! The first version of this file probed a `u8` shift with a composed operand,
//! aimed at the plan gate's 8-bit composition pin. It passed — vacuously.
//! `Shr`/`BitAnd` are not in the JIT's region vocabulary at all, so those
//! requests died at `UnsupportedOp` long before reaching the gate, and the test
//! would equally have passed against a `synthesize` that panicked on everything
//! else. Printing what each request *actually* returned is what surfaced the real
//! defect, which was both simpler and worse than the one being hunted.

use unpopped::backend::LowerError;
use unpopped::jit::{JitBudget, JitError, JitRequest, StubCompiler, synthesize};
use unpopped::pattern::PatternNode;
use unpopped_cpu_c::CpuC;
use unpopped_vocab::{ArchSku, ElementKind, OpCategory, OperandDesc, TargetId};

fn bind(i: u8) -> PatternNode {
    PatternNode::Bind(i)
}

fn op(name: &str, operands: Vec<PatternNode>) -> PatternNode {
    PatternNode::Op {
        op: name.to_string(),
        operands,
        consumers: None,
        extract: Vec::new(),
    }
}

/// `n` elements, 256-byte aligned — the alignment is what lets the planner
/// vectorize, so it is load-bearing here rather than incidental.
fn request(region: PatternNode, n_inputs: u8, dtype: ElementKind, n: i64) -> JitRequest {
    let d = OperandDesc::new(1, &[n], &[1], dtype, 256);
    JitRequest {
        region,
        n_inputs,
        op_category: OpCategory::BinaryElementwise,
        operands: vec![d; usize::from(n_inputs) + 1],
        // `ArchSku` still converts, so a CUDA caller's migration is `.into()`.
        target: ArchSku::Sm89.into(),
        fused_op_id: "probe".to_string(),
        budget: JitBudget {
            max_compile_ms: 1000,
        },
    }
}

/// Run `synthesize`, turning a panic into a failure that names what broke.
fn synth_or_fail(req: &JitRequest, what: &str) -> Result<(), JitError> {
    std::panic::catch_unwind(|| synthesize(req, &CpuC, &StubCompiler))
        .unwrap_or_else(|_| {
            panic!(
                "PANICKED across the JIT trust boundary on {what}. jit.rs requires the \
                 Synthesizer never to unwind: a request the generator cannot serve must \
                 return a typed JitError, because this seam faces another process's \
                 caller. Use `try_generate`, not `generate`, in `synthesize`."
            )
        })
        .map(|_| ())
}

/// **The regression test.** A cell the planner vectorizes past what the backend
/// emits is a typed decline, not a panic.
#[test]
fn a_schedule_the_backend_cannot_emit_declines_rather_than_panicking() {
    // u8 over 256 aligned elements -> Vectorized { width: 8 }; CpuC v1 is scalar-only.
    let req = request(op("Add", vec![bind(0), bind(1)]), 2, ElementKind::U8, 256);

    let err = synth_or_fail(&req, "a vectorized u8 Add")
        .expect_err("CpuC v1 cannot emit a vectorized cell, so this must not synthesize");

    // Specificity matters — this enum's own doc calls it load-bearing. The caller
    // must be able to tell "wrong schedule for this backend" (re-request another
    // cell) from "this dtype is unsupported" (give up on the dtype).
    match err {
        JitError::BackendDeclined(LowerError::UnsupportedSchedule { ref detail }) => {
            assert!(
                detail.contains("Vectorized"),
                "the decline should name the schedule it could not emit, got {detail:?}"
            );
        }
        other => panic!(
            "expected a schedule-specific decline so the caller can re-request a \
             different cell; got {other:?}"
        ),
    }
}

/// The positive control.
///
/// Without it, the test above is satisfied by a `synthesize` that declines
/// everything. `n = 7` is not vectorizable, so the same op at the same dtype must
/// go through — proving the decline above is about the *schedule*, and not about
/// u8, or `Add`, or a JIT that refuses all work.
#[test]
fn the_same_op_and_dtype_synthesizes_when_the_cell_is_scalar() {
    let req = request(op("Add", vec![bind(0), bind(1)]), 2, ElementKind::U8, 7);

    synth_or_fail(&req, "a scalar u8 Add").expect(
        "a scalar-cell u8 Add is exactly what CpuC v1 serves; if this declines, the \
         test above proves nothing — it would be measuring a JIT that refuses \
         everything rather than one that refuses the vectorized cell",
    );
}

/// The op-vocabulary decline stays intact and stays *distinct*.
///
/// This is the arm that made the first version of this file vacuous: `Shr` never
/// reaches the plan gate or the backend. Pinning it separately keeps that visible,
/// so a future reader does not mistake an `UnsupportedOp` for evidence about what
/// the backend can lower.
#[test]
fn an_unknown_region_op_declines_before_lowering_and_says_so() {
    let req = request(op("Shr", vec![bind(0), bind(1)]), 2, ElementKind::U8, 7);

    let err =
        synth_or_fail(&req, "an unknown region op").expect_err("Shr is not in the vocabulary");

    assert!(
        matches!(err, JitError::UnsupportedOp(ref s) if s == "Shr"),
        "an unknown op must decline as UnsupportedOp before any lowering is \
         attempted — it says nothing about what the backend can emit; got {err:?}"
    );
}

/// **A JIT request can now name a non-CUDA target — this test could not be
/// written before `JitRequest::target` existed.**
///
/// # What was actually broken
///
/// The field was `arch: ArchSku`, a closed four-variant CUDA enum. The *derived
/// key* had been target-neutral since `structure_key` began taking a `TargetId`
/// — `jit.rs` converts before keying, so cached-artifact identity was already
/// correct — but **the request path could not express a non-CUDA target at all.**
/// A Vulkane JIT request had nowhere to put one. That asymmetry is why this was
/// an API-expressiveness gap rather than a wire or cache-soundness one, and why
/// it could land after `0.2.0` instead of blocking it.
///
/// # What this asserts, and what it deliberately does not
///
/// It asserts the request is **expressible and reaches the synthesizer**, and
/// that whatever comes back is a typed outcome rather than an unwind. It does
/// **not** assert that CpuC serves a `vulkan:` cell — CpuC is a C99 emitter and
/// has no opinion about Vulkan capability sets. Asserting success would be
/// testing a thing that is not true; asserting a specific decline would pin
/// CpuC's incidental behaviour on a target it does not model.
///
/// The load-bearing part is that **the outcome is decided by the synthesizer
/// rather than by the type system refusing to hold the request.** Before this
/// change the failure was `error[E0560]` at the call site.
#[test]
fn a_jit_request_can_name_a_non_cuda_target() {
    let target = TargetId::parse("vulkan:st16")
        .expect("`vulkan:st16` is a well-formed KISS §6.8 token — namespace, colon, capability");

    // Same shape as `request()`, with a target no `ArchSku` can spell.
    let d = OperandDesc::new(1, &[256], &[1], ElementKind::F32, 256);
    let req = JitRequest {
        region: op("Add", vec![bind(0), bind(1)]),
        n_inputs: 2,
        op_category: OpCategory::BinaryElementwise,
        operands: vec![d; 3],
        target,
        fused_op_id: "non_cuda_probe".to_string(),
        budget: JitBudget {
            max_compile_ms: 1000,
        },
    };

    assert_eq!(
        req.target.as_str(),
        "vulkan:st16",
        "the request must carry the target verbatim — carry, do not interpret"
    );
    assert_ne!(
        req.target.namespace(),
        "cuda",
        "if this were a cuda: token the test would prove nothing"
    );

    // Reaches the synthesizer and returns a typed outcome either way. The
    // `catch_unwind` is the point: a target CpuC does not model must not unwind
    // across a seam that faces another process's caller.
    let _outcome = synth_or_fail(&req, "an Add on a non-CUDA target");
}
