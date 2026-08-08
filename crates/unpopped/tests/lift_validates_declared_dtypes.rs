//! A lifted op's declared dtype list is an unvalidated caller input — until now.
//!
//! `lift_elementwise(src, name, dtypes)` takes the op's identity and its accepted
//! dtypes as *parameters*: the lifter recovers the body from source, but it
//! cannot know what the op should be called or which dtypes it ought to serve.
//! Those are the caller's assertions.
//!
//! It passed them straight into the `OpDef` unchecked. A caller could lift a
//! float-only body (`expf`) and declare it as accepting `I32`, producing a stored
//! op that claims a dtype it cannot honour.
//!
//! # Why this matters more for a catalog than for a one-shot call
//!
//! An op is lifted **once** and generated **on demand**, so an unhonourable
//! declaration is not a transient error — it is a broken catalog entry, and the
//! failure surfaces to whoever first requests that cell rather than to whoever
//! stored it. Worse, it degrades quietly in at least one direction: a mixed
//! int/float dtype list makes `recipe.rs` decline to emit a recipe at all, so the
//! entry silently loses its `pattern:`/`decompose:` halves.
//!
//! Validating at intake moves the error to the party that can fix it, holding the
//! information needed to fix it.

use unpopped::lift::{LiftError, lift_elementwise};
use unpopped_vocab::ElementKind;

/// A grid-stride elementwise kernel whose body is float-only (`expf`).
const EXP_SRC: &str = r#"
extern "C" __global__ void k(const float* in0, float* out, long long n) {
    for (long long i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
        out[i] = expf(in0[i]);
    }
}
"#;

/// A body that IS admissible at an integer dtype — plain infix arithmetic.
const ADD_SRC: &str = r#"
extern "C" __global__ void k(const int* in0, const int* in1, int* out, long long n) {
    for (long long i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
        out[i] = in0[i] + in1[i];
    }
}
"#;

/// The claim: a declared dtype the lifted body cannot honour is refused.
#[test]
fn a_declared_dtype_the_body_cannot_honour_is_refused() {
    let err = lift_elementwise(EXP_SRC, "expop", &[ElementKind::F32, ElementKind::I32])
        .expect_err("expf has no integer lowering, so declaring I32 must not be accepted");

    match err {
        LiftError::DtypeNotAdmissible { dtype, ref detail } => {
            assert_eq!(
                dtype,
                ElementKind::I32,
                "the refusal must name the OFFENDING dtype, not the first one"
            );
            assert!(
                detail.contains("integer"),
                "the refusal should carry the gate's reason so the caller can act on it, got {detail:?}"
            );
        }
        other => panic!(
            "expected a dtype-specific refusal the caller can act on (drop the dtype \
             and retry) rather than a generic one; got {other:?}"
        ),
    }
}

/// Positive control #1: the same body with an honourable dtype list still lifts.
///
/// Without this, the test above is satisfied by a lifter that refuses everything.
#[test]
fn the_same_body_lifts_when_the_declared_dtypes_are_honourable() {
    let lifted = lift_elementwise(EXP_SRC, "expop", &[ElementKind::F32])
        .expect("expf at f32 is exactly what this lifter is for");
    assert_eq!(lifted.op.dtypes, vec![ElementKind::F32]);
    assert_eq!(lifted.n_inputs, 1);
}

/// Positive control #2: the gate is not simply rejecting every integer dtype.
///
/// Plain infix `+` IS admissible at `I32` (the plan gate admits wrapping
/// Add/Sub/Mul at integer dtypes). If this failed, the check above would be
/// measuring "integers are refused" rather than "unhonourable declarations are
/// refused" — a much weaker and wrong property.
#[test]
fn an_integer_dtype_is_accepted_when_the_body_can_honour_it() {
    let lifted = lift_elementwise(ADD_SRC, "addop", &[ElementKind::I32])
        .expect("infix add is admissible at I32 — wrapping integer arithmetic");
    assert_eq!(lifted.op.dtypes, vec![ElementKind::I32]);
}
