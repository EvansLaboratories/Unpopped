// SPDX-License-Identifier: MIT OR Apache-2.0
//! **U19 slice 4c-3: the `asm volatile` template table** (`docs/options-u19-slice5-decode-census.md`). Each recognised PTX
//! template lifts to the IR form that means the same, with its approximation stated (`Numerics { approx, ftz }`); anything else
//! declines with the template text. Structure is asserted on the lifted kernel (an approximation must not lift as the accurate
//! operation) and the value on the oracle.

use unpopped::exec::LaunchShape;
use unpopped::ir::{Numerics, RoundMode, ScalarExpr, UnaryOp};
use unpopped::kernel::{CollectiveKind, KernelFn, ShuffleMode, Stmt};
use unpopped::oracle::TypedBuffer;
use unpopped::oracle::kernel_eval::{KernelArg, run_kernel};
use unpopped::oracle::kernel_launch::run_launch_with_subgroup;
use unpopped_convert::{Frontend, KernelLiftError, lift_kernel, lift_kernel_launched};

fn parse(src: &str) -> Option<tree_sitter::Tree> {
    let mut p = tree_sitter::Parser::new();
    p.set_language(&tree_sitter_cpp::LANGUAGE.into()).ok()?;
    p.parse(src, None)
}

const C: Frontend = Frontend::new("c", "void", &[], "out", "in", parse);

fn lift(src: &str) -> Result<KernelFn, KernelLiftError> {
    lift_kernel(&C, src, "k")
}

fn f32s(v: &[f32]) -> KernelArg {
    KernelArg::Buffer(TypedBuffer::from_f32(&[v.len() as i64], v))
}

fn out_of(a: &KernelArg) -> Vec<f64> {
    let KernelArg::Buffer(o) = a else { panic!() };
    o.to_f64_vec()
}

/// The qualified node of the first `Assign` in the kernel.
fn qualified(k: &KernelFn) -> (UnaryOp, Numerics) {
    for s in &k.body.0 {
        if let Stmt::Assign {
            value: ScalarExpr::Qualified(inner, n),
            ..
        } = s
        {
            if let ScalarExpr::Unary(op, _) = **inner {
                return (op, *n);
            }
        }
    }
    panic!("no qualified unary assignment in {:?}", k.body);
}

#[test]
fn ex2_lg2_rcp_lift_as_qualified_approximations() {
    let src = |t: &str| {
        format!(
            "void k(float* out, const float* a) {{ float y; asm volatile(\"{t} %0, %1;\" : \"=f\"(y) : \"f\"(a[0])); out[0] = y; }}"
        )
    };
    let cases = [
        ("ex2.approx.ftz.f32", UnaryOp::Exp2, true),
        ("ex2.approx.f32", UnaryOp::Exp2, false),
        ("lg2.approx.ftz.f32", UnaryOp::Log2, true),
        ("rcp.approx.ftz.f32", UnaryOp::Recip, true),
    ];
    for (tmpl, op, ftz) in cases {
        let k = lift(&src(tmpl)).unwrap_or_else(|e| panic!("{tmpl}: {e}"));
        let (got_op, n) = qualified(&k);
        assert_eq!(got_op, op, "{tmpl}");
        assert_eq!(
            n,
            Numerics {
                round: RoundMode::NearestEven,
                ftz,
                approx: true
            },
            "{tmpl}"
        );
    }
}

#[test]
fn the_approximations_compute_within_their_stated_error() {
    // 2^x, log2 x, 1/x at x = 3.5: the oracle's value for an `approx` node is within 4 ulp of the exact result
    let run = |t: &str, x: f32| {
        let k = lift(&format!("void k(float* out, const float* a) {{ float y; asm volatile(\"{t} %0, %1;\" : \"=f\"(y) : \"f\"(a[0])); out[0] = y; }}")).unwrap();
        let mut args = vec![f32s(&[0.0]), f32s(&[x])];
        run_kernel(&k, &mut args).expect("runs");
        out_of(&args[0])[0]
    };
    for (t, x, want) in [
        ("ex2.approx.ftz.f32", 3.5f32, 3.5f32.exp2()),
        ("lg2.approx.ftz.f32", 3.5, 3.5f32.log2()),
        ("rcp.approx.ftz.f32", 3.5, 1.0 / 3.5),
    ] {
        let got = run(t, x) as f32;
        let ulp = want.abs() * f32::EPSILON;
        assert!((got - want).abs() <= 4.0 * ulp, "{t}: {got} vs {want}");
    }
}

#[test]
fn an_xor_shuffle_exchanges_lanes() {
    let k = lift_kernel_launched(
        &C,
        "void k(float* out) { float y; float x = (float)threadIdx.x; asm volatile(\"shfl.sync.bfly.b32 %0, %1, %2, 0x1f, 0xffffffff;\" : \"=f\"(y) : \"f\"(x), \"r\"(1)); out[threadIdx.x] = y; }",
        "k",
        LaunchShape::new([8, 1, 1], [1, 1, 1]).with_subgroup(1, 32),
    )
    .unwrap();
    assert!(k.body.0.iter().any(|s| matches!(
        s,
        Stmt::Collective {
            kind: CollectiveKind::Shuffle {
                mode: ShuffleMode::Xor,
                ..
            },
            ..
        }
    )));
    let mut args = vec![f32s(&[0.0; 8])];
    run_launch_with_subgroup(&k, &mut args, Some(4)).expect("runs");
    // lane i receives lane i^1's value
    assert_eq!(
        out_of(&args[0]),
        vec![1.0, 0.0, 3.0, 2.0, 5.0, 4.0, 7.0, 6.0]
    );
}

#[test]
fn async_group_and_grid_dependency_statements() {
    let k = lift_kernel_launched(
        &C,
        "void k(float* out) { asm volatile(\"cp.async.commit_group;\n\" ::); asm volatile(\"cp.async.wait_group %0;\n\" ::\"n\"(3)); \
         asm volatile(\"griddepcontrol.wait;\"); asm volatile(\"griddepcontrol.launch_dependents;\"); out[0] = 1.0f; }",
        "k",
        LaunchShape::new([1, 1, 1], [1, 1, 1]),
    )
    .unwrap();
    let kinds: Vec<_> = k
        .body
        .0
        .iter()
        .map(|s| match s {
            Stmt::AsyncCommit => "commit".to_string(),
            Stmt::AsyncWait { pending } => format!("wait{pending}"),
            Stmt::GridDepWait => "dep-wait".to_string(),
            Stmt::GridDepLaunchDependents => "dep-launch".to_string(),
            _ => "other".to_string(),
        })
        .collect();
    assert_eq!(
        kinds,
        ["commit", "wait3", "dep-wait", "dep-launch", "other"]
    );
}

fn not_yet(src: &str, needle: &str) {
    match lift(src) {
        Err(KernelLiftError::NotYetLifted { construct, .. }) => {
            assert!(construct.contains(needle), "`{construct}` lacks `{needle}`");
        }
        other => panic!("expected NotYetLifted({needle}), got {other:?}"),
    }
}

#[test]
fn an_unknown_template_or_a_partial_form_declines_with_its_text() {
    let one = |t: &str| {
        format!(
            "void k(float* out, const float* a) {{ float y; asm volatile(\"{t}\" : \"=f\"(y) : \"f\"(a[0])); out[0] = y; }}"
        )
    };
    not_yet(&one("sin.approx.f32 %0, %1;"), "sin.approx.f32");
    not_yet(&one("ex2.approx.f32 %0, %1, %2;"), "ex2.approx.f32");
    not_yet(&one("ex2.approx.f64 %0, %1;"), "ex2.approx.f64");
    // a shuffle with a partial member mask, and a wait_group whose count is not a literal
    not_yet(
        "void k(float* out, float x) { float y; asm volatile(\"shfl.sync.bfly.b32 %0, %1, %2, 0x1f, 0xffff;\" : \"=f\"(y) : \"f\"(x), \"r\"(1)); out[0] = y; }",
        "shfl.sync.bfly",
    );
    not_yet(
        "void k(float* out, int n) { asm volatile(\"cp.async.wait_group %0;\" :: \"n\"(n)); out[0] = 1.0f; }",
        "wait_group",
    );
    // a width other than the full 32 (`0x1f`) is a segmented shuffle: not this table's form
    not_yet(
        "void k(float* out, float x) { float y; asm volatile(\"shfl.sync.bfly.b32 %0, %1, %2, 0xf, 0xffffffff;\" : \"=f\"(y) : \"f\"(x), \"r\"(1)); out[0] = y; }",
        "shfl.sync.bfly",
    );
    // Control: the table's own spelling lifts.
    assert!(lift(&one("ex2.approx.ftz.f32 %0, %1;")).is_ok());
}

#[test]
fn an_uninitialised_scalar_starts_at_zero() {
    let k = lift("void k(float* out) { float y; int n; out[0] = y; out[1] = (float)n; }").unwrap();
    let mut args = vec![f32s(&[9.0, 9.0])];
    run_kernel(&k, &mut args).expect("runs");
    assert_eq!(out_of(&args[0]), vec![0.0, 0.0]);
}
