// SPDX-License-Identifier: MIT OR Apache-2.0
//! **U19 slice 4a: `#pragma unroll` and `__ldg`** (`docs/options-u19-cfamily-body-frontend.md`). An unroll pragma is a request
//! to the emitter, never a meaning: it lands in `Stmt::For::unroll` and the loop computes the same either way. `__ldg(&a[i])`
//! is a plain read (the read-only cache path is a hint).

use unpopped::kernel::{KernelFn, Stmt, Unroll};
use unpopped::oracle::TypedBuffer;
use unpopped::oracle::kernel_eval::{KernelArg, run_kernel};
use unpopped_convert::{Frontend, KernelLiftError, lift_kernel};

fn parse(src: &str) -> Option<tree_sitter::Tree> {
    let mut p = tree_sitter::Parser::new();
    p.set_language(&tree_sitter_cpp::LANGUAGE.into()).ok()?;
    p.parse(src, None)
}

const C: Frontend = Frontend::new("c", "void", &[], "out", "in", parse);

fn lift(src: &str) -> Result<KernelFn, KernelLiftError> {
    lift_kernel(&C, src, "k")
}

/// The `unroll` hint of every `For` in the kernel, in source order.
fn hints(k: &KernelFn) -> Vec<Unroll> {
    fn walk(b: &[Stmt], out: &mut Vec<Unroll>) {
        for s in b {
            match s {
                Stmt::For { unroll, body, .. } => {
                    out.push(*unroll);
                    walk(&body.0, out);
                }
                Stmt::If { then, els, .. } => {
                    walk(&then.0, out);
                    walk(&els.0, out);
                }
                Stmt::While { body, .. } => walk(&body.0, out),
                _ => {}
            }
        }
    }
    let mut v = Vec::new();
    walk(&k.body.0, &mut v);
    v
}

fn kernel(pragma: &str) -> String {
    format!(
        "void k(float* out, const float* a) {{\n{pragma}\nfor (int i = 0; i < 4; i++) {{ out[i] = __ldg(&a[i]) * 2.0f; }} }}"
    )
}

#[test]
fn an_unroll_pragma_binds_the_next_loop() {
    for (pragma, want) in [
        ("#pragma unroll", Unroll::Full),
        ("#pragma unroll 4", Unroll::Count(4)),
        ("#pragma unroll 1", Unroll::None),
        ("", Unroll::None),
    ] {
        let k = lift(&kernel(pragma)).unwrap();
        assert_eq!(hints(&k), vec![want], "`{pragma}`");
    }
}

#[test]
fn a_pragma_binds_only_the_statement_right_after_it() {
    let k = lift(
        "void k(float* out, const float* a) {\n#pragma unroll\nout[9] = 1.0f;\nfor (int i = 0; i < 4; i++) { out[i] = a[i]; }\n\
         #pragma unroll 2\nfor (int j = 0; j < 4; j++) { out[j] = a[j]; } }",
    )
    .unwrap();
    // the first loop is not the statement after the first pragma, so it has no hint; the second has its own
    assert_eq!(hints(&k), vec![Unroll::None, Unroll::Count(2)]);
}

#[test]
fn the_hint_changes_no_result_and_ldg_is_a_plain_read() {
    let a = [1.5f32, -2.0, 3.25, 0.5];
    let run = |pragma: &str| {
        let k = lift(&kernel(pragma)).unwrap();
        let mut args = vec![
            KernelArg::Buffer(TypedBuffer::from_f32(&[4], &[0.0; 4])),
            KernelArg::Buffer(TypedBuffer::from_f32(&[4], &a)),
        ];
        run_kernel(&k, &mut args).expect("runs");
        let KernelArg::Buffer(o) = &args[0] else {
            panic!()
        };
        o.to_f64_vec()
    };
    let want: Vec<f64> = a.iter().map(|&v| f64::from(v * 2.0)).collect();
    assert_eq!(run(""), want);
    assert_eq!(run("#pragma unroll"), want);
    assert_eq!(run("#pragma unroll 4"), want);
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
fn other_directives_and_bad_counts_decline() {
    not_yet(&kernel("#pragma omp parallel"), "#pragma omp parallel");
    not_yet(&kernel("#pragma unroll 0"), "count");
    not_yet(&kernel("#pragma unroll x"), "count");
    not_yet(&kernel("#pragma unroll 2 3"), "trailing");
    not_yet(
        "void k(float* out, const float* a) { out[0] = __ldg(a + 1); }",
        "address is not",
    );
    // Control: the accepted spelling lifts.
    assert!(lift(&kernel("#pragma unroll 2")).is_ok());
}
