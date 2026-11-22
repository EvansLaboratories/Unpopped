// SPDX-License-Identifier: MIT OR Apache-2.0
//! **U19 slice 5 groundwork: what the real decode kernel needed** (`docs/options-u19-slice5-decode-census.md`). The `LiftContext`
//! (launch, dynamic shared bytes, macro definitions, null buffers) and the preprocessor pass; file-scope `constexpr` constants,
//! `sizeof` and constant-sized arrays; struct pointer fields built by a brace list; a `bool` that is a normalised byte; integer
//! `min`/`max`; a null check on an optional buffer; vectors read and written through a pointer. Each lifted kernel runs on the
//! oracle against a Rust reference; each decline names the construct.

use unpopped::exec::LaunchShape;
use unpopped::kernel::KernelFn;
use unpopped::oracle::TypedBuffer;
use unpopped::oracle::kernel_eval::{KernelArg, KernelTrap};
use unpopped::oracle::kernel_launch::run_launch;
use unpopped_convert::{Frontend, KernelLiftError, LiftContext, lift_kernel_in};

fn parse(src: &str) -> Option<tree_sitter::Tree> {
    let mut b = src.to_string();
    for q in [
        "__host__",
        "__device__",
        "__forceinline__",
        "__grid_constant__",
        "__shared__",
    ] {
        b = b.replace(q, &" ".repeat(q.len()));
    }
    let mut p = tree_sitter::Parser::new();
    p.set_language(&tree_sitter_cpp::LANGUAGE.into()).ok()?;
    p.parse(b, None)
}

const C: Frontend = Frontend::new("c", "void", &[], "out", "in", parse);

fn ctx() -> LiftContext {
    LiftContext::new(LaunchShape::new([1, 1, 1], [1, 1, 1]))
}

fn lift(src: &str, c: &LiftContext) -> Result<KernelFn, KernelLiftError> {
    lift_kernel_in(&C, src, "k", c)
}

fn f32s(v: &[f32]) -> KernelArg {
    KernelArg::Buffer(TypedBuffer::from_f32(&[v.len() as i64], v))
}

fn run(k: &KernelFn, out: usize, extra: Vec<KernelArg>) -> Result<Vec<f64>, KernelTrap> {
    let mut args = vec![f32s(&vec![0.0; out])];
    args.extend(extra);
    run_launch(k, &mut args)?;
    let KernelArg::Buffer(o) = &args[0] else {
        panic!()
    };
    Ok(o.to_f64_vec())
}

fn not_yet(r: Result<KernelFn, KernelLiftError>, needle: &str) {
    match r {
        Err(KernelLiftError::NotYetLifted { construct, .. }) => {
            assert!(construct.contains(needle), "`{construct}` lacks `{needle}`");
        }
        other => panic!("expected NotYetLifted({needle}), got {other:?}"),
    }
}

#[test]
fn preprocessor_branches_follow_the_contexts_definitions_and_the_files_own_defines() {
    let src = "#if (__V__ >= 11) && (!defined(__A__) || (__A__ >= 800))\n#define EN\n#endif\n\
               void k(float* out) {\n#ifdef EN\n out[0] = 1.0f;\n#elif defined(OTHER)\n out[0] = 3.0f;\n#else\n out[0] = 2.0f;\n#endif\n\
               #ifndef EN\n out[1] = 7.0f;\n#endif\n}";
    let one = |v: i64, a: Option<i64>, extra: Option<&str>| {
        let mut c = ctx().with_define("__V__", v);
        if let Some(a) = a {
            c = c.with_define("__A__", a);
        }
        if let Some(e) = extra {
            c = c.with_define(e, 1);
        }
        run(&lift(src, &c).unwrap(), 2, vec![]).unwrap()
    };
    assert_eq!(one(13, Some(800), None), vec![1.0, 0.0]); // EN defined by the file
    assert_eq!(one(13, None, None), vec![1.0, 0.0]); // no arch: the `!defined` arm
    assert_eq!(one(13, Some(700), None), vec![2.0, 7.0]); // arch too old: EN not defined, #else, #ifndef
    assert_eq!(one(10, Some(800), None), vec![2.0, 7.0]); // version too old
    assert_eq!(one(10, Some(800), Some("OTHER")), vec![3.0, 7.0]); // #elif taken
}

#[test]
fn a_directive_the_pass_cannot_model_declines_and_line_numbers_survive_blanking() {
    not_yet(
        lift(
            "#define SQ(x) ((x) * (x))\nvoid k(float* out) { out[0] = 1.0f; }",
            &ctx(),
        ),
        "function-like macro `SQ`",
    );
    not_yet(
        lift(
            "#if __X__ ?\nvoid k(float* out) { out[0] = 1.0f; }\n#endif",
            &ctx(),
        ),
        "preprocessor condition",
    );
    not_yet(
        lift("#if 1\nvoid k(float* out) { out[0] = 1.0f; }", &ctx()),
        "no `#endif`",
    );
    // the inactive region and every directive line are blanked, not removed: the decline after them still names line 6
    let r = lift(
        "#if 0\nvoid gone() {}\n#endif\n#define A 1\nvoid k(float* out) {\n out[0] = frobnicate(1.0f);\n}",
        &ctx(),
    );
    match r {
        Err(KernelLiftError::NotYetLifted { line, .. }) => assert_eq!(line, 6),
        other => panic!("{other:?}"),
    }
}

#[test]
fn file_scope_constants_sizeof_and_constant_sized_arrays() {
    let k = lift(
        "constexpr unsigned int VEC = 4;\nconstexpr unsigned int BYTES = VEC * sizeof(float) + sizeof(half*) + 2 * sizeof(half);\n\
         constexpr float SCALE = 5e4;\nconstexpr float HALF = 0.5f;\n\
         void k(float* out) { float v[VEC]; for (unsigned int i = 0; i < VEC; i++) { v[i] = (float)i * HALF; } \
         out[0] = SCALE; out[1] = (float)BYTES; out[2] = v[3]; out[3] = (float)sizeof(half); out[4] = (float)sizeof(float*); }",
        &ctx(),
    )
    .unwrap();
    // BYTES = 4*4 + 8 + 2*2 = 28
    assert_eq!(
        run(&k, 5, vec![]).unwrap(),
        vec![50_000.0, 28.0, 1.5, 2.0, 8.0]
    );
}

#[test]
fn dynamic_shared_memory_is_the_callers_byte_count() {
    let src = "void k(float* out) { extern __shared__ float s[]; s[15] = 4.5f; out[0] = s[15]; s[16] = 1.0f; out[1] = s[16]; }";
    let k = lift(src, &ctx().with_dynamic_shared_bytes(64)).unwrap();
    // 64 bytes of float is 16 elements: index 15 is in range, index 16 traps
    let r = run(&k, 2, vec![]);
    assert!(
        r.is_err(),
        "index 16 of a 16-element array must trap, got {r:?}"
    );
    let ok = lift(
        &src.replace(" s[16] = 1.0f; out[1] = s[16];", ""),
        &ctx().with_dynamic_shared_bytes(64),
    )
    .unwrap();
    assert_eq!(run(&ok, 2, vec![]).unwrap(), vec![4.5, 0.0]);
    not_yet(lift(src, &ctx()), "no dynamic shared memory given");
    not_yet(
        lift_kernel_in(
            &C,
            src,
            "k",
            &LiftContext::new(LaunchShape::new([1, 1, 1], [1, 1, 1])).with_dynamic_shared_bytes(2),
        ),
        "no dynamic shared memory given",
    );
}

#[test]
fn a_null_check_on_a_buffer_is_true_unless_the_caller_declared_it_null() {
    let src = "void k(float* out, const float* mask) { out[0] = 1.0f; if (mask && mask[0] == 0.0f) { return; } out[0] = 2.0f; \
               out[1] = (mask && !mask[0]) ? 5.0f : 6.0f; }";
    let args = || vec![f32s(&[0.0])];
    assert_eq!(
        run(&lift(src, &ctx()).unwrap(), 2, args()).unwrap(),
        vec![1.0, 0.0]
    );
    assert_eq!(
        run(
            &lift(src, &ctx().with_null_buffer("mask")).unwrap(),
            2,
            args()
        )
        .unwrap(),
        vec![2.0, 6.0]
    );
}

#[test]
fn a_struct_with_pointer_fields_is_built_by_a_brace_list_or_a_copy() {
    let src = "struct View { unsigned int n; const float* data; };\nvoid k(float* out, const float* a) { View v; out[0] = 1.0f; }";
    not_yet(lift(src, &ctx()), "brace list");
    let k = lift(
        "struct Inner { unsigned int d; };\nstruct View { Inner in; unsigned int n; const float* data; float* dst; };\n\
         void k(float* out, const float* a, const __grid_constant__ Inner cfg) { \
         View v = {cfg, 2, a + 1, out + 3}; View w = v; v.dst[0] = v.data[v.n] + (float)w.in.d; w.dst[1] = w.data[0]; }",
        &ctx(),
    )
    .unwrap();
    let a = [10.0f32, 20.0, 30.0, 40.0];
    // v.data = a + 1, v.n = 2: v.data[2] = a[3] = 40, plus w.in.d (7) lands at out[3]; w.data[0] = a[1] = 20 lands at out[4]
    let mut args = vec![
        f32s(&[0.0; 5]),
        f32s(&a),
        KernelArg::Struct(vec![KernelArg::Int(7)]),
    ];
    run_launch(&k, &mut args).unwrap();
    let KernelArg::Buffer(o) = &args[0] else {
        panic!()
    };
    assert_eq!(o.to_f64_vec(), vec![0.0, 0.0, 0.0, 47.0, 20.0]);
}

#[test]
fn a_bool_is_a_normalised_byte_and_integer_min_max_take_the_common_type() {
    let k = lift(
        "void k(float* out, int n, unsigned int u) { bool a = n > 2; bool b = 5u; bool c = n - 3; \
         out[0] = a ? 1.0f : 2.0f; out[1] = (float)b; out[2] = (float)c; \
         out[3] = (float)min(n, 3); out[4] = (float)max(u, 7u); out[5] = max(1.5f, 0.5f); out[6] = (float)min(n, (int)u); }",
        &ctx(),
    )
    .unwrap();
    let args = vec![KernelArg::Int(5), KernelArg::Int(9)];
    // a = true; b = 5 normalised to 1; c = 5 - 3 = 2 normalised to 1; min(5,3) = 3; max(9,7) = 9; max(1.5,0.5) = 1.5; min(5,9) = 5
    assert_eq!(
        run(&k, 7, args).unwrap(),
        vec![1.0, 1.0, 1.0, 3.0, 9.0, 1.5, 5.0]
    );
}

#[test]
fn a_vector_is_read_and_written_through_a_pointer_one_lane_at_a_time() {
    let k = lift(
        "void k(float* out, const float* a) { float4 v = ((const float4*)a)[1]; v.y = v.y + 100.0f; ((float4*)out)[0] = v; \
         *((float4*)(out + 4)) = ((const float4*)a)[0]; }",
        &ctx(),
    )
    .unwrap();
    let a = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let mut args = vec![f32s(&[0.0; 8]), f32s(&a)];
    run_launch(&k, &mut args).unwrap();
    let KernelArg::Buffer(o) = &args[0] else {
        panic!()
    };
    // out[0..4] = a[4..8] with lane y raised by 100; out[4..8] = a[0..4]
    assert_eq!(
        o.to_f64_vec(),
        vec![5.0, 106.0, 7.0, 8.0, 1.0, 2.0, 3.0, 4.0]
    );
    not_yet(
        lift(
            "void k(float* out, const float* a) { int4 v = ((const float4*)a)[0]; out[0] = 1.0f; }",
            &ctx(),
        ),
        "vector",
    );
}

#[test]
fn an_elif_after_a_taken_branch_is_never_taken_even_when_true() {
    // `#if 1` is taken, so the true `#elif 1` after it must stay blanked (only one branch of a chain is live)
    let src = "#if 1\n#define FIRST\n#elif 1\n#define SECOND\n#endif\nvoid k(float* out) {\n#ifdef SECOND\n out[0] = 2.0f;\n#else\n out[0] = 1.0f;\n#endif\n}";
    assert_eq!(
        run(&lift(src, &ctx()).unwrap(), 1, vec![]).unwrap(),
        vec![1.0]
    );
}

#[test]
fn constant_sizes_fold_with_subtraction_and_shifts() {
    let k = lift(
        "constexpr unsigned int A = 10 - 3;\nconstexpr unsigned int B = 1 << 4;\nconstexpr unsigned int C = 64 >> 2;\n\
         void k(float* out) { float v[A]; v[6] = 1.0f; out[0] = (float)sizeof(float) + (float)A; out[1] = (float)B; out[2] = (float)C; }",
        &ctx(),
    )
    .unwrap();
    assert_eq!(run(&k, 3, vec![]).unwrap(), vec![11.0, 16.0, 16.0]);
}

#[test]
fn a_constant_condition_decides_whether_the_right_side_is_evaluated() {
    // `0 && x[1000]` must not read (an out-of-range load would trap); `1 || x[1000]` likewise; `1 && x[0]` and `0 || x[0]` must
    let k = lift(
        "void k(float* out, const float* x) { \
         out[0] = (0 && x[1000] > 0.0f) ? 1.0f : 2.0f; out[1] = (1 || x[1000] > 0.0f) ? 3.0f : 4.0f; \
         out[2] = (1 && x[0] > 5.0f) ? 5.0f : 6.0f; out[3] = (0 || x[0] > 5.0f) ? 7.0f : 8.0f; }",
        &ctx(),
    )
    .unwrap();
    assert_eq!(
        run(&k, 4, vec![f32s(&[9.0])]).unwrap(),
        vec![2.0, 3.0, 5.0, 7.0]
    );
}

#[test]
fn a_brace_list_must_give_every_field() {
    not_yet(
        lift(
            "struct V { unsigned int n; const float* d; };\nvoid k(float* out, const float* a) { V v = {1}; out[0] = 1.0f; }",
            &ctx(),
        ),
        "every field must be given",
    );
}

fn halves(v: &[f64]) -> KernelArg {
    KernelArg::Buffer(TypedBuffer::from_f64_as(
        unpopped_vocab::ElementKind::F16,
        &[v.len() as i64],
        v,
    ))
}

#[test]
fn a_vector_local_whose_address_is_taken_is_memory_backed_and_reads_as_half2() {
    // the decode kernel's idiom: 16 bytes loaded as int4, viewed as 4 half2, each widened to a float2
    let k = lift(
        "void k(float* out, const half* a) { int4 tmp = ((const int4*)a)[0]; \
         for (int i = 0; i < 4; i++) { ((float2*)out)[i] = __half22float2(((half2*)&tmp)[i]); } }",
        &ctx(),
    )
    .unwrap();
    let h = [1.0, 2.5, -3.0, 0.5, 8.0, -0.25, 100.0, 0.125];
    assert_eq!(run(&k, 8, vec![halves(&h)]).unwrap(), h.to_vec());
}

#[test]
fn a_half2_is_written_through_the_pun_then_stored_as_int4() {
    let k = lift(
        "void k(half* out, const float* a) { int4 tmp; \
         for (int i = 0; i < 4; i++) { ((half2*)&tmp)[i] = __float22half2_rn(((const float2*)a)[i]); } \
         ((int4*)out)[0] = tmp; }",
        &ctx(),
    )
    .unwrap();
    // 0.1 is not exact in half: it rounds to 0.0999755859375; 70000 overflows half to infinity
    let a = [1.0f32, 2.5, -3.0, 0.1, 8.0, -0.25, 70000.0, 0.125];
    let mut args = vec![halves(&[0.0; 8]), f32s(&a)];
    run_launch(&k, &mut args).unwrap();
    let KernelArg::Buffer(o) = &args[0] else {
        panic!()
    };
    let want = [
        1.0,
        2.5,
        -3.0,
        0.099_975_585_937_5,
        8.0,
        -0.25,
        f64::INFINITY,
        0.125,
    ];
    assert_eq!(o.to_f64_vec(), want.to_vec());
}

#[test]
fn a_vector_without_an_initializer_and_without_its_address_taken_declines() {
    not_yet(
        lift("void k(float* out) { float4 v; out[0] = 1.0f; }", &ctx()),
        "without an initializer",
    );
}

#[test]
fn a_pointer_compared_with_nullptr_is_the_null_check() {
    let src = "void k(float* out, const float* m) { if (m != nullptr) { out[0] = 1.0f; } if (nullptr == m) { out[1] = 2.0f; } \
               out[2] = (m == nullptr) ? 3.0f : 4.0f; }";
    let present = lift(src, &ctx()).unwrap();
    assert_eq!(
        run(&present, 3, vec![f32s(&[0.0])]).unwrap(),
        vec![1.0, 0.0, 4.0]
    );
    let absent = lift(src, &ctx().with_null_buffer("m")).unwrap();
    assert_eq!(
        run(&absent, 3, vec![f32s(&[0.0])]).unwrap(),
        vec![0.0, 2.0, 3.0]
    );
}
