// SPDX-License-Identifier: MIT OR Apache-2.0
//! **U19 slice 4d: structs** (`docs/options-u19-slice5-decode-census.md`): scalar fields, array fields and nested structs,
//! scalarized (one local per scalar, one `Private` allocation per array); a by-value struct kernel parameter becomes one
//! `ParamKind::Struct`. The test parser blanks `__grid_constant__`/`__device__`/`__forceinline__` to same-length spaces.

use unpopped::exec::LaunchShape;
use unpopped::kernel::{KernelFn, ParamKind};
use unpopped::oracle::TypedBuffer;
use unpopped::oracle::kernel_eval::KernelArg;
use unpopped::oracle::kernel_launch::run_launch;
use unpopped_convert::{Frontend, KernelLiftError, lift_kernel_launched};

fn parse(src: &str) -> Option<tree_sitter::Tree> {
    let mut blanked = src.to_string();
    for q in ["__grid_constant__", "__device__", "__forceinline__"] {
        blanked = blanked.replace(q, &" ".repeat(q.len()));
    }
    let mut p = tree_sitter::Parser::new();
    p.set_language(&tree_sitter_cpp::LANGUAGE.into()).ok()?;
    p.parse(blanked, None)
}

const C: Frontend = Frontend::new("c", "void", &[], "out", "in", parse);

fn lift(src: &str) -> Result<KernelFn, KernelLiftError> {
    lift_kernel_launched(&C, src, "k", LaunchShape::new([1, 1, 1], [1, 1, 1]))
}

fn run(k: &KernelFn, out: usize, extra: Vec<KernelArg>) -> Vec<f64> {
    let mut args = vec![KernelArg::Buffer(TypedBuffer::from_f32(
        &[out as i64],
        &vec![0.0; out],
    ))];
    args.extend(extra);
    run_launch(k, &mut args).expect("runs");
    let KernelArg::Buffer(o) = &args[0] else {
        panic!()
    };
    o.to_f64_vec()
}

const DEFS: &str = "struct Inner { unsigned int d; float w; };\n\
    struct State { alignas(16) float o[4]; float m; float d; Inner in; };\n";

#[test]
fn a_struct_local_has_scalar_array_and_nested_members_that_start_at_zero() {
    let k = lift(&format!(
        "{DEFS}void k(float* out) {{ State st; out[0] = st.m + st.o[2] + (float)st.in.d; \
         st.m = 1.5f; st.d = 2.5f; st.in.d = 7u; st.in.w = 0.25f; \
         for (int i = 0; i < 4; i++) {{ st.o[i] = (float)i * 10.0f + st.m; }} \
         st.m += st.d; st.in.d += 1u; \
         out[1] = st.m; out[2] = st.o[3]; out[3] = (float)st.in.d; out[4] = st.in.w; out[5] = st.o[0]; }}"
    ))
    .unwrap();
    assert_eq!(run(&k, 6, vec![]), vec![0.0, 4.0, 31.5, 8.0, 0.25, 1.5]);
}

#[test]
fn a_struct_is_copied_by_value() {
    let k = lift(&format!(
        "{DEFS}void k(float* out) {{ State a; a.m = 3.0f; a.o[1] = 5.0f; a.o[3] = 11.0f; a.in.d = 9u; \
         State b = a; a.m = 100.0f; a.o[1] = 200.0f; \
         State c; c = b; b.in.d = 1u; \
         out[0] = b.m; out[1] = b.o[1]; out[2] = (float)b.in.d; out[3] = c.m; out[4] = c.o[1]; out[5] = (float)c.in.d; out[6] = b.o[3]; out[7] = c.o[3]; }}"
    ))
    .unwrap();
    // b and c took a's values at the copy; later writes to a (and to b.in.d) reach neither c nor each other
    assert_eq!(
        run(&k, 8, vec![]),
        vec![3.0, 5.0, 1.0, 3.0, 5.0, 9.0, 11.0, 11.0]
    );
}

#[test]
fn a_struct_reference_parameter_aliases_the_callers_instance_and_a_value_parameter_copies_it() {
    let k = lift(&format!(
        "{DEFS}__device__ void bump(State& s, float by) {{ s.m += by; s.o[0] = s.m; s.in.d = 42u; }}\n\
         __device__ float peek(State s) {{ s.m = 999.0f; return s.o[0]; }}\n\
         __device__ float total(const State& s) {{ return s.m + s.o[0]; }}\n\
         void k(float* out) {{ State st; st.m = 1.0f; bump(st, 2.5f); bump(st, 1.0f); float p = peek(st); \
         out[0] = st.m; out[1] = st.o[0]; out[2] = (float)st.in.d; out[3] = p; out[4] = total(st); }}"
    ))
    .unwrap();
    assert_eq!(run(&k, 5, vec![]), vec![4.5, 4.5, 42.0, 4.5, 9.0]);
}

#[test]
fn a_struct_kernel_parameter_is_one_param_struct_of_flattened_scalars() {
    let k = lift(
        "struct Pair { unsigned int a; float b; };\nstruct Cfg { float scale; int n; Pair p; };\n\
         void k(float* out, const __grid_constant__ Cfg cfg) { out[0] = cfg.scale * (float)cfg.n; out[1] = (float)cfg.p.a + cfg.p.b; }",
    )
    .unwrap();
    let ParamKind::Struct { fields } = &k.params[1].kind else {
        panic!("{:?}", k.params[1])
    };
    let names: Vec<_> = fields.iter().map(|f| f.name.as_str()).collect();
    assert_eq!(names, ["cfg.scale", "cfg.n", "cfg.p.a", "cfg.p.b"]);
    let args = vec![KernelArg::Struct(vec![
        KernelArg::Float(2.5),
        KernelArg::Int(4),
        KernelArg::Int(7),
        KernelArg::Float(0.5),
    ])];
    assert_eq!(run(&k, 2, args), vec![10.0, 7.5]);
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
fn what_cannot_be_scalarized_declines_by_name() {
    let h = "void k(float* out) {";
    not_yet(
        &format!("struct P {{ float* p; }};\n{h} P x; out[0] = 1.0f; }}"),
        "pointer field",
    );
    not_yet(
        &format!("struct P {{ float v[3]; }};\n{h} P x[2]; out[0] = 1.0f; }}"),
        "struct declarator",
    );
    not_yet(
        &format!("struct P {{ float v; }};\n{h} P x; out[0] = x.nope; }}"),
        "member `nope`",
    );
    not_yet(
        &format!("struct P {{ float v; }};\n{h} P x; out[0] = x; }}"),
        "used as a value",
    );
    not_yet(
        &format!(
            "struct P {{ float v; float get() {{ return v; }} }};\n{h} P x; out[0] = x.nope(); }}"
        ),
        "no such method",
    );
    not_yet(
        "struct P { float v; };\nvoid k(float* out, const P p) { p.v = 1.0f; out[0] = 1.0f; }",
        "struct parameter",
    );
    // an array of scalars is a struct parameter's flattened field since census slice 5 (tests/kernel_body_param_arrays.rs);
    // a pointer field is still not a parameter
    not_yet(
        "struct P { float* v; };\nvoid k(float* out, const P p) { out[0] = 1.0f; }",
        "only scalar fields",
    );
    // Control: the plain spelling lifts.
    assert!(
        lift(&format!(
            "struct P {{ float v; }};\n{h} P x; x.v = 1.0f; out[0] = x.v; }}"
        ))
        .is_ok()
    );
}

#[test]
fn an_empty_brace_list_value_initialises_every_field_to_zero() {
    // `S x{};` and `S x = {};` are value-initialisation (a functor `F op{};` is the real-world spelling), also with an array field
    // and a nested struct; a short non-empty list still declines
    let k = lift(
        "struct In { int a; float b; }; struct S { float x; int n; In in; float v[2]; \
           float get() const { return x + (float)n + in.b + v[1]; } };\n\
         void k(float* out) { S s{}; S t = {}; out[0] = s.get() + 1.0f; t.x = 4.0f; t.v[1] = 2.0f; out[1] = t.get(); out[2] = s.get(); }",
    )
    .expect("lifts (was a decline: every field must be given)");
    assert_eq!(run(&k, 3, vec![]), vec![1.0, 6.0, 0.0]);
    assert!(
        lift("struct S { float x; int n; }; void k(float* out) { S s{1.0f}; out[0] = s.x; }")
            .is_err(),
        "a list shorter than the fields still declines"
    );
}
