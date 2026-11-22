// SPDX-License-Identifier: MIT OR Apache-2.0
//! **U19 slice 5: the constructs the FlashInfer paged-decode kernel needed** after the context groundwork
//! (`docs/options-u19-slice5-decode-census.md`). Constant-trip loops that are not counted loops, a memory read in one arm of
//! `?:`, an `if`/`else` that returns in an inlined function, `static_cast`/`reinterpret_cast`, a struct's array field as a
//! pointer, `true`/`false`, the two `cp.async` asm templates with their shared-address integer, and buffer roles derived from
//! use (so a barrier under a bound loaded from a read-only buffer is uniform). Each lifted kernel runs on the oracle.

use unpopped::exec::LaunchShape;
use unpopped::kernel::{BufferRole, KernelError, KernelFn, ParamKind};
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
    LiftContext::new(LaunchShape::new([1, 1, 1], [1, 1, 1])).with_dynamic_shared_bytes(64)
}

fn lift(src: &str) -> Result<KernelFn, KernelLiftError> {
    lift_kernel_in(&C, src, "k", &ctx())
}

fn f32s(v: &[f32]) -> KernelArg {
    KernelArg::Buffer(TypedBuffer::from_f32(&[v.len() as i64], v))
}

fn halves(v: &[f64]) -> KernelArg {
    KernelArg::Buffer(TypedBuffer::from_f64_as(
        unpopped_vocab::ElementKind::F16,
        &[v.len() as i64],
        v,
    ))
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
fn a_constant_trip_loop_that_is_not_counted_runs_once_per_value() {
    // the butterfly `for (offset = BDX / 2; offset > 0; offset /= 2)`: 8, 4, 2, 1
    let k = lift(
        "constexpr unsigned int BDX = 16;\nvoid k(float* out) { float s = 0.0f; float p = 1.0f; \
         for (unsigned int o = BDX / 2; o > 0; o /= 2) { s += (float)o; p = p * 2.0f; } out[0] = s; out[1] = p; \
         float d = 0.0f; for (int i = 10; i >= 2; i -= 3) { d += (float)i; } out[2] = d; \
         float q = 0.0f; for (unsigned int j = 1; j < 100; j <<= 2) { q += (float)j; } out[3] = q; }",
    )
    .unwrap();
    // 8+4+2+1 = 15 over 4 trips (2^4 = 16); 10+7+4 = 21; 1+4+16+64 = 85
    assert_eq!(run(&k, 4, vec![]).unwrap(), vec![15.0, 16.0, 21.0, 85.0]);
}

#[test]
fn a_constant_loop_that_never_ends_or_runs_long_declines() {
    not_yet(
        lift("void k(float* out) { for (int i = 10; i != 0; i -= 3) { out[0] = 1.0f; } }"),
        "more than 64 trips",
    );
    // a long loop of the counted form is a `For`, not an unroll
    assert!(
        lift("void k(float* out) { for (int i = 0; i < 1000; i += 1) { out[0] = 1.0f; } }").is_ok()
    );
    // a long loop that only the unroll could take declines
    not_yet(
        lift("void k(float* out) { for (int i = 1; i != 100000; i *= 1) { out[0] = 1.0f; } }"),
        "more than 64 trips",
    );
}

#[test]
fn a_read_in_one_arm_of_a_conditional_happens_only_when_that_arm_is_taken() {
    let k = lift("void k(float* out, const float* a, int i) { out[0] = (i < 2) ? a[i] : -1.0f; out[1] = (i >= 2) ? -2.0f : a[i]; }").unwrap();
    let go = |i: i32| {
        run(
            &k,
            2,
            vec![f32s(&[10.0, 20.0]), KernelArg::Int(i128::from(i))],
        )
    };
    assert_eq!(go(1).unwrap(), vec![20.0, 20.0]);
    // i = 5 is out of range: the load must not run (a hoisted unconditional load would trap)
    assert_eq!(go(5).unwrap(), vec![-1.0, -2.0]);
}

#[test]
fn an_if_else_that_returns_in_both_arms_lifts_in_an_inlined_function() {
    let k = lift(
        "float f(int x) { if (x > 2) { return 1.0f; } else { if (x > 0) { return 2.0f; } else { return 3.0f; } } }\n\
         float g(int x) { if (x > 5) { return 10.0f; } else if (x > 1) { return 20.0f; } else { return 30.0f; } }\n\
         void k(float* out, int n) { out[0] = f(n); out[1] = g(n); }",
    )
    .unwrap();
    let go = |n: i32| run(&k, 2, vec![KernelArg::Int(i128::from(n))]).unwrap();
    assert_eq!(go(3), vec![1.0, 20.0]);
    assert_eq!(go(1), vec![2.0, 30.0]);
    assert_eq!(go(-4), vec![3.0, 30.0]);
    assert_eq!(go(9), vec![1.0, 10.0]);
    // a return buried in a loop still declines
    not_yet(
        lift(
            "float f(int x) { for (int i = 0; i < 3; i++) { if (i == x) { return 1.0f; } } return 0.0f; }\nvoid k(float* out) { out[0] = f(1); }",
        ),
        "inside a loop",
    );
}

#[test]
fn static_cast_reinterpret_cast_and_boolean_literals() {
    let k = lift(
        "void k(float* out, const unsigned int* a, float v) { out[0] = (float)static_cast<int>(v); \
         const float* p = reinterpret_cast<const float*>(a); out[1] = p[0]; \
         bool t = true; bool f = false; out[2] = t ? 1.0f : 2.0f; out[3] = f ? 1.0f : 2.0f; \
         out[4] = (float)static_cast<unsigned int>(7); }",
    )
    .unwrap();
    let bits = KernelArg::Buffer(TypedBuffer::from_u32(&[1], &[1.5f32.to_bits()]));
    assert_eq!(
        run(&k, 5, vec![bits, KernelArg::Float(2.75)]).unwrap(),
        vec![2.0, 1.5, 1.0, 2.0, 7.0]
    );
}

#[test]
fn a_struct_array_field_is_a_pointer_to_its_private_allocation() {
    let k = lift(
        "struct S { float o[4]; float m; };\nvoid fill(S& s) { for (int i = 0; i < 2; i++) { ((float2*)s.o)[i] = make_float2((float)i, (float)i + 10.0f); } s.m = 5.0f; }\n\
         void k(float* out) { S s; fill(s); out[0] = s.o[0]; out[1] = s.o[1]; out[2] = s.o[2]; out[3] = s.o[3]; out[4] = s.m; }",
    )
    .unwrap();
    assert_eq!(run(&k, 5, vec![]).unwrap(), vec![0.0, 10.0, 1.0, 11.0, 5.0]);
}

#[test]
fn cp_async_asm_with_a_shared_address_lifts_to_copy_async() {
    let fill = |bytes: &str| {
        lift(&format!(
            "void k(float* out, const half* g) {{ extern __shared__ uint8_t smem[]; half* s = (half*)smem; \
             uint32_t si = static_cast<uint32_t>(__cvta_generic_to_shared(s)); int nb = {bytes}; \
             asm volatile(\"cp.async.cg.shared.global.L2::128B [%0], [%1], %2, %3;\\n\" ::\"r\"(si), \"l\"(g), \"n\"(16), \"r\"(nb)); \
             asm volatile(\"cp.async.commit_group;\\n\" ::); asm volatile(\"cp.async.wait_group %0;\\n\" ::\"n\"(0)); \
             __syncthreads(); out[0] = (float)s[0]; out[1] = (float)s[3]; out[2] = (float)s[7]; }}"
        ))
        .unwrap()
    };
    let g = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    // all 16 bytes land
    assert_eq!(
        run(&fill("16"), 3, vec![halves(&g)]).unwrap(),
        vec![1.0, 4.0, 8.0]
    );
    // 8 source bytes: the first four halves land, the rest are zero-filled
    assert_eq!(
        run(&fill("8"), 3, vec![halves(&g)]).unwrap(),
        vec![1.0, 4.0, 0.0]
    );
    // 0 source bytes: nothing is read, everything is zero
    assert_eq!(
        run(&fill("0"), 3, vec![halves(&g)]).unwrap(),
        vec![0.0, 0.0, 0.0]
    );
}

#[test]
fn a_predicated_cp_async_copies_only_when_the_predicate_holds() {
    let src = "void k(float* out, const half* g, int pred) { extern __shared__ uint8_t smem[]; half* s = (half*)smem; \
         for (int i = 0; i < 8; i++) { s[i] = (half)9.0f; } \
         uint32_t si = static_cast<uint32_t>(__cvta_generic_to_shared(s)); \
         asm volatile(\"{\\n .reg .pred p;\\n setp.ne.b32 p, %0, 0;\\n @p cp.async.cg.shared.global.L2::128B [%1], [%2], %3;\\n}\\n\" \
         ::\"r\"((int)pred), \"r\"(si), \"l\"(g), \"n\"(16)); \
         asm volatile(\"cp.async.commit_group;\\n\" ::); asm volatile(\"cp.async.wait_group %0;\\n\" ::\"n\"(0)); \
         __syncthreads(); out[0] = (float)s[0]; out[1] = (float)s[7]; }";
    let k = lift(src).unwrap();
    let g = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    assert_eq!(
        run(&k, 2, vec![halves(&g), KernelArg::Int(1)]).unwrap(),
        vec![1.0, 8.0]
    );
    // no fill: a false predicate leaves the destination as it was
    assert_eq!(
        run(&k, 2, vec![halves(&g), KernelArg::Int(0)]).unwrap(),
        vec![9.0, 9.0]
    );
    not_yet(
        lift(&src.replace("[%1], [%2], %3", "[%1], [%2]")),
        "asm template",
    );
}

#[test]
fn buffer_roles_follow_use_and_make_a_barrier_under_a_loaded_bound_uniform() {
    let ok = lift("void k(float* out, int* n) { int c = n[0]; for (int i = 0; i < c; i++) { __syncthreads(); } out[0] = 1.0f; }").unwrap();
    let role = |k: &KernelFn, i: usize| match &k.params[i].kind {
        ParamKind::Buffer { role, .. } => *role,
        other => panic!("{other:?}"),
    };
    assert_eq!(role(&ok, 0), BufferRole::ReadWrite);
    assert_eq!(
        role(&ok, 1),
        BufferRole::Read,
        "`int* n` is never stored to"
    );
    // Control: the same loop over a buffer the kernel also writes is divergent
    match lift(
        "void k(float* out, int* n) { int c = n[0]; for (int i = 0; i < c; i++) { __syncthreads(); } n[1] = 0; out[0] = 1.0f; }",
    ) {
        Err(KernelLiftError::Invalid(KernelError::BarrierDivergent)) => {}
        other => panic!("{other:?}"),
    }
}
