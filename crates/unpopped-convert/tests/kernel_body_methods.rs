// SPDX-License-Identifier: MIT OR Apache-2.0
//! **U19 slice 4e: member functions and early returns** (`docs/options-u19-slice5-decode-census.md`). `obj.f(args)` inlines with
//! `this` bound to the instance; a struct's conversion operator is inlined where the instance is used as a number; a callee may
//! `return` early from a tail `if`. The struct under test is Baracuda's `uint_fastdiv` verbatim (its `divmod` has an early
//! `return`, a 64-bit multiply, a shift by `l_ - 1` and `uint64_t(x)` functional casts), checked against Rust's `n / d`. The
//! division constants are computed here exactly as FlashInfer's constructor does.

use unpopped::kernel::KernelFn;
use unpopped::oracle::TypedBuffer;
use unpopped::oracle::kernel_eval::{KernelArg, run_kernel};
use unpopped_convert::{Frontend, KernelLiftError, lift_kernel};

fn parse(src: &str) -> Option<tree_sitter::Tree> {
    let mut blanked = src.to_string();
    for q in [
        "__host__",
        "__device__",
        "__forceinline__",
        "__grid_constant__",
    ] {
        blanked = blanked.replace(q, &" ".repeat(q.len()));
    }
    let mut p = tree_sitter::Parser::new();
    p.set_language(&tree_sitter_cpp::LANGUAGE.into()).ok()?;
    p.parse(blanked, None)
}

const C: Frontend = Frontend::new("c", "void", &[], "out", "in", parse);

fn lift(src: &str) -> Result<KernelFn, KernelLiftError> {
    lift_kernel(&C, src, "k")
}

const FASTDIV: &str = "struct uint_fastdiv {\n  uint32_t d_, m_, l_;\n\
  __host__ __device__ __forceinline__ void divmod(uint32_t n, uint32_t& q, uint32_t& r) const {\n\
    if (d_ <= 1) {\n      q = n;\n      r = 0;\n      return;\n    }\n\
    uint32_t hi = uint32_t((uint64_t(m_) * uint64_t(n)) >> 32);\n\
    q = (hi + ((n - hi) >> 1)) >> (l_ - 1);\n    r = n - q * d_;\n  }\n\
  __host__ __device__ __forceinline__ operator unsigned int() const { return d_; }\n};\n\
struct Cfg { uint_fastdiv page_size; };\n";

/// FlashInfer's `uint_fastdiv(d)`: `l = ceil(log2 d)`, `m = ((1 << 32) * ((1 << l) - d)) / d + 1`.
fn magic(d: u32) -> (u32, u32) {
    let mut l = 0u32;
    while l < 32 && (1u64 << l) < u64::from(d) {
        l += 1;
    }
    let m = ((1u64 << 32) * ((1u64 << l) - u64::from(d)) / u64::from(d) + 1) as u32;
    (m, l)
}

fn u32s(v: &[u32]) -> KernelArg {
    KernelArg::Buffer(TypedBuffer::from_u32(&[v.len() as i64], v))
}

fn out_u32(a: &KernelArg) -> Vec<f64> {
    let KernelArg::Buffer(b) = a else { panic!() };
    b.to_f64_vec()
}

/// `uint_fastdiv` as the real file spells it, but with the 64-bit multiply-shift replaced by 32-bit division: everything else
/// (the struct parameter, the early `return`, the by-reference outputs, the nested member call) is the real shape.
const DIV32: &str = "struct uint_fastdiv {
  uint32_t d_, m_, l_;
  __host__ __device__ __forceinline__ void divmod(uint32_t n, uint32_t& q, uint32_t& r) const {
    if (d_ <= 1) {
      q = n;
      r = 0;
      return;
    }
    q = n / d_;
    r = n - q * d_;
  }
};
struct Cfg { uint_fastdiv page_size; };
";

#[test]
fn a_struct_method_with_an_early_return_and_by_reference_outputs_agrees_with_integer_division() {
    let k = lift(&format!(
        "{DIV32}void k(unsigned int* out, const unsigned int* a, const __grid_constant__ Cfg cfg) {{          for (int i = 0; i < 8; i++) {{ unsigned int q; unsigned int r; cfg.page_size.divmod(a[i], q, r); out[2 * i] = q; out[2 * i + 1] = r; }} }}"
    ))
    .unwrap();
    let ns = [0u32, 1, 6, 7, 8, 123_456, 4_000_000_000, u32::MAX];
    // d = 7, 10, 16, 3 take the division path; d = 1 and d = 0 take the early `return` (q = n, r = 0)
    for d in [7u32, 10, 16, 3, 1, 0] {
        let mut args = vec![
            u32s(&[0; 16]),
            u32s(&ns),
            KernelArg::Struct(vec![
                KernelArg::Int(i128::from(d)),
                KernelArg::Int(0),
                KernelArg::Int(0),
            ]),
        ];
        run_kernel(&k, &mut args).expect("runs");
        let want: Vec<f64> = ns
            .iter()
            .flat_map(|&n| {
                if d <= 1 {
                    [f64::from(n), 0.0]
                } else {
                    [f64::from(n / d), f64::from(n % d)]
                }
            })
            .collect();
        assert_eq!(out_u32(&args[0]), want, "d = {d}");
    }
}

/// **A pinned gap, not a feature** (found lifting Baracuda's flat decode kernel): the real `divmod` computes
/// `uint32_t((uint64_t(m_) * uint64_t(n)) >> 32)`, a shift on a `U64` in a kernel whose compute type is `U32`. The IR's type
/// rules (`types::node_ty`) refuse a shift or a bit operation on any integer type other than the compute type, though the oracle
/// evaluates them at any integer type. A float kernel meets the same refusal on every `>>`, `&` and `|` of index math. When the
/// rule is relaxed this test flips to a lift, and the differential above should then run the real `divmod`.
#[test]
fn the_real_fastdiv_divmod_is_outside_ir_until_shifts_are_legal_at_every_integer_type() {
    let r = lift(&format!(
        "{FASTDIV}void k(unsigned int* out, const unsigned int* a, const __grid_constant__ Cfg cfg) {{          unsigned int q; unsigned int r; cfg.page_size.divmod(a[0], q, r); out[0] = q; out[1] = r; }}"
    ));
    match r {
        Err(KernelLiftError::OutsideIr { construct, .. }) => {
            assert!(
                construct.contains("IntegerOperand") && construct.contains("U64"),
                "{construct}"
            );
        }
        other => panic!("expected the pinned IR gap, got {other:?}"),
    }
}

#[test]
fn a_method_writes_the_instance_through_this_and_the_conversion_operator_reads_it() {
    let k = lift(
        "struct Acc { float s; unsigned int n; void add(float v) { s += v; n += 1u; } float mean() const { return s / (float)n; } \
         operator unsigned int() const { return n; } };\n\
         void k(float* out) { Acc a; a.add(2.0f); a.add(3.5f); a.add(6.5f); unsigned int count = a; \
         out[0] = a.s; out[1] = (float)count; out[2] = a.mean(); out[3] = (float)(a + 10u); }",
    )
    .unwrap();
    let mut args = vec![KernelArg::Buffer(TypedBuffer::from_f32(&[4], &[0.0; 4]))];
    run_kernel(&k, &mut args).expect("runs");
    assert_eq!(out_u32(&args[0]), vec![12.0, 3.0, 4.0, 13.0]);
}

#[test]
fn an_early_return_becomes_an_if_whose_else_is_the_rest() {
    let k = lift(
        "__device__ float clampz(float x) { if (x < 0.0f) { return 0.0f; } if (x > 1.0f) { return 1.0f; } float y = x * 2.0f; return y; }\n\
         __device__ void settle(float x, float& out_v) { if (x < 0.0f) { out_v = -1.0f; return; } out_v = x + 100.0f; }\n\
         void k(float* out, const float* a) { for (int i = 0; i < 4; i++) { float v = 7.0f; settle(a[i], v); out[i] = clampz(a[i]); out[4 + i] = v; } }",
    )
    .unwrap();
    let a = [-3.0f32, 0.25, 1.0, 5.5];
    let mut args = vec![
        KernelArg::Buffer(TypedBuffer::from_f32(&[8], &[0.0; 8])),
        KernelArg::Buffer(TypedBuffer::from_f32(&[4], &a)),
    ];
    run_kernel(&k, &mut args).expect("runs");
    let clamp = |x: f32| {
        if x < 0.0 {
            0.0
        } else if x > 1.0 {
            1.0
        } else {
            x * 2.0
        }
    };
    let settle = |x: f32| if x < 0.0 { -1.0 } else { x + 100.0 };
    let want: Vec<f64> = a
        .iter()
        .map(|&x| f64::from(clamp(x)))
        .chain(a.iter().map(|&x| f64::from(settle(x))))
        .collect();
    assert_eq!(out_u32(&args[0]), want);
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
fn what_is_not_a_tail_return_or_a_plain_method_declines_by_name() {
    let s = "struct A { float v; float get() const { return v; } };\n";
    not_yet(
        &format!("{s}void k(float* out) {{ A a; out[0] = a.nope(); }}"),
        "no such method",
    );
    not_yet(
        &format!("{s}void k(float* out) {{ float f = 1.0f; out[0] = f.get(); }}"),
        "whose object is not a struct",
    );
    not_yet(
        "__device__ float f(float x) { if (x < 0.0f) { return 0.0f; } else { return 1.0f; } }\nvoid k(float* out) { out[0] = f(1.0f); }",
        "`if`/`else`",
    );
    not_yet(
        "__device__ float f(float x) { for (int i = 0; i < 2; i++) { if (x < 0.0f) { return 0.0f; } } return x; }\nvoid k(float* out) { out[0] = f(1.0f); }",
        "inside a loop",
    );
    not_yet(
        "__device__ float f(float x) { return x; float y = 1.0f; }\nvoid k(float* out) { out[0] = f(1.0f); }",
        "followed by more statements",
    );
    not_yet(
        &format!("{s}void k(float* out) {{ A a; out[0] = a; }}"),
        "no conversion operator",
    );
    // Control: the plain spelling lifts.
    assert!(
        lift(&format!(
            "{s}void k(float* out) {{ A a; a.v = 2.0f; out[0] = a.get(); }}"
        ))
        .is_ok()
    );
}
