//! `u32` arithmetic is modelled as **unsigned**, checked against Rust's own
//! `u32` rather than against a C compiler.
//!
//! # Why this exists alongside the e2e test
//!
//! `cpu_end_to_end.rs` has a `u32` leg that compiles and runs the emitted kernel
//! through a real C compiler. That is the stronger test — it checks the
//! *emitter* — but it **skips silently when no C compiler is on PATH**, which is
//! the case on at least one machine this repo is developed on. A test that
//! cannot run where it is written is not a test there.
//!
//! So this file verifies the half that is machine-independent: the **oracle's**
//! arithmetic model, differentially against Rust's native `u32`. Two genuinely
//! different implementations of one specification — the oracle evaluates in
//! `i128` with an explicit mask, Rust uses hardware 32-bit wrapping ops — so
//! agreement is evidence rather than a tautology.
//!
//! # Which mutation these actually catch
//!
//! Seeding "model u32 as signed" turns
//! `shr_of_a_composed_sum_is_logical_not_arithmetic` red, and nothing else. That
//! is not a weakness in the other tests — it is where the property lives.
//!
//! `+`, `-`, `*` and the bitwise ops produce identical **bit patterns** signed or
//! unsigned, and the result then round-trips through `u32` storage, which
//! launders any sign difference: `-1_294_967_296` and `3_000_000_000` encode to
//! the same 32 bits. So a single op cannot observe the model at all. Only a
//! composed expression can — an intermediate at or above 2³¹ that a signed model
//! sign-extends, feeding a `>>` that then propagates the sign.
//!
//! Correspondingly, the `wrap_for` calls on the shift and bitwise results are
//! **defensive consistency, not independently observable**: their inputs are
//! already non-negative, so signed and unsigned wrapping agree there. The one
//! load-bearing call is the one in `arith`. Saying which is which beats implying
//! every path is proven.
//!
//! What it does NOT cover, stated so nobody reads it as more: it does not check
//! that the emitted C says `unsigned int`, and it does not check that a C
//! compiler agrees. The first is asserted below as a text property; the second
//! needs the e2e leg and a compiler.

use unpopped::cpu_c::CpuC;
use unpopped::ir::{BinaryOp, OpDef, input};
use unpopped::oracle::{TypedBuffer, evaluate};
use unpopped::{build_plan, generate};
use unpopped_vocab::{ArchSku, ElementKind, OpCategory, OperandDesc, structure_key};

/// Values chosen to straddle the signed/unsigned boundary. Anything at or above
/// `2^31` reads as negative under a signed model, which is the bug being ruled
/// out; the small values are the control that ordinary arithmetic still works.
const A: &[u32] = &[
    0,
    1,
    7,
    2_147_483_647, // i32::MAX — last value a signed model gets right
    2_147_483_648, // i32::MAX + 1 — first value it gets wrong
    3_000_000_000,
    4_294_967_295, // u32::MAX
];

fn oracle_eval(op: &OpDef, a: &[u32], b: &[u32]) -> Vec<f64> {
    let n = a.len() as i64;
    let d = OperandDesc::new(1, &[n], &[1], ElementKind::U32, 4);
    let operands = vec![d; 3];
    let key = structure_key(OpCategory::BinaryElementwise, &operands, ArchSku::Sm89);
    let plan = build_plan(op, &key);
    let bufs = vec![
        TypedBuffer::from_u32(&[n], a),
        TypedBuffer::from_u32(&[n], b),
    ];
    evaluate(&plan, &operands, &bufs, &[])
        .into_iter()
        .next()
        .expect("one output")
        .to_f64_vec()
}

fn check(name: &str, op: OpDef, b: &[u32], rust: impl Fn(u32, u32) -> u32) {
    let got = oracle_eval(&op, A, b);
    assert_eq!(got.len(), A.len(), "{name}: wrong lane count");
    for (i, &g) in got.iter().enumerate() {
        let want = f64::from(rust(A[i], b[i]));
        assert!(
            (g - want).abs() < 0.5,
            "{name} lane {i}: a={} b={} — oracle {g}, Rust u32 {want}. \
             A negative oracle value here means u32 is being modelled as SIGNED.",
            A[i],
            b[i]
        );
        assert!(g >= 0.0, "{name} lane {i}: u32 result {g} is negative");
    }
}

#[test]
fn add_wraps_modulo_2_32_not_into_negatives() {
    let b: Vec<u32> = vec![0, 1, 5, 1, 1, 2_000_000_000, 1];
    check(
        "add",
        OpDef::elementwise("addu32", 2, &[ElementKind::U32], input(0) + input(1)),
        &b,
        u32::wrapping_add,
    );
}

#[test]
fn sub_and_mul_wrap_unsigned() {
    let b: Vec<u32> = vec![1, 2, 9, 1, 1, 7, 3];
    check(
        "sub",
        OpDef::elementwise("subu32", 2, &[ElementKind::U32], input(0) - input(1)),
        &b,
        u32::wrapping_sub,
    );
    check(
        "mul",
        OpDef::elementwise("mulu32", 2, &[ElementKind::U32], input(0) * input(1)),
        &b,
        u32::wrapping_mul,
    );
}

/// The load-bearing case, and it only appears in a **composed** expression.
///
/// # A single `>>` cannot detect this, and my first attempt at this test did
///
/// Loads zero-extend, so a bare `in0 >> k` starts from a non-negative wide value
/// and an `i128` shift of a non-negative value is logical either way. The result
/// then rounds back through `u32` storage, which launders any sign difference —
/// `-1_294_967_296` and `3_000_000_000` encode to the same 32 bits. Seeding a
/// signed model left every single-op assertion green.
///
/// The divergence needs an **intermediate** that a signed model would make
/// negative: `(a + b)` at or above 2³¹ sign-extends to a negative wide value, and
/// the following `>>` then propagates the sign. `(2_147_483_000 + 1000) >> 1` is
/// `1_073_742_000` unsigned and `-1_073_741_648` — encoding to `3_221_225_648` —
/// under an arithmetic shift. Different answer, no overflow, no panic.
///
/// This shape is legal at `u32` precisely because `unsigned int` does not
/// promote, so a composed operand reads the same inlined or hoisted (the
/// sub-`int` composition pin in `plan.rs` deliberately excludes it).
#[test]
fn shr_of_a_composed_sum_is_logical_not_arithmetic() {
    // Every lane's (a + b) lands at or above 2^31, where a signed model turns
    // negative — that is the whole point of the vector.
    let a: Vec<u32> = vec![
        2_147_483_000,
        3_000_000_000,
        4_000_000_000,
        2_147_483_648,
        4_294_967_295,
    ];
    let b: Vec<u32> = vec![1_000, 500_000_000, 100, 0, 0];
    let c: Vec<u32> = vec![1, 1, 3, 31, 31];

    let n = a.len() as i64;
    let d = OperandDesc::new(1, &[n], &[1], ElementKind::U32, 4);
    let operands = vec![d; 4]; // 3 inputs + 1 output
    let key = structure_key(OpCategory::BinaryElementwise, &operands, ArchSku::Sm89);
    let op = OpDef::elementwise(
        "shr_sum_u32",
        3,
        &[ElementKind::U32],
        (input(0) + input(1)).binary(BinaryOp::Shr, input(2)),
    );
    let plan = build_plan(&op, &key);
    let bufs = vec![
        TypedBuffer::from_u32(&[n], &a),
        TypedBuffer::from_u32(&[n], &b),
        TypedBuffer::from_u32(&[n], &c),
    ];
    let got = evaluate(&plan, &operands, &bufs, &[])
        .into_iter()
        .next()
        .expect("one output")
        .to_f64_vec();

    for (i, &g) in got.iter().enumerate() {
        let want = f64::from(a[i].wrapping_add(b[i]) >> c[i]);
        assert!(
            (g - want).abs() < 0.5,
            "lane {i}: ({} + {}) >> {} — oracle {g}, Rust u32 {want}.              A mismatch here means the intermediate sum was sign-extended and the              shift propagated the sign: u32 modelled as SIGNED.",
            a[i],
            b[i],
            c[i]
        );
    }
}

#[test]
fn bitwise_ops_agree_with_rust() {
    let b: Vec<u32> = vec![0xFFFF_FFFF, 0, 0xF0F0_F0F0, 1, 0xFFFF_0000, 0x0F0F_0F0F, 42];
    for (name, bop, f) in [
        (
            "and",
            BinaryOp::BitAnd,
            (|x: u32, y: u32| x & y) as fn(u32, u32) -> u32,
        ),
        ("or", BinaryOp::BitOr, |x: u32, y: u32| x | y),
        ("xor", BinaryOp::BitXor, |x: u32, y: u32| x ^ y),
    ] {
        check(
            name,
            OpDef::elementwise(
                "bitu32",
                2,
                &[ElementKind::U32],
                input(0).binary(bop, input(1)),
            ),
            &b,
            f,
        );
    }
}

/// The emitter spells `u32` as `unsigned int`, so the C compiler performs the
/// unsigned arithmetic the oracle models.
///
/// A text assertion, not a behavioural one — the behavioural check needs a C
/// compiler and lives in `cpu_end_to_end.rs`. It is here because the two halves
/// only mean something together: an oracle modelling unsigned semantics for a
/// kernel emitted as `int` would agree with nothing.
#[test]
fn the_emitted_kernel_is_spelled_unsigned() {
    let op = OpDef::elementwise("addu32", 2, &[ElementKind::U32], input(0) + input(1));
    let d = OperandDesc::new(1, &[7], &[1], ElementKind::U32, 4);
    let key = structure_key(OpCategory::BinaryElementwise, &[d, d, d], ArchSku::Sm89);
    let src = generate(&op, &key, &CpuC).source;
    assert!(
        src.contains("unsigned int"),
        "u32 must lower to `unsigned int`, or the oracle's unsigned model \
         describes a kernel the emitter did not write:\n{src}"
    );
    assert!(
        !src.contains(" int in0") && !src.contains("(int "),
        "u32 kernel must not spell a bare signed `int` operand:\n{src}"
    );
}
