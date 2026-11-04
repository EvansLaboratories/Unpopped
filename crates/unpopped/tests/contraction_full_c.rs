// SPDX-License-Identifier: MIT OR Apache-2.0
//! **A full `[M,N]` C operand on a contraction** (ROADMAP U7 S1a-2, an Unpopped extension: KISS
//! pins no C operand). `D = alpha * (A.B) + beta * C` is an epilogue over the K-sum and an extra
//! input; a rank-2 extra input is a full C read at `[m, n]`, a rank-1 one stays the `[N]` bias
//! read at `n`. `StructureKey` already tells them apart (`a_full_c_and_a_bias_have_different_keys`
//! in `unpopped-vocab`), so a kernel cached for one is never found for the other.
//!
//! The reference below is written without the oracle. The shapes are `3x5 . 5x4` with `C` `3x4`,
//! so a C read at `[n]`, `[m]`, `[n, m]` or through the wrong row stride is a different number,
//! and every `C` value is distinct.

use unpopped::ir::{OpDef, input, konst, reduced};
use unpopped::kernel::check_kernel;
use unpopped::kernel_lift::{ContractionExtra, lift_contraction_opdef_with};
use unpopped::oracle::kernel_eval::{KernelArg, run_kernel};
use unpopped::oracle::{TypedBuffer, evaluate};
use unpopped::plan::try_build_plan;
use unpopped_vocab::{
    ArchSku, CellFacts, ContractionRoles, ElementKind, OperandDesc, derive_structure_key,
};

const M: i64 = 3;
const K: i64 = 5;
const N: i64 = 4;

fn buf(shape: &[i64], f: impl Fn(usize) -> f64) -> TypedBuffer {
    let n: i64 = shape.iter().product();
    let bytes = (0..n as usize).flat_map(|i| f(i).to_le_bytes()).collect();
    let strides = (0..shape.len())
        .map(|i| shape[i + 1..].iter().product())
        .collect();
    TypedBuffer::new(ElementKind::F64, shape.to_vec(), strides, bytes)
}

fn val(i: usize) -> f64 {
    [0.3, -1.7, 2.9, 0.11, -0.45, 1.3, 0.77, -2.2][i % 8] * (1.0 + (i / 8) as f64 * 0.07)
}

fn op() -> OpDef {
    OpDef::contraction_bias(
        "dc",
        &[ElementKind::F64],
        reduced(0) * konst(2.0) + input(2) * konst(0.5),
    )
}

/// The operands as descriptors: lhs, rhs, the extra input as given, out.
fn descs(extra: &TypedBuffer) -> Vec<OperandDesc> {
    let d = |tb: &TypedBuffer| {
        OperandDesc::new(
            tb.shape.len(),
            &tb.shape,
            &tb.strides,
            ElementKind::F64,
            256,
        )
    };
    let (a, b) = (buf(&[M, K], val), buf(&[K, N], |i| val(i + 3)));
    vec![
        d(&a),
        d(&b),
        d(extra),
        OperandDesc::new(2, &[M, N], &[N, 1], ElementKind::F64, 256),
    ]
}

fn key_of(
    ops: &[OperandDesc],
) -> Result<unpopped_vocab::StructureKey, unpopped_vocab::DeriveDecline> {
    let mut facts = CellFacts::new();
    facts.contraction_declared = true;
    facts.weight_operand = Some(1);
    facts.contraction_roles = Some(ContractionRoles::gemm(2, 0));
    derive_structure_key(ops, &facts, ArchSku::Sm89)
}

/// The oracle's bytes for `lhs.rhs` with `extra` as `Input(2)`.
fn oracle(extra: &TypedBuffer) -> Vec<u8> {
    let (a, b) = (buf(&[M, K], val), buf(&[K, N], |i| val(i + 3)));
    let ops = descs(extra);
    let key = key_of(&ops).expect("the key derives");
    let op = op();
    let plan = try_build_plan(&op, &key).expect("plans");
    evaluate(&plan, &ops, &[a, b, extra.clone()], &[])
        .remove(0)
        .bytes
}

/// `2 * sum_k a[m,k] b[k,n] + 0.5 * c(m, n)`, written out, `k` ascending.
fn reference(c: impl Fn(i64, i64) -> f64) -> Vec<u8> {
    let (a, b) = (
        |m: i64, k: i64| val((m * K + k) as usize),
        |k: i64, n: i64| val((k * N + n) as usize + 3),
    );
    let mut out = Vec::new();
    for m in 0..M {
        for n in 0..N {
            let mut acc = 0.0f64;
            for k in 0..K {
                acc += a(m, k) * b(k, n);
            }
            out.extend((acc * 2.0 + c(m, n) * 0.5).to_le_bytes());
        }
    }
    out
}

#[test]
fn a_full_c_is_read_at_its_own_row_and_column() {
    let c = buf(&[M, N], |i| 10.0 + i as f64 * 0.37);
    let cv = |m: i64, n: i64| 10.0 + (m * N + n) as f64 * 0.37;
    assert_eq!(oracle(&c), reference(cv));
    // The control: a different C (transposed index) is a different answer.
    assert_ne!(oracle(&c), reference(|m, n| cv(n.min(M - 1), m)));
}

#[test]
fn a_full_c_of_equal_rows_is_the_bias_it_replaces() {
    let bias = buf(&[N], |i| 10.0 + i as f64 * 0.37);
    let c = buf(&[M, N], |i| 10.0 + (i as i64 % N) as f64 * 0.37);
    assert_eq!(oracle(&c), oracle(&bias));
}

#[test]
fn the_lifted_kernel_agrees_with_the_oracle_on_a_full_c() {
    let c = buf(&[M, N], |i| 10.0 + i as f64 * 0.37);
    let want = oracle(&c);
    let kernel = lift_contraction_opdef_with(&op(), ElementKind::F64, ContractionExtra::FullC)
        .expect("the full-C contraction lifts");
    assert_eq!(check_kernel(&kernel), Ok(()));
    let (a, b) = (buf(&[M, K], val), buf(&[K, N], |i| val(i + 3)));
    let out = TypedBuffer::new(
        ElementKind::F64,
        vec![M, N],
        vec![N, 1],
        vec![0u8; want.len()],
    );
    let mut args = vec![
        KernelArg::Buffer(a),
        KernelArg::Buffer(b),
        KernelArg::Buffer(out),
        KernelArg::Buffer(c),
    ];
    for e in [M, N, K] {
        args.push(KernelArg::Int(i128::from(e)));
    }
    run_kernel(&kernel, &mut args).expect("the kernel runs");
    let KernelArg::Buffer(got) = &args[2] else {
        unreachable!()
    };
    assert_eq!(got.bytes, want);
    // The bias form of the same op reads `C[n]`, which is a different kernel and a different answer.
    let wrong =
        lift_contraction_opdef_with(&op(), ElementKind::F64, ContractionExtra::Bias).unwrap();
    assert_ne!(wrong, kernel);
}

/// The role-driven key (`derive_structure_key`) does not inspect an extra operand; the legacy
/// bridge does (`unpopped-vocab`). So a C of the wrong shape reaches the oracle, which refuses
/// it by name instead of reading out of bounds.
#[test]
#[should_panic(expected = "a full C is [M,N] = [3,4]")]
fn a_c_of_the_wrong_shape_is_refused_by_the_oracle() {
    oracle(&buf(&[N, M], |i| i as f64));
}

#[test]
fn a_column_major_c_is_read_through_its_own_strides() {
    // The same logical C, stored column-major: the oracle reads physical strides.
    let logical = |m: i64, n: i64| 10.0 + (m * N + n) as f64 * 0.37;
    let mut bytes = vec![0u8; (M * N) as usize * 8];
    for m in 0..M {
        for n in 0..N {
            let at = (n * M + m) as usize * 8;
            bytes[at..at + 8].copy_from_slice(&logical(m, n).to_le_bytes());
        }
    }
    let col_major = TypedBuffer::new(ElementKind::F64, vec![M, N], vec![1, M], bytes);
    assert_eq!(oracle(&col_major), reference(logical));
}
