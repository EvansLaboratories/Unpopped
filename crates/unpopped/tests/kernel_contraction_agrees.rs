// SPDX-License-Identifier: MIT OR Apache-2.0
//! **The layers agree on contractions** (`docs/ir-kernel-layer.md` section 7): a float matmul
//! (plain or batched) with the identity epilogue, lifted into the kernel layer and run by the
//! kernel interpreter, produces byte for byte what `oracle::evaluate` produces.
//!
//! The oracle reads each operand by its PHYSICAL strides at the logical coordinate, widens both
//! factors to `f64`, multiplies and accumulates in `f64` with `k` ascending, and rounds once at
//! the store. The corpus has an f32-cancelling column (1e8, 1, -1e8), NaN, infinities and signed
//! zeros, and the operands are also run transposed (column-major strides), because a lift that
//! assumed dense row-major would agree on every dense case and still be wrong.

use unpopped::ir::{ContractionAxes, Expr, OpDef, ScalarExpr, input, konst, reduced};
use unpopped::kernel::check_kernel;
use unpopped::kernel_lift::{LiftDecline, lift_contraction_opdef};
use unpopped::oracle::kernel_eval::{KernelArg, run_kernel};
use unpopped::oracle::{TypedBuffer, evaluate};
use unpopped::plan::try_build_plan;
use unpopped_vocab::{
    ArchSku, CellFacts, ContractionRoles, ElementKind, OperandDesc, derive_structure_key,
};

const LHS_VALUES: [f32; 12] = [
    1.0e8,
    1.0,
    -1.0e8,
    0.5,
    2.0,
    f32::NAN,
    1.0,
    2.0,
    f32::INFINITY,
    -0.0,
    0.1,
    -3.0,
];
const RHS_VALUES: [f32; 8] = [1.0, -1.0, 0.5, 2.0, 0.0, -0.0, 3.0, 0.25];

fn bytes_f32(vs: impl Iterator<Item = f32>) -> Vec<u8> {
    vs.flat_map(f32::to_le_bytes).collect()
}

/// A buffer of `shape` whose element at logical `coords` is `value(linear)` (linear in logical
/// row-major order), stored with `strides`.
fn buffer(shape: &[i64], strides: &[i64], value: impl Fn(usize) -> f32) -> TypedBuffer {
    let n: i64 = shape.iter().product();
    let extent: i64 = shape
        .iter()
        .zip(strides)
        .map(|(s, st)| (s - 1) * st)
        .sum::<i64>()
        + 1;
    let mut data = vec![0f32; extent as usize];
    for lin in 0..n {
        let (mut rem, mut off) = (lin, 0i64);
        for (s, st) in shape.iter().zip(strides).rev() {
            off += (rem % s) * st;
            rem /= s;
        }
        data[off as usize] = value(lin as usize);
    }
    TypedBuffer::new(
        ElementKind::F32,
        shape.to_vec(),
        strides.to_vec(),
        bytes_f32(data.into_iter()),
    )
}

fn matmul() -> OpDef {
    OpDef::contraction(
        "mm",
        &[ElementKind::F32],
        ContractionAxes::matmul(),
        reduced(0),
    )
}

fn batched() -> OpDef {
    OpDef::contraction(
        "bmm",
        &[ElementKind::F32],
        ContractionAxes::batched_matmul(),
        reduced(0),
    )
}

/// Run `op` through the oracle and through the lifted kernel on `lhs`, `rhs`.
fn both(op: &OpDef, lhs: &TypedBuffer, rhs: &TypedBuffer, out_shape: &[i64]) -> (Vec<u8>, Vec<u8>) {
    let out_strides: Vec<i64> = out_shape
        .iter()
        .enumerate()
        .map(|(i, _)| out_shape[i + 1..].iter().product())
        .collect();
    let dtype = lhs.dtype;
    let desc =
        |tb: &TypedBuffer| OperandDesc::new(tb.shape.len(), &tb.shape, &tb.strides, dtype, 256);
    let out = OperandDesc::new(out_shape.len(), out_shape, &out_strides, dtype, 256);
    let ops = [desc(lhs), desc(rhs), out];
    let rank = lhs.shape.len();
    let mut facts = CellFacts::new();
    facts.contraction_declared = true;
    facts.weight_operand = Some(1);
    facts.contraction_roles = Some(ContractionRoles::gemm(rank, usize::from(rank == 3)));
    let key = derive_structure_key(&ops, &facts, ArchSku::Sm89).expect("the key derives");
    let plan = try_build_plan(op, &key).expect("the contraction plans");
    let want = evaluate(&plan, &ops, &[lhs.clone(), rhs.clone()], &[])
        .remove(0)
        .bytes;

    let got = run_lifted(op, lhs, rhs, out_shape, out_strides, want.len());
    (want, got)
}

/// Lift `op`, run the kernel on `lhs`, `rhs`, and return the first `len` bytes of its output.
fn run_lifted(
    op: &OpDef,
    lhs: &TypedBuffer,
    rhs: &TypedBuffer,
    out_shape: &[i64],
    out_strides: Vec<i64>,
    len: usize,
) -> Vec<u8> {
    let dtype = lhs.dtype;
    let rank = lhs.shape.len();
    let kernel = lift_contraction_opdef(op, dtype).expect("the contraction lifts");
    assert_eq!(
        check_kernel(&kernel),
        Ok(()),
        "a lifted kernel is well formed"
    );
    let n_out: i64 = out_shape.iter().product();
    let mut args = vec![
        KernelArg::Buffer(lhs.clone()),
        KernelArg::Buffer(rhs.clone()),
        KernelArg::Buffer(TypedBuffer::new(
            dtype,
            out_shape.to_vec(),
            out_strides,
            vec![0u8; 8 * n_out as usize],
        )),
    ];
    // Extents: [B,] M, N, K.
    let (b, m, n, k) = if rank == 3 {
        (Some(lhs.shape[0]), lhs.shape[1], rhs.shape[2], lhs.shape[2])
    } else {
        (None, lhs.shape[0], rhs.shape[1], lhs.shape[1])
    };
    for e in b.into_iter().chain([m, n, k]) {
        args.push(KernelArg::Int(i128::from(e)));
    }
    run_kernel(&kernel, &mut args).expect("the kernel runs");
    let KernelArg::Buffer(got) = &args[2] else {
        unreachable!()
    };
    got.bytes[..len].to_vec()
}

fn lhs_value(i: usize) -> f32 {
    LHS_VALUES[i % LHS_VALUES.len()]
}

fn rhs_value(i: usize) -> f32 {
    RHS_VALUES[i % RHS_VALUES.len()]
}

#[test]
fn a_plain_matmul_agrees() {
    let lhs = buffer(&[3, 5], &[5, 1], lhs_value);
    let rhs = buffer(&[5, 4], &[4, 1], rhs_value);
    let (want, got) = both(&matmul(), &lhs, &rhs, &[3, 4]);
    assert_eq!(got, want);
}

#[test]
fn transposed_operands_agree() {
    // Column-major lhs ([3,5] with strides [1,3]) and rhs ([5,4] with strides [1,5]).
    let lhs = buffer(&[3, 5], &[1, 3], lhs_value);
    let rhs = buffer(&[5, 4], &[1, 5], rhs_value);
    let (want, got) = both(&matmul(), &lhs, &rhs, &[3, 4]);
    assert_eq!(got, want);
    // Control: the same logical values stored dense give the same answer, so the strided case is
    // exercising the view and not different data.
    let dense = both(
        &matmul(),
        &buffer(&[3, 5], &[5, 1], lhs_value),
        &buffer(&[5, 4], &[4, 1], rhs_value),
        &[3, 4],
    );
    assert_eq!(want, dense.0);
}

#[test]
fn a_batched_matmul_agrees() {
    let lhs = buffer(&[2, 3, 5], &[15, 5, 1], lhs_value);
    let rhs = buffer(&[2, 5, 4], &[20, 4, 1], rhs_value);
    let (want, got) = both(&batched(), &lhs, &rhs, &[2, 3, 4]);
    assert_eq!(got, want);
}

#[test]
fn the_accumulator_is_f64() {
    // Row 0 of the lhs is (1e8, 1, -1e8, 0.5, 2) and column 0 of the rhs is all 1.0 for the
    // values used here: the f64 sum is 3.5; an f32 accumulator would lose the 1 next to 1e8.
    let lhs = buffer(&[1, 5], &[5, 1], lhs_value);
    let rhs = buffer(&[5, 1], &[1, 1], |_| 1.0);
    let (want, got) = both(&matmul(), &lhs, &rhs, &[1, 1]);
    assert_eq!(got, want);
    let v = f32::from_le_bytes([got[0], got[1], got[2], got[3]]);
    assert_eq!(v, 3.5, "1e8 + 1 - 1e8 + 0.5 + 2 in f64");
}

#[test]
fn forms_the_lift_does_not_cover_decline_with_a_reason() {
    // Control: the plain matmul lifts.
    assert!(lift_contraction_opdef(&matmul(), ElementKind::F32).is_ok());
    let ew = OpDef::elementwise("e", 1, &[ElementKind::F32], Expr(input(0).0));
    assert_eq!(
        lift_contraction_opdef(&ew, ElementKind::F32),
        Err(LiftDecline::NotContraction)
    );
    // A fused epilogue (here a scale of the accumulator).
    let scaled = OpDef::contraction(
        "s",
        &[ElementKind::F32],
        ContractionAxes::matmul(),
        Expr(ScalarExpr::Mul(
            Box::new(ScalarExpr::Reduced(0)),
            Box::new(ScalarExpr::Const(2.0)),
        )),
    );
    // A fused epilogue lifts in tolerance mode (the tests below).
    assert!(lift_contraction_opdef(&scaled, ElementKind::F32).is_ok());
    // An epilogue that reads an input is not a function of the K-sum.
    let reads_input = OpDef::contraction(
        "ri",
        &[ElementKind::F32],
        ContractionAxes::matmul(),
        reduced(0) + input(0),
    );
    assert_eq!(
        lift_contraction_opdef(&reads_input, ElementKind::F32),
        Err(LiftDecline::UnsupportedLeaf(
            "an Input in a contraction epilogue"
        ))
    );
    // An integer contraction.
    let int = OpDef::contraction(
        "i",
        &[ElementKind::I32],
        ContractionAxes::matmul(),
        reduced(0),
    );
    assert_eq!(
        lift_contraction_opdef(&int, ElementKind::I32),
        Err(LiftDecline::UnsupportedForm("a non-float contraction"))
    );
}

#[test]
fn the_k_order_is_ascending_and_the_accumulator_f64_when_compute_is_f64() {
    // 1e16 + 1 - 1e16 + 1 + 0.1 + 0.2 + 0.3: the f64 fold is order-sensitive (1e16 swallows the 1),
    // so a descending or reordered k loop gives a different last-place answer. At f32 compute the
    // final rounding hides the order, which is why this case is f64.
    let vals = [1.0e16, 1.0, -1.0e16, 1.0, 0.1, 0.2, 0.3];
    let f64_bytes = |vs: &[f64]| vs.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<u8>>();
    let lhs = TypedBuffer::new(ElementKind::F64, vec![1, 7], vec![7, 1], f64_bytes(&vals));
    let rhs = TypedBuffer::new(
        ElementKind::F64,
        vec![7, 1],
        vec![1, 1],
        f64_bytes(&[1.0; 7]),
    );
    let op = OpDef::contraction(
        "mm64",
        &[ElementKind::F64],
        ContractionAxes::matmul(),
        reduced(0),
    );
    let (want, got) = both(&op, &lhs, &rhs, &[1, 1]);
    assert_eq!(got, want);
    // Control: the descending fold really is different, so the case discriminates.
    let desc: f64 = vals.iter().rev().fold(0.0, |a, v| a + v);
    let asc: f64 = vals.iter().fold(0.0, |a, v| a + v);
    assert_ne!(desc.to_bits(), asc.to_bits());
    assert_eq!(
        f64::from_le_bytes(got[..8].try_into().unwrap()).to_bits(),
        asc.to_bits()
    );
}

// ---------------------------------------------------------------------------------------------
// Fused epilogues, in tolerance mode (`docs/ir-kernel-layer.md` 7.1). The oracle hands the K-sum to
// the epilogue as an unrounded `f64`; the lift sets `Reduced(0) := Cast(f64 sum -> compute)`. So
// `F64` compute is byte-exact and `F32` compute is compared to within `MAX_ULPS` ulps (NaN equal,
// infinities exact), the oracle being a tolerance referee, as for `eval_contraction`. The bound is
// for well-conditioned epilogues (condition number `|s f'(s) / f(s)|` of about 1 or less). Measured
// on this corpus: `0.37 * s + 1` cancels near `s = -2.7` and moved by 71 ulps; `silu` at `s = -5`
// has condition number about 5 and moved by 3. Neither is a lift defect (rounding `s` once is the
// whole difference), so those shapes are not in the list below. `tanh` is evaluated at the compute
// dtype by the oracle as well, so it agrees exactly and is no probe of the tolerance; the shifted
// scale is (`the_tolerance_is_not_vacuous`).

/// The PM-approved bound for tolerance mode (2026-10-29).
const MAX_ULPS: u64 = 2;

fn ulps_f32(a: f32, b: f32) -> u64 {
    if a.is_nan() || b.is_nan() {
        return if a.is_nan() && b.is_nan() {
            0
        } else {
            u64::MAX
        };
    }
    let key = |x: f32| {
        let bits = i64::from(x.to_bits() as i32);
        if bits < 0 {
            i64::from(i32::MIN) - bits
        } else {
            bits
        }
    };
    key(a).abs_diff(key(b))
}

fn max_ulps_f32(want: &[u8], got: &[u8]) -> u64 {
    let f = |b: &[u8]| -> Vec<f32> {
        b.chunks(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
            .collect()
    };
    f(want)
        .into_iter()
        .zip(f(got))
        .map(|(a, b)| ulps_f32(a, b))
        .max()
        .unwrap_or(0)
}

fn buffer_f64(shape: &[i64], value: impl Fn(usize) -> f64) -> TypedBuffer {
    let n: i64 = shape.iter().product();
    let bytes = (0..n as usize)
        .flat_map(|i| value(i).to_le_bytes())
        .collect();
    let strides = (0..shape.len())
        .map(|i| shape[i + 1..].iter().product())
        .collect();
    TypedBuffer::new(ElementKind::F64, shape.to_vec(), strides, bytes)
}

/// Benign values: the epilogues below are not defined (or not interesting) on NaN and infinity.
fn benign(i: usize) -> f64 {
    [0.3, -1.7, 2.9, 0.11, -0.45, 1.3, 0.77, -2.2][i % 8] * (1.0 + (i / 8) as f64 * 0.07)
}

fn epilogues(dtype: ElementKind) -> Vec<(&'static str, OpDef)> {
    let mk = |name: &'static str, e: Expr| {
        OpDef::contraction(name, &[dtype], ContractionAxes::matmul(), e)
    };
    vec![
        (
            "scale and shift",
            mk("scale", reduced(0) * konst(0.37) + konst(10.0)),
        ),
        ("tanh", mk("tanh", reduced(0).tanh())),
        ("sigmoid", mk("sigmoid", reduced(0).sigmoid())),
    ]
}

#[test]
fn f64_compute_is_byte_exact_for_every_epilogue() {
    let lhs = buffer_f64(&[3, 5], benign);
    let rhs = buffer_f64(&[5, 4], |i| benign(i + 3));
    for (name, op) in epilogues(ElementKind::F64) {
        let (want, got) = both(&op, &lhs, &rhs, &[3, 4]);
        assert_eq!(got, want, "{name}");
    }
}

#[test]
fn f32_epilogues_are_within_the_tolerance() {
    let lhs = buffer(&[3, 5], &[5, 1], |i| benign(i) as f32);
    let rhs = buffer(&[5, 4], &[4, 1], |i| benign(i + 3) as f32);
    for (name, op) in epilogues(ElementKind::F32) {
        let (want, got) = both(&op, &lhs, &rhs, &[3, 4]);
        let worst = max_ulps_f32(&want, &got);
        assert!(
            worst <= MAX_ULPS,
            "{name}: {worst} ulps apart (bound {MAX_ULPS})"
        );
    }
}

#[test]
fn the_tolerance_is_not_vacuous() {
    // Control: rounding the K-sum to f32 before the epilogue really moves some cell, so the
    // tolerance cases are not secretly byte-exact.
    let mut moved = false;
    for m in 1..30i64 {
        let lhs = buffer(&[m, 5], &[5, 1], |i| benign(i % 16) as f32);
        let rhs = buffer(&[5, 4], &[4, 1], |i| benign(i + 3) as f32);
        let op = OpDef::contraction(
            "t",
            &[ElementKind::F32],
            ContractionAxes::matmul(),
            reduced(0) * konst(0.37) + konst(10.0),
        );
        let (want, got) = both(&op, &lhs, &rhs, &[m, 4]);
        assert!(max_ulps_f32(&want, &got) <= MAX_ULPS);
        moved |= want != got;
    }
    assert!(
        moved,
        "some cell differs between the unrounded and rounded K-sum"
    );
}

#[test]
fn a_wide_accumulator_is_required_under_an_epilogue() {
    // 1e8 + 1 - 1e8 + 0.5 + 2 = 3.5 in f64; an f32 accumulator loses the 1 and gets 2.5.
    let op = OpDef::contraction(
        "dbl",
        &[ElementKind::F32],
        ContractionAxes::matmul(),
        reduced(0) * konst(2.0),
    );
    let lhs = buffer(&[1, 5], &[5, 1], lhs_value);
    let rhs = buffer(&[5, 1], &[1, 1], |_| 1.0);
    let (want, got) = both(&op, &lhs, &rhs, &[1, 1]);
    assert!(max_ulps_f32(&want, &got) <= MAX_ULPS);
    assert_eq!(f32::from_le_bytes(got[..4].try_into().unwrap()), 7.0);
}

#[test]
fn a_batched_epilogue_agrees() {
    let lhs = buffer_f64(&[2, 3, 5], benign);
    let rhs = buffer_f64(&[2, 5, 4], |i| benign(i + 1));
    let op = OpDef::contraction(
        "bt",
        &[ElementKind::F64],
        ContractionAxes::batched_matmul(),
        reduced(0).tanh(),
    );
    let (want, got) = both(&op, &lhs, &rhs, &[2, 3, 4]);
    assert_eq!(got, want);
}

// ---------------------------------------------------------------------------------------------
// Ill-conditioned epilogues state their own bound (spec 7.1). Rounding the K-sum `s` once moves it
// by at most half an ulp, so a post `f` can move by `k / 2` ulp of its result, `k = |s f'(s) / f(s)|`;
// the bound for a cell is `ceil(k / 2) + 1` (the `+ 1` covers the post's own rounding). Each case
// shows both sides: every cell is inside its stated bound, and the stated bound is not vacuous
// because the default bound is broken somewhere (so loosening it is what the case needs).

/// The `f64` K-sums of an `[m, 5] x [5, 4]` product, row-major.
fn k_sums(lhs: &[f32], rhs: &[f32], m: usize) -> Vec<f64> {
    (0..m)
        .flat_map(|i| (0..4).map(move |j| (i, j)))
        .map(|(i, j)| {
            (0..5)
                .map(|k| f64::from(lhs[i * 5 + k]) * f64::from(rhs[k * 4 + j]))
                .sum()
        })
        .collect()
}

/// Run `epilogue` over `m` rows of the benign corpus; return `(ulps, bound)` per cell, with the
/// bound from `condition(s)`.
fn ulps_against_bound(epilogue: Expr, condition: impl Fn(f64) -> f64, m: usize) -> Vec<(u64, u64)> {
    let l: Vec<f32> = (0..m * 5).map(|i| benign(i % 16) as f32).collect();
    let r: Vec<f32> = (0..20).map(|i| benign(i + 3) as f32).collect();
    let lhs = buffer(&[m as i64, 5], &[5, 1], |i| l[i]);
    let rhs = buffer(&[5, 4], &[4, 1], |i| r[i]);
    let op = OpDef::contraction(
        "ill",
        &[ElementKind::F32],
        ContractionAxes::matmul(),
        epilogue,
    );
    let (want, got) = both(&op, &lhs, &rhs, &[m as i64, 4]);
    let f = |b: &[u8]| -> Vec<f32> {
        b.chunks(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
            .collect()
    };
    k_sums(&l, &r, m)
        .into_iter()
        .zip(f(&want).into_iter().zip(f(&got)))
        .map(|(s, (w, g))| (ulps_f32(w, g), (condition(s) / 2.0).ceil() as u64 + 1))
        .collect()
}

fn assert_states_its_own_bound(cells: &[(u64, u64)]) {
    let over_default = cells.iter().filter(|(u, _)| *u > MAX_ULPS).count();
    assert!(
        over_default > 0,
        "the default bound is never broken, so the case needs no bound of its own"
    );
    for (ulps, bound) in cells {
        assert!(ulps <= bound, "{ulps} ulps exceed the stated bound {bound}");
    }
    // Tight: the stated bound is within a factor of three of what happens on the worst cell.
    let (worst, bound) = cells
        .iter()
        .copied()
        .max_by_key(|(u, _)| *u)
        .expect("cells");
    assert!(
        bound <= 3 * worst,
        "bound {bound} is more than three times the worst case {worst}"
    );
}

#[test]
fn a_cancelling_shift_states_its_own_bound() {
    // 0.37 s + 1, condition number |0.37 s / (0.37 s + 1)|.
    let cells = ulps_against_bound(
        reduced(0) * konst(0.37) + konst(1.0),
        |s| (0.37 * s / (0.37 * s + 1.0)).abs(),
        12,
    );
    assert_states_its_own_bound(&cells);
}

#[test]
fn silu_states_its_own_bound() {
    // x * sigmoid(x): d/dx = sigmoid(x) (1 + x (1 - sigmoid(x))); condition number |x f' / f|.
    let sig = |x: f64| 1.0 / (1.0 + (-x).exp());
    let cells = ulps_against_bound(
        reduced(0).silu(),
        move |s| (1.0 + s * (1.0 - sig(s))).abs(),
        12,
    );
    assert_states_its_own_bound(&cells);
}
