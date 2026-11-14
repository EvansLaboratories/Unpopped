// SPDX-License-Identifier: MIT OR Apache-2.0
//! Contraction and row-reduce evaluation, with the fold steps they share.
//!
//! A section of the oracle (ROADMAP U30): `oracle.rs` glob-imports it, so every path it had is unchanged.

use super::*;

// ===========================================================================

/// Evaluate a rank-2 dense contraction `[m,k]·[k,n] → [m,n]` (the matmul axes the
/// plan gate has already asserted). Independent of the CUDA skinny-SIMT emitter:
/// a plain triple loop with the SAME accumulation ORDER (K ascending) and the
/// SAME `Reduced(0)` epilogue over the per-`(m,n)` K-sum.
///
/// Accumulates the K-sum in `f64` — the accurate reference. Like the reductions,
/// this oracle is a tolerance referee, not a bit-for-bit device mirror: the
/// emitter's `f32`/`double` accumulator rounding is absorbed by the tolerant
/// comparator (`Fidelity::Tolerant`), while an exactly-representable cell (small
/// integers) matches bit-for-bit.
///
/// v1 scope mirrors the emitter and `derive_contraction`: each operand is read
/// by its PHYSICAL strides at the logical (role) coordinate, so transposed/
/// permuted (`lhs_order`/`rhs_order` non-identity), batched, and broadcast-batch
/// operands all index correctly — the layout lives in the strides, not in a
/// second permutation applied on top of them (see [`read_strided`] and the
/// module's independence discipline). The epilogue reads only `Reduced(0)` (no
/// `Input`/`Coord`), plus the optional fused per-column bias.
pub(super) fn eval_contraction(
    plan: &KernelPlan<'_>,
    operands: &[OperandDesc],
    inputs: &[TypedBuffer],
    params: &[f64],
) -> TypedBuffer {
    let (epilogue, accum) = match plan.access {
        Access::Contraction {
            epilogue, accum, ..
        } => (epilogue, accum),
        _ => unreachable!("eval_contraction requires Access::Contraction"),
    };
    let n_in = plan.n_inputs as usize;
    let lhs = &operands[0];
    let rhs = &operands[1];
    // rank-3 lhs ⇒ batched `[B,M,K]·[B,K,N]`; rank-2 ⇒ plain (bdim = 1).
    let batched = lhs.rank == 3;
    let (bdim, m, k, n) = if batched {
        (lhs.shape[0], lhs.shape[1], lhs.shape[2], rhs.shape[2])
    } else {
        (1, lhs.shape[0], lhs.shape[1], rhs.shape[1])
    };
    let rhs_k = if batched { rhs.shape[1] } else { rhs.shape[0] };
    assert_eq!(
        rhs_k, k,
        "oracle: contraction K mismatch (lhs cols {k} vs rhs rows {rhs_k})"
    );

    let mut out = alloc_output(plan, operands, n_in, 0);
    let out_dt = plan.out_dtype_of(0);

    for bi in 0..bdim {
        for mi in 0..m {
            for ni in 0..n {
                // K-sum, `kk` ascending — the emitter's inner loop order.
                let pair = |kk: i64| {
                    let (lc, rc): (Vec<i64>, Vec<i64>) = if batched {
                        (vec![bi, mi, kk], vec![bi, kk, ni])
                    } else {
                        (vec![mi, kk], vec![kk, ni])
                    };
                    let a = read_strided(inputs, operands, 0, &lc, ViewRead::Direct).f64();
                    let b = read_strided(inputs, operands, 1, &rc, ViewRead::Direct).f64();
                    (a, b)
                };
                // an integer policy stores its i32 accumulator directly (the epilogue is the identity, checked by the gate)
                if let Some(v) = accum_policy::contract_int(accum, k, pair) {
                    let off = if batched {
                        bi * out.strides[0] + mi * out.strides[1] + ni * out.strides[2]
                    } else {
                        mi * out.strides[0] + ni * out.strides[1]
                    };
                    store_val(Val::Int(v), out_dt, &mut out.bytes, off as usize);
                    continue;
                }
                // a reduced-precision policy states its own conversion and rounding; the default accumulates in f64
                let acc = accum_policy::contract(accum, k, pair).unwrap_or_else(|| {
                    (0..k).fold(0.0f64, |acc, kk| {
                        let (a, b) = pair(kk);
                        acc + a * b
                    })
                });
                // Epilogue over Reduced(0) = the accumulator, plus the optional
                // fused bias: Input(i>=2) is the per-column `[N]` bias, read at
                // column `ni` (broadcast over rows) or a full `[M,N]` C at `[mi,ni]`; batched v1 carries
                // neither. No
                // Coord leaves. lhs/rhs (0/1) never appear.
                let reduced_leaf = |s: u8| {
                    assert_eq!(s, 0, "contraction epilogue reads only Reduced(0)");
                    Val::Float(acc)
                };
                let bias_leaf = |i: u8| {
                    assert!(
                        i as usize >= 2 && (i as usize) < n_in,
                        "contraction epilogue Input({i}) must be a fused bias (2..n_inputs)"
                    );
                    // A rank-1 operand is the per-column `[N]` bias; a rank-2 one is a full
                    // `[M,N]` C read at `[m,n]` (ROADMAP U7 S1a-2). The key separates them.
                    let extra = &operands[i as usize];
                    let at: &[i64] = match extra.rank {
                        1 => &[ni],
                        2 => &[mi, ni],
                        r => panic!("contraction epilogue Input({i}) has rank {r}, not 1 or 2"),
                    };
                    assert!(
                        extra.rank == 1 || (extra.shape[0] == m && extra.shape[1] == n),
                        "oracle: a full C is [M,N] = [{m},{n}], got {:?}",
                        extra.shape
                    );
                    read_strided(
                        inputs,
                        operands,
                        i as usize,
                        at,
                        input_read(plan, i as usize),
                    )
                };
                let ev = Eval {
                    dtype: plan.dtype,
                    params,
                    leaf: &bias_leaf,
                    reduced: &reduced_leaf,
                    coord: &panic_coord,
                };
                let posted = eval(epilogue, &ev);
                let off = if batched {
                    bi * out.strides[0] + mi * out.strides[1] + ni * out.strides[2]
                } else {
                    mi * out.strides[0] + ni * out.strides[1]
                };
                store_val(posted, out_dt, &mut out.bytes, off as usize);
            }
        }
    }
    out
}

/// One fold step for the have-flag NaN-propagating reduction combines.
pub(super) fn fold_step(
    rop: ReduceOp,
    e: Val,
    int_acc: bool,
    acc_f: &mut f64,
    acc_i: &mut i128,
    have: &mut bool,
) {
    match rop {
        ReduceOp::Sum | ReduceOp::Mean => {
            if int_acc {
                *acc_i = wrap_bits(acc_i.wrapping_add(e.i128()), 64);
            } else {
                *acc_f += e.f64();
            }
        }
        ReduceOp::Prod => {
            if int_acc {
                *acc_i = wrap_bits(acc_i.wrapping_mul(e.i128()), 64);
            } else {
                *acc_f *= e.f64();
            }
        }
        ReduceOp::Max | ReduceOp::Min | ReduceOp::MaxNum | ReduceOp::MinNum => {
            extremum_fold(rop, e, (int_acc, acc_f, acc_i), have);
        }
        _ => bit_fold(rop, e.i128(), (int_acc, acc_i), have),
    }
}

/// One step of `Max` / `Min` / `MaxNum` / `MinNum`: the first element seeds, a strictly better one replaces. The
/// float rules differ in NaN: `Max` / `Min` keep a NaN once seen; `MaxNum` / `MinNum` skip a NaN element and
/// replace a NaN accumulator, so a run of only NaN stays NaN. An integer has no NaN, so the four coincide in pairs.
pub(super) fn extremum_fold(
    rop: ReduceOp,
    e: Val,
    (int_acc, acc_f, acc_i): (bool, &mut f64, &mut i128),
    have: &mut bool,
) {
    let max = matches!(rop, ReduceOp::Max | ReduceOp::MaxNum);
    let ignore_nan = matches!(rop, ReduceOp::MaxNum | ReduceOp::MinNum);
    if int_acc {
        let v = e.i128();
        if !*have || (if max { v > *acc_i } else { v < *acc_i }) {
            *acc_i = v;
        }
    } else {
        let v = e.f64();
        let better = if max { v > *acc_f } else { v < *acc_f };
        let take = if ignore_nan {
            !*have || acc_f.is_nan() || (!v.is_nan() && better)
        } else {
            !*have || v.is_nan() || better
        };
        if take {
            *acc_f = v;
        }
    }
    *have = true;
}

/// The arms of [`fold_step`] not above: `And`, `Or`, `Xor`.
/// One step of an arg-reduction: `idx` is the element's position in the row-major order of the reduced axes. The
/// first element seeds; the rule is KISS-OPS-6.11-0007's: NaN orders as the **greatest** value and ties go to the
/// lower index. So `ArgMax` takes the first NaN and then stays (a NaN accumulator is final), and `ArgMin` never takes a
/// NaN while a non-NaN exists (a NaN accumulator is replaced by the first number; a NaN element is skipped). Only a
/// strictly better element replaces, so the first of equal values is kept.
pub(super) fn arg_fold(
    rop: ReduceOp,
    e: Val,
    idx: i64,
    (int_acc, acc_f, acc_i, best): (bool, &mut f64, &mut i128, &mut i64),
    have: &mut bool,
) {
    let max = matches!(rop, ReduceOp::ArgMax);
    let take = if !*have {
        true
    } else if int_acc {
        let v = e.i128();
        if max { v > *acc_i } else { v < *acc_i }
    } else {
        let v = e.f64();
        if max {
            !acc_f.is_nan() && (v.is_nan() || v > *acc_f)
        } else {
            !v.is_nan() && (acc_f.is_nan() || v < *acc_f)
        }
    };
    if take {
        if int_acc {
            *acc_i = e.i128();
        } else {
            *acc_f = e.f64();
        }
        *best = idx;
    }
    *have = true;
}

/// One step of an integer bit fold, seeded by the first element (no identity literal, as for `Max`/`Min`).
/// A float input is declined by the plan gate, and no other op reaches here: both are bugs.
pub(super) fn bit_fold(rop: ReduceOp, v: i128, (int_acc, acc): (bool, &mut i128), have: &mut bool) {
    assert!(
        int_acc && matches!(rop, ReduceOp::And | ReduceOp::Or | ReduceOp::Xor),
        "oracle: {rop:?} here is declined by the plan gate (a float bit fold, or a non-bit op)"
    );
    *acc = match (*have, rop) {
        (false, _) => v,
        (true, ReduceOp::And) => *acc & v,
        (true, ReduceOp::Or) => *acc | v,
        (true, _) => *acc ^ v,
    };
    *have = true;
}

// ===========================================================================
// C. RowReduce (per-row staged reductions, then a full-width epilogue).
// ===========================================================================

pub(super) fn eval_row_reduce(
    plan: &KernelPlan<'_>,
    operands: &[OperandDesc],
    inputs: &[TypedBuffer],
    params: &[f64],
) -> TypedBuffer {
    let (stages, epilogue) = match plan.access {
        Access::RowReduce { stages, epilogue } => (stages, epilogue),
        _ => unreachable!(),
    };
    let n_in = plan.n_inputs as usize;
    let rank = plan.key.rank as usize;
    let last = (rank - 1) as u8;
    let out = &operands[n_in];
    let k = out.shape[rank - 1];
    let total = prod(&out.shape[..rank]).max(0);
    let n_rows = if k > 0 { total / k } else { 0 };

    let mut result = alloc_output(plan, operands, n_in, 0);
    let out_dt = plan.out_dtype_of(0);

    for row in 0..n_rows {
        let base = row * k;
        // Run each stage in order → Reduced(0..n_stages).
        let mut reduced_vals: Vec<Val> = Vec::with_capacity(stages.len());
        for st in stages {
            // a product starts at the multiplicative identity (KISS-OPS-6.11-0002)
            let mut acc = if matches!(st.op, ReduceOp::Prod) {
                1.0f64
            } else {
                0.0f64
            };
            let mut have = false;
            for j in 0..k {
                let leaf = |i: u8| {
                    read_flat(
                        inputs,
                        operands,
                        i as usize,
                        role_index(plan, i as usize, last, base, j, row),
                    )
                };
                let reduced_leaf = |s: u8| reduced_vals[s as usize];
                let ev = Eval {
                    dtype: plan.dtype,
                    params,
                    leaf: &leaf,
                    reduced: &reduced_leaf,
                    coord: &panic_coord,
                };
                let e = eval(&st.pre, &ev).f64();
                match st.op {
                    ReduceOp::Sum | ReduceOp::Mean => acc += e,
                    ReduceOp::Prod => acc *= e,
                    ReduceOp::Max => {
                        if !have || e.is_nan() || e > acc {
                            acc = e;
                        }
                        have = true;
                    }
                    ReduceOp::Min => {
                        if !have || e.is_nan() || e < acc {
                            acc = e;
                        }
                        have = true;
                    }
                    ReduceOp::And
                    | ReduceOp::Or
                    | ReduceOp::Xor
                    | ReduceOp::MaxNum
                    | ReduceOp::MinNum
                    | ReduceOp::ArgMax
                    | ReduceOp::ArgMin => {
                        panic!("oracle: RowReduce {:?} stage is unsupported", st.op)
                    }
                }
            }
            let finalized = if matches!(st.op, ReduceOp::Mean) {
                acc / k as f64
            } else {
                acc
            };
            reduced_vals.push(Val::Float(finalized));
        }
        // Epilogue: full-width output, Reduced(s) = the finalized stage folds.
        for j in 0..k {
            let leaf = |i: u8| {
                read_flat(
                    inputs,
                    operands,
                    i as usize,
                    role_index(plan, i as usize, last, base, j, row),
                )
            };
            let reduced_leaf = |s: u8| reduced_vals[s as usize];
            let ev = Eval {
                dtype: plan.dtype,
                params,
                leaf: &leaf,
                reduced: &reduced_leaf,
                coord: &panic_coord,
            };
            let v = eval(epilogue, &ev);
            store_val(v, out_dt, &mut result.bytes, (base + j) as usize);
        }
    }
    result
}

// ===========================================================================
// D. Scan (serial prefix fold — the bit-reference).
