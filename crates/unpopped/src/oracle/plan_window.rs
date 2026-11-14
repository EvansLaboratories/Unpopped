// SPDX-License-Identifier: MIT OR Apache-2.0
//! Scan, window and im2col evaluation.
//!
//! A section of the oracle (ROADMAP U30): `oracle.rs` glob-imports it, so every path it had is unchanged.

use super::*;

// ===========================================================================

pub(super) fn eval_scan(
    plan: &KernelPlan<'_>,
    operands: &[OperandDesc],
    inputs: &[TypedBuffer],
    params: &[f64],
) -> TypedBuffer {
    let (pre, post) = match plan.access {
        Access::Scan { pre, post, .. } => (pre, post),
        _ => unreachable!(),
    };
    let (sop, axis, reverse, exclusive) = match plan.schedule {
        Schedule::Scan {
            op,
            axis,
            reverse,
            exclusive,
            ..
        } => (op, axis, reverse, exclusive),
        _ => unreachable!(),
    };
    let n_in = plan.n_inputs as usize;
    let rank = plan.key.rank as usize;
    let last = (rank - 1) as u8;
    let out = &operands[n_in];
    // The scanned axis `axis` splits the tensor as (outer, k, inner): a "row" is one (outer, inner) pair, its element
    // `j` sits `inner` apart. Along the innermost axis `inner` is 1 and this is the dense row walk.
    let ax = usize::from(axis);
    let k = out.shape[ax];
    let inner = prod(&out.shape[ax + 1..rank]).max(1);
    let outer = prod(&out.shape[..ax]).max(0);
    let n_rows = if k > 0 { outer * inner } else { 0 };
    let int_acc = is_int(plan.dtype);

    let mut result = alloc_output(plan, operands, n_in, 0);
    let out_dt = plan.out_dtype_of(0);

    // Monoid identity in each domain.
    let (ident_f, ident_i) = scan_identity(sop, plan.dtype);

    for row in 0..n_rows {
        let base = (row / inner) * k * inner + row % inner;
        let mut acc_f = ident_f;
        let mut acc_i = ident_i;
        let mut have = false;
        for jj in 0..k {
            let j = if reverse { k - 1 - jj } else { jj };
            let leaf = |i: u8| {
                let at = if inner == 1 {
                    role_index(plan, i as usize, last, base, j, row)
                } else {
                    base + j * inner // a strided scan: every input is full-width dense
                };
                read_flat(inputs, operands, i as usize, at)
            };
            // pre has no running prefix.
            let ev_pre = Eval {
                dtype: plan.dtype,
                params,
                leaf: &leaf,
                reduced: &panic_reduced,
                coord: &panic_coord,
            };
            let v = eval(pre, &ev_pre);

            // The running prefix bound as Reduced(0) in `post`.
            let write = |prefix: Val, result: &mut TypedBuffer| {
                let reduced_leaf = |s: u8| {
                    assert_eq!(s, 0, "scan post reads only Reduced(0)");
                    prefix
                };
                let ev = Eval {
                    dtype: plan.dtype,
                    params,
                    leaf: &leaf,
                    reduced: &reduced_leaf,
                    coord: &panic_coord,
                };
                let out_val = eval(post, &ev);
                store_val(
                    out_val,
                    out_dt,
                    &mut result.bytes,
                    (base + j * inner) as usize,
                );
            };

            match sop {
                ReduceOp::Sum | ReduceOp::Prod => {
                    let combine = |a: Val, b: Val| -> Val {
                        if int_acc {
                            let r = match sop {
                                ReduceOp::Sum => a.i128().wrapping_add(b.i128()),
                                _ => a.i128().wrapping_mul(b.i128()),
                            };
                            Val::Int(wrap_bits(r, op_width(plan.dtype)))
                        } else {
                            let r = match sop {
                                ReduceOp::Sum => a.f64() + b.f64(),
                                _ => a.f64() * b.f64(),
                            };
                            Val::Float(r)
                        }
                    };
                    let acc = if int_acc {
                        Val::Int(acc_i)
                    } else {
                        Val::Float(acc_f)
                    };
                    if exclusive {
                        write(acc, &mut result);
                        let n = combine(acc, v);
                        set_acc(n, int_acc, &mut acc_f, &mut acc_i);
                    } else {
                        let n = combine(acc, v);
                        set_acc(n, int_acc, &mut acc_f, &mut acc_i);
                        let acc2 = if int_acc {
                            Val::Int(acc_i)
                        } else {
                            Val::Float(acc_f)
                        };
                        write(acc2, &mut result);
                    }
                }
                ReduceOp::Max | ReduceOp::Min => {
                    let ident = if int_acc {
                        Val::Int(ident_i)
                    } else {
                        Val::Float(ident_f)
                    };
                    let cur = if int_acc {
                        Val::Int(acc_i)
                    } else {
                        Val::Float(acc_f)
                    };
                    let update = |acc_f: &mut f64, acc_i: &mut i128, have: &mut bool| {
                        if int_acc {
                            let x = v.i128();
                            let better = if matches!(sop, ReduceOp::Max) {
                                x > *acc_i
                            } else {
                                x < *acc_i
                            };
                            if !*have || better {
                                *acc_i = x;
                            }
                        } else {
                            let x = v.f64();
                            let better = if matches!(sop, ReduceOp::Max) {
                                x > *acc_f
                            } else {
                                x < *acc_f
                            };
                            if !*have || x.is_nan() || better {
                                *acc_f = x;
                            }
                        }
                        *have = true;
                    };
                    if exclusive {
                        let prefix = if have { cur } else { ident };
                        write(prefix, &mut result);
                        update(&mut acc_f, &mut acc_i, &mut have);
                    } else {
                        update(&mut acc_f, &mut acc_i, &mut have);
                        let acc2 = if int_acc {
                            Val::Int(acc_i)
                        } else {
                            Val::Float(acc_f)
                        };
                        write(acc2, &mut result);
                    }
                }
                ReduceOp::Mean => {
                    // a running mean (an Unpopped extension, KISS#766): the running sum over the count folded so
                    // far, so an exclusive lane 0 is 0 / 0 = NaN
                    let count = (if exclusive { jj } else { jj + 1 }) as f64;
                    if exclusive {
                        write(Val::Float(acc_f / count), &mut result);
                        acc_f += v.f64();
                    } else {
                        acc_f += v.f64();
                        write(Val::Float(acc_f / count), &mut result);
                    }
                }
                ReduceOp::And
                | ReduceOp::Or
                | ReduceOp::Xor
                | ReduceOp::MaxNum
                | ReduceOp::MinNum
                | ReduceOp::ArgMax
                | ReduceOp::ArgMin => {
                    panic!("oracle: Scan rejects {sop:?}")
                }
            }
        }
    }
    result
}

/// Store a combined Val back into the active accumulator domain.
pub(super) fn set_acc(v: Val, int_acc: bool, acc_f: &mut f64, acc_i: &mut i128) {
    if int_acc {
        *acc_i = v.i128();
    } else {
        *acc_f = v.f64();
    }
}

/// The scan monoid identity in (float, int) domains. Max/Min identities are the
/// type extremes (the int extreme is only meaningful — and only computed — for an
/// integer dtype; the float slot carries `±inf`).
pub(super) fn scan_identity(sop: ReduceOp, dt: ElementKind) -> (f64, i128) {
    let ext = |neg: bool| if is_int(dt) { int_extreme(dt, neg) } else { 0 };
    match sop {
        ReduceOp::Sum => (0.0, 0),
        ReduceOp::Prod => (1.0, 1),
        ReduceOp::Max => (f64::NEG_INFINITY, ext(true)),
        ReduceOp::Min => (f64::INFINITY, ext(false)),
        ReduceOp::Mean => (0.0, 0),
        ReduceOp::And
        | ReduceOp::Or
        | ReduceOp::Xor
        | ReduceOp::MaxNum
        | ReduceOp::MinNum
        | ReduceOp::ArgMax
        | ReduceOp::ArgMin => {
            panic!("oracle: Scan rejects {sop:?}")
        }
    }
}

// ===========================================================================
// E. Window (sliding-window pool; count_include_pad divisor).
// ===========================================================================

#[allow(
    clippy::too_many_lines,
    reason = "`eval_window` is one linear evaluation of a window plan"
)]
pub(super) fn eval_window(
    plan: &KernelPlan<'_>,
    operands: &[OperandDesc],
    inputs: &[TypedBuffer],
    params: &[f64],
) -> TypedBuffer {
    let (pre, post) = match plan.access {
        Access::Window { pre, post, .. } => (pre, post),
        _ => unreachable!(),
    };
    let (wop, axis, size, stride, dilation, pad_lo, _pad_hi, count_include_pad) =
        match plan.schedule {
            Schedule::Window {
                op,
                axis,
                size,
                stride,
                dilation,
                pad_lo,
                pad_hi,
                count_include_pad,
            } => (
                op,
                axis,
                size,
                stride,
                dilation,
                pad_lo,
                pad_hi,
                count_include_pad,
            ),
            _ => unreachable!(),
        };
    let n_in = plan.n_inputs as usize;
    let rank = plan.key.rank as usize;
    let last = (rank - 1) as u8;
    let in0 = &operands[0];
    let out = &operands[n_in];
    // The pooled axis `axis` splits the tensor as (outer, k, inner): a "row" is one (outer, inner) pair and its
    // elements sit `inner` apart. Along the innermost axis `inner` is 1 and this is the dense row walk.
    let ax = usize::from(axis);
    let k_in = in0.shape[ax];
    let k_out = out.shape[ax];
    let inner = prod(&out.shape[ax + 1..rank]).max(1);
    let outer = prod(&out.shape[..ax]).max(0);
    // KISS-OPS-6.13-0011: a reader declines a window with no in-bounds tap (the oracle used to
    // answer with the monoid identity). The oracle has no error channel, so it panics, as its
    // other precondition failures do, with the typed error's text.
    if let Err(e) = crate::window_taps::check_window_taps(
        k_in,
        k_out,
        u32::from(size),
        u32::from(stride),
        u32::from(dilation),
        u32::from(pad_lo),
    ) {
        panic!("oracle: {e}");
    }
    let n_rows = if k_out > 0 { outer * inner } else { 0 };
    let int_acc = is_int(plan.dtype);

    let (size, stride, dil, plo) = (
        i64::from(size),
        i64::from(stride),
        i64::from(dilation),
        i64::from(pad_lo),
    );

    let mut result = alloc_output(plan, operands, n_in, 0);
    let out_dt = plan.out_dtype_of(0);

    for row in 0..n_rows {
        let (ob, ib) = (row / inner, row % inner);
        let base = ob * k_in * inner + ib;
        let out_base = ob * k_out * inner + ib;
        for o in 0..k_out {
            // Fold the valid taps.
            // a product starts at the multiplicative identity
            let one = matches!(wop, ReduceOp::Prod);
            let mut acc_f = if one { 1.0f64 } else { 0.0f64 };
            let mut acc_i: i128 = i128::from(one);
            let mut best_f = 0.0f64;
            let mut best_i: i128 = 0;
            let mut cnt = 0i64;
            let mut have = false;
            for kk in 0..size {
                let p = o * stride - plo + kk * dil;
                if p < 0 || p >= k_in {
                    continue;
                }
                let leaf = |i: u8| {
                    let at = if inner == 1 {
                        role_index(plan, i as usize, last, base, p, row)
                    } else {
                        base + p * inner // a strided window: every input is full-width dense
                    };
                    read_flat(inputs, operands, i as usize, at)
                };
                let ev = Eval {
                    dtype: plan.dtype,
                    params,
                    leaf: &leaf,
                    reduced: &panic_reduced,
                    coord: &panic_coord,
                };
                let v = eval(pre, &ev);
                match wop {
                    ReduceOp::Sum | ReduceOp::Mean => {
                        if int_acc {
                            acc_i = wrap_bits(acc_i.wrapping_add(v.i128()), op_width(plan.dtype));
                        } else {
                            acc_f += v.f64();
                        }
                        cnt += 1;
                    }
                    ReduceOp::Prod => {
                        if int_acc {
                            acc_i = wrap_bits(acc_i.wrapping_mul(v.i128()), op_width(plan.dtype));
                        } else {
                            acc_f *= v.f64();
                        }
                        cnt += 1;
                    }
                    ReduceOp::Max => {
                        if int_acc {
                            let x = v.i128();
                            if !have || x > best_i {
                                best_i = x;
                            }
                        } else {
                            let x = v.f64();
                            if !have || x.is_nan() || x > best_f {
                                best_f = x;
                            }
                        }
                        have = true;
                    }
                    ReduceOp::Min => {
                        if int_acc {
                            let x = v.i128();
                            if !have || x < best_i {
                                best_i = x;
                            }
                        } else {
                            let x = v.f64();
                            if !have || x.is_nan() || x < best_f {
                                best_f = x;
                            }
                        }
                        have = true;
                    }
                    ReduceOp::And
                    | ReduceOp::Or
                    | ReduceOp::Xor
                    | ReduceOp::MaxNum
                    | ReduceOp::MinNum
                    | ReduceOp::ArgMax
                    | ReduceOp::ArgMin => panic!("oracle: Window rejects {wop:?}"),
                }
            }
            // Finalize.
            let prefix: Val = match wop {
                ReduceOp::Sum | ReduceOp::Prod => {
                    if int_acc {
                        Val::Int(acc_i)
                    } else {
                        Val::Float(acc_f)
                    }
                }
                ReduceOp::Mean => {
                    // Float-only (asserted upstream).
                    let div = if count_include_pad {
                        size as f64
                    } else {
                        cnt as f64
                    };
                    Val::Float(if count_include_pad || cnt > 0 {
                        acc_f / div
                    } else {
                        0.0
                    })
                }
                ReduceOp::Max | ReduceOp::Min => {
                    let (ident_f, ident_i) = scan_identity(wop, plan.dtype);
                    if int_acc {
                        Val::Int(if have { best_i } else { ident_i })
                    } else {
                        Val::Float(if have { best_f } else { ident_f })
                    }
                }
                ReduceOp::And
                | ReduceOp::Or
                | ReduceOp::Xor
                | ReduceOp::MaxNum
                | ReduceOp::MinNum
                | ReduceOp::ArgMax
                | ReduceOp::ArgMin => unreachable!(),
            };
            // Epilogue.
            let leaf = |i: u8| {
                let at = if inner == 1 {
                    role_index(plan, i as usize, last, base, o, row)
                } else {
                    base + o * inner
                };
                read_flat(inputs, operands, i as usize, at)
            };
            let reduced_leaf = |s: u8| {
                assert_eq!(s, 0, "window post reads only Reduced(0)");
                prefix
            };
            let ev = Eval {
                dtype: plan.dtype,
                params,
                leaf: &leaf,
                reduced: &reduced_leaf,
                coord: &panic_coord,
            };
            let v = eval(post, &ev);
            store_val(
                v,
                out_dt,
                &mut result.bytes,
                (out_base + o * inner) as usize,
            );
        }
    }
    result
}

// ===========================================================================
// G. Im2Col (2-D expanding gather — BIT-EXACT raw copy).
// ===========================================================================

pub(super) fn eval_im2col(
    plan: &KernelPlan<'_>,
    operands: &[OperandDesc],
    inputs: &[TypedBuffer],
) -> TypedBuffer {
    let (kernel, stride, pad, dilation) = match plan.schedule {
        Schedule::Im2Col {
            kernel,
            stride,
            pad,
            dilation,
        } => (kernel, stride, pad, dilation),
        _ => unreachable!(),
    };
    let n_in = plan.n_inputs as usize;
    let in0 = &operands[0];
    let (n, c_in, h_in, w_in) = (in0.shape[0], in0.shape[1], in0.shape[2], in0.shape[3]);
    let (kh, kw) = (i64::from(kernel.0), i64::from(kernel.1));
    let (sh, sw) = (i64::from(stride.0), i64::from(stride.1));
    let (ph, pw) = (i64::from(pad.0), i64::from(pad.1));
    let (dh, dw) = (i64::from(dilation.0), i64::from(dilation.1));
    // Conv output spatial extents (the key carries no extents — derive here).
    let o_h = (h_in + 2 * ph - dh * (kh - 1) - 1) / sh + 1;
    let o_w = (w_in + 2 * pw - dw * (kw - 1) - 1) / sw + 1;

    let dt = in0.dtype;
    let sz = elem_size(dt);
    let total = n * c_in * kh * kw * o_h * o_w;

    let mut out = alloc_output(plan, operands, n_in, 0);
    let in_bytes = &inputs[0].bytes;
    let in_base = inputs[0].base_offset;

    for t in 0..total {
        let col = t % (o_h * o_w);
        let row_full = t / (o_h * o_w);
        let oh = col / o_w;
        let ow = col - oh * o_w;
        let patch = row_full % (c_in * kh * kw);
        let nn = row_full / (c_in * kh * kw);
        let cc = patch / (kh * kw);
        let kij = patch - cc * (kh * kw);
        let ki = kij / kw;
        let kj = kij - ki * kw;
        let in_h = oh * sh - ph + ki * dh;
        let in_w = ow * sw - pw + kj * dw;
        let dst = t as usize * sz;
        if in_h >= 0 && in_h < h_in && in_w >= 0 && in_w < w_in {
            let src = (((nn * c_in + cc) * h_in + in_h) * w_in + in_w) + in_base;
            let src_b = src as usize * sz;
            out.bytes[dst..dst + sz].copy_from_slice(&in_bytes[src_b..src_b + sz]);
        }
        // else: typed zero == zeroed bytes (already zero-initialized).
    }
    out
}
