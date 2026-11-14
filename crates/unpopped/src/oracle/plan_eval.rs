// SPDX-License-Identifier: MIT OR Apache-2.0
//! The public entry point [`evaluate`] and the per-access evaluators (elementwise, reduction, contraction, row reduce, scan, window, im2col), with the layout math they share.
//!
//! A section of the oracle (ROADMAP U30): `oracle.rs` glob-imports it, so every path it had is unchanged.

use super::*;

// ===========================================================================
// Layout / coordinate math.
// ===========================================================================

/// How input operand `k` is addressed: at the iteration coordinate through the
/// operand's own strides, through a permutation of them, or through the view's
/// own explicit strides ([`View::Layout`]).
#[derive(Clone, Copy)]
pub(super) enum ViewRead<'a> {
    Direct,
    Perm(&'a [u8]),
    Strides(&'a [i64]),
}

/// The read mode of input operand `k`. Mirrors the emitter's `input_perm` from
/// `plan.views` (an upstream layout fact), independently.
pub(super) fn input_read<'a>(plan: &'a KernelPlan<'_>, k: usize) -> ViewRead<'a> {
    match plan.views.get(k) {
        Some(View::Permute { perm }) => ViewRead::Perm(perm.as_slice()),
        Some(View::Layout { stride, .. }) => ViewRead::Strides(stride.as_slice()),
        _ => ViewRead::Direct,
    }
}

/// Unravel a linear index over `extents`, LAST-AXIS-FASTEST.
pub(super) fn unravel(mut lin: i64, extents: &[i64]) -> Vec<i64> {
    let mut c = vec![0i64; extents.len()];
    for d in (0..extents.len()).rev() {
        c[d] = lin % extents[d];
        lin /= extents[d];
    }
    c
}

/// Product of a slice of extents.
pub(super) fn prod(s: &[i64]) -> i64 {
    s.iter().product()
}

/// Read input operand `i` at coordinate vector `coords`, honoring signed strides
/// (broadcast = stride 0; flipped = negative stride), a `Permute` view, and the
/// buffer's runtime base offset. The independent analog of the emitter's
/// `offset_expr`: `off = base + Σ_d coords[d] * strides[perm[d]]`.
pub(super) fn read_strided(
    inputs: &[TypedBuffer],
    operands: &[OperandDesc],
    i: usize,
    coords: &[i64],
    read: ViewRead<'_>,
) -> Val {
    let od = &operands[i];
    let tb = &inputs[i];
    let rank = od.rank as usize;
    let mut off = tb.base_offset;
    for (d, &c) in coords.iter().enumerate().take(rank) {
        off += c * match read {
            ViewRead::Direct => od.strides[d],
            ViewRead::Perm(p) => od.strides[p[d] as usize],
            ViewRead::Strides(s) => s[d],
        };
    }
    assert!(
        off >= 0,
        "oracle: negative element offset {off} on operand {i}"
    );
    Val::Raw(read_elem(&tb.bytes, off as usize, od.dtype), od.dtype)
}

/// Read input operand `i` at FLAT element index `idx` (the RowReduce/Scan/Window
/// role-based load — `in_i[idx]`), honoring the base offset.
pub(super) fn read_flat(
    inputs: &[TypedBuffer],
    operands: &[OperandDesc],
    i: usize,
    idx: i64,
) -> Val {
    let od = &operands[i];
    let tb = &inputs[i];
    let off = tb.base_offset + idx;
    assert!(
        off >= 0,
        "oracle: negative flat offset {off} on operand {i}"
    );
    Val::Raw(read_elem(&tb.bytes, off as usize, od.dtype), od.dtype)
}

/// The role-based flat index for RowReduce/Scan/Window operand `i` at `(row, j)`,
/// classified by the SAME split as [`rr_role`]: RowStreamed `base+j`,
/// ColBroadcast `j`, RowScalar `row`.
pub(super) fn role_index(
    plan: &KernelPlan<'_>,
    i: usize,
    last: u8,
    base: i64,
    j: i64,
    row: i64,
) -> i64 {
    match rr_role(plan.key.operands[i], last) {
        RrRole::RowStreamed => base + j,
        RrRole::ColBroadcast => j,
        RrRole::RowScalar => row,
    }
}

/// Allocate a dense (row-major) zeroed output buffer for output operand `j`.
pub(super) fn alloc_output(
    plan: &KernelPlan<'_>,
    operands: &[OperandDesc],
    n_inputs: usize,
    j: usize,
) -> TypedBuffer {
    let od = &operands[n_inputs + j];
    let rank = od.rank as usize;
    let shape: Vec<i64> = od.shape[..rank].to_vec();
    let strides: Vec<i64> = od.strides[..rank].to_vec();
    let odt = plan.out_dtype_of(j);
    let size = prod(&shape).max(0) as usize;
    TypedBuffer::new(odt, shape, strides, vec![0u8; packed_bytes(odt, size)])
}

// ===========================================================================
// The public entry point.
// ===========================================================================

/// A [`View::Layout`] carries its own extents; they must be the iteration extents
/// (the output operand's shape, for the elementwise access it is admitted on). The
/// structure key abstracts extents away, so this is where a mismatch is caught.
pub(super) fn assert_layout_view_extents(plan: &KernelPlan<'_>, operands: &[OperandDesc]) {
    for (k, v) in plan.views.iter().enumerate() {
        if let View::Layout { shape, .. } = v {
            let out = &operands[plan.views.len()];
            assert_eq!(
                shape.as_slice(),
                &out.shape[..out.rank as usize],
                "oracle: input {k} Layout view extents differ from the iteration extents"
            );
        }
    }
}

/// Interpret `plan` over `operands` (inputs THEN outputs, `len ==
/// key.n_operands`), `inputs` (the input storage images), and `params` (runtime
/// scalar `Param` values), returning the computed output storage images.
///
/// This is a SEPARATE code path from the CUDA emitter — the whole point of the
/// differential test. See the module docs for the v1 access-pattern scope and
/// the v2 deferrals (Contraction / RowSort / gather-scatter panic here).
///
/// # Panics
/// On a v2-deferred access pattern, an unsupported dtype, or a malformed plan.
#[must_use]
pub fn evaluate(
    plan: &KernelPlan<'_>,
    operands: &[OperandDesc],
    inputs: &[TypedBuffer],
    params: &[f64],
) -> Vec<TypedBuffer> {
    assert_layout_view_extents(plan, operands);
    match plan.access {
        Access::Elementwise => eval_elementwise(plan, operands, inputs, params),
        Access::Reduction { .. } => vec![eval_reduction(plan, operands, inputs, params)],
        Access::RowReduce { .. } => vec![eval_row_reduce(plan, operands, inputs, params)],
        Access::Scan { .. } => vec![eval_scan(plan, operands, inputs, params)],
        Access::Window { .. } => vec![eval_window(plan, operands, inputs, params)],
        Access::Im2Col { .. } => vec![eval_im2col(plan, operands, inputs)],
        Access::Contraction { .. } => vec![eval_contraction(plan, operands, inputs, params)],
        Access::RowSort { .. } => eval_row_sort(plan, operands, inputs),
    }
}

// ===========================================================================
// H. RowSort (stable pair-sort, NaN-greatest).

/// Total order on the sort key with **NaN as the maximum**.
///
/// `Access::RowSort`'s doc pins it: *NaN orders GREATEST in both* — so
/// ascending puts NaN last and **descending puts NaN FIRST**. That is the arm a
/// reader expects to be wrong, so it is one comparator used twice rather than
/// two comparators that could drift apart.
///
/// `-0.0` and `0.0` compare Equal, matching the emitter's `<` on floats. The
/// sort is stable, so equal keys keep their original index order and the
/// signed-zero pair is not reordered.
pub(super) fn sort_key_cmp(a: f64, b: f64) -> core::cmp::Ordering {
    match (a.is_nan(), b.is_nan()) {
        (true, true) => core::cmp::Ordering::Equal,
        (true, false) => core::cmp::Ordering::Greater,
        (false, true) => core::cmp::Ordering::Less,
        (false, false) => a.partial_cmp(&b).expect("neither operand is NaN"),
    }
}

/// Per-row stable sort of the last axis.
///
/// The body is pinned to exactly `Input(0)` by `validate_row_sort`, so there is
/// no pre- or post-map to apply — this permutes, it does not compute. Values
/// move as **raw bits** rather than through `f64`, because a values-sort is a
/// permutation: a payload that round-tripped through `f64` would normalise a
/// signalling NaN and silently differ from what the kernel writes.
pub(super) fn eval_row_sort(
    plan: &KernelPlan<'_>,
    operands: &[OperandDesc],
    inputs: &[TypedBuffer],
) -> Vec<TypedBuffer> {
    let (order, out_kind) = match plan.access {
        Access::RowSort { order, out, .. } => (order, out),
        _ => unreachable!(),
    };
    let n_in = 1usize;
    let rank = plan.key.rank as usize;
    let in_od = &operands[0];
    let k_in = in_od.shape[rank - 1];
    let n_rows = if k_in > 0 {
        prod(&in_od.shape[..rank]).max(0) / k_in
    } else {
        0
    };

    let n_outputs = usize::from(matches!(out_kind, SortOut::Both)) + 1;
    let mut outs: Vec<TypedBuffer> = (0..n_outputs)
        .map(|j| alloc_output(plan, operands, n_in, j))
        .collect();

    for row in 0..n_rows {
        let base = row * k_in;
        let keys: Vec<f64> = (0..k_in)
            .map(|j| read_flat(inputs, operands, 0, base + j).f64())
            .collect();
        // `sort_by` is stable, and reversing the comparator does not change
        // which pairs compare Equal — so ties keep ascending original index in
        // BOTH orders, which is what the emitter's (key, index) pair-sort does.
        let mut perm: Vec<usize> = (0..k_in as usize).collect();
        perm.sort_by(|&a, &b| {
            let c = sort_key_cmp(keys[a], keys[b]);
            match order {
                SortOrder::Asc => c,
                SortOrder::Desc => c.reverse(),
            }
        });

        for (j, ob) in outs.iter_mut().enumerate() {
            // The output width may be SHORTER than the input for a TopK cap;
            // `validate_row_sort` allows exactly that and nothing else, so the
            // cap is read off the operand rather than from `SortLimit`.
            let k_out = operands[n_in + j].shape[rank - 1];
            let take = k_out.min(k_in).max(0) as usize;
            let obase = (row * k_out) as usize;
            let writes_indices =
                matches!((out_kind, j), (SortOut::Indices, 0) | (SortOut::Both, 1));
            for (t, &src) in perm.iter().take(take).enumerate() {
                if writes_indices {
                    store_val(
                        Val::Int(src as i128),
                        ElementKind::I32,
                        &mut ob.bytes,
                        obase + t,
                    );
                } else {
                    let bits = read_elem(
                        &inputs[0].bytes,
                        (inputs[0].base_offset + base + src as i64) as usize,
                        in_od.dtype,
                    );
                    let odt = plan.out_dtype_of(j);
                    write_elem(&mut ob.bytes, obase + t, odt, bits);
                }
            }
        }
    }
    outs
}

// ===========================================================================
// A. Elementwise (+ multi-output / hetero).
// ===========================================================================

pub(super) fn eval_elementwise(
    plan: &KernelPlan<'_>,
    operands: &[OperandDesc],
    inputs: &[TypedBuffer],
    params: &[f64],
) -> Vec<TypedBuffer> {
    let n_in = plan.n_inputs as usize;
    let rank = plan.key.rank as usize;
    let n_outputs = plan.n_outputs as usize;
    let out0 = &operands[n_in];

    let bodies = plan.output_bodies();
    let mut outs: Vec<TypedBuffer> = (0..n_outputs)
        .map(|j| alloc_output(plan, operands, n_in, j))
        .collect();

    // A GATHER is `Access::Elementwise` with one data-dependent read address —
    // the plan gate admits it nowhere else — so it belongs in this walk rather
    // than beside it. `None` for every ordinary elementwise op, and then every
    // branch below is inert.
    let gather = crate::plan::gather_of(plan.read_index);

    // A SCATTER is the mirror: `Access::Elementwise`, single output, one
    // data-dependent WRITE address. Its OOB policy is pinned to `Skip` by the
    // gate ("bespoke scatter/scatter_add/index_add/bincount all skip an OOB
    // target"), so there is no ZeroFill/Clamp branch to write here.
    //
    // **Two semantics worth stating rather than leaving to the reader:**
    //
    // 1. The destination starts ZEROED (`alloc_output`). That is right for
    //    `AtomicAdd` — scatter-add accumulates into zero — and it is a real
    //    assumption for `AtomicMax`/`Min`, where an all-negative source would
    //    max against the 0 that was already there. A kernel whose caller
    //    pre-initialises the destination differently will differ, and correctly.
    // 2. `Assign` with DUPLICATE indices is last-writer-wins with no defined
    //    order on device. This walks the iteration space in order, so it
    //    produces ONE valid outcome, not THE outcome. Compare against it only on
    //    a unique-index corpus; the other combines are order-independent and do
    //    not carry that caveat.
    let scatter = crate::plan::scatter_of(plan.write_index);

    // ⚠️ THE ITERATION SPACE IS THE SOURCE'S FOR A SCATTER, not the output's.
    //
    // Every other elementwise op produces one output element per iteration, so
    // walking the output shape is the same walk. **A scatter does not**: its
    // destination can be smaller than its source (bincount is the extreme —
    // thousands of elements into a handful of bins), and walking the output
    // would visit the bins and silently drop every source element past the
    // first few. The value operand is the input that is not the index; a scatter with
    // ONE input (bincount) indexes itself, so that input is the source. (Without this,
    // `usize::from(idx_op == 0)` named operand 1, the OUTPUT, and bincount counted only
    // the first `bins` elements of its data.)
    let src_slot = scatter.map(|(idx_op, ..)| {
        if n_in == 1 {
            0
        } else {
            usize::from(idx_op == 0)
        }
    });
    let ext: Vec<i64> = match src_slot {
        Some(v) => operands[v].shape[..rank].to_vec(),
        None => out0.shape[..rank].to_vec(),
    };
    let n_out = prod(&ext).max(0);

    for lin in 0..n_out {
        let coords = unravel(lin, &ext);

        // Resolve the index ONCE per output element, before any body runs.
        //
        // **Out-of-range is a STORE predicate, not a load behaviour** — the
        // policy docs are explicit that no OOB load occurs, because the emitter
        // clamps the load address in-bounds and guards the store. So the address
        // is always clamped here, and the OOB decision changes what is written:
        // `Skip` leaves the cell alone, `ZeroFill` writes zero, `Clamp` stores
        // the clamped read as an ordinary value. Reading OOB and then discarding
        // would agree on every output byte and disagree on whether the load
        // happened, which is exactly the difference a sanitizer sees.
        let mut gcoords = coords.clone();
        let mut oob_zero = false;
        if let Some((g, idx_op, axis, oob, _)) = gather {
            let axis = axis as usize;
            let extent = operands[g].shape[axis];
            let raw = i64::try_from(
                read_strided(
                    inputs,
                    operands,
                    idx_op as usize,
                    &coords,
                    input_read(plan, idx_op as usize),
                )
                .i128(),
            )
            .unwrap_or(i64::MIN);
            gcoords[axis] = raw.clamp(0, (extent - 1).max(0));
            if raw < 0 || raw >= extent {
                match oob {
                    crate::ir::OobPolicy::Skip => continue,
                    crate::ir::OobPolicy::ZeroFill => oob_zero = true,
                    crate::ir::OobPolicy::Clamp => {}
                }
            }
        }

        let leaf = |i: u8| {
            let c = if gather.is_some_and(|(g, ..)| g == i as usize) {
                &gcoords
            } else {
                &coords
            };
            read_strided(
                inputs,
                operands,
                i as usize,
                c,
                input_read(plan, i as usize),
            )
        };
        let coord = |d: u8| Val::Float(coords[d as usize] as f64);
        let ev = Eval {
            dtype: plan.dtype,
            params,
            leaf: &leaf,
            reduced: &panic_reduced,
            coord: &coord,
        };
        // Evaluate all output bodies first (they share the input reads), then store.
        let vals: Vec<Val> = (0..n_outputs)
            .map(|j| {
                if oob_zero {
                    // The op's zero fill, in the output's own domain: `Val::Int(0)`
                    // for an integer output (an exact 0 rather than a float that
                    // has to round back), `Val::Float(0.0)` otherwise (a float
                    // output has no integer encoding, and embedding's output is a
                    // float: `encode_int` panicked on it before).
                    if is_int(plan.out_dtype_of(j)) {
                        Val::Int(0)
                    } else {
                        Val::Float(0.0)
                    }
                } else {
                    eval(bodies[j], &ev)
                }
            })
            .collect();
        // Scatter redirects the STORE address along one axis. Resolved after the
        // bodies run, because an OOB target guards the store rather than the
        // read — the same shape as gather's policy, mirrored.
        let mut scoords = coords.clone();
        let mut combine = None;
        if let Some((idx_op, axis, cmb, _oob, _)) = scatter {
            let axis = axis as usize;
            let extent = operands[n_in].shape[axis];
            let raw = i64::try_from(
                read_strided(
                    inputs,
                    operands,
                    idx_op as usize,
                    &coords,
                    input_read(plan, idx_op as usize),
                )
                .i128(),
            )
            .unwrap_or(i64::MIN);
            // v1 pins the policy to Skip, so an out-of-range target is dropped.
            if raw < 0 || raw >= extent {
                continue;
            }
            scoords[axis] = raw;
            combine = Some(cmb);
        }

        for (j, val) in vals.into_iter().enumerate() {
            let od = &operands[n_in + j];
            let dst = if scatter.is_some() { &scoords } else { &coords };
            let off: i64 = dst
                .iter()
                .enumerate()
                .take(rank)
                .map(|(d, &c)| c * od.strides[d])
                .sum();
            let odt = plan.out_dtype_of(j);
            let val = match combine {
                None | Some(crate::ir::WriteCombine::Assign) => val,
                Some(cmb) => {
                    // Read-modify-write. Serial here, and that is exact for
                    // these three: integer add is associative and commutative,
                    // and max/min are too — the gate admits max/min for integers
                    // only, which is also what keeps `max(-0.0, +0.0)` from ever
                    // arising. Float `AtomicAdd` is order-dependent in general
                    // and is exact only on a corpus of integer-valued floats
                    // inside 2^24 (f32) / 2^53 (f64); see docs/deferred.md.
                    let prev = Val::Raw(read_elem(&outs[j].bytes, off as usize, odt), odt);
                    match cmb {
                        crate::ir::WriteCombine::AtomicAdd => {
                            if is_int(odt) {
                                Val::Int(prev.i128() + val.i128())
                            } else {
                                Val::Float(prev.f64() + val.f64())
                            }
                        }
                        crate::ir::WriteCombine::AtomicMax => Val::Int(prev.i128().max(val.i128())),
                        crate::ir::WriteCombine::AtomicMin => Val::Int(prev.i128().min(val.i128())),
                        crate::ir::WriteCombine::Assign => unreachable!("handled above"),
                    }
                }
            };
            store_val(val, odt, &mut outs[j].bytes, off as usize);
        }
    }
    outs
}

// ===========================================================================
// B. Reduction (fold over the reduced axes; post epilogue).
// ===========================================================================

pub(super) fn eval_reduction(
    plan: &KernelPlan<'_>,
    operands: &[OperandDesc],
    inputs: &[TypedBuffer],
    params: &[f64],
) -> TypedBuffer {
    let (rop, axes, keepdim, post) = match plan.access {
        Access::Reduction {
            op,
            axes,
            keepdim,
            post,
            accum: _,
        } => (*op, *axes, *keepdim, post),
        _ => unreachable!(),
    };
    let n_in = plan.n_inputs as usize;
    let rank = plan.key.rank as usize;
    let in0 = &operands[0];

    // Reduced set: EMPTY mask ⇒ the legacy last-axis default.
    let reduced: Vec<usize> = if axes.is_empty() {
        vec![rank - 1]
    } else {
        (0..rank).filter(|&d| axes.is_set(d as u8)).collect()
    };
    let kept: Vec<usize> = (0..rank).filter(|&d| !reduced.contains(&d)).collect();
    let kept_ext: Vec<i64> = kept.iter().map(|&a| in0.shape[a]).collect();
    let red_ext: Vec<i64> = reduced.iter().map(|&a| in0.shape[a]).collect();
    let divisor = prod(&red_ext) as f64;
    let n_kept = prod(&kept_ext).max(0);
    let n_red = prod(&red_ext).max(0);

    let mut out = alloc_output(plan, operands, n_in, 0);
    let out_dt = plan.out_dtype_of(0);
    let int_acc = is_int(plan.dtype);
    // Integer reductions: `I32`/`I64` (the emitter's scope) and, from 0.40.13, the narrow `I8`/`U8`/
    // `I16`/`U16` (ROADMAP E19), which fold at the dtype's width like the elementwise `Add`/`Mul`: the
    // accumulator is exact `i128` and the store wraps it to the dtype, which for `Sum` and `Prod`
    // equals wrapping at every step, and `Max`/`Min` are exact. Integer Mean is out of scope (the
    // emitter asserts `!(int_acc && Mean)`); keep the oracle's out-of-scope-PANIC convention
    // (cf. eval_scan/eval_window/eval_row_reduce) rather than returning the un-divided sum for a
    // buildable integer Mean plan. Every integer dtype is now in scope for `Sum`/`Prod`/`Max`/`Min`.
    if int_acc {
        assert!(
            matches!(
                plan.dtype,
                ElementKind::I32
                    | ElementKind::I64
                    | ElementKind::I8
                    | ElementKind::U8
                    | ElementKind::I16
                    | ElementKind::U16
                    | ElementKind::U32
                    | ElementKind::U64
            ),
            "oracle: integer reduction dtype {:?} out of scope (I8/U8/I16/U16/I32/U32/I64/U64 only)",
            plan.dtype
        );
        assert!(
            !matches!(rop, ReduceOp::Mean),
            "oracle: integer Mean reduction out of scope (emitter rejects int_acc && Mean)"
        );
    }

    for klin in 0..n_kept {
        let kc = unravel(klin, &kept_ext);
        // Fold over the reduced-axis cartesian product.
        let mut acc_f = if matches!(rop, ReduceOp::Prod) {
            1.0
        } else {
            0.0
        };
        let mut acc_i: i128 = if matches!(rop, ReduceOp::Prod) { 1 } else { 0 };
        let mut have = false;
        let mut best_idx = 0i64;
        for rlin in 0..n_red {
            let rc = unravel(rlin, &red_ext);
            let mut coords = vec![0i64; rank];
            for (j, &a) in kept.iter().enumerate() {
                coords[a] = kc[j];
            }
            for (j, &a) in reduced.iter().enumerate() {
                coords[a] = rc[j];
            }
            let leaf = |i: u8| {
                read_strided(
                    inputs,
                    operands,
                    i as usize,
                    &coords,
                    input_read(plan, i as usize),
                )
            };
            let ev = Eval {
                dtype: plan.dtype,
                params,
                leaf: &leaf,
                reduced: &panic_reduced,
                coord: &panic_coord,
            };
            let e = eval(plan.body, &ev);
            if matches!(rop, ReduceOp::ArgMax | ReduceOp::ArgMin) {
                let state = (int_acc, &mut acc_f, &mut acc_i, &mut best_idx);
                arg_fold(rop, e, rlin, state, &mut have);
            } else {
                fold_step(rop, e, int_acc, &mut acc_f, &mut acc_i, &mut have);
            }
        }
        // Finalize (Mean divides by the reduced-extent product; k==0 ⇒ 0/0 = NaN).
        let finalized: Val = if matches!(rop, ReduceOp::ArgMax | ReduceOp::ArgMin) {
            Val::Int(i128::from(best_idx))
        } else if int_acc {
            Val::Int(acc_i)
        } else if matches!(rop, ReduceOp::Mean) {
            Val::Float(acc_f / divisor)
        } else {
            Val::Float(acc_f)
        };
        // Apply the post epilogue (default identity Reduced(0)).
        let reduced_leaf = |s: u8| {
            assert_eq!(s, 0, "reduction post reads only Reduced(0)");
            finalized
        };
        let ev = Eval {
            dtype: plan.dtype,
            params,
            leaf: &(|i: u8| panic!("reduction post reads no Input({i})")),
            reduced: &reduced_leaf,
            coord: &panic_coord,
        };
        let posted = eval(post, &ev);
        // Output offset: keepdim ⇒ per-input-axis; collapse ⇒ per-kept-position.
        let off: i64 = if keepdim {
            kept.iter()
                .enumerate()
                .map(|(j, &a)| kc[j] * out.strides[a])
                .sum()
        } else {
            kept.iter()
                .enumerate()
                .map(|(j, _)| kc[j] * out.strides[j])
                .sum()
        };
        store_val(posted, out_dt, &mut out.bytes, off as usize);
    }
    out
}

// ===========================================================================
// B'. Contraction (matmul).
