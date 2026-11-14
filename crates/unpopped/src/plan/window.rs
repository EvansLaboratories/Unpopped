// SPDX-License-Identifier: MIT OR Apache-2.0
//! Section of the plan gate; split from `plan.rs` (ROADMAP U30). Paths unchanged.

use super::*;

/// Validate an [`Access::Window`] op at build time (increment 7; AOT — a window
/// never crosses the JIT trust boundary, so a panic here is an author-error
/// backstop). Mirrors [`validate_scan`]'s operand-role + layout checks, with the
/// pooling-specific window-parameter gate:
///
/// - **`Prod`** — a windowed product is not in the pool family (KISS has `avg_pool` and
///   `max_pool` only); since 0.42.9 it is admitted as an Unpopped extension, the product of the
///   in-bounds taps from the identity `1`. `Max`/`Min`/`Sum`/`Mean` are admitted.
/// - **`Mean` requires a float dtype** — an integer average has rounding
///   semantics (i32 `sum/count` truncates); avg_pool is float-only. `Max`/`Min`/
///   `Sum` are legal on the integer base too (max/min-pool select; sum-pool
///   wraps, matching the reduction Sum contract).
/// - **`axis < rank`** — any axis; a non-innermost axis (since 0.42.12) needs every
///   input full-width dense, the oracle walking it with the axis stride.
/// - **`size`/`stride`/`dilation >= 1`** — a zero window / stride / dilation is a
///   degenerate (empty-window) config.
/// - **`2*pad_lo <= span` and `2*pad_hi <= span`** where the tap footprint is
///   `span = dilation*(size-1) + 1` — each edge window must overlap the input by
///   at least 1 tap (the bespoke `pool1d` `pad*2 <= window` constraint,
///   generalized to dilation). A pad exceeding half the window would place an
///   entire edge window in padding.
///
/// The `in_len → out_len` window arithmetic (`out_len = floor((in_len + pad_lo +
/// pad_hi - dilation*(size-1) - 1)/stride) + 1`) is a **runtime-launch-arg caller
/// precondition**, NOT a plan-time check: [`StructureKey`] deliberately abstracts
/// numeric extents away (it carries per-operand contiguity/broadcast/flip, never
/// shapes), so the plan gate cannot see `in_len`/`out_len` — the same trust level
/// as RowReduce's `k`/`n_out` and `Coord`'s exact-integer extent bound. The output
/// operand's LAYOUT (forward-dense contiguous, downsampled extent) IS keyed and is
/// checked here.
#[allow(
    clippy::too_many_arguments,
    reason = "one argument per declared window parameter"
)]
pub(super) fn validate_window(
    op: &OpDef,
    key: &StructureKey,
    axis: u8,
    size: u8,
    stride: u8,
    dilation: u8,
    pad_lo: u8,
    pad_hi: u8,
) {
    let Access::Window {
        op: wop, pre, post, ..
    } = &op.access
    else {
        unreachable!("validate_window on a non-Window op");
    };
    let name = &op.name;

    // Mean (avg_pool) is float-only: an integer average rounds (i32 sum/count
    // truncates). Max/Min/Sum ride the integer base too.
    if matches!(wop, ReduceOp::Mean) {
        assert!(
            matches!(
                key.dtype,
                ElementKind::F16
                    | ElementKind::Bf16
                    | ElementKind::F32
                    | ElementKind::F32Strict
                    | ElementKind::F64
            ),
            "OpDef '{name}': Window Mean (avg_pool) requires a float dtype, got \
             {:?} — an integer average has rounding semantics (miss honestly; \
             integer max/min/sum-pool are legal)",
            key.dtype
        );
    }

    let n = op.n_inputs as usize;
    assert!(
        (1..MAX_OPERANDS).contains(&n),
        "Window n_inputs {} out of [1, MAX_OPERANDS)",
        op.n_inputs
    );
    assert!(
        key.n_operands as usize == n + 1,
        "Window expects n_inputs+1 operands (inputs then output); got {}",
        key.n_operands
    );
    let rank = key.rank as usize;
    assert!(rank >= 1, "Window needs a pooled axis (rank >= 1)");
    let last = (rank - 1) as u8;

    // Any axis of the rank (since 0.42.12). The innermost axis keeps the row-iteration skeleton (with column-broadcast
    // and per-row operands); another axis is strided, and every input must then be full-width dense.
    assert!(
        axis <= last,
        "Window axis {axis} is outside the rank {rank} (axes 0..={last})"
    );
    if axis != last {
        for i in 0..n {
            let o = key.operands[i];
            assert!(
                o.bcast.is_empty() && o.contig == Contiguity::Contig && !o.flipped,
                "Window over a non-innermost axis needs input {i} full-width dense (no broadcast, contiguous, not reversed)"
            );
        }
    }

    // Window-parameter legality — a degenerate (empty-window) config is a reject.
    assert!(
        size >= 1,
        "Window size must be >= 1 (an empty window has no taps)"
    );
    assert!(stride >= 1, "Window stride must be >= 1");
    assert!(dilation >= 1, "Window dilation must be >= 1");
    // span = the tap footprint (dilation*(size-1)+1); each edge window must overlap
    // the input by >= 1 tap, i.e. 2*pad <= span (bespoke `pool1d` pad*2 <= window,
    // generalized to dilation). u32 arithmetic avoids u8 overflow for large params.
    let span = u32::from(dilation) * (u32::from(size) - 1) + 1;
    assert!(
        2 * u32::from(pad_lo) <= span,
        "Window pad_lo {pad_lo} exceeds half the window span {span} \
         (dilation*(size-1)+1) — an entire low-edge window would fall in padding; \
         2*pad_lo <= span (bespoke pool1d pad*2 <= window)"
    );
    assert!(
        2 * u32::from(pad_hi) <= span,
        "Window pad_hi {pad_hi} exceeds half the window span {span} \
         (dilation*(size-1)+1) — an entire high-edge window would fall in padding; \
         2*pad_hi <= span (bespoke pool1d pad*2 <= window)"
    );

    // Operand roles + layout legality (mirrors validate_scan). Input 0 is the
    // row-streamed pooled tensor: `base = row*k_in` + the tap walk `idx = base+p`
    // assume a dense, forward inner axis, so it must be Contig and NOT flipped.
    let mut input0_streamed = false;
    #[allow(
        clippy::needless_range_loop,
        reason = "`i` is reported in the error and compared with 0 as well as indexing `key.operands`"
    )]
    for i in 0..n {
        let o = key.operands[i];
        match rr_role(o, last) {
            RrRole::RowStreamed => {
                assert!(
                    o.contig == Contiguity::Contig,
                    "Window streamed input {i} must be contiguous (base = row*k_in assumes a dense pooled axis)"
                );
                assert!(
                    !o.flipped,
                    "Window streamed input {i} must not be reversed along an axis (idx = base+p reads forward-dense; a flipped view reads mirrored/OOB)"
                );
                input0_streamed |= i == 0;
            }
            RrRole::ColBroadcast => {
                assert!(
                    !o.bcast.is_set(last) && (0..last).all(|d| o.bcast.is_set(d)),
                    "Window column input {i} must broadcast every outer axis and vary along the pooled axis (in_i[p])"
                );
                assert!(!o.flipped, "Window column input {i} must not be reversed");
            }
            RrRole::RowScalar => {
                assert!(
                    rank >= 2 && (0..last).all(|d| !o.bcast.is_set(d)),
                    "Window row-scalar input {i} needs rank >= 2 and no outer-axis broadcast (in_i[row]); an all-broadcast operand is a true scalar (bake as Const)"
                );
                assert!(
                    !o.flipped,
                    "Window row-scalar input {i} must not be reversed"
                );
            }
        }
    }
    assert!(
        input0_streamed,
        "Window Input0 must be the row-streamed pooled tensor, not a column-broadcast weight or a per-row scalar"
    );
    // The output is full-width forward-dense contiguous (a DOWNSAMPLED extent — the
    // caller sizes it via the window formula, a runtime precondition — but the same
    // layout class as the input's inner axis).
    // Not `out.bcast`: the key's mask is over the iteration frame, so a size-1 own axis
    // (a global pool, a top-1) beside a wider input reads as broadcast. `contig` is
    // `Broadcast` for a genuine stride-0 axis of extent > 1, so it alone rejects those.
    let out = key.operands[n];
    assert!(
        out.contig == Contiguity::Contig && !out.flipped,
        "Window output must be full-width forward-dense contiguous (empty bcast, not flipped)"
    );

    // Expression legality (mirrors validate_scan): `pre` (per-tap pre-map) runs
    // BEFORE the fold, so it must NOT read the window result (`Reduced` rejected in
    // `pre`); `post` (per-output epilogue) reads the result as the single
    // `Reduced(0)` leaf. Coord is rejected upstream (non-elementwise); Param is
    // f32-only (emitter). Input indices must be in range.
    fn check(e: &ScalarExpr, n_inputs: u8, allow_reduced: bool, ctx: &str, name: &str) {
        match e {
            ScalarExpr::Input(i) => assert!(
                *i < n_inputs,
                "Window '{name}' {ctx} Input({i}) >= n_inputs {n_inputs}"
            ),
            ScalarExpr::Reduced(s) => {
                assert!(
                    allow_reduced,
                    "Window '{name}' {ctx} must not read Reduced({s}) — the window result does not exist in the pre-map (it reads taps only)"
                );
                assert!(
                    *s == 0,
                    "Window '{name}' {ctx} Reduced({s}) — the window result is the single Reduced(0) leaf"
                );
            }
            ScalarExpr::Const(v) => assert!(
                v.is_finite(),
                "Window '{name}' {ctx} Const must be finite, got {v}"
            ),
            ScalarExpr::Param(_)
            | ScalarExpr::Coord(_)
            | ScalarExpr::CoordInt(..)
            | ScalarExpr::ConstInt(..)
            | ScalarExpr::Local(_)
            | ScalarExpr::Exec(_) => {}
            ScalarExpr::Unary(_, x) => check(x, n_inputs, allow_reduced, ctx, name),
            ScalarExpr::Cast(x, _)
            | ScalarExpr::Bitcast(x, _)
            | ScalarExpr::Qualified(x, _)
            | ScalarExpr::Extract(x, _)
            | ScalarExpr::BitcastPacked(x, ..) => check(x, n_inputs, allow_reduced, ctx, name),
            ScalarExpr::Pack(xs) => {
                for x in xs {
                    check(x, n_inputs, allow_reduced, ctx, name);
                }
            }
            ScalarExpr::Add(a, b)
            | ScalarExpr::Sub(a, b)
            | ScalarExpr::Mul(a, b)
            | ScalarExpr::Div(a, b)
            | ScalarExpr::Binary(_, a, b) => {
                check(a, n_inputs, allow_reduced, ctx, name);
                check(b, n_inputs, allow_reduced, ctx, name);
            }
            ScalarExpr::Select(c, a, b) | ScalarExpr::Fma(c, a, b) => {
                check(c, n_inputs, allow_reduced, ctx, name);
                check(a, n_inputs, allow_reduced, ctx, name);
                check(b, n_inputs, allow_reduced, ctx, name);
            }
        }
    }
    check(pre, op.n_inputs, false, "pre-map", name);
    check(post, op.n_inputs, true, "post-epilogue", name);
}

/// Validate an [`Access::Im2Col`] op at build time (increment 11; AOT — an im2col
/// never crosses the JIT trust boundary, so a panic here is an author-error
/// backstop). Unlike the rank-agnostic-innermost emitters (Scan/Window/RowSort
/// pin `axis == rank-1`), im2col has a FIXED rank-4 -> rank-3 conv layout, so this
/// gate PINS the operand ranks + layout the way `validate_window` pins the pooled
/// axis:
///
/// - **single input** (`n_inputs == 1`, no weight operand — a weighted window is
///   the deferred windowed-contraction), `n_operands == 2` (input then output).
/// - **input rank-4 forward-dense NCHW** — `key.rank == 4` (the input is also the
///   widest operand, so the shared `key.rank` is the input rank); input0 `Contig`,
///   `!flipped`, empty broadcast. The emitter's `(((n*C+c)*H_in+in_h)*W_in+in_w)`
///   address math assumes a dense row-major NCHW input.
/// - **output rank-3 forward-dense** — `Contig`, `!flipped`, empty broadcast. The
///   EXPANDED extent `[N, C*kh*kw, oH*oW]` is a runtime caller precondition (only
///   LAYOUT is keyed — the layout-only gate admits the expansion).
/// - **geometry legality** — `kh,kw >= 1`, `stride.* >= 1`, `dilation.* >= 1`
///   (`pad.*` is a `u8`, always `>= 0`). A zero kernel/stride/dilation is a
///   degenerate config.
/// - **`out_dtype == None`** (a pure gather preserves dtype;
///   `assert_valid_out_dtype` double-gates this) and **`body == Input(0)`** (a pure
///   raw-bit copy — no pre/post in v1).
///
/// The `(H_in,W_in) -> (oH,oW)` conv arithmetic
/// (`oH = (H_in + 2*pad_h - dilation_h*(kh-1) - 1)/stride_h + 1`, `oW` symmetric)
/// is a **runtime-launch-arg caller precondition**, NOT a plan-time check:
/// [`StructureKey`] deliberately abstracts numeric extents away (it carries
/// per-operand contiguity/broadcast/flip, never shapes), so the plan gate cannot
/// see `H_in`/`oH` — the same trust level as Window's `k_in -> k_out` and RowReduce's
/// `k`/`n_out`. It is on-device-validated via `initcheck` (full write + in-bounds
/// source reads); see the increment-11 brief §3/§6.
pub(super) fn validate_im2col(
    op: &OpDef,
    key: &StructureKey,
    kernel: (u8, u8),
    stride: (u8, u8),
    pad: (u8, u8),
    dilation: (u8, u8),
) {
    let Access::Im2Col { .. } = &op.access else {
        unreachable!("validate_im2col on a non-Im2Col op");
    };
    let name = &op.name;
    let _ = pad; // pad.* is u8 (>= 0); baked into the caller's oH/oW, checked on-device.

    // G1 — single input (no weight operand), input + output operands only.
    assert!(
        op.n_inputs == 1,
        "Im2Col '{name}': v1 takes a SINGLE input (n_inputs {} != 1) — a weighted \
         window is the deferred windowed-contraction, not an im2col",
        op.n_inputs
    );
    assert!(
        key.n_operands == 2,
        "Im2Col '{name}': expects n_inputs+1 = 2 operands (input then output); got {}",
        key.n_operands
    );

    // G2 — input rank-4 forward-dense NCHW. The shared `key.rank` is the widest
    // operand rank; the rank-4 NCHW input is the widest (output is rank-3), so
    // `key.rank == 4` pins the input rank. A rank-3 (or other) input drives the max
    // below 4 and rejects here (the miss is honest, not silently wrong).
    assert!(
        key.rank == 4,
        "Im2Col '{name}': v1 pins a rank-4 NCHW input [N,C,H_in,W_in] (key.rank {} \
         != 4) — 1-D/3-D im2col are deferred rank variants",
        key.rank
    );
    let in0 = key.operands[0];
    assert!(
        in0.contig == Contiguity::Contig,
        "Im2Col '{name}': input 0 must be dense contiguous NCHW (the \
         (((n*C+c)*H_in+in_h)*W_in+in_w) address math assumes row-major)"
    );
    assert!(
        !in0.flipped,
        "Im2Col '{name}': input 0 must not be reversed along an axis (a flipped view \
         reads mirrored/OOB source coords)"
    );

    // G3 — output rank-3 forward-dense. Only LAYOUT is keyed (the expanded extent
    // [N, C*kh*kw, oH*oW] is a runtime precondition), so the gate admits the
    // expansion so long as the output is forward-dense contiguous.
    //
    // The key's `bcast` mask is NOT consulted. It is over the iteration frame
    // (KISS-CLASSIFY §6.5-0014 as amended by KISS#519, §6.6-0008), so it also records
    // "absent from this frame axis" (the rank-3 output never spans frame axis 0) and
    // "own axis of extent 1 beside a wider frame" (N = 1 beside C > 1, or a 1x1 output
    // beside a wider input). Neither is a broadcast, and a mask test rejected both.
    // A genuine own-axis broadcast (extent > 1, stride 0) makes `contig` read
    // `Broadcast`, and a strided output reads `Strided`/`InnerContig`, so the
    // `contig == Contig` clause alone rejects every output that is not dense.
    let out = key.operands[1];
    assert!(
        out.contig == Contiguity::Contig && !out.flipped,
        "Im2Col '{name}': output must be forward-dense contiguous (empty broadcast, \
         not flipped) — the [N, C*kh*kw, oH*oW] column matrix is written densely"
    );

    // G4 — geometry legality (a zero kernel/stride/dilation is degenerate).
    assert!(
        kernel.0 >= 1 && kernel.1 >= 1,
        "Im2Col '{name}': kernel (kh,kw) = ({},{}) must be >= 1 on each axis",
        kernel.0,
        kernel.1
    );
    assert!(
        stride.0 >= 1 && stride.1 >= 1,
        "Im2Col '{name}': stride (sh,sw) = ({},{}) must be >= 1 on each axis",
        stride.0,
        stride.1
    );
    assert!(
        dilation.0 >= 1 && dilation.1 >= 1,
        "Im2Col '{name}': dilation (dh,dw) = ({},{}) must be >= 1 on each axis",
        dilation.0,
        dilation.1
    );

    // G5 — dtype-preserving pure gather: out_dtype None (double-gated in
    // assert_valid_out_dtype) and body pinned to a raw Input(0) (no pre/post — a
    // composed body would need arithmetic the raw-bit copy does not emit).
    assert!(
        op.out_dtype.is_none(),
        "Im2Col '{name}': a pure gather preserves dtype — out_dtype must be None, got \
         {:?}",
        op.out_dtype
    );
    assert!(
        matches!(op.body, ScalarExpr::Input(0)),
        "Im2Col '{name}': body must be exactly Input(0) (a raw-bit copy) — v1 has no \
         pre/post map"
    );
}

/// Validate an [`Access::RowSort`] op at build time (increment 8; AOT — a sort
/// never crosses the JIT trust boundary, so a panic here is an author-error
/// backstop). Mirrors [`validate_scan`]/[`validate_window`]'s operand-role +
/// layout checks, with the sort-specific gates:
///
/// - **`stable == true` only** — v1 ALWAYS emits the stable pair-sort (`(key,
///   original-index)` with index tie-break), so stability is free; an unstable
///   network would emit byte-identical code under a different symbol (dead
///   keying) and admitting-then-ignoring the flag would be dishonest keying.
///   `stable: false` rejects.
/// - **dtype** — admit `F32|F32Strict|F64|F16|Bf16|I32|I64`; reject `U32`
///   (index-only, no value dtype) and `S8|U8` (v1 de-scope; the bespoke argsort
///   covers small ints; liftable by widening this gate + the validator cells).
/// - **out_dtype ↔ [`SortOut`] state coupling** — `Indices ⇒ out_dtype ==
///   Some(I32)`; `Values | Both ⇒ out_dtype == None` (Both's I32 out_idx is
///   emitter-hardwired, off the symbol, not out_dtype). Double-gated (also
///   `assert_valid_out_dtype`).
/// - **`body` must be exactly `Input(0)`** — v1 has no pre/post; this single
///   equality replaces the recursive `check` machinery of validate_scan/window
///   (it rejects Param/Coord/Reduced/Const/composed bodies in one stroke).
/// - **`extra_out_bodies` empty** — multi-output does not ride RowSort (also
///   caught by `assert_valid_multi_output`; double-gated). `Both` carries its
///   second buffer via the `SortOut` state + the 3-operand key, NEVER a body.
/// - **operand count** — `Values | Indices` carry `n_inputs + 1` operands
///   (`[in0, out]`); `Both` carries `n_inputs + 2` (`[in0, out_val, out_idx]`).
/// - **Input 0 layout** — `RowStreamed` + `Contig` + `!flipped` (the emitters
///   read `in0[base+j]` stride-free; a flipped/strided input reads mirrored/OOB).
/// - **Output layout** — every output (out / out_val AND out_idx for `Both`) is
///   empty bcast + `Contig` + `!flipped`. This gate is LAYOUT-only (it does NOT
///   assert the output WIDTH), so a `TopK` cell's narrower `[batch, k_out]` output
///   passes as written — no relaxation needed.
///
/// There is no axis field: RowSort is innermost-axis by definition (matching
/// Scan/Window's `axis == rank-1` posture). The bitonic variant's `k <= 1024`
/// bound is NOT checkable here (the structure key carries no numeric extents) —
/// it is a `launch_note` precondition + on-device-validated contract, the same
/// trust level as smemrow/blockscan; the base rank sort has no length limit.
///
/// - **`limit` (increment 10 TOPK/BOTTOMK)** — `Full` is today's whole-row sort;
///   `TopK` caps the writeback to the first `k_out` ranks. `k_out` is the OUT
///   operand's inner extent, a `long long` launch arg (the Window `(n_out, k_in,
///   k_out)` precedent) — the structure key carries NO numeric extent, so the
///   `k_out <= k_in` precondition is NOT expressible as a plan assert. It is a
///   runtime launch precondition, on-device-validated by `initcheck` (proves all
///   `k_out` slots written AND no over-write past `k_out`), the same trust tier as
///   the bitonic `k <= 1024`. `Indices`/`Both` under `TopK` inherit argsort's
///   `k_in <= 2^31-1` cap (the `I32` index).
pub(super) fn validate_row_sort(
    op: &OpDef,
    key: &StructureKey,
    _order: SortOrder,
    stable: bool,
    out: SortOut,
    _limit: SortLimit,
) {
    let name = &op.name;

    // v1 emits only the stable pair-sort. Reject stable=false before anything else
    // so the message is unambiguous.
    assert!(
        stable,
        "OpDef '{name}': row_sort v1 emits only the stable pair-sort (stable=true); \
         unstable declined (an unstable network would emit byte-identical code under \
         a different symbol — dead keying)"
    );

    // dtype gate: the v1 comparable value set. U32 is index-only (no value dtype);
    // S8/U8 are a v1 de-scope (bespoke argsort covers small ints).
    assert!(
        matches!(
            key.dtype,
            ElementKind::F32
                | ElementKind::F32Strict
                | ElementKind::F64
                | ElementKind::F16
                | ElementKind::Bf16
                | ElementKind::I32
                | ElementKind::I64
        ),
        "OpDef '{name}': row_sort dtype {:?} is out of the v1 set \
         (F32/F32Strict/F64/F16/Bf16/I32/I64) — U32 is index-only and S8/U8 are a \
         v1 de-scope (miss honestly)",
        key.dtype
    );

    // out_dtype ↔ SortOut state coupling (double-gated with assert_valid_out_dtype).
    // G1: Indices ⇒ Some(I32); Values | Both ⇒ None (Both's I32 out_idx is
    // emitter-hardwired off the symbol, not carried by out_dtype).
    match out {
        SortOut::Indices => assert!(
            op.out_dtype == Some(ElementKind::I32),
            "OpDef '{name}': a row argsort (SortOut::Indices) produces an I32 index \
             output — out_dtype must be Some(I32), got {:?}",
            op.out_dtype
        ),
        SortOut::Values | SortOut::Both => assert!(
            op.out_dtype.is_none(),
            "OpDef '{name}': a values-sort (Values) / fused two-output (Both) is \
             dtype-preserving on output 0 — out_dtype must be None (Both's I32 \
             out_idx is emitter-hardwired), got {:?}",
            op.out_dtype
        ),
    }

    // body must be exactly Input(0): v1 has no pre/post, so a single equality check
    // rejects Param/Coord/Reduced/Const/composed bodies in one stroke.
    assert!(
        matches!(op.body, ScalarExpr::Input(0)),
        "OpDef '{name}': row_sort body must be exactly Input(0) (v1 has no pre/post \
         map); got {:?}",
        op.body
    );

    // Multi-output does not ride RowSort (also caught by assert_valid_multi_output).
    assert!(
        op.extra_out_bodies.is_empty(),
        "OpDef '{name}': row_sort is single-output (a permutation is not a ScalarExpr \
         body) — extra_out_bodies must be empty"
    );

    // Operand count + rank. G2: Values/Indices write one output ([in0, out]);
    // Both writes two ([in0, out_val, out_idx]).
    let n = op.n_inputs as usize;
    assert!(
        n == 1,
        "OpDef '{name}': row_sort streams exactly one row operand (n_inputs=1), got {n}"
    );
    let n_outputs = if matches!(out, SortOut::Both) { 2 } else { 1 };
    assert!(
        key.n_operands as usize == n + n_outputs,
        "row_sort expects n_inputs+{n_outputs} operands (input then {n_outputs} \
         output buffer(s)); got {}",
        key.n_operands
    );
    let rank = key.rank as usize;
    assert!(rank >= 1, "row_sort needs a sorted axis (rank >= 1)");
    let last = (rank - 1) as u8;

    // Input 0 layout: the row-streamed sorted tensor. The emitters read
    // `in0[base+j]` stride-free, so it must be Contig and NOT flipped (a
    // flipped/strided operand reads mirrored/OOB).
    let in0 = key.operands[0];
    assert!(
        rr_role(in0, last) == RrRole::RowStreamed,
        "OpDef '{name}': row_sort Input0 must be the row-streamed sorted tensor, not \
         a column-broadcast or per-row-scalar operand"
    );
    assert!(
        in0.contig == Contiguity::Contig,
        "OpDef '{name}': row_sort input must be contiguous (base = row*k assumes a \
         dense sorted axis)"
    );
    assert!(
        !in0.flipped,
        "OpDef '{name}': row_sort input must not be reversed along an axis (idx = \
         base+j reads forward-dense; a flipped view reads mirrored/OOB — use the \
         Desc order, not a flipped operand)"
    );

    // Output layout: full-width forward-dense contiguous. G3: output 0 (out for
    // Values/Indices, out_val for Both) always; and for Both, operand 2 (out_idx)
    // too — it is written forward-dense out_idx[base + r/p].
    // Not `out.bcast`: the key's mask is over the iteration frame, so a size-1 own axis
    // (a global pool, a top-1) beside a wider input reads as broadcast. `contig` is
    // `Broadcast` for a genuine stride-0 axis of extent > 1, so it alone rejects those.
    let out0 = key.operands[n];
    assert!(
        out0.contig == Contiguity::Contig && !out0.flipped,
        "OpDef '{name}': row_sort output must be forward-dense contiguous (empty \
         bcast, not flipped); the width may be < the input for a TopK cap"
    );
    if matches!(out, SortOut::Both) {
        let out_idx = key.operands[n + 1];
        assert!(
            out_idx.contig == Contiguity::Contig && !out_idx.flipped,
            "OpDef '{name}': row_sort Both out_idx (operand 2) must be forward-dense \
             contiguous (empty bcast, not flipped); width may be < input for a TopK cap"
        );
    }
}

/// Vector width in elements for a [`VecWidth`] bucket.
pub(super) fn vec_width_elems(v: VecWidth) -> u32 {
    match v {
        VecWidth::V8 => 8,
        VecWidth::V4 => 4,
        VecWidth::V2 => 2,
        VecWidth::Scalar => 1,
    }
}
