// SPDX-License-Identifier: MIT OR Apache-2.0
//! Section of the plan gate; split from `plan.rs` (ROADMAP U30). Paths unchanged.

use super::*;

/// Short tag for an [`Access`] variant, for the multi-output rejection message.
pub(super) fn access_tag(a: &Access) -> &'static str {
    match a {
        Access::Elementwise => "Elementwise",
        Access::Reduction { .. } => "Reduction",
        Access::RowReduce { .. } => "RowReduce",
        Access::Contraction { .. } => "Contraction",
        Access::Scan { .. } => "Scan",
        Access::Window { .. } => "Window",
        Access::RowSort { .. } => "RowSort",
        Access::Im2Col { .. } => "Im2Col",
    }
}

/// `true` if `v` is an **address-affecting** view — one that changes which
/// producer element the strided emitter reads at each iteration coordinate, and
/// therefore forces the [`Schedule::Strided`] schedule and cannot be
/// vectorized/packed. In v1 that is [`View::Permute`] (a transposed read, offset
/// remap `c{d}·stride[perm[d]]`) and [`View::Broadcast`] (stride-0 axes — already
/// non-contiguous). [`View::Identity`] and a same-rank [`View::Reshape`] (an
/// identity linear-index map) are NOT addressing: they read at the iteration
/// coordinate exactly like a view-free operand, so an all-identity op stays
/// byte-identical to a view-free one.
pub fn view_is_addressing(v: &View) -> bool {
    matches!(
        v,
        View::Permute { .. } | View::Broadcast { .. } | View::Layout { .. }
    )
}

/// `true` if any of `op`'s per-input views is address-affecting
/// ([`view_is_addressing`]). `false` for a view-free op (empty `views`) and for
/// an all-`Identity`/same-rank-`Reshape` op — the byte-identical cases.
pub(crate) fn op_has_addressing_view(op: &OpDef) -> bool {
    op.views.iter().any(view_is_addressing)
}

/// Item-01 **layout-view** admissibility gate — runs at the TOP of
/// [`build_plan`] (the house pattern), with an independent emitter backstop in
/// the backend. A view-free op (empty `views`, every
/// pre-item-01 constructor) returns immediately, and an all-[`View::Identity`] op
/// returns after the length check — so the established path is unchanged and
/// emission stays byte-identical. For an op carrying a real (non-`Identity`)
/// view, the v1 rules, all honest AOT panics (an author/generator error — views
/// never cross the JIT trust boundary, so a panic is the backstop, not a silent
/// wrong-bind):
///
/// 1. **Shape**: `views.len() == n_inputs` (index `i` ↔ `Input(i)`).
/// 2. **Access**: a non-`Identity` view is [`Access::Elementwise`]-only in v1.
///    A reduction/row-reduce/contraction op has its OWN axis machinery (reduced
///    axes, K-contraction, feature broadcast) that a per-input read-through would
///    double-count; those reject (pass through only a trivially-`Identity` view).
/// 3. **Single-output**: a viewed input on a multi-output op is a deferred
///    composition in v1 (the multi-store DAG × the stride remap is unproven) —
///    reject. A viewed single-output op is the whole item-01 surface.
/// 4. **Validity** (every view): [`View::is_valid`] against `key.rank` (a
///    `Permute` must be a true permutation of `0..rank`).
/// 5. **`Permute` ⊥ `Broadcast`**: a permuted input's operand key must have an
///    EMPTY broadcast mask — v1 keeps the transpose remap and the stride-0
///    broadcast orthogonal (a permuted-and-broadcast operand is deferred). The
///    offset remap `c{d}·stride[perm[d]]` then folds cleanly with no per-axis
///    broadcast-skip interaction.
/// 6. **`Broadcast` agreement**: the view's declared `bcast` axes must be a
///    SUBSET of the operand key's broadcast mask. Emission is key-driven (the
///    strided emitter reads `OperandKey::bcast`), so the view is the *named*
///    form of what the key already encodes (per the `View::Broadcast` doc) — a
///    view claiming a broadcast the key doesn't have would be a silent lie the
///    emitter ignores. Validate-only in v1: it changes no address math.
/// 7. **`Reshape` scope**: v1 accepts only a `producer_rank == key.rank`
///    (same-rank, identity linear-index map) reshape, carried for
///    recognition/keying and emitted as identity address math. A rank-change
///    reshape is genuine rank-change emit (items 03/10) and rejects here.
pub(super) fn assert_valid_views(op: &OpDef, key: &StructureKey) {
    if op.views.is_empty() {
        return; // view-free — every pre-item-01 op, unchanged.
    }
    let name = &op.name;
    assert_eq!(
        op.views.len(),
        op.n_inputs as usize,
        "OpDef '{name}': views.len() ({}) must equal n_inputs ({})",
        op.views.len(),
        op.n_inputs
    );
    if op.views.iter().all(View::is_identity) {
        return; // all-Identity == view-free: byte-identical emission, no gate.
    }
    // From here at least one real (address- or recognition-bearing) view.
    assert!(
        matches!(op.access, Access::Elementwise | Access::Contraction { .. }),
        "OpDef '{name}': a non-Identity View is admitted only on Access::Elementwise \
         or Access::Contraction in v1 — a {}-class op has its own axis machinery \
         (reduced/feature axes) that a per-input read-through would double-count; a \
         view on it must be Identity",
        access_tag(&op.access)
    );
    assert!(
        op.n_outputs() == 1,
        "OpDef '{name}': a viewed input on a multi-output op ({} outputs) is a \
         deferred composition in v1 (the multi-store DAG × the per-operand stride \
         remap is unproven) — miss honestly",
        op.n_outputs()
    );
    let rank = key.rank;
    for (i, v) in op.views.iter().enumerate() {
        assert!(
            v.is_valid(rank),
            "OpDef '{name}': input {i} view {v:?} is invalid for iteration rank \
             {rank} (a Permute must be a true permutation of 0..rank)"
        );
        // Input operands are the first `n_inputs` key operands (inputs then
        // outputs). The array is fixed [OperandKey; MAX_OPERANDS], so indexing a
        // valid input slot never panics; a smaller n_operands reads a default
        // (empty-broadcast) key, which the checks below treat conservatively.
        let o = key.operands[i];
        match v {
            View::Identity => {}
            View::Permute { .. } => {
                assert!(
                    o.bcast.is_empty(),
                    "OpDef '{name}': input {i} has a Permute view AND a broadcast \
                     mask ({:#04x}) — v1 keeps the transpose remap and stride-0 \
                     broadcast orthogonal (a permuted-and-broadcast operand is \
                     deferred)",
                    o.bcast.0
                );
            }
            View::Broadcast { bcast } => {
                assert!(
                    bcast.0 & !o.bcast.0 == 0,
                    "OpDef '{name}': input {i} Broadcast view declares axes \
                     ({:#04x}) the operand key does not broadcast ({:#04x}) — the \
                     key drives address math, so the named view must agree (a \
                     view-only broadcast the emitter ignores would be a silent lie)",
                    bcast.0,
                    o.bcast.0
                );
            }
            View::Layout { shape, .. } => {
                assert!(
                    matches!(op.access, Access::Elementwise),
                    "OpDef '{name}': input {i} Layout view is admitted only on Access::Elementwise in v1 (its extents are the iteration extents, which a {}-class op does not have)",
                    access_tag(&op.access)
                );
                assert!(
                    o.bcast.is_empty(),
                    "OpDef '{name}': input {i} has a Layout view AND a broadcast mask ({:#04x}): the view's strides are the truth for this input, so the operand must carry no key broadcast (write a broadcast as a stride 0 in the view instead)",
                    o.bcast.0
                );
                assert!(
                    shape.len() == usize::from(rank),
                    "OpDef '{name}': input {i} Layout view has {} extents for iteration rank {rank}",
                    shape.len()
                );
            }
            View::Reshape { producer_rank } => {
                assert!(
                    *producer_rank == rank,
                    "OpDef '{name}': input {i} Reshape view producer_rank \
                     ({producer_rank}) != iteration rank ({rank}) — a rank-change \
                     reshape is genuine rank-change emit (items 03/10), out of \
                     item-01 scope; v1 accepts only a same-rank (identity \
                     linear-map) reshape"
                );
            }
        }
    }
}

/// Whether `dtype` may address a gather or scatter: any integer index dtype (`I8`, `U8`, `I16`,
/// `U16`, `I32`, `U32`, `I64`, `U64`); a float address is meaningless (U58, union row Mojo 49).
#[must_use]
pub fn is_index_dtype(dtype: ElementKind) -> bool {
    matches!(
        dtype,
        ElementKind::I8
            | ElementKind::U8
            | ElementKind::I16
            | ElementKind::U16
            | ElementKind::I32
            | ElementKind::U32
            | ElementKind::I64
            | ElementKind::U64
    )
}

/// The single **gathered** input of `read_index` (increment 4), or `None` for an
/// index-free / all-[`ReadIndex::Direct`] op. Returns the gathered input's slot
/// plus its [`ReadIndex::Indexed`] fields `(gathered_input, index_operand, axis,
/// oob, index_dtype)`. v1 admits **at most one** gathered input (the plan gate
/// `assert_valid_gather` rejects more), so the first match is the only one; the
/// emitter and its backstop read the gather off this one accessor to stay in
/// lockstep.
pub fn gather_of(
    read_index: &[ReadIndex],
) -> Option<(usize, u8, u8, crate::ir::OobPolicy, ElementKind)> {
    read_index.iter().enumerate().find_map(|(i, r)| match r {
        ReadIndex::Indexed {
            index_operand,
            axis,
            oob,
            index_dtype,
        } => Some((i, *index_operand, *axis, *oob, *index_dtype)),
        ReadIndex::Direct => None,
    })
}

/// `true` if `op` has a [`ReadIndex::Indexed`] input (a gather; increment 4).
/// `false` for an index-free op (empty `read_index`) and an all-`Direct` op — the
/// byte-identical cases. Forces the [`Schedule::Strided`] schedule and is the
/// `contract`/`pattern` honest-miss trigger.
pub(crate) fn op_has_gather(op: &OpDef) -> bool {
    op.read_index.iter().any(|r| !r.is_direct())
}

/// `true` if `op` carries any **runtime base offset** (a [`BaseOffset::Runtime`]
/// input OR a `Runtime` output; BASE_OFFSET SLICE). `false` for an offset-free op
/// (empty `base_offsets` + `Zero` output) and for a non-empty all-`Zero` op — the
/// byte-identical cases (presence is defined as `any(Runtime)`, so a normalized
/// all-`Zero` vec is identical to empty). THE single presence oracle: the plan
/// schedule gate, the `assert_valid_offsets` validator, the emitter's name suffix/
/// launch args/pointer bumps, and the `pattern` honest-miss all read it, so they
/// can never disagree about whether an op is offsetted.
pub(crate) fn op_has_offset(op: &OpDef) -> bool {
    op.base_offsets.iter().any(|b| !b.is_zero()) || !op.out_base_offset.is_zero()
}

/// Increment-4 **GATHER** admissibility gate — runs at the TOP of [`build_plan`]
/// (the house pattern), with an independent emitter backstop in
/// the backend. An index-free op (empty
/// `read_index`, every pre-increment-4 constructor) returns immediately, and an
/// all-[`ReadIndex::Direct`] op returns after the length check — so the
/// established path is unchanged and emission stays byte-identical. For an op
/// carrying a real gather the v1 rules, all honest AOT panics (an author/
/// generator error — the gather role never crosses the JIT trust boundary):
///
/// 1. **Shape**: `read_index.len() == n_inputs` (index `i` ↔ `Input(i)`).
/// 2. **Access**: [`Access::Elementwise`]-only in v1 — a reduction/row-reduce/
///    contraction op has its own axis machinery a per-input indexed read would
///    double-count.
/// 3. **Single-output** in v1 (the multi-store DAG × the address substitution is
///    unproven).
/// 4. **One gathered input** in v1 — the emitter handles exactly one substituted
///    axis; a second data-dependent address (and combining OOB predicates across
///    two gathers) is deferred. Every bespoke gather/index_select/embedding
///    gathers exactly one input, so this covers the charter surface.
/// 5. Per gathered input `g` with `Indexed { index_operand, axis, oob,
///    index_dtype }`:
///    - `index_operand < n_inputs` and `index_operand != g` (an input can't index
///      itself).
///    - `index_dtype ∈ {I32, I64}` — an **integer** index (a float index address
///      is meaningless; the emitted load type must be an int).
///    - `axis < key.rank`.
///    - the index operand must NOT itself be gathered (`read_index[index_operand]`
///      is `Direct`) — a data-dependent index-of-an-index is out of v1 scope.
///    - the gathered input must NOT also carry an address-affecting [`View`]
///      (Permute/Broadcast) — gather ⊥ view in v1 (a gathered-and-permuted
///      operand's composed address math is unproven; reject rather than
///      mis-emit).
///    - the gathered axis of the DATA operand must have a real stride (its key
///      broadcast mask must NOT set `axis`) — the substituted `idx·stride[axis]`
///      term needs a live stride; a broadcast gathered axis is a degenerate
///      no-op.
pub(super) fn assert_valid_gather(op: &OpDef, key: &StructureKey) {
    if op.read_index.is_empty() {
        return; // index-free — every pre-increment-4 op, unchanged.
    }
    let name = &op.name;
    assert_eq!(
        op.read_index.len(),
        op.n_inputs as usize,
        "OpDef '{name}': read_index.len() ({}) must equal n_inputs ({})",
        op.read_index.len(),
        op.n_inputs
    );
    if op.read_index.iter().all(ReadIndex::is_direct) {
        return; // all-Direct == index-free: byte-identical emission, no gate.
    }
    assert!(
        matches!(op.access, Access::Elementwise),
        "OpDef '{name}': a gather (Indexed read) is Access::Elementwise-only in v1 \
         — a {}-class op has its own axis machinery that a per-input indexed read \
         would double-count",
        access_tag(&op.access)
    );
    assert!(
        op.n_outputs() == 1,
        "OpDef '{name}': a gathered input on a multi-output op ({} outputs) is a \
         deferred composition in v1 — miss honestly",
        op.n_outputs()
    );
    let n_gathered = op.read_index.iter().filter(|r| !r.is_direct()).count();
    assert!(
        n_gathered == 1,
        "OpDef '{name}': v1 admits exactly one gathered input, got {n_gathered} — a \
         second data-dependent address (and combined OOB predicates) is deferred"
    );
    let (g, index_operand, axis, _oob, index_dtype) =
        gather_of(&op.read_index).expect("one gathered input checked above");
    assert!(
        (index_operand as usize) < op.n_inputs as usize,
        "OpDef '{name}': gather index_operand ({index_operand}) >= n_inputs ({})",
        op.n_inputs
    );
    assert!(
        index_operand as usize != g,
        "OpDef '{name}': gather input {g} names ITSELF as its index_operand — an \
         input cannot index itself"
    );
    assert!(
        is_index_dtype(index_dtype),
        "OpDef '{name}': gather index_dtype must be an integer index dtype \
         (I8/U8/I16/U16/I32/U32/I64/U64), got {index_dtype:?} — a float index address is meaningless"
    );
    assert!(
        (axis as usize) < key.rank as usize,
        "OpDef '{name}': gather axis ({axis}) >= iteration rank ({})",
        key.rank
    );
    assert!(
        op.read_index[index_operand as usize].is_direct(),
        "OpDef '{name}': the gather index operand ({index_operand}) must not ITSELF \
         be gathered — an index-of-an-index is out of v1 scope"
    );
    // gather ⊥ view (v1): the gathered input must not carry an address-affecting
    // view. An index-free/all-identity `views` is fine (the common case).
    if !op.views.is_empty() {
        assert!(
            !view_is_addressing(&op.views[g]),
            "OpDef '{name}': gathered input {g} also carries an address-affecting \
             View ({:?}) — gather ⊥ view in v1 (the composed address math is \
             unproven)",
            op.views[g]
        );
    }
    // The gathered axis of the DATA operand needs a live stride (the substituted
    // `idx·stride[axis]` term); a broadcast gathered axis is a degenerate no-op.
    // Input operands are the first `n_inputs` key slots.
    assert!(
        !key.operands[g].bcast.is_set(axis),
        "OpDef '{name}': gathered input {g} broadcasts the gathered axis ({axis}) \
         — the substituted idx·stride[axis] term needs a live stride"
    );
}

/// `true` if `op` scatters (a [`WriteIndex::ScatterIndexed`] output; increment 5).
/// `false` for a [`WriteIndex::Direct`] op — the byte-identical case. Forces the
/// [`Schedule::Strided`] schedule and is the `contract`/`pattern` honest-miss
/// trigger.
pub(crate) fn op_has_scatter(op: &OpDef) -> bool {
    !op.write_index.is_direct()
}

/// The scattered output's [`WriteIndex::ScatterIndexed`] fields
/// `(index_operand, axis, combine, oob, index_dtype)`, or `None` for a
/// [`WriteIndex::Direct`] op. The emitter and its backstop read the scatter off
/// this one accessor to stay in lockstep (mirror of [`gather_of`]).
pub fn scatter_of(
    write_index: &WriteIndex,
) -> Option<(u8, u8, WriteCombine, crate::ir::OobPolicy, ElementKind)> {
    write_index.scatter()
}

/// Whether the [`WriteCombine`] is legal for `out_dtype` at the emitter's v1
/// atomic-primitive coverage (used by the gate and the emitter backstop):
///
/// - `Assign` — legal for every dtype (a plain store).
/// - `AtomicAdd` — legal for `f32`/`f64` (native FP atomicAdd) and `i32`/`i64`
///   (native / `unsigned long long` reinterpret). f16/bf16 need the bespoke CAS
///   helper (`baracuda_atomic.cuh`), which the header-light generated source
///   can't include — deferred. u8/s8 need a sub-word CAS — deferred.
/// - `AtomicMax`/`AtomicMin` — **integer only** in v1 (`i32`/`i64`); float has no
///   native `atomicMax`/`atomicMin` (a CAS emulation is a follow-up).
pub fn combine_legal_for_dtype(combine: WriteCombine, out_dtype: ElementKind) -> bool {
    match combine {
        WriteCombine::Assign => true,
        WriteCombine::AtomicAdd => matches!(
            out_dtype,
            ElementKind::F32
                | ElementKind::F32Strict
                | ElementKind::F64
                | ElementKind::I32
                | ElementKind::I64
        ),
        WriteCombine::AtomicMax | WriteCombine::AtomicMin => {
            matches!(out_dtype, ElementKind::I32 | ElementKind::I64)
        }
    }
}

/// Increment-5 **SCATTER** admissibility gate — runs at the TOP of [`build_plan`]
/// (the house pattern), with an independent emitter backstop in
/// the backend. A write-Direct op (every
/// pre-increment-5 constructor) returns immediately, so the established path is
/// unchanged and emission stays byte-identical. For an op carrying a real scatter
/// the v1 rules, all honest AOT panics (an author/generator error — the scatter
/// role never crosses the JIT trust boundary):
///
/// 1. **Access**: [`Access::Elementwise`]-only in v1 (a reduction/row-reduce/
///    contraction has its own axis machinery a scatter would double-count).
/// 2. **Single-output** in v1.
/// 3. **Not also a gather** — a fused gather+scatter (address-in AND address-out)
///    is a deferred composition; each ships separately.
/// 4. For the scatter role `ScatterIndexed { index_operand, axis, combine, oob,
///    index_dtype }`:
///    - `index_operand < n_inputs`.
///    - `index_dtype ∈ {I32, I64}` — an integer index (the emitted load type must
///      be an int; a float destination address is meaningless).
///    - `axis < key.rank`.
///    - the index operand must NOT itself be gathered (no index-of-an-index).
///    - `oob == Skip` — the only bespoke-matched scatter policy in v1
///      (Clamp/ZeroFill are gather-side; a scattered ZeroFill would need a
///      separate zeroing pass over the untouched destination).
///    - the `combine` op must be legal for the OUTPUT dtype
///      ([`combine_legal_for_dtype`] — AtomicMax/Min integer-only, atomicAdd
///      f32/f64/i32/i64).
///    - the scattered axis of the DESTINATION (the OUTPUT key slot) must have a
///      real stride (its broadcast mask must NOT set `axis`) — the substituted
///      `idx·stride_out[axis]` term needs a live stride.
///    - the output must NOT carry an address-affecting [`View`] (scatter ⊥ view
///      in v1) — views are an input-read property; a scattered output view is
///      unproven.
pub(super) fn assert_valid_scatter(op: &OpDef, key: &StructureKey) {
    let Some((index_operand, axis, combine, oob, index_dtype)) = scatter_of(&op.write_index) else {
        return; // write-Direct — every pre-increment-5 op, unchanged.
    };
    let name = &op.name;
    assert!(
        matches!(op.access, Access::Elementwise),
        "OpDef '{name}': a scatter (ScatterIndexed write) is Access::Elementwise-only \
         in v1 — a {}-class op has its own axis machinery a scattered write would \
         double-count",
        access_tag(&op.access)
    );
    assert!(
        op.n_outputs() == 1,
        "OpDef '{name}': a scattered multi-output op ({} outputs) is a deferred \
         composition in v1 — miss honestly",
        op.n_outputs()
    );
    assert!(
        !op_has_gather(op),
        "OpDef '{name}': a fused gather+scatter (data-dependent read AND write) is a \
         deferred composition in v1 — each ships separately"
    );
    assert!(
        (index_operand as usize) < op.n_inputs as usize,
        "OpDef '{name}': scatter index_operand ({index_operand}) >= n_inputs ({})",
        op.n_inputs
    );
    assert!(
        is_index_dtype(index_dtype),
        "OpDef '{name}': scatter index_dtype must be an integer index dtype \
         (I8/U8/I16/U16/I32/U32/I64/U64), got {index_dtype:?} — a float destination address is meaningless"
    );
    assert!(
        (axis as usize) < key.rank as usize,
        "OpDef '{name}': scatter axis ({axis}) >= iteration rank ({})",
        key.rank
    );
    assert!(
        matches!(oob, crate::ir::OobPolicy::Skip),
        "OpDef '{name}': scatter OOB policy must be Skip in v1 (bespoke \
         scatter/scatter_add/index_add/bincount all skip an OOB target), got {oob:?}"
    );
    let out_dtype = op.out_dtype.unwrap_or(key.dtype);
    assert!(
        combine_legal_for_dtype(combine, out_dtype),
        "OpDef '{name}': scatter combine {combine:?} is not legal for output dtype \
         {out_dtype:?} in v1 (AtomicMax/Min integer-only; atomicAdd f32/f64/i32/i64; \
         f16/bf16/u8 atomics need the bespoke CAS helper the header-light source \
         can't include)"
    );
    // The scattered axis of the DESTINATION (last key slot) needs a live stride.
    let out_slot = (key.n_operands as usize).saturating_sub(1);
    assert!(
        !key.operands[out_slot].bcast.is_set(axis),
        "OpDef '{name}': scattered output broadcasts the scattered axis ({axis}) — \
         the substituted idx·stride_out[axis] term needs a live stride"
    );
    // scatter ⊥ view (v1): `views` is a per-INPUT slice (an output view would ride
    // a separate future field, not expressible today), so there is no output-view
    // to reject here — the scatter output offset is always an identity remap. The
    // INPUT operands may still carry views (the value operand could be a transposed
    // read); that composes cleanly (input views are handled by `offset_expr`),
    // so it is deliberately NOT rejected.

    // Body must be a bare identity value read or a Const (review #5 CRITICAL): the
    // deterministic gather-sum base (`emit_scatter_gathersum`) sums `in{val_op}`
    // DIRECTLY rather than lowering `op.body`, so a composed body (e.g.
    // `Input(0)*Param(0)`) would silently compute `Sum(updates)` instead of
    // `Sum(f(updates))` — diverging from both the op and its own atomic variant
    // (which DOES lower the body). Bespoke scatter/scatter_add copy the value
    // verbatim and bincount stores a constant, so v1 pins the body accordingly; a
    // fused scatter body is a deferred v1 composition. `val_op` matches the
    // emitter's derivation exactly.
    let val_op = (0..op.n_inputs as usize)
        .find(|&k| k != index_operand as usize)
        .unwrap_or(0);
    assert!(
        matches!(&op.body, ScalarExpr::Input(v) if *v as usize == val_op)
            || matches!(&op.body, ScalarExpr::Const(_)),
        "OpDef '{name}': a v1 scatter body must be the identity value read Input({val_op}) \
         or a constant (bincount), got a composed body — a fused scatter transform is a \
         deferred v1 composition (the deterministic gather-sum base sums the value operand \
         directly and would silently drop it)"
    );
}

/// BASE_OFFSET SLICE admissibility gate — runs at the TOP of [`build_plan`] (the
/// house pattern), with an independent emitter backstop in
/// the backend. An offset-free op (empty
/// `base_offsets` + a `Zero` output) returns immediately, and a non-empty all-`Zero`
/// op returns after the length check — so the established path is unchanged and
/// emission stays byte-identical. For an op carrying a real `Runtime` offset the v1
/// rules, all honest AOT panics (an author/generator error — the offset value is a
/// per-launch scalar that never crosses the JIT trust boundary, so a panic is the
/// backstop, not a silent wrong-emit):
///
/// 1. **Shape (G1)**: a non-empty `base_offsets` ⇒ `len == n_inputs` (index `i` ↔
///    `Input(i)`; the same rule as `views`/`read_index`). Checked even for an
///    all-`Zero` non-empty vec — a malformed length is an author bug either way.
/// 2. **Access (G2)**: any `Runtime` ⇒ [`Access::Elementwise`] only. A
///    reduction/row-reduce/contraction/scan/window/sort op has role-aware
///    addressing (RowStreamed/ColBroadcast/RowScalar) that needs per-role offset
///    semantics — a single entry-bump is unsound there, so it is de-scoped.
/// 3. **Single-output (G3)**: any `Runtime` ⇒ `extra_out_bodies` empty. A runtime
///    offset on a multi-output op is a deferred composition (the multi-store DAG ×
///    the pointer bump is unproven) — rope uses Option B (one single-output launch
///    per lane parity).
///
/// The schedule-forcing rule (**G4**: any `Runtime` ⇒ [`Schedule::Strided`]) lives
/// in the `Access::Elementwise` arm of [`build_plan`] — a runtime offset shifts the
/// effective base pointer, so the keyed `align_bytes → VecWidth` fact the
/// `float4`/`ld.128` path relies on becomes a lie even for a width-multiple value.
/// **OOB is a caller precondition** (the k/n_out trust model, like gather/scatter's
/// `gext`/`sext` and RowSort's `k<=1024`): `off + <maximal declared-extent address>`
/// must land in-bounds; the emitted code forms no address before the bump and clamps
/// no offset — it is a `launch_note` + on-device-validated contract, not a checked
/// bound.
pub(super) fn assert_valid_offsets(op: &OpDef, _key: &StructureKey) {
    if op.base_offsets.is_empty() && op.out_base_offset.is_zero() {
        return; // offset-free — every pre-increment op, unchanged.
    }
    let name = &op.name;
    // G1 arity: a non-empty vec must be length-matched (an all-Zero non-empty vec
    // is still a length claim the author must get right — checked before the
    // all-Zero early-out below).
    if !op.base_offsets.is_empty() {
        assert_eq!(
            op.base_offsets.len(),
            op.n_inputs as usize,
            "OpDef '{name}': base_offsets.len() ({}) must equal n_inputs ({})",
            op.base_offsets.len(),
            op.n_inputs
        );
    }
    if !op_has_offset(op) {
        return; // all-Zero == offset-free: byte-identical emission, no gate.
    }
    // G2 access: a runtime offset is Elementwise-only in v1.
    assert!(
        matches!(op.access, Access::Elementwise),
        "OpDef '{name}': a runtime BaseOffset is Access::Elementwise-only in v1 — a \
         {}-class op has role-aware addressing (RowStreamed/ColBroadcast/RowScalar) \
         that needs per-role offset semantics (a later increment)",
        access_tag(&op.access)
    );
    // G3 single-output: a runtime offset on a multi-output op is a deferred
    // composition (the multi-store DAG × the pointer bump is unproven).
    assert!(
        op.n_outputs() == 1,
        "OpDef '{name}': a runtime BaseOffset on a multi-output op ({} outputs) is a \
         deferred composition in v1 — rope uses Option B (one single-output launch \
         per lane parity)",
        op.n_outputs()
    );
}
