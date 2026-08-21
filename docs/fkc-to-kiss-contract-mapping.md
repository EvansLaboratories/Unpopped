# FKC → KISS-Contract: the field-by-field mapping

**The discharge KISS #236 asked for.** The architect ruled that a KISS-Contract
**replaces** a consumer's contract format as the travelling document rather than
being derived from it, and then named the escape hatch:

> I did NOT settle whether the seven-section core is expressive enough to carry
> everything FKC carries. If it isn't, this ruling becomes a defect report
> against the seven sections rather than an obligation on you. **An unmappable
> field is a finding about KISS, not about you.**

So this is a mapping attempt with every unmappable field named. It is not an
argument against the ruling and does not ask for one — it is the list.

**Measured against:** KISS `origin/main` `5437be5` (`spec/contract.md`), and the
FKC this workspace actually emits, taken from
`crates/unpopped/examples/sample_bundle.md` rather than from `contract.rs`'s
`push_str` calls, so the left-hand column is a document that exists rather than
code that produces one.

## What is not in scope

**The op vocabulary.** `op_kind: AddElementwise` is Fuel's name set, and the
ruling is explicit that a consumer keeps its names whether or not it keeps its
own document. Re-basing onto KISS-Ops for `op_identity` is a translation
obligation on this crate, not a gap in KISS, and `recipe::semantics_dag` already
does it.

**The internal representation.** Deriving an FKC-shaped value in memory stays
allowed. What the ruling forbids is emitting one *as the travelling document*.

## The mapping

| FKC field | KISS-Contract home | Note |
|---|---|---|
| `kernel` | Identity `kernel_name` | |
| `entry_point` | Identity `entry_point` | |
| `accept.structure_key` | Identity `accept_predicate` | Same token, carried verbatim. |
| `kernel_revision_hash` | Identity `revision_hash` | |
| `fkc_version` | Identity `contract_version` | |
| `op_kind` | Identity `op_identity` | After re-basing onto KISS-Ops (above). |
| `semantics` | Semantics `op_dag` + `semantics_kind` | `machine-checkable-IR` for a generated kernel. |
| `blurb` | Semantics `human_annotation` | Explicitly free-text and outside the exact-byte scope (§6.4-0001). |
| `precision.notes` | Semantics `human_annotation` | Same home; see the note below on it being **one** field. |
| `accept.inputs[].name` | Interface `positional_signature` | |
| `accept.inputs[].dtypes` | Capabilities `supported_dtype_set` | |
| `accept.inputs[].layout{…}` | Identity `accept_predicate` | The structure_key's per-operand sub-key already encodes contiguity, broadcast mask, vector width and flip. FKC states them as prose; KISS states them as bytes. Same content, and the bytes are the stricter form. |
| `caps.in_place` | Interface `in_place` | |
| `caps.alignment_bytes` | Interface `alignment_bytes` | |
| `caps.count_unit` | Interface `count_unit` | |
| `caps.awkward_layout_strategy` | Capabilities `awkward_layout_strategy` | |
| `dtypes` | Capabilities `supported_dtype_set` | |
| `cost.class` | Capabilities `cost` (class half) | |
| `cost.flops`, `cost.bytes_moved` | Capabilities `cost` (expression half) | §6.7-0006 requires expressions over the launch-scalar symbols — `"1 * n"` and `"12 * n"` are already that shape. |
| `cost.provenance` | Guarantees `cost_provenance` | |
| `precision.bit_stable_on_same_hardware` | Guarantees `bit_stability` | |
| `precision.max_ulp` | Guarantees `math_precision` / `per_backend_ulp_tiers` | |
| `precision.audited` | Guarantees `audited_status` | |
| `determinism` | Guarantees `determinism_class` | |
| `return.outputs[].dtype_rule` | Semantics `op_dag` | Derivable rather than stated: `passthrough(in0)` is what the DAG already says. A derivation is not a loss. |
| `return.outputs[].shape_rule` | Semantics `op_dag` | Same; §6.4 carries an output-shape consistency obligation. |
| `provider.name` | Provenance `bundle_envelope.provider_id` | |
| `provider.kernel_source` | Provenance `kernel_source` | |
| `provider.revision_base` | Provenance `revision_base` | |
| `provider.backend` | Provenance `kernel_source` (partly) + Identity `target_capability` | Split across two homes; see the note below. |
| `seam_profiles` | Provenance `negotiation_metadata` | Explicitly opaque (§6.9-0008), which is the right treatment for a consumer-specific profile list. |

## The unmappable fields

Three, and I have graded my confidence in each, because a gap I am wrong about
costs the architect more than one I miss.

### 1. `return.outputs[].layout_guarantee` — CONFIDENT

FKC says `layout_guarantee: contiguous`: a statement about the layout the kernel
**writes**. Searching `spec/contract.md` for an output-layout guarantee returns
nothing, in any of the seven sections.

**Why the structure_key does not cover it.** The key's operand array includes the
output operand with its contiguity — but that is the **accept predicate**: what
the caller must supply. A guarantee about what the kernel *writes into* that
buffer is a different claim with a different direction.

**Why it may nonetheless be vacuous at this schema**, which I would rather state
than have found: for every cell this workspace emits, the kernel writes exactly
the layout it was handed, so the accept predicate and the write guarantee
coincide. **They come apart the first time a kernel accepts a strided output and
writes it densely, or normalizes a layout.** So this is a real hole in the format
that no current kernel falls into — which is the kind that gets discovered by the
first party who needs it.

The Guarantees section is the natural home; it currently carries only numeric and
determinism guarantees.

### 2. `provider.link_registry` — CONFIDENT

The symbol name of the generated link registry a runtime `include!`s
(`baracuda_link_registry`). `bundle_envelope` is
`{provider_id, revision_base, derivation_lineage, contained}` (§6.9-0009) — there
is no slot for a **linking artifact's identifier**.

**Why it is not just an implementation detail.** The registry is how a consumer
resolves `entry_point` to a callable at load. A contract that names the entry
point but not where the roster of entry points lives moves that knowledge
out-of-band, which is the thing a self-delimiting document is supposed to
prevent. §1's *"the single self-delimiting document that travels with every
provided kernel"* is the clause this pressures.

It may well belong in KISS-Synth or KISS-Announce rather than KISS-Contract — I
am naming the gap, not its home.

### 3. `return.outputs[].aliasing` — LOW CONFIDENCE, LISTED FOR COMPLETENESS

`aliasing: none` is close to `in_place: false` and is probably the same claim for
every single-output kernel. It comes apart only for **multi-output** kernels,
where output-to-output aliasing is expressible and `in_place` says nothing about
it. This workspace emits multi-output kernels, so the case is reachable — but I
have not constructed one where the two claims differ, and I would not file this
alone.

## One thing that is not a gap but is worth the architect seeing

**`blurb` and `precision.notes` both map to `human_annotation`, and there is one
of it.** FKC carries two free-text fields with different subjects — what the
kernel does, and why its precision claim reads as it does. Collapsing them into
one field loses which is which, and the surviving text has to carry the
distinction in prose.

That is the `DeclinedOp::Access` shape — distinct things under one label,
separable only by text — and this workspace spent a release removing an instance
of it. It is a **very** minor instance: `human_annotation` is explicitly
non-normative and never byte-compared, so nothing conformance-bearing rests on
the distinction. Noting it because the architect asked for the list, not because
it should block anything.

## What this costs, stated plainly

Every field except the three above has a home, and two of the three are things no
kernel currently emitted actually depends on. **The seven sections are expressive
enough to carry FKC**, and the ruling stands on this evidence rather than being
weakened by it.

The port is therefore work rather than a blocker, and the two recorded withholds
(`no importable Fuel OpKind`, the blanket scatter-family return) are real rather
than exempt, exactly as the ruling says — §2.7 enumerates them as the withholds
the neutral hub abolishes.
