# Idiom lifting: parsers lift optimizations to intent, emitters re-lower

**Design doc only. No code changes.** It answers three PM tasks of 2026-09-30:
- the `[TASK]` *"restart-then-idiom-lifting-design scope=design-doc-only"*
  (§§0-6);
- the addendum adding a technique registry (§7);
- the addendum adding the hardware-feature mapping, autotuning and the same-GPU
  A/B (§§8-10).

Every citation below
was read at **`unpopped@b61a8be`**, **`baracuda@ab2e0bf`** and **`fuel@5bd7933`**
(each an `origin/main` fetched on 2026-09-29). Line numbers refer to those
revisions.

## 0. The question, and the answer this doc builds on

CireSnave approved harvesting llama.cpp/ggml (MIT) optimizations into our CUDA
emission. He then asked, as the PM relayed it:

> *"Should we also look at adding their tweaks not just to Unpopped's emitters but
> also to Unpopped's parsers? ... if an Unpopped parser sees a pattern that is an
> optimization for a specific type of task for a specific architecture, it should
> be able to tell Unpopped about that and Unpopped's IR should be capable of
> storing that as an non-architecture-specific notation ... Do the parsers even
> need to know about the optimizations or just enough to strip out the
> optimizations and leave the underlying algorithms in place?"*

The PM's answer, which this doc designs:

- **Parsers lift idioms to arch-neutral intent.** `__dp4a` becomes an int8
  dot-accumulate, a warp-shuffle reduce becomes a group reduction, and `mma`
  becomes a tile matmul-accumulate.
- **Tuning parameters become hints.** Tile sizes, unroll factors and block shape
  are recorded as hints tagged with provenance (source and arch). They are never
  semantics.
- **Emitters re-lower per target.**
- **Rule 1:** lifting preserves semantics, and a round-trip test proves it.
- **Rule 2:** an optimization that changes numerics is recorded in the intent and
  never stripped.

The answer to *"do the parsers even need to know"* is **yes, but only enough to
recognize an idiom and name its intent**. A parser that only strips an
optimization can't tell a numerics-neutral one (a shuffle tree over integers)
from a numerics-changing one (`__expf`). Rule 2 needs exactly that distinction,
so the parser must know it. §2 shows that Unpopped's current lifter already gets
this wrong.

## 1. The finding that shapes the design: the intents mostly exist already

The PM's example intents aren't new IR. Unpopped's `Access` enum
(`crates/unpopped/src/ir.rs:1604`) already represents each of them as
*semantics*. Emitters already supply the cooperative machinery as a *schedule*:

| PM's intent | Existing IR node | Cooperative lowering already emitted today |
|---|---|---|
| group reduction | `Access::Reduction` (`ir.rs:1616`), `Access::RowReduce` (`ir.rs:1656`) | baracuda `cuda.rs:4660-4700`: `warp_sum`/`block_sum` via `__shfl_down_sync` and a `__shared__ smem[32]` tree |
| group scan | `Access::Scan` (`ir.rs:1688`) | baracuda `cuda.rs:5374-5376`: `__shared__ warp_buf/warp_off[32]` |
| tile matmul-accumulate | `Access::Contraction` (`ir.rs:1672`) with `AccumSpec` (`ir.rs:1925`) | SIMT only today. `AccumSpec::WideFloat` is its sole variant, and its doc anticipates *"Tensor-core/TF32 policies join as variants with honest contract flips"* |

`plan::Schedule::RowReduce` (`crates/unpopped/src/plan.rs:25` ff.) describes
itself as *"one block per output row (warp-shuffle + shared-memory tree
reduce)"*. **So Unpopped already has the model the PM asked for:** a
non-cooperative semantic node plus an emitter-chosen cooperative schedule. What it
lacks is the parse-side direction. A source kernel that already contains the
cooperation can't be lifted back to the node, because `convert.rs`'s residue
lists refuse `__shared__`, `__shfl` and `__syncthreads` outright
(`crates/unpopped/src/convert.rs:61-88`).

### 1a. This is the `groupshared` scope call, made explicitly

`docs/groupshared-sizing.md` escalated a scope question: *does Unpopped's IR ever
model workgroup-cooperative execution?* **Adopting this design answers it: no, the
IR never models cooperation, and parsers lift cooperative source into the
existing non-cooperative intent.** The residue refusal for
`__shared__`/`__shfl`/`__syncthreads`/`groupshared` then changes meaning:

- **Today:** refuse, because the construct is hand-optimized.
- **After this design:** refuse only if no idiom recognizer claims the construct.

The change is deliberate and goes through the PM gate as that ruling. It is not a
quiet reversal. `groupshared-sizing.md` said closing the bucket needs *"a new IR
shape modeling a fixed-size cooperative scratch buffer ... and an explicit
barrier/phase ordering"*. For the reductions, softmax and norms in that bucket,
that is wrong, as §1 shows: the IR already holds their semantics. The claim is
replaced in that doc by this PR, not caveated.

Of the 12 kernel families in that bucket, `reduce`, `reduce_last_dim`,
`rms_norm_last_dim`, `softmax`, `arg_reduce_last_dim` and `layer_norm_last_dim`
have an existing intent node, subject to the gaps listed in §3.
`flash_attention`, `matmul_q4_0_tiled` and `qmatvec_q4_0` need the §3 additions.
The backward kernels are unassessed.

## 2. Rule 2 already has a live violation in Unpopped

`crates/unpopped/src/lift.rs:436-440`:

```rust
"expf" | "__expf" => UnaryOp::Exp,
"logf" | "__logf" => UnaryOp::Log,
"sqrtf" | "__fsqrt_rn" => UnaryOp::Sqrt,
```

`__expf` and `__logf` are CUDA's fast-math approximations: reduced accuracy, with
error that grows with |x|. `expf` is the accurate libm function. The lifter maps
both spellings to the same `UnaryOp::Exp`. The shared C-family lowering spells
`UnaryOp::Exp` as `expf` at f32 (`crates/unpopped/src/cfamily.rs:738`), and
baracuda's emitter output shows it (e.g. `cuda.rs:11344`). So `__expf(x)` → lift → re-emit
yields `expf(x)`. That is a different function, and nothing records that the
source was approximate. This is exactly what rule 2 forbids: *stripping* a
numerics-changing optimization.

(`__fsqrt_rn` is IEEE round-to-nearest sqrt, the same result as non-fast-math
`sqrtf`, so that line is correct.)

- **Blast radius, measured:** `convert.rs:38` imports the same `unary_fn`, so both
  lifters are affected.
- **The mapping is pinned by tests in two repos:**
  - `unpopped` `convert.rs:954` (`cuda_lifts_unary_intrinsic`) and
    `lift.rs:500` (`lifts_unary_intrinsic`)
  - `baracuda` `crates/baracuda-cuda-parse/tests/convert.rs:84`
- **Why the fix is not in this PR:** the scope is docs only. Also, a fix changes
  a lift result that an adopter's test asserts, so it has to be sequenced with
  baracuda. It is reported to the PM as a `[FINDING]`, with the fix shape in §4.3.
- **This defect is the harness's first positive control (§5.3).** On device, the
  current mapping must fail the round-trip.

A related boundary problem: `unary_fn` and `binary_fn` are **CUDA spellings that
live in neutral core** (`lift.rs:436,457`). baracuda's parser crate supplies its
own `Frontend` (`baracuda-cuda-parse/src/lib.rs:72`), but it can't supply its own
intrinsic table, because `Frontend` (`convert.rs:121-141`) has no field for one.
§4.2 moves the table.

## 3. The intent set

**Principle: add numerics axes to existing nodes before adding nodes.** Every row
below either reuses an `Access` variant or extends one. A new node is proposed
only where a family's *semantics* doesn't fit.

### 3.1 What is intent and what is a hint: the governing test

> **If a parameter can change the output bits on a fixed input *outside the
> freedom the intent already declares*, it is intent. Otherwise it is a hint.**

This test resolves the tension that makes the PM's two rules look contradictory:

- For a float reduction, tile shape and block size change summation order, so
  they change the bits.
- Rule 1 wants the round trip to reproduce the bits. Rule 2 wants rounding
  changes recorded in the intent.

The resolution is that the intent declares a **reassociation freedom**, and hints
choose a point inside it:

| Numerics axis (intent) | Values | Where it attaches |
|---|---|---|
| accumulator dtype and overflow | `F32`, `F64`, `I32Wrap`, `I32Sat` | `AccumSpec` / `ReductionAccum` (new variants) |
| reassociation freedom | `Reassociable` (any order, band from `required_fidelity`), `Pinned` (source order is semantics) | same |
| product precision | exact, or hardware-defined (tensor-core internal rounding) | `AccumSpec` (new variant) |
| approximate function | `Exact` \| `Approx{bound}` per `UnaryOp` node | `ScalarExpr` qualifier (new) |
| denormal mode | `Preserve` \| `FlushToZero` | `OpDef` (new field) |
| reformulations that round differently | e.g. online softmax | the node's own variant (§3.2) |

**Hints carry the order within `Reassociable`: the tree shape, the tile and the
split.** The original-arch re-emit honors the hints and reproduces the bits. A
different target may ignore them, and must then emit a variant whose fidelity
relative to the lifted source is declared. That uses the existing
`VariantFidelity::DeterministicallyDivergent` (`backend.rs:154` ff.), which
*"may never be selected silently"*. No new selection policy is needed.

**`I32Wrap` vs `I32Sat` is not pedantry.**
- CUDA `__dp4a` wraps on overflow.
- Vulkan's `OpSDotAccSatKHR` saturates.

They are different arithmetic, and both are 4-wide.

**This is also why "4-wide" doesn't belong in the intent.** The intent is *int8
dot-accumulate into i32, wrapping*. Width 4 is the lowering on `sm_61+`. A
`dot8`-shaped instruction on another vendor is the same intent. That is a
deliberate refinement of the PM's example wording.

### 3.2 Per kernel family

Families come from what Fuel actually runs through baracuda (kernels under
`baracuda-kernels-sys/kernels/`; Fuel has no `.cu` of its own).

**Quant matmul / mmvq**
(`gguf/mmvq.cu`, `include/baracuda_mmvq_batched.cuh`, `include/baracuda_mmvq_multim.cuh`)
- **Intent:** `Access::Contraction`, with the weight as a packed `I4`/`I8`
  operand plus its scale sibling. That is the sk4 model already pinned by
  `unpopped-vocab/tests/scale_sibling_model.rs` and `structure_key.rs:696` ff.
- **Numerics:**
  - The f32-activation path dequantizes into a float accumulate:
    `AccumSpec::WideFloat`, `Reassociable`.
  - The q8_1 path is an exact `I32Wrap` dot per block, followed by a
    float combine across blocks.
  - **The activation quantization to q8_1 is its own op in the graph, never
    folded into the matmul**. It is the numerics change, and rule 2 keeps it
    visible.
- **Gaps:**
  - `Contraction` v1 is *"rank-2 single-K dense row-major"* with uniform dtype.
  - It needs mixed operand dtypes.
  - It needs a **block-dequant read-through**: element `k` reads `scale[k / B]`
    under a named block layout. That is a `View` in the `OpDef::views` sense
    (`ir.rs:1970`), not a new `Access`.
  - ggml's nibble order (low nibbles = elements 0..15 of a Q4_0 block) is a
    *layout name* with provenance. It is not semantics.

**Attention**
(`flash_sdpa.cuh`, `flash_decoding.cuh` (WMMA), vendored FA2/FlashInfer)
- **Intent:** a **new `Access::Attention`** node. Score contraction, softmax and
  value contraction can't be expressed as a chain of today's nodes without
  materializing the score matrix, which is the thing flash attention exists to
  avoid.
- **Numerics:**
  - `softmax: ThreePass | Online { kv_block }`. Online softmax rescales at block
    boundaries, so `kv_block` changes bits, and by §3.1 it is intent, not a hint.
  - `mma` inputs use the hardware-defined product-precision axis.
- **Gaps:** the node itself. Causal/ALiBi masks. Paged KV (the index-read
  machinery of `ReadIndex`, `ir.rs:2096`, is the likely base).

**RMSNorm**
(`norm/rms_norm_fp.cu` → `baracuda_norm.cuh`)
- **Intent:** `Access::RowReduce` with 1 stage. `ir.rs:1650` ff. names it as an
  instance.
- **Numerics:** `Reassociable`, plus a per-kernel accumulator dtype.
- **Gaps:** RowReduce v1 is *"single input"*. The γ weight needs a
  broadcast-along-rows second input. The doc already names LayerNorm's per-column
  weight as the follow-up.

**RoPE**
(`attention/rope_fp.cu`)
- **Intent:** already expressible, for the interleaved `(2i ↔ 2i+1)` form.
  `crates/unpopped/ondevice/README.md:1139-1153` records all 3 blockers closed:
  - the cos/sin cache read;
  - the stride-2 output;
  - the pair-partner read, as a `+1` `BaseOffset::Runtime` (`ir.rs:2155`).

  A two-launch generated decomposition is **memcmp bit-exact** against the
  bespoke `launch_rope_apply_fp<float>`, at 0.77× its throughput. The rotate-half
  `(i ↔ i+d/2)` form is not recorded there, and I have not measured it.
- **Numerics:** `sin`/`cos` carry an `Approx` qualifier if the source computes
  them with `__sinf`/`__cosf`, which is the same trap as §2.
- **Gaps:** none in the IR. The gap is on the parse side: the lifter refuses the
  pair read as a *"non-elementwise index"*. That is the same refusal category as
  the corpus's 22-file bucket, though I haven't checked which files in that
  bucket are RoPE.
  Fusing the two launches back into one is a hint/lowering matter (§3.1), not
  intent.

**Softmax**
(`baracuda_softmax.cuh`)
- **Intent:** `Access::RowReduce` with 2 stages. `ir.rs:1650` ff. names it as an
  instance.
- **Numerics:** `Exp` qualifier (`Exact`/`Approx`) and `Reassociable`.
- **Gaps:** none structural.

**Dequant**
(`gguf/dequantize.cu`)
- **Intent:** `Access::Elementwise` over the packed operand and its scale sibling.
- **Numerics:** exact (a scale multiply).
- **Gaps:** the same block-dequant read-through `View` as quant matmul.

Warp-shuffle reduce and `mma`, the PM's other two examples, are covered above:
they lift to the `Reduction`/`RowReduce` and `Contraction` rows.

## 4. Hint and provenance storage

### 4.1 Where hints live, and where they must not

A hint **must not** reach any of the following:

- **`OpDef`**, whose derived `PartialEq`/`Serialize` is the op's identity.
- **The recipe / semantics DAG** (`recipe.rs:76`), which is the neutral
  KISS-Ops op-DAG.
- **`StructureKey`.** `capability.rs` already establishes why: *"a key is a pure
  function of the request, so two candidates for one request cannot differ in
  it"*.

Hints are a **sidecar that travels with a lift result and is consumed by variant
emission**. `capability.rs` already names that seam as the home of
schedule variation: *"That somewhere is `Backend::lower_variants`"*.

```rust
// Shape, not code. Names are provisional.
pub struct Lifted {
    pub op: OpDef,            // intent: semantics + numerics axes (§3.1)
    pub n_inputs: u8,
    pub hints: Vec<Hint>,     // NEW: advisory, never identity
}

pub struct Hint {
    pub kind: HintKind,       // Tile{m,n,k} | Unroll(n) | Block(x,y,z) |
                              // ReduceTree(TreeSpec) | VecWidth(n) | Layout(name) | …
    pub observed_for: TargetId, // the arch the source was tuned for, e.g. `cuda:sm61`
    pub origin: Origin,       // InHouse | ThirdParty{..} (§4.3)
    pub source_span: Option<(usize, usize)>, // byte range in the lifted file
}
```

- **Emitters take hints as input to `lower_variants`.** A variant built from a
  hint names it in `Variant::tag` / `launch_note` (`backend.rs:268-290`).
- **The revision hash covers hint-driven changes automatically:**
  `kernel_revision_hash` is FNV-1a over emitted source (`backend.rs:56-61`), so a
  hint that changes code changes the hash. Cache validity needs no new mechanism.

### 4.2 Idiom recognizers live in the frontend crate, not in core

By the umbrella rule, *"no emitter-specific code ... unless it is in an
emitter-specific sub-crate"*, and CireSnave's parser/emitter correction quoted in
`docs/parser-emitter-ownership-map.md`, a CUDA idiom table is baracuda's.
`Frontend` (`convert.rs:121`) grows two fields:

- an intrinsic table: CUDA name → `(UnaryOp, qualifier)`, replacing core's
  `unary_fn`/`binary_fn`;
- a list of idiom recognizers: CST pattern → `(intent, hints)`.

`baracuda-cuda-parse` then supplies both. Core keeps only the neutral types
(`Hint`, `Origin`, the numerics axes) and the walker.

⚠️ **Adding these fields is a breaking change.** `Frontend` is not
`#[non_exhaustive]` (`convert.rs:120-121`), and baracuda builds it with a struct
literal (`baracuda-cuda-parse/src/lib.rs:72`). The same is true of `Lifted`
(`lift.rs:112-113`, `#[derive(Debug)]` only), so a new `hints` field breaks
anyone who constructs one. Both belong in a breaking release, not in a minor.
(This once named the `ArchSku::Sm61` release, which option C dropped; see §6.)
Marking them `#[non_exhaustive]` in that release keeps later additions minor.

### 4.3 Origin: where "I am not a plageurist" is enforced

```rust
pub enum Origin {
    InHouse,                     // written in this ecosystem; nothing to credit
    ThirdParty {
        project: &'static str,   // "llama.cpp / ggml"
        repository: &'static str,// upstream URL
        commit: &'static str,    // the exact upstream revision the technique was read at
        path: &'static str,      // upstream file, with a line range where applicable
        license: &'static str,   // SPDX: "MIT"
        notice: &'static str,    // the upstream copyright line, verbatim
    },
}
```

It is an enum rather than an optional struct so that "no origin recorded" can't
be written. Every hint and every registry row must choose one of the two.

`Origin` is carried in three places:

1. **In each `Hint`**, so a lifted artifact names what it copied.
2. **In the technique catalogue entry (§7)** that owns the recognizer or lowering.
3. **Through `Provenance`** (`backend.rs:36`, `#[non_exhaustive]`, so a field add
   is minor). A new `techniques: Vec<Origin>` field means a *generated kernel*
   credits every harvested technique it used.

(3) is what makes credit reach the artifact rather than stop at the repo.

**The MIT notice obligation is met where the code is copied.** Only lowerings copy
code, and they live in baracuda. baracuda already maintains
`crates/baracuda-kernels-sys/LICENSE-thirdparty.md`, whose line 54 is
*"llama.cpp / ggml-cuda (GGUF block-format dequant + MMVQ)"*. A recognizer copies
no upstream code; it only *reads* the idiom. It still carries `Origin`, because
the technique's identity came from upstream.

**The §2 fix, in this shape:**
- `__expf` → `UnaryOp::Exp` qualified `Approx{bound}`, where `bound` is the
  documented error formula for the intrinsic.
- A hint carries `observed_for` (the source's CUDA target, e.g. `cuda:sm89`)
  and the exact spelling, so the CUDA re-emit reproduces `__expf`, and another
  target picks any implementation within `bound`.
- Until the qualifier exists, the interim fix is to stop mapping
  `__expf`/`__logf`, so they become `Unrecognized`, i.e. an honest refusal. That
  still changes the pinned tests in both repos.

## 5. The round-trip harness (rule 1)

### 5.1 What "numerically equal" means

**Bit-identical, on the original arch, with hints honored.** Precedent:
`crates/unpopped/ondevice/lift_roundtrip_validate.cu` already requires
*"exhaustive BIT-exactness (memcmp), not a tolerance"*. A banded comparison would
hide a numerics change inside the band, and the §2 defect is that kind of
change: `__expf` vs `expf` differs by a few ULP near zero, and by more as |x|
grows.

A second, weaker leg runs on other targets: the re-emit with hints dropped must
land inside `oracle::required_fidelity` (`oracle.rs:3176`) for the intent's
numerics class, and must declare its `VariantFidelity`. Note
`docs/normative-seams.md` §3: `ulp_bound` is a CUDA table applied to every
target, so this weaker leg inherits that known limitation for non-CUDA targets.

### 5.2 Three legs

| Leg | Runs where | Asserts |
|---|---|---|
| **A. structural** | no GPU needed. It runs in whichever repo holds *both* that language's frontend and its emitter: baracuda for CUDA, Unpopped for Slang | `lift(s)` is deterministic; `lift(emit_orig(lift(s))) == lift(s)`, with the intent AND the hints equal. Catches hint loss and intent drift without hardware. It presupposes the frontend accepts its own emitter's output. That is unmeasured today, and it is worth knowing in its own right. |
| **B. device, original arch** | the emitter's repo (baracuda for CUDA), built on CireSnave's GPU machines before merge (umbrella-model precedent) | `run(s)` and `run(emit_orig(lift(s)))` are `memcmp`-equal over random inputs plus edge seeds: ±0, ±Inf, NaN, denormals, and integer overflow for `I32Wrap` |
| **C. device, other arch** | same | within `required_fidelity`, and the variant's declared fidelity is honest |

Leg B belongs in the emitter's repo, because Unpopped can't depend on
`baracuda-cuda-emit` without a cycle (it depends on `unpopped`). **The cost of
that split is already measured:**
- The existing device leg moved to `baracuda-cuda-emit/tests/lift.rs:68` in
  `b68116c` (2026-08-06).
- Unpopped's `.cu` header and `ondevice/README.md` still told readers to run
  `cargo test -p unpopped --lib lift::tests::dump_lift_roundtrip`. That test has
  not existed in this repo since then: `git grep dump_lift_roundtrip` finds no
  definition here, and the same query finds it at baracuda `lift.rs:68`.
- **The drift is not limited to this harness.** All 21 `cargo test -p unpopped
  dump_*` run lines under `crates/unpopped/ondevice/` (14 in `README.md`, 7 in
  `.cu` headers) name tests that exist only in `baracuda-cuda-emit`. That is 13
  distinct test names:
  - checked by a `fn <name>\b` query over each repo's tracked files;
  - 0 files here and 1 file each in baracuda;
  - control: the same query finds `fn lifts_fused_multiply_add` in `lift.rs`.

  The README's `cargo run -p unpopped --bin kernelgen` names a binary that lives
  at baracuda `crates/baracuda-cuda-emit/src/bin/kernelgen.rs`. This PR corrects
  all of them, and adds a note that the harnesses live here while their
  generators live in baracuda.

A cross-repo leg needs a guard on the side that *can* run, which is the lesson of
memory `panic-text-is-a-cross-repo-api`.

### 5.3 Controls: a round-trip that can't go red certifies nothing

- **Fixture per recognizer.** Any recognizer rule without a round-trip fixture
  fails a census. The recognizers live in each frontend crate (§4.2), so the
  census is an exported check that each frontend crate runs over its own `Frontend`.
  Unpopped runs it over the in-tree Slang frontend. It uses the same shape as
  `unpopped-conformance/tests/tests_live_in_the_crate_they_guard.rs`: a census
  that closes the class, not a per-instance check.
- **Positive control for leg B: today's `__expf` mapping.** Lifting
  `out[i] = __expf(in0[i])` and re-emitting gives `expf`, and leg B must report
  `FAIL` on it. If leg B passes on the current mapping, the harness is broken.
  This is the *reachable* mutation, not a contrived one: it is live in two repos
  (memory `the-reachable-mutation-is-the-silent-one`).
- **Positive control for leg A:** drop one hint in the re-lift and assert
  inequality.
- **Leg B's inputs must include values where the idiom matters.** The existing
  harness's random inputs are in `[-4, 4]`. `__expf`'s error grows with |x|, so
  edge seeds must include large |x|, or the control can pass by accident.

## 6. How Sm61 enters the arch vocabulary

> **Updated 2026-10-03: option C (board #106) settled this, and item 2 below is
> superseded.** `TargetId` is the architecture identity. `ArchSku` stays closed
> as a baracuda-cutlass dispatch SKU and gets **no** `Sm61` variant; the
> followers that took it as an arch identity (baracuda's NVRTC compiler, the
> seam request) move to `TargetId`. Item 3 shipped in `0.14.1` (#34). So
> Unpopped's only remaining surface is the telemetry ingest follow-up named
> under item 2.

Three separate surfaces, and **only two were Unpopped's**:

1. **The `cuda:sm61` token (vocabulary): baracuda's, not ours.**
   - `TargetId` validates grammar and never vocabulary (`target.rs` module doc,
     §6.8-0004). `cuda:sm61` satisfies §6.8-0001/0005 as `TargetId::parse`
     in `target.rs` implements them. Since `0.14.1` this is executed:
     `capability.rs`'s token test parses `cuda:sm61` and resolves it to its row.
   - The `cuda:` namespace's closed token set is owned by `baracuda-cuda-vocab`
     (`src/lib.rs:105-114`, `{sm80, sm89, sm90, sm90a}`). KISS#514 repointed
     `cuda`'s reference implementation there. baracuda adds `sm61`.
2. ~~**`ArchSku::Sm61` (in `unpopped-vocab/src/layout.rs`): ours, and a breaking
   change on purpose.**~~ **Superseded by option C (2026-10-03): no variant is
   added.** The followers below move to `TargetId` instead, which is the
   "alternative" this item once argued against. The arguments are kept for the
   record.
   - One Unpopped follow-up remains: `telemetry::merge_reports` reads a compute
     capability through `arch_sku_of`, which maps only 8.x and 9.x, so it drops
     telemetry from every other arch. It will mint a `TargetId` from the
     capability once baracuda (the `cuda:` namespace owner) publishes the
     capability-to-token rule.
   - The enum is *"Intentionally NOT `#[non_exhaustive]` ... New variants are a
     deliberate breaking-change event."* Pre-1.0 that means **0.11.x → 0.12.0**,
     with the number allocated by the PM at gate time (CLAUDE.md §9).
   - Both followers compile-trap on it, which is the intent:
     - baracuda `nvrtc.rs:80-84` `arch_flag`;
     - Fuel `telemetry/baracuda_provider.rs:256-263` `arch_sku_digits`, an
       exhaustive match that has no wildcard.
   - Fuel imports `ArchSku` only through `baracuda_kernels_types`, which
     re-exports `unpopped_vocab::*` (Fuel `jit_carrier.rs:52`). So Fuel follows
     once baracuda re-pins.
   - *Alternative, not recommended now:* stop growing `ArchSku` and move the
     followers to `TargetId`. That is the long-run direction `target.rs`
     documents, but it is a larger cross-repo migration than one variant.
3. **The `cuda_capabilities(6, 1)` row: ours, additive. Shipped in `0.14.1`
   (#34)**, together with rows 10.0–12.1. The original plan follows.
   - The table starts at 7.0 and returns `None` for 6.1 today, by design (*"an
     unknown compute capability returns `None`, never a nearby row"*).
   - The row is transcribed from NVIDIA's *Technical Specifications per Compute
     Capability* when implemented. It is not asserted here from memory.

**What Sm61 changes for lowering, the reason it matters to this design:**
- sm_61 has `__dp4a` and no tensor cores (`mma` needs sm_70+). A
  `Contraction` lifted from an `mma` kernel on sm_89 therefore re-lowers to dp4a
  or SIMT on sm_61. That is the same intent under a different lowering, and the
  point of §3.
- Feature facts like "has dp4a" are capability-set vocabulary, owned by the
  namespace owner (baracuda-cuda-vocab / KISS #171's capability manifest), not by
  `TargetCapabilities`, which carries only resource numbers.

**Not ours to decide:**
- baracuda's README (`:418`) marks ≤ sm_75 **unsupported**, and
  `baracuda_mmvq_multim.cuh:103` says *"baracuda targets sm_80+"*.
- Whether baracuda *emits* for sm_61 is baracuda's call. Unpopped only makes the
  arch *representable*.

## 7. The technique registry: structured data, not prose

CireSnave asked, as the PM relayed it (PM `[TASK]`, 2026-09-30T05:38Z):

> *"For the optimizations that any of Unpopped's emitters have for any
> architecture, are we keeping a list of those somewhere so other Unpopped emitter
> implementers can try each one or a variant of it on their own architectures? If
> not, should we?"*

**No such list exists today.** The PM measured it, and I re-checked:
- **Arch-neutral algebraic rewrites are shared:** the e-graph in `optimize.rs`,
  proven by `tests/optimizing_never_changes_the_answer.rs`. These rewrite the
  *intent* and are **not** registry entries.
- **Arch-specific lowering techniques exist only as code inside each emitter.**
- `docs/catalog.md` catalogues *kernels*, not techniques. It is also marked
  **⛔ RETIRED 2026-08-14** because *"the catalog has no consumer"*: it
  *"never named a requesting party"*.
- In baracuda, a search of `@ab2e0bf` for `technique catalog` and its spellings
  found 0 hits, and `gh issue list --search technique` found nothing. As a control,
  the same kind of query finds `docs/design/kernel-specialization.md:52`
  "Structural predicate catalog".
- The llama.cpp "Phase-A catalogue" the PM mentions is not on baracuda's
  `origin/main` at that revision.

**So yes, we should keep one, and it belongs in Unpopped.** The umbrella model
exists for *"third-party discoverability"* (memory
`unpopped-umbrella-emitter-model`). A list that tells any emitter implementer
which techniques exist, on whose hardware, and with what evidence is that purpose.

**The retired catalog's failure is the registry's first requirement: name the
consumer.** It has three, each concrete:
1. **CireSnave's question itself:** implementers of other Unpopped emitters.
2. **`unpopped-slang`, next.** Its RowReduce schedule (§10) is the first work
   item that would *read* the `block-tree-reduce` row to decide what to build.
   That is the first consumer who can say "that isn't what I need".
3. **baracuda's catalogue (§7.4)**, as the first *writer* of third-party rows.

If none of these reads it by the time the schema lands, that is the signal the
catalog retirement recorded, and the registry should stop there.

It doesn't breach *"no emitter-specific code in core"*, for three reasons:
- It is **data, not code**.
- Every vendor's spelling sits in it symmetrically (§8).
- The code it points at stays in the emitter crates.

### 7.1 Format and location

- **The file:** `crates/unpopped/techniques/registry.json`.
- **Why JSON:** `serde_json` is already an `unpopped` dependency
  (`crates/unpopped/Cargo.toml:53`), and `unpopped-vocab/kiss/*.json` is the
  existing precedent for normative data files.
- **It ships inside the published crate**, and is exposed as
  `unpopped::techniques::REGISTRY` (the bytes, via `include_str!`). An
  out-of-tree emitter's own tests then read *the same bytes* Unpopped's CI read.
  That is what makes the cross-repo check in §7.3 possible.

### 7.2 Schema (v1)

```jsonc
{
  "schema": "unpopped-technique-registry-v1",
  "features": [                      // generic hardware features — §8's rows
    { "id": "int8-dot-accumulate",
      "attrs": ["width", "signedness", "overflow"],   // MUST be stated per spelling
      "spellings": [ { "namespace": "cuda", "form": "__dp4a", "since": "sm_61",
                       "attrs": { "width": 4, "signedness": "s8*s8|u8*u8", "overflow": "wrap" } } ] }
  ],
  "techniques": [
    { "id": "block-tree-reduce",
      "summary": "warp-shuffle partials, then a shared-memory tree across warps",
      "serves":  { "access": ["Reduction", "RowReduce"], "numerics": { "assoc": "Reassociable" } },
      "needs":   ["subgroup-shuffle", "workgroup-shared-memory", "workgroup-barrier"],
      "fidelity": "DeterministicallyDivergent",      // vs the intent's reference order
      "variants": [ "subgroup-arithmetic instead of shuffle", "single-warp rows skip the smem stage" ],
      "limits":   [ "tree width baked to 32 lanes is a wave64 hazard (§8)" ],
      "origin":   { "kind": "in-house" },            // or an Origin record (§4.3)
      "implementations": [
        { "emitter": "baracuda-cuda-emit",
          "repo": "ciresnave/baracuda", "at": "<sha>",
          "status": "implemented",                    // implemented | proposed | tried-no-gain | declined
          "code":   { "path": "crates/baracuda-cuda-emit/src/cuda.rs", "symbol": "block_sum_" },
          "parity": { "test": "<test fn name>", "path": "<test file>" },
          "benchmark": { "harness": "<path>", "hardware": "<device id>", "result": "<figure>",
                         "ref": "<sha>", "runs": 0, "dispersion": "<max/min>" } } ] }
  ]
}
```

- **`implementations[].status` includes negative results.** `tried-no-gain`
  (with its benchmark) and `declined` (with the reason) are what let *"other
  emitter implementers try each one"* without retrying what already lost.
- **`benchmark` carries its predicate and ref** (CLAUDE.md §7): `hardware`,
  `ref`, `runs` and `dispersion`. A bare figure becomes a claim about the present
  once repeated. `runs`/`dispersion` exist because baracuda measured a box where
  *"12 of 12 repeatedly-run cells vary by more than ±5%"*, recorded in
  `unpopped-vocab/tests/the_noise_floor_gates_flips_not_first_decisions.rs:4-7`.
- **The `features` table is keyed by the generic feature, and every spelling
  must state the numerics attributes.** The generic name alone would overclaim
  equivalence. x86's `VPDPBUSD` multiplies **u8 × s8**, CUDA's `__dp4a`
  intrinsic overloads are same-signed, and Vulkan offers wrapping and saturating
  forms (§3.1). All four are "4-wide int8 dot-accumulate", and they are not
  interchangeable without the attributes.
- **Seed rows** are in-house techniques Unpopped's own ecosystem already ships:
  - `block-tree-reduce`: baracuda `cuda.rs:4660-4700`.
  - `warp-scan`: *"Kogge-Stone warp scan + cross-warp exclusive offset"*,
    baracuda `cuda.rs:5346`.
  - `split-k`: the `"splitk"` variant tag, `backend.rs:274`.
    - Parity harness: `ondevice/splitk_validate.cu`.
    - Benchmark: `audit_reduce_softmax.cu` measured 242 vs 171 GB/s (1.41×) over
      the legacy reducer for sum axis-0 `[65536×1024]` (`ondevice/README.md:416`).
      That is one recorded run, with no dispersion, which is exactly the gap the
      `runs`/`dispersion` fields exist to expose.
  - `vectorized-access`: `Schedule::Vectorized`, `plan.rs:25` ff.

  Then baracuda's llama.cpp batch once it lands, each row with an `Origin`.

### 7.3 The conformance test: the registry cannot overclaim

This is a test in `crates/unpopped/tests/technique_registry.rs`, in the crate
that owns the file (memory `a-test-in-the-wrong-crate-never-runs`). It asserts:

1. **Schema:**
   - ids are unique;
   - every `needs` names a `features` id;
   - every spelling states every attribute its feature declares.
2. **`serves.access` resolves against the real enum.**
   - Resolution goes through an **exhaustive** `match` over `Access`, written in
     `src/techniques.rs`.
   - It has to live in `src/`, because `Access` is `#[non_exhaustive]` to
     everything outside the crate, and `tests/*.rs` is outside it. An integration
     test could only match with a `_` arm, which is exactly what goes stale.
   - Inside the crate, adding a variant breaks the build and forces a decision. A
     string list would silently go stale.
3. **In-tree rows resolve to real code** (`unpopped-cpu-c`, `unpopped-slang`):
   - `code.path` is a tracked file, enumerated from the index as `git ls-files`
     does (CLAUDE.md §5b), never by walking the disk;
   - `code.symbol` occurs in it;
   - `parity.test` is a `#[test] fn` in that crate's own test target.
4. **Out-of-tree rows are checked on the side that can run them.** Unpopped CI
   can't read baracuda.
   - `unpopped::techniques::verify_rows(emitter, repo_root)` is exported, and
     baracuda's test suite calls it on its own checkout, against the registry
     bytes of the `unpopped` version it pins.
   - Unpopped's side asserts only that every out-of-tree `implemented` row carries
     `repo` and a full `at` sha. A row that can't be pinned can't be checked by
     anyone.
   - Precedent: memory `panic-text-is-a-cross-repo-api`, where the guard lives on
     the side that can run it.
5. **Mutation controls, inside the test itself.** In-memory rows with a missing
   path, a missing symbol, and a parity test that doesn't exist must each be
   **rejected**. Otherwise a checker that returned `Ok` unconditionally would pass
   every real row (memory `guards-that-cannot-fire`).

The test **cannot** verify that a benchmark figure is true. It verifies that the
figure names its harness, hardware, ref and run count, so that anyone can
re-measure it. That boundary is stated in the test's own doc.

### 7.4 How baracuda's catalogue feeds it

baracuda's catalogue (not yet on `origin/main`) is the *code* side. An entry there
owns the recognizer, for the parse side via `Frontend` (§4.2), and the lowering,
for the emit side via `lower_variants`. The registry row is the *published* side:
- `id` is shared;
- `implementations[]` points at baracuda's code at a pinned sha;
- `origin` carries the llama.cpp `Origin`.

**One id across the recognizer, the lowering and the registry row** means a
technique can't be emitted without being liftable, or lifted without being
listed, unnoticed. Memory `unpopped-holds-the-kiss-emit-pen` makes the point that
one author makes drift less likely, while a shared definition makes the asymmetry
impossible. The shared id is the second kind.

## 8. Hardware-feature mapping: the table the registry's `features` encode

CireSnave's goal, as the PM relayed it (PM `[TASK]`, 2026-09-30T05:40Z):

> *"my RTX 4070 is accessible through CUDA but is also accessible through Slang ->
> SpirV -> Vulkan... most of the CUDA optimizations for my RTX 4070 would apply
> equally well through the other path... It would be game changing to take the
> vast CUDA kernel libraries and use those as input to generate AMD kernels that
> are as optimized."*

Keyed by generic feature.

⚠️ **The spellings below are from memory of the vendors' documentation. None was
checked for this doc.** Each becomes a registry `spelling` only when an entry
cites its source, and the per-device columns are *queried* (e.g. `vulkaninfo`),
never asserted. This is the same rule `capability.rs` applies to its own table.

| Generic feature | CUDA | Slang → SPIR-V → Vulkan | AMD (HIP) | CPU |
|---|---|---|---|---|
| subgroup shuffle / reduce | `__shfl_*_sync`, `__reduce_*_sync`; 32 lanes | `WaveReadLaneAt` / `WaveActiveSum` → `OpGroupNonUniform*` (Vulkan 1.1 subgroup ops); size queried | `__shfl*`, DPP; **wave64** (GCN/CDNA), wave32 or 64 (RDNA) | SIMD permutes; or a serial loop (one "lane") |
| workgroup shared memory | `__shared__` | `groupshared` → `Workgroup` storage class | LDS | not needed (a stack or cache-resident buffer) |
| workgroup barrier | `__syncthreads` | `GroupMemoryBarrierWithGroupSync` → `OpControlBarrier` | `__syncthreads` | none (one thread) |
| int8 dot-accumulate | `__dp4a` (sm_61+) | `VK_KHR_shader_integer_dot_product` (`OpSDot`/`OpUDot`/`OpSUDot`, `…AccSat`) | `v_dot4*` builtins, where the ISA has them | x86 VNNI `VPDPBUSD(S)` (u8×s8); Arm `SDOT`/`UDOT`/`USDOT` |
| tile matmul-accumulate | `mma.sync` / `wmma` (sm_70+) | `VK_KHR_cooperative_matrix` (device-reported shapes) | MFMA (CDNA), WMMA (RDNA3+) | Intel AMX, Arm SME: varies, often absent |
| 16-bit storage and arithmetic | `__half`, `half2` | `VK_KHR_16bit_storage`, `shaderFloat16` | packed fp16 | F16C (convert only), AVX512-FP16 |
| wide vector access | `float4` loads | `vec4` loads, alignment decorations | `dwordx4` | SIMD loads |

**Not transferable. Marked so the registry says so, never implies otherwise:**
- `cp.async` pipelines (sm_80);
- TMA and `wgmma` (sm_90/90a);
- thread-block clusters and distributed shared memory (sm_90);
- inline PTX;
- `__ldg` and other cache-policy hints (at most an advisory decoration elsewhere);
- warp-specialized producer/consumer schemes built on named barriers.

A registry `feature` for these carries `"transferable": false`, with the reason.
The technique may still be *listed*, so another vendor can look for its own
analogue, but no `implementations[]` row on another target may claim it.

**The transfer hazard is an assumption, not a feature:** a lane count of 32. A
CUDA tree reduction written for 32 lanes is wrong or wasteful on wave64. That is
why the lifted tree shape is a *hint* whose `observed_for` is the source's `cuda:`
target (§4.1), and
never intent. The re-lowering for AMD reads the subgroup size from the target,
not from the source.

## 9. Autotuning: lifted hints are starting points, not answers

"As optimized on AMD" is a claim, and it needs a measurement behind it:

1. **Seed.** Hints lifted from CUDA (tile, unroll, block shape, tree shape) are
   the tuner's *first* candidate on any target, and never its answer on a target
   other than `observed_for`.
2. **Search.** A per-target tuner enumerates the legal space:
   - `capability.rs` already states what the target *permits*;
   - `Backend::lower_variants` already turns each point into a `Variant`;
   - the candidates share a `structure_key` and differ by revision hash, which
     `capability.rs` calls *"precisely the shape a consumer's profiler keys
     siblings on"*.

   No new identity machinery is needed.
3. **Decide with the existing rule.** The tuner calls `unpopped_vocab::winner_of`
   (`dispatch.rs:324`) and `merge` (`dispatch.rs:566`), and doesn't reimplement
   ranking (memory `one-rule-one-author`).
   - `MIN_FLIP_MARGIN` gates a flip of an existing decision, *not* the first one.
     The test cited in §7.2 pins that.
   - So a first decision on a noisy box needs repeated runs and a recorded
     dispersion, or it isn't a decision.
4. **Record.** The winner goes into the registry row for that target, with the
   benchmark fields of §7.2 (`hardware`, `ref`, `runs`, `dispersion`). It also
   goes into the dispatch table, which already records *"Baracuda's measured
   default"* while Fuel remains the runtime selector (`backend.rs:262-266`).

The tuner is target-side code. It lives with whoever owns the target's runtime
(baracuda for CUDA; Vulkane for Vulkan). Unpopped supplies the variant enumeration
and the recording schema.

## 10. First cross-backend proof: same GPU, two paths

**The RTX 4070 through CUDA vs. the same RTX 4070 through Slang → SPIR-V →
Vulkan.** Holding the hardware fixed makes any difference the *path's*, which is
the clean first experiment. AMD adds a hardware variable on top.

What's ready and what isn't, measured at `unpopped@b61a8be`:
- **Ready:** `unpopped-slang` emits the **scalar contiguous `Elementwise` path
  only**. Its own refusal lists `Reduction`/`RowReduce`/`Contraction`/… as
  follow-ups (`crates/unpopped-slang/src/lib.rs:298`). So the first A/B can only
  be an elementwise kernel. That proves the *plumbing*: same IR, two emitters, one
  device, bit comparison plus timing. It proves nothing about optimization
  transfer.
- **The first A/B that tests the thesis** is `RowReduce` (softmax/rmsnorm).
  - It needs `unpopped-slang` to gain the RowReduce schedule, using the §8 rows:
    subgroup ops, `groupshared`, barrier.
  - It then compares against cuda-emit's `block-tree-reduce` on the same inputs.
  - The kernel classes are exact or `Reassociable`, so the bits comparison uses
    `required_fidelity`, and the throughput ratio is recorded with its §7.2
    benchmark fields.
- **A cross-lane dependency:** running SPIR-V on Vulkan is Vulkane's runtime, not
  Unpopped's. The A/B needs their harness or a minimal one of ours, and is a
  coordination item, not a unilateral one.

## 11. What this implies, as sequenced work (not started; each is its own PR)

1. **§2 fix:** stop stripping `__expf`/`__logf`. Coordinated with baracuda's
   pinned test.
2. **Numerics axes (§3.1):**
   - new `AccumSpec`/`ReductionAccum` variants;
   - the `Approx` qualifier;
   - `denormal` on `OpDef`.

   `ReductionAccum` is the sk4 seam, which is KISS-coordinated (memory
   `unpopped-kiss-coordination-state`). Its variants land at the synchronized
   cut, not unilaterally.
3. **`Lifted.hints`, the `Frontend` intrinsic/idiom fields, and moving
   `unary_fn`/`binary_fn` out of core (§4.1-4.2):** breaking, because neither
   struct is `#[non_exhaustive]`. Rides the 0.12.0 event with item 6.
4. **`Origin` + `Provenance.techniques` (§4.3):** an additive minor.
   `Provenance` *is* `#[non_exhaustive]` (`backend.rs:35`).
5. **Round-trip harness (§5):**
   - Unpopped: the exported census, and leg A for the in-tree Slang frontend.
   - baracuda: legs A and B for CUDA, since both the frontend and the emitter are
     theirs.
   - Leg C: with each runtime owner.
6. ~~**`ArchSku::Sm61` + the capability row (§6):** the 0.12.0 breaking event, gated
   on baracuda-cuda-vocab adding `sm61`.~~ The row shipped in `0.14.1` (#34).
   `ArchSku::Sm61` is dropped by option C (2026-10-03, §6): no variant, no
   breaking event.
7. **Intent gaps (§3.2):** block-dequant `View`, RowReduce second input,
   mixed-dtype `Contraction`, and `Access::Attention` last.
8. **Technique registry (§7):** the schema, `REGISTRY`, `verify_rows`, and the
   conformance test with its mutation controls, all *before* any data row. Then
   the in-house seed rows. baracuda's llama.cpp batch follows once its catalogue
   lands.
9. **Slang RowReduce schedule (§10)** in `unpopped-slang`. It is the prerequisite
   for the first A/B that tests optimization transfer. The A/B itself is
   coordinated with Vulkane.
10. **Autotuner (§9):** target-side, owned by each runtime. Unpopped's part is the
    variant enumeration and the registry's recording fields.

**KISS impact.** A lift that succeeds *with hints* is still KISS-Consume's
"lifted" outcome. No new refusal category is needed, and none is proposed here.
If KISS-Consume later needs to *state* the idiom-lifting rules, Unpopped holds
that pen, and the self-imposed cosigner norm applies.
