# Can `Scan` express qwen3_5's linear attention? (M6 assessment)

Docs-only assessment for M6 of the joint plan (`docs/joint-gpu-milestone-plan.md`):
the Qwen 27B-class model is arch `qwen3_5`, and its linear-attention layers are a
recurrence. The PM asked whether Unpopped's `Scan` schedule can express it.

**Short answer: no.** `Scan` folds one scalar per lane with a commutative monoid.
The qwen3_5 recurrence carries a 128 × 128 matrix per head, updated by a
data-dependent rank-1 correction. That is a different kind of state, not a missing
`Scan` parameter.

**But M6 is not blocked on new IR.** Decoding one token at a time composes from
schedules that already exist, with the runtime holding the state. Prefill composes
too, except for one step (a small triangular solve), which has a slow but correct
fallback. A purpose-built schedule is a performance question for when M6 is
scheduled, not a capability gap for bring-up.

Nothing below has been run. "Composes" means the IR has the access kinds; no test
has planned these exact shapes yet (see §5).

## 1. Sources

- **Model config:** `hf://models/Qwen/Qwen3.8-27B/config.json`, read 2026-10-07.
  - `model_type: qwen3_5`, 64 layers, `full_attention_interval: 4`, so 48
    `linear_attention` and 16 `full_attention` layers.
  - `linear_num_key_heads: 16`, `linear_num_value_heads: 48`,
    `linear_key_head_dim: 128`, `linear_value_head_dim: 128`.
  - `linear_conv_kernel_dim: 4`, and `mamba_ssm_dtype: float32` (the state is f32).
- **Modeling code:** `transformers` `main` at commit `c633787a7a` (2026-10-05),
  `src/transformers/models/qwen3_5/modeling_qwen3_5.py` (file sha256 prefix
  `aac2a1bc`). The reference kernels are `torch_recurrent_gated_delta_rule`
  (per token) and `torch_chunk_gated_delta_rule` (chunked prefill). The layer is
  `Qwen3_5GatedDeltaNet`.
- **Unpopped:** `origin/main` at `6ff089d`. `Access::Scan` and `OpDef::scan`
  (`crates/unpopped/src/ir.rs`), `Access::Window` (same file), and
  `MAX_OPERANDS = 8` (`crates/unpopped-vocab/src/structure_key.rs:50`).

## 2. The recurrence (gated delta rule)

Per value head, with state `S` of shape `[k_head_dim, v_head_dim]` = 128 × 128 in
f32, and per token `t`:

```
S   = S * exp(g_t)                 # scalar decay per head
kv  = Sᵀ k_t                       # contract over k_head_dim
d   = beta_t * (v_t - kv)          # the "delta" correction
S   = S + k_t ⊗ d                  # rank-1 update
o_t = Sᵀ q_t                       # contract over k_head_dim
```

`q` and `k` are L2-normalised (optionally inside the kernel), and `q` is scaled
by `1/√k_head_dim`.

## 3. Why `Scan` cannot express it

`Scan` (`OpDef::scan`) is a **prefix fold of one scalar per lane** along the
innermost axis:

- the monoid is `op ∈ {Sum, Prod, Max, Min}` (`Mean` is rejected);
- `pre` is an elementwise map before the fold, and `post` an elementwise map of
  the running scalar `reduced(0)`;
- the axis is `rank - 1` (v1).

The delta rule fails each of those:

1. **The carried state is a matrix, not a scalar per lane.** Each step reads all
   of `S` (`Sᵀk`) to produce the correction, so a lane cannot fold
   independently.
2. **The combine is not one of the four monoids.** The step is an affine map on
   `S`: `S ↦ S·A_t + B_t`, with `A_t = exp(g_t)(I − β_t k_t k_tᵀ)`. Affine maps
   do compose associatively, which is what makes the chunked form possible. But
   composing two of them is a 128 × 128 matrix product, not an elementwise
   combine.
3. **The output is a contraction of the state**, not an elementwise map of it.

So extending `Scan` with another `ReduceOp` would not reach it. Expressing it
directly would take a new access kind: a matrix-state recurrence with a
contraction in its step.

## 4. What does compose from existing IR

### 4.1 Decode (one token per step), the M6 bring-up path

Each step is the five lines of §2 over a resident `S`. The runtime (fuel) owns `S`
and the per-token loop, as it does for a KV cache:

| step | IR |
|---|---|
| `S * exp(g_t)` | elementwise; `g_t` broadcast over `S` via strides |
| `kv = Sᵀ k_t` | `Contraction` (it has a `Batch` axis role for the heads), or a `RowReduce` |
| `d = beta_t (v_t − kv)` | elementwise |
| `S + k_t ⊗ d` | elementwise; the outer product comes from two broadcast reads |
| `o_t = Sᵀ q_t` | `Contraction` |

The state is f32, which is also what the P40 wants (§1 of the joint plan).

### 4.2 Prefill (chunked form, chunk 64)

| piece | IR |
|---|---|
| `cum_decay = cumsum(g)` within a chunk | **`Scan` with `Sum`**: this is the one place `Scan` fits |
| pairwise decay `exp(c_i − c_j)` with a strictly-upper `-inf` mask | elementwise: broadcast reads, plus `Coord`/`Select` for the mask |
| `k_beta·kᵀ`, `q·kᵀ`, `ut·v`, `k·S` | batched `Contraction` |
| **unit-lower-triangular solve** (64 × 64, `solve_triangular`) | **no IR.** Each row depends on every earlier row. The reference has a loop fallback with the same dependency. |
| the loop over chunks carrying `S` | sequential; the runtime drives a sequence of `Contraction`s |

**Fallback:** run prefill as the §4.1 decode step, token by token. It is correct,
and it is slower for long prompts. For bring-up that is acceptable.

### 4.3 The causal conv1d (kernel 4) before the recurrence

`Access::Window` defers it by name: *"causal_conv1d (needs a weight operand →
windowed contraction)"* (`ir.rs`, the `Window` docs). As one elementwise op over
shifted reads it needs 4 shifted inputs, 4 per-tap weights and 1 output: **9
operands, over `MAX_OPERANDS = 8`.** Two chained elementwise ops of 2 taps each
fit. That is composable, but untested.

## 5. Not assessed

- No test plans these shapes. The next step would be a test in the shape of
  `crates/unpopped/tests/qwen3_dense_ops_plan_for_sm61.rs` covering §4.1, which
  should come before any claim that decode "works".
- Whether baracuda's emitter handles `Contraction` with a `Batch` axis at these
  shapes. That is baracuda's side.
- The full-attention layers of qwen3_5 differ from Qwen3's: `attn_output_gate`,
  `partial_rotary_factor: 0.25` and interleaved M-RoPE. Not assessed here.
- The gated RMSNorm (`Qwen3_5RMSNormGated`) is likely a `RowReduce` with a gate
  input, but this is not checked.
- The MTP head.

## 6. Options, for when M6 is scheduled

1. **Compose (no IR change):** decode via §4.1, and prefill via the token-by-token
   fallback, or via chunks with the triangular solve done by a loop of small ops.
   This is enough for correctness on the 27B model.
2. **A triangular-solve intent:** this closes the one prefill hole for the
   chunked form.
3. **A matrix-state recurrence access:** one fused kernel per layer, for
   performance. This is design work. It belongs with the idiom-lifting intents
   (`docs/idiom-lifting-design.md` §3), and should start only after option 1
   measures a need.
