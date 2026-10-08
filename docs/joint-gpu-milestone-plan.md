# Joint milestone plan: a real model, bulletproof on the P40 (sm_61) and the RTX 4070 (sm_89)

PM task to the Unpopped lane, 2026-10-07: coordinate one plan across baracuda,
fuel, lightbulb and Unpopped. **This is a plan, not a status report.** Every fact
below names the ref it was read at; re-read it before acting on it.

## 0. The goal, as the PM relayed it

CireSnave's definition of *bulletproof*, for the P40 (sm_61) and the RTX 4070 (sm_89):

1. A real model runs end to end on both cards.
2. Its results match a CPU reference within tolerance, on the real cards.
3. There is a speed number per card.
4. All of the above also holds through the lightbulb API.

Qwen3.x comes first. A Qwen 27B-class model across cards comes later.

> The PM relayed this as a summary, not a verbatim quote. Before M0 starts, the
> PM should pin the exact wording, the Qwen3.x size, and the 27B-class model name
> (see §5, Q1).

## 1. Where each project stands (measured 2026-10-07)

> **Read every row as "the code exists", not "it works".** These rows come from reading code and from each lane's answers. The first real run (fuel M0, Qwen3-0.6B) refuted a "works today" claim. Treat every other capability below as unmeasured until M0, M1 or M4 shows it on a real checkpoint.

| repo | ref read | relevant fact |
|---|---|---|
| unpopped | `29e8f68` | `cuda:sm61` parses and resolves to a sourced capability row (`capability.rs`, #34). Keys take `TargetId`, so an sm_61 key can be formed. **`plan.rs` never reads the target**: 0 hits for `TargetId`, while the same grep finds it in `jit.rs:34,72`. No plan decision is per-sm today. |
| baracuda | `20139e5` | Hand-written kernels cover every dense Qwen3 op (GEMM, RMSNorm, RoPE incl. YaRN, SwiGLU, softmax, SDPA/flash with GQA, embedding, add), plus MoE. `baracuda-kernels-sys/build.rs:8-16` builds only `sm80`/`sm89`/`sm90a`. **No sm_61 target exists**, and PTX forward-compatibility never runs older. The README (437-455) records the sm_61-through-latest ruling and says sm_61 is in-progress Phase B, via parse → Unpopped → emit, not hand-written SIMT. NVRTC `arch_flag` already accepts `cuda:sm61` (`baracuda-cuda-emit/src/nvrtc.rs:150-170`), but only construction is tested. There is no CUDA CI runner, and no Qwen run or transformer-block run exists. |
| fuel | `6dfc824` | Qwen3 models exist (`fuel-transformers/src/models/lazy_qwen3.rs`, `lazy_quantized_qwen3.rs`, `lazy_qwen3_moe.rs`): the code exists, but see below for whether a real checkpoint loads. CLIs are in `fuel-lazy-examples/src/bin/` and `fuel-examples/examples/quantized-qwen3`. Backends: CPU, CUDA (via baracuda `alpha.81`), Vulkan, Metal. A ULP/relative/absolute compare harness exists (`fuel-dispatch/src/fkc/verify/ulp.rs`), but it is per-op, not per-model. At `6dfc824` the fuel lane reported **no model-level CPU-vs-CUDA parity test for Qwen3** and **no tok/s harness** (`quantized-qwen3/main.rs` times model load only). **M0 added both** at `de4d370`: a model-level compare harness against a CPU fixture, and a CPU speed harness (see M0). Both harnesses have run on CUDA (M1: compare at fuel `17236d20`, timed run via fuel#321, `29a9ae0b`). `fuel-parallel` is a leaf crate that no model crate consumes. **A real Qwen3 checkpoint did not load** (fuel M0 run, 2026-10-07). Qwen3-0.6B GGUF failed with `attn_q.weight has 2097152 elements, expected 1048576`, because Qwen3 decouples `head_dim` from `hidden_size / num_heads` (0.6B: 16 × 128 = 2048, not 1024). This was fuel GAP-279 (forward guard) plus a loader bug in the quantized path, and it is **fixed for Qwen3 dense by fuel#317** (merged 2026-10-07T08:31Z, `de4d370`): `unsloth/Qwen3-0.6B-GGUF` `Q4_K_M` loads on CPU. Multi-step decode is validated against a fixture regenerated with context (fuel#320, `17236d20`; see M0), and the CUDA run of M1's compare passed (see M1). Qwen3-MoE and the 27B class are not shown yet. An earlier report that fuel runs Qwen3 end to end held only for synthetic configs. Board #106 Step A merged (#310, `c02da348`). |
| lightbulb | `fd44479` | It serves an OpenAI-compatible `POST /v1/chat/completions` and `/v1/completions`. Fuel is behind `--features fuel-engine`; the default engine is candlelight. Fuel is a **git pin `d90b481`**, not fuel's main. A Qwen3 GGUF loader exists (`src/model_fuel/loader_gguf_qwen3.rs`), but loading a real checkpoint through it is unmeasured; it runs on fuel, whose real-checkpoint load fails today (fuel row). **The CUDA device index is hardcoded to 0** in both engines (`src/model_fuel/device.rs:24`, `src/model/parallel_model_manager.rs:265`). `src/multi_gpu/` is wired into nothing. The API has no timing field. Auth is off without `DATABASE_URL`, with a default bind of `0.0.0.0:8080`. Security item 2 (pending) will refuse a non-loopback bind without real auth. End-to-end tests use TinyLlama and are `#[ignore]`. |

### Hardware

| card | sm | where | who can reach it |
|---|---|---|---|
| RTX 4070 Laptop | 89 | PM laptop (this box) | every lane on this box; baracuda verifies there today |
| RTX 4060, 8 GB | 89 | LAN desktop 192.168.4.23 | **no lane today** (baracuda reports no access) |
| Radeon VII | (AMD) | LAN desktop | irrelevant to CUDA; relevant later to Vulkan/Slang |
| P40, 24 GB | 61 | not yet arrived; host will be the LAN desktop (PM, 2026-10-07) | — (Q3) |

Two hardware facts shape the sm_61 path:

- **The P40 runs fp16 math at about 1/64 the fp32 rate** (GP102; the P100 is the
  Pascal part with fast fp16). On sm_61, f16 is a *storage* type and f32 is the
  *compute* type. int8 `__dp4a` is the fast path.
- **CUDA 13.x cannot target sm_61.** 13.3 starts at `compute_75`. The 12.9.2
  toolkit (user-local here, NVRTC verified to compile sm_61) can. The P40 box
  needs a 12.x runtime/NVRTC and a driver that still supports Pascal. Which
  driver branch that is gets confirmed from NVIDIA's release notes when the box
  is set up, not from memory.

## 2. Ownership (PM ruling 2026-10-07, option C)

- **Unpopped:** capability rows (sm_61 included) and the per-sm **plan** decisions.
- **baracuda:** the CUDA **spelling** per sm (`baracuda-cuda-emit`), the hand-written kernels, and their build targets.
- **fuel:** the model, runtime, device selection, CPU reference, compare and bench harnesses.
- **lightbulb:** serving, device configuration, the API-level test.
- **CireSnave:** hardware, LAN access, and the exact model choices.

## 3. Milestones

Each milestone ends in an observable clearing event, named in its row.

### M0: the yardstick (CPU only, no GPU needed) — owner: **fuel**

- Pick the model: the smallest Qwen3.x that is still "real" (Q1). Use the same checkpoint and format (safetensors or GGUF, and which quant) everywhere downstream.
- **CPU reference run:** fuel CPU backend, fixed prompt set, greedy decoding. Store the per-step logits and token ids as a fixture.
- **Compare harness, model level** (copy the pattern of `fuel-model-llama/tests/paged_decode_parity.rs`): for a candidate device, the same prompts give (a) logits within a stated tolerance per dtype, and (b) an identical greedy token sequence for the first N tokens. Tolerances are written down *before* any GPU run, so the GPU cannot set its own bar.
- **Speed harness:** prefill tok/s and decode tok/s, with warm-up, at a fixed prompt length and batch size, plus the card name, driver and toolkit in the record.
- Controls: the harness must go red on a perturbed logit fixture and on a one-token swap.
- **Clears when:** a fuel PR merges the fixture, the compare harness and the speed harness, with both controls shown red.
- **Merged: fuel#317, `de4d370`, 2026-10-07T08:31Z.** Checkpoint: `unsloth/Qwen3-0.6B-GGUF`, `Q4_K_M`, so the **quantized tier** of §5.1 applies (`2e-2 · max|logit_ref|`, against this same-quantization CPU reference). The fixture is `fuel-transformers/tests/fixtures/qwen3_0_6b_cpu_reference.json`: 2 prompts × 32 greedy steps, the top-16 logits per step, and `max|logit|` over the full vocabulary. **The fixture was regenerated by fuel#320 (merged 2026-10-07T20:52Z, `17236d20`), and every step of it is now valid.** The fixture at `de4d370` was valid for step 0 only. Its generator (`run_greedy`) fed each decode step one token through the stateless `QuantizedQwen3Model::forward(&self, …)`, with no KV cache, so steps 1–31 had neither the prompt nor earlier tokens in context (found by fuel, verified by the PM at fuel `f754bd1`). From `17236d20`, `run_greedy` re-feeds the whole growing sequence at every step and takes the last position's logits. A CPU run of the KV-cache decode path matches the new fixture exactly (max|Δ| = 0, 0/64 tokens mismatched, fuel's report). The harness, both controls and the `#[ignore]`d speed harness are in `fuel-transformers/tests/qwen3_cpu_yardstick.rs`.
- **Two gaps against §5.1 found at `de4d370`; fixed by fuel#318, merged 2026-10-07T10:28Z as `28b8a35` (squash of `1319633`, the head Unpopped reviewed):**
  1. **The calibration measured determinism, not reduction-order spread** (the regenerated fixture at `17236d20` records the spread as `0.0` again). The original rerun used the same process, thread count and prompt. #318 reruns at `RAYON_NUM_THREADS=1` against the default, and the spread is still `0.0`. **That is structural, not a harness defect.** fuel's loader keeps only `Q4_0` Linear weights quantized and dequantizes every other GGML dtype to F32 at load (`lazy_quantized_qwen3.rs:253-254, 342-343` at fuel `17236d20`). This checkpoint's Linear weights are `Q4_K`/`Q6_K`, so they become F32, and its matmuls are plain F32 `Op::MatMul`. On the CPU that is `byte_kernels::matmul_f32_capacity` (`fuel-cpu-backend/src/byte_kernels.rs:4700`, registered for `(MatMul, F32, Cpu)` through `matmul_f32_cpu_wrapper`, `fuel-dispatch/src/dispatch.rs:5364`). It is a sequential triple loop with no threads, so no thread count can change a sum's order. (Corrected 2026-10-08. This line used to say the matmuls were `fuel-quantized` `k_quants::matmul`, a serial `vec_dot` per output; no measurement had checked that, and fuel's loader contradicts it. fuel confirmed the corrected path, and Unpopped read the kernel at `17236d20`.) §5.1 is amended to a probe that can see a difference.
  2. **A real deviation could pass the compare unchecked.** A reference top-16 id was checked only if it was also in the *candidate's* top-16. **Fixed in #318:** a live candidate is looked up in its full logit row (`CandidateLogits::Full`). The stored-fixture path treats a missing id as `-∞`, which fails. Both controls exist: an id pushed out goes red, and a near-cutoff reorder stays green.

  M1 numbers taken against a fuel ref before `17236d20` (fuel#320, where the regenerated fixture landed) are provisional and are not recorded here.

### M1: sm_89 end to end through fuel — owners: **fuel**, then **baracuda** for kernel faults

- ~~**Depends on fuel GAP-279** and the quantized-loader fix~~: **cleared** for the M0 checkpoint by fuel#317 (`de4d370`). Fuel's first runs on 2026-10-07 compared against the context-free fixture, so their decode-step divergences measured the fixture, not the engine, and are not recorded here.
- **Compare: passed on the RTX 4070 at fuel `17236d20`** (fuel's report, 2026-10-07T21:10Z; test `m1_cuda_matches_cpu_reference` in `qwen3_cpu_yardstick.rs`, device `cuda:0`). Both prompts × 32 steps: 0/64 greedy tokens differ, no step fails the bound, max|Δ| = 9.16e-5.
  - **Ratio, upper bound: 2.2e-4**, far under §5.1's 0.25. It is computed as the global max|Δ| over the smallest per-step bound in the fixture (`2e-2` × 20.72 = 0.414), so it is an upper bound on the per-step maximum.
  - **What max|Δ| covers:** the reference's top-16 ids at each step, not the full vocabulary §5.1 names, because the fixture stores only the top 16. A full-vocabulary logit is checked only through those ids.
  - **Not teacher-forced:** the candidate feeds its own greedy tokens. With 0/64 mismatches, the two are the same here.
  - **Which ops ran on CUDA: matmul is proven on CUDA at fuel `17236d20`.** Fuel faulted the `(MatMul, F32, Cuda)` dispatch entry (`gemm_dense::matmul_f32`), and the test failed at the first matmul of prefill. Two earlier signs could not show this. GPU memory peaked at 157 MB, against 397 MB for the weights file alone (about 2.4 GB once dequantized to F32), and a test that needs a device fails without one whatever runs where. **Per-category sweep (fuel's report, 2026-10-08T01:39Z, ref `17236d20`, bisected fault injection):** embedding lookup, RMSNorm, binary mul, softmax, SiLU and dense matmul each fail the test when faulted, so each executes on CUDA in this model's forward pass. **RoPE's dedicated kernel (`rope_f32`) is never hit:** Qwen3's RoPE here runs as generic elementwise mul/add (also CUDA-confirmed), not fuel's fused RoPE kernel. **Every MatMul closed by logging (fuel finding `fuel-M1-matmul-category-close`, same ref):** a logging pass-through on `gemm_dense::matmul_f32` over a full prefill and decode shows every distinct MatMul shape went through that one CUDA entry: the q/k/v/o projections, FFN gate/up/down, attention score QK^T (batch [1,16]/[1,8], GQA n_rep=2, m=1 n=seq k=128), attention value (m=1 n=128 k=seq), and lm_head (n=151936 k=1024, once per step). No MatMul used another registration. **Add is inferred** (shared code with mul), not observed.
- **Timed CUDA run (fuel#321, `29a9ae0b`, warm, split):** prefill 21 tokens in 4.812 s (4.36 tok/s); first decode call 4.360 s (one-time graph/session setup plus 1 token, not a rate); steady-state decode 30 tokens in 10.882 s (2.76 tok/s). RTX 4070 Laptop GPU, driver 616.92. Toolkit: nvcc release 13.3 (V13.3.33); CUDA UMD 13.4 per the `nvidia-smi` header. The slow steady-state decode is suspected to come from a per-call weight upload; that is **not measured** (no H2D trace). **M1 does not mean fast.**
- **M1 met at fuel `17236d20`:** 0/64 greedy-token mismatches, max|Δ| 9.16e-5 (ratio 2.2e-4 against the smallest per-step bound 0.414, an upper bound). Compute on CUDA is proven by fault injection or per-op logging for embedding, RMSNorm, mul, softmax, SiLU and every MatMul; add is inferred (shared code with mul); RoPE runs as elementwise mul/add, because the dedicated rope kernel is not on the path. The reference is fuel's own corrected CPU run, not an independent implementation. Performance: prefill 4.36 tok/s, steady-state decode 2.76 tok/s (fuel#321; RTX 4070 Laptop, driver 616.92).
- Run M0's model on the RTX 4070 through the fuel CUDA backend, then the M0 compare and speed harnesses.
- Any kernel mismatch goes to baracuda with the op name and the inputs.
- Do the 4060 on the LAN desktop too, once someone can reach it (Q3). Two cards of the same sm make a cheap control on driver and box effects.
- **Clears when:** the compare passes on the 4070, and the tok/s number is recorded with its ref.

### M2: sm_89 through lightbulb — owner: **lightbulb**

- Move lightbulb's fuel dependency off the git pin (`d90b481`) to a **crates.io version** at or after the ref M1 cleared at. CireSnave's rule of 2026-10-07 (CLAUDE.md §9, *"Sources"*): cross-repo dependencies are crates.io versions, with no new `git =` dependencies. The known obstacle: crates.io `fuel-core` is an unrelated project (lightbulb `Cargo.toml:233-235`), so fuel must publish under names that are free. That is fuel's and lightbulb's to settle, through the PM.
- Make the CUDA device index configurable per deployment (it is hardcoded to 0 today). This is not needed for one card per box, but it is needed the moment two cards share a box (M6).
- API-level test: run M0's prompts through `POST /v1/chat/completions` with greedy parameters. The tokens must match the CPU fixture. tok/s comes from the harness, wall-clock over `usage.completion_tokens`, since the API has no timing field.
- If the test crosses machines, security item 2 needs a Postgres-backed key. Loopback on the GPU box needs none.
- **Clears when:** that test passes on the 4070, run by the harness, not by hand.

### M3: sm_61 build path (no P40 needed yet) — owners: **Unpopped** (plan), **baracuda** (emission, build)

Split along the ruling:

- **U1, Unpopped: per-sm plan decisions** (design approved by the PM 2026-10-07). An audit of `plan.rs` found one decision that must differ between sm_61 and sm_89 in Unpopped's domain: **16-bit float arithmetic**.
  - NVIDIA's 12.9.1 throughput table gives sm_61 2 fp16 results per clock per SM against 128 for fp32. sm_89 does 128 for both.
  - U1 adds the two rates to `TargetCapabilities`, and adds `KernelPlan::half_arith() -> HalfArith{Native,ViaF32}`, read from `key.target`. It is additive, and no golden moves: ViaF32 only when both rates are sourced and fp16 < fp32.
  - baracuda's packed `__h*2` path (`cuda.rs:427`) honours ViaF32 in B1. It is bit-identical to the scalar float path by design, so this changes speed, not numerics.
  - **Not U1's:** vector width comes from the key, and `max_vector_bytes` is 16 in every row. The plan sizes no shared memory. `mma` vs SIMT for `Contraction` is a feature fact owned by baracuda-cuda-vocab (`idiom-lifting-design` §6). The cuBLAS compute type is baracuda's. Both are in §6.
- **U2, Unpopped: the Qwen3 op set as IR, planned for sm_61.** This was corrected 2026-10-07 by baracuda's `docs/sm61-parse-emit-gap-analysis.md`. The sm_61 route is IR → `baracuda-cuda-emit` (plain scalar CUDA, with no tensor-core, async-copy or dp4a intrinsics) → NVRTC. It is not a parse of the hand-written kernels, so parse coverage is not the question. U2 is a test showing that every Qwen3 dense op Unpopped's IR can express plans and generates for `cuda:sm61` on the same schedule as `cuda:sm89`, differing only in `half_arith`: residual add and SwiGLU (f32, f16), RMSNorm, two-stage softmax, a RoPE pair lane, embedding, and matmul. **Attention has no IR node** (`Access::Attention`, `idiom-lifting-design` §11). For bring-up it composes from `Contraction` + `RowReduce` softmax + `Contraction`. GEMM can also use cuBLAS (`DenseGemmPlan`).
- **B1, baracuda: sm_61 emission and build.** An NVRTC sm_61 compile of every kernel M1 needs, through the 12.9 toolchain, and a decision on the hand-written kernels: an `sm61` build feature, or the emit route only. baracuda's README records the emit route as the plan.
- Tests run on this box: compile-only for sm_61. The numerics of the emitted kernels can be checked on the 4070 by forcing the sm_61 *plan* and running it on sm_89. That is the same-GPU, two-path control from `idiom-lifting-design` §10, applied across plans.
- **Clears when:** every op in M0's model has an sm_61 artifact that NVRTC accepts, and the sm_61-plan-on-sm_89 run passes M0's compare.

### M4: sm_61 on the real P40 — owners: **fuel**, **baracuda**; blocked on **hardware**

- Prerequisite: a P40 installed and reachable by a lane (Q2, Q3), with a 12.x runtime and a Pascal-capable driver.
- Run M0's compare and speed harnesses through the fuel CUDA backend.
- **Clears when:** the compare passes on the P40, and the tok/s number is recorded.

### M5: sm_61 through lightbulb — owner: **lightbulb**

- The M2 test, pointed at the P40 box. **Clears when:** it passes there.

### M6: Qwen 27B-class across cards — owners: **fuel** (split), **lightbulb** (serving); later

- **The model is a different architecture from the bring-up model.** `PORTFOLIO-ROADMAP.md` (local-model plan, agreed with CireSnave 2026-09-27, lines 562-563) names Qwen/Qwen3.8-27B. Its arch is `qwen3_5`:
  - hybrid, with 48 linear-attention layers and 16 full-attention layers;
  - a native MTP head;
  - hidden size 5120 and 64 layers;
  - run quantized on 3 P40s.
- That makes M6 a larger fuel and lightbulb item than the 0.6B bring-up:
  - a `qwen3_5` model and loader (lightbulb refuses `qwen35` GGUF today);
  - linear-attention kernels, a recurrent scan that M0–M5 never exercise. **`Scan` cannot express it** (a scalar monoid per lane against a 128 × 128 matrix state per head). But decode composes from existing IR, with fuel holding the state, and chunked prefill composes except a 64 × 64 triangular solve, which has a token-by-token fallback. See `docs/qwen3_5-linear-attention-assessment.md`; nothing in it has been run yet.
  - MTP decoding.
- Memory arithmetic, for weights only, without KV cache:
  - 27B at f16 is about 54 GB, so three P40s;
  - at int8, about 27 GB, so two P40s;
  - at 4-bit, about 15 GB, so one P40.
  The quant choice decides whether "across cards" is needed at all (Q1).
- Today nothing in fuel depends on `fuel-parallel`, and lightbulb's `src/multi_gpu/` is unwired. Layer or pipeline split across processes or devices is new integration work, and it needs M2's device-index configuration.
- Mixed sm_61 + sm_89 in one model means two plans in one run. That is exactly what U1's per-target plans make possible.
- **Clears when:** a split run passes M0's compare (re-baselined for the larger model) and is served by lightbulb.

## 4. Order and parallelism

```
M0 ──► M1 ──► M2
 │      │
 │      └──► M3 (U1, U2, B1 run in parallel; start now, needs no P40) ──► M4 ──► M5
 │                                                                         ▲
 └── hardware: P40 arrival + LAN access ───────────────────────────────────┘
M2 + M4 ──► M6
```

- **Start now, in parallel:** M0 (fuel), U1 and U2 (Unpopped), B1 (baracuda), and lightbulb's device-index work.
- The critical path is M0, then M1, then M4. M4 waits on hardware no lane controls.

## 5. Decisions (PM, 2026-10-07) and what stays open

- **Assignments:** M0 → fuel. U1 and U2 → Unpopped (U1 starts now). B1 → baracuda, after its registry-build PR. Device-index work → lightbulb.
- **Q1, models. Decided:** the first model is the smallest Qwen3 *dense* model that exercises the real op set (bring-up), with the 27B-class model later. Fuel picks the checkpoint in M0 and names it in its PR.
  - **Added to M6:** lightbulb refuses `qwen35` GGUF (`src/gguf/mod.rs:1034`). If the 27B-class model is Qwen3.5, M6 needs that refusal replaced by a real loader first. Owner: lightbulb, with fuel for the model.
- **Q2, toolchain and host.** Decided: CUDA 12.9.2 at `C:\Users\cires\AppData\Local\cuda-toolkits\12.9.2` (beside 13.3) is the sm_61 toolchain, and the P40 host is the LAN desktop. Open, and CireSnave's, tracked by the PM: P40 arrival and cooling.
- **Q3, LAN access. Open,** on the PM's board. Until it's answered, lanes reach the desktop only through its HTTP endpoints. M4 cannot run before this is answered.
- **Q4, tolerance. Decided:** a per-dtype logit bound plus greedy-token identity over a fixed prompt set. The numeric bounds are proposed below.

### 5.1 Proposed tolerance bounds (Q4)

Compare **teacher-forced**: feed both runs the reference's tokens, so one near-tie cannot cascade into a whole divergent continuation. At each step, over the full vocabulary:

| weights / compute | logit bound, per step | basis |
|---|---|---|
| f32 / f32 | `max|Δ| ≤ 1e-3 · max|logit_ref|` | reduction order is the only difference |
| f16 or bf16 storage / f32 accumulate | `max|Δ| ≤ 2e-2 · max|logit_ref|` | rounding of intermediate activations to 16 bits between layers |
| quantized (int8, 4-bit) | the same as f16, **against a CPU reference that uses the same quantized weights** | quantization error belongs to the model, not the device; comparing with an unquantized reference measures the wrong thing |

**Greedy-token identity:** for every prompt, the first 32 greedily generated tokens are identical to the reference's. A divergence is excused only at a step where the reference's top-1/top-2 margin is below that step's logit bound (a real tie). Excused steps are reported by count, and more than 1 per prompt fails.

**Calibrate before trusting the numbers.** In M0, run the CPU reference twice with a different reduction order (or thread count). The bound must be at least 4× the spread that measures, or it is too tight to be stable; it must also go red on M0's perturbed-logit control, or it is too loose to mean anything. If either check fails, M0 replaces these numbers with measured ones, and the PR says so.

**Amended 2026-10-07, corrected 2026-10-08 (bounds unchanged): the CPU rerun cannot see a difference for this model.** Its matmuls run in fuel's single-threaded F32 CPU kernel (see M0, gap 1), so a CPU-vs-CPU rerun at any thread count gives a spread of exactly 0, and "bound ≥ 4× spread" passes for every bound. The first run that changes the reduction order is the GPU's F32 GEMM (`gemm_dense::matmul_f32`; M1's nonzero max|Δ| shows the order differs). The 2026-10-07 text gave a different reason: a quantized `vec_dot`, serial over K, with the GPU splitting K through MMVQ. That reason was wrong for this checkpoint, which never runs a quantized matmul. The conclusion (no CPU rerun probe; M1 measures the ratio) stands. Only its mechanism changed. So:

- **M1 records the ratio `max|Δ| / bound`** at every step, and reports its maximum over all steps and prompts, next to pass/fail.
- **At or below 0.25, the bound has the required 4× margin. Above 0.25, it is too tight.** This is the same 4× rule, applied to the first measured order change.
- **Ratio above 0.25: investigate the kernel, or loosen the bound, never silently.** A loosened bound goes into this section with the measured ratio and the reason (PM ruling on #47, 2026-10-07).
- **The ratio is an upper-bound reading, and it is labelled that way wherever it is quoted.** It measures the device under test against the reference, not independent noise, so a kernel error inflates it just as rounding does. Fuel's first M1 run prints only the global `max|Δ|`. It reports `global max|Δ| / (smallest per-step bound)`, which is an upper bound on the per-step maximum, and says so (PM ruling on #47, 2026-10-07).
- **Optional CPU probe:** an F32 matmul with a different K order, for example fuel's threaded `gemm` entry in `fuel-cpu-backend` `ops.rs`, if this model's dispatch can be pointed at it. Fuel says that entry is not on this model's path today. (The 2026-10-07 text proposed scalar against SIMD `vec_dot`. That probe changes nothing here, because no `vec_dot` runs.)
- **The "too loose" side is unchanged:** the perturbed-logit control must still go red.

On the P40, f16 compute is slow enough (§1) that every kernel computes in f32 there. So the f16-storage row applies to sm_61 unchanged.

## 6. Blockers

| blocker | blocks | owner |
|---|---|---|
| P40 hardware not here | M4, M5, M6 | CireSnave |
| No lane can run code on the LAN desktop (the P40 host) | the 4060 control in M1; M4, M5 | PM board (Q3) |
| lightbulb refuses `qwen35` GGUF | M6, if the 27B-class model is Qwen3.5 | lightbulb |
| ~~`plan.rs` is arch-blind (no 16-bit arithmetic decision per sm)~~ **Cleared by U1 (#40)** | — | Unpopped |
| f16 GEMM on sm_61 must use cuBLAS **f32 compute** (`DenseGemmPlan`); fp16 compute runs at 2 results per clock per SM | M3, M4 speed | baracuda (B1) |
| `mma` needs sm_70+: `Contraction` on sm_61 must lower to SIMT or dp4a. This is a feature fact in baracuda-cuda-vocab | M3 | baracuda (B1) |
| Whether `cuda_bf16.h` packed bf16 ops compile under NVRTC sm_61 | M3, if the model is bf16 | baracuda (B1) |
| No sm_61 build or emit path proven | M3, M4 | baracuda (B1) |
| lightbulb device index hardcoded to 0; fuel pin stale | M2 (pin), M6 (index) | lightbulb |
| ~~A real Qwen3 checkpoint does not load (fuel GAP-279)~~ **Cleared for Qwen3-0.6B `Q4_K_M` by fuel#317 (`de4d370`)**; MoE and 27B not shown | — | fuel |
| ~~M0's compare can pass a real deviation~~ **Fixed by fuel#318 (`28b8a35`).** Its calibration cannot measure a reduction-order spread for this model (the CPU F32 matmul is single-threaded), so M1 must report `max|Δ|/bound` instead (§5.1, amended) | M1, M4 (trust in the compare verdict) | fuel |
| ~~M0's fixture is valid for step 0 only~~ **Regenerated with context by fuel#320 (`17236d20`).** | — | fuel |
| No CUDA CI runner anywhere | regression protection after each milestone | open: a self-hosted runner needs CireSnave |

## 7. After this plan: AMD and Intel through Vulkan

Sequenced **after** M0–M5, in CireSnave's order, as relayed by the PM on 2026-10-07.
The fuel lane's finding at fuel `6dfc8248` (kernel table vs shader sources):

- 66 of 167 authored kernels ship embedded SPIR-V (39.5%).
- Missing whole: `layer_norm`, `rms_norm`, `flash_attention` (and its backward), `scatter_add`, `concat`, `index_select`, the optimized matmul, and most non-f32 casts.
- AMD and NVIDIA through Vulkan are live-verified on real hardware; Intel has no evidence.

So AMD and Intel LLM support is gated on those missing kernels, not on architecture. Fuel's device auto-pick (`fuel-examples/src/lib.rs:71-91`) also does not include Vulkan.
