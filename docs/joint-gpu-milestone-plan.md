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

| repo | ref read | relevant fact |
|---|---|---|
| unpopped | `29e8f68` | `cuda:sm61` parses and resolves to a sourced capability row (`capability.rs`, #34). Keys take `TargetId`, so an sm_61 key can be formed. **`plan.rs` never reads the target**: 0 hits for `TargetId`, while the same grep finds it in `jit.rs:34,72`. No plan decision is per-sm today. |
| baracuda | `20139e5` | Hand-written kernels cover every dense Qwen3 op (GEMM, RMSNorm, RoPE incl. YaRN, SwiGLU, softmax, SDPA/flash with GQA, embedding, add), plus MoE. `baracuda-kernels-sys/build.rs:8-16` builds only `sm80`/`sm89`/`sm90a`. **No sm_61 target exists**, and PTX forward-compatibility never runs older. The README (437-455) records the sm_61-through-latest ruling and says sm_61 is in-progress Phase B, via parse → Unpopped → emit, not hand-written SIMT. NVRTC `arch_flag` already accepts `cuda:sm61` (`baracuda-cuda-emit/src/nvrtc.rs:150-170`), but only construction is tested. There is no CUDA CI runner, and no Qwen run or transformer-block run exists. |
| fuel | `6dfc824` | Qwen3 models exist (`fuel-transformers/src/models/lazy_qwen3.rs`, `lazy_quantized_qwen3.rs`, `lazy_qwen3_moe.rs`), with CLIs in `fuel-lazy-examples/src/bin/` and `fuel-examples/examples/quantized-qwen3`. Backends: CPU, CUDA (via baracuda `alpha.81`), Vulkan, Metal. A ULP/relative/absolute compare harness exists (`fuel-dispatch/src/fkc/verify/ulp.rs`), but it is per-op, not per-model. The fuel lane reports **no model-level CPU-vs-CUDA parity test for Qwen3**; the pattern to copy is `fuel-model-llama/tests/paged_decode_parity.rs`. **No tok/s harness:** `quantized-qwen3/main.rs` times model load only, and `ROADMAP.md:869-880` lists an end-to-end tok/s harness as future work. `fuel-parallel` is a leaf crate that no model crate consumes. The fuel lane reports Qwen3 running end to end on CPU and CUDA today. That is a self-report; M0 and M1 are where it gets checked. Board #106 Step A merged (#310, `c02da348`). |
| lightbulb | `fd44479` | It serves an OpenAI-compatible `POST /v1/chat/completions` and `/v1/completions`. Fuel is behind `--features fuel-engine`; the default engine is candlelight. Fuel is a **git pin `d90b481`**, not fuel's main. Qwen3 GGUF loads (`src/model_fuel/loader_gguf_qwen3.rs`). **The CUDA device index is hardcoded to 0** in both engines (`src/model_fuel/device.rs:24`, `src/model/parallel_model_manager.rs:265`). `src/multi_gpu/` is wired into nothing. The API has no timing field. Auth is off without `DATABASE_URL`, with a default bind of `0.0.0.0:8080`. Security item 2 (pending) will refuse a non-loopback bind without real auth. End-to-end tests use TinyLlama and are `#[ignore]`. |

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

### M1: sm_89 end to end through fuel — owners: **fuel**, then **baracuda** for kernel faults

- Run M0's model on the RTX 4070 through the fuel CUDA backend, then the M0 compare and speed harnesses.
- Any kernel mismatch goes to baracuda with the op name and the inputs.
- Do the 4060 on the LAN desktop too, once someone can reach it (Q3). Two cards of the same sm make a cheap control on driver and box effects.
- **Clears when:** the compare passes on the 4070, and the tok/s number is recorded with its ref.

### M2: sm_89 through lightbulb — owner: **lightbulb**

- Bump the fuel git pin from `d90b481` to the ref M1 cleared at.
- Make the CUDA device index configurable per deployment (it is hardcoded to 0 today). This is not needed for one card per box, but it is needed the moment two cards share a box (M6).
- API-level test: run M0's prompts through `POST /v1/chat/completions` with greedy parameters. The tokens must match the CPU fixture. tok/s comes from the harness, wall-clock over `usage.completion_tokens`, since the API has no timing field.
- If the test crosses machines, security item 2 needs a Postgres-backed key. Loopback on the GPU box needs none.
- **Clears when:** that test passes on the 4070, run by the harness, not by hand.

### M3: sm_61 build path (no P40 needed yet) — owners: **Unpopped** (plan), **baracuda** (emission, build)

Split along the ruling:

- **U1, Unpopped: per-sm plan decisions.** Make `plan.rs` read the target's capabilities instead of assuming sm_80+. For sm_61:
  - f16 operands compute in f32;
  - no `mma` lowering for `Contraction` (SIMT or dp4a);
  - vector widths and shared-memory limits come from the 6.1 capability row.
  Each decision gets a test that names `cuda:sm61` and `cuda:sm89` and shows they differ where they must.
- **U2, Unpopped: conversion coverage for Qwen3's ops.** baracuda's recorded sm_61 route is parse → Unpopped IR → emit. `convert.rs` lifts elementwise, reduction and scan today. Inventory which of RMSNorm, softmax, RoPE, SwiGLU, embedding (gather) and attention lift today, and close the gaps in priority order. GEMM can use cuBLAS (`DenseGemmPlan`), which supports sm_61 under CUDA 12.x, so it is not on this list unless the cuBLAS route fails.
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

On the P40, f16 compute is slow enough (§1) that every kernel computes in f32 there. So the f16-storage row applies to sm_61 unchanged.

## 6. Blockers

| blocker | blocks | owner |
|---|---|---|
| P40 hardware not here | M4, M5, M6 | CireSnave |
| No lane can run code on the LAN desktop (the P40 host) | the 4060 control in M1; M4, M5 | PM board (Q3) |
| lightbulb refuses `qwen35` GGUF | M6, if the 27B-class model is Qwen3.5 | lightbulb |
| `plan.rs` is arch-blind | M3 | Unpopped (U1) |
| No sm_61 build or emit path proven | M3, M4 | baracuda (B1) |
| lightbulb device index hardcoded to 0; fuel pin stale | M2 (pin), M6 (index) | lightbulb |
| No CUDA CI runner anywhere | regression protection after each milestone | open: a self-hosted runner needs CireSnave |

## 7. After this plan: AMD and Intel through Vulkan

Sequenced **after** M0–M5, in CireSnave's order, as relayed by the PM on 2026-10-07.
The fuel lane's finding at fuel `6dfc8248` (kernel table vs shader sources):

- 66 of 167 authored kernels ship embedded SPIR-V (39.5%).
- Missing whole: `layer_norm`, `rms_norm`, `flash_attention` (and its backward), `scatter_add`, `concat`, `index_select`, the optimized matmul, and most non-f32 casts.
- AMD and NVIDIA through Vulkan are live-verified on real hardware; Intel has no evidence.

So AMD and Intel LLM support is gated on those missing kernels, not on architecture. Fuel's device auto-pick (`fuel-examples/src/lib.rs:71-91`) also does not include Vulkan.
