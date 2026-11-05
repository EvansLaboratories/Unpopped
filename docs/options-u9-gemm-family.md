# Options note: the GEMM family at the kernel layer (ROADMAP U9, issue #120)

Status: **scope approved by the PM 2026-11-04 (option A): conformance kernels over the kernel layer, no new IR, one PR each; this note
first.** Option B (plan-layer `Access::Contraction` generalization) stays held on D2; **`StructureKey` is not touched.** Closes
nothing by itself: gap rows 3.13 and 3.15 of `docs/ir-union-design.md` stay open for the plan layer; this note says what the kernel
layer can already state and prove, and what is blocked on whom.

## Why no new IR

Every part of U9 that a kernel can state is control flow over things that exist at `origin/main` b8d376a: per-thread `Load`/`Store`/
`Let`/`For`, `Stmt::Mma` with its sparse and block-scaled atoms (0.42.30), `Collective` and `AtomicRmw`/`AtomicCas` (including on
`Shared`), `Block` and `Device` barriers (0.42.13), and (in the open #292) `CopyAsync`. A GEMM, an epilogue, a split-K fix-up or a
grouped problem table is *a program*, not a new statement. What the IR cannot say is an *intent*: "this is a GEMM with this epilogue,
choose the tiling", and that is the plan layer.

## What each U9 item becomes

| Item (row 3.13 / 3.15) | Kernel-layer form | Kernel (one PR each) |
|---|---|---|
| EVT-style epilogue: C read, bias, row/col broadcast, activation | the epilogue is an expression over the accumulator and extra `Load`s | **K1** fused epilogue `relu(alpha*acc + beta*C + bias[col])` (first; built right after this note) |
| Epilogue with aux store, multi-output, fused reduction (dbias, amax) | extra `Store`s; a reduction by `AtomicRmw` (`Add`/`Max`) or a block `Collective` | **K2** epilogue writing `D` and an aux `relu` mask, with a per-column `dbias` sum and a global `amax` |
| Split-K / sliced-K | K slices on different threads, combined through `Shared` + `Block` barrier (in-block) or `AtomicRmw Add` (cross-block) | **K3** both combines, compared with the unsplit result |
| Stream-K fix-up | a `Device` barrier between the partial phase and the fix-up phase (the S6a model) | **K4** |
| Grouped / pointer-array GEMM | a problem table (`m, n, k, offsets`) in buffers, one thread-group per problem, a loop over problems. **Real pointer arrays need pointer values** (`Place::Ptr`, D-R4-3), which stay held until a consumer asks (`docs/options-u3-remainder.md`); offsets into one buffer are the faithful kernel-layer form | **K5** |
| Structured sparse and block-scaled GEMM | a K loop of `Stmt::Mma` with the extras (metadata nibble, scales) accumulating in the fragment | **K6** (needs the MMA fragment plumbing; the largest) |
| Implicit-GEMM convolution fprop | the im2col index math inside the GEMM loop (NHWC, pad, stride), no staged matrix | **K7**; dgrad/wgrad and grouped conv follow only if a consumer asks |
| Mixed-input dequant (int4 weights, f16 activations) | the sub-byte load plus a `Cast` and a scale in the loop | held: the kernel layer's sub-byte `Load` has to be read at source first (not done) |

Every kernel uses `I32` (or integer-valued data) wherever the claim does not depend on float rounding, so the reference is a plain host
loop and equality is exact; the MMA kernel (K6) uses the atoms' own exact-product rule. Each kernel ships as an oracle test
(`tests/kernel_gemm_family*.rs`) and a cpu-c differential (compiled with a host C compiler, bytes compared), with a seeded-mutation pass
(the epilogue constants, a swapped operand, a dropped barrier, an off-by-one K slice), and data chosen so that a mutant cannot hide.

## What stays blocked, and on whom

| Plan-layer item | Needs | Owner |
|---|---|---|
| Mixed A/B/accumulator/output dtypes on `Access::Contraction` | a per-operand-dtype `StructureKey` (`STRUCTURE_KEY_VERSION` bump) | Fuel and KISS (propose-first, ROADMAP D2); parked by the PM 2026-10-24 |
| EVT as a plan node (epilogue grammar) | an epilogue clause in the KISS contract | KISS |
| Grouped / ptr-array as a plan op | a bundle ABI | Fuel |
| Split-K / stream-K / tile shape as a plan *parameter* | a schedule language (ROADMAP P9) | CireSnave's spend, parked |
| Structured sparsity and block-scaled as plan ops | new `OpKind`s and operand roles | KISS |

None of these is started or half-started here; the kernels above are evidence of what the plan-layer node will have to mean, and the
differential against a host loop is what the plan layer's lowering is later checked against.

## Decisions for the PM

| # | Question | Recommendation |
|---|---|---|
| Q1 | The kernel list K1..K7 and the one-PR-each order | As above; K6 and K7 last |
| Q2 | Each kernel is tests only, so a patch bump each (PM allocates) | Yes |
| Q3 | Mixed-input dequant held until the sub-byte `Load` is read at source | Yes |
| Q4 | Real pointer arrays stay held (`Place::Ptr`), grouped GEMM by offsets | Yes |

## Build order, once approved

K1, K2, K3, K4, K5, K7, K6; each with its cpu-c differential and mutation pass; the CHANGELOG entry of each
names what the kernel pins.
