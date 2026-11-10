# Changelog

**Why this file exists, and it is not bookkeeping.**

Measured 2026-09-06 by diffing each published `.crate` tarball against the tree —
a practice relayed by baracuda as *"keep the last release's consumer code and run
it against the new artifact; it answers what did this fix break, which no gate
written for the fix can ask."*

⚠️ **A public-API diff of `unpopped` 0.9.0 vs the tree reported PURELY ADDITIVE:
292 items → 293, nothing removed. Underneath were TWO behaviour changes in
`unpopped-vocab`, one of them NUMERIC — and both sat at a version number
IDENTICAL to the published one.** A signature diff cannot see a same-signature
behaviour change, and a version check cannot see a tree that never bumped.

---

## `0.44.37` — docs, the Slang sizing docs and corpus harness moved to fishstringer

`docs/slang-corpus-sizing.md`, `docs/slang-naming-convention-note.md`, `docs/slang-second-blocker-sizing.md` and `crates/unpopped/examples/slang_corpus_report.rs` (with its `[[example]]` entry) are deleted: fishstringer#13 (main 255f66f) holds them verbatim with a provenance header, the harness as `fishstringer-slang-parse/examples/slang_corpus_report.rs`. `ROADMAP.md` E12 and a `convert.rs` comment now point there; the other mentions (`docs/groupshared-sizing.md`, `docs/ir-union-design.md`, the Slang survey, CHANGELOG) are dated history and keep the old file names. `docs/ir-union-surveys/slang.md` and `docs/ir-union-evidence/slang-wave*.md` stay (IR-union evidence). No behaviour change.

---

## `0.44.36` — docs, options note for peer and multicast scopes and multimem ops (ROADMAP U12)

`docs/options-u12-peer-multicast.md`: what the seven languages have for cluster-peer shared memory, peer device buffers and `multimem` (evidence in the wave-2c files, ThunderKittens `multimem.cuh` read for the dtype and order table), what Unpopped lacks at 8ad1f43 (one address space per launch, no device rank, `Scope` ends at `Device`, no pointer values), three options (A places plus one `Stmt`, B statements for every form, C defer) and a recommendation: A in four slices (cluster-peer places, devices and peer buffers, multicast stores and reductions, then the one breaking `MulticastLoadReduce`). Four rulings are asked of the PM. Docs only; no code.

---

## `0.44.35` — docs, the master capability list's Unpopped column refreshed for the R5.7 rows (F3, D4, C4, C4b, F7)

`docs/ir-union-master-capabilities.md`: the Unpopped column is re-read at `origin/main` 5c12a76b now that R5.7 PRs #319 to #326 have merged. D4 (a combine function and a tuple state: `ReduceWith`/`ScanWith`, `CallTuple`, `CollectiveTuple`) and C4 (`MaskedLoad`/`MaskedStore` with a user fill) go 🟡 to ✅. F3 (Int32 accumulators landed; TF32, 3xTF32 and fp8 fast-accumulate absent), C4b (clamp and reflect are an index remap, not a named mode) and F7 (the convolution family is kernels, no fused conv op) stay 🟡, and their per-row notes say what landed and what is still missing. G3 and G4 are unchanged. No code change; the other seven columns are untouched.

---

## `0.44.34` — feature, a tuple state for a combining collective: KernelFn::results, Stmt::CallTuple and Stmt::CollectiveTuple, ROADMAP R5.7 D4b

Triton's `tl.reduce` and `tl.associative_scan` fold a tuple with a user combine (argmax carries `(value, index)`, Welford `(mean, m2, count)`), which a scalar combine (`ReduceWith`/`ScanWith`, 0.44.28) cannot express. Option B of `docs/options-d4b-tuple-state-combine.md` (PM-approved 2026-11-09): the state is `k` scalars, each its own type.

**New `Stmt` variants (`Stmt` stays exhaustive, as ruled): `Stmt::CallTuple { callee, args, results }` and `Stmt::CollectiveTuple { kind, values, results, scope }`.** Also `KernelFn::results: Vec<u32>` (the values after `result`; `KernelFn` is `#[non_exhaustive]`, so a new defaulted field breaks no literal), `KernelFn::with_results` and `KernelFn::all_results`.

- `CallTuple` calls a module function that returns several values and receives all of them, one local each, in order (`result` then `results`); `Stmt::Call` still receives the first value only. The checker types each result local against its value, the oracle and the resumable evaluator carry every value across the frame, and `unpopped-cpu-c` emits a callee with one out-pointer per value (`res`, `res1`, ...), the first byte-identical to before.
- `CollectiveTuple` takes `kind: ReduceWith | ScanWith { combine }`, `values` (one per component) and `results` (one local per component). The combine is a module function of `2k` scalar parameters (the component types, then again) that returns `k` values and reaches no barrier; the checker holds it to that, refuses an empty tuple, a mismatch of values and results, any other kind, a scope other than Subgroup or Block, and a combine that is not `(a0.., b0..) -> (r0..)` over the component types. The author asserts associativity, the oracle folds left to right in lane order, a float component is compared within a tolerance, there is no exclusive scan (as for the scalar form). It is a barrier site: uniformity marks every result divergent and refuses it under thread-dependent control.
- `unpopped-cpu-c` emits it as a block-level construct between two thread loops (`cv<n>_<k>` per component, the running state `acc<n>_<k>`, the combine called with both states and one `&cr<n>_<k>` per component).

Tests: `kernel_call_tuple` (the argmax step called in a serial loop and in a launch, hand-worked), `kernel_collective_tuple` (Triton's argmax over `(value, index)` and Chan's Welford merge over `(mean, m2, count)`, reduce and scan, block and subgroup, hand-worked values; the checker's refusals; a non-commutative combine that fixes the argument order; the call-graph cycle through a tuple collective; the divergence rule), and their `unpopped-cpu-c` twins `kernel_call_tuple_emit_agrees` and `kernel_collective_tuple_emit_agrees` (byte for byte; Welford within 2 ulps because the C evaluates a float chain in its own way). Seeded mutants all caught: 7 of the call and 16 of the collective (oracle and checker) and 3 and 6 of the emitted C.

External break surface, measured 2026-11-09 at `origin/main` of baracuda `316bb006`, vulkane `73ff8c0`, fishstringer `53fe72d`, fuel `de12d8efc` and KISS `2727c22`: no `match` over `unpopped::kernel::Stmt` outside this repo. baracuda's `extern_ptx.rs` constructs `Stmt::{Let, Load, Store, ExternCall}` and vulkane's and fishstringer's `lower_kernel_default_is_the_typed_decline.rs` call `Stmt::barrier`, which a new variant leaves alone (the same grep finds 49 files here, so the pattern works).

---

## `0.44.33` — docs-only, correct the meaning of the FireLine seam in the interface plan (definitions shared, no runtime hop required)

`docs/fireline-unpopped-interface-plan.md` section 8 heading said Fuel talks to Unpopped "via FireLine, no direct wiring". CireSnave's decision 395 (2026-11-09) says every seam has its definitions in FireLine and crates use them; nothing has to hop through FireLine at run time. Heading corrected; the content (which definitions cross) is unchanged.

---

## `0.44.32` — docs-only, options note for a tuple state in a combining collective (R5.7 D4 slice b)

Adds `docs/options-d4b-tuple-state-combine.md`: why the tuple state cannot be a packed local today (the kernel evaluator refuses a packed value; a callee has one result), three representations, and a recommendation (a struct of scalars: multi-result callees and a `CollectiveTuple` statement). No code.

---

## `0.44.31` — tests, the implicit-GEMM convolution family (forward, data gradient, filter gradient) with stride, dilation and padding at the kernel layer, ROADMAP R5.7 F7

`kernel_conv_family` (oracle, against a host scatter reference) and `kernel_conv_family_emit_agrees` (`unpopped-cpu-c` against the oracle): an NHWC convolution with stride (3, 1), dilation (2, 3), padding (1, 1), a non-square image and filter and taps off the image at both ends of both axes, as three kernel-layer kernels with the im2col index math in the loops and no staged matrix: `fprop`, `dgrad` (each input element gathers the taps that reach it, with the divisibility and range test of a strided convolution) and `wgrad`. No IR change: U9 K7 was the stride-1, dilation-1 forward kernel; this generalises it and adds the two backward convolutions CUTLASS names beside it.

---

## `0.44.30` — feature, Stmt::MaskedLoad and Stmt::MaskedStore (predicated access with a user fill value), ROADMAP R5.7 C4

`kernel::Stmt::{MaskedLoad { local, place, mask, fill }, MaskedStore { place, value, mask }}` (`Stmt` is `#[non_exhaustive]`): a load declares its local as the element where `mask != 0` and as `fill` (any value of the element type, not only zero) where the mask is zero; a store writes only where the mask holds. Where a mask is false the memory is not touched: no bounds check and no race record, so an index outside the buffer under a false mask is legal and a false lane cannot race a true one. The checker types the fill and the mask; the uniformity analysis treats a value as divergent when its mask or fill is; the oracle and `unpopped-cpu-c` run it (the C is a bounds-checked access inside `if ((mask) != 0)`).

`kernel_padding_modes` (oracle, hand-worked rows) and `kernel_padding_modes_emit_agrees` (`unpopped-cpu-c` against the oracle): a 1-D window read at positions -2..7 of a five-element array in each padding mode. Zero, NaN, constant and -inf fill are a `Stmt::MaskedLoad`; replicate (clamp) and reflect are an index remap (Min, Max) in front of a load. No IR change: no language in the survey has reflect or replicate as a built-in load mode, so the modes are expressible and this is the evidence.

---

## `0.44.29` — feature, a collective folded with a user combine function (CollectiveKind::ReduceWith and ScanWith), ROADMAP R5.7 D4

`kernel::CollectiveKind::{ReduceWith { combine }, ScanWith { combine }}` (the enum is `#[non_exhaustive]`): every lane of a `Subgroup` or `Block` receives the fold (or the inclusive prefix fold) of the scope's values under a plain module function `(a: T, b: T) -> T` with no barrier, left to right in lane order. The author asserts the function is associative, so a device may reassociate it. `check_module` resolves the function (the collective is a call edge, so a cycle is refused) and `check_kernel` alone refuses it; the oracle runs it; `unpopped-cpu-c` spells a call per lane. Not built: an exclusive scan (no identity) and a tuple (packed) state.

---

## `0.44.28` — feature, AccumSpec::Int32 and Int32Sat (s8/u8 into an s32 accumulator) for a contraction, ROADMAP U7 S1b integer part

`ir::AccumSpec::{Int32, Int32Sat { k_step }}` (`AccumSpec` is `#[non_exhaustive]`): `I8`/`U8` operands, exact products, an `i32` accumulator that wraps (`Int32`, the exact sum modulo 2^32) or clamps into the `i32` range after every `k_step` elements (`Int32Sat`, PTX `.satfinite`). The oracle models both; the plan gate admits them only for an `I8`/`U8` cell with an `I32` accumulator and output in the key, `out_dtype = Some(I32)` and the identity epilogue, else the new `ContractionDecline::IntegerPolicyMismatch { detail }`. No `StructureKey` change (the key already carries `wdt`/`acc`/`out`). The kernel lift declines them as it does every policy but `WideFloat`. `Fp8FastAccum` is not built (no documented hardware model).

---

## `0.44.27` — feature, RoundMode::NearestAway (ties away from zero) for Cast; arithmetic qualifiers refuse it

`ir::RoundMode::NearestAway` (PTX `.rna`, C `round`): a `Cast` rounds to nearest with ties away from zero, in the oracle for every inexact conversion (float to integer, f64 to f32, integer to f32/f64, 16-bit floats, MX elements). `unpopped-cpu-c` spells it for float to integer (`round`) and declines the other inexact conversions with a reason. A `Qualified` arithmetic operation takes only the four IEEE directions, so `NearestAway` there is a new `TypeError::IllegalQualifier` and a plan-gate decline. Stochastic rounding is still not here. The enum is `#[non_exhaustive]`.

---

## `0.44.26` — feature, canonical text and parser for Layout, Swizzle and ComposedLayout (swz(B,M,S)(shape:stride)) for KISS Contract 6.5 output_layouts

`unpopped-vocab`: `Display` for `Swizzle` (`swz(B,M,S)`) and `ComposedLayout` (`swz(B,M,S)(shape:stride)`), and `FromStr` for `Layout`, `Swizzle` and `ComposedLayout`. One canonical spelling: no spaces, no leading `+` or zero, no `-0`, only the swizzle shift may be negative; nesting is bounded by `MAX_DEPTH` before recursion. New `LayoutError::Parse(reason)` (the enum is `#[non_exhaustive]`). Round trip tested against the two entries of KISS `output_layout_vectors`.

---

## `0.44.25` — docs, doc comments that said Call, AtomicCas and the module type were unchecked, and the U11 note that said nothing was built

`kernel.rs` said `check_kernel` refuses `AtomicCas` and that nothing checks or runs `Stmt::Call` or `Module`; both are checked and run. `docs/options-u11-extern-and-inline-asm.md` said "nothing built" after typed `Extern` and `Stmt::ExternCall` landed (0.43.0). Comments only.

---

## `0.44.24` — fix, a flaky cpu-c row-sort test (two tests wrote the same C file name in parallel)

`kernel_emit_row_sort_agrees`: `degenerate_rows_agree` and `full_sorts_agree_in_every_form` built the same tag (`sort_F32_Asc_Both_0_2`, `..Desc..`), so the parallel tests overwrote each other's `.c` and `.exe`. The tag now includes the shape. Test-only.

---

## `0.44.23` — docs-only, the plan for moving the IR and lowering contract into FireLine interface crates

Adds `docs/fireline-unpopped-interface-plan.md`: what moves, what stays, what breaks on the first release, and the drift test. No code or version change.

---

## `0.44.22` — master capability list, every cell checked against primary sources (docs only)

`docs/ir-union-master-capabilities.md` promotes the 46-row capability grid into the master list of everything the seven kernel languages (CUDA, CUTLASS, CuTe, Slang, Mojo, Triton, ThunderKittens) can do, against the Unpopped IR at b3e44429. Every cell cites its primary source (version, path or URL, quote) in `docs/ir-union-evidence/`; the unknown count is 0. New rows C4b (padding fill modes), F9 (MMA operand count and kinds) and I1 (normalisation family, several axes). The R5.7 rows come first, each language column names the version it was read at, and a schedule-only marker flags rows KISS does not describe. The Unpopped column is refreshed (A2, A3, B1, C1, D6, E3, E4, E6, H1 had landed). KISS owns the verdict column, keyed by these row ids.

---

## `0.44.21` — tensor-core attention conformance kernel A6 (ROADMAP U10; tests only)

`kernel_attention_mma.rs` and `unpopped-cpu-c/tests/kernel_attention_mma_emit_agrees.rs` pin attention with both GEMMs on `Stmt::Mma`, written with the kernel layer alone: `S = scale * Q K^T` as two MMAs over two key tiles, a scalar row softmax between block barriers (the lanes exchange `S` and `P` through global scratch, with the atom's own fragment layout supplied as id tables), and `O = P V` as a third MMA, on an `F32` toy `m4n4k8` atom (the shipped half-precision atoms have no C element type in cpu-c). Scores pass 89 and the row maximum falls in both key tiles. Oracle against a host loop; C against `run_launch` within 2 ulps. See `docs/options-u10-attention.md`.

---

## `0.44.20` — paged-KV decode attention conformance kernel A4 (ROADMAP U10; tests only)

`kernel_attention_paged.rs` and `unpopped-cpu-c/tests/kernel_attention_paged_emit_agrees.rs` pin flash-style decode attention over a paged KV cache written with the kernel layer alone: K and V in a pool of pages, an `I32` block table in a buffer (options note Q5, no pointers) mapping (sequence, logical page) to a pool page, and per-sequence lengths (a partial last page, exactly one page, an odd count, and a zero-length sequence that yields zeros). Unused table entries name a poison page, so a wrong page, slot or length changes the answer. Oracle against a host gather; C against `run_launch` within 2 ulps. See `docs/options-u10-attention.md`.

---

## `0.44.19` — masked attention conformance kernel A3 (ROADMAP U10; tests only)

`kernel_attention_masked.rs` and `unpopped-cpu-c/tests/kernel_attention_masked_emit_agrees.rs` pin A2's online-softmax attention with a per-key visibility test (`j + lag <= row`; `lag = 0` is causal), an optional ALiBi bias `slope * (row - j)` on visible scores, and the defined result for a row that sees no key: zeros, not NaN (options note Q4). Oracle against a host loop in the same `f32` order; C against `run_launch` within 2 ulps. See `docs/options-u10-attention.md`.

---

## `0.44.18` — flash-style attention conformance kernel A2 (ROADMAP U10; tests only)

`kernel_attention_flash.rs` and `unpopped-cpu-c/tests/kernel_attention_flash_emit_agrees.rs` pin an online-softmax attention written with the kernel layer alone: no score matrix, per row and output column a running max, sum and accumulator, rescaled by `exp(m - m')` when a key raises the max. The oracle result matches A1's naive kernel and the host loop; the C matches `run_launch` within 2 ulps. See `docs/options-u10-attention.md`.

---

## `0.44.17` — naive attention conformance kernel A1 (ROADMAP U10; tests only)

`kernel_attention_naive.rs` and `unpopped-cpu-c/tests/kernel_attention_naive_emit_agrees.rs` pin a single-head attention written with the kernel layer alone (scores into a scratch buffer, row max, `exp(s - max)` sum, weighted values; `F32`, one thread per query row, scores past 89 so an `exp` without the max subtraction overflows). Oracle against a host loop; C compiled and run against `run_launch` within 2 ulps. Nine seeded mutants are caught by the oracle test (the overflow ones depend on the oracle rounding `F32` locals, the fix in the same queue); one is equivalent and documented. See `docs/options-u10-attention.md`.

---

## `0.44.16` — the kernel oracle rounds a floating local to its own type

`Let`, `Assign` and a call's scalar argument now round a floating value to the local's type (`F32`, `F16`, `BF16`, `F32Strict`) when it is written, as the serial C does for a `float` variable; before, the oracle kept the `f64` result in an `F32` local between statements, so `exp(234)` stayed finite and `2^24 + 1 - 2^24` was `1` where `float` C gives `inf` and `0`. Found writing the U10 attention kernel A1. `tests/kernel_f32_local_rounding.rs` is red on the old behaviour (an overflowing local, an assigned local, an `F64` control); the full `unpopped`, `unpopped-cpu-c` and `unpopped-vocab` suites pass unchanged, so no test moved and no tolerance was touched. Recorded in `docs/ir-kernel-layer.md` section 5.

---

## `0.44.15` — docs: options note for U10 attention at the kernel layer; ROADMAP U9b marked done

`docs/options-u10-attention.md` proposes scope A for ROADMAP U10 (kernel-layer attention conformance kernels A1 to A4 and A6 with cpu-c differentials and seeded-mutation passes, no new IR; backward held) and lists the plan-layer items that stay blocked on KISS and Fuel. The U9b row is updated to match #305 and #306. No code changes.

---

## `0.44.14` — cpu-c spells `I4`/`U4` kernel buffers and sub-byte `Cast` (ROADMAP U9b S2; additive: a refusal becomes an acceptance)

A kernel `Load` of an `I4`/`U4` buffer element is spelled as a shift and mask of the packed byte (low nibble first; `I4` sign-extended by xor-and-subtract), a widening `Cast` as the plain C conversion and a narrowing `Cast` as a wrap modulo 16. The buffer argument is `const unsigned char*`. `kernel_subbyte_load_refused.rs` becomes `kernel_subbyte_emit_agrees.rs` (C compiled and run against `run_launch` over all 16 nibble values and the narrowing wrap); five seeded mutants caught. S3 adds the dequantizing GEMM conformance kernel (`out[r][c] = sum_k (w4[r][k] - zero[g]) * scale[g] * x[k][c]`, `I4` and `U4` weights; `kernel_dequant_gemm.rs`, `kernel_dequant_gemm_emit_agrees.rs`; five seeded mutants caught: nibble order, zero point ignored, scale group ignored, zero group ignored, k stride).

---

## `0.44.13` — sub-byte `Cast` endpoints `I4`/`U4` (ROADMAP U9b S1; additive: a refusal becomes an acceptance)

Ruled by the PM 2026-11-05 (`docs/options-u9b-subbyte-dequant.md`). `check_cast` now accepts `I4` and `U4` against an integer endpoint with `Saturation::Wrap`
(widening is exact: `I4` sign-extends, `U4` zero-extends; narrowing wraps modulo 16); `types::{is_subbyte_int, subbyte_range}` are new. The oracle evaluates
a sub-byte value as a local, a constant or a narrowing `Cast`. The checker refuses, typed, a `Store` of a sub-byte element (sub-byte buffers are read-only in
kernels in v1: a store is a read-modify-write of a byte two threads share) and any operator over a sub-byte operand (widen it with a `Cast` first); a
saturating cast or one against a float endpoint is still refused. `int_range` is unchanged (47 call sites treat it as "an arithmetic integer"). The C emitter
still does not spell an `I4` buffer (slice S2, pinned by `kernel_subbyte_load_refused.rs`). Tests: `kernel_subbyte_load` (6); nine seeded mutants caught.

---

## `0.44.12` — CUDA Tile IR reader spike (a report and a throwaway probe; docs and tools only)

`docs/cuda-tile-ir-reader-spike.md` and `tools/spikes/cuda_tile_probe.py`: NVIDIA/cuda-tile read at `0192668`, parse and validate only,
no lowering. 105 ops, 29 bytecode files (25 parsed, 4 of 4 structural rejections matched), 96 of 105 ops used by the 92 text tests;
the suite is not a lowering oracle and the open dialect has no `mbarrier`/`wgmma`/`cp.async` ops. Nothing copied from the repository.

---

## `0.44.11` — GEMM-family kernel K6: a K loop of `Stmt::Mma` accumulating in the fragment (ROADMAP U9; tests only)

`unpopped-cpu-c/tests/kernel_gemm_mma_chain_emit_agrees.rs`: two `Stmt::Mma` run back to back over two K slices, the second reading
the first result as its `C` (`d` is `c`), for a dense atom, a 2:4 sparse atom, a block-scaled atom and both (the toy 4-lane
atom of the extras tests). For each the compiled C agrees with the launch interpreter byte for byte, and the interpreter chained
result equals the sum of the two slices run as separate single-slice MMAs on a zero `C`, which pins the accumulation itself.
Four seeded product mutants (the oracle ignoring `C` in the integer and in the float path, the C emitter starting its
accumulator at zero, its K loop one short) are all caught. The sparse WGMMA/tcgen05 atoms and a real warp-level tile loop
wait for a second lane-table source (ROADMAP U7 / U8b).

---

## `0.44.10` — GEMM-family kernel K7: implicit-GEMM convolution forward (ROADMAP U9; tests only)

`conv_implicit_gemm` (`tests/support/conv.rs`, copied into cpu-c): NHWC forward convolution as a GEMM over `(r, s, c)` with the
im2col index math inside the loop and zero padding by skipping taps outside the image, no staged matrix. The shapes make an
index mix-up loud: a non-square image (4x5), a non-square filter (3x2), padding on height only, three input channels, two
output channels, two batches (one block each). Oracle, the C compiled with a host compiler, and a direct host convolution
agree. Twelve seeded mutants (pad dropped or added, filter or input coordinates swapped, batch ignored, bounds swapped
or off by one, channel loop short, output coordinates swapped, accumulator not accumulated) are all caught. dgrad, wgrad
and grouped convolution follow only if a consumer asks.

---

## `0.44.9` — GEMM-family kernel K5: grouped GEMM by a problem table (ROADMAP U9; tests only)

`grouped_gemm` (`tests/support/grouped.rs`, copied into cpu-c): three GEMMs of different shapes `(3,2,4)`, `(4,4,2)`, `(2,4,3)`
in one launch, one block per problem, a `U32` problem table `[P, 6]` of `m, n, k` and element offsets into one flat buffer per
operand (the kernel-layer form of a grouped / pointer-array GEMM until pointer values exist, ROADMAP U9c); a thread outside
its problem's `m x n` does nothing, and `d` is exactly the packed outputs so a stray write runs off or overwrites a neighbour.
Oracle, the C compiled with a host compiler, and the host loop agree. Eleven seeded mutants (block index unused, bounds
swapped or removed, `k` end wrong, each of the three offsets ignored, row strides swapped, accumulator not accumulated) are
all caught.

---

## `0.44.8` — GEMM-family kernel K4: split-K with a `Device`-barrier fix-up (ROADMAP U9; tests only)

`split_k_with_fixup`, the stream-K shape without atomics: the two blocks of a `1x1x2` grid write half-K partials to a
global workspace, a `Device` barrier orders the phases, and block 0 adds them into `out`. Oracle, the C (the S6a phase
loop) and the host loop agree. Five seeded mutants (barrier removed or demoted to `Block`, colliding workspace slots, one
partial read, slice not chosen by block) are caught; moving the fix-up to block 1 survives and is equivalent (either block
computes the same `out`).

---

## `0.44.7` — GEMM-family kernel K3: split-K in a block and across blocks (ROADMAP U9; tests only)

`split_k_in_block` (two thread layers per output sum their half of K, publish through `Shared`, a `Block` barrier, one adds
the two) and `split_k_across_blocks` (a `1x1x2` grid, each block adds its half into `out` with a device-scope
`AtomicRmw Add`), both equal to the unsplit product on mixed-sign data; oracle, C compiled with a host compiler, and host
loop agree. Nine seeded mutants (slice end short, slice start wrong, barrier removed, one partial only, wrong slot, atomic
add to max or exchange, block index unused) are all caught.

---

## `0.44.6` — re-vendor KISS `dtype_manifest.json` (KISS#772; vendored data and its pin only)

`crates/unpopped-vocab/kiss/dtype_manifest.json` re-vendored verbatim from KISS `e786097`: only the `generated_from` member
changed (it names the emitting tool now); no entry, count or schema moved. Blob `129690fe` read back equal to KISS, 0 CR bytes;
the content pin in `tests/kiss_byte_match.rs` (length 3,650, FNV-1a-64) and the `kiss/README.md` table moved with it.

---

## `0.44.5` — CHANGELOG fragments and `tools/stamp-version.py` (the version-stamp pilot; tools and docs only)

`changelog.d/<topic>.md` fragments and `python tools/stamp-version.py <version>`, ported from coderipper's pilot to
Unpopped's layout: a PR adds a fragment and never edits `CHANGELOG.md`, a `Cargo.toml` version, `Cargo.lock` or the cpu-c
README pins; the one command sets the version in the root path-dependency lines, the four crates, the four `unpopped*`
lockfile entries and the README pins, writes one `## `<version>` — <kind>` heading from the fragments above the newest entry,
and deletes them. Self-tested on a throwaway fragment (`--dry-run`, then a real run, `cargo check --workspace --locked`
green, CRLF kept, a repeat of the same version refused). This PR carries only its fragment and no number: the gate stamps it.

---

## `0.44.4` — GEMM-family kernel K2: aux outputs, `dbias` and `amax` by atomics (ROADMAP U9; tests only, stacked on 0.44.3)

`aux_outputs` in `tests/support/gemm.rs`: the same main loop, then `D = acc`, a mask `acc > 0`, the per-column sum of
`relu(acc)` by `AtomicRmw Add` (`dbias`) and the maximum of `acc` by `AtomicRmw Max` (`amax`, started at `i32::MIN`), on
a mixed-sign dataset and an all-negative one (on the second, `max(relu(acc))` would be 0, not the maximum, which a
positive-only dataset cannot tell apart). Oracle, the C compiled with a host compiler, and the host loop agree. Eight
seeded mutants (add to max, max to min, no relu in `dbias`, mask flipped or inverted, `dbias` by row, `D` clamped,
`amax` of the clamped value) are all caught; the last survived the first, positive-only dataset.

---

## `0.44.3` — GEMM-family kernel K1: the fused-epilogue integer GEMM (ROADMAP U9; tests only)

`tests/kernel_gemm_family.rs` and `unpopped-cpu-c/tests/kernel_gemm_family_emit_agrees.rs` (builder `tests/support/gemm.rs`,
copied into cpu-c): `out = relu(2 * (A.B) + 3 * C + bias[col])`, one thread per output on a 4x4 launch, `I32` so a plain
host loop is the exact reference, with data that clamps some outputs and not others. The oracle, the C compiled with a
host compiler, and the host loop agree. Twelve seeded mutants (alpha, beta, C, bias dropped; bias indexed by row; relu
dropped or flipped; K loop short; A or B transposed; output transposed; accumulator not accumulated) are all caught.

---

## `0.44.2` — options note for U9 (the GEMM family at the kernel layer); docs only

`docs/options-u9-gemm-family.md` (ROADMAP U9, issue #120): scope A approved by the PM 2026-11-04 (kernel-layer conformance
kernels K1..K7, no new IR, one PR each), what each U9 item becomes, and the plan-layer items that stay blocked on Fuel, KISS
and the PM (D2, P9). `StructureKey` is not touched.

---

## `0.44.1` — the double-buffered `cp.async` conformance kernel (ROADMAP U8 S4; tests only, stacked on 0.44.0)

`tests/kernel_pipeline_double_buffer.rs` and `unpopped-cpu-c/tests/kernel_pipeline_emit_agrees.rs` (the builder is
`tests/support/pipeline.rs`, with a copy in cpu-c): four threads sum a four-tile pipeline through a two-stage `Shared`
ring, prefetching tile `k + 1` while consuming tile `k` of the mirror thread, one group committed per iteration. The
correct kernel sums to `[436, 432, 428, 424]` and the C agrees byte for byte; waiting for too little, dropping the read
barrier and dropping the reuse barrier are each a typed trap (`UninitializedShared`, `UninitializedShared`,
`DataRace`). Writing the third found that the 0.44.0 oracle logged the destination write only at the wait, so a refill
issued before another thread's read passed silently; 0.44.0 now logs the issue as a write too.

---

## `0.44.0` — **BREAKING**: `Stmt::{CopyAsync, AsyncCommit, AsyncWait}`, `cp.async` in the IR (ROADMAP U8 S1)

`Stmt` is exhaustive, so three new variants break every downstream `match`; `KernelError` gains `AsyncUnsupported` and
`KernelTrap` gains `AsyncRace { alloc }` and `AsyncUnwaited` (both `#[non_exhaustive]`). `CopyAsync { dst, src, elems }`
copies 4, 8 or 16 bytes from a readable global buffer place to a `Shared` allocation place; `AsyncCommit` closes the
thread's open copies into a group (an empty commit still makes one, as on hardware); `AsyncWait { pending }` lands the
older groups. `check_kernel` refuses a destination that is not a `Shared` allocation, a source that is not a global
buffer, a write-only source, mismatched element types and other sizes. The oracle reads the source at issue and writes
the destination at the wait; touching a destination element before it lands is `AsyncRace`, a copy never waited for is
`AsyncUnwaited`, the issue and the landing are both logged `Shared` writes (the hardware may write any time after issue, so
a refill races a reader unless a barrier follows the read; another thread reads a landed copy after a barrier).
`unpopped-cpu-c` performs the copy at issue and spells commit and wait as nothing. Not checked statically: the
destination's alignment to its size, and whether a wait covers a copy (both depend on run-time values and control
flow; the oracle traps on the second). Tests: `kernel_async_copy` (14), `kernel_async_copy_emit_agrees` (4, real host C
compiler); 15 seeded oracle/checker mutants and 3 emitter mutants all caught.

---

## `0.43.3` — cpu-c tests no longer reach outside their crate (sweep SUP-004; tests only)

`kernel_emit_argmax_agrees.rs` took `arg_loop.rs` from `../../unpopped/tests/support`, a path the published `unpopped-cpu-c`
package does not contain. The helper is now a copy in `unpopped-cpu-c/tests/support/` (header says to change both).

---

## `0.43.2` — options note for U8 (copy atoms, pipelines and roles); docs only, stacked on 0.43.1

`docs/options-u8-copy-atoms.md` (ROADMAP U8, issue #119): what exists, the line between IR and idiom, four slices
(S1 `cp.async` and async groups recommended first; S2 `ldmatrix`/`stmatrix` held on a second lane-table source; S3
`mbarrier`/TMA/descriptors waits for the Baracuda sm_90 emitter; S4 pipelines and roles as recipes, no IR node) and six
questions for the PM. Nothing is built.

---

## `0.43.1` — the oracle and cpu-c run an `ExternCall` through its reference (additive, ROADMAP U11 step 2; stacked on 0.43.0)

The oracle runs `Stmt::ExternCall` as a call of the declaration's `reference` kernel on the calling thread, in both evaluators (`run_module`, and `run_launch_module` with a frame stack per thread); a declaration with no reference traps `Unsupported("an Extern with no reference body")`. `unpopped-cpu-c` spells it as a C call of the reference (the reference is emitted with the other callees); a declaration with no reference, or with a body of its own target `"cpu-c"` which it would have to honour, is refused with a typed `UnsupportedKernel` rather than silently replaced. Only a device leg can show an asm body means what its reference means (step 3). A declared `sync` effect is not modelled by the reference run: the oracle rendezvouses only if the reference itself holds a barrier. Tests: `kernel_extern_run` (6) and `kernel_extern_emit_agrees` (6, byte-for-byte against the oracle on a host C compiler); four oracle and five emitter seeded mutants all red.

## `0.43.0` — typed `Extern` declarations and `Stmt::ExternCall`, checked (**breaking**: `Stmt` gains a variant; ROADMAP U11 step 1; version proposed, the PM allocates)

`Module` gains `externs: Vec<ExternDecl>` (`#[serde(default)]`, so older modules load) and `Stmt::ExternCall { decl, args, results }`. An `ExternDecl` has scalar `params` and `results`, declared `Effects` (`pure`, `sync`, `volatile`; `reads`/`writes` reserved and empty in v1 because there are no pointer operands), a `TargetBody` per target (`BodyText::Asm` with `{N}` operand placeholders, or `BodyText::Symbol`) and an optional `reference` kernel. `check_module` holds a declaration to its own words (`pure` excludes volatile, sync and memory effects, and needs a result; no `Thread` sync; one body per target; placeholders within the operands; a reference is a plain scalar kernel with exactly the declared signature and no extern call) and a call to its declaration (arity, argument and result types, result locals declared by the call, a sync scope the launch has). The uniformity rule treats a `sync` call as a barrier site (also through `Call`) and marks the results of a call that is not pure, or has a divergent argument, as divergent. `check_kernel` alone refuses an `ExternCall`. **The oracle and cpu-c do not run it yet** (typed `Unsupported` / refusal): running the `reference` is step 2. Tests: `kernel_extern_check` (10 tests); nineteen seeded mutants (one per check) all red.

## `0.42.31` — options note for U11, typed `Extern` and inline asm (docs only)

`docs/options-u11-extern-and-inline-asm.md`: the shape of a typed `ExternDecl` (declared effects, per-target bodies, an optional reference kernel) and `Stmt::ExternCall`, approved by the PM as written (Q1-Q6). Nothing is built; no behaviour changes.

## `0.42.30` — cpu-c spells `Stmt::Mma` extra operands (additive, ROADMAP U7 S3b; version proposed, the PM allocates)

`unpopped-cpu-c` now spells an atom with 2:4 metadata and/or block scales instead of refusing it: the extras get id tables from the atom like A, B and C; a sparse `A` is expanded into a dense matrix through the metadata (the two other positions of a group are zero); scales multiply the factors as `(a * sa) * (b * sb)` per `k` block, in the oracle's order. Preconditions the C cannot check (the oracle traps on them): broadcast lanes agree, a metadata nibble is `i0 < i1`. A statement whose `extra` list is not the atom's is refused with a typed error. Tests: `kernel_mma_extras_emit_agrees` compiles and runs the C against `run_launch` byte for byte for a sparse s8 atom, a scaled f32 atom, both together and a broadcast scale. Eight seeded mutants (kept positions swapped, compressed slot swapped, scale block index, `SFB` stride, sparse `A` ignored, metadata mask, `SFA`/`SFB` swapped, `SFB` stride +1) all red.

## `0.42.29` — 2:4 sparse and block-scaled MMA atoms (additive in behaviour; `Stmt::Mma` gains a field, ROADMAP U7 S3)

An `MmaAtom` may now carry **extra operands**, numbered after C: `with_sparse_24(a, (meta_dtype, meta_layout))` stores `A` compressed as `M x K/2` and adds an `M x K/4` metadata operand (the low four bits of an entry are `i0 | i1 << 2`, the kept positions in a group of four `k`, `i0 < i1`); `with_block_scale(block, dtype, (sfa, sfb))` adds the scales `SFA` (`M x K/block`) and `SFB` (`K/block x N`), applied as `(a * sa) * (b * sb)` per `k` in `f64` (float accumulators only). The flags compose. An extra operand's layout need only **cover** its matrix, since scales are broadcast on hardware; lanes that hold one element must hand over the same bits (else the oracle traps). `Stmt::Mma` gains `extra: Vec<u32>` (the `Private` fragments of the extra operands, in `MmaAtom::extras()` order; `#[serde(default)]`), which the checker holds to the atom's operand list, types and lengths; `MmaAtom::values` and `dims` are no longer `const`. New `MmaAtomError::{NotACover, BadFlag, FlagTwice}`. **No PTX `mma.sp` or block-scaled constructor ships**: their metadata and scale-factor lane tables are not checked against anything but their author, so the flags are exercised by a toy 4-lane atom in `kernel_mma_flags`; `unpopped-cpu-c` refuses an atom with extra operands with a typed error (next slice). Tests: expected D is a plain triple loop over the dense matrices (sparse `A` expanded by the host, scales applied per element), with controls (compressed `A` read as dense differs; unit scales differ); metadata with `i0 > i1` and `i0 == i1` trap; a broadcast scale agrees or traps; refusals of every bad flag. Six seeded mutants (kept positions swapped, `SFB` dropped, scale block index wrong, equal positions accepted, broadcast disagreement ignored, sparse `A` dims left at `K`) all red.

## `0.42.28` — cpu-c spells `Stmt::Mma` (additive, ROADMAP U7 S2)

`unpopped-cpu-c` now emits a `Stmt::Mma` as a block-level construct between two thread loops: per subgroup of the atom's lane count it gathers the lanes' `Private` A, B and C fragments into the logical matrices through id tables baked from the atom (`MmaAtom::id`), forms `D = A.B + C` (floats from `C` in ascending `k` in `double`, rounded once when stored; integers exact in `long long`), and scatters D back. The shipped f16 and bf16 atoms are refused with a typed error naming the dtype (this emitter has no half C type); the s8 atom and a toy `F32`-operand atom with the PTX layouts (atoms are data) are spelled. Tests: `kernel_mma_emit_agrees` compiles and runs the C and compares it byte for byte with `run_launch`, including a case where the f32 sum is inexact; four seeded mutants (C dropped, accumulator `float`, D scattered to one subgroup's lanes, B read transposed) all red. **Builds on #283 (0.42.27).**

## `0.42.27` — `Stmt::Mma`: a warp-level matrix multiply-accumulate in the IR, checker and oracle (additive, ROADMAP U7 S2)

`Stmt::Mma { atom, a, b, c, d }` is a rendezvous of one subgroup (like `Collective`): each lane hands over its A, B and C fragments (`Private` arrays) and receives its share of `D = A.B + C`. **The checker** requires a launch whose subgroup size is exactly the atom's lane count, fragments that are `Private`, static, of the atom's element type and `values` long, and uniform control (`BarrierDivergent` otherwise); new `KernelError::MmaUnsupported`. **The oracle** gathers the lanes' fragments into the logical matrices through the atom's layouts, sums the exact products and `C` in `f64` in ascending `k` order and rounds once to the accumulator type (integers: exact, wrapping), and scatters `D` back; an unstored fragment slot traps `UninitializedPrivate`. A device may sum in another order, so compare floats within a tolerance. `unpopped-cpu-c` treats it as a barrier site and **refuses to spell it yet** (next slice). Tests: `kernel_mma` runs the f16, bf16 and s8 atoms with fragments placed by the PTX tables written independently, against a plain triple loop; refusals and the trap each have a control. Seeded mutants (B indexed transposed, C dropped, D scattered to the wrong lane) all red; dropping the integer `wrap` is an equivalent mutant (the store truncates to the same bits). **Stacked on #282 (0.42.26).**

## `0.42.26` — `unpopped::mma::MmaAtom`, MMA atoms as data (additive, ROADMAP U7 S2 first step)

A new `MmaAtom` type: the logical `M x N x K`, the A/B/accumulator types, the lane count and the thread-value `Layout` of A, B and C/D (column-major logical matrices; index `lane + lanes*value`). `MmaAtom::new` refuses a layout that is not a bijection onto its matrix, a scope other than `Subgroup`, and a wrong size. Shipped atoms: PTX `mma.sync` `m16n8k16` f16 and bf16 into f32, and `m16n8k32` s8 into s32; the `m16n8k8` tf32 atom waits for a tf32 operand notion (no tf32 `ElementKind`). No `Stmt` yet: `Stmt::Mma` (checker, oracle, cpu-c) follows. Tests: `mma_atoms` checks every fragment element against the PTX fragment tables written independently as `(row, col)` formulas; three seeded stride mutants all red. The tables and the layouts were both written from the same PTX ISA text, so they agree on a shared reading; an hardware or CUTLASS-derived cross-check is still open.

## `0.42.25` — cpu-c spells `Private` arrays (additive, ROADMAP U7 S2 groundwork)

The serialized thread-loop C now accepts a `Private` allocation: one slice of `n` elements per thread of the block (thread `t`'s element `o` is `t*n + o`, bounds-checked against `n`), also across `Device` barriers. It was refused before ("only Shared is spelled"). A `Private` array passed to a callee is still refused (its slice is not the callee's whole array); `Constant` and dynamic extents are still refused. `Stmt::Mma` fragments (U7 S2) need this. Tests: `kernel_private_array_emit_agrees` (C vs the launch oracle, per-thread-distinct data, with and without a Device barrier; two seeded mutants, a shared slice and a one-thread stride, both red).

## `0.42.24` — a full `[M,N]` C operand on a contraction (additive, ROADMAP U7 S1a-2)

**An Unpopped extension, not KISS** (KISS pins no C operand). A contraction's extra input `Input(2..)` that is a dense row-major `[M,N]` buffer is read at the output element `[m,n]`, so `D = alpha*(A.B) + beta*C` is an ordinary epilogue; a rank-1 extra input stays the `[N]` per-column bias. `StructureKey` already tells the two apart through `OperandKey::bcast` (bias broadcast over M), so no key change and no KISS RFC (PM ruling, 2026-11-03; `crates/unpopped-vocab/tests/a_full_c_and_a_bias_have_different_keys.rs` is the proof). Changed: the legacy key bridge admits a dense `[M,N]` extra operand (a strided, column-major, row-broadcast, wrong-shape or rank-3 one still declines); `oracle::eval_contraction` reads by operand rank and refuses a wrong-shape C by name; `kernel_lift::ContractionExtra::{Bias, FullC}` and `lift_contraction_opdef_with` (the default stays `Bias`). The role-driven `derive_structure_key` does not inspect an extra operand, as before. **Epilogue only** (PM ruling D2): C is added after the K-sum in the wide type, also under an `AccumSpec` policy; C as the initial accumulator, and its `u*|C|` bound, belong to `Stmt::Mma` (S2). Evidence per row, not a device run: an independent reference, the compiled-C differential (f32, f64, dense and column-major C), and 11 seeded mutants, all caught (one survivor, a rank-3 extra, was found and given its test).

## `0.42.23` — reduced-precision accumulation policies for a contraction (additive, ROADMAP U7 S1a; an Unpopped extension, not KISS)

`AccumSpec` gains four precision policies, each with a `k_step` (the atom's K): `Tf32` (inputs rounded to TF32, ties away from zero), `Tf32x3` and `Bf16x3` (head and remainder split, product `hh + hl + lh`, the `ll` term dropped) with an `f32` accumulator, and `F16Accum` (an `f16` accumulator, nearest even, overflow beyond 65520). The oracle's model: converted inputs, exact products, a sum in `f64` inside each step of `k_step` elements, the accumulator rounded after every step, so the bytes are deterministic (`oracle/accum_policy.rs`). `OpDef::with_accum` selects one; the operator stays a sum over `K` (the in-crate guard `accumspec_is_a_precision_policy_and_never_an_operator_choice` gets its arms). **Gate:** a policy needs `F32` operands (TF32 family) or `F16` (`F16Accum`) and `k_step >= 1`, else `ContractionDecline::AccumPolicyMismatch`. **Lift:** `lift_contraction_opdef` declines every policy but `WideFloat`, typed, so `unpopped-cpu-c` has no policy reference yet. **The declared bound per emitter** (a policy may be claimed only with a stated bound: `(gamma_K + gamma_s + d(P)) * sum|a~ b~|`, zero for a bit-exact reproduction) is in `docs/accum-policies.md`, with the mapping to KISS's two-valued `MathFidelity`. **Not here (S1b, with the mixed-dtype `StructureKey` change the PM deferred):** `Int32`/`Int32Sat` and `Fp8FastAccum`, which need an output dtype different from the inputs. **Tests:** `tests/contraction_accum_policy.rs` (7) holds the oracle to a reference written independently (TF32 by `f64::round`, bf16 and f16 through the `half` codec) over a corpus built so each policy departs from the plain sum and `k_step` matters, plus the plan and lift declines; three unit tests check the oracle's TF32 and its own f64-to-f16 rounding against exact integer arithmetic and `half`'s `from_f32`. Fifteen seeded mutants caught (TF32 truncating, keeping 11 bits; the `x3` forms dropping or adding a term; the accumulator never rounded or rounded to the wrong type; `k_step` ignored; the f16 overflow threshold, subnormal quantum and tie rule; the gate and the lift decline removed). **Finding:** `half::f16::from_f64` (2.7.1) mis-rounds values near a tie (already filed upstream as VoidStarKat/half-rs#151; 55 per million inputs with default features, 456 without); the oracle rounds exactly and says why.

## `0.42.22` — options note for U7, MMA atoms and accumulation policy (docs only)

`docs/options-u7-mma-atoms.md`: what U7 would build now that U3 and U6 are done, and the six rulings it needs. Read at `origin/main` `7c05d3c`. KISS has no clause for any of it (matmul is an order-invariant contraction compared under a tolerance; `MathFidelity` is two-valued), so it is an Unpopped extension to propose upstream when it lands. The note proposes two layers: a plan-level `AccumSpec` growth (TF32, 3xTF32, bf16x3, f16, int32, fp8 fast-accumulate) with accumulate-into-C and a deterministic oracle model held to a declared bound, then a kernel-level `Stmt::Mma` over data-driven `MmaAtom` values whose lane layouts are U6 `Layout`s; WGMMA and `tcgen05` are staged behind U8 and U4 S7. It also records that mixed A/B dtypes need a `StructureKey` version bump (a change other repos see) and recommends deferring that slice. ROADMAP U7 links the note. No code changed; nothing is built.

## `0.42.21` — ROADMAP row U2b for the effects model (docs only)

D6 of `docs/options-u2-lift-declines.md` gets its own row so #113 can close: D1-D5 are built, and the effects model waits for a consumer. No code changed; the version is a placeholder for the PM to allocate.

## `0.42.20` — a concept map of the IR for the KISS kernel-body layer (docs only)

`docs/kiss-kernel-body-concept-map.md`, written for the KISS lane (`jkkstudy`) at the PM's request after CireSnave agreed (2026-11-03) that this IR is source material for a new KISS normative sub-standard with Unpopped as one implementation. Per construct (function and launch, types and expressions, statements, memory spaces, barriers, atomics/fences/the race rule, collectives, MMA atoms, source leaves): the observable behaviour, what is language-common versus target-dependent, and the evidence (oracle, independent references, compiled-C differential, seeded mutants). It says plainly that nothing has run on a device, that the MMA kernel-body construct is a proposal, and lists eleven places where Unpopped decided something KISS might decide differently. Read at `origin/main` `aae2739`. No code changed; the version is a placeholder for the PM to allocate (0.42.19 is S1a, in review).

## `0.42.18` — the `Cluster` scope (additive, ROADMAP U4 S6b-1)

`LaunchShape` gains `cluster: Option<[u32; 3]>` (`with_cluster(dims)`, `#[serde(default)]`): blocks per cluster per axis. The grid must be a multiple of it on every axis and no extent may be zero (`BadLaunch` otherwise). **Checker:** a `Cluster` barrier, fence or atomic scope is admitted only in a launch that has cluster dimensions (`BadBarrierScope`, `SyncUnsupported`, `AtomicUnsupported` otherwise); `Thread` stays refused. **Oracle:** the blocks of one cluster rendezvous at a `Cluster` barrier (`BlockStop::Wait { scope, at }`, `grid_release` returns `Released::{Finished, Grid, Clusters}`); the barrier orders the accesses of that cluster's blocks and no others (a per-cluster epoch in `Edges` and `Stamp`); clusters are independent, so a cluster that finished does not strand another, while a block that finished inside a waiting cluster, blocks of a cluster at different barriers, or a `Device` wait beside a `Cluster` wait are `BarrierMismatch`. A `Cluster`-scope atomic or fence reaches the blocks of its cluster and no further (two atomics whose narrower scope does not contain both threads are a `DataRace`, as for `Block` and `Subgroup`). **Uniformity:** a kernel that holds a `Cluster` (or `Device`) barrier treats `BlockIdx` as divergent, so such a barrier under `if blockIdx == 0` is `BarrierDivergent`. **cpu-c:** a top-level `Cluster` barrier is spelled as a barrier of the whole grid (the `Device` phases of 0.42.14), a valid and stronger schedule since clusters are independent; it gives the bytes the cluster-scoped oracle gives for a race-free kernel. **Still open:** named barriers (a subset of a block with a participant count) wait for a consumer or S7 (`docs/options-u4-s6b-cluster-and-named-barriers.md`). **Tests:** `tests/kernel_cluster.rs` (8: in-cluster visibility, no ordering across clusters, two barriers as two phases, the divergent check, cluster-scope atomics, release/acquire at cluster scope, tiling on the x and y axes, the launch and scope checks), a cpu-c differential, two unit tests of `grid_release`; the conformance test's refused-barrier example is now a `Thread` barrier. Eleven seeded mutants caught: `cluster_of` ignoring x or y (the y one survived until a test read a block in the next cluster along y), the cluster epoch not counted, not ordering, or ordering across clusters, the `Cluster` scope reaching every block, `BlockIdx` uniform for a `Cluster` barrier, the tiling unchecked, the scope admitted without dimensions, a finished cluster released, and a stranded block not detected. The mutant script now prints an anchor that does not match instead of reporting the unmutated run.

## `0.42.16` — the union surveys re-scored (docs only)

`docs/ir-union-rescore.md`: the seven surveys of `docs/ir-union-surveys/` (519 capability rows, written 2026-10-24/25, before the kernel layer) re-asked against `origin/main` `a634b09` (0.42.14). 144 rows changed; each carries the enum variant, rule or test that makes the new answer true. Totals: Yes 29 -> 93, Partial 108 -> 168, No 370 -> 246, n/a 12; the rough coverage index (Yes plus half of Partial, over rows that are not n/a) goes from 16% to 35%. The remaining `No` rows are grouped by what they would take (asynchronous data movement and pipelines 71, matrix units 37, resources and host concerns 36, scheduling recipes 36, tile and fragment abstractions 22, numerics 19, escape hatches 17, unclassified 8; keyword-assigned, so indicative) and tied to ROADMAP rows. The surveys themselves are untouched, and rows that did not change were not re-verified. ROADMAP section 6.7 links the result. No code changed; the version is a patch bump allocated by the PM at the gate (0.42.15, the `Cluster` scope, is in review).

## `0.42.14` — the cpu-c reference runs a top-level `Device` barrier (additive, ROADMAP U4 S6a, the consumer slice)

`unpopped-cpu-c`'s serialized thread loop now spells a `Device` barrier at the top level of a launched kernel's body, as the oracle runs it (0.42.13). The barrier cuts the body into **phases of the whole grid**: every block is resident, so each local and each `Shared` array becomes an array of the grid (`l<i>_all`, `sh<i>_all`), and each phase is one pass over the blocks (`for blk2 { for blk1 { for blk0 {` ascending, as the oracle runs them) in which `l<i>` and `sh<i>` point at the block's own slice, so the block-level code of a phase is exactly what a single block used. The scalar parameters are copied into the locals once, before the first phase. **Refused, typed:** a `Device` barrier anywhere but the top level of the body (inside an `If` or a `For`), and a grid of more than 65 536 resident threads (blocks times threads). `Cluster` and named barriers are still refused (S6b). **Also:** `docs/options-u4-s6b-cluster-and-named-barriers.md`, the options note for the rest of S6 (a `Cluster` scope with cluster dimensions on `LaunchShape`, recommended next; named barriers, recommended to wait for a consumer or S7), and the conformance test that uses a barrier `CpuC` must refuse now uses a `Cluster` one (a `Device` barrier is spelled). **Tests:** `unpopped-cpu-c/tests/kernel_device_barrier_emit_agrees.rs` (7) compiles the C with a host compiler and compares the written buffers with `run_launch` byte for byte: per-block partials then a grid-wide sum, two phases with a `Block` barrier mixed in, a per-thread local and a per-block `Shared` array carried across the barrier on a 3x2 grid, a 3x2 grid with a scalar parameter, the phases and per-grid storage visible in the C, and the two refusals. Seven seeded mutants of the emitter: locals not per block, `Shared` not per block, scalar parameters never set, the resident cap removed, the cap never checked, and the block index built with the grid's x and y extents swapped are caught; setting the scalar parameters in every phase is equivalent (the value is the same each time). The swapped-extents mutant survived two tests whose phase-1 data came from global memory (colliding block slices were harmless there) and a mutant script whose anchor never matched; it is caught once the locals and `Shared` of a 3x2 grid have to persist across the barrier.

## `0.42.13` — a `Device` barrier synchronizes the grid (additive, ROADMAP U4 S6a)

`Stmt::Barrier { scope: Device, .. }` is now admitted by `check_kernel` and run by the oracle. **How:** the launch driver used to run blocks independently, one after another, so it had nothing to wait for. Now every block is resident: blocks run in order, each to its end or to a `Device` barrier (`advance` stops a block when all its threads wait at one `Device` barrier), and when every block waits at the same barrier the grid is released (`grid_released`) and each block goes on with its threads, locals and `Shared` memory as they were. With no `Device` barrier this is the old order exactly. A block that finishes while others wait, or blocks waiting at different barriers, are `BarrierMismatch`. **Memory:** a `Device` barrier orders every access before it before every access after it, whatever the blocks (`Stamp::dev_epoch`), so "each block writes its partial, `Device` barrier, every block reads all partials" is race-free and a `DataRace` without the barrier or with only a `Block` barrier. **Uniformity:** a `Device` barrier's threads are the whole grid, so a kernel that holds one treats `BlockIdx` as divergent: a `Device` barrier under `if blockIdx == 0` is `BarrierDivergent` (the same code with a `Block` barrier stays legal). **Not yet (S6b):** `Cluster` scope (the launch has no cluster dimensions) and named barriers stay `BadBarrierScope`; `unpopped-cpu-c` still spells only `Block` barriers (typed refusal), so its serialized thread loop does not run a `Device` barrier yet (a phase loop over blocks is the next consumer slice). **Tests:** `tests/kernel_device_barrier.rs` (8: the partial/sum kernel race-free with the barrier and a race without it or with a `Block` one, locals kept across the barrier, two barriers as two phases, `Block` and `Device` barriers mixed, a one-block grid, the divergent check, `Cluster` refused) and a unit test of `grid_released`; the exec-scope refusal test now admits `Device`. Seven seeded mutants caught: the device epoch not ordering, not counted, different barriers released together, threads not made runnable, `BlockIdx` taken as uniform, the device flag never set, and a block never stopping at the barrier; an eighth (a finished first block not stranding) was equivalent, and the redundant clause was removed.

## `0.42.12` — a window over any axis (additive, ROADMAP U2 D5, window half)

An `Access::Window` may pool any axis, not only the innermost: the same strided scheme as the scan of 0.42.11. The tensor splits as `(outer, k, inner)` around the pooled axis; a "row" is one `(outer, inner)` pair, its elements sit `inner` apart, the output has the pooled axis's `k_out` and every other extent unchanged, and every input must be full-width dense (no broadcast, contiguous, not reversed). **Gate:** `plan::validate_window` accepts `axis < rank` (an axis outside the rank is rejected, `outside the rank`) and checks the dense inputs for a non-innermost axis. **Oracle:** `eval_window` keeps separate input and output bases and strides. **Lift:** `lift_window_opdef` takes the pooled axis from the op: the window position loop, the bound test and the tap loads all use it, so the kernel is the same statements with the pooled axis in the window position; the launched window lift (`lift_window_opdef_launched`) grid-strides the outermost loop, whose iterations are independent output cells, so it needed no change. An axis outside the rank declines typed. `unpopped-cpu-c` needs no change. **Tests:** `tests/window_axis.rs` holds the oracle to a pool written over explicit coordinates on a `[2, 5, 4]` tensor (`Sum`, a NaN-carrying `Max`, `Mean` with both divisors; two padded, strided geometries; every axis), pins the output extents, and holds the lift to the oracle byte for byte; the compiled-C window differential gains an axis-0 case; the plan unit test that pinned the rejection now pins the admission and the out-of-rank rejection. Seven seeded mutants caught: the oracle's input base, output base, tap stride and store stride each dense, and the lift's output loop, position counter and bound test each on the innermost axis.

## `0.42.11` — a scan over any axis (additive, ROADMAP U2 D5, scan half)

An `Access::Scan` may now scan any axis of its rank, not only the innermost. The tensor splits as `(outer, k, inner)` around the scanned axis `k`; a "row" is one `(outer, inner)` pair and its element `j` sits `inner` apart in the dense layout (along the innermost axis `inner` is 1 and this is the old row walk, unchanged). **Gate:** `plan::validate_scan` accepts `axis < rank`; for a non-innermost axis every input must be full-width dense (no broadcast, contiguous, not reversed), so there is no per-column or per-row operand to index differently; an axis outside the rank is rejected (`outside the rank`). **Oracle:** `eval_scan` walks the rows with the axis stride. **Lift:** `lift_scan_opdef` builds its shape with the scanned axis as the reduced one (the counters already walk every axis), so the kernel is the same statements over a different loop nest; an axis outside the rank declines typed. **Not yet:** the launched lift (`lift_scan_opdef_launched`) still scans the innermost axis only and declines the others (`a scan over a non-innermost axis in a launched scan`), and a window over a non-innermost axis (D5's second half) is still refused. `unpopped-cpu-c` needs no change. **Tests:** `tests/scan_axis.rs` holds the oracle to a scan written over explicit coordinates on a `[2, 3, 4]` tensor (`Sum` and a NaN-carrying `Max`, every axis, all four direction and exclusivity cells), pins an axis-0 scan by hand, and holds the lift to the oracle byte for byte; the compiled-C scan differential gains an axis-0 case; two older tests that pinned the decline now pin the lift and the out-of-rank decline. Six seeded mutants caught: the oracle's base, read stride, store stride and `inner` each wrong, the lift scanning the innermost, and the launched lift accepting any axis.

## `0.42.10` — a running `Mean` scan (additive, ROADMAP U2 D1; an Unpopped extension, not KISS)

An `Access::Scan` may now combine with `Mean`: a running mean. Lane `j` of an inclusive scan is `(x_0 + ... + x_j) / (j + 1)`; lane `j` of an exclusive scan is the sum of the lanes before it over `j`, so **lane 0 is `0 / 0`, NaN** (the mean of nothing, as a mean over an empty run is). A reverse scan counts from the far end. The sum is an `f64` fold and the quotient is taken in `f64` and rounded once at the store; a fused `post` reads the mean as `Reduced(0)`. **Float only:** an integer running mean rounds, so the plan gate rejects it (`requires a float dtype`) and the lift declines (`an integer Mean`). **Not a KISS scan:** KISS-OPS-6.11-0003 takes the monoids `{sum, prod, max, min}` and a running mean is not one; whether KISS accepts, declines or decomposes it is ThinkersJournal/KISS#766, which has no reply, so the PM approved building it as an Unpopped extension (2026-11-03) to be revisited if KISS rules otherwise. **Where:** `plan::validate_scan` admits it at a float dtype (the unit test that pinned the rejection now pins the admission and the integer rejection), the oracle (`eval_scan`, `scan_identity`) folds it, and the serial scan lift (`lift_scan_opdef`) counts the elements folded so far in the `have` local and divides the `F64` accumulator by it (`scan_stored`); `unpopped-cpu-c` needs no change. The **launched** scan lift still declines it (a running mean has no declared bound). **Tests:** `tests/scan_mean.rs` holds the oracle to a running mean written in Rust (F64 and F32, all four direction and exclusivity cells), pins the corpus by hand, and holds the lift to the oracle byte for byte (NaN equal to NaN); the compiled-C scan differential gains a `Mean` case. Six seeded mutants caught: the count ignoring exclusivity, the divisor being the row length, the lift never counting, the lift storing the undivided sum, the launched decline removed, and the plan's float gate removed.

## `0.42.9` — a `Prod` window: the product of the in-bounds taps (additive, ROADMAP U2 D2; an Unpopped extension, not KISS)

An `Access::Window` may now combine with `Prod`. Each output is the product of the in-bounds taps of its window from the multiplicative identity `1`; floats multiply in `f64` and round once at the store, integers wrap at their width, and a skipped (padding) tap contributes nothing, so an edge window multiplies only the taps it has. **This is not a KISS op:** KISS has `avg_pool` and `max_pool` only; the proposal is ThinkersJournal/KISS#767. **KISS#767 was corrected:** it first said KISS left the value of a window with no in-bounds tap unstated, which was wrong; KISS-OPS-6.13-0011 makes such a window inexpressible and Unpopped already declines it (`window_taps`), so the empty product never arises and no padding ruling was needed. **Where:** `plan::validate_window` no longer asserts against `Prod` (the unit test that pinned the rejection now pins the admission), the oracle (`eval_window`) seeds the accumulator at `1` and multiplies, and the serial window lift (`lift_window_opdef`) seeds at `1`; `unpopped-cpu-c` needs no change. **Tests:** `tests/window_prod.rs` holds the oracle to a product written in Rust over the same window arithmetic (F64, F32, and an I32 product that wraps), pins an edge window, and holds the lift to the oracle byte for byte; the compiled-C window differential gains a `Prod` case. Four seeded mutants caught: the oracle's seed `0`, its float and integer products as sums, and the lift's seed `0`.

## `0.42.8` — `Stmt::Fence` runs: release and acquire fences (additive, ROADMAP U4 S5)

Option A of `docs/question-u4-memory-model.md` (CireSnave, 2026-11-03), second half. `Stmt::Fence { scope, order }` was a refused shape; the oracle now runs it by the C++ fence rules on top of 0.42.7's edges (`oracle/kernel_sync.rs`). A **release fence** (`Release`, `AcqRel`, `SeqCst`) snapshots what the thread knows; every later atomic of that thread, whatever its own order, publishes that snapshot on its element (at the narrower of the fence's and the atomic's scope). An **acquire fence** (`Acquire`, `AcqRel`, `SeqCst`) takes in the records of every atomic read the thread made before it. A fence orders no memory of its own. So relaxed message passing (write data; release fence; relaxed flag write | relaxed flag read; acquire fence; read data) is race-free, and a `DataRace` if either fence is missing, wrong (`Acquire` where `Release` belongs), on the wrong side of its atomic, or too narrow for the threads. **Checker:** a `Relaxed` fence (it would order nothing) and the scopes `Thread` and `Cluster` are `SyncUnsupported`; `Subgroup`, `Block`, `Device` are admitted. `SeqCst` is treated as `AcqRel`, as for atomics. A wait made of a relaxed read and an acquire fence is a `CrossBlockWait` like one made of an acquire atomic. `unpopped-cpu-c` emits nothing for a fence (its serial thread loop is one valid schedule). **Tests:** `tests/kernel_fence.rs` (8: the order table, no fences, fence on the wrong side of the atomic, scope reach, a fence covering every earlier read, the named wait, the checker's refusals, a fence outside a launch). Seven seeded mutants caught: release fence ignored, acquire fence taking nothing, a release snapshot that is empty, the fence's scope ignored, a fence-based wait not counted as a wait, a `Relaxed` fence admitted, and (after a test was added for it) the held reads keeping only the last one.

## `0.42.7` — atomic memory order and scope: release/acquire edges in the oracle (additive, ROADMAP U4 S4)

CireSnave ruled option A of `docs/question-u4-memory-model.md` on 2026-11-03 (KISS has no memory-model wording; nothing to defer to). `AtomicRmw` and `AtomicCas` now accept every `MemoryOrder` and the scopes `Subgroup`, `Block` and `Device`; `check_kernel` still refuses `Thread` and `Cluster` (no semantics yet), typed. **The oracle's rule (`oracle/kernel_sync.rs`):** each thread keeps a knowledge clock; a release-class atomic (`Release`, `AcqRel`, `SeqCst`) leaves a record on its element, an acquire-class atomic (`Acquire`, `AcqRel`, `SeqCst`) joins the records into the reader, and a plain access is ordered before a later one when the later thread has acquired from the earlier up to that access (on top of the existing barrier ordering). A relaxed read-modify-write leaves the record alone (it continues the release sequence). An edge needs both atomics' narrower scope to contain both threads. Message passing (write data; release the flag | acquire the flag; read data) is race-free with release/acquire and a `DataRace` with `Relaxed` on either side. **Scope:** two atomics on one element whose narrower scope does not contain both threads are a `DataRace` (a `Block`-scope atomic used by two blocks). **`SeqCst` is accepted and treated as `AcqRel`: the oracle does not model a single total order, so nothing here checks sequential consistency.** **`KernelTrap::CrossBlockWait`:** the oracle runs blocks in order, so a block that runs past the step limit after an acquire-class atomic while later blocks have not run is that wait, named, not a hang (until the `Device` barrier model of S6). A runaway in the last block, or one with no acquire, is still `StepLimit`. **Also:** `run_launch_with_limit` runs a launch with a lower per-thread statement limit than `STEP_LIMIT` (additive; it bounds the interpreter, not the language); `unpopped-cpu-c` is unchanged (its serial thread loop is one valid schedule, so order and scope change nothing there). **Tests:** `tests/kernel_atomic_order.rs` (13: the order table, a `CAS` acquire, a transitive chain, a release sequence continued by a relaxed add, two releasers on one flag, block and subgroup scopes, a narrow-scope release giving no edge to another block, the release access itself ordered, `CrossBlockWait` and its two controls); two refusal tests in `kernel_atomic_cas.rs` and `kernel_atomic_fetch.rs` now refuse `Thread`/`Cluster` instead. Ten seeded mutants caught: acquire never joining, release never recording, the edge-scope guard removed, a record forgetting the releaser's knowledge, `SeqCst` not acquiring, the atomic scope check removed, a `CrossBlockWait` without an acquire, and (after tests were added for them) an off-by-one in the knowledge test, a release clearing other threads' records, and `CrossBlockWait` in the last block.

## `0.42.6` — a `Prod` stage in `Access::RowReduce` (additive, ROADMAP U2 D3)

A row reduction may now have a `Prod` stage. It multiplies the per-element `pre` values along the row from the multiplicative identity `1` (KISS-OPS-6.11-0002) in the stage's `F64` accumulator and rounds once to the compute dtype, like the other stages; a later stage or the epilogue reads the rounded value. **Where:** the plan gate no longer asserts against it (`plan::validate_row_reduce`), the oracle folds it (`eval_row_reduce`: the accumulator starts at `1`, `acc *= e`), and the serial kernel lift (`lift_row_reduce_opdef`) seeds the accumulator at `1`. `unpopped-cpu-c` needs no change (the lifted kernel is ordinary statements). **Still declined:** the launched lift (`lift_row_reduce_opdef_launched`) refuses a `Prod` stage with a new typed reason, because a float product is not order-independent and no bound is declared; the reason string changed from "a Prod or empty stage list" to "an empty stage list" (serial) and the launched text above, so a caller matching the old string needs the new one. **Tests:** `tests/row_reduce_prod.rs` holds the oracle to a product written in Rust (mixed signs, a zero giving `-0`, `0 * inf` giving NaN, all ones, an `F32` overflow, a one-element row so the identity cannot leak), the serial lift to the oracle byte for byte in `F32` and `F64`, a product feeding a later epilogue, and the launched decline; `unpopped-cpu-c/tests/kernel_emit_row_reduce_agrees.rs` compares the compiled C with the interpreter. Four seeded mutants caught: the oracle folding `Prod` as a sum, the oracle's seed `0`, the lift's seed `0`, and the launched decline removed.

## `0.42.5` — `ArgMin` follows KISS: it never selects a NaN while a number exists (fix, ROADMAP U5 P3)

**The 0.42.2 `ArgMin` was a defect against KISS.** It used PyTorch's rule (NaN is the minimum, so the first NaN's index) where KISS-OPS-6.11-0007 makes NaN the **greatest** value in the total order, so `argmin` never selects a NaN while a non-NaN exists. The first ruling (PM, 2026-11-02) chose PyTorch's rule before KISS was read; the PM then ruled that KISS wins. Now: `ArgMax` is the first NaN's index if any (unchanged); `ArgMin` is the first index of the smallest non-NaN, and index 0 when every element is NaN (both ops). Integers are unchanged. **Behaviour change** for any caller that relied on `ArgMin` returning a NaN's index: a row `[1, NaN, 0]` gave `1`, now gives `2`; a row `[NaN, 1, 2]` gave `0`, now `1`. It is a fix to the stated contract, so the version is a patch. `ArgMax`/`ArgMin` are KISS-Ops's `argmax`/`argmin` (the §6.13 table, read from `sort_network`'s index output), so KISS-OPS-6.11-0007 is their authority. **Tests:** the corpus gains an all-NaN row, the reference fold is written from the total order (`greater`), and the kernel loop of `tests/support/arg_loop.rs` and the C differential (`kernel_emit_argmax_agrees.rs`) state the same order. Five seeded mutants of `arg_fold` are caught: the old PyTorch `ArgMin`, `ArgMin` that skips NaN but never replaces a NaN seed, one that never skips an element NaN, and two on `ArgMax` (a NaN accumulator not final; a NaN element not winning).

## `0.42.4` — the cpu-c reference for collectives; a call that reaches a collective is a barrier site (additive, ROADMAP U5 P5)

`unpopped-cpu-c`'s serialized thread loop now spells `Stmt::Collective`. It is a **block-level** construct between two thread loops, like a barrier: the loop before it stores every thread's contribution (`cv<n>[t]`, and the shuffle or broadcast operand `co<n>[t]`), the construct walks the groups of the scope (the whole block, or consecutive runs of the launch's subgroup size: its required minimum, which is the oracle's default; `1` when the launch states none) and writes each lane's result into `l<result>[lane]`, and the next loop starts from there. The rules are the oracle's (0.42.3), written out in C: the shuffles with CUDA's keep-your-own-value rule and the wrapping index, the votes and the ballot mask, the left-fold reduce and the inclusive and exclusive scans (lane 0 of an exclusive scan gets the identity; the first element seeds otherwise), integers wrapping through the unsigned type and floats rounding in their own width. A kernel with a collective takes the barrier's restrictions (no `Return`, no `While` holding it, uniform control: `check_kernel` already proved it), a collective inside a function other than the launched entry is refused typed, and so is a collective value outside `I32`, `I64`, `U32`, `U64`, `F32`, `F64`. **In `unpopped`:** a call that reaches a collective is a barrier site, as one that reaches a `Barrier` is (`kernel_module.rs`): under divergent control, or with a divergent scalar argument, it is `BarrierDivergent`; the oracle runs a collective in a callee (`tests/kernel_collective.rs`). **Tests:** `unpopped-cpu-c/tests/kernel_collective_emit_agrees.rs` compiles the C with a host compiler and compares the written buffer with `run_launch` byte for byte: integer reductions and scans in every op and both scopes (exclusive scans of `Sum`, `And`, `Or`, `Xor`), float reductions and scans with a NaN in one subgroup, every shuffle mode with operands 0, 1, 3 and 9 and broadcasts of lanes 0, 2, 6 and 11 (past the scope wraps), votes and the ballot mask including a float predicate; a seeded `+`-to-`-` mutant of the C is noticed. Ten seeded mutants of the new code are caught: in the emitter, the exclusive scan's lane 0 not the identity, its update not skipped for lane 0, the `And` identity not all ones, shuffle `Up` excluding lane 0, `Idx` clamping instead of wrapping, the ballot's bit order reversed (it survived until the vote input had zeros in an asymmetric pattern), the float `Max` ignoring a NaN element, the float `Sum` as a difference, a subgroup grouped as the block; and in `unpopped`, a call that reaches a collective not counted as a barrier site.

## `0.42.3` — collectives run: shuffle, vote, ballot, reduce, scan, broadcast (additive, ROADMAP U5 P5)

PM ruling 2026-11-02 (4): `Stmt::Collective { kind, value, result, scope }`, a shape since 0.42.0, now runs in the oracle with **full participation** of a `Subgroup` or a `Block`. A collective is a rendezvous like a barrier that also exchanges registers: the resumable evaluator stops each thread at the statement holding its contribution, and when every thread of the scope has arrived the launch driver computes every lane's result (`oracle/kernel_collective.rs`) and puts it in the lane's `result` local. **The rules** (lane `i` of `n`, ascending): `Shuffle` `Idx` and `Broadcast` read lane `operand mod n`; `Up`/`Down`/`Xor` read lane `i - operand`, `i + operand`, `i ^ operand` or keep the lane's own value when there is no such lane (CUDA's rule); `Vote` `All`/`Any` give `1`/`0` as a `U32`, `Ballot` the `U32` mask (a ballot is a `Subgroup` operation, at most 32 lanes); `Reduce` gives every lane the left fold in lane order; `Scan` the prefix fold, inclusive or exclusive (lane 0 of an exclusive scan gets the identity of `Sum`/`And`/`Or`/`Xor`; an exclusive `Max`/`Min` has none and is refused). The folds are `Sum`, `Max`, `Min` and, on integers, `And`, `Or`, `Xor`; a float `Sum` is the left fold, which a device may reassociate (compare within a tolerance). **A collective does not order memory**: it exchanges registers and is not a fence, so accesses on either side of it race unless a barrier orders them. **Checker (`kernel_check/collective.rs`):** a launch is required (`ExecWithoutLaunch`), the scope is `Subgroup` or `Block` (`BadBarrierScope`), the value is an integer, `F32` or `F64`, a lane operand is an integer (`TypeMismatch`), the result local is declared by the statement (a vote's is `U32`, the rest take the value's type), and unsupported combinations are `CollectiveUnsupported` (a ballot over a block, an exclusive `Max`/`Min`, a bit fold of floats, `Mean`, `Prod`, `MaxNum`, `ArgMax`). **A collective must sit in uniform control**, like a barrier (`kernel_uniform.rs`): under a divergent `If` or loop it is `BarrierDivergent`, and a lane that finishes while its scope waits is `BarrierMismatch`. **Not yet:** `unpopped-cpu-c` refuses a collective (typed: "a Collective is not emitted yet"); the block-level C reference is the next slice. **Tests:** `tests/kernel_collective.rs` (11: reductions, bit folds, scans, shuffles of every mode with an out-of-range lane, broadcasts, votes and the ballot mask, a float sum, the divergent refusal, a collective that does not order memory with a barrier control, the checker's refusals), with every expected array written out by hand. Ten seeded mutants caught: a shuffle `Up` as `Down`, `Idx` and `Broadcast` without the wrap, an exclusive scan that includes the lane, `And` as `Or`, the ballot's bit order, a collective that orders memory, a ballot over a block admitted, an exclusive `Max` admitted, and the divergence check removed. Three functions the slice pushed over the 50-line bound were reshaped (`collective_stop`, `status_after`, a split test).

## `0.42.2` — `ArgMax` / `ArgMin` reduce to an `I64` index (additive, ROADMAP U5 P3)

PM ruling 2026-11-02 (3): `ReduceOp::ArgMax` and `ArgMin` in an `Access::Reduction` give the **index** of the extreme, by PyTorch's rule: the **first** index of the extreme, and **NaN wins** (it is the maximum and the minimum for this purpose), so a run with a NaN gives the first NaN's index, and a run of only NaN gives index `0`. The index is the position in the row-major order of the reduced axes: one axis gives that axis's coordinate; all axes give the flat index (`torch.argmax(x)` with no `dim`). The op names its output: `OpDef::out_dtype` must be `Some(I64)` and the post the identity (`assert_valid_out_dtype` admits an `I64` output for `Sum`, `ArgMax` and `ArgMin`). **Refused, typed:** at the plan gate (`PlanError::InadmissibleOpAtDtype`) an arg-reduction without the `I64` output or with a non-identity post, and in a scan, window or row reduction; in the kernel lift (`LiftDecline::UnsupportedForm`) always, because the kernel layer states it as a loop over locals (`tests/support/arg_loop.rs`, the sequential form); the KISS-Ops recipe has no token for it (an honest miss). **Tests:** `tests/reduce_argmax.rs` (5: the oracle against a fold written in Rust at `F32`, `F64` and `I32`, over the last axis, axis 0 and all axes; the corpus pins ties, a NaN first, a NaN in the middle with a larger number after it, an all-negative row and equal zeros; the gate declines; the lift declines; a kernel loop run by the interpreter equals the oracle) and `unpopped-cpu-c/tests/kernel_emit_argmax_agrees.rs` (the same loop compiled as C equals the interpreter, and a seeded `>`-to-`<` flip of the C is noticed). Six seeded mutants caught: the NaN accumulator not final, a NaN element never taking over, ties won by the last element at an integer and at a float dtype, the `I64`-output decline removed, and a scan admitted at the gate. `scan_identity` stays under the 50-line bound.

## `0.42.1` — `MaxNum` / `MinNum` reduce ignoring NaN (additive, ROADMAP U5 P1)

PM ruling 2026-11-02 (5): `ReduceOp::MaxNum` and `MinNum`, shapes since 0.42.0, now fold. A NaN element is skipped and a NaN accumulator is replaced by the next element, so a run with a number in it gives that number's extremum and a run of **only NaN gives NaN** (unlike `Max`/`Min`, which keep a NaN once seen). The first element seeds the fold and a strictly better one replaces it, so the first of two equal zeros is kept, as `Max` keeps it. They reduce in an `Access::Reduction` only, at a float dtype (an integer has no NaN, so there they equal `Max`/`Min`); a scan, window or row reduction declines at the plan gate (`PlanError::InadmissibleOpAtDtype`, "folds in a Reduction only") and in the kernel lift. The kernel lift spells the rule as `if !have { acc = v } else if acc != acc { acc = v } else if v == v && better(v, acc) { acc = v }`, and `unpopped-cpu-c` compiles it. **Tests:** `tests/reduce_maxnum.rs` (5: the oracle against an independent fold at `F32` and `F64` on both axes, the lifted kernel against the oracle, a corpus that tells a NaN-ignoring extremum from a NaN-propagating one, the integer equivalence with `Max`/`Min`, the scan declines); `unpopped-cpu-c/tests/kernel_emit_maxnum_reduce_agrees.rs` (the emitted C against the interpreter, and a seeded flip of the C's NaN probe is noticed). Five seeded mutants caught: the oracle treating `MaxNum` as `Max`, its NaN seed not replaced, the lift treating it as `Max`, the lift probing the element where it means the accumulator, and a scan admitted at the gate (a sixth, dropping the element's `!is_nan` guard, is equivalent: a NaN never compares better). `extremum_step` and `fold_step` stay under the 50-line bound (`nan_arm`, `extremum_fold`).

## `0.42.0` — BREAKING: `ReduceOp` gains `And`/`Or`/`Xor` (built) and `MaxNum`/`MinNum` (shapes); `Stmt::Collective` (shape) (ROADMAP U5)

PM rulings 2026-11-02 on `docs/options-u5-collectives-and-richer-reductions.md`: (1) option A, staged like U4: break the exhaustive enums **once**, behind typed refusals, so the later slices are additive; (2) a custom combine stays at the kernel layer; (3) arg-reductions follow PyTorch; (4) collectives v1 take full participation of the scope; (5) `MaxNum`/`MinNum` ignore NaN and an all-NaN run gives NaN. This release is the one break.

**What breaks** (every exhaustive `match` on these types): `ReduceOp` gains `And`, `Or`, `Xor`, `MaxNum`, `MinNum`, and **is now `#[non_exhaustive]`**, so this is the last release in which adding a `ReduceOp` variant breaks a downstream `match` (a downstream match needs a `_` arm from now on). `Stmt` gains `Collective { kind, value, result, scope }` (`Stmt` stays exhaustive on purpose); new `CollectiveKind` (`Shuffle`, `Vote`, `Reduce`, `Scan`, `Broadcast`), `ShuffleMode`, `VoteKind`, all `#[non_exhaustive]`; `KernelError` gains `CollectiveUnsupported`. **Who is affected:** any crate that matches `ReduceOp` or `Stmt` exhaustively. Read from sibling `origin/main` 2026-11-02, all of which pin `unpopped` 0.40.0 and so see nothing until they upgrade: baracuda has 16 files naming `ReduceOp::`, vulkane 1 file naming it and 1 using `Stmt::`, bonefish 2 files using `Stmt::`; Fuel depends on `unpopped-vocab` only.

**What is new and runs:** `ReduceOp::{And, Or, Xor}` reduce **integers** in an `Access::Reduction` (any axes, `keepdim`; `I32`, `I64`, `U32`, `U64` in the oracle; `I32` and `I64` lift into the kernel layer and compile through `unpopped-cpu-c`). The fold is seeded by the first element, with no identity literal (as `Max`/`Min`). The kernel interpreter gains integer `BinaryOp::{BitAnd, BitOr, BitXor}` (it had none; the CUDA and Slang parsers need them too). **What is refused, typed:** at the plan gate (`PlanError::InadmissibleOpAtDtype`) `MaxNum`/`MinNum` for every access and dtype, a bit op on a float dtype, and a bit op in a scan, window or row reduction; in the kernel lift (`LiftDecline::UnsupportedForm`) the same, and a bit op in a launched (block-parallel) reduction; at `check_kernel`, `Stmt::Collective` (`KernelError::CollectiveUnsupported`); the KISS-Ops recipe has no token for the new ops (an honest miss, as for `Mean`). **Tests:** `tests/reduce_bit_ops.rs` (6: the oracle against an independent fold at four integer widths and both axes, the lifted kernel against the oracle, the corpus tells the three ops apart, the declines), `tests/kernel_collective_shape.rs` (3: every kind refused, the same kernel without it checks, every kind round-trips), and `unpopped-cpu-c/tests/kernel_emit_bit_reduce_agrees.rs` (the emitted C against the interpreter, 12 reductions, and a seeded `^`-to-`|` mutant of the C is noticed). Nine seeded mutants of the new code are caught: the oracle's `Xor` as `Or`, its seed as `0`, the lift's `Xor` as `Or`, its fold that never accumulates, the kernel interpreter's `BitAnd` as `BitOr`, a float bit reduction admitted, a bit scan admitted, `MaxNum` admitted, `Collective` admitted. Source fns touched stay under the 50-line bound (`fold_step` and `eval_int` were reshaped for it).

## `0.41.15` — docs: an options note for U5, collectives and richer reductions (ROADMAP U5)

`docs/options-u5-collectives-and-richer-reductions.md`: five pieces (P1 NaN-ignoring `MaxNum`/`MinNum`, P2 `And`/`Or`/`Xor` and `Any`/`All` reductions, P3 `Access::ArgReduce`, P4 tuple-state custom combines, P5 collectives as kernel-layer statements that rendezvous like a barrier), what exists today, which of them break an exhaustive enum (`ReduceOp` and `Stmt` do, `Access` does not), three options, a recommendation (A, staged like U4: one breaking release behind typed refusals, then additive slices; P4 stays at the kernel layer), and the five rulings it needs. Docs only; no code changes.

## `0.41.14` — docs: the U4 memory-model question for CireSnave (ROADMAP U4 S4, S5)

`docs/question-u4-memory-model.md`: what the oracle should promise for atomic `order` and `scope` and for `Stmt::Fence`, which cannot be built without a decision. It states what the oracle checks today (a race rule over barrier epochs, not a simulation of a machine), then three options: **A**, race classification with release/acquire happens-before edges (recommended, the default if no answer), **B**, ignore order, **C**, explore litmus outcomes; and two sub-questions A forces (a wait on another block's write as a typed trap rather than a hang until the `Device` barrier model exists; `SeqCst` accepted and treated as `AcqRel`, said so in the CHANGELOG). Docs only; no code changes.

## `0.41.13` — a `Shared` array passed to a callee in the C reference (additive, ROADMAP U3 R1 / M5c of `docs/options-u3-calls.md`)

`unpopped_cpu_c::lower_module` now spells `CallArg::Alloc(a)` under a launch: the block's `Shared` array `sh<a>` is passed to the callee's four buffer arguments with a declared one-dimensional shape `{N}` and stride `{1}` (`ash<a>`, `ast<a>`, base `0`), so the callee's buffer parameter indexes it, bounds-checked, exactly as it indexes a buffer, and a callee forwards it to a nested callee as it forwards any buffer. The two helper arrays are declared only when a module is being lowered: a lone launched kernel's C is byte-identical to 0.41.12 (pinned by `a_lone_launched_kernel_gets_no_array_shape_declarations`). Still refused, typed: a `Private` array (a launched entry's allocations are `Shared` only in this emitter, as before), and a callee that holds a barrier under a launch. `tests/kernel_module_emit_agrees.rs` (13 tests) compiles the C and compares it with `run_launch_module` byte for byte: a `Shared` array filled in one callee and mirrored by another across an entry barrier, the array forwarded through a nested callee with the array named rather than the first (a decoy at index 0). Six seeded mutants caught: in the emitter (the array index lost, the helper arrays always declared, never declared) and on the generated C (the extent one too short, the stride zero, the output buffer passed in place of the array).

## `0.41.12` — allocations through a call: a callee's array parameter is the entry's `Shared`/`Private` array (additive, ROADMAP U3 R1 / M4c of `docs/options-u3-calls.md`)

`CallArg::Alloc(a)` now runs: the entry's allocation `a` (`Shared` or `Private`) is passed **by alias** to a callee's buffer parameter, and the callee reads and writes it through `Place::Logical` with its one index. The access is the allocation's, not a global buffer's: bounds-checked against the allocation (`AllocOutOfBounds`), logged by the race detector (`DataRace`, `UninitializedShared`), ordered by the entry's or the callee's barriers, and an atomic on it is the `Shared` atomic. A `Private` array stays the calling thread's own. A callee may forward its array parameter to a nested callee with `CallArg::Buffer`. In the oracle a buffer-map entry at or above `ALLOC_BIND` names an allocation (`ALLOC_BIND + index`), and the allocation operations resolve the index against the **entry's** declarations (`Machine::alloc_k`), since only the entry declares any. **The checker is stricter for this case:** an allocation is passed only to a parameter that declares rank 1 (`Some(1)`), because the callee's index count must be known (a `None` or a higher rank is `CallArgMismatch`); element type and the `Constant`-to-writer rule are as before. Still refused: a callee that declares allocations of its own (run time `Unsupported`), and `unpopped-cpu-c` still refuses an allocation passed to a callee (the C pass-through is the next slice). `tests/kernel_module_alloc.rs` (8 tests: a `Shared` array filled in one call and read in another across an entry barrier, the same without the barrier is caught, a barrier inside the callee orders the array, forwarding to a nested callee, a `Private` array is the calling thread's own, an access past the allocation, the array named rather than the first, the checker's element/rank matching); six seeded mutants (stores or loads not routed to the allocation, the binding's index lost, allocations looked up in the callee instead of the entry, an unknown-rank parameter accepted, forwarding that drops the binding) are all caught.

## `0.41.11` — `lower_module` under a launch: a barrier-free callee reads the thread's ids (additive, ROADMAP U3 R1 / M5b of `docs/options-u3-calls.md`)

`unpopped_cpu_c::lower_module` now accepts a **launched** entry. The entry is spelled by the serialized thread loop as before; a `Call` there is a C call inside one thread's run, and a callee (a plain C function, its locals ordinary variables) takes the thread's ids as six extra arguments `ti0..2`, `bi0..2` (before `res`): `ThreadIdx` and `BlockIdx` are those arguments, `BlockDim` and `GridDim` constants of the entry's launch, and a callee that calls another forwards its own. A function other than the entry that holds a `Barrier` is **refused up front, by name** ("called under a launch"): the thread loop cannot be split in the middle of a C function. Unlaunched modules and `lower_kernel` are unchanged. `tests/kernel_module_emit_agrees.rs` (9 tests now) compiles the C and compares it with `run_launch_module` byte for byte: a callee reading `BlockIdx`/`ThreadIdx`/`BlockDim` on a 2-block grid, ids forwarded through a nested call with a per-thread result, and the barrier-callee refusal. Mutants caught: in the emitter, the ids not passed, the signature without ids, the callee spelling ids from the loop's variables, the barrier-callee guard removed, and the forwarded ids wrong; on the generated C, the thread id passed as the block id, the block id dropped, the block size constant wrong. `function_text` stays under the 50-line bound.

## `0.41.10` — `unpopped_cpu_c::lower_module`: a module as serial C (additive, ROADMAP U3 R1 / M5 of `docs/options-u3-calls.md`)

New public function `unpopped_cpu_c::lower_module(&Module, entry) -> Result<GeneratedKernel, LowerError>`. Each function the entry reaches through calls is a `static int unpopped_cpu_fn_<i>_<name>(...)` in dependency order, taking the arguments a lone kernel takes (a buffer is four C arguments, a scalar one) plus `T* res` when it has a result; the entry is `int unpopped_cpu_kernel_<name>(...)`, spelled as a lone kernel is. A `Call` is a C call: a buffer argument forwards the caller's four arguments (the alias), a scalar is cast and passed by value, the callee's result local is declared up front and stored through `res` on every exit (an early `Return` included), and a nonzero status (`1` out of bounds, `2` zero step) is returned at once by the caller. `CpuC::lower_kernel` of a lone kernel is byte-identical to 0.41.9 (the function text moved into a shared helper). **Refused, typed:** a launched entry (the next slice: ids and the thread loop through a call), an allocation passed to a callee, anything `check_module` refuses. `tests/kernel_module_emit_agrees.rs` compiles the C with a host compiler and compares the written buffers with `oracle::run_module` byte for byte (a result through `res`, a buffer forwarded to the caller buffer it names, an early `Return` with callee locals isolated, a callee out of bounds returning status 1 from the entry, the launched refusal). Eight seeded mutants are caught: five in the emitter (the wrong buffer forwarded, the status check dropped, the result not bound to the caller's local, the scalar argument not passed, the result declaration not hoisted) and three on the generated C (the early-`Return` store, the final store, a swallowed status).

## `0.41.9` — a barrier inside a callee: a call that reaches one is a barrier site (additive, ROADMAP U3 R1 / M4b of `docs/options-u3-calls.md`, Q6)

`check_module` no longer refuses a callee that holds a `Barrier`. Instead it computes, per function, whether it **reaches** a barrier (directly or through its own calls, over the acyclic graph), and the uniformity rule treats a `Call` of such a function as a barrier site: it is `BarrierDivergent` under control that may differ between threads (a divergent `If`/`While`/`For` bound, or after a divergent `Break`/`Continue`), and also when a **scalar argument is divergent**, because the callee was checked assuming its scalar parameters are uniform (buffer and allocation arguments are not scalars; a load from one is divergent in the callee as everywhere). A call's result local is still marked divergent. At run time a thread stopped at a barrier inside a callee keeps its function, locals and buffer binding in the thread's state (`ThreadRun.fn_idx`/`bufmap`) and resumes inside that callee after the release, then returns to the caller's frame. Still refused, typed: a callee with its own launch or a `Shared`/`Constant` allocation, an allocation passed to or declared by a callee (the next slice), a barrier in a callee of an unlaunched entry (`ExecWithoutLaunch`). `tests/kernel_module_barrier.rs` (8 tests: the barrier in a callee orders the block, without it the same exchange is a `DataRace`, the caller's frame and a swapped buffer binding survive the callee's barrier, a divergent call site, a divergent scalar argument, the property is transitive through a nested call, an unlaunched entry); seven seeded mutants (not transitive, the call not a barrier site, a divergent argument ignored, no call reaching a barrier, the thread's function not persisted across the barrier, its buffer binding not persisted, a barrier inside an `If`/loop of a callee not counted) are all caught. `kernel_uniform::stmt` (62 lines since 0.40) is split into `defines`/`control` to stay under the 50-line bound; the 24 `tests/kernel_barrier.rs` tests, which pin the analysis' known over-rejections, pass unchanged.

## `0.41.8` — `run_launch_module`: a launched entry calls plain functions (additive, ROADMAP U3 R1 / M4a of `docs/options-u3-calls.md`)

`oracle::kernel_launch::run_launch_module(&Module, entry, args)` runs a launched entry whose threads `Call` the module's plain functions. In the resumable evaluator a call is a `Frame::Call` on the thread's own stack holding the caller's locals, buffer map and function index; the callee's body runs above it, and the frame is popped (and the result copied to the caller's `result` local) when the body ends or the callee `Return`s (a `Return` now unwinds to the nearest call frame, or ends the thread when there is none). A callee runs on the calling thread inside the entry's launch: it reads that thread's `ThreadIdx`/`BlockIdx` and its accesses are logged against that thread and the caller's buffer, so the race detector sees through a call (N threads writing one element inside a callee are a `DataRace`). `check_module` now checks a callee reachable from a launched entry inside the entry's launch (its ids are legal), still refusing a callee that declares its own launch, a `Shared`/`Constant` allocation or a `Barrier`. **Typed refusals still:** a callee holding a barrier (slice M4b: the transitive divergence check), an allocation passed to a callee or declared by one. `tests/kernel_module_launch.rs` (7 tests: each thread calls with its own arguments, a callee reads the calling thread's ids, the race detector through a call, nested calls and a `Return` ending only the callee, the caller's locals and buffer map after a call, a result reaching the calling thread, the barrier-in-callee refusal); seven seeded mutants (the buffer map not restored, the caller's locals not restored, `Return` ending the whole thread, the callee not checked inside the launch, the callee's locals not fresh, the result dropped, the module not handed to the resumable evaluator) are all caught. `kernel_resume` `step` is refactored (`break_loop`, `unwind_to_call`, `Frame::seq`, `call_stmt`) to stay under the 50-line bound; the existing resume and launch tests pass unchanged.

## `0.41.7` — `oracle::run_module` runs a `Call` (additive, ROADMAP U3 R1 / M3 of `docs/options-u3-calls.md`)

`oracle::run_module(&Module, entry, args)` checks the module with `check_module` (`Malformed` otherwise) and runs the entry on the recursive evaluator, where a `Stmt::Call` is a **frame swap on the one machine**: the callee's function, fresh locals and buffer map replace the caller's for the length of its body and are restored after. A scalar argument is evaluated in the caller's frame and bound by value to the callee's parameter local; a buffer argument binds the callee's parameter to the *root* argument the caller's buffer already names, so a buffer passed by alias is the caller's buffer (two parameters may alias one buffer, a nested call composes); a `Return` ends the callee only; the callee's result local is copied to the caller's `result` local. **Typed refusals:** a launched entry or one with allocations (`Unsupported`, slice M4), an allocation passed to a callee, a callee with allocations, a callee that returns without setting its result local (a definite-assignment check is a later refinement). `run_kernel` and `run_launch` still refuse a `Call`. `tests/kernel_module_run.rs` (6 tests: a result back to the caller, a buffer passed to the caller buffer it names, two parameters aliasing one buffer, callee locals isolated and an early `Return`, a call in a loop and a call inside a call, the typed refusals); six seeded mutants (buffer map ignored, callee locals not fresh, result not copied back, caller frame not restored, scalar argument not bound, the module not checked first) are all caught.

## `0.41.6` — `check_module`: a `Call` is well formed inside a `Module` (additive, ROADMAP U3 R1 / M2 of `docs/options-u3-calls.md`)

`kernel::check_module(&Module, entry)` checks every function of the module and resolves each `Stmt::Call` against its callee: the callee exists (`BadCallee`), the argument count matches (`CallArity`), each argument fits its parameter (`CallArgMismatch`: a scalar of the parameter's type; a caller buffer of the same element type, a compatible rank and a role the caller really has, so a callee cannot write a caller's read-only buffer; a caller allocation of the same element type, one-dimensional, a `Constant` one only for a callee that does not write it), the result local gets the callee's result type (a `TypeMismatch` otherwise, `CallResult` if the callee has none), and the call graph is acyclic (`CallCycle`). A called function is a plain function: no launch shape, no `Shared`/`Constant` allocation, no `Barrier` (a barrier through a call needs the transitive divergence check, slice M4): `CallUnsupported`. **Unchanged:** `check_kernel` alone still refuses a `Call`, and every evaluator (`run_kernel`, `run_launch`) still refuses a module's call, so nothing runs a call yet (M3). New `KernelError` variants `BadCallee`, `CallArity`, `CallArgMismatch`, `CallCycle`, `CallResult` (the enum is non-exhaustive). `tests/kernel_module_check.rs` (8 tests) pins each rule; eight seeded mutants (arity unchecked, cycle undetected, role unchecked, element type unchecked, a result asked of a void callee, a launched callee admitted, only the entry checked, a scalar argument's type unchecked) are all caught.

## `0.41.5` — docs: an options note for `Stmt::Call` and the module checker (ROADMAP U3 R1)

`docs/options-u3-calls.md`: what a callee is at check time and at run time, as recommendations the PM rules on (Q1 the result local, Q2 how a buffer reaches a callee, Q3 the callee's environment, Q4 how a call runs, Q5 what `check_module` checks, Q6 barriers and divergence through a call, Q7 effects) and the five slices M1-M5 with the mutants each must seed. **Q2 is ruled (option b) and built** in 0.41.4 (#244), before the 0.41.x publish. Docs only; no code changes.

## `0.41.4` — `Stmt::Call` takes `CallArg`s; `KernelFn.result` (shape only, ROADMAP U3 R1 / M1 of `docs/options-u3-calls.md`)

PM ruling 2026-11-02 (Q2 of the calls options note, option b): reshape the `Call` of 0.41.0 **before 0.41.x is published**, so 0.41.0's `args: Vec<ScalarExpr>` is never on crates.io and no consumer sees a break. `Stmt::Call.args` is now `Vec<CallArg>`, `CallArg::{Scalar(ScalarExpr), Buffer(u32), Alloc(u32)}`: a scalar by value, a caller buffer parameter or allocation by alias, matched to the callee's parameter of the same position. `KernelFn` gains `result: Option<u32>` (the local a `Call` receives; `#[serde(default)]`, so an older serialized kernel reads back with `None`) and `with_result`. **Still shape only:** `check_kernel` refuses a `Call` with `CallUnsupported` (pinned by `tests/kernel_call_shape.rs`, which also round-trips a module with all three argument kinds and the `result` default). Nothing in a sibling builds a `Call` (the vulkane and bonefish tests build only `Barrier`/`AtomicRmw`). The module checker, the call semantics and the evaluators are slices M2-M5 of the note.

## `0.41.3` — `Stmt::AtomicCas` runs (additive, ROADMAP U4 slice S3)

The shape was reserved by 0.41.0 and refused; this lifts the refusal. `AtomicCas { place, expected, desired, old, order, scope }` compares the element with `expected` and, if they are equal, stores `desired`, as one indivisible step, and hands back the value it found (into `old`) either way. **Admitted form:** an integer element of a `ReadWrite` global buffer or a `Shared` allocation, `expected` and `desired` of the element's type (a mismatch is `TypeMismatch`), `Relaxed` order and `Device` scope; a float element is `AtomicUnsupported` (compare-and-exchange is integer only). The oracle logs a CAS as an atomic access (it races only with a plain access) and runs it in the global and the `Shared` path through one step function shared with `AtomicRmw`; `unpopped-cpu-c`'s serialized thread loop spells it as `if (old == e) place = d;` and binds `old`. Which thread wins a contended CAS depends on arrival order, which the IR does not specify, so `tests/kernel_atomic_cas.rs` asserts properties (exactly one winner of a CAS from zero, a failed CAS leaves the cell and reports what it found, a CAS loop reaches the maximum, a `Shared` CAS has one winner, the refusals, a CAS racing a plain access), and two cpu-c tests compile the C and compare it with the interpreter. Also: `check_stmt`, `check_atomic` and `eval_int` (80, 57 and 70 lines since 0.41.0) are split under the 50-line Codacy bound. Seeded mutants all caught: a CAS that always writes, never writes, compares with `!=`, leaves `old` unstored (the interpreter and the C), and a checker that accepts a float element; the emitted C with `==` as `!=` is caught; the emitted C with `old` unbound is caught by the CAS-loop test not terminating (a hang, not an assertion).

## `0.41.2` — the cpu-c kernel emitter spells a launch as a serialized thread loop; `Lowering::exec` (additive, ROADMAP U4 / I3)

PM-approved 2026-11-02 (the first consumer of the U4 shapes and the I3 oracle promise). `unpopped-cpu-c`'s `lower_kernel` now accepts a kernel with a `LaunchShape`: serial C that runs the blocks in order and, in a block, the threads in order, splitting the thread loop at each `Block` barrier (a barrier-holding `If` or counted `For` is spelled once around its phases, its condition or bounds read at thread 0 because `check_kernel` proved them uniform). `Shared` allocations are per-block arrays; execution ids `ThreadIdx`, `BlockIdx`, `BlockDim`, `GridDim` are spelled from the launch; an `AtomicRmw` (all seven combines, the old value) is a plain read-modify-write. New in `unpopped`: `Lowering::exec` / `LoweringBuilder::exec`, the seam that spells an `Exec` leaf (default: the same in-band decline as before, so no existing emitter changes). **Refused, typed:** a barrier of any scope but `Block`; a `While` holding a barrier; a `Break`/`Continue` leaving, or a `Return` in a kernel with, a barrier; `Private`/`Constant`/dynamic allocations; subgroup and cluster ids; a block over 1024 threads; `AtomicCas`, `Fence`, `Call`. An unlaunched kernel's C is byte-identical to 0.41.0. `tests/kernel_launch_emit_agrees.rs` compiles the C with a host C compiler and compares written buffers with `run_launch` byte for byte (a shared histogram with two barriers, the block-tree reductions across six geometries, 3-D ids, the six associative combines, the launched integer scatters, an early `Return`) and holds `Exchange` and the fetch-add old value to the properties the IR promises, not to the order. `tests/lowering_exec_seam.rs` pins the seam. Eight seeded mutants (the barrier not splitting the loop, `BlockIdx` as `ThreadIdx`, a dropped modulo in `ThreadIdx(1)`, atomic `Max` as `Min`, the old value not stored, the block `For` counter not broadcast, `Return` leaving the kernel, the seam ignoring the id) are all caught.

## `0.41.1` — docs: an options note for what remains of U3 (ROADMAP U3)

`docs/options-u3-remainder.md`: the four open items (`Stmt::Call` with a module type, pointer values, texture and tensor spaces with texture/sampler resources, `ProgramModel`/`BlockSpmd` lowering), each with the gap rows that list it, the measured need, and whether it breaks. Recommendation, ruled the same day: `Stmt::Call` rides U4's one `Stmt` break (built in 0.41.0, #241); a `program_model` declaration field is additive; pointer values (byte-offset load and store first), textures (Fuel measured 0 of 192 sampling kernels) and the `BlockSpmd` lowering wait for a consumer. The note itself changes no code.

## `0.41.0` — BREAKING: `Stmt` takes the shapes U4 needs once; `AtomicRmw` returns the old value and has `And`/`Or`/`Xor`/`Exchange` (ROADMAP U4 slice 1)

PM ruling 2026-11-02 on `docs/options-u4-sync-and-atomics.md`: option B, break `Stmt` once and stage the semantics behind typed refusals, so the later U4 slices are additive. **What breaks** (every exhaustive `match` on `Stmt`, every constructor of these statements): `Stmt::Barrier` gains `kind: BarrierKind` (only `Full`); `Stmt::AtomicRmw` gains `old: Option<u32>`, `order: MemoryOrder` and `scope: Scope`; `Stmt::AtomicCas { place, expected, desired, old, order, scope }` and `Stmt::Fence { scope, order }` are new. `Stmt::barrier(scope)` and `Stmt::atomic_rmw(op, place, value)` build the 0.40 forms. **What is new and runs:** an `AtomicRmw` with `old` set stores the element's previous value in that local (the order the threads reach the element is unspecified, so tests assert properties: a fetch-add of 1 over `n` threads returns a permutation); `AtomicOp` gains `And`, `Or`, `Xor` and `Exchange` (additive, `AtomicOp` is non-exhaustive); an `F32`/`F64` `Exchange` is admitted, a float `And`/`Or`/`Xor`/`Max`/`Min` is not; the `old` local is thread-divergent for the barrier check. **What is reserved and refused, typed:** an `order` other than `Relaxed`, a `scope` other than `Device` (`KernelError::AtomicUnsupported`), `AtomicCas` (`AtomicUnsupported`) and `Fence` (new `KernelError::SyncUnsupported`). `unpopped-cpu-c`'s kernel emitter refuses the new statements as it refused the old ones (it emits no launches). **Who is affected:** any crate that matches or builds `Stmt::Barrier`/`Stmt::AtomicRmw` (the checks run against sibling `origin/main`: vulkane and bonefish build them only in tests that pin the typed decline). **`Stmt::Call { callee, args, result }` and `Module { fns }` are in the same break** (PM ruling 2026-11-02 on `docs/options-u3-remainder.md`: the callee is a `KernelFn` in a `Module`, so downstream breaks once); `check_kernel` refuses a `Call` with the new `KernelError::CallUnsupported` until the module checker and the call semantics land, and `tests/kernel_call_shape.rs` pins that, a module round trip and one mutant (a checker that accepts it). `tests/kernel_atomic_fetch.rs` (9 tests) pins the semantics and the refusals; nine seeded mutants (the old value not stored, the new value returned, the shared path returning the new value, `Exchange` keeping the old, `Or` as `And`, the old local not divergent, the order not checked, the scope not checked, a float `Max` admitted) are all caught.

## `0.40.22` — docs: an options note for U4 (synchronization and atomics); the U3 and U4 ROADMAP rows corrected

`docs/options-u4-sync-and-atomics.md`: what the kernel layer has for barriers and atomics, what the union surveys list as missing, which emitter lanes could use a slice (measured: none implements the kernel-layer path for these statements today; bonefish has named a future `sycl::atomic_ref` spelling), the seven IR shapes a slice adds with whether each breaks `Stmt` consumers, three slicing options and a recommendation (break `Stmt` once, stage the semantics behind typed refusals). Nothing is built; it needs a ruling. ROADMAP U3 said "open, after U2" although its spec, capability types, resource map and row 4 are done; the row now says what is done and what is open. No code change.

## `0.40.21` — docs: ROADMAP X7 retraction (0.40.20 said two survey docs state no ref)

The X7 row merged in 0.40.20 said `thunderkittens.md` and `slang.md` do not state the ref their line cites are into, that a header line was the fix, and that `u3-row3-resource-map-and-frontend.md` had a stale `Frontend` description left over. All three were wrong: `thunderkittens.md` gives the commit it read (`a575e66`), `slang.md` gives its fetch date (2026-10-24, no SHA), and `u3-row3` section 1 is headed "read from `origin/main` (4b5bedc, 0.32.24)", so each is a dated reading. My search had looked for the wrong wording ("read at"). The row now says so and that nothing remains for X7. The 0.40.20 line-cite fixes themselves stand. No code change.

## `0.40.20` — docs: stale line cites named or dated (ROADMAP X7)

X7 audited: of 83 line cites with an identifier beside them, the survey docs and `idiom-lifting-design.md` pin their own ref by design and are left alone; the live cites that had drifted now name the symbol and date the line (`classify_vec_width` in ROADMAP C6, `normative-seams.md`, `deferred.md`; `Access` and `AccumSpec` in `groupshared-sizing.md` and `ir-union-design.md`; the removed `CUDA_RESIDUE` cite; a `recipe.rs` comment). ROADMAP X7 records the audit, the method's limit (a window test, then reading) and what remains. No code change.

## `0.40.19` — the plan oracle evaluates `U32` and `U64` reductions; an integer `Mean` reduction is a typed decline (ROADMAP E19)

`oracle::evaluate` now folds `Sum`, `Prod`, `Max` and `Min` at `U32` and `U64` by the same rule as the other integer dtypes (wrap at the dtype's width, `Max`/`Min` exact), with values above `i32::MAX` and `i64::MAX` in `tests/oracle_narrow_int_reduction.rs`; with it every integer dtype is in scope for those four folds.

PM ruling 2026-11-01 (option b): an integer `Mean` reduction has no defined semantics (no rounding rule for the quotient), so the plan gate now declines it with a typed `PlanError::InadmissibleOpAtDtype` at every integer dtype. **A tightening:** such an op used to plan and then panic in the oracle or the emitter's assert; a float `Mean` and the other integer folds are unchanged. No new semantics until an emitter lane needs them. `tests/integer_mean_is_declined.rs` (red first) pins both halves; three seeded mutants (the decline also hitting a float `Mean`, no decline at all, `U64` dropped from the oracle) are caught.

## `0.40.18` — tests: the any/all/count reductions at narrow integer dtypes agree with the host (ROADMAP E19 tail)

`tests/oracle_narrow_int_predicate_reductions.rs`: the heterogeneous-output shapes the plan gate's rule 4 admits (`count` = `Sum` of `in != 0` with an `I64` output; `any`/`all` = `Max`/`Min` of `in != 0` with a `Reduced(0) != 0` post and a `U8` output) at `I8`, `U8`, `I16`, `U16` through `oracle::evaluate`, against references computed on the host (rows of all zero, one nonzero, all nonzero, and values nonzero only above the low byte). It passed on its first run, so it was mutation-checked: integer `Max`/`Min` swapped, `CmpNe` compared at 8 bits, and an integer `Sum` that drops its elements are all caught. No code change; this closes the untested tail 0.40.16 named.

## `0.40.17` — docs/tests: no in-repo tolerant-band migration is pending (ROADMAP C5)

The 0.40.15 ratchet table and ROADMAP C5 said the remaining hand-written tolerant bands in `complex_domain` were "to migrate" to `compare_for_plan`. They are not: every remaining site is a test of `compare` on hand-built buffers (there is no plan, so `compare_for_plan` cannot apply) or a pattern in a test of the derivation's own output. The ratchet table's reasons and the ROADMAP row now say so; no behaviour change.

## `0.40.16` — the plan oracle evaluates integer reductions at `I8`, `U8`, `I16` and `U16` (additive, ROADMAP E19)

`oracle::evaluate` panicked on any integer reduction narrower than `I32` ("integer reduction dtype I8 out of scope"), so openclownfish X2 PR1 (#33), which lifts narrow-integer elementwise ops and scans, could not check a narrow reduction against the reference. It now evaluates `Sum`, `Prod`, `Max` and `Min` at `I8`/`U8`/`I16`/`U16` by the IR's own rule: arithmetic at an integer dtype wraps at the dtype's width, so a `Sum` is the wrapping fold and a `Prod` the wrapping product (the accumulator is exact `i128` and the store wraps, which equals wrapping at every step for `Sum` and `Prod`), and `Max`/`Min` are exact. `tests/oracle_narrow_int_reduction.rs` compares the oracle with the host's own `wrapping_add`, `wrapping_mul`, `max` and `min` over the native types, with values chosen to overflow; it was committed failing first. Integer `Mean` and the dtypes `U32`/`U64` are still out of scope (the same panic). The plan gate is unchanged. Not covered by this test: a heterogeneous-output narrow reduction (the any/all/count shapes); those now reach the oracle instead of panicking at the dtype check and are unverified here.

## `0.40.15` — `oracle::compare_for_plan`: the band a cell is entitled to, not one the caller picks (additive, ROADMAP C5)

PM ruling 2026-11-01 (option A + D + C of the C5 options note). `oracle::compare_for_plan(expected, actual, plan, operands)` derives the band with `required_fidelity` and compares under it, so there is no band argument to widen; `compare_for_plan_with_exactness` takes what a device measured. `compare_for_plan_within(.., band)` accepts a caller's band only if it is no looser than the entitlement (a tighter one is applied, a looser one is an `Err` that says so; a tolerant band against a bit-exact entitlement is looser, and so is a band larger in either tolerance). When `required_fidelity` declines (a target with no accuracy table, an unrateable body) both return `"no entitlement"` instead of guessing; the caller then chooses a band and calls `compare` knowingly. `compare` itself is unchanged, so nothing breaks. `tests/fidelity_literals_ratchet.rs` counts the textual uses of the tolerant `Fidelity` constructor in every tracked `.rs` file (from `git ls-files`, a directory walk of `crates/` otherwise, and it says which ran) and fails on any file above or below its table entry, so a new hand-written band needs a stated reason and the table only tightens. The breaking `Entitled` type (option B) is revisited if the ratchet shows callers still loosening. `tests/compare_for_plan.rs` (6 tests) and the ratchet were committed failing first; six seeded mutants are caught (a bit-exact entitlement never looser, the looser check skipped, only one of the two tolerances compared, a missing entitlement guessed as bit-exact, a new literal in an unlisted file).

## `0.40.14` — docs: an options note for compelling callers to use `required_fidelity` (ROADMAP C5)

`docs/options-compelled-fidelity.md`: measured counts of hand-picked `Fidelity` literals and `compare` call sites in Unpopped and the sibling repos (predicate and refs stated), the one asymmetry that shapes the fix (a band tighter than the entitlement is harmless, a looser one is the defect), four options, and a recommendation (`compare_for_plan` that derives the band, a refusal of a looser caller band, a ratchet test). Nothing is built; it needs a ruling. No code change.

## `0.40.12` — lift an N-D global-id elementwise store and a 2-D row loop from OpenCL C (additive, ROADMAP E18 part 2)

`convert::lift_elementwise` accepts a row-major flat index built from identifiers, `*`, `+` and parentheses (`out[y * w + x] = a[y * w + x] + b[y * w + x]` with `x`, `y` from `get_global_id`): every read must carry exactly the store's index text, so a read flattened differently (`b[x * w + y]`, transposed) still declines. The op is the same pointwise `OpDef` the 1-D store gives; the rank lives in the structure key. `convert::lift_reduction` accepts a 2-D row loop, `out[r] = acc` after `for (c = 0; c < n; ++c) acc += in[r * n + c]`, as the last-axis reduction, only when `r` is declared as `get_global_id(0)` and the read index is exactly `r*n+c` for the loop variable and bound; a read that ignores the row, or a row taken from `get_global_id(1)`, declines. `tests/opencl_idioms_lift.rs` takes openclownfish's snippets (written failing first); three seeded mutants, two caught and one equivalent (accepting `-` in a flat index changes nothing, because equal texts are pointwise whatever the operator). This completes ROADMAP E18.

## `0.40.11` — integer `Select` and comparisons per backend, so an integer `Rem` composes (additive, ROADMAP E16)

Under `Backend::total_integer_ops` the plan gate now also admits an Elementwise `Select` and a comparison (`CmpEq` .. `CmpGe`) at `I32`/`I64`/`U32`/`U64`, which is what `docs/int-div-shift-semantics.md`'s integer `Rem`, `Select(b == 0, 0, a - (a / b) * b)`, needs (Rockfish#31 could not spell it). A backend that does not opt in is refused exactly as before; Baracuda, Vulkane and openclownfish are unaffected and opt in separately. The kernel interpreter's integer evaluator gains `Select` (the one arm taken, the condition tested `!= 0`). The exactness of 64-bit integer comparisons in the plan oracle, which this relies on, is 0.40.10.

`tests/kernel_int_div_shift_lift_agrees.rs` now also runs `Rem` and all six comparisons at the four dtypes, plan oracle against lifted kernel, over the full cross product of boundary values; `tests/integer_div_gate_is_opt_in.rs` and a gate-level unit test pin the unchanged refusal. Seeded mutants: the `Select` arm and the comparison arm ignoring the flag, and an inverted kernel `Select`, are caught.

## `0.40.10` — FIX: the plan oracle's integer `Max`, `Min` and comparisons are exact above 2^53

`oracle::evaluate` took `Max`, `Min` and every comparison at an INTEGER compute dtype through `f64`, which holds an integer exactly only up to 2^53. At `I64`/`U64` that is a **silent wrong value**: `max(2^53 + 1, 2^53 + 1)` came out `2^53`, and `i64::MAX == i64::MAX - 1` came out true. They are now exact `i128`. The unit test `oracle::int64_exactness_tests` compares the oracle with the host's own integer operators over `Max`, `Min` and all six comparisons at both dtypes and fails before this change.

**Reach, measured at 0.40.8 (`b31af2b`):** no published plan could reach it. The plan gate refuses an integer elementwise `Max`/`Min` and an integer elementwise comparison, and the reduction-predicate comparison admits only leaves against 0 or 1. The wrong value was latent in the oracle and was found while testing the integer `Rem` work (which admits comparisons per backend); it is fixed here first, on its own, because it is a correctness fix. Older versions were not measured. A hand-built plan that bypasses the gate would have hit it.

## `0.40.9` — lift a `Mean` post and exclusive/reverse scans from OpenCL C spellings (additive, ROADMAP E18 part 1)

`convert::lift_reduction` recognises `out[0] = acc / (float)n` after `acc += in[i]` over `i < n` as `ReduceOp::Mean`; the divisor must be the loop bound, so `acc / (float)k` for any other `k` still declines. `convert::lift_scan` reads `out[i] = acc; acc += in[i];` (the store before the update) as exclusive and a `for` counting down to zero (`i >= 0` with `--i`, `i--` or `i -= 1`) as reverse; the two compose. A descending loop that stops anywhere else is declined rather than read as the forward scan, which it would otherwise have been silently. Forward inclusive scans are unchanged. `tests/opencl_idioms_lift.rs` takes openclownfish's snippets as input and parses them with `tree-sitter-cpp` (a new dev-dependency: `in` and `out` are keywords in the Slang grammar). The N-D `get_global_id(d)` and 2-D row-loop idioms are the next E18 PR.

## `0.40.8` — the plan-level oracle evaluates integer `Div` by the total rule (fix, ROADMAP E16)

`oracle::evaluate` carried `unreachable!("integer Div is refused at the plan gate")` for an integer `Div`. 0.40.7 let a backend that declares `Backend::total_integer_ops` past that gate, so a plan for such a backend panicked in the oracle. It now follows `docs/int-div-shift-semantics.md` (single source; not restated): the same rule the kernel interpreter has, written separately. `tests/kernel_int_div_shift_lift_agrees.rs` runs `Div`, `Shl` and `Shr` at `I32`/`I64`/`U32`/`U64` over the full cross product of boundary values through the plan oracle and the lifted kernel, byte for byte; two seeded mutants (a zero divisor giving 0, floor instead of truncating division) are caught. The gate is unchanged, and so are Baracuda, Vulkane and openclownfish (not opted in).

**Found while testing, not fixed here:** an integer `Rem` has no IR form, and its `Select` composition is refused at the plan gate at an integer dtype ("v1 select is float-only"), so a backend cannot be handed a `Rem` through `try_generate` yet (ROADMAP E16).

## `0.40.7` — integer `Div` and `ConstInt` at an integer compute dtype, per backend (additive, ROADMAP E16)

New defaulted `Backend::total_integer_ops(&self) -> bool` (`false`). A backend returning `true` declares that its `arith`/`binary`/`const_int` seams spell integer `Div`, `Shl`/`Shr` and integer constants by the total rule in `docs/int-div-shift-semantics.md` (the single source; not restated here). `try_generate` and `generate_variants` then admit an Elementwise integer `Div` at `I32`/`I64`/`U32`/`U64` and a `ConstInt` leaf at an integer compute dtype; new `plan::try_build_plan_for` and `plan::build_plan_for` take the backend. `try_build_plan` and `build_plan` are unchanged and still refuse both. A defaulted trait method breaks no implementor, so this is additive.

**Baracuda, Vulkane and openclownfish are unaffected**: they do not override the method, so they are refused exactly as before and must opt in separately, each after adopting the guards and the boundary vectors. **Rockfish is the first opt-in.** `tests/integer_div_gate_is_opt_in.rs` pins the unchanged refusal of a backend that does not opt in.

## `0.40.6` — docs: two stale line cites in `ondevice/README.md` (ROADMAP X7, part)

`contract.rs:245` is now `contract.rs:345` (the `op.n_outputs() > 1` guard) and `pattern.rs:229` is now `pattern.rs:260-261` (the `Access::Elementwise` guard), each named by what it guards so the next drift is readable. No code change.

## `0.40.5` — docs: ROADMAP E18 (OpenCL C idioms for `convert.rs`) and the C5 caller count

ROADMAP gains E18 (openclownfish X1: a `Mean` post, exclusive/reverse scan spellings and N-D `get_global_id`/2-D row loops, each a new recognizer; `Frontend` has no idiom hook). C5 now carries the count: zero non-test callers of `oracle::required_fidelity` at `496c192`. No code change.

## `0.40.4` — float `AtomicRmw::Add` (F32/F64) and a launched float Scatter add (additive)

`Stmt::AtomicRmw` admits `Add` on an `F32`/`F64` element (global buffer or `Shared`); float `Max`/`Min` stay refused. **Accumulation order is unspecified** (PM ruling 2026-11-01, option A): a result is defined only inside the band `gamma_(n-1) * sum(abs)` with `n` = the cell's hit count plus the initial value (`docs/ir-kernel-layer.md` section 5a, which also records that this band deliberately differs from KISS-CONTRACT-6.8-0012 as clause-pinned in KISS#759). `lift_scatter_opdef_launched` now accepts an `F32`/`F64` `AtomicAdd` destination; an `F16`/other float destination still declines. Tests: `tests/kernel_parallel_float_scatter_band.rs` (every cell in the band on five geometries, untouched cells exact, two host orders differ yet both in the band, a dropped update falls outside), `tests/kernel_atomic_rmw.rs`.

## `0.40.3` — docs: panic text as a cross-repo API, measured (ROADMAP C11)

`docs/normative-seams.md` section 6 now records a measurement (predicate, refs, per-repo counts, what it does and does not show) in place of the reported 34. No code change.

## `0.40.2` — `AtomicRmw` on a `Shared` allocation (additive)

`Stmt::AtomicRmw` now also accepts a `Place::Alloc` into a `Shared` allocation (integer element, relaxed, same race rule as a global element: an atomic conflicts with a plain access, never with another atomic). `Private` and `Constant` allocations are refused (`KernelError::AtomicUnsupported`). The interpreter traps an atomic on a `Shared` element nothing has stored (`UninitializedShared`). No existing kernel changes meaning. Tests: `tests/kernel_atomic_shared.rs` (a block counter; the barrier after the init and the barrier before the read each deleted is a `DataRace`; the refusals).

## `0.40.1` — one author for the window output-extent rule (no behaviour change)

`window_taps::window_out_extent` now calls the same function as `shape::windowed_extent` (a crate-private `windowed_extent_wide` over `i64`), instead of restating the rule: the reader-side tap check and plan shape inference cannot drift. The only difference is that the arithmetic now saturates on an adversarial extent instead of overflowing. Test: `tests/window_extent_has_one_author.rs`.

## `0.40.0` — Breaking: `Stmt::AtomicRmw`; E1d-6: `lift_scatter_opdef_launched`

**Breaking.** `kernel::Stmt` is exhaustive on purpose and gains a variant, `AtomicRmw { op: AtomicOp, place, value }` (new `AtomicOp { Add, Max, Min }`, new `KernelError::AtomicUnsupported`). A consumer that matches `Stmt` exhaustively must handle it (in this repo: `check_kernel`, the uniformity analysis, the interpreter and its resumable stepper, and `unpopped-cpu-c`, which refuses it as it refuses a `Barrier`). Semantics: relaxed, device-wide, global integer buffer element, no return value; two atomics never race, an atomic and a plain access to one element do (`docs/ir-kernel-layer.md` section 5a). Consumers checked: the only code outside this repo that names `Stmt` is a Vulkane test that pushes a `Barrier`; it does not match exhaustively.

New `kernel_lift::lift_scatter_opdef_launched(op, dtype, rank, block, grid)`: the source's outermost axis is a grid-stride loop and every combine is an `AtomicRmw`; integer `AtomicAdd`/`AtomicMax`/`AtomicMin` are bit-exact against the serial scatter for every `grid`/`block`; `Assign` and a float add decline. Tests: `tests/kernel_atomic_rmw.rs`, `tests/kernel_parallel_scatter_agrees.rs`.

## `0.39.3` — E1d-6: `lift_gather_opdef_launched` (additive)

New `kernel_lift::lift_gather_opdef_launched(op, dtype, rank, block, grid)`: the gather's outermost output axis is a grid-stride loop; bit-exact against the serial gather for every `grid`/`block` and every out-of-range policy. (A launched Scatter follows in 0.40.0, with the atomic statement it needs.) Tests: `tests/kernel_parallel_gather_agrees.rs`.

## `0.39.2` — E1d-5: `lift_scan_opdef_launched`, a chunked block scan; E1d-6: grid-stride `lift_window_opdef_launched`, `lift_im2col_opdef_launched`, `lift_contraction_opdef_launched` and `lift_row_sort_opdef_launched` (additive)

`unpopped-cpu-c` now ships a `README.md` (the publish warned that it had none).

New `kernel_lift::lift_scan_opdef_launched(op, dtype, rank, block, grid)`: one block per row (blocks stride over the leading axes; with none, block 0 does the one row), the innermost axis scanned in three phases. Thread `t` takes the contiguous chunk `[t*c, min(t*c + c, n))` of scan positions (`c = ceil(n / block)`, ascending or descending for a reverse scan), folds it into a total, the totals go through a Kogge-Stone inclusive scan in `Shared` (`Barrier`s around each round), and each thread re-walks its chunk from its offset and writes the prefix exactly as the serial scan does (exclusive, inclusive, a fused post on the prefix rounded once). Integer `Sum`/`Prod`/`Max`/`Min` and float `Max`/`Min` equal the serial scan's bytes; a float `Sum` meets the E1d-2 bound for every output, equals the written order bit for bit, is reproducible and grid-independent, and at `block = 1` equals the serial scan; a float `Prod` and `Mean` decline. `scan_pass` in `kernel_lift.rs` now shares its load/pre-map step (`scan_loads`) with the new lift. Tests: `tests/kernel_parallel_scan_agrees.rs` (six order-independent kinds in four direction/exclusivity combinations on six geometries; the written order; the bound per prefix; no leading axis; a fused post; the declines; a dropped tree round, chunks that run to the end of the row, and two seeded production mutants each seen). **E1d-6.** `lift_window_opdef_launched`, `lift_im2col_opdef_launched` and `lift_contraction_opdef_launched` (batch or `M` axis, each thread the serial `k`-ascending accumulation) and `lift_row_sort_opdef_launched` (rows; a rank-1 sort declines) make the outermost output axis a grid-stride loop (the shared `grid_stride_outermost`, which `lift_elementwise_opdef_launched` now also uses); every output cell is computed from its own taps in the serial order, so all three are bit-exact against the serial kernels for every `grid` and `block`. Tests: `tests/kernel_parallel_grid_stride_agrees.rs` (every pool kind with and without `count_include_pad`, an im2col with batch, channels, stride, padding and dilation, a plain and a batched matmul, a row sort with values and indices in both orders over ties/NaN/both zeros, five geometries including more threads than cells and a non-power-of-two block; an unstrided loop and a wrong start are seen, the former by the race checker). Also records the wavefront ruling below in the docs.

## `0.39.1` — the plan-level oracle takes a shift count modulo the width (openclownfish#26)

`oracle::evaluate` on an integer `Shl`/`Shr` used to answer `0` (`Shl`) or the sign fill / `0` (`Shr`) for a count outside `0..width`; only the kernel interpreter and the cpu-c kernel emitter modelled rule 3 of `docs/int-div-shift-semantics.md` (count read as unsigned bits, modulo the promoted width: 64 for 64-bit types, 32 for every narrower one). A backend that adopted the ruled guards therefore disagreed with the referee on every out-of-range count (found by openclownfish's differential test). The oracle now follows the rule: `1 << 32` at `I32` is `1`, `-8 >> 32` is `-8`, a count of `-1` reads as `W - 1`. Inputs inside `0..width` are unchanged, and the plan gate still states the caller contract for emitters that have not adopted the guards. Tests: `tests/oracle_shift_counts_follow_the_ruled_rule.rs` (`I32`/`U32`/`I64`/`U64`, `Shl` and `Shr`, six values by eight counts through `oracle::evaluate`, against the host's `wrapping_shl`/`wrapping_shr`; red before the change at `1 << 32`), and the inline `integer_bitwise_and_shift_ops_wrap_at_the_promotion_width` expectations are updated.

## `0.39.0` — Breaking: a window with no in-bounds tap is declined, not answered with the monoid identity (KISS-OPS-6.13-0010/-0011)

CireSnave board 345 "B" (KISS 0.6.53, KISS#751): an output window must have at least one tap on a real input element (`p = o*stride - pad_lo + kk*dilation` inside `[0, k_in)`), a producer must not write a shape that breaks it and **a reader must decline it**. The plan gate's `2*pad <= dilation*(size-1) + 1` assert does not enforce this at dilation > 1 (control `n=1, k=2, d=2, s=1, p=1` passes it and its one window has no real tap), and the plan gate sees no extents, so the rule is checked where the extents are known.

**Breaking — old behaviour.** `oracle::evaluate` on a window op used to return the monoid identity (`0` for `Sum`/`Mean`, the identity for `Max`/`Min`, `0` for a `Mean` with a zero valid-tap count) for a window with no in-bounds tap. It now **panics** with `oracle: Window output N has no in-bounds tap (...)` (the oracle has no error channel; its other precondition failures panic too). Adopters whose differential tests feed such a shape, and rely on the identity result, must stop producing the shape. New: `unpopped::window_taps::{window_out_extent, check_window_taps, WindowNoRealTap}` (KISS extent formula and the typed check), `kernel_lift::lift_window_opdef_for_extents` (declines with the new `LiftDecline::WindowWithoutRealTap`), and the extent-free `lift_window_opdef` is unchanged. Tests: `tests/window_without_real_tap_declines.rs` (the KISS control; an enumeration over 4,000-odd geometries proving the helper and showing geometries that pass the span assert and are still declined; the oracle panics on the control and still evaluates a legal window; two seeded mutants, one per call site, each go red).

## `0.38.6` — E1d-4: the launched reduction takes several reduced axes, no kept axis and a fused post (additive); C7 Q5 recorded

`lift_reduction_opdef_launched` no longer declines (1) a reduction over **several reduced axes**: thread `t` takes the indices `t, t+B, ...` of the outermost reduced axis and folds everything under each in row-major order, `Mean` divides by the product of the extents; (2) a reduction with **no kept axis** (one output): block 0 does the row, other blocks idle (`if BlockIdx == 0`); (3) a **fused post**: thread 0 rounds the finalised fold once and the post reads it, as the serial lift does. Order-independent folds equal the serial kernel's bytes; a float `Sum`/`Mean` equals the written order; at `block = 1` every post equals the serial bytes. The declines that remain: a float `Prod` and an integer `Mean`. Tests: `tests/kernel_parallel_reduction_forms.rs`; the old "rank-1 declines" assertion in `kernel_parallel_reduction_agrees.rs` is removed.

**C7 Q5 ruled** (the Unpopped lane's user answered "Go with option C." in session 2026-10-31; reported by the lane, unverified by the PM): the normative rule is that every output window has at least one tap on a real input element; KISS-OPS-6.13-0010/-0011 (board 345 "B") adds that a reader MUST decline any axis where a window has no in-bounds tap. `2*pad <= dilation*(size-1) + 1` is NOT sufficient at dilation > 1 (n=1, k=2, d=2, s=1, p=1 passes it; this entry first said it was sufficient, which was wrong). Recorded in `ROADMAP.md` C7 and `docs/normative-seams.md` section 7.


## `0.38.5` — E1d-3: `lift_row_reduce_opdef_launched`, a block-tree `Access::RowReduce` (additive)

New `kernel_lift::lift_row_reduce_opdef_launched(op, dtype, rank, block, grid)`: one block per row (blocks stride over the outermost leading axis), each stage a block-tree reduction whose single rounded result thread 0 stores in `Shared` and a `Barrier` broadcasts to every thread as `Reduced(s)`, then the epilogue runs over the row's columns strided by thread. Float only (`F32`/`F64`), `rank >= 2`; a rank-1 row reduction, a `Prod` or empty stage list, a non-power-of-two `block` and a zero `grid` decline with a typed `LiftDecline`. `Max`/`Min` stages are order-independent and equal the serial lift's bytes; a `Sum`/`Mean` stage meets the E1d-2 bound and its written order (`docs/e1d-parallel-mapping.md`, "E1d-3 as built"); at `block = 1` the whole kernel equals `lift_row_reduce_opdef`'s bytes. The block-tree fold (`Fold`: publish, log-step rounds, partner `have`) is now one helper in `kernel_lift/block_reduce.rs` shared by both launched lifts; `lift_reduction_opdef_launched` is unchanged in behaviour. Tests: `tests/kernel_parallel_row_reduce_agrees.rs` (Max/Min bit-exact; the Sum/Mean bound and written order; a two-stage centred quotient bit-exact to the written order; softmax and RMS norm within 4 ulps of the serial lift; reproducible and grid-independent; the declines; a dropped round, an unstrided slice and a missing broadcast `Barrier` each seen).

## `0.38.4` — E1d-2: a launched reduction accepts a float `Sum`/`Mean` with a declared bound (additive)

`lift_reduction_opdef_launched` no longer declines a float `Sum` or `Mean`. PM ruling 2026-10-31: allowed with a declared bound, `|result - exact| <= gamma_(n-1) * sum|x_i|` (`u = 2^-53`, plus the final rounding's half ulp for `F32`; against the serial kernel, twice that), and reproducible run to run for a fixed `(grid, block)`. The order is written in `docs/e1d-parallel-mapping.md` ("E1d-2 as built"): thread `t` folds `t, t+B, t+2B, ...` ascending into an `f64`, then `acc[t] = acc[t] + acc[t+s]` for `s = B/2 ... 1`, one rounding at the end; at `B = 1` it is the serial order. Float `Prod` and integer `Mean` still decline (`LiftDecline::UnsupportedForm`, message reworded). Tests: `tests/kernel_parallel_reduction_float_sum_bound.rs` (the bound over five rows including a cancelling one; run-to-run and grid independence; the kernel's bytes equal a plain-Rust reimplementation of the written order; a dropped round and an unstrided slice are seen). The old decline test is renamed `a_form_with_neither_order_independence_nor_a_bound_declines` and now uses a float `Prod`.

## `0.38.3` — integer division and shifts are total at the kernel layer (additive)

PM ruling 2026-10-31 on `docs/int-div-shift-semantics.md`, as recommended. In the kernel interpreter (`run_kernel`) and the cpu-c *kernel* emitter (`CpuC::lower_kernel`), for `I32`/`I64`/`U32`/`U64`: `a / 0` gives `a`; signed `MIN / -1` gives `MIN`; otherwise the quotient truncated toward zero; a shift count is read as unsigned bits modulo the promoted width (64 for 64-bit types, 32 for every narrower one, `S8`/`U8` included: unchanged promote-then-truncate); `<<` is defined for a negative or overflowing left operand and `>>` is arithmetic on signed types, logical on unsigned. Before, the interpreter declined these (`Unsupported`) and the emitter declined them. Integer remainder is the guarded form over the total `Div` (`b == 0` gives `0`; `MIN % -1` is `0` by the wrap), pinned in the tests. Not changed: the intent-layer plan gate still refuses integer `Div` and the `Shl`/`Shr` caller contract stands on the plan path until each emitter adopts the guards (ROADMAP E16). Tests: `crates/unpopped/tests/kernel_int_div_shift_semantics.rs` (interpreter against host-native integers, four seeded mutants red) and `crates/unpopped-cpu-c/tests/kernel_emit_int_div_shift_agrees.rs` (compiled C against the interpreter on the ruled vectors; three seeded mutants of the emitted source red, with an unmutated control).

**Fix found on the way.** Releases 0.37.1 through 0.38.1 were published with four raw NUL bytes in `crates/unpopped/src/contract.rs` (the `revision_hash_of` byte-string literal and its doc comment) and two in this file (behaviour identical, they compiled; no release is yanked): a backslash-zero escape typed through a tool that interprets it. It compiled and meant the same value, which is why nothing caught it; they are now the two-character escape. New guard `crates/unpopped-conformance/tests/no_text_file_contains_a_nul_byte.rs` fails on a NUL in any text file of `crates/`, `docs/`, `.github/` and the root (population control; seen red on an injected NUL).


## `0.38.2` — docs: ROADMAP C7 records CireSnave's rulings on the KISS seam questions

The five questions of ROADMAP row C7 (move versus compute, backend dtypes, vector width, accuracy, window overlap) are ruled by CireSnave (board 338/345, relayed by the PM 2026-10-31); the row now states each ruling and that Q5's exact wording is pending board 345. No code change.

## `0.38.1` — `cfamily::strided_offset_expr` and `cfamily::unravel_decomp`: the strided spellings emitters re-spelled (additive)

Two emitters each kept a private copy of the strided element-offset expression (`baracuda-cuda-emit`'s `offset_expr`/`offset_expr_coord`, `openclownfish-clc`'s `offset_expr`; openclownfish#15), and `baracuda-cuda-emit` the row-major unravel (`emit_unravel_decomp`). They are now one spelling in `unpopped::cfamily`: `strided_offset_expr(bcast, coord, stride, rank, perm)` (`c{d}*{stride}_{si}` per non-broadcast axis, `perm` for a `View::Permute` read, `0` when every axis broadcasts) and `unravel_decomp(axes, indent, lin, ctype, coord, shape, guard)` (the target's wide integer is a parameter: `long long` for CUDA C, `long` for OpenCL C). The format strings are those of the private copies, so switching changes no generated byte; each emitter's own tests are the check when it does. Tests: `tests/cfamily_strided_spelling.rs` (golden strings; three seeded mutants, dropping the broadcast filter, ignoring `perm`, dropping the divide, each go red).

## `0.38.0` — BREAKING: `ArtifactKind` and `SynthKernel` are `#[non_exhaustive]`; `HipCodeObject` and `SpirV` added

`jit::ArtifactKind` gains `HipCodeObject` (ELF code-object bytes: hipRTC / `hipcc --genco`; Rockfish R7d) and `SpirV` (little-endian bytes of the word stream), and both it and `jit::SynthKernel` are now `#[non_exhaustive]`, so a later target is an additive variant. **Migration:** a `match` on `ArtifactKind` needs a wildcard arm (refuse an artifact you do not know, as a loader already refuses `Stub`); build a `SynthKernel` with the new `SynthKernel::new(entry_point, source, artifact, kind)` instead of a struct literal. `Compiler::artifact_kind()` keeps its `Ptx` default; a compiler for another target overrides it. Payloads are unchanged: a compiled artifact is `Vec<u8>` plus a kind; SPIR-V words straight from an emitter ride `GeneratedKernel::with_words` (0.37.1), not `SynthKernel`. No in-tree consumer matched the enum or built a `SynthKernel` literal (the one construction is in `jit.rs`).

## `0.37.1` — `GeneratedKernel` can carry a binary: `with_words` / `words` / `revision_hash` (additive)

`GeneratedKernel::with_words(Vec<u32>)` and `words() -> Option<&[u32]>` let a binary target (Vulkane's SPIR-V emitter is the first) return its compiled word stream beside the source text it was built from; `GeneratedKernel::new` is unchanged, so every existing backend compiles and behaves as before, and `generate`'s provenance stamp keeps the words. New `GeneratedKernel::revision_hash()` is what the contract's `kernel_revision_hash` and the link-registry row now use: FNV-1a over the source, and, when words are present, over a `\0words\0` tag and each word little-endian, so recompiling the same source into different words re-mints the revision. A kernel without words hashes byte for byte as before (pinned against the FNV-1a value for `"a"`). The ABI/binding manifest (ROADMAP E5) is not part of this change. Tests: `tests/generated_kernel_carries_words.rs` and the `contract.rs` hash test.

## `0.37.0` — BREAKING: the core carries no Slang grammar (`parse_slang`, `SLANG`, `tree-sitter-slang` removed)

`unpopped::convert::{parse_slang, SLANG}` and the `tree-sitter-slang` dependency are gone, and the `convert` feature now enables only `tree-sitter` (the shared C-family walkers and the `Frontend` type). The Slang front end is Vulkane's: use `vulkane-slang-parse` 0.17.0 (`vulkane_slang_parse::{SLANG, parse_slang}`) with `lift`, `lift_elementwise`, `lift_reduction`, `lift_scan` here, or build a `Frontend` with `Frontend::new`. CUDA left the same way in 0.30.0 (`baracuda-cuda-parse`). The walkers, `Frontend`, `Frontend::new` and `with_intrinsics` are unchanged. **MSRV claim, as measured:** `unpopped`, `unpopped-vocab` and `unpopped-cpu-c` build on the declared 1.85 (`cargo +1.85 build -p unpopped -p unpopped-vocab -p unpopped-cpu-c`), and `unpopped-vocab` and `unpopped-cpu-c` also test on it; `unpopped`'s own tests run on stable because the dev-dependency `tree-sitter-slang` needs rustc 1.90 (the `msrv` CI job is split accordingly). This crate's own tests and the `slang_corpus_report` example build a Slang `Frontend` from `tree-sitter-slang`, now a dev-dependency only; `tests/neutral_core_names_no_cuda_grammar.rs` guards the shipped manifest and `convert.rs` (its controls prove the shipped/dev split lands where it should).

## `0.36.0` — BREAKING: `unpopped-slang` leaves the workspace (it is Vulkane's, as `vulkane-slang-emit`)

The Slang emitter moved to Vulkane (PM ruling 2026-10-30, CireSnave: *"Vulkane now owns Slang in our ecosystem."*): `vulkane-slang-emit` 0.17.0 is published with the code and its history, and the three conformance cells that spanned both emitters (`cell_token_is_not_a_computation`, `dtype_lowering_coverage`, `lower_kernel_default_is_the_typed_decline`) live in its `tests/`, with `unpopped-cpu-c` as their dev-dependency. This release deletes `crates/unpopped-slang`, its workspace dependency and the conformance crate's dev-dependency. The first two cells keep their `CpuC` halves here (`cell_token_is_not_a_computation`, `lower_kernel_default_is_the_typed_decline`); the dtype matrix, which compared the two emitters, is gone from this repo.

**Breaking for consumers of the line.** `unpopped-slang` ends at 0.35.3 on crates.io (deprecated since 0.34.13, nothing yanked); there is no `unpopped-slang` 0.36. A crate that depends on `unpopped-slang` tracking the `unpopped` line can no longer do so: it must move to `vulkane-slang-emit`. Known consumers, read from each repo's `origin/main`: `baracuda-cuda-emit` (a test-only dev-dependency, `unpopped-slang = "0.35.3"`, used by its cross-backend tests) and `unpopped-harness` (pinned `0.22.0`, unaffected until it bumps). Published `unpopped-slang` versions keep resolving; no `unpopped` API changes in this release (`parse_slang` and the `SLANG` residue stay until the next breaking step).

## `0.35.5` — E1d-1: a block-tree Reduction launched with Shared and Barrier (additive; one behaviour change)

`kernel_lift::lift_reduction_opdef_launched(op, dtype, rank, block, grid)` lifts a single-axis `Access::Reduction` as one cooperating block per output: each thread folds a strided slice of the reduced axis, the partials go through a `Shared` array and a log-step tree with a `Barrier` per round, thread 0 stores, and blocks stride over the outermost kept axis. It takes only order-independent kinds (integer `Sum`/`Prod`/`Max`/`Min`, float `Max`/`Min`), so it is bit-exact against the serial kernel except for NaN payload and the sign of a zero when both zeros occur in a `Max`/`Min`; float `Sum`/`Prod`/`Mean`, integer `Mean`, a fused post, several reduced axes, no kept axis and a non-power-of-two `block` decline with a typed `LiftDecline` (`BadLaunch` for the launch shape). Geometry is the caller's, never derived. Proof: `tests/kernel_parallel_reduction_agrees.rs` (`run_launch` equals the serial kernel over six kinds, three layouts and seven geometries; six broken copies each disagree or trap).

**Behaviour change in the barrier checker.** `check_kernel`'s barrier-uniformity rule treated every `Exec` leaf as thread-divergent, so a `Barrier` inside a loop whose bounds read `BlockIdx`, `BlockDim` or `GridDim` was rejected with `BarrierDivergent`. Those three read the same in every thread of a block, so they are now uniform; `ThreadIdx`, `LaneId`, `SubgroupId` and `SubgroupSize` stay divergent and the four pinned over-rejections are unchanged. The checker only accepts more kernels, never fewer. Pinned by `a_barrier_under_block_level_exec_ids_is_accepted_and_under_thread_ids_is_not` in `tests/kernel_barrier.rs`, which fails on the old rule (the block-strided loop is rejected) and on a rule that makes `ThreadIdx` uniform (the last assertion).

## `0.35.4` — E1d-0: an Elementwise kernel launched with a grid-stride loop (additive)

`kernel_lift::lift_elementwise_opdef_launched(op, dtype, rank, block, grid)` lifts an Elementwise op with its outermost output axis as a grid-stride loop on `grid` blocks of `block` threads along axis 0, and a `LaunchShape` to match; `LiftDecline::BadLaunch` (new variant of a `#[non_exhaustive]` enum) refuses a zero `block` or `grid`. The geometry is the caller's, never derived from an extent, and the parallel axis is declared by the lift from the op family, not found by inspecting loops (`docs/e1d-parallel-mapping.md`). The kernel interpreter's integer evaluator now accepts a wrapping integer-to-integer `Cast` (the U32 execution ids become the I32 counter); a saturating one still refuses. Proof: `run_launch` equals `run_kernel` byte for byte across grids 1/3/7 and blocks 1 to 256 over three ops with special values, and four seeded mutants (drop the stride, drop the block factor of the start, drop the cast wrap, accept any saturation) each go red. A new conformance test fails the build if any `crates/*/src` file contains a launch spelling.

## `0.35.3` — `Frontend` carries its own math-call tables

**Additive; no breaking change.** `Frontend` gains `unary_fn` and `binary_fn` (function pointers from a source-language math-call name to the IR op), defaulting in `Frontend::new` to the CUDA spellings it has always recognised, and a const `Frontend::with_intrinsics(unary, binary)` to override them. A language that overloads the math names (OpenCL C `exp`/`sqrt`/`pow`) can now lift them; asked for by openclownfish. `Frontend` is `#[non_exhaustive]` and `new` keeps its six arguments, so no caller changes. First slice of ROADMAP I1; the CUDA default tables stay in core until `baracuda-cuda-parse` supplies its own. The `parse` docs now say the shared walkers need a `tree-sitter-cpp`-shaped grammar (`indices` field).

## `0.35.2` — docs: E1d parallel mapping design note and the US10949182B2 re-read

**Docs only; no code, no API, no behaviour change.** `docs/e1d-parallel-mapping.md` designs the target-neutral parallel transform (declared-parallel axes, a caller-chosen grid with grid-stride loops, block-tree `RowReduce`) and re-reads US10949182B2's independent claims against it: outside as read, not clear. `docs/PATENT-SWEEP.md` row 10 now records the estimated expiry Google Patents shows (2037-11-16, read 2026-10-31), where it said none was shown.

## `0.35.1` — docs: integer division, remainder and shift semantics, measured per target (design note)

**Docs only; no code, no API, no behaviour change.** `docs/int-div-shift-semantics.md` records what C11, the C++ draft, PTX ISA 9.4, LLVM, OpenCL C 3.0, GLSL 4.60.8, SPIR-V and WGSL each specify for integer division by zero, `MIN / -1` and a shift count at or above the width, with sections cited, and recommends one total IR rule (WGSL's division cases, shift count modulo the width). Awaiting the PM's ruling; integer `Div` stays rejected and `Shl`/`Shr` keep their caller contract until then.

## `0.35.0` — BREAKING: `tree-sitter` 0.26 → 0.27 (the `convert` feature)

`Frontend::parse` returns a `tree_sitter::Tree`, so the grammar-runtime version is part of this crate's public API: every external `Frontend` (Baracuda's CUDA frontend, openclownfish, `vulkane-slang-parse`) must build its `Tree` with `tree-sitter` 0.27 to link against this release. No `unpopped` source changed; `tree-sitter-slang` 0.3.1 (already the latest) builds against 0.27. CireSnave's standing rule is every dependency on its latest version. `cargo test --workspace --all-features` (129 test results, 0 failed) and `clippy --workspace --all-targets --all-features -D warnings` pass.

## `0.34.13` — `unpopped-slang` is deprecated: the final release

**Docs and metadata only; no code, no API, no behaviour change.** `unpopped-slang` carries a crate-doc notice, a "DEPRECATED" description and the crates.io `deprecated` maintenance badge, all pointing to `vulkane-slang-emit` (Vulkane owns Slang, CireSnave 2026-10-30; handover `vulkane` `docs/design/slang-parser-emitter-handover.md`, vulkane#107). Only `unpopped-slang` is published at this version; the other crates stay at their published versions. The crate is removed from the tree in the next PR.

## `0.34.12` — docs: ownership map records Slang as Vulkane's

**Docs only; no code, no API, no behaviour change.** `docs/parser-emitter-ownership-map.md` sections 1 and 9 now say Vulkane owns Slang (CireSnave 2026-10-30), with the handover in `vulkane` `docs/design/slang-parser-emitter-handover.md` (vulkane#107). `unpopped-slang` is untouched and stays in-tree until `vulkane-slang-emit` is published.

## `0.34.11` — docs: ROADMAP E10 closed as not checkable in `validate_row_sort`; E1c and E2 move to Vulkane

**Docs only; no code, no API, no behaviour change.** `k <= 1024` is the contract of the out-of-tree CUDA bitonic variant (the base rank sort has no limit), and `validate_row_sort` runs on a `StructureKey` with no numeric extents, so it cannot check it. ROADMAP owner column for E1c and E2 is now Vulkane (PM ruling 2026-10-30).

## `0.34.10` — feat: the cpu_c kernel emitter spells integer compute (`I32`, `I64`, `U32`, `U64`) (E1b-int)

**Behaviour change in `CpuC::lower_kernel` only; no API change.** A `KernelFn` whose compute dtype is `I32`, `I64`, `U32` or `U64` was `LowerError::UnsupportedKernel` and is now emitted as serial C; every other dtype (16-bit, narrow, packed) is still that typed decline. In the integer forms, `+ - *` wrap in two's complement through the unsigned type (C's signed overflow is undefined), `Select` tests its condition against zero and moves the arm, `Max`/`Min` and the comparisons yield the integer 1/0, and `& | ^` are the C operators. Integer `/`, shifts and the logical operators decline with a typed reason (the oracle's rules for a zero divisor and an oversized shift are not ones the emitter has been taught).

`unpopped-cpu-c/tests/kernel_emit_int_reduce_agrees.rs` (reductions over both axes, scans in both directions and exclusivities, a scan pre-map, 1-D windows with padding, stride and dilation) and `kernel_emit_int_scatter_agrees.rs` (gather with `Skip`/`Clamp`/`ZeroFill`, scatter with `Assign`/`AtomicAdd`/`AtomicMax`/`AtomicMin`, `bincount` at both index widths) compile and run lifted kernels and compare output bytes with `run_kernel`: bit-exact. The corpus wraps (`MAX + 1`, a product past the width), has an all-negative `Max` row and an all-positive `Min` row, and a destination that starts non-zero.

Seeded mutants on the emitted C, all caught: a `Max` turned into a `Min`, the `Max` first-element rule removed (the accumulator then stays 0 on an all-negative row), the wrapping add narrowed to 16 bits, the scatter's negative-index and one-past-the-end tests off by one, an `AtomicMax` that kept the smaller, and a `bincount` that added 2. (A first narrowing mutant on one row was equivalent, the row cancels under truncation; it is judged on a row that does not.) `integer_compute_is_a_typed_decline` is replaced by `unsupported_integer_widths_are_still_a_typed_decline`, which pins the 16-bit refusal. Test harness: `kernel_run.rs` binds `U64`, and `kernel_data.rs` builds 8-byte `I64`/`U64` elements (a zeroed `I64` buffer had been half its size).

The serial loop is a correctness reference, not a performance or device-timing claim (PM ruling 2026-10-30).

## `0.34.9` — test: cpu_c kernel emitter agrees with the interpreter on Gather, Scatter and Im2Col (E1b, part 4; E1b complete for float kernels)

**Tests and docs only; no API or behaviour change.** `unpopped-cpu-c/tests/kernel_emit_gather_scatter_agrees.rs` and `kernel_emit_im2col_agrees.rs` compile and run lifted kernels and compare output bytes with `run_kernel` (bit-exact except NaN sign/payload). Gather: `Skip`, `Clamp` and `ZeroFill` with `I32`, `I64` and `U32` index buffers, both axes, a broadcast 1-D index and `F64`. Scatter: `Assign` and `AtomicAdd` with the same index types, both axes, `index_add` and `F64`. Im2Col: padding, stride, dilation, batch and channels, a rectangular input and window, `F64`. The output starts non-zero (`7.0`), so `Skip` is distinguishable from `ZeroFill` and an unwritten cell shows; the index corpus holds negatives, the last valid row, one past it, far-out values and each index type's extremes. The emitter needed no change.

Seeded mutants on the emitted C, all caught: the gather and scatter negative-index and one-past-the-end tests off by one, the clamp's lower branch inverted, a scatter-add turned into an assign, and, for Im2Col, the clamp's upper limit one too high (row and column), the tap test inverted (row and column) and the pad value changed from the typed zero. (Two mutants I first tried, `>` to `>=` in a clamp at zero, were equivalent and are not kept.)

**Not emitted yet, and pinned:** an integer compute dtype (`Scatter`/`Gather` over `I32`/`I64` data, `Bincount`, `AtomicMax`/`AtomicMin` on integers, integer reductions and scans) is `LowerError::UnsupportedKernel` from `admit` (`integer_compute_is_a_typed_decline`); that is the next emitter change, not a test gap.

Harness correction: `same_bytes` judged every output buffer by the kernel's compute dtype, so an `I32` index buffer written by an `F32` kernel (RowSort's) was compared as `f32` with NaNs canonicalised. It now takes each buffer's own dtype; `nan_is_equal_only_to_nan` gains the control that two integers that are both NaN as `f32` still differ. No earlier result changed (the indices there are small, so never NaN as `f32`). `tests/support/kernel_run.rs` also binds `I64` and `U32` buffers.

Also recorded (docs): the 2026-10-30 parser/emitter ownership assignments (SPIR-V, GLSL, HLSL, WGSL to Vulkane; HIP C++ to Rockfish; SYCL/DPC++ to Bonefish; Metal and Slang owners open) in `docs/parser-emitter-ownership-map.md` section 9, which supersedes that map's earlier "Vulkane is not a stakeholder" finding.

The serial loop is a correctness reference, not a performance or device-timing claim (PM ruling 2026-10-30).

## `0.34.8` — test: cpu_c kernel emitter agrees with the interpreter on RowReduce and RowSort (E1b, part 3)

**Tests and docs only; no API or behaviour change.** `unpopped-cpu-c/tests/kernel_emit_row_reduce_agrees.rs` and `kernel_emit_row_sort_agrees.rs` compile and run lifted `Access::RowReduce` and `Access::RowSort` kernels and compare output bytes with `run_kernel`. RowReduce: the four identity epilogues, an all-negative `Max` and all-positive `Min` row, NaN/infinity rows, and softmax, a centred quotient and an RMS norm (`F32` within 3 ulps, tolerance mode; `F64` exact). RowSort: every `SortOut`, full and capped (`k_out` 1, 3, 11), both directions, `F32`/`F64`, rank 2 and 3, over a corpus of ties, both zeros, infinities and NaNs of several signs and payloads; values are bit-exact except NaN sign/payload, indices exact. The emitter needed no change.

Seeded mutants, all caught, on the emitted C: the stage accumulator declared `float` (sum 3.5 becomes 2.5); a float stage accumulator under softmax tolerance mode leaves the 3-ulp bound (row `[0, -18.42 x 100]`, so the tolerance is not vacuous); the sort comparator with its two tie-breaks (`<=` and `<`) swapped and with its NaN test inverted each change the output bytes.

The serial loop is a correctness reference, not a performance or device-timing claim (PM ruling 2026-10-30). Not here: Im2Col, Gather, Scatter (E1b, next PR).

## `0.34.7` — test: cpu_c kernel emitter agrees with the interpreter on Contraction and Window (E1b, part 2)

**Tests and docs only; no API or behaviour change.** `unpopped-cpu-c/tests/kernel_emit_contraction_window_agrees.rs` compiles and runs lifted matmul (`F32`/`F64`, dense and column-major operands) and batched matmul kernels, and 1-D `Sum`/`Max`/`Min`/`Mean` windows (padded and strided, both `Mean` divisors, dilation, `F32`/`F64`), comparing output bytes with `run_kernel` (bit-exact except NaN sign/payload, below). The emitter needed no change. The two `Mean` divisors are shown to give different bytes, so one divisor for both could not pass.

Seeded mutants, both caught: the contraction accumulator declared `float` (3.5 becomes 2.5) and the window accumulator declared `float` (1 becomes 0), made on the emitted C.

One harness correction: a NaN made by an operation (`inf - inf`, `0 * inf`) has the host's sign (x86: negative) where the interpreter makes the positive one. The exact comparison is therefore **bit-exact except NaN sign/payload**: any two NaNs are equal (`canonical_nans` in `tests/support/kernel_run.rs`), which is not semantic here because the spec treats NaN as one value and no op reads a NaN's sign or payload. Every non-NaN byte (including `-0.0` vs `0.0` and the infinities) is still compared exactly, and `nan_is_equal_only_to_nan` proves NaN vs a number, `+inf` vs NaN and finite vs NaN still fail. Earlier E1a/E1b tests never produced such a NaN, so none changed result.

The serial loop is a correctness reference, not a performance or device-timing claim (PM ruling 2026-10-30). Not here: RowReduce, RowSort, Im2Col, Gather, Scatter.

## `0.34.6` — test: cpu_c kernel emitter agrees with the interpreter on Reduction and Scan (E1b, part 1)

**Tests and docs only; no API or behaviour change.** `unpopped-cpu-c/tests/kernel_emit_reduce_agrees.rs` lowers lifted `Access::Reduction` and `Access::Scan` kernels (`lift_reduction_opdef`, `lift_scan_opdef`) with `CpuC::lower_kernel`, compiles the C, runs it and compares output bytes with `run_kernel` (bit-exact except NaN sign/payload since 0.34.7; before it, byte-exact): every `ReduceOp` over the last axis and over the outer axis with `keepdim`, every scan monoid forward-inclusive and reverse-exclusive, `F32` and `F64`. The corpus holds NaN (first and mid-row), signed zeros, infinities, an all-negative row and a row that cancels in `f32` but not in `f64`. Fused posts (tolerance mode) agree within 3 ulps for `F32` and byte-exact for `F64`. The emitter needed no change: the E1a statement spelling already covers these kernels.

Seeded mutants, each caught: the generated accumulator declared `float` instead of `double` (reduction row 0 gives `-0.5` not `0.5`), the scan prefix declared `float` (element 2 gives `0` not `1`), and the same prefix mutant under a fused post, which leaves the 3-ulp bound (so the tolerance is not vacuous). The kernel checker rejects an `F32` accumulator, so these mutate the emitted source; `tests/support/kernel_run.rs` gains `run_c_mutated` for that.

The serial loop is a correctness reference, not a performance or device-timing claim (PM ruling 2026-10-30). Not here: Contraction, Window, RowReduce, RowSort, Im2Col, Gather, Scatter (E1b, next PRs).

## `0.34.5` — feat: `Backend::lower_kernel` and the cpu_c serial-C kernel emitter (E1a)

**Additive.** `Backend::lower_kernel(&KernelFn)` is a new **defaulted** trait method (PM ruling 2026-10-30), so every existing backend, Baracuda's and Fuel's included, compiles unchanged; the default returns the new typed decline `LowerError::UnsupportedKernel` (not `UnsupportedOp`, never a panic). `Lowering` gains a `local` seam and `local_types`, so a typed node over a kernel `Local` can be spelled and typed.

`unpopped-cpu-c` overrides it: a `KernelFn` becomes **serial C**, one logical thread, returning `0`, `1` (out of bounds) or `2` (zero loop step), which are the interpreter's traps. Coverage in E1a: `F32`/`F64` compute, buffers and scalars the C type table spells, no launch, allocation, barrier or packed local; anything else is `UnsupportedKernel`. Constants are cast to the compute type so an `f32` expression rounds per operation like `float`, not once at double.

**This is a correctness reference. The serial loop is not a performance claim and no 4070 or other device timing follows from it** (PM ruling 2026-10-30; the parallel mapping is stage E1d, its own note and ruling).

Tests: `unpopped-cpu-c/tests/kernel_emit_agrees.rs` compiles the emitted C with the host compiler and compares output bytes with `run_kernel` (F64 byte-exact, since 0.34.7 bit-exact except NaN sign/payload; F32 compound expressions within 3 ulps, the spec's tolerance mode, plus an exact per-operation check in Rust that the interpreter cannot make), with 12 seeded mutants all caught; `unpopped-conformance/tests/lower_kernel_default_is_the_typed_decline.rs` pins the default for `Slang`, a bare backend and an overriding backend, with five sabotages (panic, `Ok`, `UnsupportedOp`, empty and unattributed reason) all caught. The host-compiler driver moved verbatim to `tests/support/cc.rs`. Also fixed: cpu_c's module docs said unsupported schedules "panic"; they return `UnsupportedSchedule`.

Not here (E1b onward): Reduction, Scan, Contraction, Window, RowReduce, RowSort, Im2Col, Gather, Scatter through this path; Slang; a `ResourceMap` parameter (Slang needs one; cpu_c's argument layout is fixed and documented in `kernel.rs`, so the E1 note's `bindings` parameter is deferred to the Slang stage).

## `0.34.4` — docs: E1 design note, reference emitters beyond `Schedule::Scalar`

**Docs only; no code, API or behaviour change.** `docs/e1-emit-from-the-kernel-layer.md` frames ROADMAP row E1: the options (grow `Backend::lower` per schedule, emit from a lifted `KernelFn`, or stop), a recommendation (emit from `KernelFn`, staged cpu-c first, differential against `run_kernel`), the trait shape and argument ABI it needs, and four decisions for the PM. Also records that cpu-c's module docs say "panics" where the emitter returns a typed `UnsupportedSchedule` decline.

## `0.34.3` — docs: PATENT-SWEEP full independent-claim read for the 0.34.x publish

**Docs only; no code, API or behaviour change.** `docs/PATENT-SWEEP.md` gains a section that reads every independent claim, in full, of the hits the earlier passes read from excerpts only (US10726606B2, US12412230B2, WO2013070636A1, US20100122264A1) or left unread (US20120254497A1, US20080109795A1, US12299766B2, EP4318225A1, US20240045677A1, US20240045654A1, EP1620793B1). Wording is "outside as read", not legal advice; none of the independent claims read is met by the `unpopped*` crates as described, and the section records re-read triggers and the one broad claim (US20100122264A1 claim 1) that no adopter has been charted against.

## `0.34.2` — feat: `Barrier` in `run_launch`, the resumable evaluator and barrier-aware data races (U3 row 4, PR 3)

**Additive.** `run_launch` now runs kernels with `Stmt::Barrier`. Each thread of a block is a resumable evaluation (an explicit frame stack, `oracle::kernel_resume`, because a barrier can sit inside a loop); a phase runs every runnable thread until it reaches a barrier or finishes, and when every thread of the barrier's scope (the block, or the subgroup) waits at the same barrier statement they all advance. `KernelTrap::BarrierMismatch` (new): a thread finished while peers waited for it (a `Return` before a barrier, which the checker allows), or the threads of a scope waited at different barriers and none could be released.

**`DataRace` is now barrier-aware.** Every access carries who made it and how many barriers had released around it. Two accesses to one element, one of them a write, conflict unless ordered: same thread, or the same block with a block barrier between them, or the same subgroup with a subgroup or block barrier between them. Accesses by different blocks are never ordered. Readers are tracked until the next write, so the write-after-read hazard (a fast thread's next-round store against a slow thread's read of the last round) is caught.

`run_kernel` and the lifting differential keep the recursive evaluator (design note section 5). The resumable one is a second implementation of loop semantics, so `tests/kernel_resume_agrees.rs` runs nested loops with `Break`, `Continue`, an early `Return` and a `While` through both and compares the output over six inputs that reach different branches.

Tests: `tests/kernel_barriers_run.rs` (tile reverse, barriers in a loop with and without the second barrier, subgroup versus block scope, global ordering inside and across blocks, mismatch by `Return` at both scopes), `tests/kernel_resume_agrees.rs`, and unit tests on `release` for the guards a checked kernel cannot reach. Every new rule was mutated in the source: 33 mutants, each turned a named test red; seven survived the first round (a readers-by-thread-index bug, a missing zero-step test, and the `release` guards that no checked kernel can fire), and each now has a test that kills it.

**With this row 4 is complete:** ids, launch shape, shared/private arrays, barriers with a uniformity rule, and a driver that runs them. Not here: `Constant` initializers, atomics, cluster scope, pointer values (design note section 8).

## `0.34.1` — feat: `run_launch`, the per-thread driver for barrier-free kernels (U3 row 4, PR 2)

**Additive.** `oracle::kernel_launch::{run_launch, run_launch_with_subgroup}` run every thread of a launched kernel, block by block (x fastest), threads one after another. `Exec` ids evaluate (`ThreadIdx`, `BlockIdx`, `BlockDim`, `GridDim`, `LaneId`, `SubgroupId`, `SubgroupSize`); the subgroup size is the launch range's minimum unless the caller passes one inside the range. `Shared` (per block) and `Private` (per thread) allocations are real storage, sized by `Static` or by the integer argument a `Param` extent names.

`KernelTrap` (already `#[non_exhaustive]`) gains `AllocOutOfBounds`, `UninitializedShared`, `UninitializedPrivate` and `DataRace`. **Shared memory is not zero-filled**: a read of an element no thread stored traps, so a missing barrier cannot hide behind a zero. **`DataRace`**: two threads touching one element with at least one write, for global buffers across the whole launch and for shared memory within a block. That covers the store/store case and a read of another thread's store; sequential execution would otherwise let the order the threads happen to run in satisfy it. v0's "later store wins" was sound for one thread only.

Declined by name, not guessed: a `Barrier` statement (PR 3), reading a `Constant` allocation (`AllocDecl` has no initializer), and anything `run_kernel` already declines. `run_kernel` still runs only unlaunched kernels and now says to use `run_launch` for the others. `run_launch` on an unlaunched kernel is `ArgMismatch`.

Tests: `tests/kernel_launch.rs`; each trap has a program that raises it and its nearest legal variant. 16 source mutants of the driver each turned a named test red; two survived at first (a "several readers" bit that cannot fire without barriers, now removed with a note for PR 3, and a shared-read race test that accepted either of two traps, now exact).

## `0.34.0` — feat (BREAKING): execution ids, launch shape, memory-space allocations and `Barrier` in the kernel layer (U3 row 4, PR 1: types and checker)

**Breaking:** `ir::ScalarExpr` and `ir::DagNode`/`DagKey` gain one variant, `Exec(exec::ExecId)` (decision D-R4-1: one leaf, a `#[non_exhaustive]` payload), and `kernel::Stmt` gains `Barrier { scope }`. Every exhaustive `match` on them needs an arm (inside this workspace: `backend`, `cfamily`, `contract`, `jit`, `optimize`, `oracle`, `pattern`, `plan`, `recipe`, `types`, `kernel_lift`, `unpopped-slang`). An `Exec` leaf is opaque like `Local`: never an `Input`, never folded, typed `U32` (D-R4-2), refused by the plan gate (`types::has_local` now covers it), declined by every `Backend` with a typed decline, and declined by the OpDef lifts (`UnsupportedLeaf("Exec")`).

**Additive:** new module `exec` (`Scope`, `ExecId`, `LaunchShape`, `SubgroupSize`, `MemorySpace`, `Extent`, `AllocDecl`); `KernelFn::{launch, allocs}` (both `#[serde(default)]`; `KernelFn::new` sets them empty, so every existing producer is unchanged) with `with_launch`/`with_allocs`; `kernel::Place::Alloc { alloc, index }` (arrays by id, no pointer values: D-R4-3); eleven `KernelError` variants (`ExecWithoutLaunch`, `ExecAxisOutOfRange`, `BadLaunch`, `SubgroupIdWithoutSubgroup`, `AllocSpace`, `AllocExtent`, `WriteToConstant`, `SharedWithoutLaunch`, `BadBarrierScope`, `BarrierDivergent`; a `Barrier` in an unlaunched kernel is `ExecWithoutLaunch`). `check_kernel` moved to a private `kernel_check` module and is re-exported from `kernel` unchanged.

**D-R4-5 over-rejects, on purpose.** `BarrierDivergent` is a conservative taint rule (every `Exec` read and every global, shared or private load is divergent; control and loop-carried flow are followed to a fixed point). It never accepts a divergent barrier and refuses some uniform programs; the four the design note names are pinned in `tests/kernel_barrier.rs` as `rejected_1..4`, so a relaxation is a deliberate test flip.

**Not run yet.** The interpreter traps `KernelTrap::Unsupported` (by name) on a kernel with a launch or allocations; `run_launch` is PR 2 and 3 of the row. A kernel can *check* and not *run* for now.

**Migration note for the two consumers:** re-vendor for the two new variants; the leaf list is `ScalarExpr::Exec(ExecId)`, `DagNode::Exec`, `Stmt::Barrier`, `Place::Alloc`, `KernelFn.launch/allocs`. No change to `ElementKind`, `OperandDesc`, `ArchSku`, `Frontend` or `Backend`.

Tests: `tests/kernel_exec.rs` (every checker rule with a one-change mutation and a control) and `tests/kernel_barrier.rs`; each rule was also mutated in the source and caught.

## `0.33.1` — docs: U3 row 4 design note, execution scopes and memory spaces

`docs/u3-row4-scopes-and-memory-spaces.md`: one `ScalarExpr::Exec` leaf, `LaunchShape`, `Place::Alloc` arrays (no pointer values), `Stmt::Barrier` with a conservative uniformity rule, and the per-thread driver, as one breaking increment (0.33.0). Design only; no code. The barrier-uniformity check (D-R4-5) is conservative and will reject some correct programs; the cases are listed in the note for a later relaxation row.

`SECURITY.md` added with the supported-versions text: the latest line gets security fixes, an older line only when the backport is small, no period promised. Lines 0.25.x-0.32.x are not supported by default under it.

## `0.33.0` — feat (BREAKING): `Frontend` is `#[non_exhaustive]`; the declared `ResourceMap` and the `lift_*_with` entry points (U3 row 3, #113)

**Breaking:** `convert::Frontend` is now `#[non_exhaustive]`, so a struct literal no longer compiles outside this crate. Build one with the new `const fn Frontend::new(name, kernel_marker, residue, out_name, in_prefix, parse)` (today's six fields, in order); `SLANG` is unchanged. **Migration:** Baracuda's `baracuda-cuda-parse` `CUDA` const (`Frontend { .. }` at `lib.rs:72`) becomes one `Frontend::new(..)` call. Fuel builds no `Frontend` (a grep of `fuel/crates` finds none).

**Additive:** `resource_map::{ResourceMap, DeclaredBuffer, ResourceMapError}`; `convert::{lift_elementwise_with, lift_reduction_with, lift_scan_with, lift_with}` taking `Option<&ResourceMap>` (the four existing `lift_*` keep their signatures and call them with `None`); `LiftProvenance { Declared, Inferred }`; `LiftError::{UndeclaredResource, BadResourceMap}` (`LiftError` was already `#[non_exhaustive]`, both map to `UnrecognizedButExpressible`). With a map, the output is the one `Write`/`ReadWrite` buffer, operand `K` is the `K`th `Read` buffer in declaration order whatever the source calls it, and a declared `Params` field is `ScalarExpr::Param(K)`; a name the map does not declare is `UndeclaredResource`, never a fallback to inference. `Lifted` gains `provenance` and `resources` and becomes `#[non_exhaustive]` (a grep of `baracuda/crates` for `Lifted {` finds no consumer literal).

**Migration note for the two consumers (PM requirement c):** no change to `ElementKind`, `OperandDesc`, `ArchSku` or `ScalarExpr` (`git diff 4b5bedc -- crates/unpopped/src/ir.rs crates/unpopped-vocab/src` is empty), so Fuel's contact is untouched and Baracuda needs no new `ScalarExpr` arm or typed decline from this row; row 4 carries the new leaves. No array members exist in Fuel's measured corpus (147 kernels, 635 `Params` fields, none array, vector or nested), so std140 and scalar layout agree today; `ParamField` cannot represent one, a limit stated for future corpora. Tests: `tests/resource_map_lift.rs` (9). Mutation-checked, each killed by named tests: operand index reversed (3 tests), provenance hard-coded `Declared` (2), the undeclared-name branch replaced by fallback (2), `ResourceMap::check` skipped (1); `position` to `rposition` is an equivalent mutant (names are unique) and was replaced by the reversal. Proposed number; the PM allocates (0.33.0 first of the 0.33.x release, row 4 follows as 0.33.1).


## `0.32.27` — docs: the U3 row 3 design note, the resource map and the `Frontend` change (#113)

`docs/u3-row3-resource-map-and-frontend.md`, docs only, no code: `Frontend` goes `#[non_exhaustive]` with a `const fn new`; a per-call `ResourceMap` (buffers plus a `Params` layout whose `LayoutRule` has no default) enters through `*_with` siblings of the four `lift_*` entry points, which keep their signatures; `Lifted` records `Declared` or `Inferred`; the layout rule is data in the map, not a `Frontend` field. Names no change to `ElementKind`, `OperandDesc`, `ArchSku` or `ScalarExpr`. The PM's rulings on its open questions are in its section 6. Proposed number; the PM allocates.

## `0.32.25` — test: runtime base offsets on the gather and scatter lifts (U2, #113)

`tests/kernel_gather_scatter_offset_agrees.rs`: a gather and a scatter-add with `BaseOffset::Runtime` on every operand agree with the oracle on offset, junk-padded buffers, leave the padding untouched, and disagree once any one input's offset is dropped. Test and docs only; no lift change (the offset is the argument buffer's `base_offset`, so the lifted kernels were already offset-free).

## `0.32.24` — feat (additive): `bincount` lifts (U2, #113)

`lift_scatter_opdef` now lifts `OpDef::bincount`, the one scatter whose output dtype is not its compute dtype: one integer input that is also its own index, a constant body, and an `I32` count destination, lifted as `out[x[i]] += ConstInt(c)` at the count dtype (`tests/kernel_bincount_agrees.rs`, 3: `I32` and `I64` data against the oracle, the in-range total and the unhit bin as controls, and the declines). Any other per-output dtype declines (`"a per-output dtype that is not a count"`: a non-integer constant, a non-add combine, a float count). Writing the test found the oracle bug fixed in 0.32.20 (`bincount` counted only the first `bins` elements of its input). Mutation-checked, each killed by a named test: the increment off by one, the add-only guard removed, the destination local typed as the compute dtype, and the count path bypassed. No trait or struct shape change.

## `0.32.23` — feat (additive): `lift_scatter_opdef`, the scatter lift (U2, #113)

`lift_scatter_opdef` lifts an `Access::Elementwise` op whose output is a `WriteIndex::ScatterIndexed` write (`scatter`, `scatter_add`, `index_add`; `Assign`, `AtomicAdd`, `AtomicMax`, `AtomicMin`; index dtype `I32`/`I64`/`U32`; full-shape and 1-D indexes on axis 0 and 1), byte-identical to the oracle (`tests/kernel_scatter_agrees.rs`). The iteration is the source's; the destination extent along the axis is one more scalar parameter; out-of-range targets are skipped; the atomics are a serial read-modify-write in the oracle's order, so even a duplicate target under `Assign` agrees (last writer, same order). The kernel does not zero the destination (the caller owns the start value; a test adds into a destination holding 100). New `LiftDecline::NotScatter`; a float atomic max/min, a non-`Skip` policy and `bincount` (per-output dtype) decline with a reason. Mutation-checked, each killed by named tests: `CmpGe` to `CmpGt`, `CmpLt` to `CmpLe`, the add combine replaced by a max, the max and min combines swapped, and the destination written at the counters instead of the index. No trait or struct shape change. Proposed number (after #173's 0.32.18); the PM allocates.

## `0.32.22` — feat (additive): `lift_contraction_opdef` accepts views on its factors (U2, #113)

A matmul whose operands carry `View::Permute` (what a planner emits for a transposed factor) now lifts instead of declining. The oracle reads a contraction's factors through the bound operand's own strides and never reads the view, so the view is recognition metadata and the kernel is the view-free one; this is the opposite of the Elementwise lift (#113 views), where `Permute` is lift code, and reordering coordinates here would transpose twice. A view on a fused bias, which the oracle does read, declines (`"a view on a fused bias"`). `tests/kernel_contraction_view_agrees.rs` (4): plain and batched matmuls with either or both factors transposed and declared by a `Permute`, the kernel byte-identical to the view-free op's, a control that the same bytes bound with the wrong strides change the product (the binding carries the transpose), and the bias decline. Mutation-checked: the bias-view check skipping the bias (1 red), views declined again (4 red). No trait or struct shape change. Proposed number (after the Elementwise views at 0.32.21, which this stacks on); the PM allocates.

## `0.32.21` — feat (additive): `lift_elementwise_opdef` lifts input views (U2, #113)

An Elementwise op whose inputs are read through `View::Permute`, `Layout`, `Broadcast` or `Reshape` now lifts. `Permute` is lift code: iteration axis `d` indexes producer axis `perm[d]`, so the producer coordinate on axis `perm[d]` is the loop counter of axis `d` and the producer buffer is bound as it is. `Layout`, `Broadcast` (stride 0), `Reshape` and a flipped operand are binding contracts, like a base offset: the caller binds the view's extents and strides on the same storage, and the kernel is the identity one. Every other lift still declines a view (`"a non-identity view"`) because it would ignore it, and a malformed one (wrong length, not a permutation) declines `"an ill-formed view"`. `tests/kernel_view_agrees.rs` (6): rank-2 and two rank-3 permutations (one not its own inverse, which is the only thing that tells the two directions apart), a strided `Layout` slice with a base offset, a broadcast, a same-rank reshape, a flipped operand, `F32`/`F64`/`I32`, and the declines; a control reruns each permute case with the view dropped (traps or differs). Mutation-checked: the permute ignored (2 red), the permutation applied backwards (1 red, the rank-3 case only), the validity check removed (1 red). Open: a view on a Contraction. No trait or struct shape change. Proposed number (after #174's 0.32.20, which this stacks on); the PM allocates.

## `0.32.20` — fix: `oracle::evaluate` on `OpDef::bincount` counted only the first `bins` elements of its input

**Affected: every release from `v0.8.0` through `v0.32.1`** (the line is present in the tags `v0.8.0`, `v0.8.1`, `v0.24.0`, `v0.25.1` and `v0.32.1`, and absent in `v0.7.0`; not bisected between those two). `eval_elementwise` takes the iteration space of a scatter from its *source*, and named the source as `operands[usize::from(idx_op == 0)]`, "the input that is not the index". `bincount` has one input and it is its own index, so that expression named operand 1, the **output**: the oracle walked `bins` cells instead of `len(x)` and counted only the first `bins` elements (14 values into 6 bins gave `[2, 1, 2, 0, 0, 1]`; the right answer is `[4, 2, 3, 0, 0, 1]`). `scatter`, `scatter_add` and `index_add` are unaffected, having a separate value operand. A scatter with a single input now takes that input as its source. This is the very case the comment on that line names ("bincount is the extreme: thousands of elements into a handful of bins") and it had no test; `bincount_counts_its_whole_input_not_the_first_bins_elements` has 14 values into 6 bins and failed before the change with exactly the wrong counts above. **What this changes for a consumer:** the oracle's *answer* for a `bincount` whose input is longer than its bin count; anything that compared a bincount kernel against this oracle at that size was comparing against a wrong reference (a kernel that counted correctly would have been reported as the one in error). The emitted code is unchanged. Found while lifting `bincount` to the kernel layer, where the lifted kernel disagreed with the oracle and was the one that was right. No API change. Proposed number; the PM allocates.

## `0.32.19` — feat (additive): the Elementwise-class lifts accept runtime base offsets (U2, #113)

`check_plain` no longer declines an Elementwise op carrying `BaseOffset::Runtime` on its inputs and/or output. The IR defines the offset as a launch argument added to the operand's base before any address math; the kernel layer's argument buffer already carries that (`TypedBuffer::base_offset`, applied by the interpreter to every access), so the lifted kernel is the offset-free one and the caller binds the offset in the buffer. The plan admits an offset on Elementwise only, so any other access still declines it (`"a runtime base offset outside Elementwise"`), as does a `base_offsets` whose length is not `n_inputs`. `tests/kernel_base_offset_agrees.rs` (3): offsets on every operand at `F32`/`F64`/`I32`, on one operand only, and the declines; each buffer sits in junk padding on both sides, the output's padding is checked untouched, and a control reruns the kernel with an input's offset dropped to show the agreement depends on it. Mutation-checked: the interpreter ignoring a buffer's `base_offset` (2 red), the non-Elementwise decline removed (1 red) and the length check removed (1 red). Open: how a source-emitting backend carries the offset (the U3 seam, no argument ABI). No trait or struct shape change. Proposed number; the PM allocates.

## `0.32.18` — feat (additive): `lift_gather_opdef`, the gather lift; the oracle's `ZeroFill` no longer panics on a float output (U2, #113)

`lift_gather_opdef` lifts an `Access::Elementwise` op with one `ReadIndex::Indexed` input (gather, `index_select`, embedding) at any index dtype `I32`/`I64`/`U32` and every `OobPolicy`, byte-identical to the oracle (`tests/kernel_gather_agrees.rs`: full-shape and 1-D broadcast indexes, axis 0 and 1, `F32`/`F64`, a body over a second input, an index corpus of negatives, one-past-the-end and the index type's extremes). The source's extent along the gathered axis is one more scalar parameter; the other lifts still decline a gather (`check_plain`), and a plain op declines with the new `LiftDecline::NotGather`. **Fixed on the way:** the oracle's `ZeroFill` stored `Val::Int(0)` whatever the output dtype, so an `OobPolicy::ZeroFill` gather (embedding's own policy) with a float output panicked in `encode_int` ("encode_int to unsupported dtype F32"); it now stores a float zero for a float output. The oracle zeroes its output, so `Skip` and `ZeroFill` are indistinguishable against it; `policies_differ_on_a_prefilled_output` checks them against their definitions on a pre-filled output. Mutation-checked, each killed by a named test: `CmpGe` to `CmpGt`, `CmpLt` to `CmpLe`, the data read at the counter instead of the clamped index, the clamp's `Min` to `Max` (3 red each), and the `ZeroFill` store dropped (`policies_differ_on_a_prefilled_output` only, as the oracle cannot see it). No trait or struct shape change. Proposed number (after #172's 0.32.17); the PM allocates.

## `0.32.17` — feat (additive): integer windows lift at `I32` and `I64` (U2, #113)

`lift_window_opdef` now lifts `Sum`, `Max` and `Min` pools at `I32` and `I64` (the accumulator is the dtype, as for integer reductions and scans), byte-identical to the oracle over three geometries that pad, stride and dilate (`tests/kernel_int_window_agrees.rs`: wrap-around sums, an all-negative `Max` window and an all-positive `Min` window as controls read from the oracle). An integer `Mean`, a fused post at an integer dtype and the narrower integers decline with a reason. **`RowReduce` stays float-only by the plan's rule** (`RowReduce requires a float dtype`), so there is no integer row reduction to lift; and the plan refuses a window made only of padding (`2*pad_lo <= span`), so the lift's no-valid-tap `Max`/`Min` identity is unreachable through it and is not tested. Mutation-checked: the accumulator start `0` to `1` and an `F64` result local at an integer dtype each turn the new tests red; routing the `Max`/`Min` fold's `int` flag to `false` is an equivalent mutant (an integer is never NaN, so the NaN arm it keeps is dead). No trait or struct shape change. Proposed number; the PM allocates.

## `0.32.16` — feat (additive): the U3 artifact/manifest types and the capability set, standalone (U3 rows 1 and 2)

New modules `unpopped::artifact` (`KernelArtifact` `Source`|`SpirV`, `KernelManifest`, `EntryManifest`, `Binding`, `ParamsLayout`, `ToolchainRecord`, `LayoutRule` with no default; `KernelManifest::check`) and `unpopped::capset` (`CapabilitySet`, `Capability`, `CapKey`, `CapValue`, `CapSource` reported/measured/declared, `DenormMode` defaulting to `Unknown`). Types only: the seam hold is in force, so nothing touches `GeneratedKernel`, `Backend` or `Frontend`, and no backend reads them. An unrecorded capability is unknown, never denied (`dtype_support` answers `NotExpressible`); a capability key is recorded once. The variants of `Packing`, `BindingKind` and `CapKey` are the spec sketch, not a ruling. `Params` members of a type with no plain scalar size (FP8, MX) are refused, not skipped. `tests/artifact_and_capability_types.rs` (14 tests); ten seeded mutants, all red. Proposed number; the PM allocates.

## `0.32.14` — feat (additive): integer reductions and scans lift at `I32` and `I64` (U2, #113)

`lift_reduction_opdef` and `lift_scan_opdef` now lift `Sum`, `Prod`, `Max` and `Min` at `I32` and `I64` (every scan direction and exclusivity, every reduction axis set), byte-identical to the oracle (`tests/kernel_int_reduce_scan_agrees.rs`: wrap-around sums and products, an all-negative `Max` row, an all-positive `Min` row, ties). The accumulator is the dtype itself, so the oracle's exact fold with wrapping and the kernel's agree; the scan identities `0`, `1` and the type's extremes are `ConstInt`. An integer `Mean`, a fused post at an integer dtype and the narrower integers decline with a reason (`UnsupportedForm`). The window and row-reduce lifts stay float-only. Mutation-checked, each killed by a named test: scan `Max` identity `lo` to `0`, `Min` identity `hi` to `0`, `Prod` identity `1` to `0` (all `integer_scans_agree_in_every_direction_and_exclusivity`), reduction `Prod` start `1` to `0` (`integer_reductions_agree_over_each_axis`), and an `F64` accumulator for an integer dtype (4 red). No trait or struct shape change. Proposed number; the PM allocates.

## `0.32.13` — feat (additive): `I32` row sorts lift, and the kernel interpreter compares integers (U2, #113)

`lift_row_sort_opdef` now lifts an `I32` `RowSort` (every `SortOut`, both orders, full and `TopK`), byte-identical to the oracle (`tests/kernel_row_sort_agrees.rs`: `integer_sorts_agree_in_every_form`, plus a control that the corpus holds both extremes and a tie). Other dtypes decline: `I64` plans but the oracle orders keys through `f64`, which collapses large 64-bit keys, so an exact kernel would disagree; the plan gate does not admit the other integers. The kernel interpreter's `eval_int` gains the six comparisons (1 or 0 at the compute type, operands at their own integer type), which an integer-compute kernel needs for any `If`/`Select` condition. Mutation-checked: `<` to `<=`, `<=` to `<` and `!=` to `==` in the new arm each turn `integer_sorts_agree_in_every_form` red. No trait or struct shape change. Proposed number; the PM allocates.

## `0.32.12` — feat (additive): the fused per-column bias in `lift_contraction_opdef` (tolerance mode, U2, #113)

`lift_contraction_opdef` now lifts `OpDef::contraction_bias` (a plain matmul whose epilogue reads `Input(2..n_inputs)`, a `[N]` bias read at the output column and broadcast over the rows, the oracle's rule) instead of declining `a fused bias input`. The kernel takes each bias as a rank-1 `Read` buffer after `out` (`Input(2)` is buffer 3), so a bias-free contraction keeps its parameters. Same tolerance mode as the other fused epilogues: byte-exact at `F64` compute, at most 2 ulp at `F32`; an ill-conditioned bias states its own bound (`tests/kernel_contraction_agrees.rs`: `a_cancelling_bias_states_its_own_bound`, `s + b` with `b = -0.9 s`, `k = |s / (s + b)|`, bound `ceil(k / 2) + 1`, and the default bound is shown broken so the stated one is needed). An epilogue reading `Input(0)`/`Input(1)` or past `n_inputs`, or a batched contraction with a bias, still declines (`UnsupportedLeaf` / `UnsupportedForm("a batched fused bias input")`). Mutation-checked, all red: the bias read at the row coordinate (3 red), the bias bound to `0` (3 red), the K-sum accumulator rounded to `F32` each step (10 red, 3 of them bias tests). Declines that stay, with the reason: `Mean` scan, `Prod` row-reduce stage and `Prod` window are rejected by the plan builder and the oracle (`plan.rs` Scan combine, `oracle.rs` "RowReduce Prod stage is unsupported", "Window rejects Prod"), so there is nothing to agree with. Proposed number; the PM allocates.

## `0.32.11` — feat (additive): fused post-expressions for `lift_window_opdef` (tolerance mode, U2, #113)

`lift_window_opdef` now lifts a 1-D `Window` pool with a fused post (e.g. a scaled and shifted `Mean`, a `sqrt` of a `Sum`) instead of declining. `Reduced(0)` is bound to the finalised pool value (the `F64` fold, `Mean` divisor and `Max`/`Min` identity as before) rounded once to the compute dtype (`docs/ir-kernel-layer.md` 7.1): byte-exact at `F64` compute, at most 2 ulp at `F32` (NaN equal, infinities exact). A post that reads an `Input` declines (`UnsupportedLeaf`). `tests/kernel_window_post_agrees.rs` (6): `F64` byte-exact and `F32` within 2 ulp for five posts over all four pools and both `Mean` divisors; a control that the tolerance is not vacuous; a wide-accumulator case (`1e8 + 1 - 1e8 ...`, exact value asserted); NaN/infinity cells; the `Input`-in-post decline. Mutation-checked: the post ignored (4+ red), the post bound to the wrong local (4+ red), an `F32`-rounded accumulator inside the tap fold (1 red, `a_wide_accumulator_is_required_under_a_post`; my first corpus for that test (`1, -1e8, 0.5`) let this mutant survive because both roundings land on the same `F32`, so the case is now `1e8, 1, -1e8` inside one window). No trait or struct shape change. Proposed number (assumes #166 takes 0.32.10); the PM allocates.

## `0.32.10` — feat (additive): `lift_row_sort_opdef`, the RowSort lift (U2, #113)

`lift_row_sort_opdef` lifts `Access::RowSort` (`row_sort`, `row_argsort`, `row_sort_indices`, `row_topk`, `row_bottomk`; `F32`/`F64`; ascending and descending; full and `TopK`) to a rank-sort kernel that is byte-identical to the oracle on a corpus of ties, NaNs with distinct payloads, signed zeros and infinities (`tests/kernel_row_sort_agrees.rs`). Integer keys and unstable sorts decline (`LiftDecline::UnsupportedForm`); a non-sort op declines with the new `LiftDecline::NotRowSort`. Additive: one new function and one new variant of a `#[non_exhaustive]` enum.

## `0.32.9` — docs: the U3 spec, execution scopes, memory spaces, resources, `ProgramModel` and the Frontend/Backend seam against Slang's needs (#114)

New `docs/u3-program-model-and-seam.md`. Section 1 lists what the seam assumes today and which of it is CUDA-shaped (source-text artifact, no argument ABI, a dtype-only capability query, name-convention buffers, substring kernel marker and residue list), each with its evidence. Section 2 proposes the U3 data model on top of the kernel layer (scopes and ids, memory spaces, resources, `ProgramModel`, a capability *set* with a subgroup range and reported/measured provenance). Section 3 records the five seam positions with Fuel's review of each (Fuel#514): per-entry manifest with the `Params` layout rule and the toolchain record, a declared resource map with `inferred` provenance, capability set, a measured conformance record per (device, driver, kernel) besides the flags, and Fuel owning the layout dialect as data. Section 4 sketches the types; section 5 lists what changes in the traits when this lands and what each costs; section 6 marks the unknowns UNVERIFIED (Fuel: the `Params` layout rule for 4 kernels; Vulkane: whether the `vulkan:` token set can carry a subgroup range). Docs only; no trait edit before this merges (CireSnave's hold). Proposed number; the PM allocates.

## `0.32.8` — feat (additive): fused epilogues for `lift_contraction_opdef` (tolerance mode, U2, #113)

`lift_contraction_opdef` now lifts a matmul (plain or batched) with a fused epilogue over the K-sum instead of declining: `Reduced(0)` is bound to the `F64` K-sum rounded once to the compute dtype (spec 7.1), byte-exact at `F64` compute and at most 2 ulp at `F32` for a well-conditioned epilogue. An epilogue that reads an `Input` declines (`UnsupportedLeaf`); a fused bias input still declines. `tests/kernel_contraction_agrees.rs` gains 7 tests (the last two state a bound of their own, below): `F64` byte-exact for a scale-and-shift, `tanh` and `sigmoid`; `F32` within 2 ulp; a control that the tolerance is not vacuous; a wide-accumulator case under an epilogue; a batched epilogue. **Finding recorded in the spec (7.1):** the 2 ulp bound only holds for well-conditioned epilogues. `0.37 s + 1` (cancelling near `s = -2.7`) moved by 71 ulp and `silu` at `s = -5` by 3 ulp on this corpus, from one rounding of the K-sum; neither is a lift defect, so the corpus keeps clear of them and the spec says to do the same or state a larger bound with its calculation. `tanh` agrees exactly because the oracle evaluates it at the compute dtype too, so it is no probe of the tolerance. `a_cancelling_shift_states_its_own_bound` and `silu_states_its_own_bound` give each ill-conditioned example the bound `ceil(k / 2) + 1` per cell (`k` the condition number) and show both sides: every cell is inside it, the default 2 ulp is broken somewhere, and the bound is within three times the worst cell (checked by mutating the bound: divided by four, the cells exceed it, 71 and 3 ulp against 22 and 2; multiplied by twenty, it is no longer within three times the worst cell). Mutation-checked: an `F32`-rounded accumulator inside the `k` fold (8 red). No trait or struct shape change. Proposed number; the PM allocates.

## `0.32.7` — feat (additive): fused post-expressions for `lift_scan_opdef` (tolerance mode, U2, #113)

`lift_scan_opdef` now lifts a `Scan` with a fused post (`OpDef::scan`) instead of declining. At each position the prefix is rounded once to the compute dtype (`red`), the post reads it as `Reduced(0)` and the elements loaded for that position as `Input(k)`, in both directions and both exclusivities (the exclusive write still happens before the fold). Byte-exact at `F64` compute, at most 2 ulp at `F32` (NaN equal, infinities exact), per `docs/ir-kernel-layer.md` 7.1. `tests/kernel_scan_post_agrees.rs` (5): `F64` byte-exact and `F32` within tolerance for a `sqrt` of a running sum, a prefix divided by `1 + x*x` (reads the element) and a halved running `Max`, in all four direction/exclusivity combinations; a control that the tolerance is not vacuous; a wide-prefix case (`1e8 + 1 - 1e8 + 0.5`); NaN/infinity cells. Mutation-checked: `Input` bound to the wrong local (3 red), the exclusive/inclusive write order swapped (5), an `F32`-rounded prefix inside the fold (2). No trait or struct shape change. Proposed number; the PM allocates.

## `0.32.6` — feat (additive): fused post-expressions for `lift_reduction_opdef` (tolerance mode, U2, #113)

`lift_reduction_opdef` now lifts a `Reduction` with a fused post (`OpDef::reduction_post`, e.g. `norm2 = sqrt(Sum(x*x))`) instead of declining. `Reduced(0)` is bound to the finalised fold rounded once to the compute dtype (the tolerance mode of `docs/ir-kernel-layer.md` 7.1): byte-exact at `F64` compute, at most 2 ulp at `F32` (NaN equal, infinities exact). A post that reads an `Input` declines (`UnsupportedLeaf`), as the oracle's plan gate does. New private `kernel_lift/post.rs` holds the shared `Reduced`-binding helpers (`lift_row_reduce_opdef` now uses them). `tests/kernel_reduction_post_agrees.rs` (6): `F64` byte-exact for `norm2`, an RMS norm and a scaled `Max`; `F32` within 2 ulp; a control that the tolerance is not vacuous; a NaN/infinity row; an order/width case (`1e8 + 1 - 1e8 ...`); the `Input`-in-post decline. Mutation-checked: `Mean` division dropped (3 red), the post unlifted (4), an `F32`-rounded accumulator (2). One of my first mutants (rounding the accumulator once after the loop) was equivalent to the lift's own rounding and survived honestly; the replacement rounds inside the fold. No trait or struct shape change. Proposed number; the PM allocates.

## `0.32.5` — feat (additive): `lift_row_reduce_opdef`, the first tolerance-mode lift (U2, #113)

New `kernel_lift::lift_row_reduce_opdef` (in `kernel_lift/row_reduce.rs`) lowers a float `Access::RowReduce` (staged `Sum`/`Mean`/`Max`/`Min` row folds, then a full-width epilogue; `F32` or `F64`) with the oracle's folds (an `F64` accumulator, ascending order, NaN-keeping `Max`/`Min`, `Mean` divided in `F64`). **It is not byte-exact for a non-identity epilogue**: the oracle passes each stage value on as an unrounded `f64`, which the kernel language cannot state inside a compute-dtype expression, so the lift sets `Reduced(s) := Cast(f64 acc -> compute)` (PM ruling 2026-10-29, option a). At `F64` compute that is the identity (byte-exact); at `F32` it is held to at most 2 ulp (NaN equal, infinities exact), written in the test. `docs/ir-kernel-layer.md` gains section 7.1 stating the two modes and that the oracle is a tolerance referee (as for `eval_contraction`'s fused epilogue). Other forms decline with a typed `LiftDecline` (new `NotRowReduce`). `tests/kernel_row_reduce_agrees.rs` (8): identity epilogues byte-exact for all four stage ops; `F64` byte-exact for softmax, a centred quotient and an RMS norm; `F32` within 2 ulp for the same; a control that the tolerance is not vacuous (some row differs between the unrounded and rounded stage value); a wide-accumulator case; NaN/infinity rows; extrema over rows that never cross zero; typed declines. Mutation-checked: wrong stage register, mis-divided `Mean`, an `F32`-rounded accumulator, a perturbed per-element value, an extremum accumulator starting at the sum identity (this last one survived the first corpus and is what `extremum_rows_that_never_cross_zero_agree` is for). No trait or struct shape change. Proposed number; the PM allocates.

## `0.32.4` — feat (additive): `lift_window_opdef` (U2, #113)

New `kernel_lift::lift_window_opdef` (in `kernel_lift/window.rs`) lowers a float 1-D `Access::Window` pool (`Sum`, `Mean`, `Max`, `Min`; `F32` or `F64`; identity post-expression; pooled axis innermost) the way the oracle folds it: per output cell, the taps `p = o*stride - pad_lo + kk*dilation` that fall inside the input are loaded, mapped by `pre` and folded in ascending `kk` into an `F64` accumulator; `Mean` divides by `size` (`count_include_pad`) or by the valid-tap count (zero when there is none); `Max`/`Min` keep a NaN and fall back to the monoid identity when no tap was valid; one `Cast` down at the `Store`. The window geometry is baked in as constants; the input extents and `k_out` are `I32` parameters. A fused post-expression, `Prod`, a non-innermost axis and a non-float dtype decline with a typed `LiftDecline` (new `NotWindow`). `tests/kernel_window_agrees.rs` (7): byte equality with `oracle::evaluate` for all four pools, padding, stride, dilation, no padding, a NaN kept by `Max`/`Min`, an order-sensitive `f64` sum, and both `Mean` divisors (with a control that they differ on the corpus). Mutation-checked: the exclude-pad divisor replaced by `size` (3 tests red), the include-pad divisor by the count (1), pad sign (3), dilation ignored (1), the bound off by one (4+), the valid-tap count not incremented (3). Not killable by design: a window with no valid tap (the plan's `2*pad <= span` rule forbids it), so the empty-window identity path is exercised by no test. No trait or struct shape change. Proposed number; the PM allocates.

## `0.32.3` — feat (additive): `lift_im2col_opdef` (U2, #113)

New `kernel_lift::lift_im2col_opdef` (in `kernel_lift/im2col.rs`) lowers a 2-D `Access::Im2Col` at `F32` or `F64` into six nested `For` loops over `n, c, ki, kj, oh, ow` that copy one element per output cell `(n, c*kh*kw + ki*kw + kj, oh*oW + ow)` from `(n, c, oh*sh - ph + ki*dh, ow*sw - pw + kj*dw)`, leaving the typed zero where the tap is outside the input. The scalar layer has no comparison, so "inside" is tested by clamping: `max(0, min(i, extent-1)) - i` is zero exactly when `i` is inside. Window geometry is baked in as constants; the extents `N, C, H, W, oH, oW` are `I32` scalar parameters. Other dtypes and any other access decline with a typed `LiftDecline` (new `NotIm2Col`). `tests/kernel_im2col_agrees.rs` (7): byte equality with `oracle::evaluate` for padded 3x3, batch and channels, stride 2, dilation, a rectangular input and window with different parameters on the two axes, and `f64`, on a corpus of distinct values with a NaN and a negative zero planted. Mutation-checked: pad sign flipped, bound `extent` for `extent-1`, tap row and column swapped, dilation replaced by stride (each turned 3 or more tests red). No trait or struct shape change. Proposed number; the PM allocates.

## `0.32.2` — feat (additive): `lift_contraction_opdef` (U2, #113)

New `kernel_lift::lift_contraction_opdef` lowers a float matmul (`ContractionAxes::matmul()` or `batched_matmul()`, `F32` or `F64` compute, identity epilogue, two inputs) into a kernel that does what the oracle does: nested `For` over `[b,] m, n`, an inner `For` over `k` ascending, both factors cast up to `F64` and multiplied there (an `f32` product is exact in `f64`), an `F64` accumulator, one `Cast` down at the `Store`. Operand layouts are the operands' physical strides read at the logical `(m,k)`/`(k,n)` coordinate, which is what `Place::Logical` does, so transposed and broadcast operands need nothing extra. A fused epilogue or bias, a non-float contraction and any other `ContractionAxes` decline with a typed `LiftDecline` (new `NotContraction`). `tests/kernel_contraction_agrees.rs` (6): byte equality with `oracle::evaluate` for plain, transposed (column-major) and batched matmul on a corpus with f32-cancelling magnitudes, NaN, infinities and signed zeros; an f64-compute case with order-sensitive data (1e16, 1, -1e16, ...) that discriminates the `k` order; typed declines. Mutation-checked: a descending `k` loop (4 tests red), rhs coordinate roles swapped (3), accumulator starting at 1 (4). No trait or struct shape change. Proposed number; the PM allocates.

## `0.32.1` — docs: patent re-sweep for the kernel layer, packed values, MX dtypes, layout inverses/views and the U3 seam (2026-10-29)

`docs/PATENT-SWEEP.md` gains a re-sweep addendum (the 2026-10-25 text said to re-sweep at U2): 7 new ideas (16-22), 14 queries with the patent numbers each returned, 12 hit rows (10-21) and a prioritized list of independent claims for the PM to read (US10949182B2 first; the Apple unified-IR family re-checked against statements; address-space and variant-selection patents before U3 is built). The PM's claim reads (independent claims only, Google Patents text, 2026-10-28) are recorded as verdicts: hits 10-18 and 21 are all "outside as read"; hits 19-20 (variant selection, address spaces) are not covered because U3 is not built; the MX codec is not in the publish scope. Weak or null areas are named (a CPU interpreter as a differential reference, a compiler choosing a wider accumulator, IR-level packed types, layout inverses and views, a SPIR-V manifest). Docs only. Proposed number; the PM allocates.

## `0.32.0` — feat (BREAKING): `View::Layout { shape, stride }`, a view that carries extents (U6 option B, #117)

New `View` variant: iteration coordinate `c` reads element `base_offset + sum_d c[d] * stride[d]`. `shape` must equal the op's iteration extents (a caller precondition, checked by the oracle, since the structure key still abstracts extents away); `stride` is the truth for that input, so the operand's own runtime strides are not consulted and the operand must carry no key broadcast (write a broadcast as stride 0 in the view). Admitted only on `Access::Elementwise` in v1; strides `>= 0`, extents `>= 1`, rank must equal the iteration rank (`View::is_valid`, `assert_valid_views`); counts as address-affecting like `Permute` (`view_is_addressing`). It says what the four older variants cannot: a strided slice, a reshape of a dense producer to another rank, any one-mode-per-axis `Layout`. `unpopped::view_layout` converts both ways (`layout_to_view` now returns `View::Layout` for a flat layout no producer-relative variant matches, and `None` only for a nested one). The text codec is serde, so the new variant round-trips with no codec change. The oracle reads it (`ViewRead::Strides`); `kernel_lift` already declines every non-`Identity` view, unchanged.

**Downstream:** `View` is deliberately exhaustive, so every `match` over it needs a new arm. In particular **Baracuda's CUDA emitter match on `View` needs a new `View::Layout` arm** (read `offset = base + sum c[d]*stride[d]` instead of the operand's strides). Tests: `tests/view_layout_oracle.rs` (oracle differential over strided / reshaped / broadcast-as-stride-0 reads, four gate panics, round trip); eight seeded mutations (oracle consults operand strides, oracle reverses view strides, extents check skipped, negative stride accepted, rank ignored, not addressing, no axis reversal in the bridge, bridge extent check skipped) each turned a test red. Proposed number; the PM allocates.


## `0.31.13` — feat (additive): `U64` joins the integer table (U1, #112)

`U64` is an integer `CoordInt`/`ConstInt` leaf type, a `Cast`/`Bitcast` endpoint and an integer-arithmetic type (`types::is_index_int`, `is_cast_int`, `typed_bits` 64, `int_range` `0..=u64::MAX`). Arithmetic wraps at 64 bits zero-extended; a float cast to `U64` saturates at `u64::MAX` and NaN is 0. **Limit (PM ruling B, 2026-10-28):** a `ConstInt` literal's payload is still `i64`, so a `U64` literal is `0..=i64::MAX` and a negative one is a typed `IllegalIntLeaf` (never a wrap); larger values are reached by `Cast`, arithmetic or `CoordInt`. Widening the payload (breaking) changes the `Lowering::const_int` seam and waits for it. CpuC spells `U64` constants, arithmetic and `Bitcast` (compiled and compared with the oracle) but declines any `Cast` to or from `U64` with a reason, because its conversion helpers are `long long` based. Behaviour change for callers: `check_cast`/`check_bitcast`/`infer` of `U64` used to be `Err`.

## `0.31.12` — feat (additive): `lift_scan_opdef` (U2, #113)

New `kernel_lift::lift_scan_opdef` lowers a float `Access::Scan` (`Sum`, `Prod`, `Max`, `Min`; inclusive or exclusive; forward or reverse; innermost axis; `F32` or `F64` compute; identity post-expression only; `pre` may be any body over the inputs) into a kernel that does what the oracle does: an `F64` prefix local that starts at the monoid identity, one `Cast` up per pre-mapped element, the fold in the oracle's order (`Max`/`Min` keep a NaN), the prefix written before the fold (exclusive) or after it (inclusive), one `Cast` down per `Store`. The infinite identities are built from their bit patterns (`Bitcast` of an integer constant) so a scan kernel serializes to JSON. Typed `LiftDecline` for `Mean`, an integer scan, an outer-axis scan and a fused post (new `NotScan`). `tests/kernel_scan_agrees.rs`: byte-for-byte agreement with `oracle::evaluate` for 4 monoids x 2 directions x 2 exclusivities, a pre-map, on a corpus with f32-cancelling magnitudes, an all-negative row, NaN first and mid-row, infinities and signed zeros; JSON round trip. Mutation-checked: exclusive ordering, reverse, and the Max, Min and Prod identities each turned tests red when altered. No trait or struct shape change. Proposed number; the PM allocates.

## `0.31.11` — feat (additive): `unpopped::view_layout`, `View` as a `Layout` (U6, #117)

New module, no change to `View`, the key or the text codec. `view_to_layout(view, iter_shape, producer_strides)` builds the `Layout` a view reads (modes in REVERSE axis order, since the oracle iterates last-axis-fastest and a `Layout` first-mode-fastest); `layout_to_view(layout, producer_strides)` returns the one of `Identity`/`Permute`/`Broadcast` that reads identically, and `None` otherwise (nested layout, foreign stride, permute-and-broadcast mix, ambiguous permutation), never a nearest view. Negative (flipped) strides and rank-changing `Reshape` are declined. `tests/view_layout_oracle.rs` checks every rank-2/3 permute, every rank-3 broadcast mask, identity and same-rank reshape against `oracle::evaluate`; six seeded mutations of the recognition rules (axis order, permutation direction, broadcast stride-0 check, reshape strides, ambiguity, broadcast recognition) each turned a test red. Proposed number; the PM allocates.

## `0.31.10` — feat (additive): packed `Select` mask, lane-wise (U1 slice 4, #112)

`Select(c, a, b)` accepts a PACKED condition whose lane count equals the arms': lane `i` of the result is `a`'s lane `i` when `c`'s lane `i` is non-zero, else `b`'s. Each mask lane is tested `!= 0` in its own type exactly as a scalar `Select` condition is (an integer lane by value, a float lane rounded to the compute dtype first: `-0.0` is false, NaN true), and the mask's lane type may differ from the arms'. `types::infer_ty` types it; `types::lane_of` selects the mask's lane (a scalar condition is still the same in every lane); the oracle evaluates it through `lane_of`. A packed `c` over scalar arms or with a different lane count stays `PackedOperand`. No new variants, so additive; the plan gate still declines every packed body and no emitter spells it. Behaviour change for callers: `infer_ty` of a `Select` with a packed condition used to be `Err(PackedOperand)`.
## `0.31.9` — feat (additive): `Layout::select`, `take`, `slice` (with `SliceCrd`) and `tile_to_shape` (U6 part 3, #117)

`select`/`take` pick top-level modes; `slice(&SliceCrd)` fixes some coordinates and keeps others, returning the kept layout and the offset (`kept(j) + offset == self(full coordinate)`); `tile_to_shape(target)` replicates a block to cover `target` as `(block_i, tiles_i)` per dimension, tiles after the whole block (declines non-multiples with `NotDivisible`). New file `layout_algebra/slice.rs`; tests in `tests/layout_algebra_slice.rs`, each against a deliberately broken variant. Proposed number; the PM allocates.

## `0.31.8` — feat (additive): `lift_reduction_opdef`, and the interpreter's mixed-type fallback (U2, #113)

New `kernel_lift::lift_reduction_opdef` lowers a float `Access::Reduction` (`Sum`, `Prod`, `Max`, `Min`, `Mean`; last axis or an axis set; `keepdim` or not; `F32` or `F64` compute; identity post-expression only) into a kernel that does what the oracle does: an `F64` accumulator, one `Cast` up per body value, the fold in the oracle's order, the `Max`/`Min` NaN rule (`!have || v.is_nan() || better`), `Mean` as an `F64` division by the reduced extents, one `Cast` down at the `Store`. A fused post-expression, an integer reduction or a non-last-axis form the oracle also lacks decline with a typed `LiftDecline` (new variant `NotReduction`). The interpreter gains a node-by-node fallback for mixed-type float expressions: the oracle rounds every float operation to one dtype, but a reduction accumulates in `f64` inside an `f32` kernel, so each node runs at its operands' type; a pure compute-dtype expression keeps the oracle path. `tests/kernel_reduction_agrees.rs`: byte-for-byte agreement with `oracle::evaluate` over all five reduce ops, outer-axis and keepdim, f64 compute, a body over the input, on a corpus with f32-cancelling magnitudes, an all-negative row, NaN first and NaN mid-row, infinities and signed zeros. Mutation-checked: the NaN rule, the `have` flag, the Mean divisor, keepdim, the Prod identity and the mixed-type fallback each turned tests red when deleted. No trait or struct shape change. Proposed number; the PM allocates.
## `0.31.7` — feat (additive): `Cast` rows for the OCP MX elements `F4E2M1`, `F6E2M3`, `F6E3M2` (U1, #112)

`F4E2M1`/`F6E2M3`/`F6E3M2` are `Cast` endpoints (`types::is_mx_element`; `float_format` gains their `(ebits, mbits)`). PM ruling A (2026-10-28): a Cast INTO one requires `Saturation::ClampFinite` (`Wrap`/`Clamp` are `IllegalCast` naming the fix), rounds once from the exact value under any of the four `RoundMode`s, saturates every overflow and infinity to the largest finite value (e2m1 6, e2m3 7.5, e3m2 28), turns NaN into `+0` and keeps the sign of zero. **NaN to `+0` is the IR's rule; the NaN behaviour of the vendors' conversion instructions is UNVERIFIED, so an emitter that cannot match it must decline.** The oracle reference (`cast_ref::encode_mx`/`decode_mx`) is cross-checked against a brute-force search of the explicit value grid. An MX value may only be an intermediate directly under a `Cast`: a root or `out_dtype` of an MX type is a typed plan decline (no buffer layout yet), not an oracle panic. No emitter spells them. Behaviour change for callers: `check_cast(F4E2M1, ..)` used to be `Err`.
## `0.31.6` — feat (additive): `Layout::right_inverse` and `Layout::left_inverse` (U6 part 2, #117)

`right_inverse` returns `R` with `self(R(i)) == i` for every `i < size(R)`, chaining the modes whose stride equals the extent covered so far (the two-sided inverse for a bijection onto `[0, size)`); `left_inverse` of an injective layout is `right_inverse((self, complement(self, cosize(self))))` and declines what `complement` declines. Property tests in `tests/layout_algebra_inverse.rs`, each against a deliberately broken variant. Not yet: slicing, `select`/`take`, `tile_to_shape`. Proposed number; the PM allocates.

## `0.31.5` — feat (additive): `unpopped_vocab::layout_algebra`, hierarchical layouts and their algebra (U6, #117)

New pure module: `IntTuple`, `Layout` (`shape:stride`, hierarchical, validated), `LayoutError`, `Swizzle`, `ComposedLayout`. On `Layout`: `size`/`cosize`/`rank`/`depth`/`call`/`crd2idx`/`flatten`/`coalesce`/`composition`/`complement`/`logical_divide`/`logical_product`. No dependency on the IR, `OpDef`, or the Frontend/Backend traits; no `unsafe`, no I/O, checked arithmetic. Strides are `>= 0` (negative declined); `composition` uses CuTe's sufficient divisibility condition and declines some expressible cases. `View` is NOT re-expressed over it yet (later PR, needs the first lane's agreement). Tests: `tests/layout_algebra_properties.rs`, each property run against the real operation and against a seeded-bad variant that must fail. Proposed number; the PM allocates.
## `0.31.4` — docs: the kernel layer v0 spec (U2, #113)

New `docs/ir-kernel-layer.md`: a separate `kernel.rs` artifact (`KernelFn`, `Stmt`, `Place::Logical`) plus one `ScalarExpr::Local(u32)` leaf, the scope and type rules `check_kernel` enforces and the known-bad mutations that pin them, the reference interpreter's semantics, the effects model with its six unknowns marked, the `lift_opdef` plan (Elementwise by nested `For`, since the intent layer has no integer division), and the U2 choices that constrain U3's `ProgramModel` and resource manifest. Written before the code it describes; the code landed in 0.31.0 (types, `check_kernel`) and 0.31.3 (interpreter, Elementwise lift). Docs only. Proposed number; the PM allocates.

## `0.31.3` — feat (additive): the kernel interpreter and `lift_elementwise_opdef`, with the layers-agree differential (U2, #113)

New `oracle::kernel_eval` (`run_kernel`, `KernelArg`, `KernelTrap`): runs a `KernelFn` on one logical thread in plain Rust, reusing the oracle's own evaluator for compute-dtype expressions (so a lifted body is evaluated by the same code as the original) and a small wrapped-integer evaluator for bounds, coordinates and conditions. New `kernel_lift::lift_elementwise_opdef` lowers an Elementwise `OpDef` (single output, identity views, no gather/scatter, body of `Input`/`Const`/`ConstInt` and operators) into a nested-`For` kernel, or returns a typed `LiftDecline`. `tests/kernel_layers_agree.rs`: byte-for-byte agreement with `oracle::evaluate` over every unary and binary op on f32 special values (signed zero, NaN, infinities, subnormals) with a broadcast input, f64, wrapped i32 add/sub/mul, plus, in `tests/kernel_eval_semantics.rs`, the interpreter's own semantics (loop bounds read once, `Break`/`Continue`/`Return`, step limit, typed traps). Mutation-checked: stride handling, integer wrap, `Break` and `Continue` in `While` each turned a test red when deleted (the first corpus missed the last three; tests were added). No trait or struct shape change. Proposed number; the PM allocates.
## `0.31.1` — feat (additive): `Cast` rows for `F16` and `Bf16` (U1, #112)

`F16` and `Bf16` are now `Cast` endpoints, both directions, against `{f32, f32s, f64, i8..i64, u8..u32}` and each other (`types::is_cast_float_endpoint`, `float_format`; `is_cast_float` is unchanged and still means a compute float, so an `F16` compute dtype is still declined by the plan gate). `check_cast` accepts them with the float rules (float to float: `Wrap`/`ClampFinite`; float to int: `Clamp`; int to float: `Wrap`), `Bitcast` accepts `F16`/`Bf16` against 16-bit integers, and `rounding_matters` compares exponent range AND significand (f16 to bf16 and bf16 to f16 are both inexact despite equal width). The oracle reference rounds once from the exact value (`cast_ref::encode_ieee`, all four `RoundMode`s, overflow follows the direction, `ClampFinite` clamps infinities, integers above 2^53 go through round-to-odd so there is no double rounding); it is cross-checked against the hardware `f32` conversion and exhaustively over all 65536 patterns of each format. No emitter spells them yet: CpuC has no C type and declines, Slang is unchanged (baracuda/Fuel own those spellings). Packed elements `F16`/`Bf16` now type, still declined by the plan gate until a packed seam lands. Behaviour change for callers: `check_cast(F16, ..)` and `check_bitcast(F16, U16)` used to be `Err`.

## `0.31.0` — feat (BREAKING): `ScalarExpr::Local` and the kernel layer's types and scope checker (U2, #113)

New leaf `ScalarExpr::Local(u32)` (and `DagNode::Local`): a read of a kernel-layer local. It has meaning only inside a `KernelFn`, so `types::infer`/`infer_ty` refuse it (`TypeError::LocalOutsideKernel`, new variant), the plan gate declines an `OpDef` body containing one (`types::has_local`), and every emitter-facing site declines it (`DeclinedOp::Kernel`, new variant); `ScalarExpr` is exhaustive, so every downstream `match` must choose. `types::infer_ty_in(e, compute, local_types)` types an expression inside a kernel.

New module `kernel`: `KernelFn`, `Param`/`ParamKind`/`BufferRole`, `LocalDecl`, `Place::Logical`, `Block`, `Stmt` (`Let`, `Load`, `Assign`, `Store`, `If`, `For`, `While`, `Break`, `Continue`, `Return`) and `check_kernel`, the scope and type checker with 14 `KernelError`s, each pinned by a known-bad mutation in `tests/kernel_layer.rs` (three of the checks were deleted to confirm their tests go red). Per `docs/ir-kernel-layer.md` (#145): a separate artifact, no `OpDef`/`KernelPlan`/`Frontend`/`Backend` edit. `Stmt::Call` is deferred until there is a module type (spec unknown U-2); the interpreter and `lift_opdef` follow in their own PRs. Proposed number; the PM allocates.

## `0.30.0` — refactor (BREAKING): the CUDA front end leaves the neutral core (ROADMAP P2, #112)

Removed from `unpopped::convert` (feature `convert`): `parse_cuda`, the `CUDA` `Frontend` const, the private CUDA residue list, and the optional `tree-sitter-cuda` dependency (the feature is now `dep:tree-sitter`, `dep:tree-sitter-slang`). Baracuda owns CUDA in this ecosystem and already ships the same three items as `baracuda-cuda-parse` (0.24.0): migrate with `use baracuda_cuda_parse::{CUDA, parse_cuda}`. `Frontend`, `SLANG`, `parse_slang`, `lift_elementwise`/`lift_reduction`/`lift_scan`/`lift` and the eleven shared walkers are unchanged: **no trait or struct shape changes in this release** (the `Frontend`/`Backend` seam is held until the Slang requirements are in). The old P2 hold on "KISS registry repoint (b)" was stale: KISS#514 merged 2026-09-25.

Tests: the CUDA-source tests left with the code (11 are in `baracuda-cuda-parse/tests/convert.rs` with the same bodies; `slang_and_cuda_lift_to_the_same_ir` needs both grammars and moves there). Five walker tests that happened to use CUDA source (`elementwise_and_reduction_lifters_dont_cross`, `scan_reduction_elementwise_are_disjoint`, the single-assignment / reassigned-local pair and the ternary-to-`Select` test) are re-expressed on Slang so the shared walkers keep their coverage here; `a_caller_defined_frontend_lifts_through_the_shared_walkers` now borrows the Slang grammar. New `tests/neutral_core_names_no_cuda_grammar.rs` guards the absence (with controls that find what stays). Proposed number; the PM allocates. Not published.

## `0.29.0` — feat (BREAKING): OCP MX element dtypes `F4E2M1`, `F6E2M3`, `F6E3M2`; KISS corpus re-vendored to `bafbe07` (U1b, #112)

Three new `ElementKind` rows, appended after `U64` (`ALL` 25 -> 28; `ElementKind` is deliberately exhaustive, so every downstream `match` over it must choose: breaking pre-1.0). Tokens `f4e2m1` / `f6e2m3` / `f6e3m2` (KISS-CLASSIFY 6.1-0014..-0016), storage 4 / 6 / 6 bits, kind `float`, usable (not reserved), so the closed vocabulary is 27 tokens. `ElementKind::is_mx_element` is new. An MX element in an accumulator position typed-declines: new `TokenDecline::ElementDtypeAsAccumulator` (token `<acc>` and the `(acc + mp)` field, both the lenient and strict decode) and new `DeriveDecline::ElementDtypeAsAccumulator` (declared, or defaulted to operand 0's dtype for a contraction). Both are additive on `#[non_exhaustive]`/new-variant enums but the `DeriveDecline` addition is breaking for an exhaustive matcher.

Nothing computes with them: `plan::check_dtype_is_a_compute_dtype` declines them (typed `InadmissibleOpInCell`), CpuC and Slang report them unsupported, FKC has no spelling, and no `Cast`/codec row exists (a later slice; the OCP packing convention is unverified, ROADMAP U1b).

Vendored KISS artifacts re-taken from `bafbe07` (spans corpus commits `9336a5b`, `07ccc96`, `68dc233`, `bafbe07`; see `kiss/README.md`): `structure_key_vectors.json` 28 -> 34 positive and 25 -> 27 decline vectors, 66 -> 71 derivation vectors, `dtype_manifest.json` 24 -> 27 rows, and the `source_commit` member is gone upstream, so its assertion and shape pin were removed. Proposed number; the PM allocates.

## `0.28.0` — feat (BREAKING): `ScalarExpr::BitcastPacked`, scalar to packed reinterpretation (U1 slice 4 step 3b, #112)

New `ScalarExpr::BitcastPacked(x, elem, lanes)` (and `DagNode::BitcastPacked`) reinterprets the bits of a scalar as `Packed(elem, lanes)`, lane 0 least significant; `U32` to `Packed(I8, 4)` is the example. Total lane bits must equal the operand's width (`IllegalBitcast`), lanes are 2 or 4 (`BadLanes`), the operand must be scalar (`PackedOperand`). It is the inverse of step 3a's `Bitcast` of a packed value. `ScalarExpr` is deliberately exhaustive, so every downstream `match` over it must choose to handle or decline the new node (PM ruling (a), 2026-10-28). The oracle slices `Extract(BitcastPacked(..), i)` from the operand's bits (`types::lane_of` declines it, since there is no scalar shift node). The optimizer and pattern matcher treat it as opaque and the plan gate still declines every packed body. Proposed number; the PM allocates.

## `0.27.3` — feat (additive): `Bitcast` of a packed value to a scalar (U1 slice 4 step 3a, #112)

`infer_ty` accepts `Bitcast(packed, scalar)` when lane bits times lane count equal the target width (`Packed(I8, 4)` to `U32`; lane 0 least significant); new `types::check_packed_bitcast` names the refusal, and the oracle assembles the bits. A `Bitcast` of a packed value to a mismatched width is now `IllegalBitcast` (it was `PackedOperand`). The plan gate still declines every packed body. Scalar to packed is the later `BitcastPacked` node. Proposed number; the PM allocates.

## `0.27.2` — feat (additive): lane-wise operators on packed values (U1 slice 4 step 2, #112)

`infer_ty` types `Add`, `Sub`, `Mul`, `Fma`, `Min`/`Max`, a whole-value `Select` (scalar condition, packed arms) and `Cast` over packed operands lane-wise (same lane count, no broadcast, scalar rules on the lane types; every other operator still refuses with `PackedOperand`). New `types::lane_of` is the reference meaning of `Extract`; the oracle evaluates through it. The type rules now live in one table (`node_ty`) used by scalar and packed expressions alike; scalar results are unchanged. The plan gate still declines every packed body (no emitter seam yet). Proposed number; the PM allocates.

## `0.27.1` — docs: ROADMAP rows with no owner or blocker now name one (weekly sweep 2026-10-26)

E2-E11, I2, C3-C5, C7, C11, X3 and X7 said only "open"; each now names its owner and its trigger or blocker. No code change.

## `0.27.0` — feat (BREAKING): packed values, `Pack`/`Extract`, `Ty` and `infer_ty` (U1 slice 4 step 1, #112)

**Breaking:** `ScalarExpr` gains the variants `Pack(Vec<ScalarExpr>)` and `Extract(Box<ScalarExpr>, u8)` (and `DagNode` gains `Pack` and `Extract`); every exhaustive match over them must handle or decline the new nodes. Additive: `types::Ty`, `types::infer_ty`, `types::has_packed`, the `TypeError` arms `BadLanes`, `BadLane`, `NotPacked`, `PackedOperand`, `PackedResult`, and `DeclinedOp::Packed`. `infer` is unchanged for every existing body and refuses a packed result with `PackedResult`. The plan gate declines any body containing a packed node (no emitter has a seam yet), so no existing op plans differently. Design: `docs/ir-typed-expressions.md` section 9.

## `0.26.5` — feat (additive, KISS-CLASSIFY-6.8-0018): per-target vector-access byte cap (KISS-CLASSIFY-6.8-0018, KISS#674)

The `vec` width of `sk6` was capped at a literal 16 bytes. It is now the cap the key's target namespace declares: `CellFacts::vec_cap_bytes: Option<u32>` (`None` = `DEFAULT_VEC_CAP_BYTES` = 16). `cuda` and `vulkan` declare 16, so **no existing key moves**; the token set stays `{v1,v2,v4,v8}`. A caller keying a target that declares another cap sets the field; it is the namespace document's value, not a device query (`DeviceCapability::widest_vec_width` is legality on the running part and is unchanged). The vendored derivation corpus gains KISS's three `vec_cap_*` vectors (66 total); a cap that is ignored fails two of them.

## `0.26.4` — fix: integer routing in `lower_expr`/`lower_dag` types every operand (Pip review of #130)

`int_node` routed on the first operand's type only (a `Select`'s first arm), so an emitter that skipped the plan gate and lowered a mixed integer/float node was spelled by `arith`/`select`, or by `int_op` at the first operand's type: a miscompile. Every value operand is now typed; a mix is `Decline::UnsupportedOp { op: DeclinedOp::Integer }` naming the types. Well-typed nodes and the plan gate are unchanged.

## `0.26.3` — fix: Slang `lower()` decline text for i16/u16

The `unpopped-slang` decline text said i16/u16 "wait on vocabulary v5"; v5 shipped (vulkane 0.14.0), so it now says they require `i16` in the target's `<arith>` field. **Error text only, no behaviour change** (`supports_dtype` is unchanged). Adopters that match the old substring `wait on vocabulary v5` must match `require `i16` in the same field` instead. ROADMAP X5.

## `0.26.2` — docs: patent sweep record

`docs/PATENT-SWEEP.md`: a due-diligence patent sweep of the IR, lowering and backends at 0.25.1 (covers the 0.26.x line). Not legal advice. Independent claims of 13 patents read in full, none overlap on the crates (two Google kernel-caching patents charted, adopter runtimes flagged); four more listed unread. No code change.

## `0.26.1` — docs: layout/shape frontier (baracuda SA-9) folded into the union design

`docs/ir-union-design.md` section 8b decides output views, rank-changing reshape and general strides (U6), multi-K einsum and TF32 (U7). No code change.

## `0.26.0` — BREAKING: integer index expressions, `CoordInt`, `ConstInt` and integer arithmetic (I1 of #112)

**Breaking, pre-1.0 (second number).** New variants: `ir::ScalarExpr::{CoordInt, ConstInt}`, `ir::DagNode::{CoordInt, ConstInt}`, `backend::DeclinedOp::Integer`, `types::TypeError::{IllegalIntLeaf, IntegerOperand}`; new `Lowering` seams `coord_int`, `const_int`, `int_op` (default: decline) and `backend::IntOp`; every exhaustive `match` over `ScalarExpr`/`DagNode`/`TypeError` gains arms. New public items: `ir::{coord_int, konst_int}`, `types::{check_int_leaf, int_binary_ok, int_spine_type, is_index_int}`.

`CoordInt(axis, T)` is the exact output coordinate (no float round-trip, so no 2^24 bound) and `ConstInt(v, T)` an exact constant, `T` one of `I32`/`I64`/`U32`. Inside a float op, `Add`/`Sub`/`Mul` over integers wrap in two's complement at the type's width (never C's undefined signed overflow), `Min`/`Max` keep the type, the six comparisons yield the compute type's 1/0, and `Select` takes any condition and integer arms. `Div`, unary math, `Fma` and the other binary ops refuse an integer operand with `TypeError::IntegerOperand`; an integer COMPUTE dtype is unaffected. The plan gate admits an integer value anywhere `infer` accepts it (it was root-or-under-`Cast` only); the optimizer returns such a body unchanged. Spec: `docs/ir-typed-expressions.md` section 8.

cpu_c spells the constants and every operation (widening to an unsigned type for the wrapping forms); it has no strided schedule, so it declines `CoordInt` through the default seam, as it does `Coord`. **Baracuda's CUDA emitter needs `coord_int`, `const_int` and `int_op` to lower these**; until then it declines naming the leaf.

Proof: `tests/integer_index_expressions.rs` (type rules each with a control, gate, oracle at the extremes) and a real-compiler differential over 3 types x 6 edge constants x 11 forms (136 kernels) against the oracle's exact `i128` evaluation; a mutation of the `Sub` spelling turned it red.

## `0.25.1` — test fix: exact powers of two in the optimizer tests

Two optimizer tests looped `k` over `-1074..=1023` with `f64::powi(k)`, which is `1 / 2^-k` on some libms; `2^1074` overflows, so the subnormal powers of two came out `0` there and the tested population differed by platform. They now build `2^k` from its bit pattern (`pow2_f64`), and a test pins it at `k = 1023, -1022, -1023, -1074`. Test-only; no public API change. Found by the 0.25.0 libm sweep (a `powi(-130)` fixture that was `0.0` on Linux).

## `0.25.0` — BREAKING: numerics qualifiers, `ScalarExpr::Qualified` (I1 of #112)

**Breaking, pre-1.0 (second number).** New variants: `ir::ScalarExpr::Qualified`, `ir::DagNode::Qualified`, `backend::DeclinedOp::Qualified`, `pattern::PatternError::QualifiedUnsupported`; new `Lowering::qualified` seam (default: decline); every exhaustive `match` over `ScalarExpr`/`DagNode` gains an arm. New public items: `ir::{Numerics, QualOp}`, `Expr::{qualified, rounded, ftz, approx}`, `numerics::{eval_qualified_f32, approx_within, ftz_f32, APPROX_ULP, LOG2_ABS_FLOOR}`, `types::has_qualified`.

`Qualified(op, Numerics{round, ftz, approx})` evaluates one of `Add`/`Sub`/`Mul`/`Div`/`Fma`/`Sqrt`/`Rsqrt`/`Recip`/`Exp2`/`Log2` under an IEEE directed rounding mode, PTX-style flush-to-zero (operands and result, sign kept) and/or an approximate form with a stated bound (4 ulp of f32 on the finite-normal domain; `Log2` with a 2^-20 absolute floor). Spec: `docs/ir-numerics-qualifiers.md`. v1: f32/f32s, Elementwise, single output, not combined with `Cast`/`Bitcast`. The optimizer returns such a body unchanged; pattern derivation and recipes decline it. A backend that cannot honour a qualifier DECLINES (it never lowers the bare operation): cpu_c spells nearest-even, `ftz` and `approx` (as the exact function) and declines every directed mode naming it; Slang declines `Qualified`; the CUDA emitter declines until baracuda spells `__fadd_rd`-style forms.

Proof: the directed-rounding reference is checked against exact `i128` integer rounding (add/sub/mul/fma, bracket checks for div/sqrt, an IEEE 7.4/6.3 overflow and zero-sign table); `ftz` is compiled and run through a real C compiler lane by lane with a no-`ftz` control; `approx` forms are compiled and checked within the bound over 199 lanes per operation with a failing control; seven reference mutations and six cpu_c mutations each turned a test red.

## `0.24.1` — numerics qualifiers: the spec (docs only)

`docs/ir-numerics-qualifiers.md`: `ScalarExpr::Qualified(child, Numerics{round, ftz, approx})` over `Add`/`Sub`/`Mul`/`Div`/`Fma`/`Sqrt`/`Rsqrt`/`Recip`/`Exp2`/`Log2`; IEEE directed rounding including overflow and zero sign; PTX-style flush-to-zero (operands and result); `approx` as a stated 4-ulp bound on a stated domain, checkable against the oracle; backends that cannot honour a qualifier decline, never drop it. No code; the implementation is 0.25.0 (Part of #112).

## `0.24.0` — BREAKING: typed expressions, `Cast` and `Bitcast` (I1 of #112/#113)

**Breaking, pre-1.0 (second number).** New variants: `ir::ScalarExpr::Cast`, `ir::ScalarExpr::Bitcast`, `ir::DagNode::Cast`/`Bitcast`, `pattern::PatternError::TypedExprUnsupported`; every exhaustive `match` on `ScalarExpr`, `DagNode` or `PatternError` in an adopter must add an arm. Additive: `ir::{RoundMode, Saturation, CastSpec}` (`#[non_exhaustive]` enums), `Expr::cast`/`bitcast`, the `types` module (`infer`, `check_cast`, `check_bitcast`, `TypeError`), `backend::DeclinedOp::{Cast, Bitcast}`, and the lowering seams `Lowering::{cast, bitcast, compute, type_name}` with `LoweringBuilder::{cast, bitcast, compute_dtype, type_name}`. The seams default to a typed decline, so an existing emitter compiles unchanged and declines a typed op.

Every `ScalarExpr` node now has a type (`types::infer`; spec in `docs/ir-typed-expressions.md`). An expression with no `Cast`/`Bitcast` types to the compute dtype at every node, so every existing op is unchanged. `Cast{to, round, sat}` converts a value (float/int in the v1 table: `f32 f32s f64 i8 i16 i32 i64 u8 u16 u32`) with explicit rounding and saturation: float to int is saturating with NaN to 0, so no emitter reaches C undefined behaviour; float narrowing overflows to infinity or, with `ClampFinite`, to the largest finite value. `Bitcast` reinterprets bits between equal widths. Plan gate (`check_typed_admissibility`): Elementwise, single-output, `f32/f32s/f64` compute, store type equal to the body type, and a non-compute-typed value only at the root or directly under another `Cast`/`Bitcast`. The optimizer returns a typed expression unchanged. `unpopped-cpu-c` spells both (rounding other than nearest-even declines for inexact int/float narrowing; float to int supports all four modes); `unpopped-slang` declines. KISS pattern and recipe derivation decline.

Proof: `cpu_end_to_end` compiles and runs 53 kernels (float to 7 integer types in all four rounding modes, int to int wrap/clamp, int to float, f64 to f32 with and without `ClampFinite`, bitcast) over 50 edge values each against the oracle, whose conversion (`cast_ref`) is computed from exact integers and IEEE bits. Five seeded bugs (trunc to floor, no clamp, bitcast as value cast, oracle without `ClampFinite`, oracle NaN to 1) each turn it red. Not in this release: half/fp8/sub-byte endpoints, packed values, integer `Coord`/`ConstInt`/integer arithmetic, rounding/ftz/approx qualifiers, and anything touching fp4/fp6 (D5, CireSnave). Numbered 0.24.0; the PM allocates the final number.


## `0.23.0` — BREAKING: `ScalarExpr::Fma`, the first node of the union IR (I1 of #112/#113)

**Breaking, pre-1.0 (second number).** New variants: `ir::ScalarExpr::Fma`, `ir::DagNode::Fma`, `pattern::PatternError::FmaUnsupported`; every exhaustive `match` on `ScalarExpr` or `PatternError` in an adopter must add an arm. Additive and non-breaking: `backend::DeclinedOp::Fma` (the enum is `#[non_exhaustive]`), `Lowering::fma` and `LoweringBuilder::fma` (the builder defaults to a typed decline, so an existing emitter compiles unchanged and declines `Fma` until it supplies a spelling), and `Expr::fma`.

`Fma(a, b, c)` is `a * b + c` with ONE rounding. It is not `Add(Mul(a, b), c)` and the optimizer never rewrites either into the other (zero rules, pinned by `fma_is_never_folded_or_rewritten`). Admissible at `f32`/`f32s`/`f64`; `f16`/`bf16` miss honestly (a half FMA can only be spelled by widening, fusing and rounding twice), as do integer, complex and bool dtypes (`check_fma_admissibility`, `fma_admissibility.rs`). `unpopped-cpu-c` spells `fmaf`/`fma`; `unpopped-slang` declines (no single-rounding guarantee on every Vulkan device); KISS pattern and recipe derivation decline (no FKC §4.1 op, no confirmed KISS-Ops token). Proof it is fused: `cpu_end_to_end::fma_is_fused_through_a_real_compiler` runs an optimized kernel on an input where one rounding gives 2^-24 and two give 0; replacing `fmaf` with `a*b+c` turns it red. Not in this release: `Cast`/`Bitcast`, rounding and ftz qualifiers, packed values, integer `Coord`/`Select`, the kernel layer (the rest of I1 and I2), and anything touching fp4/fp6 (D5, CireSnave). Numbered 0.23.0; the PM allocates the final number.

## `0.22.19` — I0 of the IR-as-full-union programme: the cooperation ruling replaces idiom-lifting 1a (docs and comments only)

Replaces `idiom-lifting-design.md` 1a and the groupshared "answered" banners with the 2026-10-24 ruling (the IR models cooperation); records design-doc rulings D1-D8 (D5 pending CireSnave, board 290); adds `docs/ir-union-capability-grid.md` (Unpopped's IR beside the seven languages); ROADMAP 6.7 rows U0-U12 link tracking issues #111-#123; fixes the stale `Contraction`/`AxisRole::Batch` doc comments in `ir.rs`, the TF32 comment in `element.rs`, and three stale statements in `joint-gpu-milestone-plan.md`, `normative-seams.md`, `slang-second-blocker-sizing.md`. No behaviour change. The Slang `lower()` i16/u16 message is deliberately untouched (adopters match error text). Numbered 0.22.19; the PM allocates the final number.

## `0.22.18` — the IR as the full union of kernel languages: gap matrix and design (docs only)

`docs/ir-union-design.md` plus seven raw surveys in `docs/ir-union-surveys/` (CUDA, CUTLASS, Slang, CuTe, Mojo, Triton, ThunderKittens). CireSnave's directive (2026-10-24, quoted verbatim in the doc) and the PM rulings: the IR models cooperation; the IR is built from the full union of the seven languages while parser/emitter modules exist only for CUDA, CUTLASS and Slang. The doc consolidates 22 gap rows, proposes an intent layer plus a kernel layer, orders 12 breaking increments (none started), states the module interface for the CUDA, CUTLASS and Slang owners, and lists decisions D1-D8 for the PM. No code.

## `0.22.17` — integer `AtomicAdd` is deterministic for every integer dtype, not four

`ir::is_integer_kind` is documented as a local mirror of `plan::is_int_dtype` but listed 4 of its 11 dtypes (i32, i64, u8, i8). `WriteCombine::is_fp_atomic_add` therefore classed an integer `AtomicAdd` into u32, u64, i16, u16, i4, u4 or b1 as a floating-point atomic add, and the CUDA emitter routed that scatter to the non-deterministic variant. The safe direction, but a wrong claim about an order-independent integer sum. The mirror now lists all 11. Found by the CUDA coverage survey (2026-10-25) and confirmed in source. Test: `atomic_add_integer_classification_matches_plan_for_every_dtype` compares the two over `ElementKind::ALL` (red before the fix on `I16`; green after) and checks that only `AtomicAdd` is ever order-dependent.

## `0.22.16` — ROADMAP records the Slang corpus baseline (E12)

Docs only. `slang_corpus_report` over Fuel's 147 `.slang` kernels (`fuel` `ac29b53c9`, this crate at 0.22.14): 0 accepted, 94 unrecognized, 53 inexpressible (36 `groupshared`, 16 `InterlockedCompareExchange`, 1 `RWByteAddressBuffer`). The first measurement since #22/#25; Fuel uses its own binding layout, so it is the baseline for E13 and not a defect.

## `0.22.15` — ROADMAP records the C14 Vulkan NaN measurement

Docs only. C14 is done (`unpopped-harness` 0.0.16, #16): AMD Radeon 610M and RTX 4070 Laptop both show 0 mismatches over all 2^32 f32 inputs on the NaN, infinity and sign-of-zero probes, so no device is measured not preserving NaN and C2 (N2) stays parked. E2 notes that the Slang non-finite constant spelling is fixed (#106) and the declined ops are a different gap.

## `0.22.14` — `unpopped-slang` spells NaN and infinity constants as Slang, not C macros

A body carrying a NaN or infinity constant used to emit `INFINITY` / `NAN`, which Slang does not define: `slangc` failed with `undefined identifier 'INFINITY'`. Found 2026-10-24 by the harness's C14 probe (`|x| == +inf` on a Vulkan device); the crate docs had recorded it as a known seam follow-up, which is no reason to leave it. The `constant` seam now uses `slang_const`: NaN, +inf and -inf become `asfloat(0x7fc00000u)`, `asfloat(0x7f800000u)` and `asfloat(0xff800000u)`; finite constants still go through `const_lit` unchanged, so no existing golden moves. Tests: `non_finite_constants_are_not_spelled_as_c_macros` (fails with the old spelling; verified by restoring it), and the `slangc`-gated `emitted_slang_compiles_with_slangc` gains a NaN/+inf/-inf kernel (run locally with `slangc` 1.4.350.0: ok).

## `0.22.13` — ROADMAP section 6 records the PM's rulings on the audit calls

Docs only. Calls 1, 2, 4, 6, 8, 9, 11 are parked and keep their rows; call 5 (N2) is parked on KISS's recommendation with a new unblocker row (C14, measure a Vulkan device's float-controls); call 3 (`vec_width`) waits on baracuda #252, then this lane writes the plan; call 10 is deferred; call 7 is still to be asked. E2 gains `copysign`/`nextafter` (harness 0.0.15), P6 the binary and ternary corpora, and C15 records #104.

## `0.22.12` — `f8e6m2` is reserved, so there are 21 usable dtypes, not 22

Tests and comments only; no library behaviour change. KISS reported (2026-10-24, KISS-CLASSIFY-6.1-0001): 24 recognized, **21 usable, 3 reserved** (`f8e4m3fnuz`, `f8e5m2fnuz`, `f8e6m2`). `ElementKind::is_reserved` already said so, but `dtype_lowering_coverage.rs` counted two reserved and 22 usable, and passed only because its table carried a row for the reserved `f8e6m2` (as `ByDesign(MX_SCALE)`). That row moves to `the_reserved_dtypes_never_lower`; the table is 21 rows and `usable = 21`. The same "22 usable" in `kiss_byte_match.rs`, `kiss_dtype_manifest.rs`, `docs/normative-seams.md` and `docs/deferred.md` is corrected (and "the other 17" becomes 16). `f8e8m0` stays the one scale-only usable dtype (6.1-0013). The move/compute boundary is still our reading, not a KISS clause (KISS states none; `docs/normative-seams.md` says so).

## `0.22.11` — spec-to-roadmap audit; the docs stop saying `ulp_bound` takes no target

Docs and one comment only; no behaviour change. `ROADMAP.md` gains section 6: 60 rows for every undone item found in 23 spec/design docs and not already a row (open / held / done with the PR; 11 need a CireSnave or PM call). `ulp_bound` has taken an `AccuracyKey { target }` since 95d2ed6, but `docs/conformance.md`, `docs/normative-seams.md`, `docs/deferred.md` and the KNOWN LIMIT comment on `ulp_bound` still said the band is CUDA's for every backend; the claims are replaced, not caveated, and the open dtype and precision-mode axes are stated. The README stops quoting versions (it said `unpopped` 0.2.0, `unpopped-vocab` 0.3.0, emitters 0.14.4 against a 0.22.x workspace) and counts nine migration rows, not eight. `ROADMAP.md` section 4 now says moving `ArchSku` waits on baracuda's registry repoint first.

## `0.22.10` — the two published library crates are declared public API to CodeRipper (`public_api_crates`)

`reachability` counts callers inside one crate, so it listed 179 `pub` items of `unpopped` (98) and `unpopped-vocab` (81) that adopters and this workspace's test crates call. A new workspace-root `.coderipper.toml` declares both packages under `[reachability] public_api_crates`, one entry each with its reason, instead of 179 per-item waivers. All 179 are now printed `[declared-public]` with that reason and none fails the run (measured at coderipper `d14a22b6`, `check reachability --workspace`). This declares, it does not prove: a `pub` item with no caller anywhere is declared public too. No code change, no API change.

---

## `0.22.9` — the lifter states that its two bracket-eats cannot fail (CodeRipper unused-return-values)

`parse_input_ref` and `parse_call` discarded `eat`'s result; both are entered only after the caller peeked the token, so it always succeeds. The result is now bound and `debug_assert!`ed. No behaviour change in release, no API change.

## `0.22.8` — `tools/renamed_names_left_in_prose.py` declares its licence (CodeRipper SPDX, last file)

One comment line. No code, no API change.

## `0.22.7` — docs only: ROADMAP re-baselines CodeRipper on the repinned workflow

No code, no API change.

## `0.22.6` — every `.rs` file declares its licence (CodeRipper licence/SPDX)

One `// SPDX-License-Identifier: MIT OR Apache-2.0` line at the top of each of the 104 `.rs` files, matching `license` in Cargo.toml. No code, no API change.

## `0.22.5` — every `#[allow]` carries a `reason` (CodeRipper lint-suppression)

37 attributes gained a `reason = "..."`; none was removed or widened. No code, no API change.

## `0.22.4` — test only: the `docs/` half of a seam guard moves to `unpopped-conformance` (CodeRipper SUP-004)

`the_declaration_form_is_a_seam` read `../../../docs/normative-seams.md`, outside its crate and so outside the published `.crate`. The doc check is now `a_doc_discussing_plan_body_names_both_access_shapes` in the never-published conformance crate, with a control that it engages. No API change.

## `0.22.3` — docs and CI only: CodeRipper baseline rows, repin portfolio.yml to 30e012b7

No code, no API change.

## `0.22.2` — CI only: advisory CodeRipper portfolio check

Adds a non-blocking `portfolio` job (reusable workflow `ciresnave/coderipper`, pinned by SHA). No code, no API change.

## `0.22.1` — docs only: ROADMAP drops the landed rows (#6, the `0.20.3` publish)

No code, no API change.

## `0.22.0` — BREAKING: `Backend::supports_dtype` returns `DtypeSupport` (closes #6)

`fn supports_dtype(&self, dtype, target) -> bool` becomes `-> DtypeSupport`: `Supported` / `Unsupported` (the device or this backend lacks it: change hardware) / `NotExpressible` (the target's vocabulary cannot say: wait for a spec). `bool` reported both refusals as `false`. The enum is `#[non_exhaustive]`; `DtypeSupport::is_supported()` is the old answer for a caller that only needs to know whether to emit. **Migration:** change the return type; `true` becomes `Supported`, `false` becomes `Unsupported` unless the refusal is "the vocabulary has no token to say" (then `NotExpressible`). A wrong guess between the two declines identically; only the remedy differs. Callers: `.is_supported()`.

In tree: `unpopped-cpu-c` has no vocabulary-gated refusals (the MX scales and `fnuz` pair are `Unsupported` by design). `unpopped-slang`: `i8`/`u8`/`bool` on a target without the `i8` arith token is `Unsupported`; `i16`/`u16` on a target without an `i16` token is `NotExpressible`, because the published `<arith>` vocabulary names no 16-bit arithmetic token (Vulkane's finding: *silence, not denial*). Out of tree: baracuda's CUDA emitter moves in the same release. Variant names are proposed here; Vulkane (who raised it) confirms before merge.

## `0.21.1` — CI only: the published-vs-workspace divergence gate (closes #4)

No code, no API change. `tools/published_divergence.sh` and a `divergence` job in `verify.yml`: for each publishable crate whose declared version is already on the registry, the registry's `src/` must equal the workspace's (`diff -rq --strip-trailing-cr`). A version not yet published passes as unreleased; a registry or network failure fails. The decision uses the sparse index, because `static.crates.io` answers 403 for a file that does not exist. The script runs its own control first (an injected line moves the differing count 0 to 1) and prints its population. Verified locally at `f0f25620`: 4 compared, 0 differ; a line appended to `unpopped-vocab` makes it exit 1; a bumped, unpublished `unpopped-slang` is reported unreleased.

## `0.21.0` — BREAKING: the per-device decline (`ExactnessRecord` names what it was baked against; `Admission`; the ledger file)

Step 2 of `docs/design-per-device-decline.md`, with the PM's rulings of 2026-10-22. Pre-1.0, so the second number is the major.

**Breaking, all in `unpopped::exactness` and its readers:**
- `DeviceStamp::new(hw, driver, compiler_version, compiler_args)` and a new `DeviceStamp::compiler_version`, part of the stamp's identity: a `slangc` upgrade now invalidates a record (it did not before; the arguments do not carry the version).
- `ExactnessRecord` is `#[non_exhaustive]` and gains `kernel_hash`, `emitter_version`, `coverage` (`Coverage { kind: Exhaustive | Sampled, tuples, seed }`) and `extra` (fields a newer writer added, kept on write). `ExactnessRecord::new` takes `kernel_hash`, `emitter_version`, `coverage`.
- A record applies only to a kernel with the same `kernel_hash`, and **only an `Exhaustive` `Exact` admits**: a sampled `Exact` is kept as evidence and admits nothing (it did before; the module docs always said it could not). A sampled `Flushes` still declines.
- `ExactnessLedger::lookup(stamp, kernel_hash, body, dtype)` takes the hash. `contract_with_exactness` and `required_fidelity_with_exactness` take `Option<Measured>` (`Measured::new(stamp, kernel_hash, ledger)`) instead of a `(stamp, ledger)` pair, and `MeasuredOn` gains `kernel_hash`.

**Added:** `ExactnessLedger::find` (`Finding::Exact | Withheld(Unmeasured | Flushes | StaleKernel | SampledOnly)`; `Unmeasured` stays distinct from `Flushes`); `ExactnessLedger::admit(stamp, kernel_hash, body, dtype, Requirement)` returning a typed `Admission { Admitted, ClaimWithheld(..), Declined(..), UseArgs(ArgsOffer) }` that Unpopped never turns into an error (the adopter chooses; Fuel owns the ledger location, measure-on-first-run and applying `UseArgs`); `AdmissionTally` (the reporter for the safe failure, per stamp and reason); and the ledger file, JSON Lines, schema `unpopped-exactness-ledger/1`, owned here (`ExactnessRecord::to_line` / `parse_line`, `ExactnessLedger::to_jsonl` / `parse_jsonl`). An unreadable line is a counted `LineError` with its line number, never a panic and never dropped silently; unknown fields round-trip. The wire form is the one `unpopped-harness` 0.0.9 already writes, pinned by a test with a literal line.

**Not here:** `emitter_version` is still a string the producer supplies (Unpopped cannot ask a `Backend` its version; a `Backend` method is a candidate for a later release), and `Backend::supports_dtype` (#6) is its own PR in this release line. `unpopped-harness` moves to this record in its own PR.

## `0.20.21` — docs only: `ROADMAP.md` schedules #6 and #4

No code, no API change. #6 (three-state `supports_dtype`) is scheduled into the 0.21 Breaking minor; #4 stays open until `verify.yml` runs the published-vs-workspace comparison, which is now a roadmap row.

## `0.20.20` — docs only: the per-device decline design (`docs/design-per-device-decline.md`)

A design note, no code, no key bytes, no API change. It separates four gaps (persistence of the ledger, plan-level decline, compiler-argument variants, staleness) and records the PM's rulings: one `ExactnessRecord` type gaining `kernel_hash`, `compiler_version`, `coverage` and `emitter_version` in a 0.21 Breaking minor; `Declined` a typed value the adopter chooses what to do with; Fuel owns the ledger location and applying a variant. `Unmeasured` stays distinct from `Flushes`. The stamp carries `slangc` arguments but not its version, so today a compiler upgrade does not invalidate an `Exact` record.

## `0.20.19` — additive: a device that flushes subnormals declines Exact (`MeasuredOn`, `*_measured`, `*_with_exactness`)

`contract::MeasuredOn { stamp, ledger, dtype }` with `ulp_bound_measured` and `precision_of_measured`; `contract_with_exactness` and `oracle::required_fidelity_with_exactness`. The existing functions are unchanged and delegate with `None` (`AccuracyKey` keeps its shape: a lifetime parameter on it would break every caller). A `Flushes` record for the body on exactly the stamp declines the bound on any namespace (a CUDA relu stops claiming `correctly_rounded`); an `Exact` record admits a non-CUDA body only if the table rates every operator 0 (agreeing with a reference that has an error budget is not correct rounding); no record changes nothing, so an unmeasured device cannot claim Exact. A stamp for another target than the cell's is ignored. Exhaustive 2^32 exists only for unary f32 bodies: a binary or f64 body is admitted only by a measured run, and nothing is extrapolated.

## `0.20.18` — additive: `unpopped::exactness`, the record of what a device measured (PM-allocated)

New `DeviceStamp` (an `HwStamp` plus driver version and compiler arguments), `ExactnessRecord`, `ExactnessVerdict { Exact, Flushes }`, `ExactnessLedger::lookup` and `body_key` (dtype + the IR's serde-JSON, pinned by a test). Nothing reads a ledger yet: the next change wires it into `AccuracyKey` so a `Flushes` record declines Exact. Records apply to exactly their stamp, body and dtype; a changed driver or argument set matches nothing; disagreeing records resolve to `Flushes`. Exhaustive 2^32 exists only for unary f32 bodies, so a binary or f64 body has a record only if a measured run exists, and is declined otherwise (no extrapolation).

## `0.20.17` — docs only: the harness family moved out of scratch

`ROADMAP.md` row 3 and a status line in `docs/harness-family.md`: the harness lives in `ThinkersJournal/unpopped-harness` (cpu-c, CUDA, Vulkan). No code, no key bytes, no API change.

## `0.20.16` — slang: name the subnormal mode (`required_slangc_args`, header line)

Exhaustive 2^32 f32 relu, cpu-c reference vs Slang through `slangc` -> SPIR-V -> Vulkan (`unpopped-harness` 0.0.2/0.0.3, AMD Radeon 610M and RTX 4070 Laptop): the default Slang kernel differs on exactly 8,388,607 inputs, the negative subnormals (the device flushes them in `x < 0.0`, then returns `x`; cpu-c and CUDA return 0). With `slangc -denorm-mode-fp32 preserve` (SPIR-V `DenormPreserve 32`): zero mismatches on both. The mode is a compile argument, not source text.

New `unpopped_slang::required_slangc_args(dtype)` (`-denorm-mode-fp32 preserve` for F32/F32Strict, `-denorm-mode-fp64 preserve` for F64, none for integers). Every float kernel's header gains one comment line, `// slangc: <args>` (comment only: the lifter ignores it; the kernel body, name and cell are unchanged). Module docs: a device must NOT be gated on `VkPhysicalDeviceFloatControlsProperties` alone (the RTX 4070 reports `shaderDenormPreserveFloat32 = false` yet preserves when asked and flushes by default); how a plan declines waits on a `denorm` field in the `vulkan:` token, which is vulkane's vocabulary, not KISS's (KISS lane ruling 2026-10-21). Until then an `Exact` float Slang kernel is only as exact as the flag the consumer passes. Not checked here: the `f64` flag on a device (no f64 run), and the `slangc` self-hosted test still `#[ignore]`.

## `0.20.15` — docs only: `max(x, 0)` does not scrub NaN

`ir.rs` (`UnaryOp::Relu`, `Expr::relu`) said `max(x, 0)` scrubs NaN. That is IEEE `fmax` (`BinaryOp::FmaxIeee`); `BinaryOp::Max` propagates NaN and keeps A on ties, so `Max(x, 0)` is bit-identical to `Relu` (exhaustive 2^32 f32 comparison through cpu-c, Exact tier; other emitters not run). Comments only, no code.

## `0.20.14` — docs only: roadmap row and design note for the harness family

`ROADMAP.md` row 3 and `docs/harness-family.md`: external test/profile harnesses that wrap Unpopped, kept in scratch until a second target runs (CireSnave's request, PM ruling 2026-10-19). No code, no key bytes, no API change.

## `0.20.13` — tests only: mutation audit of `oracle.rs`, cluster 9 (walkers' index arithmetic)

`cargo-mutants` 27.1.0 over the evaluators' index and layout arithmetic. `oracle::walk_pins` pins a top-k row sort with both outputs (each output is sized from its own
operand), a skipping gather (indices just outside the axis write nothing), a scatter whose index operand is longer than its value operand (the walk is over the value
operand), a middle-axis reduction placed by the output strides (collapsed and keepdim), and `required_fidelity` for a u8 mask beside an f32 value (tolerant, not bit-exact).
The three survivors left in these functions are equivalent and are listed with their reasons in the PR. No code, no key bytes, no API change.

## `0.20.12` — tests only: mutation audit of `oracle.rs`, cluster 8 (value plumbing)

`cargo-mutants` 27.1.0 over the plumbing around the evaluators. `oracle::plumbing_pins` pins `raw_sign_masks` (narrow floats only), `round_to_compute`
(through the compute dtype, f64 untouched), `unravel` (last axis fastest), `store_val` (a same-dtype raw value moves verbatim, a different one converts,
a real or integer value into a complex slot gets a zero imaginary part), `alloc_output` (each output of a multi-output op takes its own descriptor), and the
`Neg`, `Abs` and `Select` arms of `eval` on narrow floats (payload-preserving sign edits; the select condition is tested in the compute dtype). No code,
no key bytes, no API change.

## `0.20.11` — tests only: mutation audit of `oracle.rs`, cluster 7 (storage encoders and codecs)

`cargo-mutants` 27.1.0 over the storage encoders and bit codecs. `oracle::encode_pins` pins `encode_int` and `encode_float` for the four-bit and one-bit
dtypes, `encode_complex` (real half low, imaginary half high), `encode_int_as_complex`, the `Val::complex` projection for float, integer and complex
raw values, and `unit_roundoff`. The 28 survivors left in this group are each provably equivalent (a `|` over disjoint bit sets, or a comparison whose equal case returns
earlier) and are listed with their reasons in the PR. No code, no key bytes, no API change.

## `0.20.10` — tests only: mutation audit of `oracle.rs`, cluster 6 (`arith_steps`)

`cargo-mutants` 27.1.0 over `arith_steps`, the count that sizes the derived tolerance band. `oracle::arith_pins` builds expressions whose operands
cost 1, 2, 3 and 5 steps, so a combining `+` that became `-` or `*` cannot land on the right total: leaves cost nothing, infix and named binary nodes cost
one plus both operands, a unary costs one plus its operand, and a `Select` costs only its three subexpressions. No code, no key bytes, no API change.

## `0.20.9` — tests only: mutation audit of `oracle.rs`, cluster 5 (`eval_binary`)

`cargo-mutants` 27.1.0 over `eval_binary`. `oracle::binary_pins` reads back the storage bits of `Copysign`, `Max` and `Min` on `e5m2`, `e4m3fn`,
`f16` and `bf16`: the sign mask, either operand winning, a NaN operand returned with its payload in either position, and a tie between `+0` and `-0`
keeping the first operand's bits (the `f64` detour agrees on every ordinary value and loses exactly those). No code, no key bytes, no API change.

## `0.20.8` — tests only: mutation audit of `oracle.rs`, cluster 4 (`compare`)

`cargo-mutants` 27.1.0 over `compare`. `oracle::compare_pins` checks which rejection each branch of the comparator returns (not just that it
returns `Err`): the first differing element in bit-exact mode, the byte-length case, infinities and NaN in the complex and float paths, the
tolerance band `abs + rel * max(|e|, |a|)` per component for complex, exactly for integers and in `f64` for floats, and that a mixed
complex/real or int/float pair takes the real projection rather than the component-wise or exact-integer path. No code, no key bytes, no API change.

## `0.20.7` — tests only: mutation audit of `oracle.rs`, cluster 3 (`eval_im2col`)

`cargo-mutants` 27.1.0 over `eval_im2col`. `oracle::im2col_pins` checks the evaluator against a direct six-deep loop over
(batch, channel, tap, output position) that never divides or takes a remainder, across non-square, strided, padded and
dilated geometries, a non-zero buffer base offset, and size-1 batch and channel axes (admitted by the plan gate since 0.20.6).
No code, no key bytes, no API change.

## `0.20.6` — fix: the Im2Col, Window and row-sort gates no longer read a size-1 axis as a broadcast

`build_plan` panicked ("output must be forward-dense contiguous" / "input 0 must be a full dense tensor") for ordinary shapes:
an Im2Col with batch 1 beside two or more channels, a single-channel batch (N ≥ 2, C = 1), a Window pooling a row to one
element (a global pool), and a top-1 / arg-max row sort. The key's broadcast mask is over the iteration frame, and an own
axis of extent 1 beside a wider frame axis counts as stride 0 there, so the gates' `bcast.is_empty()` clauses rejected dense
tensors. The clauses are removed: `contig` already reads `Broadcast` for a genuine stride-0 axis of extent > 1, so the
dense-layout clause alone rejects every operand that is not dense. Found while pinning `eval_im2col` (mutation audit of
`oracle.rs`). Gates only loosen: every plan that built before still builds, and no key bytes change. The emitters have not been
exercised on the newly admitted shapes.

## `0.20.5` — tests only: mutation audit of `structure_key.rs` and `oracle.rs`, clusters 1 and 2

`cargo-mutants` 27.1.0. Two earlier test-only PRs merged to `main` **without a version bump**, so `main` stayed at `0.20.4`; their
tests ship in this release: #71 (`structure_key.rs` codec and helper pins, and a `synthesis_lowers_the_optimized_body`
integration test) and #72 (`oracle.rs` cluster 1: the fp8/f16 encoders and the scalar evaluators). This release adds cluster 2: the
fold, row-reduce, scan and window evaluators (`oracle::fold_pins`). The survivors that are provably equivalent are listed in the
PRs. No code, no key bytes, no API change.

## `0.20.4` — tests only: mutation audit of the contraction and plan checks

`cargo-mutants` 27.1.0 over `check_contraction`, `validate_contraction_roles`, `contraction_epilogue_admissible`,
`ContractionDecline`/`PlanError`, `ContractionRoles`, the contraction keyer and the legacy bridge. Survivors became pin tests
(`plan::contraction_check_pins`, `structure_key::contraction_mutation_pins`); the proven-equivalent ones are listed in the
PR. No code, no key bytes, no API change.

## `0.20.3` — tests and docs only: two `allow(dead_code)` and one ignored doctest removed

CireSnave 5.1g (`#[ignore]` and `#[allow(dead_code)]` are last resorts). `oracle.rs` `coverage`: spurious allow
removed. `tests/common`: `OtherStub` moved into `provenance.rs`, its only user, so the shared module no longer
needs a crate-wide `allow(dead_code)`. `sku.rs` `ozaki_slices`: the ```` ```ignore ```` example (it named a crate
we do not depend on) is a real doctest. No code, no key bytes, no API change. One `#[ignore]` remains
(`emitted_slang_compiles_with_slangc`), a documented last resort: see `ROADMAP.md` row 1.

## `0.20.2` — comments only: `rm` is a permission to round, never a requirement

Two comments (the `from_op` bridge in `structure_key.rs`, and `tests/ctr_roles.rs`) said an `f32`
contraction "rounds its inputs". KISS-OPS 6.17-0003 makes `rm` a permission (KISS#593); the comments now
say so. No code, no key bytes, no API change. `ROADMAP.md` refreshed to 0.20.1.

## `0.20.1` — a contraction cell the plan cannot serve is a typed refusal, not a panic

`try_build_plan` gains `check_contraction`: the `Access::Contraction` arm of the plan gate no longer
`assert!`s. A key derived from declared roles (`ContractionRoles`, 0.19.0) can name a cell no `OpDef` pairs
with, so the refusal is reachable by *requesting* a cell and owes a typed decline (KISS-EMIT §6.8-0004).

- **Added** `PlanError::UnservableContraction(ContractionDecline)` and `ContractionDecline`
  (`RolesInvalid`, `KeyHasNoContractionFacts`, `EpilogueInadmissible`, `BatchDisagrees`,
  `BatchNotOutermost`), both `#[non_exhaustive]`, so this is additive. The first three are the old
  asserts, now typed; the last two are new: the op and the key disagree on whether the cell is batched
  (e.g. a rank-3 key whose caller declared no batch axis), and a real batch axis stored inside the matrix
  axes (the v1 emitters bind batch as the outermost stride; a broadcast batch axis is exempt). Rank 4 and
  above is `RolesInvalid` (v1 lowers one batch coordinate).
- **Behaviour:** `try_build_plan` / `try_generate` return `Err` where `build_plan` panicked; `build_plan` is
  the same gate's panicking wrapper, with a similar message. `try_generate` maps it into
  `LowerError::InadmissiblePlan`, so backends inherit the decline without code.
- **Not covered, deliberately:** each backend's own limits (a tile schedule, an accumulator spelling, a
  dtype list). Those remain the backend's decline.

## `0.20.0` — `ContractionAxes` (unpopped-vocab) is now `ContractionRoles` (BREAKING)

`unpopped::ir::ContractionAxes` (the *op's* per-input axis roles) and `unpopped_vocab::ContractionAxes` (0.19.0's
declared M/N/K and batch axes of the *key*) were two public types with one name. The vocab type is renamed, with
what hangs off it, while the surface is new. No behaviour and no token byte changes.

| 0.19.x | 0.20.0 |
|---|---|
| `unpopped_vocab::ContractionAxes` | `unpopped_vocab::ContractionRoles` (same API: `new`, `gemm`, `with_batch_axes`, `batch_axes`) |
| `CellFacts::contraction_axes` | `CellFacts::contraction_roles` |
| `DeriveDecline::ContractionAxesNotDeclared` (wire `ContractionAxesNotDeclared`) | `DeriveDecline::ContractionRolesNotDeclared` (wire `ContractionRolesNotDeclared`) |

`unpopped::ir::ContractionAxes` is unchanged. The 0.19.0 entry below names the old spellings, as shipped.

## `0.19.1` — the derivation vectors are re-vendored from KISS `026feba` (63 vectors)

Tests and docs only; no behaviour, API or token byte changes. KISS#592 corrected its `<acc>` warning
(the undeclared default is operand 0's dtype, not the weight's): one vector is renamed
(`ctr_weight_first_undeclared_accumulator_defaults_to_operand0_dtype`) and one added
(`ctr_weight_second_undeclared_accumulator_defaults_to_operand0_dtype`). Measured: PASS 63 | FAIL 0 |
DECLINE-DIFFERS 0 (of 63).

## `0.19.0` — the ctr stage of KISS 0.4.3: declared M/N/K and batch roles (BREAKING)

KISS 0.4.3's 62 derivation vectors now byte-match (PASS 62, from 35 at 0.18.1; `KNOWN_GAP` is empty).
No `STRUCTURE_KEY_VERSION` change: the token grammar is `sk6`'s.

**One derivation, role-driven.** A contraction's M/N/K and batch axes are *declared*
(KISS-CLASSIFY-6.6-0016, -0023), never read from operand positions; the old positional rank-2/3
deriver, which also hardcoded `<acc>`/`<mp>` from the lhs dtype and differed from the spec on 26 of the
27 new vectors, is gone from the conformant path.

- **Added** `AxisRef`, `ContractionAxes` (+ `ContractionAxes::gemm(rank, batch)` for the plain
  `[B..,M,K].[B..,K,N]` declaration), `BatchRoleDecline`, `CellFacts::contraction_axes`.
- **Derivation:** M/N/K size classes from the declared axes; the batch class is that of the **product**
  of the declared batch extents (a declared extent-1 axis is batched; no batch axis, no `b<class>`, at any
  rank); **rank above 3 derives**; `<acc>` is the declared accumulator, else operand 0's dtype (§6.7-0006;
  the weight's only when the weight is operand 0, so a weight-first caller must declare it); `<mp>` is the
  declared `math_fidelity`.
- **BREAKING — a declared contraction with no `contraction_axes` now declines**
  (`DeriveDecline::ContractionAxesNotDeclared`, this crate's name, as `WeightRoleNotDeclared`). Callers of
  `derive_structure_key{,_token}` that set `contraction_declared` must set the axes
  (`ContractionAxes::gemm(rank, batch)` is the old positional meaning) and, **to keep their 0.18 bytes,
  declare `accumulator` (`f16`/`bf16` -> `f32`, `i8` -> `i32`) and `math_fidelity` (`rm` for `f32`)**:
  the spec default is operand 0's dtype and `st`, so an f16 cell undeclared now keys `.../f16/f16/f16/st`
  where 0.18 keyed `.../f16/f32/f16/st`. Consumers that pin ctr goldens must regenerate them if they move
  to the spec default.
- **The legacy `op` bridge (`structure_key`, `try_structure_key*`) is unchanged in bytes**: it now
  *declares* the positional GEMM roles, the `f32` accumulator and the `f32`->`rm` policy that it always
  implied, and derives through the same role path (pinned by `tests/ctr_roles.rs` and the existing suite).
- **New typed declines:** `ContractionAxesNotDeclared`, `ContractionAxisInvalid`, `WeightRoleOutOfRange`,
  `ContractionNoOutput`, `Batch(BatchRoleDecline)` (`BatchAxisOutOfRange`, `BatchAxisDuplicate`,
  `BatchAxisOverlapsContractionRole`, `BatchAxisNotInOutput`, `BatchAxisNonPositiveExtent`),
  `ReservedContractionDtype`. `ScaleRoleOutOfRange`'s wire name is now `RoleSlotOutOfRange` (KISS's
  spelling, shared with the weight slot); the variant is unchanged.
- **Not decided here:** the key says nothing about whether an emitter can bind these axes. The v1 CUDA
  emitter binds one outermost batch axis; an emitter must decline what it cannot bind (KISS keys any
  declared batch axes, e.g. `ctr_rank4_one_batch_axis_declared`). The old "batch must be storage-outermost"
  decline is no longer a key-time decline.

## `0.18.1` — KISS 0.4.3's 62 derivation vectors are vendored; 27 are pinned as the known gap

Tests and docs only; no behaviour, API or token byte changes. `structure_key_derivation_vectors.json`
is re-vendored from KISS `f88d389` (36 -> 62 vectors: 15 `ctr` token vectors, 12 role-hint declines;
`decline_contraction_out_of_scope` removed) and the harness reads each vector's `contraction`
object where `CellFacts` can express it. Measured: PASS 35 | FAIL 7 | DECLINE-DIFFERS 20. The 27 are
named in `KNOWN_GAP` and asserted exactly, so the follow-up that implements the ctr stage (0.19.0,
Breaking) has to empty the list.

## Unreleased — `0.18.0`: `sk6`, the scale dtype is keyed (BREAKING)

**`STRUCTURE_KEY_VERSION` 5 → 6** (KISS#588, conformance 0.4.0,
KISS-CLASSIFY-6.6-0022). A quantized `ctr` cell keys the dtype of its designated
scale operand as the optional last contraction coordinate `q<dtype>`
(`…/st/qf8e8m0`, `…/st/qf16`). At `sk5` an `f8e8m0` and an `f16` scale derived
one key although they decode the same stored values to different numbers.

- **Every token changes its prefix** (`sk5|` → `sk6|`); an `sk5` token now parses
  to `UnsupportedSchemaVersion { version: 5 }`. Re-derive, do not rewrite: stored
  or announced tokens must be regenerated.
- **`ContractionKey::scale: Option<ElementKind>`** (+ `with_scale`) and
  **`CellFacts::scale_operand: Option<usize>`** (canonical slot, a declared role,
  never inferred). `None` omits the coordinate; every existing caller derives the
  same body as at `sk5`.
- **New declines** in `derive_structure_key`: `ScaleRoleOutOfRange` (a hint naming
  no operand is not silently dropped) and `ReservedScaleDtype`. Both names are
  this crate's. The reader declines a `q` coordinate with an unknown dtype
  (`UnknownDtype`), a reserved one (`ReservedDtype`), a second one, or one outside
  a contraction field.
- **Re-vendored from KISS `86196f7`** (conformance 0.4.0): vectors file 28
  positive / 25 decline, derivation vectors unchanged in outcome (35 PASS, 1 SKIP).

Not covered by KISS's derivation vectors (they exclude `ctr`, KISS#543): the
derivation of `q<dtype>` is pinned only by `tests/sk6_scale_dtype.rs`.

---

## `0.17.3`: the weight role is declared (additive)

**Fixed: `<wdt>` read operand 1 positionally**, which KISS-CLASSIFY-6.6-0019
forbids. Measured at `0.17.0`: `Gemm [W:f8e4m3fn, x:f16, y:f16]` derived
`<wdt>` = `f16` (the activation) while `[x:f16, W:f8e4m3fn, y:f16]` derived
`f8e4m3fn`, so a weight-on-the-left cell got a well-formed, byte-stable token
naming the wrong dtype.

- **`CellFacts::weight_operand: Option<usize>`** names the weight operand (canonical
  slot). `<wdt>` is that operand's dtype.
- **`derive_structure_key` declines a declared contraction without it**:
  `DeriveDecline::WeightRoleNotDeclared`. This name is this crate's, **not in
  KISS's decline vocabulary**. `0.17.0`'s `derive_structure_key` silently read
  operand 1 there; that API is one release old.
- **The `op = Gemm` bridge declares operand 1 and says so**, so no token from
  `structure_key` / `try_structure_key` changes. A caller whose weight is not the
  rhs (`W @ x`, a sibling-scale operand order) moves to `derive_structure_key`.
- **Out-of-range weight slot** is `ContractionOutOfScope`, not an index panic.

Still positional: the M/N/K axis roles (§6.6-0016), and the batch role is
undefined upstream (KISS#543).

---

## Unreleased — `0.17.2`

CI and docs only: `.github/CODEOWNERS` (`* @ciresnave`), a top-level
`permissions: contents: read` in `verify.yml`, and `docs/glossary.md`. No code,
token or API change.

---

## Unreleased — `0.17.1`

Docs only. Adds `ROADMAP.md` at the repository root: the queue, what each item
waits on, and the KISS items this lane consumes with their blobs. No code,
token or API change.

---

## `0.17.0` (breaking; shipped in `0.17.1`, never published on its own)

Moves `STRUCTURE_KEY_VERSION` 4 → 5 and aligns the keyer with KISS-CLASSIFY at
`KISS@ff12265` (KISS#535, KISS#536). **Every non-trivial token changes.** If you
cache on a token, re-derive; an `sk4` token is `UnsupportedSchemaVersion`.

### What changes in the token

| Field | `sk4` | `sk5` | KISS-CLASSIFY |
|---|---|---|---|
| Field 1 | 24 op-family codes (`bin`, `une`, `red`, `gem`, …) | the **derived** cell class: `ctr` `ndx` `rdc` `ew1` `ew2` `ew3` `oth` | §6.5-0006, §6.5-0015 |
| Operand sub-key | five parts | five parts, plus `/dl` or `/dc` iff the operand carries a `quant` record | §6.3-0009a, §6.7-0004 |
| Own extent-1 axis, non-zero stride, frame wider | mask bit clear | bit **set** (an own extent-1 axis counts as stride 0) | §6.6-0008, KISS#520 |
| Innermost axis of a reduction | `v4` / `v8` when aligned | `v1` | §6.5-0009(b) |

### ⚠️ BREAKING (API)

- **`StructureKey::op: OpCategory` is replaced by `StructureKey::cell_class:
  CellClass`.** The token no longer names an op family, so none can be recovered
  from it. `seed_winner` reads `CellClass::Ctr`.
- **`OperandKey` gains `dequant: Option<DequantForm>`; `OperandDesc` gains
  `role: OperandRole` and `dequant_form: Option<DequantForm>`.** All three types
  are `#[non_exhaustive]`; construct through `new` / `with_role` /
  `with_dequant_form`. `OperandRole::Unspecified` keeps the pre-sk5 convention
  (inputs, then the output), so existing callers key as before.
- **`TokenDecline::UnknownOpFamily` is `UnknownCellClass`** (and its wire name),
  and every retired sk4 op code lands there. New variants: `BadOperandSubKey`,
  `BadDequantForm`, `ReduceNotGatedToRdc`, `RdcWithoutReduce`,
  `ContractionPresenceMismatch` (§6.6-0017, §6.6-0010, §6.7-0004).
- **`DeriveDecline` gains** `ReservedDtypeAtOperand0`, `CollapsedReduction`,
  `ReducedAxisOutOfRange` and `ContractionOutOfScope`, with a `wire_name()` in
  the KISS vocabulary.

### New

- **`derive_structure_key` / `derive_structure_key_token`** take `CellFacts`
  (declared contraction roles, index role, reduced axes, accumulator,
  math-fidelity) beside the operands, which is KISS's derivation input
  (§6.6-0012), and put operands with roles in canonical order (§6.6-0014).
  `structure_key` and `try_structure_key` keep their `op` argument and build
  the facts from it through a bridge: `Gemm` declares contraction roles,
  `Indexing`/`Embedding` an index role, a keepdim `Reduction` the axes it folded.
  **That bridge is not an op-to-class table**; the class is still derived from
  the operands.
- **`CellClass`, `DequantForm`, `OperandRole`, `CellFacts`.**

### Fixed: four keyer defects against KISS (measured by baracuda at `0.16.0`)

Each is pinned through both the new and the legacy entry point in
`tests/sk5_keyer_defects.rs`, and by KISS's vectors.

| | Clause | `sk4` | `sk5` |
|---|---|---|---|
| (a) | §6.5-0009(b) | a reduced innermost axis derived `v4`/`v8` | `v1` |
| (b) | §6.6-0009 | a collapsed reduction output was keyed as a non-reduction | typed decline `CollapsedReduction` |
| (c) | §6.6-0009 | a reduced axis at or past the rank was keyed | typed decline `ReducedAxisOutOfRange` |
| (d) | §6.1-0001 | a reserved dtype (`f8e4m3fnuz`, `f8e5m2fnuz`) at operand 0 was keyed | typed decline `ReservedDtypeAtOperand0` |

**Not a defect, and left alone:** a reserved dtype at a *non-primary* operand
still derives. Only operand 0 is a key position (§6.6-0015); KISS ruled this on
2026-10-14 and the sk4 behaviour was right.

### Conformance

Against `kiss/structure_key_derivation_vectors.json` (36 vectors, blob
`697aa543`): **35 PASS, 0 FAIL, 1 skipped**, run as
`cargo test -p unpopped-vocab --test kiss_derivation_vectors -- --nocapture`.
Only vectors with `spec_golden` are spec-anchored; the rest are
reference-deriver output, so agreement there is two derivers agreeing. The one
skip is `decline_contraction_out_of_scope`, which waits on a batch role that is
undefined (KISS#543). The codec leg (`kiss_byte_match.rs`) passes 26 positive
and 21 decline vectors. `[8,0]` derives `da` / `v1`, as KISS reads §6.3-0011.

**Not covered:** a declared contraction. `ctr` cells are keyed as before from
plain `[lhs, rhs, out]` and bias operand lists; the weight role (§6.6-0019) is
still not an input (#31), and a declared contraction whose field cannot be
derived is now a `ContractionOutOfScope` decline instead of a `gem` token with
no contraction field.

### Other

- **Vendored from KISS `ff12265`:** `structure_key_vectors.json`,
  `dtype_manifest.json` and the new `structure_key_derivation_vectors.json`. The
  CRLF/LF re-vendor hold in `kiss/README.md` is lifted (KISS measured 0 CR bytes
  from both generators).
- **`unpopped-vocab` gains a dev-dependency, `serde_json`**, for the derivation
  harness. Its normal dependencies are unchanged.
- **Docs corrected:** the `[8,0]` "deliberately not changed" note below and the
  `docs/deferred.md` "ask KISS for a target-match vector" remedy were both stale
  (the vector has been vendored and asserted since the `bc16715` copy).

---

## `0.16.0` (breaking; published)

### f16/bf16 take FP8's shape: a portable carrier and an emitted codec

The last vendor spelling in the neutral `cfamily` module is gone. baracuda#154's local
shadow keeps every CUDA byte `baracuda-cuda-emit` emits unchanged.

- **`cfamily::scalar_ctype(F16 | Bf16)` returns `"unsigned short"`**, the
  storage carrier, instead of `"__half"` / `"__nv_bfloat16"`.
- **New `cfamily::half_helpers(ElementKind)`**: portable-C99 f16/bf16 codecs
  (`unpopped_f16_load`/`_store`, `unpopped_bf16_load`/`_store`), emitted with
  the kernel. Round to nearest, ties to even; NaN encodes as the canonical
  quiet NaN. Checked bit for bit through a real C compiler against the oracle's
  codec on all 65536 patterns and on 524288 products that need rounding; the
  oracle's codec is itself checked against the `half` crate.
- **`narrow_load_fn` / `narrow_store_fn` name the codec for the halves**, so
  `promote_load_f32`, `demote_store_f32` and `cast_scalar` emit
  `unpopped_f16_load(x)` where they emitted `__half2float(x)`.
- **Removed: `cfamily::half_load_intrinsic` and `half_store_intrinsic`.**
- **`store_expr_of` encodes a uniform half store once, at the store**, as for
  FP8: a half body root is now a plain f32 expression.
- **`unpopped-cpu-c` computes f16 and bf16.**
- **Fix: a narrow-float predicate stored a wrong mask.** The U8 store of a
  `Cmp*` body over FP8 decoded the f32 `0.0`/`1.0` root as if it were a storage
  pattern, so every true lane stored `0`. Live in `unpopped-cpu-c` since FP8
  support; `store_expr_of` now converts from what the root is.
- **kiss-ref `0.3.4` -> `0.4.0`** (`kiss-ref-core`, `kiss-ops-vocab`,
  `kiss-classify-vocab`; KISS contract v3). Its enums are now
  `#[non_exhaustive]`; the in-tree `kiss_ref_diff` test reference panics on a
  `Monoid` it does not know rather than folding it as another. No published API
  of ours changes. `unpopped-cpu-c` takes `kiss-ref-core 0.4.0` as well.

---

## 2026-10-11 — everything at `0.15.0`

### ⚠️ BREAKING — `__expf` / `__logf` no longer lift (they are `Unrecognized`)

`unpopped::lift` and `unpopped::convert` (CUDA frontend) used to map CUDA's
fast-math `__expf` and `__logf` to the exact `UnaryOp::Exp` / `Log`, which the
shared lowering spells `expf` / `logf`. A lift-and-re-emit silently turned an
approximation into the accurate function: a numerics change with no record
(idiom-lifting rule 2, `docs/idiom-lifting-design.md` §2).

- **What used to lift and now is `LiftError::Unrecognized`:** any kernel body
  calling `__expf(x)` or `__logf(x)`. `expf` / `logf` are unchanged.
- **No signature changed:** `unary_fn` is `pub(crate)`. Callers that matched
  `Ok` from `lift_elementwise` on such source now get `Err`.
- **Adopter action:** tests that pin the old mapping must expect
  `Unrecognized` (baracuda's `baracuda-cuda-parse/tests/convert.rs:84` does).
  Spell `expf` / `logf` if the exact function is what the kernel means.
- **Interim shape:** an `Approx` qualifier (§4.3) is still the full fix and is
  not in this release.

## 2026-10-11 — everything at `0.14.6` (CI only)

ci: `cargo deny` job and `deny.toml`; every workflow `uses:` pinned by full
commit SHA. No code, emitted byte or golden moves; the number changes because a
pushed change changes the version.

## 2026-10-11 — everything at `0.14.5` (docs only)

docs: 0.14.4 published, stale claims corrected. No code, emitted byte or golden
moves; the number changes because a pushed change changes the version.

- `README.md` no longer says the emitters are unpublished (false since 2026-08-19).
- `crates/unpopped/README.md` says the emitters are at 0.14.4, not 0.3.0.

## 2026-10-08 — everything at `0.14.4`

### `HalfArith`: the first plan decision that differs by sm (additive)

U1 of the joint P40 (sm_61) / RTX 4070 (sm_89) plan
(`docs/joint-gpu-milestone-plan.md`). No emitted byte or golden moves.

- **`TargetCapabilities` gains `fp16_results_per_clk_per_sm` and
  `fp32_results_per_clk_per_sm`** (`Option<u32>`), transcribed from NVIDIA's
  12.9.1 "Throughput of Native Arithmetic Instructions" table (page sha256
  `06499a0c…`). sm_61 does 2 fp16 results per clock per SM against 128 for fp32;
  sm_89 does 128 for both. The table has no column for 8.7, 10.3, 11.0 or 12.1,
  so those are `None`. `from_queried` leaves both `None`, and
  `with_arith_throughput` attaches sourced figures.
- **`KernelPlan::half_arith()` and `half_arith_for(TargetId)` return `HalfArith`
  (`Native` / `ViaF32`).** `ViaF32` is returned only when both rates are sourced
  and fp16 < fp32, which today means sm_61 only. Everything else is `Native`,
  the behaviour every emitter already has.
- It is a method, not a `KernelPlan` field, because `KernelPlan` is not
  `#[non_exhaustive]` and emitters build it by struct literal.
- Consumer: baracuda-cuda-emit gates its packed native `__h*2` path on
  `Native`. That path is bit-identical to the scalar float path by design, so on
  the P40 this changes speed, not numerics.

### Also in this change set (no API or emitted-byte change)

These merged after `0.14.3` with no version change of their own, and ship under
`0.14.4` (the PM's ruling, 2026-10-07: an unpublished number covers everything
merged under it). The change set is #37–#51, except #49.

- **Docs in the crate:** `telemetry::cuda_target_of` documents that its
  `sm{major}{minor}` format is not injective in principle (#37).
- **Tests:** `unpopped-vocab` pins what the frame broadcast mask says about
  collapsed outputs (`a_collapsed_output_is_absent_not_broadcast.rs`, #38).
  `unpopped` checks that every Qwen3 dense op plans and generates for
  `cuda:sm61` (`qwen3_dense_ops_plan_for_sm61.rs`, U2, #42).
  `unpopped-conformance`'s `golden_regen_checkpoint.rs` follows the
  `docs/deferred.md` split and the 2026-11-05 checkpoint (#43, #45).
- **Repository docs (not in any crate):** the joint P40 (sm_61) / RTX 4070
  (sm_89) milestone plan, `docs/joint-gpu-milestone-plan.md` (#39, kept current
  in #41, #46, #47, #48, #50, #51); `docs/deferred.md` Section B split, with the
  f16/bf16 shadow surface counted as eleven functions (#43, #45); and the
  qwen3_5 linear-attention assessment (#44).

## Released

⚠️ **What reached crates.io, read from its API on 2026-10-03:** `0.11.0`
(2026-09-11), `0.12.0` (2026-10-01), `0.13.0` (2026-10-01), `0.14.0` (2026-10-02), `0.14.1`
(2026-10-03), for all four published crates. `0.14.3` (2026-10-03) was confirmed
from the same API on 2026-10-07; `0.14.2` was never published. `0.14.4` was
published 2026-10-08 (`unpopped-vocab` 21:32:11Z, `unpopped` 21:33:14Z,
`unpopped-cpu-c` 21:33:43Z, `unpopped-slang` 21:33:59Z, all `yanked=false`,
read back from the crates.io API); `unpopped-conformance` is `publish = false`
and stays absent. **`0.11.1`–`0.11.6` existed only in the workspace and were
never published.** A consumer went straight from `0.11.0` to `0.12.0`, so
everything those intermediate numbers carried is listed under `0.12.0`.


## 2026-10-03 — everything at `0.14.3`

### Fixed: telemetry from any compute capability now merges, and every drop is counted

**Behaviour change: H100 telemetry now counts.** `telemetry::merge_reports` used to
stamp a record through `arch_sku_of`, which has answers only for 8.x and 9.x, and
dropped every other capability before the merge saw it (sm_61, sm_70, sm_75,
Blackwell). Its 9.x answer was `cuda:sm90a`, but Fuel keys a Hopper record
`cuda:sm90`, so `merge`'s arch gate (stamp target == key target) rejected every
H100 report as well. Both drops were silent.

- **New `cuda_target_of(major, minor)`:** `cuda:sm{major}{minor}`, digits
  concatenated, no `a` suffix. This is baracuda's `cuda:` rule (it owns the
  namespace, §6.8-0004), and it is the same rule Fuel keys records with.
  `merge_reports` stamps with it.
- **Changed rows, on purpose:** 9.0 now stamps `cuda:sm90` (was `cuda:sm90a`), so
  H100 reports keyed `cuda:sm90` now merge into the `cuda:sm90` cells, where
  before they changed nothing. 8.6/8.7 now stamp `cuda:sm86`/`cuda:sm87` (was
  `cuda:sm80`); Fuel builds no key for those today, so no live row changes.
  6.x, 7.x, 10.x and 12.x now merge into their own targets. 8.0 and 8.9 are
  byte-identical.
- **New `merge_reports_counted`** returns a `MergeReport`: merged records per
  full target token, plus counts for no stamp, no capability, an unformable
  target, an unknown backend, a declined entry, and an arch-gate reject.
  `merge_reports` is the same fold with the report discarded.
- `arch_sku_of` is unchanged. It still answers "which cutlass dispatch cell".

## `0.14.2` (docs only, never published; its content shipped in `0.14.3`)

### Docs only: `TargetId` is the architecture identity; `ArchSku` is a cutlass dispatch SKU

No code, signature or behaviour change. This release records board #106 option C
(2026-10-03), which every affected lane approved with CireSnave's prior authorisation:

- **`TargetId` names every target, CUDA architectures included.** Any
  architecture is spelled as a token (`cuda:sm61`, `cuda:sm100`, …) whether or not
  `ArchSku` has a variant for it. The `unpopped-vocab::target` module docs and
  `TargetId`'s doc now say so.
- **`ArchSku` is re-scoped to a baracuda-cutlass dispatch SKU.** It stays closed
  and exhaustive, with no new variants and no freeze. A missing variant is not a
  missing architecture. Whether to add one (Blackwell `Sm100a`, say) is baracuda's
  call, made for its cutlass dispatch. Its doc also lost a pointer to a ROADMAP that
  this repo doesn't have.
- **`docs/idiom-lifting-design.md` §6:** the planned `ArchSku::Sm61` breaking
  release is marked superseded. No variant is added. The capability row shipped in `0.14.1`.

The `telemetry::merge_reports` drop this entry listed as a follow-up is fixed in `0.14.3`.

## 2026-10-03 — everything at `0.14.1`

### `cuda_capabilities` / `capabilities_for` cover sm_61 through sm_121 (additive)

These rows were missing, so they returned `None`. Each is now transcribed from
NVIDIA's own tables, parsed from the HTML rather than copied by hand. The
sources and the verification state are recorded on `cuda_capabilities`:

| New rows | Source | Toolkit accepts the arch | Verified on hardware |
|---|---|---|---|
| 6.1 | CUDA C++ Programming Guide **12.9.1** | CUDA 12.9 (NVRTC); not 13.x | no (awaits a P40) |
| 10.0, 10.3, 11.0, 12.0, 12.1 | CUDA Programming Guide (13.x edition), Sep 10, 2026 | CUDA 13.3 | no, compile-level only |

**Every existing row was re-checked against the same sources and is
unchanged.**

**Deliberately absent:**
- **10.7**: NVIDIA tabulates it, but CUDA 13.3 doesn't accept it.
- **8.8**: CUDA 13.3 accepts it, but neither table has it.
- Anything below 6.1.

**The two NVIDIA editions disagree on 12.x resident blocks per SM.** The CUDA
12.9.1 edition says 32 and the current edition says 24. The row uses **24**,
the later figure. A launcher must query the device regardless.

This is part of CireSnave's ruling that baracuda supports sm_61 through the
latest architecture via Unpopped's CUDA parse/emit. Nothing in Unpopped's IR is
arch-specific. These rows only inform which schedule variants are worth
offering.

## 2026-10-02 — everything at `0.14.0`

One breaking release that conforms `unpopped-vocab`'s structure-key derivation
to KISS-Classify as of **KISS#517 + KISS#519** (`KISS@bc16715` plus the #519
amendment). Tracking: #31.

⚠️ **Wire-visible. Tokens change for the inputs below.** If you cache on the
token, re-derive the affected entries rather than migrating them.

**Every same-rank cell with ordinary strides keys exactly as in 0.13.0.** The
KISS byte-match vectors and every pre-existing token test pass unchanged.

### Which tokens change

| Input | `0.13.0` | `0.14.0` | KISS-CLASSIFY |
|---|---|---|---|
| A lower-rank operand in a higher-rank cell, e.g. `[256]` with `[128,256]` | mask `00` | mask `01` (in frame coordinates) | §6.5-0014 (as amended by #519), §6.6-0008, §6.6-0013 |
| An own extent-1, stride-0 axis where the frame is wider, e.g. `[1,256]` strides `[0,1]` | mask `00` | mask `01` | §6.6-0008 |
| A trailing unit axis, e.g. `[4,1]` | `d4` (and possibly a packed width) | `da` / `v1` | §6.3-0011 |
| A zero-extent innermost axis, e.g. `[8,0]` | `d8` | `da` / `v1` | §6.3-0011, §6.5-0012 |
| A mixed-rank work class, e.g. `[2,64]` with `[64]` | `grid` (left-aligned frame: 4096) | `block` (right-aligned: 128) | §6.6-0013, §6.5-0010 |
| A token naming `f8e6m2` | parsed and keyed | **typed decline** (`ReservedDtype`) | §6.1-0013 |

**Unchanged on purpose:** a rank-deficient operand is still laid out from its
**own** axes. It is not `br` and not forced to `v1` merely for lacking a frame
axis. KISS#517's first text padded the layout too. That gave im2col's dense and
strided rank-3 outputs the same token, which Unpopped's im2col gate caught.
KISS#519 amended the clause.

### New API: `try_structure_key` / `try_structure_key_token`

§6.6-0021: a scale-type dtype (`f8e8m0`, `f8e6m2`) at operand 0 has no defined
primary dtype, and derivation MUST decline. The new functions return
`Result<_, DeriveDecline>` with `DeriveDecline::ScaleDtypeAtOperand0 { dtype }`.
On every other input they return exactly what the infallible pair does.

**`structure_key` / `structure_key_token` stay, undeprecated for now.** They key
a scale-first list with the scale's dtype, which is non-conformant for that one
input, and their docs say so. `#[deprecated]` arrives in the release that
migrates this workspace's own ~200 call sites.

A scale in any *other* operand slot (the sk4 sibling model) is unaffected.

### Consumers in this workspace

- **The im2col plan gate** now admits mask bit 0, the frame axis a rank-3 output
  cannot span, and still requires the output's own layout to be dense.
- **`contract::index_is_1d`:** a rank-deficient index tensor gathered along the
  **last** axis now reads as a 1-D (broadcast) index. Its frame mask has the
  leading bits set, where its empty own mask used to read as full-shape. This
  is the right reading, but it changes the emitted contract for that input. No
  in-tree test exercises a rank-deficient index.

### Vendored KISS artifacts

- Re-vendored from `KISS@bc16715`, verbatim; the files hash to KISS's blob IDs.
- The Vulkan vocabulary moved 4 → 5. It was re-verified, not just re-pinned: v5
  only adds `i16`/`i64`/`f64` arith names, and no vendored vector names them.
- KISS's §6.8-0002 discriminating target-match pairs are now asserted.

### Not in this release

- The §6.6-0019 weight-role `<wdt>` (the `gem_weight_role_discriminator`
  vector). It needs role hints in the derivation API, and has a separate ticket.
- The stride-0 vs nonzero-stride unit-axis mask question (two tokens for one
  broadcast meaning). The derivation follows the text literally, and KISS has
  raised the question with the PM.

**Cross-checked against an independent implementation:** the `[4,1]` token and
the right-aligned work class match Fuel's own deriver, run on the same inputs.
The #519 examples are pinned as goldens.

> **About the merge commit.** `9574050`, the squash of #32, is titled *"(DRAFT) …
> awaiting baracuda + Fuel sign-off"*. The squash took the PR's title as it stood
> before the sign-offs arrived. Both had landed before the merge, and both are
> recorded on #32: Fuel's by its own re-run (fuel#285), baracuda's as relayed by
> the portfolio PM. Published 2026-10-02 07:43–07:44Z from that commit, in the order
> `unpopped-vocab` → `unpopped` → `unpopped-cpu-c` → `unpopped-slang`.

## 2026-10-01 — everything at `0.13.0`

### ⚠️ Structure keys change for two zero-valued inputs (wire-visible)

Two derivations read zero as "maximally divisible", because zero passes every
`x % n == 0` test. Both were live in `0.12.0` and both are fixed (#29):

| Input | `0.12.0` derived | `0.13.0` derives | KISS-CLASSIFY |
|---|---|---|---|
| A contraction with `K = 0` (rank 2 or 3) | `k_div` `d16` | `da` | §6.5-0012, §6.6-0016 |
| An operand with `align_bytes = 0` (unspecified alignment) | a packed width, e.g. `v4` | `v1` | §6.5-0009 |

**Who is affected:** only callers who build those two inputs. Every other key is
byte-identical; the full workspace and its KISS byte-match vectors pass
unchanged. **If you cache on the token, a cached entry for either input was
keyed under the wrong cell.** Re-derive it rather than migrating it.

**Why it went unnoticed:** the per-operand sub-key reaches the divisibility
ladder only through `inner_axis`, which picks an axis of extent `> 1`, so that
path never passes `0`. The contraction path calls it on the raw K extent. An
outside report of the K case was retracted on the strength of the per-operand
path, and measuring through the public `structure_key` reversed the
retraction. Pinned by `unpopped-vocab/tests/zero_is_not_maximally_divisible.rs`,
whose three tests fail on `0.12.0`.

**Not changed here, changed in `0.14.0`:** an operand whose innermost axis has
extent `0` (e.g. `[8, 0]`) derives `da` / `v1` from axis `rank − 1`, not `d8`
from the outer axis. KISS closed the "active axis" question for §6.3-0011
(innermost active axis is axis `rank − 1`, no non-unit exclusion; §6.5-0012
buckets `E = 0` as `da`). Measured at `0.16.0` and at sk5: `[8,0]` strides
`[1,1]` keys `co/00/v1/da/f`, as KISS's
`innermost_axis_extent_zero_is_axis_rank_minus_1` vector says.

## 2026-10-01 — everything at `0.12.0`

Covers `0.11.1`–`0.12.0`, none of which was published before this.

### ⚠️ BREAKING — the `seam` feature and `unpopped::jit::seam` are removed

`0.11.0` shipped an optional `seam` feature (`seam =
["dep:fuel-kernel-seam-types"]`) and `#[cfg(feature = "seam")] pub mod seam` in
`jit.rs`, which converted Fuel's `PatternNode` into this crate's. Both are gone,
along with the `fuel-kernel-seam-types` dependency (#18).

**Why:** this crate's public API named another project's type, so every major
release of `fuel-kernel-seam-types` forced a major release of `unpopped`. It had
already happened once. A `0.10.3` → `0.11.2` bump put two incompatible
`PatternNode` types into `baracuda-cuda-emit`'s dependency graph.

**Migrate:**
- To synthesize from this crate's own `unpopped::pattern::PatternNode`, call the
  native entry point `unpopped::jit::synthesize(&JitRequest, ..)` directly.
- If you start from Fuel's `PatternNode`, the conversion now lives in
  `baracuda-cuda-emit` (`src/seam.rs`, relocated near-verbatim). That crate
  already depends on `fuel-kernel-seam-types` directly.
- `--all-features` no longer turns on anything but `convert`.

The five tests that reached synthesis only through the seam (happy path,
recursion, three typed declines) now call the native entry point, in
`tests/native_synthesize_reaches_core.rs`. Before this change, no test in the
crate called that entry point directly.

### `convert` (the CUDA/Slang lifter, off by default) lifts more

- **Local variables resolve (#22).** A local that is assigned exactly once is
  substituted into the lifted body. Compound assignments (`+=`) and `++`/`--` count
  as further assignments, so a running accumulator is never mistaken for a
  constant. The crate's own scan-vs-elementwise test caught a first draft that
  missed them.
- **Comparisons and the ternary operator lift (#25)** to `Cmp*` and `Select`.

Both target constructs the lifter previously *refused*: an unresolved
identifier, or an unrecognized comparison token. They are meant to widen what is
accepted, not to change what an already-accepted kernel lifts to. That is the
design intent. No before/after comparison over a corpus has been run to measure
it.

### `unpopped-vocab`: the `cuda:` reserved id block is gone (#15)

The four `cuda:sm*` target tokens were interned at fixed ids `0`–`3`, the last
CUDA-specific vocabulary in this neutral crate. They now intern like every
other namespace's tokens, via `TargetId::parse`. **Only code that relied on a
`TargetId`'s numeric value could notice, and that was always forbidden:** ids
are process-local and never serialized. Tokens and `==` are unchanged.

### Also in this range, with no behaviour change

- **Doc comment correction (#8):** the `VariantFidelity::DeterministicallyDivergent`
  doc comment no longer cites the retracted measurement (see the 0.11.0
  correction below). The fix landed after `0.11.0` was packaged, so `0.12.0` is
  the first published version to carry it.
- **New tests that pin existing behaviour:**
  - the one-version rule (#7);
  - dtype `storage_bits`/`kind` checked against KISS's manifest (#16);
  - `Complex64`'s width (#19).
- **Dependencies:**
  - lockfile refreshed to the latest in-range versions (#20);
  - `kiss-ref-core`/`kiss-ops-vocab`/`kiss-classify-vocab` `0.3.4` (#24), a
    dev-dependency only, never reaching consumers.
- **Design and sizing docs:** #10–#12, #14, #26, #27. The
  `golden_regen_checkpoint` date moved to 2026-10-22 (#28). CI's checkout
  action moved to v7 (#9).

## 2026-09-09 — everything at `0.11.0`

⚠️ **`unpopped-vocab 0.4.0 → 0.11.0`, `unpopped-cpu-c 0.9.0 → 0.11.0`,
`unpopped-slang 0.7.0 → 0.11.0`, `unpopped 0.10.0 → 0.11.0` — the skipped version
numbers are a DELIBERATE UNIFICATION, not a mistake, and this line exists so
nobody re-derives it later.**

**CireSnave's standing rule, quoted:** *"I always want all crates within a project
to use the same version number so developers consuming them know which go with
which."* The exception in that rule is for a **cross-project** pair — an emitter
crate in another project whose number answers *"which Unpopped does this work
with"*. `unpopped-cpu-c` and `unpopped-slang` are **inside** this project, so
their numbers answered nothing and the exception does not reach them.

**`unpopped-conformance` moves too, although it is `publish = false` and has no
external consumer.** Leaving one crate off the shared number recreates the
question the rule exists to close.

### ⚠️ For consumers: one number, written four times, and all four must move together

Before this release a consumer wrote **three different numbers** for four crates.
Now they write `0.11.0` four times.

**Every current pin excludes `0.11.0`, so nothing arrives on a `cargo update`:**

    ^0.10.0  = >=0.10.0, <0.11.0     excludes 0.11.0
    ^0.4.0   = >=0.4.0,  <0.5.0      excludes 0.11.0
    ^0.9.0   = >=0.9.0,  <0.10.0     excludes 0.11.0
    ^0.7.0   = >=0.7.0,  <0.8.0      excludes 0.11.0

⚠️ **A PARTIAL edit resolves two versions of a crate into one graph** — e.g.
moving `unpopped` to `0.11.0` while leaving `unpopped-vocab` at `^0.4.0` gives
`0.4.0` (the direct pin) *and* `0.11.0` (what `unpopped 0.11.0` requires), which
are different types with the same name. **Unification makes that mistake more
visible, not less: four identical numbers make a missed line obvious, where three
different numbers made it look plausible.**


**Remedies issue #4 for TODAY'S INSTANCE ONLY.** ⚠️ Publishing makes published ==
workspace **at one moment**; the member keeps evolving and the gap reopens on the
next unpublished commit. **A green publish is not the class being closed** —
baracuda's argument, and the portfolio gate vulkane is building is the durable
half.

### ⚠️ BREAKING — `VariantFidelity::ReassociatedDeterministic` → `DeterministicallyDivergent`

The old name asserted **reassociation** for every member of the class. The new one
names the class extensionally — deterministic on fixed hardware, different bits
from the default, undirected — and **asserts no mechanism**.

**One selection policy, so one variant** — a fifth with identical semantics is a
distinction no consumer can act on. **The FKC determinism spelling is unchanged
(`same_hardware_bitwise`)**, asserted by `the_fidelity_rename_is_wire_invisible`.

> 🔴 **CORRECTION, 2026-09-11 — this entry originally cited a RETRACTED
> measurement, and the retracted text is immutable in the published `0.11.0`.**
>
> It read: *"baracuda measured a member whose reduction tree was provably
> unchanged and whose bits differed anyway — cache-vs-recompute of an equal
> `expf`, 12,283,172 of 16,777,216 elements, worst 11 ULP, reproducible."*
>
> **That measurement was withdrawn on 2026-09-09 (baracuda#99, closed
> `not_planned`), two days before `0.11.0` shipped.** It had been taken against
> `variants[1]` — which is `prec`, declared `MorePrecise`, whose entire purpose is
> to differ from the base. Measured against `smemrow`'s own kernel: **0 ULP over
> 16,777,216 elements.** `smemrow` is `BitIdentical` and always was.
>
> ⚠️ **So the second mechanism this rename was argued from has NO measured
> instance.** It is recorded in `backend.rs` as a hypothesis rather than an
> observation.
>
> **The name is kept.** The old name asserted a mechanism for every member; this
> one asserts none, and **removing an unsupported claim is not the same act as
> adding one.** Reverting would re-assert *"every member is a reassociation"* on
> evidence no better than what was withdrawn, and would cost the one adopter a
> second 19-site rename. **If the class turns out to be reassociation-only, this
> name is less specific than it could be, which is not wrong.**
>
> **How it happened:** the retraction reached baracuda's issue and two of their
> code comments, and never reached here. ⚠️ **A RETRACTION MUST TRAVEL AS FAR AS
> THE CLAIM DID — and a correction is *more* discoverable than the original while
> still not travelling, because discoverability is pull and propagation is push.
> Nobody goes looking at a number they have already written down.**

### `ulp_bound` no longer declines an expression whose exactness is by construction

An expression built entirely from `BinaryOp::is_int_only` operators has no
rounding step on any hardware, so a **non-CUDA** target now gets `0.0` where it got
`INFINITY`, and `precision_of` rates it `("correctly_rounded", Some(0))` instead
of `("approximate", None)`.

⚠️ **Keyed on `is_int_only`, NOT on `ulp_sum(e) == 0`.** `unary_ulp` rates
`Sqrt`/`Recip`/`Floor` at 0.0 and `binary_ulp` rates
`Copysign`/`Nextafter`/`FmaxIeee` at 0.0 — **those zeros are IEEE claims about a
target's float unit**, exactly the borrowed assertion the namespace gate exists to
refuse. **A CUDA target is untouched**, asserted across four rating tiers.

Found by baracuda while adopting 0.10.0, applying this repo's own rule: *a
prescription that errs conservatively has no complainant.*

### `Access::Contraction` is documented ALWAYS COMPUTED, and its premise is guarded

"No predicate applies" is an answer; its absence read as an oversight. The K-fold
is a sum over products and the variant carries **no operator field at all**.
Guarded by an exhaustive match on `AccumSpec` — **0 exhaustive matches existed
before, so a variant that falsifies the ruling used to compile with 0 errors, 0
clippy warnings and 0 test failures.**

### ⚠️ Why the emitters move at all

`unpopped-cpu-c 0.9.0` and `unpopped-slang 0.7.0` **both require `unpopped =
"0.10.0"`** — measured from their served manifests. For a 0.x crate `^0.10.0`
**excludes 0.11.0**, so publishing `unpopped` alone would strand both emitters on
the old line while every sibling moved. **`baracuda-cuda-emit` declares all four**,
so it would then resolve two `unpopped` versions into one graph — the two-artifacts
defect this release exists to close.

**Source of both emitters is byte-identical to their published versions. They move
because a dependency did, and then again because of the unification above.**

⚠️ **This section was headed *"Why three crates and not one"* until the
unification, and the count was correct when written.** The unification made it
five, and **a heading carrying a stale number is the artifact a reader trusts
most** — it was found by a reviewer's nitpick about an unrelated count two
sections above, which is the only reason anyone re-read this one.

## 2026-09-06 — `unpopped-vocab 0.4.0` · `unpopped 0.10.0` · `unpopped-cpu-c 0.9.0` · `unpopped-slang 0.7.0`

⚠️ **Everything below shipped. It sat under "Unreleased" for forty minutes after
being published** — caught while adding the next entry, which is the only reason
it was caught at all. **A changelog whose "Unreleased" section describes released
work is worse than no changelog: it is the same object as a version number that
no longer matches its code, one level up.**

**Published and verified against the served tarballs; `unpopped-vocab 0.4.0` and
`unpopped 0.10.0` src trees are byte-identical to the merge commit.**

**Version cascade applied 2026-09-06 on the PM's ruling: `unpopped-vocab 0.4.0`,
never `0.3.3`.**

| crate | was | now | why it moved |
|---|---|---|---|
| `unpopped-vocab` | 0.3.2 | **0.4.0** | two behaviour changes, one numeric |
| `unpopped` | 0.9.0 | **0.10.0** | new public fn, a deprecation, and its vocab requirement moved |
| `unpopped-cpu-c` | 0.8.0 | **0.9.0** | ⚠️ **source BYTE-IDENTICAL to the published 0.8.0** — bumped ONLY because a dependency did |
| `unpopped-slang` | 0.6.0 | **0.7.0** | ⚠️ **source BYTE-IDENTICAL to the published 0.6.0** — bumped ONLY because a dependency did |
| `unpopped-conformance` | 0.1.0 | 0.1.0 | `publish = false`; not on the registry |

⚠️ **The two "byte-identical" rows are stated because a reader who sees a version
move with no behaviour change will otherwise assume there was one they cannot
find.** Both were verified by diffing the served `.crate` tarball against the
tree, not by inspection.

### Downstream, measured rather than reasoned

**Registry reverse-dependencies of `unpopped-vocab`: 6, all in this portfolio**
(control: `serde` returns 120,294, so the query works).

    baracuda-cuda-emit / -kernels-types / -types   require ^0.1.0  (an OLD line;
                                                    0.3.x never reached them)
    unpopped / unpopped-cpu-c / unpopped-slang     require ^0.3.2

⚠️ **And the hop that mattered was the one nobody had measured.** baracuda's
**working tree** declares `unpopped-vocab = "0.3.2"` **directly** — measured at
their `origin/main` `2b3292ca`, after fetching, because the local ref was stale.
**So the reassurance "baracuda is caret-pinned `^0.8.7` on `unpopped`, therefore
cannot reach the new vocab" was FALSE**: their direct vocab dependency would have
taken `0.3.3` on the next `cargo update`, bypassing the `unpopped` pin entirely.

**Their exposure to the numeric change is nonetheless zero, measured on their
side:** `Fp8E5M2::from_f32` has **0** call sites in baracuda. **`Fp8E4M3FN` is
unchanged by this release** — 0 diff lines between published 0.3.2 and the tree
mention E4M3, against a control of 5 mentioning E5M2 — so their two E4M3 call
sites are unaffected.


### `unpopped-vocab` 0.3.2 → 0.4.0 — the divergence that forced this release

**Two behaviour changes. Neither could ship as `0.3.3`.** The then-published
`unpopped 0.9.0` requires `unpopped-vocab = "0.3.2"`, a caret range that
**accepts 0.3.3** — so a patch bump reaches every existing consumer on their next
`cargo update`, with no compile error and no version signal.

- **`ElementKind` E5M2 `from_f32` overflow now produces INFINITY, not a saturated
  max-finite.** Overflow at `|x| >= 61440.0` (the midpoint between max-finite
  `57344` and `2^16`; the midpoint itself rounds to infinity, because
  ties-to-even picks the significand ending `00`) returns `sign | 0x7C`.
  Previously it delegated to `float8` 0.7.0, whose E5M2 encoder saturates **every**
  overflow to `0x7B` — including a literal `f32::INFINITY` — while **its own
  decoder maps `0x7C` to infinity**. ⚠️ **This is a NUMERIC change to emitted fp8
  bytes**: a value that encoded as `57344.0` now encodes as `inf`.
- **`winner_of` resolves an exact timing tie deterministically.** Previously
  `sort_by` stability made the winner, its `entry_point`, and the `margin` depend
  on the order the caller pushed candidates. The tiebreak is
  median → implementor code → entry point and carries **no** performance or
  preference meaning. ⚠️ **Emitted dispatch tables can differ for tied cells**,
  which is generated code a consumer has committed.

### `unpopped` 0.9.0 → 0.10.0

- **Added `ir::is_bit_move_row_reduce_output(stages, epilogue)`** — the
  §6.16-0011 classification for the `Access::RowReduce` shape, which had no
  helper at all. Purely additive.
- **Deprecated `ir::is_bit_move_reduce`** — it answers a fold question for shapes
  that may have several folds. Non-breaking; the two replacements have signatures
  that cannot be satisfied without the whole input→output path.
- **Corrected the `RowReduce` gating prescription in `ir.rs`'s field table.** The
  old advice classified an all-move multi-stage fold as *computed*. ⚠️ **It erred
  CONSERVATIVELY, so a consumer following it emitted pessimised-but-conforming
  code and had no symptom to report.**

### `unpopped-cpu-c` 0.8.0 → 0.9.0 · `unpopped-slang` 0.6.0 → 0.7.0

- **Published 0.8.0 source is IDENTICAL to the tree.** No bump needed on its own
  account; it moves only if its `unpopped` requirement does.

---


Versions at or before `unpopped 0.9.0` / `unpopped-cpu-c 0.8.0` /
`unpopped-vocab 0.3.2` predate this file. **Their contents were verified against
the served tarballs rather than reconstructed from memory**, which is the only
claim this file makes about them.
