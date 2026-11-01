# E1d: parallel mapping from the kernel layer, and the patent re-read it triggers

**Status: design note, docs only. Awaiting the PM's ruling.** PM task 2026-10-31, item (2): write the E1d note and re-read
US10949182B2. `docs/e1-emit-from-the-kernel-layer.md` section 6 names E1d ("grid-stride over the outer loop; block-tree
`RowReduce` with `Barrier`") and section 7.4 says E1d is the stage that triggers the patent re-read. Not legal advice and not
a freedom-to-operate opinion; see "What this read is" below.

## What exists, and who owns what

- The kernel layer already has the parallel vocabulary: `KernelFn::launch: Option<LaunchShape>`, `ScalarExpr::Exec(ExecId)`,
  `Stmt::Barrier`, `Shared` allocations, and `run_launch`, which runs every thread of a launch in plain Rust
  (`docs/u3-row4-scopes-and-memory-spaces.md`, all three PRs landed). A `LaunchShape` is a **declared property of the kernel**;
  nothing in this crate computes one.
- `run_kernel` runs a kernel as one logical thread and is the oracle for every lift differential.
- Emitters: cpu-c is serial by nature. The Slang emitter (E1c) is **Vulkane's** since 2026-10-30; CUDA is Baracuda's. So E1d's
  emitter half is not this lane's. What this lane can own is the **target-neutral half**: a transform from a serial `KernelFn`
  to a `KernelFn` with a `LaunchShape`, proved against `run_kernel` by `run_launch`, which needs no GPU and runs on GitHub.

## The design

**E1d-0, grid-stride elementwise.** A transform `parallelize(kernel, axis, block) -> KernelFn` takes a kernel whose outer
`For` is declared parallel and rewrites it to `For i in gid .. n step gridDim*blockDim`, with
`gid = BlockIdx*BlockDim + ThreadIdx` read from `Exec` leaves. `block` (threads per block) and the grid are **inputs** to the
transform, carried in the resulting `LaunchShape`; the grid is a fixed number chosen by the caller (a device-class constant,
for example some multiple of the multiprocessor count), not derived from `n`. Correctness for any grid is the point of a
grid-stride loop, so a test can run grids of 1, 3 and 7 blocks against the serial result.

**E1d-1, block-tree `RowReduce`.** One cooperating block reduces a row: each thread accumulates a strided slice into a
`Shared` cell, then a log-step tree with a `Barrier` between rounds, then thread 0 stores. To keep the block count independent
of the row count, blocks stride over rows too (`row = BlockIdx; row < rows; row += GridDim`). The accumulation **order** differs
from the serial fold, so this stage is bit-exact only where the reduction is order-independent (integer, `Max`/`Min`) and a
documented tolerance case for float `Sum`; that is a numerics ruling for the PM, not assumed here.

**Parallelism is declared, never discovered.** The lift knows which axes are independent because of the `Access` family it
lifted (every output coordinate of an Elementwise op; every row of a `RowReduce`), so it marks the axis parallel. The transform
trusts the mark; the proof is the differential, not an analysis. This is deliberate, and the patent section below is why.

**The emitters consume the result.** A backend that receives a `KernelFn` with a `launch` spells `Exec` as its thread ids
(`SV_DispatchThreadID`/`threadIdx` and so on) and reports the declared `LaunchShape` in its manifest so the host launches it.
Unpopped emits no launch directive.

## Patent re-read: US10949182B2 (MathWorks, "Systems and methods for generating code for parallel processing units")

Read 2026-10-31 on the Google Patents text, `patents.google.com/patent/US10949182B2/en`: status Active, priority 2016-11-17,
granted 2021-03-16, **estimated expiry 2037-11-16** (Google's own assumption, not a legal conclusion; the 2026-10-25 sweep
recorded "no expiry shown"). 77 claims; independent claims **1, 15, 26, 30, 44, 58, 64, 70, 74**. I read the full text of 1, 15,
26, 30 and 44. Claims 30 and 44 repeat claim 1's steps word for word (as a medium and an apparatus). I also read the opening
elements of 58, 64, 70 and 74 (they match 15 and 26 as the medium/apparatus forms); I did not read their closing elements.

| independent claim | what it requires (as read) | does the E1d design do it? |
|---|---|---|
| 1, 30, 44 | generate an IR; identify a nested loop of for-loops with loop bounds; **apply a parallel loop analysis check** to pick the loops suitable for parallel execution; **determine the number of thread blocks and threads per block based on one or more of the loop bounds**; convert the loops to a kernel; **add a kernel launch directive** carrying those two numbers | **No on three elements.** Parallelism is declared by the lift from the op family, with no analysis check; the grid is a caller-chosen constant and the grid-stride and row-stride loops make correctness and block count independent of any loop bound; no launch directive is emitted |
| 15, 58, 64 | partition the IR into serial and parallel segments; convert a **serial segment between two parallel segments into a single-thread, single-block kernel**; add a launch directive | **No.** There is no serial/parallel partitioning and no single-thread kernel is generated; thread 0 storing a reduction result inside one kernel is not a separate kernel |
| 26, 70, 74 | a table mapping sequential function calls to predefined parallel library calls (FFT, DNN, BLAS, SOLVER libraries), substituted on a match | **No.** No library-call substitution exists in this IR or in E1d |

Dependent claims 5 and 6 compute the thread count from the loop-iteration product and the device warp size, and claim 7/8 a grid
dimension from the iteration count; the design does none of these for the same reason as claim 1's fourth element.

**Reading: outside as read**, for the E1d design as written, on the claims above. The word is **not** "clear".

The design constraints that keep it there, to be enforced as written rules rather than remembered:

1. No pass in this crate derives a block or grid count, or a thread count, from a loop bound or an iteration count.
   `LaunchShape` values come from the caller or a device-class table. (A `Param(local)` extent in `LaunchShape` is a launch-time
   scalar the host supplies; it is not computed by us from a bound.)
2. No pass decides which loops are parallel by inspecting them. The mark comes from the lift's `Access` family.
3. No emitter in this repo writes a launch directive. The manifest reports the declared shape.
4. If anyone proposes an automatic dependence/independence analysis, a derived geometry, or a launch statement, **stop and
   re-read this patent first.**

## What this read is, and is not

- **Is:** the claim text of the nine independent claims as published on Google Patents, compared against a written design.
- **Is not:** a read of the specification, the USPTO file wrapper or prosecution history (which can narrow a claim's scope), or
  of any claim by number beyond those named above; not a read of what Baracuda's or Fuel's CUDA emitters do (the 2026-10-25 sweep
  said they "should be charted separately" and that is still true); not a legal opinion; and the design is not code yet, so the
  re-read must be repeated against the code when E1d-0 lands.
- The earlier sweep's "the Slang emitter hard-codes `[numthreads(256, 1, 1)]`" is no longer a statement about this repo (the Slang
  emitter moved to Vulkane); the E1d constraint that matters here is rule 3.

## Decisions for the PM

1. **Scope**: this lane owns the target-neutral transform and its `run_launch` proof (E1d-0, E1d-1); the Slang spelling is
   Vulkane's (E1c follow-on) and CUDA's is Baracuda's. Recommend yes.
2. **Declared-parallel axes** with the proof by differential, no analysis pass (rule 2). Recommend yes; it also keeps the lift
   simple.
3. **Float `Sum` order** in E1d-1: accept a documented tolerance bound against the serial fold, or restrict block-tree
   `RowReduce` to order-independent reductions first. Recommend restrict first, tolerance later with its own note.
4. **Patent rules 1-4** recorded as standing constraints in `PATENT-SWEEP.md` and `ROADMAP.md` section 5 once ruled.
5. **Stages**: E1d-0 as one PR (transform plus `run_launch` differential on grids 1, 3, 7), E1d-1 as a second.

## Proof obligations if ruled

- E1d-0: for each Elementwise family, `run_launch(parallelize(k), grid g)` equals `run_kernel(k)` byte for byte, for grids
  `{1, 3, 7}` and block sizes `{1, 32, 256}`, over the existing lift differential inputs. A mutant that drops the stride
  (each thread does one element only) must go red on `n > grid*block`; a mutant that breaks `gid` must go red.
- E1d-1: the same per reduction kind, with the order-independence caveat above; a mutant that removes a `Barrier` must be
  reported by the checker's `BarrierMismatch`/`DataRace`, not by luck.
- A text check that no source file in `crates/` contains a launch-directive spelling such as `<<<` or `dispatch` in an emitter
  that runs from this repo (rule 3), so the constraint is enforced by a test rather than by memory.

## E1d-1 as built

`kernel_lift::lift_reduction_opdef_launched(op, dtype, rank, block, grid)` (`kernel_lift/block_reduce.rs`). Per output, one
block: each thread folds `reduced = tid; < n; += BlockDim` into a private accumulator, publishes it to a `Shared` array
(`partial`, plus `partial_have` for `Max`/`Min`), and `block/2, block/4, ... 1` tree rounds (`if tid < s`, then a `Barrier`)
combine the partials with the serial fold's own step; thread 0 stores. Blocks stride over the outermost kept axis
(`row = BlockIdx; += GridDim`). Geometry is the caller's; `block` must be a power of two.

- **Accepted** (order-independent, so bit-exact against the serial kernel): integer `Sum`/`Prod`/`Max`/`Min` (wrapping is
  associative and commutative) and float `Max`/`Min` (NaN-keeping). Exceptions, as for every float kernel here: NaN payload,
  and the sign of a zero when both `+0` and `-0` occur in a `Max`/`Min` (the serial fold keeps the first, the tree whichever it
  combines first). The proof corpus has no `-0.0`.
- **Declined** with a typed `LiftDecline`: float `Prod` (float `Sum`/`Mean` are E1d-2, below), integer `Mean`, a fused post, more than one reduced axis, no kept axis (a rank-1 reduction), a non-power-of-two `block`.
  `Access::RowReduce` (float, multi-stage, tolerance-mode) is E1d-3, below.
- **The barrier checker** treated every `Exec` leaf as thread-divergent, so a barrier inside a block-strided row loop was
  rejected. `BlockIdx`, `BlockDim` and `GridDim` are now uniform (every thread of a block reads the same value); `ThreadIdx`,
  the lane and subgroup ids stay divergent, and the pinned over-rejections are unchanged.
- **Proof** (`tests/kernel_parallel_reduction_agrees.rs`): `run_launch` equals the serial kernel byte for byte over six
  reduction kinds, three axis/keepdim layouts and seven geometries (one thread to 12 blocks of 16; a 64-wide block over a
  37-wide row, so threads with nothing to fold are exercised). Six broken copies must disagree or trap: a removed first
  `Barrier` (`UninitializedShared`), a removed round `Barrier` (`DataRace`), a dropped last round, an unstrided slice, a combine
  that ignores the partner's `have` flag (seen on an all-negative `Max`), and a wrong block start.

## E1d-2 as built

`lift_reduction_opdef_launched` also accepts a float `Sum` and `Mean`, held to a declared bound instead of the serial kernel's bytes
(PM ruling 2026-10-31: "allowed with a DECLARED bound: error <= gamma_(n-1) * sum|x_i| against the oracle, and the result
reproducible run-to-run for a fixed (grid, block); write the order in the note and test both").

- **The order**, fixed by `(grid, block)` and nothing else. With `B = block`: thread `t` folds the row's elements `t, t+B, t+2B, ...`
  ascending into an `f64` accumulator starting at `0.0`; then for `s = B/2, B/4, ..., 1` each thread `t < s` sets
  `acc[t] = acc[t] + acc[t+s]` (the thread's own partial is the left operand); thread 0 holds the total, divides it by the reduced
  extent for `Mean`, and rounds once to the compute dtype. A block wider than the row leaves threads with nothing to fold: they
  contribute `0.0`, which is exact. At `B = 1` this is the serial order, so the bytes equal the serial kernel's.
- **The bound.** Let `u = 2^-53` and `gamma_k = k u / (1 - k u)`. Any order of the same `n - 1` additions in `f64` has
  `|computed - exact| <= gamma_(n-1) * sum|x_i|` (the standard summation bound); the launched sum meets it, plus the half ulp of the
  final rounding when the compute dtype is `F32`. For `Mean` everything is divided by `n`. Against the serial kernel, which meets the
  same bound, the difference is at most twice that. The bound is stated in absolute terms on purpose: on a cancelling row (`1e8`,
  `-1e8`, ...) it is large against `|exact|`, and no relative claim is made there.
- **Reproducibility.** The same `(grid, block)` gives the same bytes on every run; the bytes do not depend on `grid` (a row's order is
  fixed by `block`; `grid` only chooses which block handles which row).
- **Still declined:** float `Prod`, integer `Mean`, a fused post, several reduced axes, no kept axis (as in E1d-1).
- **Proof** (`tests/kernel_parallel_reduction_float_sum_bound.rs`): over `F32` and `F64`, `Sum` and `Mean`, six geometries (one thread to
  a block wider than the row), five rows (benign, cancelling, huge/tiny, `0.1`, mixed signs; the reference is a Neumaier sum): the
  bound against the near-exact value and against the serial kernel; run-to-run and grid independence; the kernel's bytes equal a
  plain-Rust reimplementation of the order above (bit for bit); a dropped last round and an unstrided slice are each seen.
  Scope: the proof is the kernel interpreter's (`run_launch`); an emitter that contracts `acc + x` into an FMA, or reorders the
  tree, is outside this claim and has its own test.

## E1d-3 as built

`lift_row_reduce_opdef_launched(op, dtype, rank, block, grid)` (`kernel_lift/row_reduce_launched.rs`) is `Access::RowReduce` on the
block-tree. The serial lift (`lift_row_reduce_opdef`) is the specification: it rounds each stage once to the compute dtype and runs
later stages and the epilogue on that (tolerance mode against the oracle, 2 ulp at `F32`). The launched lift keeps every one of those
choices and changes only who does the folding.

- **Per row, one block**, blocks striding over the outermost leading axis (`row = BlockIdx; += GridDim`), `rank >= 2`.
- **Stage order.** For stage `s = 0, 1, ...`: thread `t` folds columns `t, t+B, t+2B, ...` ascending into an `F64` accumulator over the
  stage's `pre` expression (which may read `Reduced(u)`, `u < s`); the partials are published and combined by the E1d-1 tree
  (`acc[t] = acc[t] (op) acc[t+k]`, `k = B/2 ... 1`, a `Barrier` after the publish and after each round); thread 0 finalises (`Mean`
  divides by the row length), rounds once to the compute dtype and stores the result in `Shared` slot `s`; a `Barrier`; every thread
  loads the slot into its `Reduced(s)` local. After the last stage every thread runs the epilogue over columns `t, t+B, ...`, reading
  the loaded stage results. The broadcast barrier is what makes a stage result visible to the next stage's and the epilogue's readers;
  the stage-result slots are distinct per stage, so no stage overwrites a slot another thread may still read.
- **Numerics.** A `Max`/`Min` stage is a set function: bit-exact against the serial lift (except NaN payload and a zero's sign, as in
  E1d-1). A `Sum`/`Mean` stage's `F64` accumulation meets the E1d-2 bound, `gamma_(n-1) * sum|x_i|`, and is rounded once exactly as the
  serial lift rounds it, so its stage value differs from the serial lift's by at most that bound plus, at `F32`, one rounding step at a
  boundary. The epilogue then runs on the stage values it is given, so a later stage or epilogue that amplifies a stage value (an `exp`,
  a division by a stage value) can move an output cell by more than the stage's own difference; the proof bounds softmax and RMS norm at
  4 ulps of the serial lift on its corpus, which is a measurement of that corpus and not a bound for every input. The result is
  reproducible for a fixed `(grid, block)` and does not depend on `grid`. At `block = 1` the tree is empty and the bytes equal the
  serial lift's.
- **Declined:** rank 1, a `Prod` stage, an empty stage list, an integer dtype, a non-power-of-two `block`, a zero `grid`.
- **Proof:** `tests/kernel_parallel_row_reduce_agrees.rs`, listed in `CHANGELOG.md` 0.38.5. Same scope statement as E1d-2: the proof is
  the kernel interpreter's (`run_launch`).

## E1d-4 as built

`lift_reduction_opdef_launched` (E1d-1, E1d-2) now also takes the three forms it declined:

- **Several reduced axes.** The fold loop nest is the serial lift's; only the outermost reduced axis is thread-strided (`i = tid; < n; += BlockDim`),
  so thread `t` folds, in row-major order, everything under the indices `t, t+B, ...` of that axis. Parallelism is limited by that axis's
  extent (a block wider than it leaves threads idle, which is exact for `Sum` and handled by `have` for `Max`/`Min`); correctness is not.
  The tree and the bound are unchanged (the bound holds for any order of the same additions); `Mean` divides by the product of the reduced
  extents.
- **No kept axis.** One output: the row is wrapped in `if BlockIdx == 0`, so only block 0 works and `grid` is irrelevant to the result.
  The condition is uniform across the block, so the `Barrier`s inside are legal.
- **A fused post.** Thread 0 holds the total; it rounds the finalised fold once to the compute dtype and the post reads that as `Reduced(0)`,
  the serial lift's own rule. A `Max`/`Min` op with a post is therefore bit-exact against the serial kernel; a `Sum`/`Mean` op is the written
  order followed by the same post, and equal to the serial bytes at `block = 1`. A post may not read an `Input` (as in the serial lift).
- **Proof:** `tests/kernel_parallel_reduction_forms.rs`.

## E1d-5 as built

`lift_scan_opdef_launched(op, dtype, rank, block, grid)` (`kernel_lift/scan_launched.rs`) is `Access::Scan` on the block. Let `B = block`, `n` the
row length, `c = ceil(n / B)`. Scan positions are numbered `j = 0..n-1` in scan order (position `j` for a forward scan, `n-1-j` for a reverse one).

- **Phase 1, totals.** Thread `t` folds the scan positions `[t*c, min(t*c + c, n))` in ascending `j` into an accumulator that starts at the monoid
  identity (`0`, `1`, or for `Max`/`Min` the first element through the `have` flag) and publishes it to `Shared`.
- **Phase 2, offsets.** For `d = 1, 2, 4, ... < B`: every thread `t >= d` sets `total[t] = total[t] (op) total[t-d]` (own value on the left), all
  reads of the round before any write (`Barrier` between). Thread `t`'s offset is the result of thread `t-1` (the identity for thread 0).
- **Phase 3, writes.** Thread `t` starts from its offset and walks its chunk again, writing the prefix as the serial scan does: before the fold when
  exclusive, after it when inclusive, a fused post reading the prefix rounded once to the compute dtype.
- **Numerics.** `Max`/`Min` and integer `Sum`/`Prod` are order-independent: bit-exact against the serial scan (NaN payload and the sign of a zero in
  `Max`/`Min` excepted, as in E1d-1). A float `Sum` output is the `f64` sum of at most `n` elements added in the order above, so it meets
  `gamma_(n-1) * sum|x_i|` over its own prefix plus the half ulp of the single rounding, is reproducible for a fixed `block`, does not depend on `grid`, and
  equals the serial scan at `B = 1`. A float `Prod` declines (as in E1d-1); `Mean` is not a monoid.
- **Chunk size.** `c` is computed in the kernel from the row length (a counting loop: the kernel integer subset has no division or shift in a float
  kernel). It decides only which thread does which positions. The launch shape stays the caller's: no pass derives a block or grid count from a
  bound (E1d rules).
- **A `have` guard on the partner is not needed** and is not emitted: chunks are contiguous from the front, so a thread with elements never has an empty lower
  partner, and an empty partner holds the identity, which a fold leaves unchanged. (A seeded mutant that ignored such a guard was not caught, which is
  how this was found.)
- **Proof:** `tests/kernel_parallel_scan_agrees.rs`.

## E1d-6 as built

`lift_window_opdef_launched`, `lift_im2col_opdef_launched`, `lift_contraction_opdef_launched` and `lift_row_sort_opdef_launched` (in `kernel_lift.rs`) are E1d-0's transform applied to four more families whose output cells
are independent: the outermost output axis becomes `i = BlockIdx*BlockDim + ThreadIdx; i < n; i += GridDim*BlockDim`, the rest of the serial kernel is
untouched. Each cell folds its own taps in the serial lift's order (window) or gathers one element (im2col), so there is no reordering, no `Shared`
memory and no `Barrier`, and the result is **bit-exact** against the serial kernel for every `(grid, block)`, floats included. Parallelism is declared from
the op family, never found by inspecting loops. A contraction thread runs the serial `k`-ascending accumulation for its output cells. A row-sort thread sorts whole rows with the serial algorithm (a total order over `(key, index)`, so bit-exact); a rank-1 sort has no leading axis and declines. Gather follows the same pattern and is not done yet; Scatter is not parallel in
general (two indices can write one cell). Proof: `tests/kernel_parallel_grid_stride_agrees.rs`.

`lift_gather_opdef_launched` is the same grid-stride over the gather's outermost output axis: each output cell reads its own index and source element and is
written by one thread, so it is bit-exact for every `(grid, block)` and every out-of-range policy (a skipped cell keeps its prior bytes). Proof:
`tests/kernel_parallel_gather_agrees.rs`. `lift_scatter_opdef_launched` grid-strides the source's outermost axis and combines with `Stmt::AtomicRmw` (`docs/ir-kernel-layer.md` section 5a), because two
source cells can name one destination cell and the serial lift's `Load` then `Store` would race (the launch interpreter reports `DataRace`). Integer
`AtomicAdd`/`AtomicMax`/`AtomicMin` are order-independent, so the destination is bit-exact against the serial kernel for every `(grid, block)`. An `Assign`
(duplicates would make the winner depend on the schedule) and a float add (order-dependent) decline. Proof: `tests/kernel_parallel_scatter_agrees.rs`,
`tests/kernel_atomic_rmw.rs`.

## Who owns a hardware width

An emitter owns the wavefront/warp width (and any other device fact: shared-memory size, register file). A `KernelPlan` carries none, so a plan stays
target-neutral; the same plan lowers to a 32-wide and a 64-wide target and each emitter reads its width from its target. The launch shape in a
`KernelFn` is a *declared property of the kernel* (E1d rules), not a hardware width. (PM ruling 2026-10-31, on Rockfish's finding that the plan carries no
wavefront size: the emitter owns it. `generate()` does not optimize, so emitters also round their own constants.)


## Provenance

Claim text read from the downloaded Google Patents page on 2026-10-31 (claims 1, 15, 26, 30, 44 in full; the dependent claims
cited by number). Kernel-layer facts from `docs/u3-row4-scopes-and-memory-spaces.md` and `docs/e1-emit-from-the-kernel-layer.md`
at `origin/main` `bcfae84`. Expiry and status are Google Patents' displayed assumptions.
