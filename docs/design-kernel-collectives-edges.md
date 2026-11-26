<!-- SPDX-License-Identifier: MIT OR Apache-2.0 -->
# Subgroup collectives: the IR defines the edge, emitters must spell it (2026-11-26)

Asked by vulkane through pm-kernels (`collective-edge-rules`, board 506 preparation; vulkane's `docs/design/506-subgroup-collectives.md` cites its sources). The oracle's edge semantics for `Stmt::Collective` were stated nowhere in the `CollectiveKind` docs; they are now (`crates/unpopped/src/kernel.rs`, `CollectiveKind::Shuffle` and `Vote`), and pinned by `tests/kernel_collective_edges.rs`.

**The rule.** The IR defines the result of every lane of a collective, including the edge lanes. An emitter **spells that rule** (for example, a select on the subgroup invocation id against the delta) and is device-tested at the edge lanes, or it declines the construct by name. It must not inherit what a device happens to do: WGSL leaves an out-of-range `subgroupShuffleUp/Down/Xor` source indeterminate, `subgroupBroadcast` needs a constant id in `[0, 128)`, HLSL has no shuffle up, down or xor intrinsic (a computed `WaveReadLaneAt` expresses them), and MSL's edge text is unverified (vulkane's note).

**Where each rule comes from (n lanes, `0..n`).**

| Case | IR rule | Source |
|---|---|---|
| `Shuffle Idx`, `operand` in or out of `0..n` | lane `operand mod n` (Euclidean: `n` reads lane 0, `-1` reads lane `n - 1`) | **Design** (portable defined semantics). PTX ISA 9.5 section 9.7.10.6 `shfl.sync.idx` takes `bval[4:0]`, the same as `operand mod n` only when `n` is 32 |
| `Up`, `Down`, `Xor` with the source outside `0..n` | the lane reads its **own** value | **Inherited from CUDA, cited:** PTX ISA 9.5 section 9.7.10.6 `shfl.sync`: "If the computed source lane index j is in range, the thread will copy the input operand a from lane j into its own destination register d; otherwise, the thread will simply copy its own input a to destination d", and `if (!pval) j = lane` |
| `delta == 0` (`Up`, `Down`), `mask == 0` (`Xor`) | every lane reads itself | follows from the two rules above |
| `delta >= n` (`Up`, `Down`), `mask >= n` (`Xor`, no lane's `i ^ mask` is in range) | every lane reads itself | same |
| `Ballot` | a `U32`, bit `i` the lane `i` predicate; **more than 32 lanes is a typed trap** | **Design** (the result type is `U32`); PTX ISA 9.5 `vote.sync.ballot.b32` is a `.b32` mask of 32 lanes |

**Scope of the claim.** The `Up`/`Down`/`Xor` rule is CUDA's for a clamp equal to the scope's size; PTX's `c` operand can also split a warp into segments, which the IR does not model. The `Idx` rule is the IR's own for any `n` (a subgroup size other than 32 has no PTX `shfl` of that width).

**Pinned:** `tests/kernel_collective_edges.rs` runs 8 lanes (values `lane + 1`) at `delta` 0, `n - 1`, `n`, `n + 1` for `Up` and `Down`, masks 0, 1, `n - 1`, `n`, `n + 1` for `Xor`, `Idx` operands `n`, `n + 1` and `-1`, and a ballot at 32 lanes (defined) and 33 (a typed trap), each with the hand-written expected array.

— lane: unpopped
