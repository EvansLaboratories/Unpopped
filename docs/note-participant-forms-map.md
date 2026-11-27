# Note: which shipped statements let the ARRIVING thread name participants (map for the PM's barrier hold)

Written 2026-11-27 by the Unpopped lane at pm-kernels' task `map-shipped-mbar-to-participant-forms`. **Read-only: no shipped statement is edited.** Source: `crates/unpopped/src/kernel.rs` and `exec.rs` at `origin/main` e57ef66e (line numbers are that file's; they move). The hold being served (PM 2026-11-27, wrapper read of US10437593B2 and US10977037B2, "plurality of threads SPECIFIED BY the first thread"; findings not advice, not a clearance, no attorney): no new barrier statements or fields where the ARRIVING THREAD NAMES THE PARTICIPANTS (a mask operand, a thread COUNT on a named barrier) and no modelling of resumption at a sync point. PTX forms listed by the task are quoted from the task, not re-read by this lane in PTX ISA 9.5 for this note (**UNVERIFIED** against the ISA text).

## The table

| Shipped statement (kernel.rs) | Field | Who sets it | Names participants? | PTX form it spells |
|---|---|---|---|---|
| `Barrier { scope, kind: Full }` (l.563) | `scope` (Subgroup, Warpgroup, Block, Cluster, Device, System) | the program text, the same for every thread; "wait until every thread of `scope` reaches this statement" | **No**: the participant set is the whole scope, fixed by the scope | `bar.sync 0`, `barrier.cluster`, `bar.warp.sync`-class, grid sync |
| **`Barrier { kind: Named { id, count } }`** (l.1370) | `id` (0 to 15), **`count`** | each ARRIVING thread writes `count`; "threads that disagree on `count` are `KernelTrap::BarrierMismatch`" (l.1368) | **A thread COUNT on a named barrier: yes**, set by the arriving thread (every arriver must agree, so the set is "any `count` threads, whole warps") | `bar.sync id, count` |
| **`Barrier { kind: NamedArrive { id, count } }`** (l.1378) | `id`, **`count`** | the arriving (non-waiting) thread | **Same: a thread count**, set by the arriver | `bar.arrive id, count` |
| `MbarInit { bar, count }` (l.755) | `count` (arrival count, at least 1) | the INITIALISING thread | a count of expected arrivals, set once at init, not by an arriving thread | `mbarrier.init.b64 [addr], count` |
| `MbarArrive { bar, tx }` (l.770) | `tx` (bytes, optional) | the arriving thread | **No count operand** (one arrival consumed per statement) and **no cta mask**; `tx` is a byte count for the transaction count, not a participant set | `mbarrier.arrive{.expect_tx}` WITHOUT the `count` and `ctaMask` operands |
| `MbarCompleteTx { bar, bytes }` (l.782) | `bytes` | the thread (or, by hardware, the copy) completing | No: a byte count; no mask | `mbarrier.complete_tx` |
| `MbarWait { bar, parity }` (l.797) | `parity` | any waiting thread | No: "any subset of the block's threads may wait" | `mbarrier.try_wait.parity` |
| `Tcgen05Commit { bar }` (l.1010) | `bar` only | the issuing thread | **No `.multicast::cluster`, no `ctaMask`**; one local barrier; "a release-class arrival like `MbarArrive` without a transaction count" | `tcgen05.commit` WITHOUT `.multicast::cluster [mbar] {, ctaMask}` |
| `Tcgen05Alloc` / `Dealloc` / `RelinquishPermit` (l.987, 995, 1003) | `ncols` | the issuing warp | No participant set; **`cta_group::1` only**, "`.exclusive` and `cta_group::2` are not modelled" (l.986): the group is fixed in the statement text | `tcgen05.alloc.cta_group::1` etc. |
| `CopyAsync { dst, src, elems, valid, cache, l2_prefetch }` (l.1041) | `valid` (source elements read) | the issuing thread | No: a data count, not a participant set; completion by `AsyncWait` of the thread's own groups | `cp.async` |
| `CopyBulk { dst, src, elems, bar }` (l.1075) | `bar` | the issuing thread | No mask or multicast; completes on the one named local `mbarrier` ("the hardware does a `MbarCompleteTx` of those bytes") | `cp.async.bulk...mbarrier::complete_tx::bytes` WITHOUT `.multicast::cluster` |
| `Place::ClusterPeer { alloc, block, index }` (l.255) | `block` (rank in the cluster) | the accessing thread | **An address, not a participant set**: names ONE block's `Shared` array to read or write; no arrive, no synchronisation of its own ("ordered by a Cluster barrier or a release/acquire pair") | `mapa` |
| `Place::Multicast` (l.287) and `MulticastLoadReduce` (l.626) | none naming devices | the launch (`LaunchShape::devices`) | The device set is the launch's, not named by the accessing thread | `multimem` |
| `Collective Shuffle { mode, operand }` / `Vote` / `Reduce` / `Scan` (l.1391 on) | `operand` (a lane index, distance or xor mask) | each lane | **A lane SOURCE, not a participant set**: the scope is whole (Subgroup or Block, "full participation"); `Xor`'s "mask" is `lane ^ operand`, not a `membermask` | `shfl.sync` (the `membermask` operand is NOT carried), `vote.sync` |

## What this shows

1. **One shipped form falls under the new wording: `BarrierKind::Named` and `NamedArrive` carry `count`**, written by the arriving thread, with a mismatch trap (kernel.rs l.1366 to 1383; oracle `oracle/kernel_named.rs`, #460's phase-keyed release). Everything else the task lists is either absent from the shipped statements or fixed by the program text for every thread. The hold's own carve-out ("block-scope full-block named barriers, conformance row 88 as written, stay ALLOWED") describes a named barrier whose `count` is the whole block; whether a `count` FIELD on the shipped statement counts as "a thread COUNT on a named barrier" is the PM's reading, not this lane's.
2. **None of the PTX multicast/mask operands is shipped:** `tcgen05.commit .multicast::cluster {ctaMask}`, `mbarrier.arrive{.multicast} ... count {, ctaMask}` and `mbarrier.expect_tx{.multicast} ... {ctaMask}` have no field in `MbarArrive`, `Tcgen05Commit` or `CopyBulk`. `MbarArrive` has no arrive-count operand at all (the PTX `count` operand is not modelled).
3. **`MbarInit.count` is set by the initialising thread** (one statement, not the arrivers), the PTX `mbarrier.init` form.
4. **Not carried at all:** `syncwarp(mask)` and the `membermask` of `shfl.sync`/`vote.sync`; the `.cta_group::1|::2` qualifier is fixed in the statement text (`::1` only).

## Held, so named here

No change is proposed. `BarrierKind` additions (leaves' open ASK), any `syncwarp(mask)` row and any named-barrier-count row stay HELD until a wrapper read says otherwise. The shipped statements and the oracle are unchanged.

— lane: unpopped
