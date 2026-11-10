# Options note: peer and multicast scopes, and multimem ops (ROADMAP U12, issue #123)

Status: **proposal, for the PM.** Written 2026-11-10 against `origin/main` 8ad1f43. PM [TASK] 2026-11-10: U12 options note first (docs only), then build the option the PM rules. U12 is the last row of the IR-union programme (`docs/ir-union-design.md` row 3.21 / I12, "Multi-GPU: peer pointers, multicast, `multimem` ld_reduce/red/st, peer TMA"); its dependencies I4 (sync and atomics) and D4 (multi-GPU in scope, last) are met.

## What is needed

Seven kernel languages reach memory that is not this block's, this device's, or this thread's. The master list (`docs/ir-union-master-capabilities.md`, H5) marks Unpopped ❌ and the evidence rows say what each language has:

| Capability | Where it exists (evidence in `docs/ir-union-evidence/*-wave2c.md`) |
|---|---|
| **Cluster-peer shared memory** (another block of the cluster's `Shared` array, by `mapa`) | PTX `mapa` (9.7.10.26); CuTe `cluster_sm90.hpp:170`; ThunderKittens `mapa.shared::cluster`; Triton Gluon tests |
| **Peer device pointers** (a buffer on another GPU, reached over NVLink or PCIe peer access) | ThunderKittens `pgl`; CUTLASS experimental `PeerDeviceMapping`; Triton `_import_peer_ptrs` (plain `int64` pointers from torch symmetric memory); Mojo host `enable_peer_access` |
| **`multimem`**: one multicast address that stands for the same buffer on every device of an NVLink domain; `ld_reduce` reads the reduction over all devices, `st` writes all, `red` reduces into all | PTX 9.7.10.15; ThunderKittens `common/multimem.cuh`; CuTe DSL `multimem_ld_reduce`, `multimem_st`, `multimem_red_add1`; Mojo inline PTX in MAX comm kernels |
| **TMA multicast** (one tile load delivered to several blocks of a cluster) | PTX `cp.async.bulk.tensor ... multicast::cluster`; CUTLASS and CuTe `SM90_TMA_LOAD_MULTICAST` (this one belongs to U8c, TMA, and is not part of this note) |

Not first-class in the other languages: Slang has none (0 hits, `slang-wave2c.md`); Triton has peer pointers only as host-supplied integers and no `multimem`; CUDA's device runtime says "There is no multi-GPU support" (the host drives it).

## What the code does not do today (measured at `origin/main` 8ad1f43)

- **One address space per launch.** `run_launch` and `run_launch_module` take one argument list: one device's buffers. `LaunchShape` has `block`, `grid`, `subgroup` and (since 0.42.18) `cluster`; there is no device count. `ExecId` has no device rank.
- **No pointer values** (decision D-R4-3: arrays by id, `Place::Ptr` held with U9c). A place names a buffer or an allocation, never a pointer, so a peer pointer cannot be passed in; a peer place has to name the peer by rank.
- **`Scope` stops at `Device`** (`Thread`, `Subgroup`, `Block`, `Cluster`, `Device`). Cluster atomics and fences are typed refusals today (`Cluster` scope on a barrier works since 0.42.18). `multimem` orders at the `.sys` scope, above `Device`.
- **Another block's `Shared` array is unreachable**: `Place::Alloc` names this block's allocation. A grep over `crates/unpopped/src` for `mapa`, `Dsmem`, `Remote` and `peer` finds nothing (control: `cluster` finds `exec.rs` `with_cluster` and the checker's cluster rules).
- `Scope`, `ExecId`, `Place` and `ParamKind` are `#[non_exhaustive]`, so a new variant of any of them is additive; **`Stmt` is exhaustive**, so a new `Stmt` variant is a break stamped as the second number.

## What the oracle would have to model

A **device** is a full copy of the launch's buffers plus its own grid. The oracle runs the grid of every device under one scheduler, so the race detector and the release/acquire happens-before machinery (`oracle/kernel_sync.rs`) see accesses across devices. A cross-device access is legal only when ordered by a `System`-scope release/acquire pair (an atomic or a fence); otherwise it is a data race and the oracle says so, as it does between blocks. `multimem` has three semantics:

- `st(addr, v)`: every device's element becomes `v`.
- `red.add(addr, v)`: every device's element becomes `element + v`.
- `ld_reduce.op(addr)`: the result is `op` over the element on every device. Float `add` has unspecified order (the same unspecified order, so the same declared band, as a float atomic: the `AtomicRmw` doc comment, section 5a); 16-bit `add` accumulates in `f32` (`acc::f32` in the PTX ThunderKittens emits) and rounds once; `min`, `max`, integer ops are exact.

ThunderKittens' wrappers (`multimem.cuh`, commit 67845f5f, read for this note) bound what the hardware path supports: `s32`/`u32` with `add`/`min`/`max`; `f32` `add` only; `f16`/`bf16` `add` (accumulating in `f32`), `min`, `max` for `ld_reduce` and `add` only for `red`; two-wide vectors of the float types; memory order `weak`, `relaxed`, `acquire` for `ld_reduce`, `weak`, `relaxed`, `release` for `st`, and always `release` at `sys` scope for `red`.

## Options

| | Peer and multicast reached by | New in the IR | Breaks |
|---|---|---|---|
| **A** | new **places**: `Place::ClusterPeer { alloc, block, index }` (another block's `Shared`), `Place::Peer { buffer, device, coords }` (a buffer on another device), `Place::Multicast { buffer, coords }` (the buffer on all devices); the existing `Load`, `Store`, `AtomicRmw`, `AtomicCas` and `Fence` act on them; `Scope::System`; `ExecId::DeviceRank` and `DeviceCount`; `LaunchShape::devices`; **one new `Stmt`**: `MulticastLoadReduce { local, place, op, order }`, because a `Load` carries no combine | places and enum variants (additive) plus one `Stmt` variant | one break, in the last slice |
| **B** | new **statements** for each form: `PeerLoad`, `PeerStore`, `PeerAtomic`, `MultimemLdReduce`, `MultimemSt`, `MultimemRed` | five or six `Stmt` variants | a break at once, and every consumer match grows six arms |
| **C** | defer peers until `Place::Ptr` (U9c) exists and only add `Scope::System` and the cluster-peer place now | places only | none now, a second design later; multimem and peer buffers wait on a consumer |

## Recommendation: A, in four slices

A keeps what a peer access *is* (a load, a store, an atomic on an element of someone else's memory) in the statements the checker, oracle, uniformity analysis and emitter already know, and puts the one thing that is genuinely new (a read that reduces across devices) in one named statement. B repeats the atomic machinery six times. C leaves the row open for no consumer-driven reason: CireSnave's directive is that the IR covers every common kernel language ("can't express" is backlog), and ThunderKittens' PGL plus the CuTe and Mojo multimem helpers are the demand evidence.

1. **S1 cluster-peer shared memory** (additive: `Place::ClusterPeer`, plus the `Cluster` scope on atomics and fences, which are typed refusals today). A block reads or writes the `Shared` array of another block of its cluster; the rank is an integer expression; legal only in a launch with cluster dimensions; ordered by a `Cluster` barrier (exists since 0.42.18) or a cluster-scope release/acquire. Oracle and cpu-c (the serial C already runs the blocks of a cluster in one process). Fixture: a cluster-wide reduction where each block publishes a partial into its own `Shared` and block 0 reads its cluster peers' after a `Cluster` barrier.
2. **S2 devices and peer buffers** (additive): `Scope::System`, `ExecId::DeviceRank`/`DeviceCount`, `LaunchShape::devices`, `Place::Peer`, an oracle `run_launch_devices` that runs all devices' grids under one scheduler with `System`-scope release/acquire edges and race detection across devices, cpu-c reference as one C function per device over a table of buffer pointers (a harness, not a real NVLink). Fixtures: an all-gather (each device writes its slice to every peer) and a ring all-reduce with a flag, hand-worked expectations.
3. **S3 multicast stores and reductions** (additive): `Place::Multicast`, with `Store` (`multimem.st`) and `AtomicRmw` (`multimem.red`) acting on it; the checker admits the dtype and op table above and refuses the rest with a typed error. Fixture: all-reduce by `red.add` into a multicast buffer, integer exact and float inside a band.
4. **S4 `MulticastLoadReduce`** (**BREAKING**: one new `Stmt`). `ld_reduce` with an op and an order, the 16-bit `f32` accumulation rule, and the second all-reduce form (`ld_reduce` then `st`). The second-number bump, listed with the new variant in the fragment, as for D4b.

Each slice carries the usual bar: red-first tests, hand-worked expected values, a Rust reference, cpu-c against the oracle, seeded mutants (a dropped system-scope edge must make the oracle report the race; a swapped op or order must change the bytes), and the five-repo grep for external `Stmt` matches before slice 4 merges.

## Out of scope here

- **PTX or CUDA emission** of any of this (`multimem`, `mapa` and peer pointers are a Baracuda emitter's job; Unpopped owns the IR, the checker and the oracle).
- **NVSwitch topology, NVLS setup, `cuMulticast*` host calls and symmetric-memory allocation** (host and runtime; the survey calls multi-GPU peer access "out of kernel IR; plan-level only", and the buffer binding of a launch stays an argument).
- **TMA multicast** (U8c) and **TMEM** (S5 of U7).
- A collective-communication library (NCCL-style ring and tree algorithms); the fixtures are kernels, not a library.

## Needs a ruling

1. **A, B or C** (recommended: A).
2. **Is a launch-level device count the right unit** (`LaunchShape::devices`), or should a multi-device program be several `Module` launches the harness steps together? (recommended: one launch with a device count, because the cross-device race check needs one schedule.)
3. **Stop at S3 for now** and schedule S4 (the one break) when a consumer asks for `ld_reduce`, or build S4 straight after S3? (recommended: build it; it is the whole of `multimem`'s read side and the grep for external matches costs one pass.)
4. **Float `multimem` reduction order**: declare it unspecified inside a band like a float atomic (recommended), or define a device-rank order the hardware does not promise.

## Sources

PTX ISA 9.5 sections named in `docs/ir-union-evidence/cuda-wave2c.md` (H5: 9.7.10.15 `multimem`, 8.2.3 Multimem Addresses, 9.7.10.26 `mapa`); the other languages' rows in the same files; ThunderKittens `include/common/multimem.cuh` and `include/types/system/pgl.cuh` at commit 67845f5f (the asm strings and type table above were read from `multimem.cuh` for this note). The PTX pages themselves were not re-read for this note; the dtype and ordering limits above are ThunderKittens' wrappers, which are a subset of what PTX allows.
