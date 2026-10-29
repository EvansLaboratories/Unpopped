# U3 spec: execution scopes, memory spaces, resources, `ProgramModel`, and the Frontend/Backend seam they force

Status: **spec only, no code, no trait edit.** Closes ROADMAP U3 / gaps 3.2, 3.3, 3.18 of
`ir-union-design.md` (section 5, increment I3) once built. Written before any trait is touched, as the
PM required (2026-10-28): *"No trait edit before the U3 spec merges."* CireSnave's standing hold applies
until then (no new `Frontend`/`Backend` trait items, no schedule families, no new emitter features in
`unpopped-cpu-c` / `unpopped-slang` beyond bug fixes).

**Why this spec exists in this order.** CireSnave 2026-10-28, as quoted in Fuel#514
(`docs/SLANG-SEAM-REQUIREMENTS.md`): *"In addition to Baracuda's CUDA parser and emitter, Fuel will be
contributing a Slang parser and emitter to Unpopped. [...] I ask because having Unpopped and Baracuda
decide that seam could make the seam very CUDA-centric."* The seam was drawn against CUDA first. This
document is that seam tested against Slang's needs, on the evidence Fuel measured, before the shapes
freeze.

Inputs read for it: Fuel#514 (147 `.slang` kernels, regex counts, Fuel `origin/main` `2eed77789`);
Fuel's item-by-item review of the five positions below (Fuel lane, 2026-10-28, quoted in section 3); the
kernel layer spec `ir-kernel-layer.md` (landed in 0.31.0 and 0.31.3, with the Reduction lift
following: types, `check_kernel`, interpreter, Elementwise and Reduction lifts) and its section 8 (which U2 choices constrain this one);
`backend.rs` and `convert.rs` at `origin/main`.

## 1. What the seam assumes today, and which of it is CUDA-shaped

Facts, read from `origin/main` (not asserted from memory):

| Item | Today | CUDA-shaped? | Evidence |
|---|---|---|---|
| Artifact | `GeneratedKernel { name: String, source: String, provenance }`, `#[non_exhaustive]` | **Yes.** A CUDA kernel is source text compiled by NVRTC; a SPIR-V word stream and a binding manifest "can't ride a `String`" (the type's own doc). | `backend.rs:96-112` |
| Argument ABI | Not stated anywhere. The artifact carries no binding or parameter description. | **Yes, by omission.** CUDA takes positional pointers plus by-value scalars; Slang/Vulkan needs descriptor indices, a set, storage-vs-uniform, a `Params` block layout (Fuel#514 R1). | Fuel#514 sec. 3 R1 |
| Capability query | `Backend::supports_dtype(dtype, target) -> DtypeSupport` is the only one. | Partly. dtype is the one capability CUDA varied by arch; Vulkan varies by subgroup size, 16-bit storage, f64, cooperative matrix, denorm mode (R4-R6). | `backend.rs:677` |
| Frontend buffer naming | `Frontend { out_name, in_prefix }`: `out`/`in{K}` (CUDA) or `output`/`input{K}` (Slang) | **Yes.** Declared by name convention, not by reading declarations; 0 of 147 Fuel kernels lift on it (4 of 147 after a naming fix). | `convert.rs`, ROADMAP E13 |
| Frontend kernel entry | `kernel_marker: &str`, a substring (`__global__`, `numthreads`) | Yes. Slang has entry attributes and two-entry files (R3). | `convert.rs` |
| Frontend refusal | `residue: &[&str]`, a substring list that refuses `groupshared`, `Interlocked*` | Yes in effect: the union design (D1) replaces refusal by a kernel-layer body, after U4/U5. | `ir-union-design.md` section 4 |
| Frontend grammar | `parse: fn(&str) -> Option<tree_sitter::Tree>` | No language, but it makes tree-sitter 0.26 part of the public type. | `convert.rs` |
| Schedule | cpu-c and Slang accept only `Schedule::Scalar` (ROADMAP E1) | Neutral, but a measurement of how little of the seam is exercised off CUDA. | ROADMAP E1 |

Nothing in this list is changed by this document.

## 2. The U3 data model (proposed; every shape below is open to the PM's ruling)

It extends the kernel layer, it does not replace it. `ir-kernel-layer.md` section 8 listed what U2 fixed;
this section builds on those five choices and says where it would have to break them.

**2.1 Execution scopes and ids.** A `Scope` enum (`Thread`, `Subgroup`, `Block`, `Cluster`, `Device`,
`Peer`) and id leaves (`ThreadId(axis)`, `BlockId(axis)`, `LaneId`, `SubgroupId`, ...) as new
`ScalarExpr` leaves (breaking, one increment). A `LaunchShape` (block dims, grid dims, cluster dims) is a
property of the **kernel**, not of any statement. Subgroup size is **not** a constant: it is a range
(2.5). *Constrained by U2:* a v0 `KernelFn` is one logical thread; ids are new leaves around that model,
so the interpreter gains a "run it for every thread" driver rather than a rewrite.

**2.2 Memory spaces and allocation.** A `MemorySpace` (`Global`, `Shared`, `Local`/private, `Constant`,
`Texture`, later `Tensor`, `DistributedShared`) on a `LocalDecl` (`#[non_exhaustive]` in v0 for exactly
this) and on pointer-typed places. `Alloc { space, ty, extent }` as a declaration statement. Shared
allocations may be static (compile-time extent) or dynamic (launch argument). *Constrained by U2:*
`Place::Logical` keeps meaning "through the operand's view"; U3 adds `Place::Ptr { base, offset }`
variants beside it. If U3 ever wants `Ptr` to replace `Logical`, the lifting differential (U2 section 7)
breaks; this spec says it will not.

**2.3 Resources.** `Resource` is the thing a binding points at: `Buffer { elem, packing, access }`,
`UniformBlock { layout }`, `Texture`, `Sampler`, `Struct`. Distinct from a `Param`: a `Param` is the
kernel function's formal argument; a `Resource` is what the *target* binds it to. Fuel's corpus: 147 of
147 kernels end with a `ConstantBuffer<Params>` at the highest binding; bf16/f16 are stored two per `u32`
in 89 files (R6), so **packing is a property of the resource, not of the arithmetic dtype**. This is the
section that keeps "storage type is not compute type" a first-class fact (the same one `supports_dtype`
already encodes for dtypes).

**2.4 `ProgramModel`.** `ThreadSpmd` (CUDA, Slang, Mojo thread code, CuTe) or `BlockSpmd` (Triton). A
kernel declares one. Lowering `BlockSpmd` to `ThreadSpmd` is a pass inside Unpopped, not each emitter's
problem (decision D6 in `ir-union-design.md`). *Constrained by U2:* if `BlockSpmd` is primary, `LocalDecl`
gains a tile type (the U2 equality rule survives).

**2.5 Capabilities.** A `CapabilitySet` per target, **not** a `bool` per question. Values are typed and
tagged with their provenance (`Reported` by the driver, `Measured` by a probe, `Declared` by a vocabulary).
Subgroup size is a **range** `(min, max, fixed)`, not a number (Fuel's change). Fuel's own evidence for the
tag: the RTX 4070 *reports* neither FTZ nor preserve for f32 yet preserves when asked (`unpopped-slang/src/lib.rs:37-41`: with
`-denorm-mode-fp32 preserve` the same run had zero mismatches on both devices), so a reported bit can be
wrong.

## 3. The five positions, with Fuel's review recorded verbatim in substance

The PM accepted these as the spec's starting position (`[ASK-ANSWER] slang-seam-5-questions`,
2026-10-28) on two conditions: (1) Fuel reviews and says yes/no per item in the PR; (2) the Vulkane-owned
vocabulary part is marked UNVERIFIED. Fuel's reply (Fuel lane 1, 2026-10-28) is quoted in substance; the
last column is what this spec now says.

| # | Position sent | Fuel's reply | Recorded here |
|---|---|---|---|
| 1 | `GeneratedKernel` gets an artifact enum (`Source(String)` \| `SpirV(Vec<u32>)`) and an optional manifest (bindings, `Params` layout, entry names, workgroup size, required features). | **YES + 3 changes.** (a) The manifest is **per entry** inside a module: workgroup size and bindings are per entry; Fuel's `compile.sh` emits one `.spv` per entry and 4 files have two entries, so allow both shapes. (b) `Params` layout must record the **layout rule** (std140 vs scalar), not only offsets: 4 kernels have fixed arrays in `Params` and std140 pads array elements to 16 bytes; **Fuel has not checked which rule its host struct assumes for those 4.** (c) Add the SPIR-V version / target env and the `slangc` profile. `Vec<u32>` suffices. | Adopted with all three. See 4.1. The std140-vs-scalar question for the 4 kernels is **UNVERIFIED** and is Fuel's to measure. |
| 2 | The Slang Frontend takes a **declared resource map**; name inference stays as a fallback. | **YES**, conditional: the map covers `Params` **fields** too, and a kernel lifted by name inference is marked **inferred** in provenance so it can be counted and cannot pass as declared. Fuel can supply expected maps for the 147 as an oracle once the types exist. | Adopted. See 4.2. |
| 3 | A capability **set** type per target; `supports_dtype` becomes a convenience over it; values come from the namespace vocabulary. | **YES + 1 change**: subgroup size is a range; every value needs provenance **reported vs measured**. *Unverified by Fuel:* whether the `vulkan:` token set can carry either. | Adopted. See 2.5. Whether the `vulkan:` token set can carry a subgroup range or a measured tag is **UNVERIFIED**: the vocabulary is Vulkane's (`kiss-vulkan-vocab`), no Vulkane lane is running, and the PM answers for it until one exists. Nothing in this spec relies on the answer. |
| 4 | Numerics flags in three places: requirement in the plan, device fact in the capability set, compile flags in the artifact. | **YES to the three places + a fourth thing that is a result, not a flag**: a **measured conformance record per (device, driver, kernel)**. Fuel GAP-276: SPIR-V `FMax`/`FMin` canonicalize the +/-0 tie on the 4070 and two source-level rewrites were re-fused by the driver, so exactness cannot be derived from flags. Also: Fuel's `compile.sh` passes **no denorm flag**, so "no denorm mode declared" must read as **unknown**, never as preserve. | Adopted with the fourth item. See 4.3. |
| 5 | **Fuel** owns the layout dialect as data; Unpopped owns the resource-map and manifest **types**. | **YES.** Conditions: the dialect is **data** (one resource-map instance per layout family, `2s1u` / `3s1u`, plus the `Params` convention document), and Fuel's pipeline layouts are later **derived from the map**. That refactor is Fuel-side and starts after this spec merges (Fuel ROADMAP 101). | Adopted. Named owners: Unpopped (this lane) for the types; Fuel lane 1 for the dialect. |

## 4. The artifact and capability types (sketch, additive on `#[non_exhaustive]` types)

**4.1 Artifact and manifest.**

```
KernelArtifact::Source(String) | SpirV(Vec<u32>)
GeneratedKernel { name, artifact, manifest: Option<KernelManifest>, provenance }
KernelManifest  { toolchain: ToolchainRecord, entries: Vec<EntryManifest> }
EntryManifest   { name, workgroup_size: [u32; 3], bindings: Vec<Binding>, params: Option<ParamsLayout>,
                  required_features: Vec<FeatureToken> }
Binding         { set: u32, index: u32, kind: BindingKind /* Storage | Uniform | Texture | Sampler */,
                  role: Read | Write | ReadWrite, elem: ElementKind, packing: Packing }
ParamsLayout    { rule: LayoutRule /* Std140 | Std430 | Scalar */, fields: Vec<ParamField { name, offset, ty }> }
ToolchainRecord { spirv_version, target_env, profile: Option<String>, flags: Vec<String> }
```

Entry-per-module and entry-per-file are both valid manifests (Fuel change (a)). `LayoutRule` is a closed
enum with **no default**: a manifest that does not say is not a manifest. `GeneratedKernel::new(name,
source)` keeps working (artifact `Source`, no manifest), so no existing backend changes.

**4.2 Resource map as a Frontend input.** `ResourceMap { buffers: Vec<DeclaredBuffer>, params:
Vec<ParamField>, layout: LayoutRule }`, supplied by the caller; `lift_*` takes `Option<&ResourceMap>`.
`Lifted` records `Provenance::Declared | Inferred`; name inference sets `Inferred` and is reported as such
(Fuel condition on item 2). The map covers `Params` fields, not only buffers.

**4.3 Numerics and conformance.** The plan already carries the requirement (`Qualified`, the exactness
admission: "a flushing device already declines exact", `tests/a_flushing_device_declines_exact.rs`). The
device fact goes in `CapabilitySet` (denorm: `Unknown | Preserve | Flush`, **default `Unknown`**). Compile
flags go in `ToolchainRecord`. The measured conformance record is a fourth artifact:
`ConformanceRecord { device, driver, kernel_identity, property, outcome }`, produced by Unpopped's
harness and **consumed** (not derived) by exactness decisions. Its schema is out of scope here; its
existence is the point.

**4.4 Capability set.**

```
CapabilitySet { entries: Vec<Capability> }
Capability    { key: CapKey, value: CapValue, source: Reported | Measured | Declared }
CapValue      = Bool(bool) | Range { min: u32, max: u32, fixed: bool } | Enum(..) | Unknown
```

`Backend::supports_dtype` stays and becomes a default method that consults `capabilities(target)`
(a new defaulted method, additive).

## 5. What changes in the traits when this lands, and what each costs

| Change | Where | Breaking? |
|---|---|---|
| `GeneratedKernel` gains `artifact`, `manifest` | `backend.rs` | No (`#[non_exhaustive]` struct; `new` kept). |
| `Backend::capabilities(&self, target) -> CapabilitySet` | `backend.rs` | No (defaulted method returning "only dtypes known"). |
| `Frontend` gains a resource-map parameter on the `lift_*` entry points | `convert.rs` | **Yes for `Frontend` struct users** if a field is added to a non-`#[non_exhaustive]` struct; the union design (I2) already plans `Frontend` `#[non_exhaustive]` with the kernel-layer path. They should land in one increment. |
| Scope ids, `Alloc`, `Place::Ptr`, `ProgramModel` | `ir.rs`, `kernel.rs` | Yes (new `ScalarExpr` leaves; one pre-1.0 minor, as decision D8 ruled). |

None of this is started. The order after this spec merges: (1) the manifest and artifact types (additive,
no consumer needs more than `Source`); (2) the capability set; (3) the resource map and the `Frontend`
change together with I2's kernel-layer path; (4) scopes and memory spaces with the interpreter's
per-thread driver; (5) a Slang emitter handoff to Fuel (ROADMAP P3) against the declared types.

## 6. Unknowns, marked so they are not read as decisions

- **UNVERIFIED (Fuel):** which layout rule Fuel's host `Params` struct assumes for the 4 kernels with
  fixed arrays (std140 pads array elements to 16 bytes).
- **UNVERIFIED (Vulkane, no lane running; the PM answers):** whether the `vulkan:` token set can carry a
  subgroup range or a reported/measured tag. This spec does not depend on it: `CapabilitySet` can be
  filled from a probe without the vocabulary.
- **UNVERIFIED (Unpopped):** whether cooperative matrix (Fuel R10: 27 GLSL files, outside the 147) fits
  `CapabilitySet` + U7's MMA atoms or needs its own seam.
- **Open:** whether `Frontend` stays a descriptor struct or becomes a trait once it has a resource map.
  `Frontend` is a plain struct holding a `fn` pointer, not a trait, and the question whether it is a
  FireLine seam crate (the shared-interface-crates project) is for the PM and the FireLine lane: this
  document only notes that its `parse` field exposes `tree_sitter::Tree`, so a seam crate for it would
  pin tree-sitter's version.
- **Open:** `Stmt::Call` and the module type (`ir-kernel-layer.md` unknown U-2) are needed before any
  device-function call, and so before most real Slang kernels lift.
