# Options: constant tables (`__constant__ T t[N] = {...}`, `static const T t[N] = {...}`) (U40, from vulkane's 8k SPIR-V slice)

Status: **options for pm-kernels' ruling; no code before the ruling.** Measured at lane/u19-walls-2 (`5c0df93c`), Baracuda corpus.

## The gap

`AllocDecl { name, space, elem, extent }` (`crates/unpopped/src/exec.rs:283`, `#[non_exhaustive]`) has no initialiser, so a table has no
IR form: `MemorySpace::Constant` exists (exec.rs:263) but an allocation in it can only be filled by stores, and a `Constant` array
cannot be written by a kernel. Vulkane's SPIR-V emitter therefore declines a `Constant` alloc by name. The C-family walker today
declines a file-scope `__constant__`/`static const` table by name too (`a declaration with a storage class other than __shared__`).

## Census demand (with controls)

Baracuda corpus (564 files, `--args --includes`): **0 kernels are blocked by a constant table.** `grep -rn '__constant__'` over the
`.cu`/`.cuh` files finds **1** line, a comment (`bitsandbytes/src/nf4_kernel.cuh:50`: "kept as a switch instead of a __constant__
array"); `kvalues` finds 0 (so the iq4 tables the PM mentioned are not in this corpus); `static const|constexpr T t[N] = {` finds 1
file, `ondevice/write_slice_doff_validate.cu:104` (a validation harness, not a kernel). Positive controls: the same greps with
`constexpr int` find 751 lines, and the one static-const file is found by the same pattern. So the demand is **vulkane's own kernels**,
not Baracuda's census.

## Options

- **A. An additive optional `init` on `AllocDecl` (recommended).** `init: Option<Vec<i128>>`-like literal list (integers, and floats
  by bit pattern; one value per element, length == extent), `#[serde(default, skip_serializing_if = "Option::is_none")]` so an
  init-less kernel serialises byte for byte as before (artifact keys and goldens unchanged). `AllocDecl::new` keeps its signature;
  `AllocDecl::with_init(values)` sets it. Checker: only `Constant` and `Private` allocations may carry one (a `Shared` one declines:
  no initialiser exists for it in any target), length == extent, each value representable in `elem`, and a `Store` into a `Constant`
  allocation is refused. Oracle: the allocation's memory is initialised from `init` at launch (a `Constant` one is read-only). Walker
  (after the ruling): a file-scope `__constant__`/`static const` array with an all-literal initialiser lifts to an initialised
  allocation; `t[i]` is an ordinary `Place::Alloc` load. **Emitters:** each must spell it or decline by name (the default is to
  decline: an emitter that ignores `init` would emit a zeroed table, so the checker/`Backend` gate must refuse an initialised
  allocation unless the backend opts in, like `total_integer_ops`). CUDA family: `__constant__ T t[N] = {...};` at file scope; cpu-c:
  `static const T t[N] = {...};`; SPIR-V (vulkane): `OpConstantComposite` array plus a `Private` or `UniformConstant` variable with an
  initialiser; Slang: `static const T t[N] = {...};`. Cost M: IR field + checker + oracle + a plan/backend gate in unpopped, then one
  small row per emitter.
- **B. A table as a read-only buffer parameter the caller binds.** No IR change: the walker lifts the table to a `Param::Buffer`
  (`Read`) named after it, and the caller passes the data at launch. Works on every emitter today, but changes the kernel's
  interface (every launch must bind it; a generated kernel's `GeneratedKernel::interface` grows a buffer), and the data lives outside
  the kernel, so the oracle test must supply it. Cost S; a fine interim for a caller that owns the launch, wrong for a kernel that is
  meant to be self-contained (vulkane's SPIR-V case).
- **C. Decline (status quo).** Cost 0; the kernels that need a table stay out of the lifted set.

## Recommendation

A, with the backend opt-in gate (no emitter silently drops an initialiser), staged as U40-1 (IR, checker, oracle, serde), U40-2 (walker),
then one row per emitter; vulkane first since it asked. Lanes: **vulkane** spells it (the requester), **baracuda-cuda-emit**,
**fishstringer-slang-emit** and **cpu-c** each spell or decline by name; **rockfish/openclownfish/bonefish** decline by name. B stays
available as an interim for a caller that binds its own buffers. No exhaustive-enum widening (`AllocDecl` is a `#[non_exhaustive]`
struct; one optional field).

## Shape decided and built (2026-11-28, `unpopped-leaves`, at pm-kernels' [TASK] `next-rows`: U66 and U40 are one row)

Option A, as recommended. `AllocDecl::init: Option<Vec<i128>>`, `#[serde(default, skip_serializing_if = "Option::is_none")]`, set by `AllocDecl::with_init`. **One value per element, by logical index; an
integer element takes its value, a float element its bit pattern as a non-negative integer** (so the field is `Eq`, a negative zero and a NaN payload survive, and no float parsing is involved).
`check_kernel` (`KernelError::AllocInit`, non_exhaustive): only `Constant` or `Private` carry one (a `Shared` one has no initializer in any target), a `Static` extent, length equal to the extent, each value in the element's
range (an integer in `int_range`, a float's bits below `2^(8*width)`), no layout beside it (undefined together until a row asks), and a store into a `Constant` stays refused (`WriteToConstant`). Oracle: a `Constant` table is one read-only block-level
instance with every element initialized; a `Private` table is every thread's own copy; a `Constant` allocation **without** `init` is still unreadable (`Unsupported`). `unpopped-cpu-c` spells both (a `static const` array of the element's
width, copied into the block's storage or into every thread's slice) and declines by name in a callee, in a kernel with a Device barrier, a peer access or several devices, and a `Constant` allocation with no values: never a zeroed table. The other emitters keep their own
decline until each opts in. The walker half (U40-2, lifting `__constant__`/`static const` literal arrays) is **not** built here. Additive: `AllocDecl` and `KernelError` are `non_exhaustive`.
