# Options: `*reinterpret_cast<const uint32_t*>(&smem_A[i][j])` (U38c, census wall: 5 Baracuda kernels)

Status: **options for pm-kernels' ruling; one option is small enough to take without a ruling.**

## The shape

`baracuda_fp8_rcr_sm89.cuh:183`: `__shared__ uint8_t smem_A[M][K]`, then `const uint32_t a0 = *reinterpret_cast<const uint32_t*>(&smem_A[frag_m0 + ar][ac]);` (four fp8 bytes read as one word for an MMA fragment). Decline today: "the address of an element of something that is not a pointer or array": `&a[i][j]` of a multi-dimensional array (U19 census, 897d4997-era slice supports the element, not its address).

## Options

- **A. `&a[i][j]` is a pointer into the flat array (recommended).** The multi-dimensional array is already one flat allocation read row-major, and a cast pointer into a shared array already addresses its bytes (`Place::AllocRaw`, `tests/kernel_body_qualified_calls.rs`). `&a[i][j]` becomes a pointer at flat element `i * J + j` (byte offset times the element size); the cast and the deref then lift as for a one-dimensional array. Additive, walker only, no IR change. Cost S.
- **B. Also accept the unaligned case.** The load assumes 4-byte alignment of `&smem_A[m][k]`; C leaves a misaligned one undefined. Option A lifts the read at the byte offset without checking; add an oracle check that traps a misaligned `u32` read of a byte array (a refinement: a trap where C is undefined). Cost S, needs the PM's ruling because it adds an oracle trap.
- **C. Decline.**

## Recommendation

A now (I will take it on lane/u19-walls-2 without waiting: additive, walker-only); B only if a kernel is found whose offset is misaligned.
