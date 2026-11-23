# Options: `vec_t<float, 128>` (U38b, census wall: 8 Baracuda kernels, FlashInfer cascade/merge)

Status: **options for pm-kernels' ruling; no code.** Read at flashinfer `vec_dtypes.cuh:2092-2160`.

## The shape

    template <size_t vec_size> struct vec_t<float, vec_size> { float4 data[vec_size / 4];
      float& operator[](size_t i) { return ((float*)(data))[i]; }
      void fill(float v) { for (i < vec_size / 4) data[i] = make_float4(v, v, v, v); }
      void load(const float* p) { for (..) data[i] = ((float4*)p)[i]; }   void store(float* p) const {..}
      template <typename T> void cast_load(const T* p) { cast_load_impl(*this, p); } ... };

Needed: (1) a partial specialisation of a class template with a value parameter, picked by `vec_t<float, 128>`; (2) a struct field that is an array of `float4`; (3) `((float*)data)[i]`, a scalar view of a vector array; (4) `((float4*)p)[i]` through a global pointer; (5) `make_float4`; (6) `cast_load`/`cast_store` through `vec_cast<>` templates (`cast_from_impl`, `if constexpr`).

## Options

- **A. Scalarize (recommended).** A struct whose member is a vector array becomes a flat array of `lanes x N` scalars (the walker already scalarizes struct array fields, U23b, and vector locals of 2 and 4 lanes); `(float*)data` is then the same flat array (layout-identical), `(float4*)p` a 4-scalar load group at element `4*i`. `make_float4` is already a vector constructor. Additive, walker only, no IR change. Cost L: five prerequisites, each with an oracle test (partial-specialisation matching by value, array-of-vector fields, scalar view of a vector array, grouped pointer loads, `cast_load` for `float` to `float` and `half` to `float`). Sub-rows U38b-1..5 so each lands with its mutants.
- **B. Native vector arrays in the IR.** Lane-indexed runtime access to a `VecAlloc` of N elements. Touches `Place`/checker/emitters (every emitter must spell it). Cost XL, breaking for out-of-crate matchers unless `Place` is non-exhaustive (it is). Not recommended: A covers the kernels without making emitters learn a new shape.
- **C. Decline;** FlashInfer kernels instantiated at `vec_size <= 8` hit the `vec_t<float, 4>` specialisations (`float4 data` scalar) which need only (2)-(5); take those first if only the small sizes matter.

## Recommendation

A, in order U38b-1 (value-parameter partial specialisation) first: it is the only prerequisite that stands alone and it also unblocks `TileShape<T>`-style traits already in the PR. No lane change for A.
