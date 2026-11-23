/*
 * HAND-FLATTENED COPY of one FlashInfer configuration, for Unpopped U19 slice 5
 * (plain `__global__`, no templates, no macros that expand to templates).
 *
 * Upstream: FlashInfer, https://github.com/flashinfer-ai/flashinfer
 *   commit eee0d75f91f64c520bfaed07e39a850ea4ddde23 (nightly-v0.6.12-20260528-3), as vendored
 *   at crates/baracuda-kernels-sys/vendor/flashinfer (see its VENDOR.md, LICENSE and NOTICE).
 * Derived from these upstream files (vendored copies):
 *   include/flashinfer/attention/decode.cuh  (BatchDecodeWithPagedKVCacheKernel, ...Device,
 *       compute_qk, update_local_state, sync_state, BatchDecodeWithPagedKVCacheDispatched)
 *   include/flashinfer/attention/state.cuh, variants.cuh, variant_helper.cuh,
 *       default_decode_params.cuh, page.cuh, fastdiv.cuh, cp_async.cuh, math.cuh, vec_dtypes.cuh
 *
 * Copyright (c) 2023 by FlashInfer team.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * WHAT WAS CHANGED (flattening only; arithmetic and memory-access order are meant to be
 * unchanged, and were checked bitwise against the original, see the harness notes):
 *   - Instantiated, for exactly ONE configuration: DTypeQ = DTypeKV = DTypeO = half,
 *     IdType = int32_t, HEAD_DIM = 128, PosEncodingMode::kNone, GROUP_SIZE = 1
 *     (num_qo_heads == num_kv_heads), DefaultAttention<false,false,false,false>,
 *     NUM_STAGES_SMEM = 2 (the compute_capacity.first >= 8 arm).
 *   - Derived constants, from BatchDecodeWithPagedKVCacheDispatched (decode.cuh:763-773):
 *       vec_size = max(16/sizeof(half), 128/32) = 8, bdx = 128/8 = 16, bdy = 1,
 *       num_threads = max(128, 16*1) = 128, bdz = 128/(16*1) = 8, tile_size_per_bdx = 4.
 *   - Templates replaced by those constants; vec_t<float,8>/vec_t<half,8>/state_t<8>/
 *     paged_kv_t/BatchDecodeParams/DefaultAttention became plain structs or arrays with the
 *     same member order; vec_cast<float,half>/<half,float> and cp_async::pred_load<128,...>
 *     became plain functions with the same bodies.
 *   - Dead branches for this configuration removed: the kRoPELlama arm (and freq, which is
 *     only read there), LogitsTransform (identity for the all-false variant), LogitsMask
 *     (always true), the unused DefaultAttention members (qo_len, kv_len, window_left,
 *     soft_cap_pre_tanh_scale, custom_mask_ptr), v_scale (Params has none, so 1.0f), the
 *     block_valid_mask test is KEPT. The partition_kv flag is KEPT (runtime).
 *   - cg::this_thread_block().sync() written as __syncthreads() (identical for a thread block).
 *   - U19 slice-5 param flatten (mechanical, no logic edit): the kernel no longer takes the
 *     `DecodeParamsHalf` struct. It takes 16 buffer (pointer) params plus ONE scalar-only
 *     `DecodeScalars` struct; `compute_qk` lost its (unread) `params` arg; `partition_kv` is a
 *     uint32 in the scalar struct. The local `paged_kv` is rebuilt from the params.
 *   - The `#if __CUDACC_VER_MAJOR__ >= 12 ... griddepcontrol` blocks and the
 *     FLASHINFER_CP_ASYNC_ENABLED fallbacks are kept verbatim.
 *
 * WHAT THIS IS NOT: a general instantiator. It covers ONE decode configuration (above), not
 * other head dims, dtypes (bf16/fp8), group sizes, RoPE, sliding window, soft-cap, ALiBi,
 * custom masks, the MLA kernel, or the single-request kernel. Do not extend it by hand.
 */
#include <cstddef>
#include <cstdint>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace flat_decode {

// ---- constants from the Dispatched function (HEAD_DIM = 128, half, GROUP_SIZE = 1) ----
constexpr uint32_t HEAD_DIM = 128;
constexpr uint32_t VEC_SIZE = 8;
constexpr uint32_t BDX = 16;
constexpr uint32_t BDY = 1;
constexpr uint32_t BDZ = 8;
constexpr uint32_t TILE_SIZE_PER_BDX = 4;
constexpr uint32_t NUM_STAGES_SMEM = 2;
constexpr uint32_t VEC_BITS = sizeof(half) * VEC_SIZE * 8;  // 128
// launch: grid (padded_batch_size, num_kv_heads), block (BDX, BDY, BDZ) = (16, 1, 8),
// dynamic smem = 2*2*4*1*8*128*sizeof(half) + max(4*128*sizeof(half*), 2*1*8*sizeof(float))
constexpr uint32_t SMEM_BYTES = 2 * NUM_STAGES_SMEM * TILE_SIZE_PER_BDX * BDY * BDZ * HEAD_DIM *
                                    sizeof(half) +
                                (TILE_SIZE_PER_BDX * 128 * sizeof(half*) > 2 * BDY * BDZ * sizeof(float)
                                     ? TILE_SIZE_PER_BDX * 128 * sizeof(half*)
                                     : 2 * BDY * BDZ * sizeof(float));

constexpr float LOG2E = 1.44269504088896340736f;  // math.cuh
constexpr float INF = 5e4;                         // math.cuh (sic: upstream uses 5e4)

// ---- fastdiv.cuh (vendored copy), plain struct ----
struct uint_fastdiv {
  uint32_t d_, m_, l_;
  __host__ __device__ __forceinline__ void divmod(uint32_t n, uint32_t& q, uint32_t& r) const {
    if (d_ <= 1) {
      q = n;
      r = 0;
      return;
    }
    uint32_t hi = uint32_t((uint64_t(m_) * uint64_t(n)) >> 32);
    q = (hi + ((n - hi) >> 1)) >> (l_ - 1);
    r = n - q * d_;
  }
  __host__ __device__ __forceinline__ operator unsigned int() const { return d_; }
};

// ---- page.cuh paged_kv_t<half, int32_t>, same member order ----
struct PagedKvHalf {
  uint_fastdiv page_size;
  uint32_t num_heads;
  uint32_t head_dim;
  uint32_t batch_size;
  uint32_t stride_page;
  uint32_t stride_n;
  uint32_t stride_h;
  half* k_data;
  half* v_data;
  int32_t* indices;
  int32_t* indptr;
  int32_t* last_page_len;
  int32_t* rope_pos_offset;
};

__device__ __forceinline__ uint32_t paged_get_length(const PagedKvHalf& kv, uint32_t batch_idx) {
  if (kv.indptr[batch_idx + 1] == kv.indptr[batch_idx]) {
    return 0;
  }
  return (kv.indptr[batch_idx + 1] - kv.indptr[batch_idx] - 1) * kv.page_size +
         kv.last_page_len[batch_idx];
}

__device__ __forceinline__ size_t paged_get_elem_offset(const PagedKvHalf& kv, size_t page_idx,
                                                        size_t head_idx, size_t entry_idx,
                                                        size_t feat_idx) {
  return page_idx * kv.stride_page + head_idx * kv.stride_h + entry_idx * kv.stride_n + feat_idx;
}

__device__ __forceinline__ size_t paged_protective_get_kv_offset(const PagedKvHalf& kv,
                                                                 int32_t page_iter,
                                                                 uint32_t head_idx,
                                                                 uint32_t entry_idx,
                                                                 uint32_t feat_idx,
                                                                 int32_t last_indptr) {
  if (page_iter < last_indptr) {
    return paged_get_elem_offset(kv, __ldg(kv.indices + page_iter), head_idx, entry_idx, feat_idx);
  } else {
    return 0;
  }
}

// ---- default_decode_params.cuh BatchDecodeParams<half,half,half,int32_t>, same member order ----
struct DecodeParamsHalf {
  half* q;
  int32_t* q_rope_offset;
  PagedKvHalf paged_kv;
  half* o;
  float* lse;
  float* maybe_alibi_slopes;
  uint32_t padded_batch_size;
  uint32_t num_qo_heads;
  int32_t q_stride_n;
  int32_t q_stride_h;
  int32_t window_left;
  float logits_soft_cap;
  float sm_scale;
  float rope_rcp_scale;
  float rope_rcp_theta;
  int32_t* request_indices;
  int32_t* kv_tile_indices;
  int32_t* o_indptr;
  int32_t* kv_chunk_size_ptr;
  bool* block_valid_mask;
  bool partition_kv;
};

// ---- U19 flatten: the ONE scalar-only struct param of the kernel (pointer fields are separate
// buffer params, see the kernel signature). Scalars of BatchDecodeParams + paged_kv_t, same order. ----
struct DecodeScalars {
  uint32_t padded_batch_size;
  uint32_t num_qo_heads;
  int32_t q_stride_n;
  int32_t q_stride_h;
  int32_t window_left;
  float logits_soft_cap;
  float sm_scale;
  float rope_rcp_scale;
  float rope_rcp_theta;
  uint32_t partition_kv;  // 0/1 (was bool)
  uint_fastdiv page_size;  // paged_kv.page_size: d_, m_, l_
  uint32_t num_heads;
  uint32_t head_dim;
  uint32_t batch_size;
  uint32_t stride_page;
  uint32_t stride_n;
  uint32_t stride_h;
};

// ---- math.cuh ----
__forceinline__ __device__ float ptx_exp2(float x) {
  float y;
  asm volatile("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
  return y;
}
__forceinline__ __device__ float ptx_log2(float x) {
  float y;
  asm volatile("lg2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
  return y;
}
__forceinline__ __device__ float ptx_rcp(float x) {
  float y;
  asm volatile("rcp.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
  return y;
}
__forceinline__ __device__ float shfl_xor_sync(float x, int lane_mask) {
  float y;
  asm volatile("shfl.sync.bfly.b32 %0, %1, %2, 0x1f, 0xffffffff;"
               : "=f"(y)
               : "f"(x), "r"(lane_mask));
  return y;
}

// ---- cp_async.cuh: pred_load<128, kPrefetch, kNoFill / kFillZero> on half ----
#if (__CUDACC_VER_MAJOR__ >= 11)
#if (!defined(__CUDA_ARCH__) || (__CUDA_ARCH__ >= 800))
#define FLASHINFER_CP_ASYNC_ENABLED
#endif
#endif

__device__ __forceinline__ void cp_async_commit_group() {
#ifdef FLASHINFER_CP_ASYNC_ENABLED
  asm volatile("cp.async.commit_group;\n" ::);
#endif
}

// wait_group<n>: n is 2 * NUM_STAGES_SMEM - 1 = 3 in the loop, 0 after it
__device__ __forceinline__ void cp_async_wait_group_3() {
#ifdef FLASHINFER_CP_ASYNC_ENABLED
  asm volatile("cp.async.wait_group %0;\n" ::"n"(3));
#endif
}
__device__ __forceinline__ void cp_async_wait_group_0() {
#ifdef FLASHINFER_CP_ASYNC_ENABLED
  asm volatile("cp.async.wait_group %0;\n" ::"n"(0));
#endif
}

__device__ __forceinline__ void cp_async_pred_load_128b_nofill(half* smem_ptr, const half* gmem_ptr,
                                                               bool predicate) {
#ifdef FLASHINFER_CP_ASYNC_ENABLED
  uint32_t smem_int_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
  asm volatile(
      "{\n"
      " .reg .pred p;\n"
      " setp.ne.b32 p, %0, 0;\n"
      " @p cp.async.cg.shared.global.L2::128B [%1], [%2], %3;\n"
      "}\n" ::"r"((int)predicate),
      "r"(smem_int_ptr), "l"(gmem_ptr), "n"(16));
#else
  if (predicate) {
    *((uint4*)smem_ptr) = *((uint4*)gmem_ptr);
  }
#endif
}

__device__ __forceinline__ void cp_async_pred_load_128b_fillzero(half* smem_ptr,
                                                                 const half* gmem_ptr,
                                                                 bool predicate) {
#ifdef FLASHINFER_CP_ASYNC_ENABLED
  uint32_t smem_int_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
  int src_in_bytes = predicate ? 16 : 0;
  asm volatile("cp.async.cg.shared.global.L2::128B [%0], [%1], %2, %3;\n" ::"r"(smem_int_ptr),
               "l"(gmem_ptr), "n"(16), "r"(src_in_bytes));
#else
  if (predicate) {
    *((uint4*)smem_ptr) = *((uint4*)gmem_ptr);
  } else {
    *((uint4*)smem_ptr) = make_uint4(0, 0, 0, 0);
  }
#endif
}

// ---- vec_dtypes.cuh: vec_t<float,8> is `float4 data[2]`; vec_t<half,8> is `int4 data[1]` ----
// Float vector helpers; the vector itself is `alignas(16) float v[8]`.
__device__ __forceinline__ void vec8f_cast_load_from_half(float* dst, const half* src_ptr) {
  // cast_load_impl: vec_t<half,8> tmp; tmp.load(src_ptr); dst.cast_from(tmp)
  // -> vec_cast<float, half>::cast<8>(dst, tmp): four __half22float2
  int4 tmp = ((const int4*)src_ptr)[0];
#pragma unroll
  for (size_t i = 0; i < VEC_SIZE / 2; ++i) {
    ((float2*)dst)[i] = __half22float2(((half2*)&tmp)[i]);
  }
}
__device__ __forceinline__ void vec8f_store(const float* src, float* ptr) {
#pragma unroll
  for (size_t i = 0; i < VEC_SIZE / 4; ++i) {
    ((float4*)ptr)[i] = ((const float4*)src)[i];
  }
}
__device__ __forceinline__ void vec8f_load(float* dst, const float* ptr) {
#pragma unroll
  for (size_t i = 0; i < VEC_SIZE / 4; ++i) {
    ((float4*)dst)[i] = ((const float4*)ptr)[i];
  }
}
__device__ __forceinline__ void vec8f_cast_store_to_half(const float* src, half* ptr) {
  // cast_store_impl: vec_t<half,8> tmp; tmp.cast_from(src); tmp.store(ptr)
  int4 tmp;
#pragma unroll
  for (size_t i = 0; i < VEC_SIZE / 2; ++i) {
    ((half2*)&tmp)[i] = __float22half2_rn(((const float2*)src)[i]);
  }
  ((int4*)ptr)[0] = tmp;
}

// ---- state.cuh state_t<8> ----
struct State {
  alignas(16) float o[VEC_SIZE];
  float m;
  float d;
};
__device__ __forceinline__ void state_init(State& st) {
#pragma unroll
  for (size_t i = 0; i < VEC_SIZE / 4; ++i) {
    ((float4*)st.o)[i] = make_float4(0.f, 0.f, 0.f, 0.f);
  }
  st.m = -INF;
  st.d = 1.f;
}
__device__ __forceinline__ float state_get_lse(const State& st) { return st.m + ptx_log2(st.d); }
__device__ __forceinline__ void state_merge(State& st, const float* other_o, float other_m,
                                            float other_d) {
  float m_prev = st.m, d_prev = st.d;
  st.m = max(m_prev, other_m);
  st.d = d_prev * ptx_exp2(m_prev - st.m) + other_d * ptx_exp2(other_m - st.m);
#pragma unroll
  for (size_t i = 0; i < VEC_SIZE; ++i) {
    st.o[i] = st.o[i] * ptx_exp2(m_prev - st.m) + other_o[i] * ptx_exp2(other_m - st.m);
  }
}

// ---- decode.cuh: compute_qk (bdx=16, tile_size = bdy*tile_size_per_bdx = 4, kNone) ----
__device__ __forceinline__ void compute_qk(float sm_scale_log2,
                                           const half* smem, const float* q_vec,
                                           uint32_t iter_base, uint32_t iter_bound, float* s,
                                           State& st, const uint32_t tx, const uint32_t tz) {
  constexpr uint32_t tile_size = BDY * TILE_SIZE_PER_BDX;
  float m_prev = st.m;
#pragma unroll
  for (uint32_t j = 0; j < tile_size; ++j) {
    alignas(16) float k_vec[VEC_SIZE];
    vec8f_cast_load_from_half(k_vec, smem + (j * BDX + tx) * VEC_SIZE);
    s[j] = 0.f;
#pragma unroll
    for (uint32_t i = 0; i < VEC_SIZE; ++i) {
      s[j] += q_vec[i] * k_vec[i];
    }
#pragma unroll
    for (uint32_t offset = BDX / 2; offset > 0; offset /= 2) {
      s[j] += shfl_xor_sync(s[j], offset);
    }
    // LogitsTransform is the identity for DefaultAttention<false,false,false,false>
    s[j] *= sm_scale_log2;  // variant.use_softmax
    bool mask = true;       // LogitsMask: no custom mask, no sliding window
    s[j] = (iter_base + tz * tile_size + j < iter_bound && mask) ? s[j] : -INF;
    st.m = max(st.m, s[j]);
  }

  float o_scale = ptx_exp2(m_prev - st.m);
  st.d *= o_scale;
#pragma unroll
  for (uint32_t j = 0; j < tile_size; ++j) {
    s[j] = ptx_exp2(s[j] - st.m);
    st.d += s[j];
  }
#pragma unroll
  for (uint32_t i = 0; i < VEC_SIZE; ++i) {
    st.o[i] = st.o[i] * o_scale;
  }
}

// ---- decode.cuh: update_local_state (vec_size=8, bdx=16, tile_size=4) ----
__device__ __forceinline__ void update_local_state(const half* smem, const float* s, State& st,
                                                   uint32_t tx) {
  constexpr uint32_t tile_size = BDY * TILE_SIZE_PER_BDX;
#pragma unroll
  for (uint32_t j = 0; j < tile_size; ++j) {
    alignas(16) float v_vec[VEC_SIZE];
    vec8f_cast_load_from_half(v_vec, smem + (j * BDX + tx) * VEC_SIZE);
#pragma unroll
    for (uint32_t i = 0; i < VEC_SIZE; ++i) {
      st.o[i] = st.o[i] + s[j] * v_vec[i];
    }
  }
}

// ---- decode.cuh: sync_state (bdz = 8 > 1, use_softmax) ----
__device__ __forceinline__ void sync_state(State& st, float* smem, float* smem_md,
                                           const uint32_t tx, const uint32_t ty,
                                           const uint32_t tz) {
  constexpr uint32_t head_dim = BDX * VEC_SIZE;
  vec8f_store(st.o, smem + (tz * BDY + ty) * head_dim + tx * VEC_SIZE);
  smem_md[(tz * BDY + ty) * 2] = st.m;
  smem_md[(tz * BDY + ty) * 2 + 1] = st.d;
  __syncthreads();
  state_init(st);
#pragma unroll
  for (uint32_t j = 0; j < BDZ; ++j) {
    float mz = smem_md[(j * BDY + ty) * 2], dz = smem_md[(j * BDY + ty) * 2 + 1];
    alignas(16) float oz[VEC_SIZE];
    vec8f_load(oz, smem + (j * BDY + ty) * head_dim + tx * VEC_SIZE);
    state_merge(st, oz, mz, dz);
  }
}

// ---- decode.cuh: BatchDecodeWithPagedKVCacheKernel + ...Device, one configuration ----
__global__ void BatchDecodeWithPagedKVCacheKernel_f16_d128_g1_none(
    half* q_ptr, int32_t* q_rope_offset, half* o_ptr, float* lse_ptr, float* maybe_alibi_slopes,
    int32_t* request_indices, int32_t* kv_tile_indices, int32_t* o_indptr,
    int32_t* kv_chunk_size_ptr, bool* block_valid_mask_ptr, half* k_data, half* v_data,
    int32_t* indices, int32_t* indptr, int32_t* last_page_len, int32_t* rope_pos_offset,
    const __grid_constant__ DecodeScalars params) {
  extern __shared__ uint8_t smem[];
  const uint32_t bx = blockIdx.x;
  const uint32_t by = blockIdx.y;
  const uint32_t tx = threadIdx.x;
  const uint32_t ty = threadIdx.y;
  const uint32_t tz = threadIdx.z;

  const half* q = q_ptr;
  half* o = o_ptr;
  float* lse = lse_ptr;
  const PagedKvHalf paged_kv = {params.page_size, params.num_heads,  params.head_dim,
                                params.batch_size, params.stride_page, params.stride_n,
                                params.stride_h,   k_data,            v_data,
                                indices,           indptr,            last_page_len,
                                rope_pos_offset};
  const bool* block_valid_mask = block_valid_mask_ptr;
  const uint32_t num_qo_heads = params.num_qo_heads;
  const bool partition_kv = params.partition_kv != 0;

  constexpr uint32_t head_dim = BDX * VEC_SIZE;
  const uint32_t batch_idx = request_indices[bx];
  const uint32_t kv_tile_idx = kv_tile_indices[bx];
  const uint32_t kv_head_idx = by;
  const uint32_t qo_head_idx = kv_head_idx * BDY + ty;
  if (block_valid_mask && !block_valid_mask[bx]) return;
  const uint32_t kv_chunk_size = *kv_chunk_size_ptr;
  const uint32_t kv_len = paged_get_length(paged_kv, batch_idx);
  const uint32_t max_chunk_size = partition_kv ? kv_chunk_size : kv_len;
  const uint32_t chunk_start = partition_kv ? kv_tile_idx * max_chunk_size : 0;
  const uint32_t chunk_end =
      partition_kv ? min((kv_tile_idx + 1) * max_chunk_size, kv_len) : kv_len;
  const uint32_t chunk_size = chunk_end - chunk_start;

  // DefaultAttention ctor, reduced to the one member this configuration reads
  const float sm_scale_log2 = params.sm_scale * LOG2E;
  half* k_smem = (half*)smem;
  half* v_smem = (half*)(smem + NUM_STAGES_SMEM * TILE_SIZE_PER_BDX * BDY * BDZ * head_dim *
                                    sizeof(half));
  size_t* kv_offset_smem = (size_t*)(smem + 2 * NUM_STAGES_SMEM * TILE_SIZE_PER_BDX * BDY * BDZ *
                                                head_dim * sizeof(half));
  float* smem_md = (float*)(smem + 2 * NUM_STAGES_SMEM * TILE_SIZE_PER_BDX * BDY * BDZ * head_dim *
                                       sizeof(half));

  alignas(16) float q_vec[VEC_SIZE];
  const uint32_t q_stride_n = params.q_stride_n;
  const uint32_t q_stride_h = params.q_stride_h;
  // POS_ENCODING_MODE == kNone: do not apply rotary embedding to q matrix
#if (__CUDACC_VER_MAJOR__ >= 12 && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
  asm volatile("griddepcontrol.wait;");
#endif
  vec8f_cast_load_from_half(q_vec, q + batch_idx * q_stride_n + qo_head_idx * q_stride_h +
                                       tx * VEC_SIZE);

  // preload k/v tiles
  uint32_t stage_idx = 0;
  const int32_t last_indptr = paged_kv.indptr[paged_kv.batch_size];

  uint32_t packed_page_iter_base = paged_kv.indptr[batch_idx] * paged_kv.page_size + chunk_start;
#pragma unroll
  for (uint32_t j = 0; j < TILE_SIZE_PER_BDX; ++j) {
    uint32_t qq, r;
    paged_kv.page_size.divmod(packed_page_iter_base + ((j * BDZ + tz) * BDY + ty) * BDX + tx, qq,
                              r);
    kv_offset_smem[((j * BDZ + tz) * BDY + ty) * BDX + tx] =
        paged_protective_get_kv_offset(paged_kv, qq, kv_head_idx, r, 0, last_indptr);
  }
  __syncthreads();

  size_t kv_offset[TILE_SIZE_PER_BDX];
#pragma unroll
  for (uint32_t iter = 0; iter < NUM_STAGES_SMEM; ++iter) {
#pragma unroll
    for (uint32_t j = 0; j < TILE_SIZE_PER_BDX; ++j) {
      kv_offset[j] =
          kv_offset_smem[((iter * BDZ + tz) * BDY + ty) * TILE_SIZE_PER_BDX + j] + tx * VEC_SIZE;
    }
#pragma unroll
    for (uint32_t j = 0; j < TILE_SIZE_PER_BDX; ++j) {
      cp_async_pred_load_128b_nofill(
          k_smem + (((stage_idx * BDZ + tz) * BDY + ty) * TILE_SIZE_PER_BDX + j) * head_dim +
              tx * VEC_SIZE,
          paged_kv.k_data + kv_offset[j],
          ((iter * BDZ + tz) * BDY + ty) * TILE_SIZE_PER_BDX + j < chunk_size);
    }
    cp_async_commit_group();
#pragma unroll
    for (uint32_t j = 0; j < TILE_SIZE_PER_BDX; ++j) {
      cp_async_pred_load_128b_fillzero(
          v_smem + (((stage_idx * BDZ + tz) * BDY + ty) * TILE_SIZE_PER_BDX + j) * head_dim +
              tx * VEC_SIZE,
          paged_kv.v_data + kv_offset[j],
          ((iter * BDZ + tz) * BDY + ty) * TILE_SIZE_PER_BDX + j < chunk_size);
    }
    cp_async_commit_group();
    stage_idx = (stage_idx + 1) % NUM_STAGES_SMEM;
  }

  State st;
  state_init(st);
  float s[BDY * TILE_SIZE_PER_BDX];

  // ceil_div(chunk_size, tile_size_per_bdx * bdy * bdz) = (x + y - 1) / y
#pragma unroll 2
  for (uint32_t iter = 0;
       iter < (chunk_size + (TILE_SIZE_PER_BDX * BDY * BDZ) - 1) / (TILE_SIZE_PER_BDX * BDY * BDZ);
       ++iter) {
    if ((iter + NUM_STAGES_SMEM) % BDX == 0) {
#pragma unroll
      for (uint32_t j = 0; j < TILE_SIZE_PER_BDX; ++j) {
        uint32_t qq, r;
        paged_kv.page_size.divmod(
            packed_page_iter_base + ((iter + NUM_STAGES_SMEM) * TILE_SIZE_PER_BDX * BDY * BDZ +
                                     ((j * BDZ + tz) * BDY + ty) * BDX + tx),
            qq, r);
        kv_offset_smem[((j * BDZ + tz) * BDY + ty) * BDX + tx] =
            paged_protective_get_kv_offset(paged_kv, qq, kv_head_idx, r, 0, last_indptr);
      }
    }
    // compute qk
    cp_async_wait_group_3();
    __syncthreads();
    compute_qk(sm_scale_log2,
               k_smem + (stage_idx * BDZ + tz) * BDY * TILE_SIZE_PER_BDX * head_dim, q_vec,
               iter * TILE_SIZE_PER_BDX * BDY * BDZ, chunk_size, s, st, tx, tz);
    __syncthreads();

#pragma unroll
    for (uint32_t j = 0; j < TILE_SIZE_PER_BDX; ++j) {
      kv_offset[j] = kv_offset_smem[((((iter + NUM_STAGES_SMEM) % BDX) * BDZ + tz) * BDY + ty) *
                                        TILE_SIZE_PER_BDX +
                                    j] +
                     tx * VEC_SIZE;
    }

    // load k tiles
#pragma unroll
    for (uint32_t j = 0; j < TILE_SIZE_PER_BDX; ++j) {
      cp_async_pred_load_128b_nofill(
          k_smem + (((stage_idx * BDZ + tz) * BDY + ty) * TILE_SIZE_PER_BDX + j) * head_dim +
              tx * VEC_SIZE,
          paged_kv.k_data + kv_offset[j],
          (((iter + NUM_STAGES_SMEM) * BDZ + tz) * BDY + ty) * TILE_SIZE_PER_BDX + j < chunk_size);
    }
    cp_async_commit_group();

    // update m/d/o states
    cp_async_wait_group_3();
    __syncthreads();
    update_local_state(v_smem + (stage_idx * BDZ + tz) * BDY * TILE_SIZE_PER_BDX * head_dim, s,
                       st, tx);
    __syncthreads();

    // load v tiles
#pragma unroll
    for (uint32_t j = 0; j < TILE_SIZE_PER_BDX; ++j) {
      cp_async_pred_load_128b_fillzero(
          v_smem + (((stage_idx * BDZ + tz) * BDY + ty) * TILE_SIZE_PER_BDX + j) * head_dim +
              tx * VEC_SIZE,
          paged_kv.v_data + kv_offset[j],
          (((iter + NUM_STAGES_SMEM) * BDZ + tz) * BDY + ty) * TILE_SIZE_PER_BDX + j < chunk_size);
    }
    cp_async_commit_group();
    stage_idx = (stage_idx + 1) % NUM_STAGES_SMEM;
  }
  cp_async_wait_group_0();
  __syncthreads();

  // sync local state of all warps inside a threadblock
  sync_state(st, reinterpret_cast<float*>(smem), smem_md, tx, ty, tz);
#pragma unroll
  for (size_t i = 0; i < VEC_SIZE; ++i) {
    // AttentionVariantBase::OutputTransform; get_v_scale(params) is 1.0f (Params has no v_scale)
    float d_rcp = (st.m != -INF) ? ptx_rcp(st.d) : 0.f;
    float v_scale_val = 1.0f;
    st.o[i] = st.o[i] * d_rcp * v_scale_val;
  }

  if (tz == 0) {
    vec8f_cast_store_to_half(st.o, o + (bx * num_qo_heads + qo_head_idx) * head_dim +
                                       tx * VEC_SIZE);
    // write lse
    if (lse != nullptr) {
      lse[bx * num_qo_heads + qo_head_idx] = state_get_lse(st);
    }
  }
#if (__CUDACC_VER_MAJOR__ >= 12 && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
  asm volatile("griddepcontrol.launch_dependents;");
#endif
}

}  // namespace flat_decode
