#include "api.h"
#include "contract.h"
#include "params.h"

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime.h>

namespace {

using namespace gtsparse_row_template_center_last;
using CenterLastFP32Params = FinalizeRowTemplateCenterLastFP32Params;

struct CenterLastFP32Setting1Cfg {
    static constexpr int BM = 128;
    static constexpr int BN = 16;
    static constexpr int BK = 16;
    static constexpr int THREADS = 64;
    static constexpr int A_SHARED_SIZE = 2048;
    static constexpr int B_SHARED_SIZE = 256;
};

struct CenterLastFP32Setting2Cfg {
    static constexpr int BM = 128;
    static constexpr int BN = 16;
    static constexpr int BK = 32;
    static constexpr int THREADS = 64;
    static constexpr int A_SHARED_SIZE = 4096;
    static constexpr int B_SHARED_SIZE = 512;
};

struct CenterLastFP32Setting3Cfg {
    static constexpr int BM = 128;
    static constexpr int BN = 64;
    static constexpr int BK = 32;
    static constexpr int THREADS = 128;
    static constexpr int A_SHARED_SIZE = 4096;
    static constexpr int B_SHARED_SIZE = 2048;
};

template <int bytes>
struct center_last_global_load;

template <>
struct center_last_global_load<16> {
    __device__ __inline__ center_last_global_load(uint4& D, void const* ptr, int pred_guard) {
        uint4& data = *reinterpret_cast<uint4*>(&D);
        asm volatile(
            "{\n"
            "  .reg .pred p;\n"
            "  setp.ne.b32 p, %5, 0;\n"
            "  mov.b32 %0, %6;\n"
            "  mov.b32 %1, %7;\n"
            "  mov.b32 %2, %8;\n"
            "  mov.b32 %3, %9;\n"
            "  @p ld.global.v4.u32 {%0, %1, %2, %3}, [%4];\n"
            "}\n"
            : "=r"(data.x), "=r"(data.y), "=r"(data.z), "=r"(data.w)
            : "l"(ptr), "r"((int)(pred_guard & 1)), "r"(data.x), "r"(data.y), "r"(data.z), "r"(data.w));
    }
};

template <>
struct center_last_global_load<8> {
    __device__ __inline__ center_last_global_load(uint4& D, void const* ptr, int pred_guard) {
        uint2 const* ptr_ldg = reinterpret_cast<uint2 const*>(ptr);
        #pragma unroll
        for (int ldg_idx = 0; ldg_idx < 2; ++ldg_idx) {
            uint2& data = *(reinterpret_cast<uint2*>(&D) + ldg_idx);
            asm volatile(
                "{\n"
                "  .reg .pred p;\n"
                "  setp.ne.b32 p, %3, 0;\n"
                "  mov.b32 %0, %4;\n"
                "  mov.b32 %1, %5;\n"
                "  @p ld.global.v2.u32 {%0, %1}, [%2];\n"
                "}\n"
                : "=r"(data.x), "=r"(data.y)
                : "l"(ptr_ldg + ldg_idx), "r"((int)(pred_guard & (1 << ldg_idx))), "r"(data.x), "r"(data.y));
        }
    }
};

template <>
struct center_last_global_load<4> {
    __device__ __inline__ center_last_global_load(uint4& D, void const* ptr, int pred_guard) {
        unsigned const* ptr_ldg = reinterpret_cast<unsigned const*>(ptr);
        #pragma unroll
        for (int ldg_idx = 0; ldg_idx < 4; ++ldg_idx) {
            unsigned& data = *(reinterpret_cast<unsigned*>(&D) + ldg_idx);
            asm volatile(
                "{\n"
                "  .reg .pred p;\n"
                "  setp.ne.b32 p, %2, 0;\n"
                "  mov.b32 %0, %3;\n"
                "  @p ld.global.u32 %0, [%1];\n"
                "}\n"
                : "=r"(data)
                : "l"(ptr_ldg + ldg_idx), "r"((int)(pred_guard & (1 << ldg_idx))), "r"(data));
        }
    }
};

__device__ __forceinline__ int logical_template_slot_count(int template_id) {
    switch (template_id) {
        case kTemplateCenter:
            return kSlotCountW1;
        case kTemplateSkip2Keep0:
            return 10;
        case kTemplateSkip2Keep1:
            return 9;
        case kTemplateSkip2Keep2:
            return 10;
        case kTemplateSkip1Hole0:
            return 18;
        case kTemplateSkip1Hole1:
            return 19;
        case kTemplateSkip1Hole2:
            return 18;
        default:
            return kSlotCountW27;
    }
}

__device__ __forceinline__ int logical_template_payload_width(int template_id) {
    switch (template_id) {
        case kTemplateCenter:
            return kPayloadWidthW1;
        case kTemplateSkip2Keep0:
        case kTemplateSkip2Keep1:
        case kTemplateSkip2Keep2:
            return kPayloadWidthW9;
        case kTemplateSkip1Hole0:
        case kTemplateSkip1Hole1:
        case kTemplateSkip1Hole2:
            return kPayloadWidthW18;
        default:
            return kPayloadWidthW27;
    }
}

__device__ __forceinline__ int logical_template_initial_bias(int template_id) {
    switch (template_id) {
        case kTemplateCenter:
            return 26;
        case kTemplateSkip2Keep0:
            return 0;
        case kTemplateSkip2Keep1:
            return 1;
        case kTemplateSkip2Keep2:
            return 2;
        case kTemplateSkip1Hole0:
            return 1;
        case kTemplateSkip1Hole1:
        case kTemplateSkip1Hole2:
        case kTemplateFull27:
        default:
            return 0;
    }
}

__device__ __forceinline__ int logical_template_boundary_bias_bump(
    int template_id,
    int prev_compact_slot) {
    if (template_id == kTemplateSkip2Keep0) {
        return (prev_compact_slot == 4) ? 1 : 2;
    }
    if (template_id == kTemplateSkip2Keep1) {
        if (prev_compact_slot == 3) {
            return 4;
        }
        if (prev_compact_slot == 7) {
            return 1;
        }
        return 2;
    }
    if (template_id == kTemplateSkip2Keep2) {
        if (prev_compact_slot == 3) {
            return 1;
        }
        if (prev_compact_slot == 8) {
            return 0;
        }
        return 2;
    }
    if (template_id == kTemplateSkip1Hole0) {
        if (prev_compact_slot < 8) {
            return prev_compact_slot & 1;
        }
        if (prev_compact_slot == 8) {
            return 1;
        }
        if (prev_compact_slot < 16) {
            return (prev_compact_slot + 1) & 1;
        }
        return 0;
    }
    if (template_id == kTemplateSkip1Hole1) {
        if (prev_compact_slot == 8 || prev_compact_slot == 9) {
            return 0;
        }
        return ((prev_compact_slot & 1) == 0) ? 1 : 0;
    }
    if (template_id == kTemplateSkip1Hole2) {
        if (prev_compact_slot < 8) {
            return prev_compact_slot & 1;
        }
        if (prev_compact_slot == 8) {
            return 1;
        }
        return (prev_compact_slot + 1) & 1;
    }
    return 0;
}

template <typename Params>
__device__ __forceinline__ const int* logical_input_row_ptr(
    const Params& p,
    int template_id,
    int input_row_offset,
    int row_local_seed) {
    if (template_id == kTemplateCenter) {
        return p.input_rows_w1 + (input_row_offset + row_local_seed) * kSlotCountW1;
    }
    if (template_id >= kTemplateSkip2Keep0 && template_id <= kTemplateSkip2Keep2) {
        const int local_template_id = template_id - kTemplateSkip2Keep0;
        return p.input_rows_w9 +
            ((local_template_id * p.template_stride) + input_row_offset + row_local_seed) * kPayloadWidthW9;
    }
    if (template_id >= kTemplateSkip1Hole0 && template_id <= kTemplateSkip1Hole2) {
        const int local_template_id = template_id - kTemplateSkip1Hole0;
        return p.input_rows_w18 +
            ((local_template_id * p.template_stride) + input_row_offset + row_local_seed) * kPayloadWidthW18;
    }
    return p.input_rows_w27 + (input_row_offset + row_local_seed) * kSlotCountW27;
}

template <int Threads, typename Params>
__device__ __forceinline__ bool sorted_tile_compact_slot_has_active(
    const Params& p,
    int template_id,
    int input_row_offset,
    int compact_slot,
    int payload_width,
    int* __restrict__ slot_active_shared) {
    if (threadIdx.x == 0) {
        *slot_active_shared = 0;
    }
    __syncthreads();

    const int* tile_slot_ptr =
        logical_input_row_ptr(p, template_id, input_row_offset, 0) + compact_slot;
    bool local_active = false;
    for (int row = threadIdx.x; row < kBM; row += Threads) {
        if (tile_slot_ptr[row * payload_width] >= 0) {
            local_active = true;
            break;
        }
    }
    if (__any_sync(0xffffffffu, local_active) && ((threadIdx.x & 31) == 0)) {
        atomicExch(slot_active_shared, 1);
    }
    __syncthreads();
    return *slot_active_shared != 0;
}

__device__ __forceinline__ void center_last_fp32_setting3_tile(
    const CenterLastFP32Params& p,
    int template_id,
    int out_row_base,
    int input_row_offset,
    int bn_base,
    float* A_shared,
    float* B_shared) {
    const int tid = static_cast<int>(threadIdx.x);
    const int Cin = p.c_in;
    const int Cout = p.c_out;
    const int loops_per_slot = Cin / CenterLastFP32Setting3Cfg::BK;
    const int slot_count = logical_template_slot_count(template_id);
    const int payload_width = logical_template_payload_width(template_id);
    const int total_k_loops = slot_count * loops_per_slot;
    const int row_pitch = payload_width * 16;
    const int logical_stride = Cin * Cout;
    const int weight_k_stride = 8 * Cout;

    float C_local[64];
    #pragma unroll
    for (int i = 0; i < 64; ++i) {
        C_local[i] = 0.0f;
    }

    float* A_shared_ptr = A_shared + tid * 4;
    float* A_shared_reduce_ptr = A_shared + ((tid / 16) * 32);
    float* B_shared_ptr = B_shared + tid * 4;
    float* B_shared_reduce_ptr = B_shared + (tid % 16);
    const int location_offset = out_row_base + (tid / 16);
    const int c_n_offset = bn_base + (tid % 16);
    const int channel_offset_a = (tid * 4) % 32;
    const int row_local_seed = tid / 8;
    const int* row_seed_ptr =
        logical_input_row_ptr(p, template_id, input_row_offset, row_local_seed);
    const int* slot_row_ptr = row_seed_ptr;

    int compact_slot = 0;
    int ci_tile = 0;
    int ci_offset = 0;
    const float* slot_weight_ptr =
        p.weight
        + logical_template_initial_bias(template_id) * logical_stride
        + (tid / 16) * Cout
        + bn_base
        + ((tid * 4) % 64);

    #pragma unroll
    for (int k_0 = 0; k_0 < total_k_loops; ++k_0) {
        const int* out_in_map_ptr_k = slot_row_ptr;
        const float* slot_weight_ci_ptr = slot_weight_ptr + ci_offset * Cout;

        __syncthreads();
        const int* input_idx_ptr = out_in_map_ptr_k;
        #pragma unroll
        for (int ax = 0; ax < 8; ++ax) {
            const int input_idx = *input_idx_ptr;
            if (input_idx != -1) {
                *(reinterpret_cast<float4*>(A_shared_ptr + ax * 512)) =
                    *(reinterpret_cast<const float4*>(
                        p.features + input_idx * Cin + ci_offset + channel_offset_a));
            } else {
                *(reinterpret_cast<float4*>(A_shared_ptr + ax * 512)) = make_float4(0.f, 0.f, 0.f, 0.f);
            }
            input_idx_ptr += row_pitch;
        }
        const float* B_k_ptr = slot_weight_ci_ptr;
        #pragma unroll
        for (int ax = 0; ax < 4; ++ax) {
            *(reinterpret_cast<float4*>(B_shared_ptr + ax * 512)) =
                *(reinterpret_cast<const float4*>(B_k_ptr));
            B_k_ptr += weight_k_stride;
        }

        __syncthreads();
        #pragma unroll
        for (int k_1 = 0; k_1 < 8; ++k_1) {
            #pragma unroll
            for (int k_2 = 0; k_2 < 4; ++k_2) {
                const int vk = (k_1 << 2) + k_2;
                #pragma unroll
                for (int i = 0; i < 64; ++i) {
                    C_local[i] +=
                        A_shared_reduce_ptr[((i / 4) * 8) * 32 + vk]
                        * B_shared_reduce_ptr[(vk * 64) + ((i % 4) * 16)];
                }
            }
        }

        ++ci_tile;
        ci_offset += CenterLastFP32Setting3Cfg::BK;
        if (ci_tile == loops_per_slot) {
            ci_tile = 0;
            ci_offset = 0;
            const int prev_compact_slot = compact_slot;
            ++compact_slot;
            if (compact_slot < slot_count) {
                const int logical_delta =
                    1 + logical_template_boundary_bias_bump(template_id, prev_compact_slot);
                slot_row_ptr += 1;
                slot_weight_ptr += logical_delta * logical_stride;
            }
        }
    }

    #pragma unroll
    for (int i = 0; i < 64; ++i) {
        const int out_row = p.out_rows[location_offset + ((i / 4) * 8)];
        const int col = c_n_offset + (i % 4) * 16;
        if (out_row >= 0 && col < Cout) {
            p.output[out_row * Cout + col] = C_local[i];
        }
    }
}

__device__ __forceinline__ void center_last_fp32_setting3_sorted_tile(
    const CenterLastFP32Params& p,
    int template_id,
    int out_row_base,
    int input_row_offset,
    int bn_base,
    float* A_shared,
    float* B_shared,
    int* slot_active_shared) {
    const int tid = static_cast<int>(threadIdx.x);
    const int Cin = p.c_in;
    const int Cout = p.c_out;
    const int loops_per_slot = Cin / CenterLastFP32Setting3Cfg::BK;
    const int slot_count = logical_template_slot_count(template_id);
    const int payload_width = logical_template_payload_width(template_id);
    const int row_pitch = payload_width * 16;
    const int logical_stride = Cin * Cout;
    const int weight_k_stride = 8 * Cout;

    float C_local[64];
    #pragma unroll
    for (int i = 0; i < 64; ++i) {
        C_local[i] = 0.0f;
    }

    float* A_shared_ptr = A_shared + tid * 4;
    float* A_shared_reduce_ptr = A_shared + ((tid / 16) * 32);
    float* B_shared_ptr = B_shared + tid * 4;
    float* B_shared_reduce_ptr = B_shared + (tid % 16);
    const int location_offset = out_row_base + (tid / 16);
    const int c_n_offset = bn_base + (tid % 16);
    const int channel_offset_a = (tid * 4) % 32;
    const int row_local_seed = tid / 8;
    const int* slot_row_ptr =
        logical_input_row_ptr(p, template_id, input_row_offset, row_local_seed);
    const float* slot_weight_ptr =
        p.weight
        + logical_template_initial_bias(template_id) * logical_stride
        + (tid / 16) * Cout
        + bn_base
        + ((tid * 4) % 64);

    for (int compact_slot = 0; compact_slot < slot_count; ++compact_slot) {
        if (sorted_tile_compact_slot_has_active<CenterLastFP32Setting3Cfg::THREADS>(
                p,
                template_id,
                input_row_offset,
                compact_slot,
                payload_width,
                slot_active_shared)) {
            int ci_offset = 0;
            for (int ci_tile = 0; ci_tile < loops_per_slot; ++ci_tile) {
                const int* out_in_map_ptr_k = slot_row_ptr;
                const float* slot_weight_ci_ptr = slot_weight_ptr + ci_offset * Cout;

                __syncthreads();
                const int* input_idx_ptr = out_in_map_ptr_k;
                #pragma unroll
                for (int ax = 0; ax < 8; ++ax) {
                    const int input_idx = *input_idx_ptr;
                    if (input_idx != -1) {
                        *(reinterpret_cast<float4*>(A_shared_ptr + ax * 512)) =
                            *(reinterpret_cast<const float4*>(
                                p.features + input_idx * Cin + ci_offset + channel_offset_a));
                    } else {
                        *(reinterpret_cast<float4*>(A_shared_ptr + ax * 512)) = make_float4(0.f, 0.f, 0.f, 0.f);
                    }
                    input_idx_ptr += row_pitch;
                }
                const float* B_k_ptr = slot_weight_ci_ptr;
                #pragma unroll
                for (int ax = 0; ax < 4; ++ax) {
                    *(reinterpret_cast<float4*>(B_shared_ptr + ax * 512)) =
                        *(reinterpret_cast<const float4*>(B_k_ptr));
                    B_k_ptr += weight_k_stride;
                }

                __syncthreads();
                #pragma unroll
                for (int k_1 = 0; k_1 < 8; ++k_1) {
                    #pragma unroll
                    for (int k_2 = 0; k_2 < 4; ++k_2) {
                        const int vk = (k_1 << 2) + k_2;
                        #pragma unroll
                        for (int i = 0; i < 64; ++i) {
                            C_local[i] +=
                                A_shared_reduce_ptr[((i / 4) * 8) * 32 + vk]
                                * B_shared_reduce_ptr[(vk * 64) + ((i % 4) * 16)];
                        }
                    }
                }
                ci_offset += CenterLastFP32Setting3Cfg::BK;
            }
        }
        if (compact_slot + 1 < slot_count) {
            const int logical_delta =
                1 + logical_template_boundary_bias_bump(template_id, compact_slot);
            slot_row_ptr += 1;
            slot_weight_ptr += logical_delta * logical_stride;
        }
    }

    #pragma unroll
    for (int i = 0; i < 64; ++i) {
        const int out_row = p.out_rows[location_offset + ((i / 4) * 8)];
        const int col = c_n_offset + (i % 4) * 16;
        if (out_row >= 0 && col < Cout) {
            p.output[out_row * Cout + col] = C_local[i];
        }
    }
}

__device__ __forceinline__ void center_last_fp32_setting2_tile(
    const CenterLastFP32Params& p,
    int template_id,
    int out_row_base,
    int input_row_offset,
    int bn_base,
    float* A_shared,
    float* B_shared) {
    const int tid = static_cast<int>(threadIdx.x);
    const int Cin = p.c_in;
    const int Cout = p.c_out;
    const int loops_per_slot = Cin / CenterLastFP32Setting2Cfg::BK;
    const int slot_count = logical_template_slot_count(template_id);
    const int payload_width = logical_template_payload_width(template_id);
    const int total_k_loops = slot_count * loops_per_slot;
    const int row_pitch = payload_width * 8;
    const int logical_stride = Cin * Cout;
    const int weight_k_stride = 16 * Cout;

    float C_local[32];
    #pragma unroll
    for (int i = 0; i < 32; ++i) {
        C_local[i] = 0.0f;
    }

    float* A_shared_ptr = A_shared + tid * 4;
    float* A_shared_reduce_ptr = A_shared + ((tid / 4) * 32);
    float* B_shared_ptr = B_shared + tid * 4;
    float* B_shared_reduce_ptr = B_shared + (tid % 4);
    const int location_offset = out_row_base + (tid / 4);
    const int c_n_offset = bn_base + (tid % 4);
    const int channel_offset_a = (tid * 4) % 32;
    const int row_local_seed = tid / 8;
    const int* row_seed_ptr =
        logical_input_row_ptr(p, template_id, input_row_offset, row_local_seed);
    const int* slot_row_ptr = row_seed_ptr;

    int compact_slot = 0;
    int ci_tile = 0;
    int ci_offset = 0;
    const float* slot_weight_ptr =
        p.weight
        + logical_template_initial_bias(template_id) * logical_stride
        + (tid / 4) * Cout
        + bn_base
        + ((tid * 4) % 16);

    #pragma unroll
    for (int k_0 = 0; k_0 < total_k_loops; ++k_0) {
        const int* out_in_map_ptr_k = slot_row_ptr;
        const float* slot_weight_ci_ptr = slot_weight_ptr + ci_offset * Cout;

        __syncthreads();
        const int* input_idx_ptr = out_in_map_ptr_k;
        #pragma unroll
        for (int ax = 0; ax < 16; ++ax) {
            const int input_idx = *input_idx_ptr;
            if (input_idx != -1) {
                *(reinterpret_cast<float4*>(A_shared_ptr + ax * 256)) =
                    *(reinterpret_cast<const float4*>(
                        p.features + input_idx * Cin + ci_offset + channel_offset_a));
            } else {
                *(reinterpret_cast<float4*>(A_shared_ptr + ax * 256)) = make_float4(0.f, 0.f, 0.f, 0.f);
            }
            input_idx_ptr += row_pitch;
        }
        const float* B_k_ptr = slot_weight_ci_ptr;
        #pragma unroll
        for (int ax = 0; ax < 2; ++ax) {
            *(reinterpret_cast<float4*>(B_shared_ptr + ax * 256)) =
                *(reinterpret_cast<const float4*>(B_k_ptr));
            B_k_ptr += weight_k_stride;
        }

        __syncthreads();
        #pragma unroll
        for (int k_1 = 0; k_1 < 8; ++k_1) {
            #pragma unroll
            for (int k_2 = 0; k_2 < 4; ++k_2) {
                const int vk = (k_1 << 2) + k_2;
                #pragma unroll
                for (int i = 0; i < 32; ++i) {
                    C_local[i] +=
                        A_shared_reduce_ptr[((i / 4) * 16) * 32 + vk]
                        * B_shared_reduce_ptr[(vk * 16) + ((i % 4) * 4)];
                }
            }
        }

        ++ci_tile;
        ci_offset += CenterLastFP32Setting2Cfg::BK;
        if (ci_tile == loops_per_slot) {
            ci_tile = 0;
            ci_offset = 0;
            const int prev_compact_slot = compact_slot;
            ++compact_slot;
            if (compact_slot < slot_count) {
                const int logical_delta =
                    1 + logical_template_boundary_bias_bump(template_id, prev_compact_slot);
                slot_row_ptr += 1;
                slot_weight_ptr += logical_delta * logical_stride;
            }
        }
    }

    #pragma unroll
    for (int i = 0; i < 32; ++i) {
        const int out_row = p.out_rows[location_offset + ((i / 4) * 16)];
        const int col = c_n_offset + ((i % 4) * 4);
        if (out_row >= 0 && col < Cout) {
            p.output[out_row * Cout + col] = C_local[i];
        }
    }
}

__device__ __forceinline__ void center_last_fp32_setting2_sorted_tile(
    const CenterLastFP32Params& p,
    int template_id,
    int out_row_base,
    int input_row_offset,
    int bn_base,
    float* A_shared,
    float* B_shared,
    int* slot_active_shared) {
    const int tid = static_cast<int>(threadIdx.x);
    const int Cin = p.c_in;
    const int Cout = p.c_out;
    const int loops_per_slot = Cin / CenterLastFP32Setting2Cfg::BK;
    const int slot_count = logical_template_slot_count(template_id);
    const int payload_width = logical_template_payload_width(template_id);
    const int row_pitch = payload_width * 8;
    const int logical_stride = Cin * Cout;
    const int weight_k_stride = 16 * Cout;

    float C_local[32];
    #pragma unroll
    for (int i = 0; i < 32; ++i) {
        C_local[i] = 0.0f;
    }

    float* A_shared_ptr = A_shared + tid * 4;
    float* A_shared_reduce_ptr = A_shared + ((tid / 4) * 32);
    float* B_shared_ptr = B_shared + tid * 4;
    float* B_shared_reduce_ptr = B_shared + (tid % 4);
    const int location_offset = out_row_base + (tid / 4);
    const int c_n_offset = bn_base + (tid % 4);
    const int channel_offset_a = (tid * 4) % 32;
    const int row_local_seed = tid / 8;
    const int* slot_row_ptr =
        logical_input_row_ptr(p, template_id, input_row_offset, row_local_seed);
    const float* slot_weight_ptr =
        p.weight
        + logical_template_initial_bias(template_id) * logical_stride
        + (tid / 4) * Cout
        + bn_base
        + ((tid * 4) % 16);

    for (int compact_slot = 0; compact_slot < slot_count; ++compact_slot) {
        if (sorted_tile_compact_slot_has_active<CenterLastFP32Setting2Cfg::THREADS>(
                p,
                template_id,
                input_row_offset,
                compact_slot,
                payload_width,
                slot_active_shared)) {
            int ci_offset = 0;
            for (int ci_tile = 0; ci_tile < loops_per_slot; ++ci_tile) {
                const int* out_in_map_ptr_k = slot_row_ptr;
                const float* slot_weight_ci_ptr = slot_weight_ptr + ci_offset * Cout;

                __syncthreads();
                const int* input_idx_ptr = out_in_map_ptr_k;
                #pragma unroll
                for (int ax = 0; ax < 16; ++ax) {
                    const int input_idx = *input_idx_ptr;
                    if (input_idx != -1) {
                        *(reinterpret_cast<float4*>(A_shared_ptr + ax * 256)) =
                            *(reinterpret_cast<const float4*>(
                                p.features + input_idx * Cin + ci_offset + channel_offset_a));
                    } else {
                        *(reinterpret_cast<float4*>(A_shared_ptr + ax * 256)) = make_float4(0.f, 0.f, 0.f, 0.f);
                    }
                    input_idx_ptr += row_pitch;
                }
                const float* B_k_ptr = slot_weight_ci_ptr;
                #pragma unroll
                for (int ax = 0; ax < 2; ++ax) {
                    *(reinterpret_cast<float4*>(B_shared_ptr + ax * 256)) =
                        *(reinterpret_cast<const float4*>(B_k_ptr));
                    B_k_ptr += weight_k_stride;
                }

                __syncthreads();
                #pragma unroll
                for (int k_1 = 0; k_1 < 8; ++k_1) {
                    #pragma unroll
                    for (int k_2 = 0; k_2 < 4; ++k_2) {
                        const int vk = (k_1 << 2) + k_2;
                        #pragma unroll
                        for (int i = 0; i < 32; ++i) {
                            C_local[i] +=
                                A_shared_reduce_ptr[((i / 4) * 16) * 32 + vk]
                                * B_shared_reduce_ptr[(vk * 16) + ((i % 4) * 4)];
                        }
                    }
                }
                ci_offset += CenterLastFP32Setting2Cfg::BK;
            }
        }
        if (compact_slot + 1 < slot_count) {
            const int logical_delta =
                1 + logical_template_boundary_bias_bump(template_id, compact_slot);
            slot_row_ptr += 1;
            slot_weight_ptr += logical_delta * logical_stride;
        }
    }

    #pragma unroll
    for (int i = 0; i < 32; ++i) {
        const int out_row = p.out_rows[location_offset + ((i / 4) * 16)];
        const int col = c_n_offset + ((i % 4) * 4);
        if (out_row >= 0 && col < Cout) {
            p.output[out_row * Cout + col] = C_local[i];
        }
    }
}

template <int K_ld_factor, int N_ld_factor, bool K_ld_check, bool N_ld_check>
__device__ __forceinline__ void center_last_fp32_setting1_tile(
    const CenterLastFP32Params& p,
    int template_id,
    int out_row_base,
    int input_row_offset,
    int bn_base,
    float* A_shared,
    float* B_shared) {
    const int tid = static_cast<int>(threadIdx.x);
    const int Cin = p.c_in;
    const int Cout = p.c_out;
    const int KTile = 16;
    const int KTilePadded = KTile * ((Cin + KTile - 1) / KTile);
    const int loops_per_slot = KTilePadded / KTile;
    const int slot_count = logical_template_slot_count(template_id);
    const int payload_width = logical_template_payload_width(template_id);
    const int total_k_loops = slot_count * loops_per_slot;
    const int row_pitch = payload_width * 16;
    const int logical_stride = Cin * Cout;

    float C_local[32];
    #pragma unroll
    for (int i = 0; i < 32; ++i) {
        C_local[i] = 0.0f;
    }

    float* A_shared_ptr = A_shared + tid * 4;
    float* A_shared_reduce_ptr = A_shared + ((tid / 4) * 16);
    float* B_shared_ptr = B_shared + tid * 4;
    float* B_shared_reduce_ptr = B_shared + (tid % 4);
    const int location_offset = out_row_base + (tid / 4);
    const int c_n_offset = bn_base + (tid % 4);
    const int channel_offset_a = (tid * 4) % 16;
    const int row_local_seed = tid / 4;
    const int* row_seed_ptr =
        logical_input_row_ptr(p, template_id, input_row_offset, row_local_seed);
    const int* slot_row_ptr = row_seed_ptr;

    int A_ld_start = 0;
    int A_ld_amount = 0;
    int A_ld_bound = 0;
    int A_pred_guard = 0;
    int B_ld_start = 0;
    int B_ld_amount = 0;
    int B_ld_bound = 0;
    int B_pred_guard = 0;
    int B_ld_amount_N = 0;
    int B_ld_K_bound = 0;
    bool B_ld_K = false;

    if constexpr (N_ld_check || K_ld_check) {
        B_ld_start = bn_base + ((tid * 4) % 16);
        B_ld_amount_N = max(0, min(B_ld_start + 4, Cout) - B_ld_start);
        B_ld_K_bound = Cin;
    } else {
        B_pred_guard = 1;
    }

    int compact_slot = 0;
    int ci_tile = 0;
    int ci_offset = 0;
    const float* slot_weight_ptr =
        p.weight
        + logical_template_initial_bias(template_id) * logical_stride
        + (tid / 4) * Cout
        + bn_base
        + ((tid * 4) % 16);

    #pragma unroll
    for (int k_0 = 0; k_0 < total_k_loops; ++k_0) {
        const int* out_in_map_ptr_k = slot_row_ptr;
        const float* slot_weight_ci_ptr = slot_weight_ptr + ci_offset * Cout;

        if constexpr (K_ld_check) {
            A_ld_start = ci_offset + channel_offset_a;
            A_ld_amount = max(0, min(A_ld_start + 4, Cin) - A_ld_start);
            A_ld_bound = A_ld_amount / (K_ld_factor / 4);
            A_pred_guard = 0;
            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                if (i < A_ld_bound) {
                    A_pred_guard |= (1 << i);
                }
            }
        } else {
            A_pred_guard = 1;
        }

        if constexpr (K_ld_check || N_ld_check) {
            B_ld_K = (ci_offset + tid * 4 / 16) < B_ld_K_bound;
            B_ld_amount = B_ld_amount_N * static_cast<int>(B_ld_K);
            B_ld_bound = B_ld_amount / (N_ld_factor / 4);
            B_pred_guard = 0;
            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                if (i < B_ld_bound) {
                    B_pred_guard |= (1 << i);
                }
            }
        }

        __syncthreads();
        const int* input_idx_ptr = out_in_map_ptr_k;
        #pragma unroll
        for (int ax = 0; ax < 8; ++ax) {
            const int input_idx = *input_idx_ptr;
            if (input_idx != -1) {
                uint4 A_loaded = make_uint4(0, 0, 0, 0);
                center_last_global_load<K_ld_factor>(
                    A_loaded,
                    p.features + input_idx * Cin + ci_offset + channel_offset_a,
                    A_pred_guard);
                *(reinterpret_cast<uint4*>(A_shared_ptr + ax * 256)) = A_loaded;
            } else {
                *(reinterpret_cast<uint4*>(A_shared_ptr + ax * 256)) = make_uint4(0, 0, 0, 0);
            }
            input_idx_ptr += row_pitch;
        }
        uint4 B_loaded = make_uint4(0, 0, 0, 0);
        center_last_global_load<N_ld_factor>(B_loaded, slot_weight_ci_ptr, B_pred_guard);
        *(reinterpret_cast<uint4*>(B_shared_ptr)) = B_loaded;

        __syncthreads();
        #pragma unroll
        for (int k_1 = 0; k_1 < 4; ++k_1) {
            #pragma unroll
            for (int k_2 = 0; k_2 < 4; ++k_2) {
                const int vk = (k_1 << 2) + k_2;
                #pragma unroll
                for (int i = 0; i < 32; ++i) {
                    C_local[i] +=
                        A_shared_reduce_ptr[((i / 4) * 16) * 16 + vk]
                        * B_shared_reduce_ptr[(vk * 16) + ((i % 4) * 4)];
                }
            }
        }

        ++ci_tile;
        ci_offset += KTile;
        if (ci_tile == loops_per_slot) {
            ci_tile = 0;
            ci_offset = 0;
            const int prev_compact_slot = compact_slot;
            ++compact_slot;
            if (compact_slot < slot_count) {
                const int logical_delta =
                    1 + logical_template_boundary_bias_bump(template_id, prev_compact_slot);
                slot_row_ptr += 1;
                slot_weight_ptr += logical_delta * logical_stride;
            }
        }
    }

    #pragma unroll
    for (int i = 0; i < 32; ++i) {
        const int out_row = p.out_rows[location_offset + ((i / 4) * 16)];
        const int col = c_n_offset + ((i % 4) * 4);
        if (out_row >= 0 && (!N_ld_check || col < Cout)) {
            p.output[out_row * Cout + col] = C_local[i];
        }
    }
}

template <int K_ld_factor, int N_ld_factor, bool K_ld_check, bool N_ld_check>
__device__ __forceinline__ void center_last_fp32_setting1_sorted_tile(
    const CenterLastFP32Params& p,
    int template_id,
    int out_row_base,
    int input_row_offset,
    int bn_base,
    float* A_shared,
    float* B_shared,
    int* slot_active_shared) {
    const int tid = static_cast<int>(threadIdx.x);
    const int Cin = p.c_in;
    const int Cout = p.c_out;
    const int KTile = 16;
    const int KTilePadded = KTile * ((Cin + KTile - 1) / KTile);
    const int loops_per_slot = KTilePadded / KTile;
    const int slot_count = logical_template_slot_count(template_id);
    const int payload_width = logical_template_payload_width(template_id);
    const int row_pitch = payload_width * 16;
    const int logical_stride = Cin * Cout;

    float C_local[32];
    #pragma unroll
    for (int i = 0; i < 32; ++i) {
        C_local[i] = 0.0f;
    }

    float* A_shared_ptr = A_shared + tid * 4;
    float* A_shared_reduce_ptr = A_shared + ((tid / 4) * 16);
    float* B_shared_ptr = B_shared + tid * 4;
    float* B_shared_reduce_ptr = B_shared + (tid % 4);
    const int location_offset = out_row_base + (tid / 4);
    const int c_n_offset = bn_base + (tid % 4);
    const int channel_offset_a = (tid * 4) % 16;
    const int row_local_seed = tid / 4;
    const int* slot_row_ptr =
        logical_input_row_ptr(p, template_id, input_row_offset, row_local_seed);

    int A_ld_start = 0;
    int A_ld_amount = 0;
    int A_ld_bound = 0;
    int A_pred_guard = 0;
    int B_ld_start = 0;
    int B_ld_amount = 0;
    int B_ld_bound = 0;
    int B_pred_guard = 0;
    int B_ld_amount_N = 0;
    int B_ld_K_bound = 0;
    bool B_ld_K = false;

    if constexpr (N_ld_check || K_ld_check) {
        B_ld_start = bn_base + ((tid * 4) % 16);
        B_ld_amount_N = max(0, min(B_ld_start + 4, Cout) - B_ld_start);
        B_ld_K_bound = Cin;
    } else {
        B_pred_guard = 1;
    }

    const float* slot_weight_ptr =
        p.weight
        + logical_template_initial_bias(template_id) * logical_stride
        + (tid / 4) * Cout
        + bn_base
        + ((tid * 4) % 16);

    for (int compact_slot = 0; compact_slot < slot_count; ++compact_slot) {
        if (sorted_tile_compact_slot_has_active<CenterLastFP32Setting1Cfg::THREADS>(
                p,
                template_id,
                input_row_offset,
                compact_slot,
                payload_width,
                slot_active_shared)) {
            int ci_offset = 0;
            for (int ci_tile = 0; ci_tile < loops_per_slot; ++ci_tile) {
                const int* out_in_map_ptr_k = slot_row_ptr;
                const float* slot_weight_ci_ptr = slot_weight_ptr + ci_offset * Cout;

                if constexpr (K_ld_check) {
                    A_ld_start = ci_offset + channel_offset_a;
                    A_ld_amount = max(0, min(A_ld_start + 4, Cin) - A_ld_start);
                    A_ld_bound = A_ld_amount / (K_ld_factor / 4);
                    A_pred_guard = 0;
                    #pragma unroll
                    for (int i = 0; i < 4; ++i) {
                        if (i < A_ld_bound) {
                            A_pred_guard |= (1 << i);
                        }
                    }
                } else {
                    A_pred_guard = 1;
                }

                if constexpr (K_ld_check || N_ld_check) {
                    B_ld_K = (ci_offset + tid * 4 / 16) < B_ld_K_bound;
                    B_ld_amount = B_ld_amount_N * static_cast<int>(B_ld_K);
                    B_ld_bound = B_ld_amount / (N_ld_factor / 4);
                    B_pred_guard = 0;
                    #pragma unroll
                    for (int i = 0; i < 4; ++i) {
                        if (i < B_ld_bound) {
                            B_pred_guard |= (1 << i);
                        }
                    }
                }

                __syncthreads();
                const int* input_idx_ptr = out_in_map_ptr_k;
                #pragma unroll
                for (int ax = 0; ax < 8; ++ax) {
                    const int input_idx = *input_idx_ptr;
                    if (input_idx != -1) {
                        uint4 A_loaded = make_uint4(0, 0, 0, 0);
                        center_last_global_load<K_ld_factor>(
                            A_loaded,
                            p.features + input_idx * Cin + ci_offset + channel_offset_a,
                            A_pred_guard);
                        *(reinterpret_cast<uint4*>(A_shared_ptr + ax * 256)) = A_loaded;
                    } else {
                        *(reinterpret_cast<uint4*>(A_shared_ptr + ax * 256)) = make_uint4(0, 0, 0, 0);
                    }
                    input_idx_ptr += row_pitch;
                }
                uint4 B_loaded = make_uint4(0, 0, 0, 0);
                center_last_global_load<N_ld_factor>(B_loaded, slot_weight_ci_ptr, B_pred_guard);
                *(reinterpret_cast<uint4*>(B_shared_ptr)) = B_loaded;

                __syncthreads();
                #pragma unroll
                for (int k_1 = 0; k_1 < 4; ++k_1) {
                    #pragma unroll
                    for (int k_2 = 0; k_2 < 4; ++k_2) {
                        const int vk = (k_1 << 2) + k_2;
                        #pragma unroll
                        for (int i = 0; i < 32; ++i) {
                            C_local[i] +=
                                A_shared_reduce_ptr[((i / 4) * 16) * 16 + vk]
                                * B_shared_reduce_ptr[(vk * 16) + ((i % 4) * 4)];
                        }
                    }
                }
                ci_offset += KTile;
            }
        }
        if (compact_slot + 1 < slot_count) {
            const int logical_delta =
                1 + logical_template_boundary_bias_bump(template_id, compact_slot);
            slot_row_ptr += 1;
            slot_weight_ptr += logical_delta * logical_stride;
        }
    }

    #pragma unroll
    for (int i = 0; i < 32; ++i) {
        const int out_row = p.out_rows[location_offset + ((i / 4) * 16)];
        const int col = c_n_offset + ((i % 4) * 4);
        if (out_row >= 0 && (!N_ld_check || col < Cout)) {
            p.output[out_row * Cout + col] = C_local[i];
        }
    }
}

__global__ void __launch_bounds__(CenterLastFP32Setting3Cfg::THREADS) center_last_fp32_setting3_kernel(
    CenterLastFP32Params p) {
    __shared__ float A_shared[CenterLastFP32Setting3Cfg::A_SHARED_SIZE];
    __shared__ float B_shared[CenterLastFP32Setting3Cfg::B_SHARED_SIZE];

    const int gnt = (p.c_out + CenterLastFP32Setting3Cfg::BN - 1) / CenterLastFP32Setting3Cfg::BN;
    const int logical = static_cast<int>(blockIdx.x);
    const int row_tile = logical / gnt;
    const int out_row_base = row_tile * CenterLastFP32Setting3Cfg::BM;
    const int bn_base = (logical % gnt) * CenterLastFP32Setting3Cfg::BN;
    const int template_id = p.template_ids[out_row_base];
    if (template_id < 0) {
        return;
    }
    const int input_row_offset = p.input_row_offsets[out_row_base];
    center_last_fp32_setting3_tile(
        p,
        template_id,
        out_row_base,
        input_row_offset,
        bn_base,
        A_shared,
        B_shared);
}

__global__ void __launch_bounds__(CenterLastFP32Setting3Cfg::THREADS) center_last_fp32_setting3_sorted_kernel(
    CenterLastFP32Params p) {
    __shared__ float A_shared[CenterLastFP32Setting3Cfg::A_SHARED_SIZE];
    __shared__ float B_shared[CenterLastFP32Setting3Cfg::B_SHARED_SIZE];
    __shared__ int slot_active_shared;

    const int gnt = (p.c_out + CenterLastFP32Setting3Cfg::BN - 1) / CenterLastFP32Setting3Cfg::BN;
    const int logical = static_cast<int>(blockIdx.x);
    const int row_tile = logical / gnt;
    const int out_row_base = row_tile * CenterLastFP32Setting3Cfg::BM;
    const int bn_base = (logical % gnt) * CenterLastFP32Setting3Cfg::BN;
    const int template_id = p.template_ids[out_row_base];
    if (template_id < 0) {
        return;
    }
    const int input_row_offset = p.input_row_offsets[out_row_base];
    center_last_fp32_setting3_sorted_tile(
        p,
        template_id,
        out_row_base,
        input_row_offset,
        bn_base,
        A_shared,
        B_shared,
        &slot_active_shared);
}

__global__ void __launch_bounds__(CenterLastFP32Setting2Cfg::THREADS) center_last_fp32_setting2_kernel(
    CenterLastFP32Params p) {
    __shared__ float A_shared[CenterLastFP32Setting2Cfg::A_SHARED_SIZE];
    __shared__ float B_shared[CenterLastFP32Setting2Cfg::B_SHARED_SIZE];

    const int gnt = (p.c_out + CenterLastFP32Setting2Cfg::BN - 1) / CenterLastFP32Setting2Cfg::BN;
    const int logical = static_cast<int>(blockIdx.x);
    const int row_tile = logical / gnt;
    const int out_row_base = row_tile * CenterLastFP32Setting2Cfg::BM;
    const int bn_base = (logical % gnt) * CenterLastFP32Setting2Cfg::BN;
    const int template_id = p.template_ids[out_row_base];
    if (template_id < 0) {
        return;
    }
    const int input_row_offset = p.input_row_offsets[out_row_base];
    center_last_fp32_setting2_tile(
        p,
        template_id,
        out_row_base,
        input_row_offset,
        bn_base,
        A_shared,
        B_shared);
}

__global__ void __launch_bounds__(CenterLastFP32Setting2Cfg::THREADS) center_last_fp32_setting2_sorted_kernel(
    CenterLastFP32Params p) {
    __shared__ float A_shared[CenterLastFP32Setting2Cfg::A_SHARED_SIZE];
    __shared__ float B_shared[CenterLastFP32Setting2Cfg::B_SHARED_SIZE];
    __shared__ int slot_active_shared;

    const int gnt = (p.c_out + CenterLastFP32Setting2Cfg::BN - 1) / CenterLastFP32Setting2Cfg::BN;
    const int logical = static_cast<int>(blockIdx.x);
    const int row_tile = logical / gnt;
    const int out_row_base = row_tile * CenterLastFP32Setting2Cfg::BM;
    const int bn_base = (logical % gnt) * CenterLastFP32Setting2Cfg::BN;
    const int template_id = p.template_ids[out_row_base];
    if (template_id < 0) {
        return;
    }
    const int input_row_offset = p.input_row_offsets[out_row_base];
    center_last_fp32_setting2_sorted_tile(
        p,
        template_id,
        out_row_base,
        input_row_offset,
        bn_base,
        A_shared,
        B_shared,
        &slot_active_shared);
}

template <int K_ld_factor, int N_ld_factor, bool K_ld_check, bool N_ld_check>
__global__ void __launch_bounds__(CenterLastFP32Setting1Cfg::THREADS) center_last_fp32_setting1_kernel(
    CenterLastFP32Params p) {
    __shared__ float A_shared[CenterLastFP32Setting1Cfg::A_SHARED_SIZE];
    __shared__ float B_shared[CenterLastFP32Setting1Cfg::B_SHARED_SIZE];

    const int gnt = (p.c_out + CenterLastFP32Setting1Cfg::BN - 1) / CenterLastFP32Setting1Cfg::BN;
    const int logical = static_cast<int>(blockIdx.x);
    const int row_tile = logical / gnt;
    const int out_row_base = row_tile * CenterLastFP32Setting1Cfg::BM;
    const int bn_base = (logical % gnt) * CenterLastFP32Setting1Cfg::BN;
    const int template_id = p.template_ids[out_row_base];
    if (template_id < 0) {
        return;
    }
    const int input_row_offset = p.input_row_offsets[out_row_base];
    center_last_fp32_setting1_tile<K_ld_factor, N_ld_factor, K_ld_check, N_ld_check>(
        p,
        template_id,
        out_row_base,
        input_row_offset,
        bn_base,
        A_shared,
        B_shared);
}

template <int K_ld_factor, int N_ld_factor, bool K_ld_check, bool N_ld_check>
__global__ void __launch_bounds__(CenterLastFP32Setting1Cfg::THREADS) center_last_fp32_setting1_sorted_kernel(
    CenterLastFP32Params p) {
    __shared__ float A_shared[CenterLastFP32Setting1Cfg::A_SHARED_SIZE];
    __shared__ float B_shared[CenterLastFP32Setting1Cfg::B_SHARED_SIZE];
    __shared__ int slot_active_shared;

    const int gnt = (p.c_out + CenterLastFP32Setting1Cfg::BN - 1) / CenterLastFP32Setting1Cfg::BN;
    const int logical = static_cast<int>(blockIdx.x);
    const int row_tile = logical / gnt;
    const int out_row_base = row_tile * CenterLastFP32Setting1Cfg::BM;
    const int bn_base = (logical % gnt) * CenterLastFP32Setting1Cfg::BN;
    const int template_id = p.template_ids[out_row_base];
    if (template_id < 0) {
        return;
    }
    const int input_row_offset = p.input_row_offsets[out_row_base];
    center_last_fp32_setting1_sorted_tile<K_ld_factor, N_ld_factor, K_ld_check, N_ld_check>(
        p,
        template_id,
        out_row_base,
        input_row_offset,
        bn_base,
        A_shared,
        B_shared,
        &slot_active_shared);
}

static void check_center_last_fp32_forward_inputs(
    const torch::Tensor& features,
    const torch::Tensor& logical_weight,
    const torch::Tensor& out_rows,
    const torch::Tensor& input_rows_w1,
    const torch::Tensor& input_rows_w9,
    const torch::Tensor& input_rows_w18,
    const torch::Tensor& input_rows_w27,
    const torch::Tensor& template_ids,
    const torch::Tensor& input_row_offsets,
    int expected_bk,
    int expected_bn,
    bool require_cin_divisible,
    bool require_cout_divisible) {
#ifndef NDEBUG
    TORCH_CHECK(features.is_cuda() && logical_weight.is_cuda());
    TORCH_CHECK(out_rows.is_cuda() && input_rows_w1.is_cuda() && input_rows_w9.is_cuda());
    TORCH_CHECK(input_rows_w18.is_cuda() && input_rows_w27.is_cuda());
    TORCH_CHECK(template_ids.is_cuda() && input_row_offsets.is_cuda());
    TORCH_CHECK(features.dtype() == torch::kFloat32);
    TORCH_CHECK(logical_weight.dtype() == torch::kFloat32);
    TORCH_CHECK(out_rows.dtype() == torch::kInt32);
    TORCH_CHECK(input_rows_w1.dtype() == torch::kInt32);
    TORCH_CHECK(input_rows_w9.dtype() == torch::kInt32);
    TORCH_CHECK(input_rows_w18.dtype() == torch::kInt32);
    TORCH_CHECK(input_rows_w27.dtype() == torch::kInt32);
    TORCH_CHECK(template_ids.dtype() == torch::kInt32);
    TORCH_CHECK(input_row_offsets.dtype() == torch::kInt32);
    TORCH_CHECK(features.dim() == 2);
    TORCH_CHECK(logical_weight.dim() == 2);
    TORCH_CHECK(out_rows.dim() == 1);
    TORCH_CHECK(template_ids.dim() == 1);
    TORCH_CHECK(input_row_offsets.dim() == 1);
    TORCH_CHECK(input_rows_w1.dim() == 3);
    TORCH_CHECK(input_rows_w9.dim() == 3);
    TORCH_CHECK(input_rows_w18.dim() == 3);
    TORCH_CHECK(input_rows_w27.dim() == 3);
    if (require_cin_divisible) {
        TORCH_CHECK(int(features.size(1)) % expected_bk == 0);
    }
    if (require_cout_divisible) {
        TORCH_CHECK(int(logical_weight.size(1)) % expected_bn == 0);
    }
    TORCH_CHECK(int(out_rows.size(0)) == int(template_ids.size(0)));
    TORCH_CHECK(int(out_rows.size(0)) == int(input_row_offsets.size(0)));
    TORCH_CHECK(int(out_rows.size(0)) % kBM == 0);
    TORCH_CHECK(int(input_rows_w1.size(0)) == kFamilyW1Templates);
    TORCH_CHECK(int(input_rows_w9.size(0)) == kFamilyW9Templates);
    TORCH_CHECK(int(input_rows_w18.size(0)) == kFamilyW18Templates);
    TORCH_CHECK(int(input_rows_w27.size(0)) == kFamilyW27Templates);
    TORCH_CHECK(int(input_rows_w1.size(2)) == kPayloadWidthW1);
    TORCH_CHECK(int(input_rows_w9.size(2)) == kPayloadWidthW9);
    TORCH_CHECK(int(input_rows_w18.size(2)) == kPayloadWidthW18);
    TORCH_CHECK(int(input_rows_w27.size(2)) == kPayloadWidthW27);
    TORCH_CHECK(int(input_rows_w1.size(1)) == int(input_rows_w9.size(1)));
    TORCH_CHECK(int(input_rows_w1.size(1)) == int(input_rows_w18.size(1)));
    TORCH_CHECK(int(input_rows_w1.size(1)) == int(input_rows_w27.size(1)));
    TORCH_CHECK(int(logical_weight.size(0)) == kNumLogicalOffsets * int(features.size(1)));
    TORCH_CHECK(features.is_contiguous());
    TORCH_CHECK(logical_weight.is_contiguous());
    TORCH_CHECK(out_rows.is_contiguous());
    TORCH_CHECK(input_rows_w1.is_contiguous());
    TORCH_CHECK(input_rows_w9.is_contiguous());
    TORCH_CHECK(input_rows_w18.is_contiguous());
    TORCH_CHECK(input_rows_w27.is_contiguous());
    TORCH_CHECK(template_ids.is_contiguous());
    TORCH_CHECK(input_row_offsets.is_contiguous());
#else
    (void)features;
    (void)logical_weight;
    (void)out_rows;
    (void)input_rows_w1;
    (void)input_rows_w9;
    (void)input_rows_w18;
    (void)input_rows_w27;
    (void)template_ids;
    (void)input_row_offsets;
    (void)expected_bk;
    (void)expected_bn;
    (void)require_cin_divisible;
    (void)require_cout_divisible;
#endif
}

static CenterLastFP32Params make_center_last_fp32_params(
    const torch::Tensor& features,
    const torch::Tensor& logical_weight,
    const torch::Tensor& out_rows,
    const torch::Tensor& input_rows_w1,
    const torch::Tensor& input_rows_w9,
    const torch::Tensor& input_rows_w18,
    const torch::Tensor& input_rows_w27,
    const torch::Tensor& template_ids,
    const torch::Tensor& input_row_offsets,
    torch::Tensor& output,
    int64_t n_out) {
    CenterLastFP32Params p;
    p.features = features.data_ptr<float>();
    p.weight = logical_weight.data_ptr<float>();
    p.out_rows = out_rows.data_ptr<int>();
    p.input_rows_w1 = input_rows_w1.data_ptr<int>();
    p.input_rows_w9 = input_rows_w9.data_ptr<int>();
    p.input_rows_w18 = input_rows_w18.data_ptr<int>();
    p.input_rows_w27 = input_rows_w27.data_ptr<int>();
    p.template_ids = template_ids.data_ptr<int>();
    p.input_row_offsets = input_row_offsets.data_ptr<int>();
    p.output = output.data_ptr<float>();
    p.n_out = static_cast<int>(n_out);
    p.c_in = static_cast<int>(features.size(1));
    p.c_out = static_cast<int>(logical_weight.size(1));
    p.padded_rows = static_cast<int>(out_rows.size(0));
    p.template_stride = static_cast<int>(input_rows_w1.size(1));
    return p;
}

template <typename Cfg>
static torch::Tensor center_last_fp32_forward_common(
    const torch::Tensor& features,
    const torch::Tensor& logical_weight,
    const torch::Tensor& out_rows,
    const torch::Tensor& input_rows_w1,
    const torch::Tensor& input_rows_w9,
    const torch::Tensor& input_rows_w18,
    const torch::Tensor& input_rows_w27,
    const torch::Tensor& template_ids,
    const torch::Tensor& input_row_offsets,
    int64_t n_out,
    bool require_cin_divisible,
    bool require_cout_divisible,
    void (*kernel_launcher)(CenterLastFP32Params)) {
    c10::cuda::CUDAGuard device_guard(features.device());
    check_center_last_fp32_forward_inputs(
        features,
        logical_weight,
        out_rows,
        input_rows_w1,
        input_rows_w9,
        input_rows_w18,
        input_rows_w27,
        template_ids,
        input_row_offsets,
        Cfg::BK,
        Cfg::BN,
        require_cin_divisible,
        require_cout_divisible);

    const int Cout = static_cast<int>(logical_weight.size(1));
    auto output = torch::empty({n_out, Cout}, features.options());
    if (n_out == 0 || Cout == 0 || out_rows.numel() == 0) {
        return output;
    }

    const auto p = make_center_last_fp32_params(
        features,
        logical_weight,
        out_rows,
        input_rows_w1,
        input_rows_w9,
        input_rows_w18,
        input_rows_w27,
        template_ids,
        input_row_offsets,
        output,
        n_out);
    kernel_launcher(p);
    return output;
}

template <int K_ld_factor, int N_ld_factor, bool K_ld_check, bool N_ld_check>
static void launch_center_last_fp32_setting1_variant(CenterLastFP32Params p) {
    const int gnt = (p.c_out + CenterLastFP32Setting1Cfg::BN - 1) / CenterLastFP32Setting1Cfg::BN;
    const int grid = (p.padded_rows / CenterLastFP32Setting1Cfg::BM) * gnt;
    center_last_fp32_setting1_kernel<K_ld_factor, N_ld_factor, K_ld_check, N_ld_check><<<
        grid,
        CenterLastFP32Setting1Cfg::THREADS,
        0,
        at::cuda::getCurrentCUDAStream()>>>(p);
}

template <int K_ld_factor, int N_ld_factor, bool K_ld_check, bool N_ld_check>
static void launch_center_last_fp32_setting1_sorted_variant(CenterLastFP32Params p) {
    const int gnt = (p.c_out + CenterLastFP32Setting1Cfg::BN - 1) / CenterLastFP32Setting1Cfg::BN;
    const int grid = (p.padded_rows / CenterLastFP32Setting1Cfg::BM) * gnt;
    center_last_fp32_setting1_sorted_kernel<K_ld_factor, N_ld_factor, K_ld_check, N_ld_check><<<
        grid,
        CenterLastFP32Setting1Cfg::THREADS,
        0,
        at::cuda::getCurrentCUDAStream()>>>(p);
}

static void launch_center_last_fp32_setting2(CenterLastFP32Params p) {
    const int gnt = (p.c_out + CenterLastFP32Setting2Cfg::BN - 1) / CenterLastFP32Setting2Cfg::BN;
    const int grid = (p.padded_rows / CenterLastFP32Setting2Cfg::BM) * gnt;
    center_last_fp32_setting2_kernel<<<grid, CenterLastFP32Setting2Cfg::THREADS, 0, at::cuda::getCurrentCUDAStream()>>>(p);
}

static void launch_center_last_fp32_setting2_sorted(CenterLastFP32Params p) {
    const int gnt = (p.c_out + CenterLastFP32Setting2Cfg::BN - 1) / CenterLastFP32Setting2Cfg::BN;
    const int grid = (p.padded_rows / CenterLastFP32Setting2Cfg::BM) * gnt;
    center_last_fp32_setting2_sorted_kernel<<<grid, CenterLastFP32Setting2Cfg::THREADS, 0, at::cuda::getCurrentCUDAStream()>>>(p);
}

static void launch_center_last_fp32_setting3(CenterLastFP32Params p) {
    const int gnt = (p.c_out + CenterLastFP32Setting3Cfg::BN - 1) / CenterLastFP32Setting3Cfg::BN;
    const int grid = (p.padded_rows / CenterLastFP32Setting3Cfg::BM) * gnt;
    center_last_fp32_setting3_kernel<<<grid, CenterLastFP32Setting3Cfg::THREADS, 0, at::cuda::getCurrentCUDAStream()>>>(p);
}

static void launch_center_last_fp32_setting3_sorted(CenterLastFP32Params p) {
    const int gnt = (p.c_out + CenterLastFP32Setting3Cfg::BN - 1) / CenterLastFP32Setting3Cfg::BN;
    const int grid = (p.padded_rows / CenterLastFP32Setting3Cfg::BM) * gnt;
    center_last_fp32_setting3_sorted_kernel<<<grid, CenterLastFP32Setting3Cfg::THREADS, 0, at::cuda::getCurrentCUDAStream()>>>(p);
}

static torch::Tensor center_last_fp32_setting1_forward_dispatch(
    const torch::Tensor& features,
    const torch::Tensor& logical_weight,
    const torch::Tensor& out_rows,
    const torch::Tensor& input_rows_w1,
    const torch::Tensor& input_rows_w9,
    const torch::Tensor& input_rows_w18,
    const torch::Tensor& input_rows_w27,
    const torch::Tensor& template_ids,
    const torch::Tensor& input_row_offsets,
    int64_t n_out) {
    const int Cin = static_cast<int>(features.size(1));
    const int Cout = static_cast<int>(logical_weight.size(1));
    if (Cin % 16 == 0) {
        if (Cout % 16 == 0) {
            return center_last_fp32_forward_common<CenterLastFP32Setting1Cfg>(features, logical_weight, out_rows, input_rows_w1, input_rows_w9, input_rows_w18, input_rows_w27, template_ids, input_row_offsets, n_out, false, false, &launch_center_last_fp32_setting1_variant<16, 16, false, false>);
        }
        if (Cout % 4 == 0) {
            return center_last_fp32_forward_common<CenterLastFP32Setting1Cfg>(features, logical_weight, out_rows, input_rows_w1, input_rows_w9, input_rows_w18, input_rows_w27, template_ids, input_row_offsets, n_out, false, false, &launch_center_last_fp32_setting1_variant<16, 16, false, true>);
        }
        if (Cout % 2 == 0) {
            return center_last_fp32_forward_common<CenterLastFP32Setting1Cfg>(features, logical_weight, out_rows, input_rows_w1, input_rows_w9, input_rows_w18, input_rows_w27, template_ids, input_row_offsets, n_out, false, false, &launch_center_last_fp32_setting1_variant<16, 8, false, true>);
        }
        return center_last_fp32_forward_common<CenterLastFP32Setting1Cfg>(features, logical_weight, out_rows, input_rows_w1, input_rows_w9, input_rows_w18, input_rows_w27, template_ids, input_row_offsets, n_out, false, false, &launch_center_last_fp32_setting1_variant<16, 4, false, true>);
    }
    if (Cin % 4 == 0) {
        if (Cout % 16 == 0) {
            return center_last_fp32_forward_common<CenterLastFP32Setting1Cfg>(features, logical_weight, out_rows, input_rows_w1, input_rows_w9, input_rows_w18, input_rows_w27, template_ids, input_row_offsets, n_out, false, false, &launch_center_last_fp32_setting1_variant<16, 16, true, false>);
        }
        if (Cout % 4 == 0) {
            return center_last_fp32_forward_common<CenterLastFP32Setting1Cfg>(features, logical_weight, out_rows, input_rows_w1, input_rows_w9, input_rows_w18, input_rows_w27, template_ids, input_row_offsets, n_out, false, false, &launch_center_last_fp32_setting1_variant<16, 16, true, true>);
        }
        if (Cout % 2 == 0) {
            return center_last_fp32_forward_common<CenterLastFP32Setting1Cfg>(features, logical_weight, out_rows, input_rows_w1, input_rows_w9, input_rows_w18, input_rows_w27, template_ids, input_row_offsets, n_out, false, false, &launch_center_last_fp32_setting1_variant<16, 8, true, true>);
        }
        return center_last_fp32_forward_common<CenterLastFP32Setting1Cfg>(features, logical_weight, out_rows, input_rows_w1, input_rows_w9, input_rows_w18, input_rows_w27, template_ids, input_row_offsets, n_out, false, false, &launch_center_last_fp32_setting1_variant<16, 4, true, true>);
    }
    if (Cin % 2 == 0) {
        if (Cout % 16 == 0) {
            return center_last_fp32_forward_common<CenterLastFP32Setting1Cfg>(features, logical_weight, out_rows, input_rows_w1, input_rows_w9, input_rows_w18, input_rows_w27, template_ids, input_row_offsets, n_out, false, false, &launch_center_last_fp32_setting1_variant<8, 16, true, false>);
        }
        if (Cout % 4 == 0) {
            return center_last_fp32_forward_common<CenterLastFP32Setting1Cfg>(features, logical_weight, out_rows, input_rows_w1, input_rows_w9, input_rows_w18, input_rows_w27, template_ids, input_row_offsets, n_out, false, false, &launch_center_last_fp32_setting1_variant<8, 16, true, true>);
        }
        if (Cout % 2 == 0) {
            return center_last_fp32_forward_common<CenterLastFP32Setting1Cfg>(features, logical_weight, out_rows, input_rows_w1, input_rows_w9, input_rows_w18, input_rows_w27, template_ids, input_row_offsets, n_out, false, false, &launch_center_last_fp32_setting1_variant<8, 8, true, true>);
        }
        return center_last_fp32_forward_common<CenterLastFP32Setting1Cfg>(features, logical_weight, out_rows, input_rows_w1, input_rows_w9, input_rows_w18, input_rows_w27, template_ids, input_row_offsets, n_out, false, false, &launch_center_last_fp32_setting1_variant<8, 4, true, true>);
    }
    if (Cout % 16 == 0) {
        return center_last_fp32_forward_common<CenterLastFP32Setting1Cfg>(features, logical_weight, out_rows, input_rows_w1, input_rows_w9, input_rows_w18, input_rows_w27, template_ids, input_row_offsets, n_out, false, false, &launch_center_last_fp32_setting1_variant<4, 16, true, false>);
    }
    if (Cout % 4 == 0) {
        return center_last_fp32_forward_common<CenterLastFP32Setting1Cfg>(features, logical_weight, out_rows, input_rows_w1, input_rows_w9, input_rows_w18, input_rows_w27, template_ids, input_row_offsets, n_out, false, false, &launch_center_last_fp32_setting1_variant<4, 16, true, true>);
    }
    if (Cout % 2 == 0) {
        return center_last_fp32_forward_common<CenterLastFP32Setting1Cfg>(features, logical_weight, out_rows, input_rows_w1, input_rows_w9, input_rows_w18, input_rows_w27, template_ids, input_row_offsets, n_out, false, false, &launch_center_last_fp32_setting1_variant<4, 8, true, true>);
    }
    return center_last_fp32_forward_common<CenterLastFP32Setting1Cfg>(features, logical_weight, out_rows, input_rows_w1, input_rows_w9, input_rows_w18, input_rows_w27, template_ids, input_row_offsets, n_out, false, false, &launch_center_last_fp32_setting1_variant<4, 4, true, true>);
}

static torch::Tensor center_last_fp32_setting1_sorted_forward_dispatch(
    const torch::Tensor& features,
    const torch::Tensor& logical_weight,
    const torch::Tensor& out_rows,
    const torch::Tensor& input_rows_w1,
    const torch::Tensor& input_rows_w9,
    const torch::Tensor& input_rows_w18,
    const torch::Tensor& input_rows_w27,
    const torch::Tensor& template_ids,
    const torch::Tensor& input_row_offsets,
    int64_t n_out) {
    const int Cin = static_cast<int>(features.size(1));
    const int Cout = static_cast<int>(logical_weight.size(1));
    if (Cin % 16 == 0) {
        if (Cout % 16 == 0) {
            return center_last_fp32_forward_common<CenterLastFP32Setting1Cfg>(features, logical_weight, out_rows, input_rows_w1, input_rows_w9, input_rows_w18, input_rows_w27, template_ids, input_row_offsets, n_out, false, false, &launch_center_last_fp32_setting1_sorted_variant<16, 16, false, false>);
        }
        if (Cout % 4 == 0) {
            return center_last_fp32_forward_common<CenterLastFP32Setting1Cfg>(features, logical_weight, out_rows, input_rows_w1, input_rows_w9, input_rows_w18, input_rows_w27, template_ids, input_row_offsets, n_out, false, false, &launch_center_last_fp32_setting1_sorted_variant<16, 16, false, true>);
        }
        if (Cout % 2 == 0) {
            return center_last_fp32_forward_common<CenterLastFP32Setting1Cfg>(features, logical_weight, out_rows, input_rows_w1, input_rows_w9, input_rows_w18, input_rows_w27, template_ids, input_row_offsets, n_out, false, false, &launch_center_last_fp32_setting1_sorted_variant<16, 8, false, true>);
        }
        return center_last_fp32_forward_common<CenterLastFP32Setting1Cfg>(features, logical_weight, out_rows, input_rows_w1, input_rows_w9, input_rows_w18, input_rows_w27, template_ids, input_row_offsets, n_out, false, false, &launch_center_last_fp32_setting1_sorted_variant<16, 4, false, true>);
    }
    if (Cin % 4 == 0) {
        if (Cout % 16 == 0) {
            return center_last_fp32_forward_common<CenterLastFP32Setting1Cfg>(features, logical_weight, out_rows, input_rows_w1, input_rows_w9, input_rows_w18, input_rows_w27, template_ids, input_row_offsets, n_out, false, false, &launch_center_last_fp32_setting1_sorted_variant<16, 16, true, false>);
        }
        if (Cout % 4 == 0) {
            return center_last_fp32_forward_common<CenterLastFP32Setting1Cfg>(features, logical_weight, out_rows, input_rows_w1, input_rows_w9, input_rows_w18, input_rows_w27, template_ids, input_row_offsets, n_out, false, false, &launch_center_last_fp32_setting1_sorted_variant<16, 16, true, true>);
        }
        if (Cout % 2 == 0) {
            return center_last_fp32_forward_common<CenterLastFP32Setting1Cfg>(features, logical_weight, out_rows, input_rows_w1, input_rows_w9, input_rows_w18, input_rows_w27, template_ids, input_row_offsets, n_out, false, false, &launch_center_last_fp32_setting1_sorted_variant<16, 8, true, true>);
        }
        return center_last_fp32_forward_common<CenterLastFP32Setting1Cfg>(features, logical_weight, out_rows, input_rows_w1, input_rows_w9, input_rows_w18, input_rows_w27, template_ids, input_row_offsets, n_out, false, false, &launch_center_last_fp32_setting1_sorted_variant<16, 4, true, true>);
    }
    if (Cin % 2 == 0) {
        if (Cout % 16 == 0) {
            return center_last_fp32_forward_common<CenterLastFP32Setting1Cfg>(features, logical_weight, out_rows, input_rows_w1, input_rows_w9, input_rows_w18, input_rows_w27, template_ids, input_row_offsets, n_out, false, false, &launch_center_last_fp32_setting1_sorted_variant<8, 16, true, false>);
        }
        if (Cout % 4 == 0) {
            return center_last_fp32_forward_common<CenterLastFP32Setting1Cfg>(features, logical_weight, out_rows, input_rows_w1, input_rows_w9, input_rows_w18, input_rows_w27, template_ids, input_row_offsets, n_out, false, false, &launch_center_last_fp32_setting1_sorted_variant<8, 16, true, true>);
        }
        if (Cout % 2 == 0) {
            return center_last_fp32_forward_common<CenterLastFP32Setting1Cfg>(features, logical_weight, out_rows, input_rows_w1, input_rows_w9, input_rows_w18, input_rows_w27, template_ids, input_row_offsets, n_out, false, false, &launch_center_last_fp32_setting1_sorted_variant<8, 8, true, true>);
        }
        return center_last_fp32_forward_common<CenterLastFP32Setting1Cfg>(features, logical_weight, out_rows, input_rows_w1, input_rows_w9, input_rows_w18, input_rows_w27, template_ids, input_row_offsets, n_out, false, false, &launch_center_last_fp32_setting1_sorted_variant<8, 4, true, true>);
    }
    if (Cout % 16 == 0) {
        return center_last_fp32_forward_common<CenterLastFP32Setting1Cfg>(features, logical_weight, out_rows, input_rows_w1, input_rows_w9, input_rows_w18, input_rows_w27, template_ids, input_row_offsets, n_out, false, false, &launch_center_last_fp32_setting1_sorted_variant<4, 16, true, false>);
    }
    if (Cout % 4 == 0) {
        return center_last_fp32_forward_common<CenterLastFP32Setting1Cfg>(features, logical_weight, out_rows, input_rows_w1, input_rows_w9, input_rows_w18, input_rows_w27, template_ids, input_row_offsets, n_out, false, false, &launch_center_last_fp32_setting1_sorted_variant<4, 16, true, true>);
    }
    if (Cout % 2 == 0) {
        return center_last_fp32_forward_common<CenterLastFP32Setting1Cfg>(features, logical_weight, out_rows, input_rows_w1, input_rows_w9, input_rows_w18, input_rows_w27, template_ids, input_row_offsets, n_out, false, false, &launch_center_last_fp32_setting1_sorted_variant<4, 8, true, true>);
    }
    return center_last_fp32_forward_common<CenterLastFP32Setting1Cfg>(features, logical_weight, out_rows, input_rows_w1, input_rows_w9, input_rows_w18, input_rows_w27, template_ids, input_row_offsets, n_out, false, false, &launch_center_last_fp32_setting1_sorted_variant<4, 4, true, true>);
}

}  // namespace

torch::Tensor finalize_row_template_center_last_fp32_setting1_forward(
    torch::Tensor features,
    torch::Tensor logical_weight,
    torch::Tensor out_rows,
    torch::Tensor input_rows_w1,
    torch::Tensor input_rows_w9,
    torch::Tensor input_rows_w18,
    torch::Tensor input_rows_w27,
    torch::Tensor template_ids,
    torch::Tensor input_row_offsets,
    int64_t n_out) {
    return center_last_fp32_setting1_forward_dispatch(
        features,
        logical_weight,
        out_rows,
        input_rows_w1,
        input_rows_w9,
        input_rows_w18,
        input_rows_w27,
        template_ids,
        input_row_offsets,
        n_out);
}

torch::Tensor finalize_row_template_center_last_fp32_setting1_sorted_forward(
    torch::Tensor features,
    torch::Tensor logical_weight,
    torch::Tensor out_rows,
    torch::Tensor input_rows_w1,
    torch::Tensor input_rows_w9,
    torch::Tensor input_rows_w18,
    torch::Tensor input_rows_w27,
    torch::Tensor template_ids,
    torch::Tensor input_row_offsets,
    int64_t n_out) {
    return center_last_fp32_setting1_sorted_forward_dispatch(
        features,
        logical_weight,
        out_rows,
        input_rows_w1,
        input_rows_w9,
        input_rows_w18,
        input_rows_w27,
        template_ids,
        input_row_offsets,
        n_out);
}

torch::Tensor finalize_row_template_center_last_fp32_setting2_forward(
    torch::Tensor features,
    torch::Tensor logical_weight,
    torch::Tensor out_rows,
    torch::Tensor input_rows_w1,
    torch::Tensor input_rows_w9,
    torch::Tensor input_rows_w18,
    torch::Tensor input_rows_w27,
    torch::Tensor template_ids,
    torch::Tensor input_row_offsets,
    int64_t n_out) {
    return center_last_fp32_forward_common<CenterLastFP32Setting2Cfg>(
        features,
        logical_weight,
        out_rows,
        input_rows_w1,
        input_rows_w9,
        input_rows_w18,
        input_rows_w27,
        template_ids,
        input_row_offsets,
        n_out,
        true,
        true,
        &launch_center_last_fp32_setting2);
}

torch::Tensor finalize_row_template_center_last_fp32_setting2_sorted_forward(
    torch::Tensor features,
    torch::Tensor logical_weight,
    torch::Tensor out_rows,
    torch::Tensor input_rows_w1,
    torch::Tensor input_rows_w9,
    torch::Tensor input_rows_w18,
    torch::Tensor input_rows_w27,
    torch::Tensor template_ids,
    torch::Tensor input_row_offsets,
    int64_t n_out) {
    return center_last_fp32_forward_common<CenterLastFP32Setting2Cfg>(
        features,
        logical_weight,
        out_rows,
        input_rows_w1,
        input_rows_w9,
        input_rows_w18,
        input_rows_w27,
        template_ids,
        input_row_offsets,
        n_out,
        true,
        true,
        &launch_center_last_fp32_setting2_sorted);
}

torch::Tensor finalize_row_template_center_last_fp32_setting3_forward(
    torch::Tensor features,
    torch::Tensor logical_weight,
    torch::Tensor out_rows,
    torch::Tensor input_rows_w1,
    torch::Tensor input_rows_w9,
    torch::Tensor input_rows_w18,
    torch::Tensor input_rows_w27,
    torch::Tensor template_ids,
    torch::Tensor input_row_offsets,
    int64_t n_out) {
    return center_last_fp32_forward_common<CenterLastFP32Setting3Cfg>(
        features,
        logical_weight,
        out_rows,
        input_rows_w1,
        input_rows_w9,
        input_rows_w18,
        input_rows_w27,
        template_ids,
        input_row_offsets,
        n_out,
        true,
        true,
        &launch_center_last_fp32_setting3);
}

torch::Tensor finalize_row_template_center_last_fp32_setting3_sorted_forward(
    torch::Tensor features,
    torch::Tensor logical_weight,
    torch::Tensor out_rows,
    torch::Tensor input_rows_w1,
    torch::Tensor input_rows_w9,
    torch::Tensor input_rows_w18,
    torch::Tensor input_rows_w27,
    torch::Tensor template_ids,
    torch::Tensor input_row_offsets,
    int64_t n_out) {
    return center_last_fp32_forward_common<CenterLastFP32Setting3Cfg>(
        features,
        logical_weight,
        out_rows,
        input_rows_w1,
        input_rows_w9,
        input_rows_w18,
        input_rows_w27,
        template_ids,
        input_row_offsets,
        n_out,
        true,
        true,
        &launch_center_last_fp32_setting3_sorted);
}

torch::Tensor finalize_row_template_center_last_fp32_forward(
    torch::Tensor features,
    torch::Tensor logical_weight,
    torch::Tensor out_rows,
    torch::Tensor input_rows_w1,
    torch::Tensor input_rows_w9,
    torch::Tensor input_rows_w18,
    torch::Tensor input_rows_w27,
    torch::Tensor template_ids,
    torch::Tensor input_row_offsets,
    int64_t n_out) {
    const int Cin = static_cast<int>(features.size(1));
    const int Cout = static_cast<int>(logical_weight.size(1));
    if (Cin % 32 == 0 && Cout % 64 == 0) {
        return finalize_row_template_center_last_fp32_setting3_forward(
            features,
            logical_weight,
            out_rows,
            input_rows_w1,
            input_rows_w9,
            input_rows_w18,
            input_rows_w27,
            template_ids,
            input_row_offsets,
            n_out);
    }
    return finalize_row_template_center_last_fp32_setting1_forward(
        features,
        logical_weight,
        out_rows,
        input_rows_w1,
        input_rows_w9,
        input_rows_w18,
        input_rows_w27,
        template_ids,
        input_row_offsets,
        n_out);
}

torch::Tensor finalize_row_template_center_last_fp32_sorted_forward(
    torch::Tensor features,
    torch::Tensor logical_weight,
    torch::Tensor out_rows,
    torch::Tensor input_rows_w1,
    torch::Tensor input_rows_w9,
    torch::Tensor input_rows_w18,
    torch::Tensor input_rows_w27,
    torch::Tensor template_ids,
    torch::Tensor input_row_offsets,
    int64_t n_out) {
    const int Cin = static_cast<int>(features.size(1));
    const int Cout = static_cast<int>(logical_weight.size(1));
    if (Cin % 32 == 0 && Cout % 64 == 0) {
        return finalize_row_template_center_last_fp32_setting3_sorted_forward(
            features,
            logical_weight,
            out_rows,
            input_rows_w1,
            input_rows_w9,
            input_rows_w18,
            input_rows_w27,
            template_ids,
            input_row_offsets,
            n_out);
    }
    return finalize_row_template_center_last_fp32_setting1_sorted_forward(
        features,
        logical_weight,
        out_rows,
        input_rows_w1,
        input_rows_w9,
        input_rows_w18,
        input_rows_w27,
        template_ids,
        input_row_offsets,
        n_out);
}
