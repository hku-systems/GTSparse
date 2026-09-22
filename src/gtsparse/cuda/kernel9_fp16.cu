#include "api.h"
#include "contract_k9.h"

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace {

using namespace gtsparse_kernel9;

struct Kernel9FP16Params {
    const half* __restrict__ features;
    const half* __restrict__ weight;
    const int* __restrict__ out_rows;
    const int* __restrict__ input_rows_w1;
    const int* __restrict__ input_rows_w4;
    const int* __restrict__ input_rows_w7;
    const int* __restrict__ input_rows_w9;
    const int* __restrict__ template_ids;
    const int* __restrict__ input_row_offsets;
    half* __restrict__ output;
    int c_in;
    int c_out;
    int padded_rows;
    int template_stride;
};

constexpr int kBN = 64;
constexpr int kBK = 32;
constexpr int kThreads = 128;
constexpr int kASharedSize = 5120;
constexpr int kBSharedSize = 2304;

__device__ __forceinline__ int template_slot_count(int template_id) {
    switch (template_id) {
        case kTemplateCenter: return 1;
        case kTemplateSkip2Keep0: return 4;
        case kTemplateSkip2Keep1: return 3;
        case kTemplateSkip2Keep2: return 4;
        case kTemplateSkip1Hole0: return 6;
        case kTemplateSkip1Hole1: return 7;
        case kTemplateSkip1Hole2: return 6;
        default: return 9;
    }
}

__device__ __forceinline__ int template_payload_width(int template_id) {
    if (template_id == kTemplateCenter) return kPayloadWidthW1;
    if (template_id <= kTemplateSkip2Keep2) return kPayloadWidthW4;
    if (template_id <= kTemplateSkip1Hole2) return kPayloadWidthW7;
    return kPayloadWidthW9;
}

__device__ __forceinline__ int template_initial_offset(int template_id) {
    switch (template_id) {
        case kTemplateCenter: return 4;
        case kTemplateSkip2Keep1:
        case kTemplateSkip1Hole0: return 1;
        case kTemplateSkip2Keep2: return 2;
        default: return 0;
    }
}

__device__ __forceinline__ int template_boundary_bump(int template_id, int slot) {
    if (template_id == kTemplateSkip2Keep0) return slot == 0 ? 2 : slot == 1 ? 0 : 1;
    if (template_id == kTemplateSkip2Keep1) return 2;
    if (template_id == kTemplateSkip2Keep2) return slot == 0 ? 1 : slot == 1 ? 0 : 2;
    if (template_id == kTemplateSkip1Hole0) return slot == 1 || slot == 3 ? 1 : 0;
    if (template_id == kTemplateSkip1Hole1) return slot == 0 || slot == 5 ? 1 : 0;
    if (template_id == kTemplateSkip1Hole2) return slot == 1 || slot == 3 ? 1 : 0;
    return 0;
}

__device__ __forceinline__ const int* input_row_ptr(
    const Kernel9FP16Params& p,
    int template_id,
    int input_row_offset,
    int row_local_seed) {
    if (template_id == kTemplateCenter) {
        return p.input_rows_w1 + (input_row_offset + row_local_seed) * kPayloadWidthW1;
    }
    if (template_id <= kTemplateSkip2Keep2) {
        return p.input_rows_w4
            + (((template_id - kTemplateSkip2Keep0) * p.template_stride)
               + input_row_offset + row_local_seed) * kPayloadWidthW4;
    }
    if (template_id <= kTemplateSkip1Hole2) {
        return p.input_rows_w7
            + (((template_id - kTemplateSkip1Hole0) * p.template_stride)
               + input_row_offset + row_local_seed) * kPayloadWidthW7;
    }
    return p.input_rows_w9 + (input_row_offset + row_local_seed) * kPayloadWidthW9;
}

__device__ __forceinline__ void mma_m16n8k16(
    float* accum,
    const half* a_fragment_half,
    const half* b_fragment_half) {
#if __CUDA_ARCH__ >= 800
    const unsigned* a = reinterpret_cast<const unsigned*>(a_fragment_half);
    const unsigned* b = reinterpret_cast<const unsigned*>(b_fragment_half);
    __asm__ __volatile__(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"
        "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};"
        : "=f"(accum[0]), "=f"(accum[1]), "=f"(accum[2]), "=f"(accum[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]),
          "f"(accum[0]), "f"(accum[1]), "f"(accum[2]), "f"(accum[3]));
    __asm__ __volatile__(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"
        "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};"
        : "=f"(accum[4]), "=f"(accum[5]), "=f"(accum[6]), "=f"(accum[7])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[2]), "r"(b[3]),
          "f"(accum[4]), "f"(accum[5]), "f"(accum[6]), "f"(accum[7]));
#elif __CUDA_ARCH__ >= 750
    const unsigned* a0 = reinterpret_cast<const unsigned*>(a_fragment_half);
    const unsigned* a1 = reinterpret_cast<const unsigned*>(a_fragment_half + 4);
    const unsigned* b0 = reinterpret_cast<const unsigned*>(b_fragment_half);
    const unsigned* b4 = reinterpret_cast<const unsigned*>(b_fragment_half + 4);
    const unsigned* b2 = reinterpret_cast<const unsigned*>(b_fragment_half + 2);
    const unsigned* b6 = reinterpret_cast<const unsigned*>(b_fragment_half + 6);
    __asm__ __volatile__(
        "mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32"
        "{%0, %1, %2, %3}, {%4, %5}, {%6}, {%7, %8, %9, %10};"
        : "=f"(accum[0]), "=f"(accum[1]), "=f"(accum[2]), "=f"(accum[3])
        : "r"(a0[0]), "r"(a0[1]), "r"(b0[0]),
          "f"(accum[0]), "f"(accum[1]), "f"(accum[2]), "f"(accum[3]));
    __asm__ __volatile__(
        "mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32"
        "{%0, %1, %2, %3}, {%4, %5}, {%6}, {%7, %8, %9, %10};"
        : "=f"(accum[4]), "=f"(accum[5]), "=f"(accum[6]), "=f"(accum[7])
        : "r"(a0[0]), "r"(a0[1]), "r"(b4[0]),
          "f"(accum[4]), "f"(accum[5]), "f"(accum[6]), "f"(accum[7]));
    __asm__ __volatile__(
        "mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32"
        "{%0, %1, %2, %3}, {%4, %5}, {%6}, {%7, %8, %9, %10};"
        : "=f"(accum[0]), "=f"(accum[1]), "=f"(accum[2]), "=f"(accum[3])
        : "r"(a1[0]), "r"(a1[1]), "r"(b2[0]),
          "f"(accum[0]), "f"(accum[1]), "f"(accum[2]), "f"(accum[3]));
    __asm__ __volatile__(
        "mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32"
        "{%0, %1, %2, %3}, {%4, %5}, {%6}, {%7, %8, %9, %10};"
        : "=f"(accum[4]), "=f"(accum[5]), "=f"(accum[6]), "=f"(accum[7])
        : "r"(a1[0]), "r"(a1[1]), "r"(b6[0]),
          "f"(accum[4]), "f"(accum[5]), "f"(accum[6]), "f"(accum[7]));
#endif
}

__device__ __forceinline__ void kernel9_fp16_tile(
    const Kernel9FP16Params& p,
    int template_id,
    int out_row_base,
    int input_row_offset,
    int bn_base,
    half* a_shared,
    half* b_shared) {
    const int tid = threadIdx.x;
    const int thread_x = tid & 31;
    const int thread_y = tid >> 5;
    const int cin = p.c_in;
    const int cout = p.c_out;
    const int loops_per_slot = cin / kBK;
    const int slot_count = template_slot_count(template_id);
    const int payload_width = template_payload_width(template_id);
    const int row_pitch = payload_width * 32;
    const int logical_stride = cin * cout;

    float accum[64];
    half a_fragment[32];
    half b_fragment[16];
    #pragma unroll
    for (int index = 0; index < 64; ++index) accum[index] = 0.0f;

    const int row_local_seed = thread_y * 8 + thread_x / 4;
    const int* slot_row_ptr = input_row_ptr(p, template_id, input_row_offset, row_local_seed);
    const half* slot_weight_ptr =
        p.weight + template_initial_offset(template_id) * logical_stride + bn_base
        + (((thread_y << 2) + (thread_x >> 3)) * cout) + ((thread_x & 7) * 8);
    const half* feature_ptr = p.features + ((thread_x & 3) * 8);
    const int reorder_offset = (thread_y % 2) * 64 + (thread_x / 4);
    const int channel_base = bn_base + thread_y / 2 * 32 + (thread_x % 4) * 2;

    int compact_slot = 0;
    int ci_tile = 0;
    int ci_offset = 0;
    const int total_k_loops = slot_count * loops_per_slot;
    for (int k_loop = 0; k_loop < total_k_loops; ++k_loop) {
        const half* feature_ci_ptr = feature_ptr + ci_offset;
        const half* weight_ci_ptr = slot_weight_ptr + ci_offset * cout;
        __syncthreads();
        #pragma unroll
        for (int row = 0; row < 4; ++row) {
            half* dst = a_shared + row * 1280 + thread_y * 320
                + (thread_x >> 2) * 40 + (thread_x & 3) * 8;
            const int input_idx = slot_row_ptr[row * row_pitch];
            if (input_idx >= 0) {
                *reinterpret_cast<uint4*>(dst) =
                    *reinterpret_cast<const uint4*>(feature_ci_ptr + input_idx * cin);
            } else {
                *reinterpret_cast<uint4*>(dst) = make_uint4(0, 0, 0, 0);
            }
        }
        #pragma unroll
        for (int row = 0; row < 2; ++row) {
            half* dst = b_shared + row * 1152 + thread_y * 288
                + (thread_x >> 3) * 72 + (thread_x & 7) * 8;
            *reinterpret_cast<uint4*>(dst) =
                *reinterpret_cast<const uint4*>(weight_ci_ptr + row * 16 * cout);
        }
        __syncthreads();

        #pragma unroll
        for (int k_half = 0; k_half < 2; ++k_half) {
            #pragma unroll
            for (int m_tile = 0; m_tile < 4; ++m_tile) {
                unsigned int address;
                __asm__ __volatile__(
                    "{ .reg .u64 addr; cvta.to.shared.u64 addr, %1; cvt.u32.u64 %0, addr; }"
                    : "=r"(address)
                    : "l"((void*)((a_shared + ((thread_y & 1) * 2560 + m_tile * 640 + k_half * 16))
                                  + ((thread_x & 15) * 40 + (thread_x >> 4) * 8))));
#if __CUDA_ARCH__ >= 750
                __asm__ __volatile__(
                    "ldmatrix.sync.aligned.m8n8.x4.shared.b16"
                    "{%0, %1, %2, %3}, [%4];"
                    : "=r"(((unsigned*)(a_fragment + m_tile * 8))[0]),
                      "=r"(((unsigned*)(a_fragment + m_tile * 8))[1]),
                      "=r"(((unsigned*)(a_fragment + m_tile * 8))[2]),
                      "=r"(((unsigned*)(a_fragment + m_tile * 8))[3])
                    : "r"(address));
#endif
            }
            #pragma unroll
            for (int n_pair = 0; n_pair < 2; ++n_pair) {
                unsigned int address;
                __asm__ __volatile__(
                    "{ .reg .u64 addr; cvta.to.shared.u64 addr, %1; cvt.u32.u64 %0, addr; }"
                    : "=r"(address)
                    : "l"((void*)((b_shared + (k_half * 1152 + (thread_y >> 1) * 32 + n_pair * 16))
                                  + ((thread_x & 15) * 72 + (thread_x >> 4) * 8))));
#if __CUDA_ARCH__ >= 750
                __asm__ __volatile__(
                    "ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16"
                    "{%0, %1, %2, %3}, [%4];"
                    : "=r"(((unsigned*)(b_fragment + n_pair * 8))[0]),
                      "=r"(((unsigned*)(b_fragment + n_pair * 8))[1]),
                      "=r"(((unsigned*)(b_fragment + n_pair * 8))[2]),
                      "=r"(((unsigned*)(b_fragment + n_pair * 8))[3])
                    : "r"(address));
#endif
            }
            #pragma unroll
            for (int m_tile = 0; m_tile < 4; ++m_tile) {
                #pragma unroll
                for (int n_pair = 0; n_pair < 2; ++n_pair) {
                    mma_m16n8k16(
                        accum + m_tile * 16 + n_pair * 8,
                        a_fragment + m_tile * 8,
                        b_fragment + n_pair * 8);
                }
            }
        }

        ++ci_tile;
        ci_offset += kBK;
        if (ci_tile == loops_per_slot) {
            ci_tile = 0;
            ci_offset = 0;
            const int previous_slot = compact_slot++;
            if (compact_slot < slot_count) {
                slot_row_ptr += 1;
                slot_weight_ptr += (1 + template_boundary_bump(template_id, previous_slot)) * logical_stride;
            }
        }
    }

    #pragma unroll
    for (int m_tile = 0; m_tile < 4; ++m_tile) {
        const int local_reorder_offset = reorder_offset + m_tile * 16;
        #pragma unroll
        for (int n_pair = 0; n_pair < 2; ++n_pair) {
            #pragma unroll
            for (int local_id = 0; local_id < 8; ++local_id) {
                const int row_local = local_reorder_offset + (((local_id / 2) % 2) * 8);
                const int out_row = p.out_rows[out_row_base + row_local];
                const int col = channel_base + n_pair * 16 + (local_id % 2) + (local_id / 4) * 8;
                if (out_row >= 0 && col < cout) {
                    p.output[out_row * cout + col] =
                        __float2half(accum[m_tile * 16 + n_pair * 8 + local_id]);
                }
            }
        }
    }
}

__global__ void __launch_bounds__(kThreads, 4) kernel9_fp16_kernel(Kernel9FP16Params p) {
    __shared__ half a_shared[kASharedSize];
    __shared__ half b_shared[kBSharedSize];
    const int channel_tiles = (p.c_out + kBN - 1) / kBN;
    const int logical = static_cast<int>(blockIdx.x);
    const int out_row_base = (logical / channel_tiles) * kBM;
    const int bn_base = (logical % channel_tiles) * kBN;
    const int template_id = p.template_ids[out_row_base];
    if (template_id < 0) return;
    kernel9_fp16_tile(
        p, template_id, out_row_base, p.input_row_offsets[out_row_base],
        bn_base, a_shared, b_shared);
}

}  // namespace

torch::Tensor kernel9_fp16_forward(
    torch::Tensor features,
    torch::Tensor logical_weight,
    torch::Tensor out_rows,
    torch::Tensor input_rows_w1,
    torch::Tensor input_rows_w4,
    torch::Tensor input_rows_w7,
    torch::Tensor input_rows_w9,
    torch::Tensor template_ids,
    torch::Tensor input_row_offsets,
    int64_t n_out) {
    c10::cuda::CUDAGuard guard(features.device());
    const int cout = static_cast<int>(logical_weight.size(1));
    auto output = torch::empty({n_out, cout}, features.options());
    Kernel9FP16Params p;
    p.features = reinterpret_cast<const half*>(features.data_ptr<at::Half>());
    p.weight = reinterpret_cast<const half*>(logical_weight.data_ptr<at::Half>());
    p.out_rows = out_rows.data_ptr<int>();
    p.input_rows_w1 = input_rows_w1.data_ptr<int>();
    p.input_rows_w4 = input_rows_w4.data_ptr<int>();
    p.input_rows_w7 = input_rows_w7.data_ptr<int>();
    p.input_rows_w9 = input_rows_w9.data_ptr<int>();
    p.template_ids = template_ids.data_ptr<int>();
    p.input_row_offsets = input_row_offsets.data_ptr<int>();
    p.output = reinterpret_cast<half*>(output.data_ptr<at::Half>());
    p.c_in = static_cast<int>(features.size(1));
    p.c_out = cout;
    p.padded_rows = static_cast<int>(out_rows.size(0));
    p.template_stride = static_cast<int>(input_rows_w1.size(1));
    const int grid = (p.padded_rows / kBM) * ((p.c_out + kBN - 1) / kBN);
    kernel9_fp16_kernel<<<grid, kThreads, 0, at::cuda::getCurrentCUDAStream()>>>(p);
    return output;
}
