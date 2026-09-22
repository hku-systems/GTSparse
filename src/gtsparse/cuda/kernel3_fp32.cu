#include "api.h"
#include "contract_k3.h"

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime.h>

namespace {

using namespace gtsparse_kernel3;

struct Kernel3FP32Params {
    const float* __restrict__ features;
    const float* __restrict__ weight;
    const int* __restrict__ out_rows;
    const int* __restrict__ input_rows_w1;
    const int* __restrict__ input_rows_w2;
    const int* __restrict__ input_rows_w3;
    const int* __restrict__ template_ids;
    const int* __restrict__ input_row_offsets;
    float* __restrict__ output;
    int c_in;
    int c_out;
    int padded_rows;
    int template_stride;
};

constexpr int kBN = 64;
constexpr int kBK = 32;
constexpr int kThreads = 128;
constexpr int kASharedSize = 4096;
constexpr int kBSharedSize = 2048;

__device__ __forceinline__ int template_slot_count(int template_id) {
    if (template_id <= kTemplateOffset2) {
        return 1;
    }
    if (template_id <= kTemplateOffsets12) {
        return 2;
    }
    return 3;
}

__device__ __forceinline__ int template_payload_width(int template_id) {
    return template_slot_count(template_id);
}

__device__ __forceinline__ int template_initial_offset(int template_id) {
    if (template_id == kTemplateOffset1 || template_id == kTemplateOffsets12) {
        return 1;
    }
    if (template_id == kTemplateOffset2) {
        return 2;
    }
    return 0;
}

__device__ __forceinline__ int template_boundary_bump(int template_id) {
    return template_id == kTemplateOffsets02 ? 1 : 0;
}

__device__ __forceinline__ const int* input_row_ptr(
    const Kernel3FP32Params& p,
    int template_id,
    int input_row_offset,
    int row_local_seed) {
    if (template_id <= kTemplateOffset2) {
        return p.input_rows_w1
            + ((template_id * p.template_stride) + input_row_offset + row_local_seed) * kPayloadWidthW1;
    }
    if (template_id <= kTemplateOffsets12) {
        return p.input_rows_w2
            + (((template_id - kTemplateOffsets01) * p.template_stride)
               + input_row_offset + row_local_seed) * kPayloadWidthW2;
    }
    return p.input_rows_w3 + (input_row_offset + row_local_seed) * kPayloadWidthW3;
}

__device__ __forceinline__ void kernel3_fp32_tile(
    const Kernel3FP32Params& p,
    int template_id,
    int out_row_base,
    int input_row_offset,
    int bn_base,
    float* a_shared,
    float* b_shared) {
    const int tid = static_cast<int>(threadIdx.x);
    const int cin = p.c_in;
    const int cout = p.c_out;
    const int loops_per_slot = cin / kBK;
    const int slot_count = template_slot_count(template_id);
    const int payload_width = template_payload_width(template_id);
    const int total_k_loops = slot_count * loops_per_slot;
    const int row_pitch = payload_width * 16;
    const int logical_stride = cin * cout;
    const int weight_k_stride = 8 * cout;

    float accum[64];
    #pragma unroll
    for (int index = 0; index < 64; ++index) {
        accum[index] = 0.0f;
    }

    float* a_shared_ptr = a_shared + tid * 4;
    float* a_reduce_ptr = a_shared + ((tid / 16) * 32);
    float* b_shared_ptr = b_shared + tid * 4;
    float* b_reduce_ptr = b_shared + (tid % 16);
    const int location_offset = out_row_base + (tid / 16);
    const int channel_offset = bn_base + (tid % 16);
    const int channel_offset_a = (tid * 4) % 32;
    const int row_local_seed = tid / 8;
    const int* slot_row_ptr = input_row_ptr(p, template_id, input_row_offset, row_local_seed);

    int compact_slot = 0;
    int ci_tile = 0;
    int ci_offset = 0;
    const float* slot_weight_ptr =
        p.weight
        + template_initial_offset(template_id) * logical_stride
        + (tid / 16) * cout
        + bn_base
        + ((tid * 4) % 64);

    #pragma unroll
    for (int k_loop = 0; k_loop < total_k_loops; ++k_loop) {
        const float* slot_weight_ci_ptr = slot_weight_ptr + ci_offset * cout;

        __syncthreads();
        const int* input_idx_ptr = slot_row_ptr;
        #pragma unroll
        for (int row = 0; row < 8; ++row) {
            const int input_idx = *input_idx_ptr;
            if (input_idx >= 0) {
                *reinterpret_cast<float4*>(a_shared_ptr + row * 512) =
                    *reinterpret_cast<const float4*>(
                        p.features + input_idx * cin + ci_offset + channel_offset_a);
            } else {
                *reinterpret_cast<float4*>(a_shared_ptr + row * 512) = make_float4(0.f, 0.f, 0.f, 0.f);
            }
            input_idx_ptr += row_pitch;
        }
        const float* weight_ptr = slot_weight_ci_ptr;
        #pragma unroll
        for (int row = 0; row < 4; ++row) {
            *reinterpret_cast<float4*>(b_shared_ptr + row * 512) =
                *reinterpret_cast<const float4*>(weight_ptr);
            weight_ptr += weight_k_stride;
        }

        __syncthreads();
        #pragma unroll
        for (int k1 = 0; k1 < 8; ++k1) {
            #pragma unroll
            for (int k2 = 0; k2 < 4; ++k2) {
                const int vk = (k1 << 2) + k2;
                #pragma unroll
                for (int index = 0; index < 64; ++index) {
                    accum[index] +=
                        a_reduce_ptr[((index / 4) * 8) * 32 + vk]
                        * b_reduce_ptr[vk * 64 + (index % 4) * 16];
                }
            }
        }

        ++ci_tile;
        ci_offset += kBK;
        if (ci_tile == loops_per_slot) {
            ci_tile = 0;
            ci_offset = 0;
            ++compact_slot;
            if (compact_slot < slot_count) {
                slot_row_ptr += 1;
                slot_weight_ptr += (1 + template_boundary_bump(template_id)) * logical_stride;
            }
        }
    }

    #pragma unroll
    for (int index = 0; index < 64; ++index) {
        const int out_row = p.out_rows[location_offset + ((index / 4) * 8)];
        const int col = channel_offset + (index % 4) * 16;
        if (out_row >= 0 && col < cout) {
            p.output[out_row * cout + col] = accum[index];
        }
    }
}

__global__ void __launch_bounds__(kThreads) kernel3_fp32_kernel(Kernel3FP32Params p) {
    __shared__ float a_shared[kASharedSize];
    __shared__ float b_shared[kBSharedSize];
    const int channel_tiles = (p.c_out + kBN - 1) / kBN;
    const int logical = static_cast<int>(blockIdx.x);
    const int row_tile = logical / channel_tiles;
    const int out_row_base = row_tile * kBM;
    const int bn_base = (logical % channel_tiles) * kBN;
    const int template_id = p.template_ids[out_row_base];
    if (template_id < 0) {
        return;
    }
    kernel3_fp32_tile(
        p,
        template_id,
        out_row_base,
        p.input_row_offsets[out_row_base],
        bn_base,
        a_shared,
        b_shared);
}

}  // namespace

torch::Tensor kernel3_fp32_forward(
    torch::Tensor features,
    torch::Tensor logical_weight,
    torch::Tensor out_rows,
    torch::Tensor input_rows_w1,
    torch::Tensor input_rows_w2,
    torch::Tensor input_rows_w3,
    torch::Tensor template_ids,
    torch::Tensor input_row_offsets,
    int64_t n_out) {
#ifndef NDEBUG
    TORCH_CHECK(features.scalar_type() == at::kFloat, "features must be float32");
    TORCH_CHECK(logical_weight.scalar_type() == at::kFloat, "weight must be float32");
    TORCH_CHECK(features.size(1) % kBK == 0, "Cin must be divisible by 32");
    TORCH_CHECK(logical_weight.size(1) % kBN == 0, "Cout must be divisible by 64");
    TORCH_CHECK(logical_weight.size(0) == 3 * features.size(1), "weight must be [3 * Cin, Cout]");
#endif
    c10::cuda::CUDAGuard guard(features.device());
    const int cout = static_cast<int>(logical_weight.size(1));
    auto output = torch::empty({n_out, cout}, features.options());
    if (n_out == 0) {
        return output;
    }

    Kernel3FP32Params p;
    p.features = features.data_ptr<float>();
    p.weight = logical_weight.data_ptr<float>();
    p.out_rows = out_rows.data_ptr<int>();
    p.input_rows_w1 = input_rows_w1.data_ptr<int>();
    p.input_rows_w2 = input_rows_w2.data_ptr<int>();
    p.input_rows_w3 = input_rows_w3.data_ptr<int>();
    p.template_ids = template_ids.data_ptr<int>();
    p.input_row_offsets = input_row_offsets.data_ptr<int>();
    p.output = output.data_ptr<float>();
    p.c_in = static_cast<int>(features.size(1));
    p.c_out = cout;
    p.padded_rows = static_cast<int>(out_rows.size(0));
    p.template_stride = static_cast<int>(input_rows_w1.size(1));

    const int channel_tiles = (p.c_out + kBN - 1) / kBN;
    const int grid = (p.padded_rows / kBM) * channel_tiles;
    kernel3_fp32_kernel<<<grid, kThreads, 0, at::cuda::getCurrentCUDAStream()>>>(p);
    return output;
}
