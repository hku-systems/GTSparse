#pragma once

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

#include "contract_k9.h"
#include "hashmap.cuh"

namespace gtsparse_kernel9_builder {

using namespace gtsparse_kernel9;

constexpr int kBuilderThreads = 256;
constexpr int kBuilderWarpsPerBlock = kBuilderThreads / 32;

static __device__ __constant__ unsigned int kTemplateRejectMaskConst[kNumTemplates] = {
    0x1efu, 0x1a6u, 0x16du, 0x0cbu, 0x049u, 0x082u, 0x124u, 0x000u,
};

static __device__ __constant__ int kTemplateSlotCountConst[kNumTemplates] = {
    1, 4, 3, 4, 6, 7, 6, 9,
};

static __device__ __constant__ int kTemplateLocalIndexConst[kNumTemplates] = {
    0, 0, 1, 2, 0, 1, 2, 0,
};

static __device__ __constant__ int kTemplatePayloadOffsetConst[kNumTemplates][kMaxPayloadSlots] = {
    {4, -1, -1, -1, -1, -1, -1, -1, -1},
    {0, 3, 4, 6, -1, -1, -1, -1, -1},
    {1, 4, 7, -1, -1, -1, -1, -1, -1},
    {2, 4, 5, 8, -1, -1, -1, -1, -1},
    {1, 2, 4, 5, 7, 8, -1, -1, -1},
    {0, 2, 3, 4, 5, 6, 8, -1, -1},
    {0, 1, 3, 4, 6, 7, -1, -1, -1},
    {0, 1, 2, 3, 4, 5, 6, 7, 8},
};

__device__ __forceinline__ int classify_template_fast(unsigned int active_mask) {
    #pragma unroll
    for (int template_id = 0; template_id < kNumTemplates; ++template_id) {
        if ((active_mask & kTemplateRejectMaskConst[template_id]) == 0u) {
            return template_id;
        }
    }
    return kTemplateFull9;
}

template <int PayloadWidth>
__device__ __forceinline__ void write_payload_row(
    int* __restrict__ template_out_rows,
    int* __restrict__ payload_segment,
    const int* __restrict__ warp_probed_rows,
    int local_template_id,
    int template_stride,
    int row_pos,
    int template_id,
    int out_row_id) {
    const int lane = threadIdx.x & 31;
    if (lane == 0) {
        template_out_rows[template_id * template_stride + row_pos] = out_row_id;
    }
    const int slot_count = kTemplateSlotCountConst[template_id];
    if (lane < slot_count) {
        const int row_base = ((local_template_id * template_stride) + row_pos) * PayloadWidth;
        payload_segment[row_base + lane] =
            warp_probed_rows[kTemplatePayloadOffsetConst[template_id][lane]];
    }
}

__device__ __forceinline__ void scatter_payload_row(
    int template_id,
    int row_pos,
    int template_stride,
    int out_row_id,
    const int* __restrict__ warp_probed_rows,
    int* __restrict__ template_out_rows,
    int* __restrict__ input_rows_w1,
    int* __restrict__ input_rows_w4,
    int* __restrict__ input_rows_w7,
    int* __restrict__ input_rows_w9) {
    if (template_id == kTemplateCenter) {
        write_payload_row<kPayloadWidthW1>(
            template_out_rows, input_rows_w1, warp_probed_rows, 0,
            template_stride, row_pos, template_id, out_row_id);
        return;
    }
    if (template_id <= kTemplateSkip2Keep2) {
        write_payload_row<kPayloadWidthW4>(
            template_out_rows, input_rows_w4, warp_probed_rows,
            kTemplateLocalIndexConst[template_id], template_stride,
            row_pos, template_id, out_row_id);
        return;
    }
    if (template_id <= kTemplateSkip1Hole2) {
        write_payload_row<kPayloadWidthW7>(
            template_out_rows, input_rows_w7, warp_probed_rows,
            kTemplateLocalIndexConst[template_id], template_stride,
            row_pos, template_id, out_row_id);
        return;
    }
    write_payload_row<kPayloadWidthW9>(
        template_out_rows, input_rows_w9, warp_probed_rows, 0,
        template_stride, row_pos, template_id, out_row_id);
}

static __global__ void build_layout_kernel(
    const int* __restrict__ template_counts,
    int* __restrict__ padded_counts,
    int* __restrict__ global_row_bases,
    int bm) {
    if (blockIdx.x != 0 || threadIdx.x != 0) {
        return;
    }
    int prefix = 0;
    for (int template_id = kNumTemplates - 1; template_id >= 0; --template_id) {
        const int count = template_counts[template_id];
        const int padded = ((count + bm - 1) / bm) * bm;
        padded_counts[template_id] = padded;
        global_row_bases[template_id] = prefix;
        prefix += padded;
    }
}

struct BuildScatterParams {
    const int* __restrict__ template_counts;
    const int* __restrict__ padded_counts;
    const int* __restrict__ global_row_bases;
    const int* __restrict__ template_out_rows;
    int* __restrict__ out_rows;
    int* __restrict__ input_rows_w1;
    int* __restrict__ input_rows_w4;
    int* __restrict__ input_rows_w7;
    int* __restrict__ input_rows_w9;
    int* __restrict__ template_ids;
    int* __restrict__ input_row_offsets;
    int template_stride;
    int max_global_rows;
};

template <int PayloadWidth>
__device__ __forceinline__ void scatter_template_rows(
    const BuildScatterParams& p,
    int template_id,
    int local_template_id,
    int* __restrict__ input_rows) {
    const int count = p.template_counts[template_id];
    const int padded = p.padded_counts[template_id];
    const int global_base = p.global_row_bases[template_id];
    for (int row = threadIdx.x; row < padded; row += blockDim.x) {
        const int global_row = global_base + row;
        p.template_ids[global_row] = template_id;
        p.input_row_offsets[global_row] = row;
        if (row < count) {
            p.out_rows[global_row] = p.template_out_rows[template_id * p.template_stride + row];
        } else {
            p.out_rows[global_row] = -1;
            int* dst = input_rows + ((local_template_id * p.template_stride) + row) * PayloadWidth;
            #pragma unroll
            for (int slot = 0; slot < PayloadWidth; ++slot) {
                dst[slot] = -1;
            }
        }
    }
}

static __global__ void scatter_runtime_kernel(BuildScatterParams p) {
    const int template_id = blockIdx.x;
    if (template_id == kNumTemplates) {
        const int padded_row_count = p.global_row_bases[0] + p.padded_counts[0];
        for (int row = padded_row_count + threadIdx.x; row < p.max_global_rows; row += blockDim.x) {
            p.out_rows[row] = -1;
            p.template_ids[row] = -1;
            p.input_row_offsets[row] = -1;
        }
        return;
    }
    if (p.padded_counts[template_id] <= 0) {
        return;
    }
    if (template_id == kTemplateCenter) {
        scatter_template_rows<kPayloadWidthW1>(p, template_id, 0, p.input_rows_w1);
    } else if (template_id <= kTemplateSkip2Keep2) {
        scatter_template_rows<kPayloadWidthW4>(
            p, template_id, kTemplateLocalIndexConst[template_id], p.input_rows_w4);
    } else if (template_id <= kTemplateSkip1Hole2) {
        scatter_template_rows<kPayloadWidthW7>(
            p, template_id, kTemplateLocalIndexConst[template_id], p.input_rows_w7);
    } else {
        scatter_template_rows<kPayloadWidthW9>(p, template_id, 0, p.input_rows_w9);
    }
}

static inline std::tuple<
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor>
finalize_runtime_tensors(
    const torch::Tensor& template_counts,
    const torch::Tensor& template_out_rows,
    const torch::Tensor& input_rows_w1,
    const torch::Tensor& input_rows_w4,
    const torch::Tensor& input_rows_w7,
    const torch::Tensor& input_rows_w9,
    int max_rows,
    int bm,
    cudaStream_t stream) {
    auto int_opts = template_counts.options().dtype(torch::kInt32);
    auto padded_counts = torch::empty({kNumTemplates}, int_opts);
    auto global_row_bases = torch::empty({kNumTemplates}, int_opts);
    build_layout_kernel<<<1, 128, 0, stream>>>(
        template_counts.data_ptr<int>(), padded_counts.data_ptr<int>(),
        global_row_bases.data_ptr<int>(), bm);

    const int max_global_rows = ((max_rows + kNumTemplates * bm + bm - 1) / bm) * bm;
    auto out_rows = torch::empty({max_global_rows}, int_opts);
    auto template_ids = torch::empty({max_global_rows}, int_opts);
    auto input_row_offsets = torch::empty({max_global_rows}, int_opts);
    BuildScatterParams p;
    p.template_counts = template_counts.data_ptr<int>();
    p.padded_counts = padded_counts.data_ptr<int>();
    p.global_row_bases = global_row_bases.data_ptr<int>();
    p.template_out_rows = template_out_rows.data_ptr<int>();
    p.out_rows = out_rows.data_ptr<int>();
    p.input_rows_w1 = input_rows_w1.data_ptr<int>();
    p.input_rows_w4 = input_rows_w4.data_ptr<int>();
    p.input_rows_w7 = input_rows_w7.data_ptr<int>();
    p.input_rows_w9 = input_rows_w9.data_ptr<int>();
    p.template_ids = template_ids.data_ptr<int>();
    p.input_row_offsets = input_row_offsets.data_ptr<int>();
    p.template_stride = static_cast<int>(input_rows_w1.size(1));
    p.max_global_rows = max_global_rows;
    scatter_runtime_kernel<<<kNumTemplates + 1, 256, 0, stream>>>(p);
    return {
        out_rows, input_rows_w1, input_rows_w4, input_rows_w7, input_rows_w9,
        template_ids, input_row_offsets, template_counts, padded_counts};
}

__device__ __forceinline__ void decode_output_coord_linear_key(
    int64_t key,
    int oD,
    int oH,
    int oW,
    int& b,
    int& d,
    int& h,
    int& w) {
    const int64_t hw = static_cast<int64_t>(oH) * oW;
    const int64_t dhw = static_cast<int64_t>(oD) * hw;
    b = static_cast<int>(key / dhw);
    key -= static_cast<int64_t>(b) * dhw;
    d = static_cast<int>(key / hw);
    key -= static_cast<int64_t>(d) * hw;
    h = static_cast<int>(key / oW);
    w = static_cast<int>(key - static_cast<int64_t>(h) * oW);
}

__host__ __device__ __forceinline__ int64_t output_coord_linear_key(
    int b,
    int d,
    int h,
    int w,
    int oD,
    int oH,
    int oW) {
    return (((static_cast<int64_t>(b) * oD + d) * oH + h) * oW + w);
}

}  // namespace gtsparse_kernel9_builder
