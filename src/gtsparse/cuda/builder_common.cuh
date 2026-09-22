#pragma once

#include <cstdlib>
#include <ATen/cuda/CUDAContext.h>
#include <cub/cub.cuh>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

#include "contract.h"
#include "hashmap.cuh"

namespace gtsparse_row_template_center_last_builder {

using namespace gtsparse_row_template_center_last;

constexpr int kBuilderThreads = 256;
constexpr int kBuilderWarpSize = 32;
constexpr int kBuilderWarpsPerBlock = kBuilderThreads / kBuilderWarpSize;

static __device__ __constant__ unsigned int kTemplateRejectMaskActualConst[kNumTemplates] = {
    0x07ffdfffu,
    0x07ffde00u,
    0x07fc01ffu,
    0x0003dfffu,
    0x000001ffu,
    0x0003de00u,
    0x07fc0000u,
    0x00000000u,
};

static __device__ __constant__ int kTemplateSlotCountConst[kNumTemplates] = {
    1,
    10,
    9,
    10,
    18,
    19,
    18,
    27,
};

static __device__ __constant__ int kTemplateLocalIndexConst[kNumTemplates] = {
    0,
    0,
    1,
    2,
    0,
    1,
    2,
    0,
};

static __device__ __constant__ int kTemplatePayloadActualOffsetConst[kNumTemplates][kMaxPayloadSlots] = {
    {13, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {0, 3, 6, 1, 4, 7, 2, 5, 8, 13, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {9, 12, 15, 10, 16, 11, 14, 17, 13, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {18, 21, 24, 19, 22, 25, 20, 23, 26, 13, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {9, 18, 12, 21, 15, 24, 10, 19, 22, 16, 25, 11, 20, 14, 23, 17, 26, 13, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {0, 18, 3, 21, 6, 24, 1, 19, 4, 22, 7, 25, 2, 20, 5, 23, 8, 26, 13, -1, -1, -1, -1, -1, -1, -1, -1},
    {0, 9, 3, 12, 6, 15, 1, 10, 4, 7, 16, 2, 11, 5, 14, 8, 17, 13, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {0, 9, 18, 3, 12, 21, 6, 15, 24, 1, 10, 19, 4, 22, 7, 16, 25, 2, 11, 20, 5, 14, 23, 8, 17, 26, 13},
};

__device__ __forceinline__ int classify_template_fast(uint32_t active_mask, int min_template_id = 0) {
    #pragma unroll
    for (int template_id = 0; template_id < kNumTemplates; ++template_id) {
        if (template_id < min_template_id) continue;
        if ((active_mask & kTemplateRejectMaskActualConst[template_id]) == 0u) {
            return template_id;
        }
    }
    return kTemplateFull27;
}

__device__ __forceinline__ int template_payload_width(int template_id) {
    if (template_id == kTemplateCenter) {
        return kPayloadWidthW1;
    }
    if (template_id >= kTemplateSkip2Keep0 && template_id <= kTemplateSkip2Keep2) {
        return kPayloadWidthW9;
    }
    if (template_id >= kTemplateSkip1Hole0 && template_id <= kTemplateSkip1Hole2) {
        return kPayloadWidthW18;
    }
    return kPayloadWidthW27;
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
    int out_row_id,
    int slot_count) {
    const int lane = threadIdx.x & 31;
    if (lane == 0) {
        template_out_rows[template_id * template_stride + row_pos] = out_row_id;
    }
    if (lane >= PayloadWidth) {
        return;
    }
    const int row_base = ((local_template_id * template_stride) + row_pos) * PayloadWidth;
    if (lane < slot_count) {
        const int src_offset = kTemplatePayloadActualOffsetConst[template_id][lane];
        payload_segment[row_base + lane] = warp_probed_rows[src_offset];
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
    int* __restrict__ input_rows_w9,
    int* __restrict__ input_rows_w18,
    int* __restrict__ input_rows_w27) {
    const int slot_count = kTemplateSlotCountConst[template_id];
    if (template_id == kTemplateCenter) {
        write_payload_row<kPayloadWidthW1>(
            template_out_rows,
            input_rows_w1,
            warp_probed_rows,
            0,
            template_stride,
            row_pos,
            template_id,
            out_row_id,
            slot_count);
        return;
    }
    if (template_id >= kTemplateSkip2Keep0 && template_id <= kTemplateSkip2Keep2) {
        write_payload_row<kPayloadWidthW9>(
            template_out_rows,
            input_rows_w9,
            warp_probed_rows,
            kTemplateLocalIndexConst[template_id],
            template_stride,
            row_pos,
            template_id,
            out_row_id,
            slot_count);
        return;
    }
    if (template_id >= kTemplateSkip1Hole0 && template_id <= kTemplateSkip1Hole2) {
        write_payload_row<kPayloadWidthW18>(
            template_out_rows,
            input_rows_w18,
            warp_probed_rows,
            kTemplateLocalIndexConst[template_id],
            template_stride,
            row_pos,
            template_id,
            out_row_id,
            slot_count);
        return;
    }
    write_payload_row<kPayloadWidthW27>(
        template_out_rows,
        input_rows_w27,
        warp_probed_rows,
        0,
        template_stride,
        row_pos,
        template_id,
        out_row_id,
        slot_count);
}

static __global__ void build_layout_kernel(
    const int* __restrict__ template_counts,
    int* __restrict__ padded_counts,
    int* __restrict__ global_row_bases,
    int bm) {
    if (blockIdx.x != 0 || threadIdx.x != 0) {
        return;
    }
    int global_prefix = 0;
    for (int template_id = kNumTemplates - 1; template_id >= 0; --template_id) {
        const int count = template_counts[template_id];
        const int padded = ((count + bm - 1) / bm) * bm;
        padded_counts[template_id] = padded;
        global_row_bases[template_id] = global_prefix;
        global_prefix += padded;
    }
}

static __global__ void build_padded_row_count_kernel(
    const int* __restrict__ padded_counts,
    int* __restrict__ padded_row_count) {
    if (blockIdx.x != 0 || threadIdx.x != 0) {
        return;
    }
    int total = 0;
    #pragma unroll
    for (int template_id = 0; template_id < kNumTemplates; ++template_id) {
        total += padded_counts[template_id];
    }
    padded_row_count[0] = total;
}

struct BuildScatterParams {
    const int* __restrict__ template_counts;
    const int* __restrict__ padded_counts;
    const int* __restrict__ global_row_bases;
    const int* __restrict__ template_out_rows;
    int* __restrict__ out_rows;
    int* __restrict__ input_rows_w1;
    int* __restrict__ input_rows_w9;
    int* __restrict__ input_rows_w18;
    int* __restrict__ input_rows_w27;
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
            for (int i = 0; i < PayloadWidth; ++i) {
                dst[i] = -1;
            }
        }
    }
}

static __global__ void scatter_runtime_kernel(BuildScatterParams p) {
    const int template_id = blockIdx.x;
    if (template_id == kNumTemplates) {
        const int padded_row_count =
            p.global_row_bases[kNumTemplates - 1] + p.padded_counts[kNumTemplates - 1];
        for (int global_row = padded_row_count + threadIdx.x; global_row < p.max_global_rows; global_row += blockDim.x) {
            p.template_ids[global_row] = -1;
        }
        return;
    }
    if (template_id > kNumTemplates) {
        return;
    }
    const int padded = p.padded_counts[template_id];
    if (padded <= 0) {
        return;
    }
    if (template_id == kTemplateCenter) {
        scatter_template_rows<kPayloadWidthW1>(p, template_id, 0, p.input_rows_w1);
    } else if (template_id >= kTemplateSkip2Keep0 && template_id <= kTemplateSkip2Keep2) {
        scatter_template_rows<kPayloadWidthW9>(p, template_id, kTemplateLocalIndexConst[template_id], p.input_rows_w9);
    } else if (template_id >= kTemplateSkip1Hole0 && template_id <= kTemplateSkip1Hole2) {
        scatter_template_rows<kPayloadWidthW18>(p, template_id, kTemplateLocalIndexConst[template_id], p.input_rows_w18);
    } else {
        scatter_template_rows<kPayloadWidthW27>(p, template_id, 0, p.input_rows_w27);
    }
}

static __global__ void build_runtime_from_dense_out_in_map_kernel(
    const int* __restrict__ dense_out_in_map,
    const int* __restrict__ dense_masks,
    int* __restrict__ template_counts,
    int* __restrict__ template_out_rows,
    int* __restrict__ input_rows_w1,
    int* __restrict__ input_rows_w9,
    int* __restrict__ input_rows_w18,
    int* __restrict__ input_rows_w27,
    int n_out,
    int template_stride,
    int min_template_id) {
    const int warp_id = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int row = blockIdx.x * kBuilderWarpsPerBlock + warp_id;
    if (row >= n_out) {
        return;
    }

    __shared__ int s_dense_rows[kBuilderThreads];
    int input_row = -1;
    bool active = false;
    int row_mask = 0;
    if (dense_masks != nullptr && lane == 0) {
        row_mask = dense_masks[row];
    }
    row_mask = __shfl_sync(0xffffffffu, row_mask, 0);
    if (lane < kNumLogicalOffsets) {
        if (dense_masks != nullptr) {
            active = ((row_mask >> lane) & 1u) != 0u;
            input_row = active ? dense_out_in_map[row * kNumLogicalOffsets + lane] : -1;
        } else {
            input_row = dense_out_in_map[row * kNumLogicalOffsets + lane];
            active = input_row >= 0;
        }
    }
    s_dense_rows[threadIdx.x] = input_row;
    __syncwarp();
    const int* warp_dense_rows = s_dense_rows + (threadIdx.x & ~31);

    const unsigned int active_mask = __ballot_sync(0xffffffffu, active);
    int template_id = -1;
    int row_pos = -1;
    if (lane == 0) {
        const unsigned int classify_mask =
            dense_masks != nullptr ? static_cast<unsigned int>(row_mask) : active_mask;
        template_id = classify_template_fast(classify_mask, min_template_id);
        row_pos = atomicAdd(template_counts + template_id, 1);
    }
    template_id = __shfl_sync(0xffffffffu, template_id, 0);
    row_pos = __shfl_sync(0xffffffffu, row_pos, 0);

    scatter_payload_row(
        template_id,
        row_pos,
        template_stride,
        row,
        warp_dense_rows,
        template_out_rows,
        input_rows_w1,
        input_rows_w9,
        input_rows_w18,
        input_rows_w27);
}

static __global__ void build_template_segment_offsets_kernel(
    const int* __restrict__ template_counts,
    int* __restrict__ segment_offsets) {
    if (blockIdx.x != 0 || threadIdx.x != 0) {
        return;
    }
    int prefix = 0;
    segment_offsets[0] = 0;
    #pragma unroll
    for (int template_id = 0; template_id < kNumTemplates; ++template_id) {
        prefix += template_counts[template_id];
        segment_offsets[template_id + 1] = prefix;
    }
}

__device__ __forceinline__ const int* payload_row_ptr_for_sort(
    int template_id,
    int row,
    int template_stride,
    const int* __restrict__ input_rows_w1,
    const int* __restrict__ input_rows_w9,
    const int* __restrict__ input_rows_w18,
    const int* __restrict__ input_rows_w27) {
    if (template_id == kTemplateCenter) {
        return input_rows_w1 + row * kPayloadWidthW1;
    }
    if (template_id >= kTemplateSkip2Keep0 && template_id <= kTemplateSkip2Keep2) {
        return input_rows_w9 +
            ((kTemplateLocalIndexConst[template_id] * template_stride) + row) * kPayloadWidthW9;
    }
    if (template_id >= kTemplateSkip1Hole0 && template_id <= kTemplateSkip1Hole2) {
        return input_rows_w18 +
            ((kTemplateLocalIndexConst[template_id] * template_stride) + row) * kPayloadWidthW18;
    }
    return input_rows_w27 + row * kPayloadWidthW27;
}

static __global__ void build_template_sort_keys_kernel(
    const int* __restrict__ template_counts,
    const int* __restrict__ segment_offsets,
    const int* __restrict__ input_rows_w1,
    const int* __restrict__ input_rows_w9,
    const int* __restrict__ input_rows_w18,
    const int* __restrict__ input_rows_w27,
    int template_stride,
    int64_t* __restrict__ sort_keys,
    int* __restrict__ sort_indices) {
    const int template_id = blockIdx.x;
    if (template_id >= kNumTemplates) {
        return;
    }
    const int count = template_counts[template_id];
    const int segment_begin = segment_offsets[template_id];
    const int slot_count = kTemplateSlotCountConst[template_id];
    for (int row = threadIdx.x; row < count; row += blockDim.x) {
        const int* payload = payload_row_ptr_for_sort(
            template_id,
            row,
            template_stride,
            input_rows_w1,
            input_rows_w9,
            input_rows_w18,
            input_rows_w27);
        unsigned int mask = 0u;
        int popcount = 0;
        #pragma unroll
        for (int slot = 0; slot < kMaxPayloadSlots; ++slot) {
            if (slot >= slot_count) {
                break;
            }
            if (payload[slot] >= 0) {
                mask |= (1u << slot);
                ++popcount;
            }
        }
        sort_keys[segment_begin + row] =
            (static_cast<int64_t>(popcount) << kNumLogicalOffsets) | static_cast<int64_t>(mask);
        sort_indices[segment_begin + row] = row;
    }
}

static __global__ void reorder_template_out_rows_kernel(
    const int* __restrict__ template_counts,
    const int* __restrict__ segment_offsets,
    const int* __restrict__ sorted_indices,
    const int* __restrict__ src_template_out_rows,
    int* __restrict__ dst_template_out_rows,
    int template_stride) {
    const int template_id = blockIdx.x;
    if (template_id >= kNumTemplates) {
        return;
    }
    const int count = template_counts[template_id];
    const int segment_begin = segment_offsets[template_id];
    const int src_base = template_id * template_stride;
    for (int row = threadIdx.x; row < count; row += blockDim.x) {
        const int src_row = sorted_indices[segment_begin + row];
        dst_template_out_rows[src_base + row] = src_template_out_rows[src_base + src_row];
    }
}

template <int PayloadWidth, int FamilyBaseTemplate, int FamilyTemplateCount>
static __global__ void reorder_family_payload_rows_kernel(
    const int* __restrict__ template_counts,
    const int* __restrict__ segment_offsets,
    const int* __restrict__ sorted_indices,
    const int* __restrict__ src_input_rows,
    int* __restrict__ dst_input_rows,
    int template_stride) {
    const int local_template_id = blockIdx.x;
    if (local_template_id >= FamilyTemplateCount) {
        return;
    }
    const int template_id = FamilyBaseTemplate + local_template_id;
    const int count = template_counts[template_id];
    const int segment_begin = segment_offsets[template_id];
    for (int row = threadIdx.x; row < count; row += blockDim.x) {
        const int src_row = sorted_indices[segment_begin + row];
        const int src_base = ((local_template_id * template_stride) + src_row) * PayloadWidth;
        const int dst_base = ((local_template_id * template_stride) + row) * PayloadWidth;
        #pragma unroll
        for (int slot = 0; slot < PayloadWidth; ++slot) {
            dst_input_rows[dst_base + slot] = src_input_rows[src_base + slot];
        }
    }
}

static inline void sort_runtime_rows_by_local_mask_(
    const torch::Tensor& template_counts,
    int total_rows,
    int template_stride,
    cudaStream_t stream,
    torch::Tensor& template_out_rows,
    torch::Tensor& input_rows_w1,
    torch::Tensor& input_rows_w9,
    torch::Tensor& input_rows_w18,
    torch::Tensor& input_rows_w27) {
    if (total_rows <= 1) {
        return;
    }

    auto int_opts = template_counts.options().dtype(torch::kInt32);
    auto key_opts = template_counts.options().dtype(torch::kInt64);
    auto segment_offsets = torch::empty({kNumTemplates + 1}, int_opts);
    build_template_segment_offsets_kernel<<<1, 1, 0, stream>>>(
        template_counts.data_ptr<int>(),
        segment_offsets.data_ptr<int>());

    auto sort_keys_in = torch::empty({total_rows}, key_opts);
    auto sort_keys_out = torch::empty({total_rows}, key_opts);
    auto sort_indices_in = torch::empty({total_rows}, int_opts);
    auto sort_indices_out = torch::empty({total_rows}, int_opts);
    build_template_sort_keys_kernel<<<kNumTemplates, kBuilderThreads, 0, stream>>>(
        template_counts.data_ptr<int>(),
        segment_offsets.data_ptr<int>(),
        input_rows_w1.data_ptr<int>(),
        input_rows_w9.data_ptr<int>(),
        input_rows_w18.data_ptr<int>(),
        input_rows_w27.data_ptr<int>(),
        template_stride,
        sort_keys_in.data_ptr<int64_t>(),
        sort_indices_in.data_ptr<int>());

    size_t sort_temp_bytes = 0;
    cub::DeviceSegmentedRadixSort::SortPairsDescending(
        nullptr,
        sort_temp_bytes,
        sort_keys_in.data_ptr<int64_t>(),
        sort_keys_out.data_ptr<int64_t>(),
        sort_indices_in.data_ptr<int>(),
        sort_indices_out.data_ptr<int>(),
        total_rows,
        kNumTemplates,
        segment_offsets.data_ptr<int>(),
        segment_offsets.data_ptr<int>() + 1,
        0,
        64,
        stream);
    auto sort_temp = torch::empty(
        {static_cast<int64_t>(sort_temp_bytes)},
        template_counts.options().dtype(torch::kUInt8));
    cub::DeviceSegmentedRadixSort::SortPairsDescending(
        sort_temp.data_ptr(),
        sort_temp_bytes,
        sort_keys_in.data_ptr<int64_t>(),
        sort_keys_out.data_ptr<int64_t>(),
        sort_indices_in.data_ptr<int>(),
        sort_indices_out.data_ptr<int>(),
        total_rows,
        kNumTemplates,
        segment_offsets.data_ptr<int>(),
        segment_offsets.data_ptr<int>() + 1,
        0,
        64,
        stream);

    auto sorted_template_out_rows = torch::empty_like(template_out_rows);
    auto sorted_input_rows_w1 = torch::empty_like(input_rows_w1);
    auto sorted_input_rows_w9 = torch::empty_like(input_rows_w9);
    auto sorted_input_rows_w18 = torch::empty_like(input_rows_w18);
    auto sorted_input_rows_w27 = torch::empty_like(input_rows_w27);
    reorder_template_out_rows_kernel<<<kNumTemplates, kBuilderThreads, 0, stream>>>(
        template_counts.data_ptr<int>(),
        segment_offsets.data_ptr<int>(),
        sort_indices_out.data_ptr<int>(),
        template_out_rows.data_ptr<int>(),
        sorted_template_out_rows.data_ptr<int>(),
        template_stride);
    reorder_family_payload_rows_kernel<kPayloadWidthW1, kTemplateCenter, kFamilyW1Templates><<<
        kFamilyW1Templates,
        kBuilderThreads,
        0,
        stream>>>(
        template_counts.data_ptr<int>(),
        segment_offsets.data_ptr<int>(),
        sort_indices_out.data_ptr<int>(),
        input_rows_w1.data_ptr<int>(),
        sorted_input_rows_w1.data_ptr<int>(),
        template_stride);
    reorder_family_payload_rows_kernel<kPayloadWidthW9, kTemplateSkip2Keep0, kFamilyW9Templates><<<
        kFamilyW9Templates,
        kBuilderThreads,
        0,
        stream>>>(
        template_counts.data_ptr<int>(),
        segment_offsets.data_ptr<int>(),
        sort_indices_out.data_ptr<int>(),
        input_rows_w9.data_ptr<int>(),
        sorted_input_rows_w9.data_ptr<int>(),
        template_stride);
    reorder_family_payload_rows_kernel<kPayloadWidthW18, kTemplateSkip1Hole0, kFamilyW18Templates><<<
        kFamilyW18Templates,
        kBuilderThreads,
        0,
        stream>>>(
        template_counts.data_ptr<int>(),
        segment_offsets.data_ptr<int>(),
        sort_indices_out.data_ptr<int>(),
        input_rows_w18.data_ptr<int>(),
        sorted_input_rows_w18.data_ptr<int>(),
        template_stride);
    reorder_family_payload_rows_kernel<kPayloadWidthW27, kTemplateFull27, kFamilyW27Templates><<<
        kFamilyW27Templates,
        kBuilderThreads,
        0,
        stream>>>(
        template_counts.data_ptr<int>(),
        segment_offsets.data_ptr<int>(),
        sort_indices_out.data_ptr<int>(),
        input_rows_w27.data_ptr<int>(),
        sorted_input_rows_w27.data_ptr<int>(),
        template_stride);

    template_out_rows = sorted_template_out_rows;
    input_rows_w1 = sorted_input_rows_w1;
    input_rows_w9 = sorted_input_rows_w9;
    input_rows_w18 = sorted_input_rows_w18;
    input_rows_w27 = sorted_input_rows_w27;
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
make_empty_runtime_tensors(const torch::TensorOptions& int_opts) {
    auto empty_1d = torch::empty({0}, int_opts);
    auto empty_w1 = torch::empty({kFamilyW1Templates, 0, kPayloadWidthW1}, int_opts);
    auto empty_w9 = torch::empty({kFamilyW9Templates, 0, kPayloadWidthW9}, int_opts);
    auto empty_w18 = torch::empty({kFamilyW18Templates, 0, kPayloadWidthW18}, int_opts);
    auto empty_w27 = torch::empty({kFamilyW27Templates, 0, kPayloadWidthW27}, int_opts);
    auto zero_counts = torch::zeros({kNumTemplates}, int_opts);
    return {
        empty_1d,
        empty_w1,
        empty_w9,
        empty_w18,
        empty_w27,
        empty_1d,
        empty_1d,
        zero_counts,
        zero_counts.clone()};
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
    const torch::Tensor& input_rows_w9,
    const torch::Tensor& input_rows_w18,
    const torch::Tensor& input_rows_w27,
    int max_rows,
    int bm,
    cudaStream_t stream) {
    auto int_opts = template_counts.options().dtype(torch::kInt32);
    auto padded_counts = torch::empty({kNumTemplates}, int_opts);
    auto global_row_bases = torch::empty({kNumTemplates}, int_opts);
    build_layout_kernel<<<1, 128, 0, stream>>>(
        template_counts.data_ptr<int>(),
        padded_counts.data_ptr<int>(),
        global_row_bases.data_ptr<int>(),
        bm);

    const int max_global_rows = ((max_rows + kNumTemplates * bm + bm - 1) / bm) * bm;
    auto out_rows = torch::empty({max_global_rows}, int_opts);
    auto template_ids = torch::empty({max_global_rows}, int_opts);
    auto input_row_offsets = torch::empty({max_global_rows}, int_opts);

    BuildScatterParams sp;
    sp.template_counts = template_counts.data_ptr<int>();
    sp.padded_counts = padded_counts.data_ptr<int>();
    sp.global_row_bases = global_row_bases.data_ptr<int>();
    sp.template_out_rows = template_out_rows.data_ptr<int>();
    sp.out_rows = out_rows.data_ptr<int>();
    sp.input_rows_w1 = input_rows_w1.data_ptr<int>();
    sp.input_rows_w9 = input_rows_w9.data_ptr<int>();
    sp.input_rows_w18 = input_rows_w18.data_ptr<int>();
    sp.input_rows_w27 = input_rows_w27.data_ptr<int>();
    sp.template_ids = template_ids.data_ptr<int>();
    sp.input_row_offsets = input_row_offsets.data_ptr<int>();
    sp.template_stride = static_cast<int>(input_rows_w1.size(1));
    sp.max_global_rows = max_global_rows;
    scatter_runtime_kernel<<<kNumTemplates + 1, 256, 0, stream>>>(sp);

    return {
        out_rows,
        input_rows_w1,
        input_rows_w9,
        input_rows_w18,
        input_rows_w27,
        template_ids,
        input_row_offsets,
        template_counts,
        padded_counts};
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
build_runtime_from_dense_out_in_map(
    torch::Tensor dense_out_in_map,
    torch::Tensor dense_masks,
    torch::Tensor template_counts,
    int max_bm,
    bool sorted,
    cudaStream_t stream) {
#ifndef NDEBUG
    check_cuda_contiguous(dense_out_in_map, "dense_out_in_map");
    TORCH_CHECK(dense_out_in_map.scalar_type() == at::kInt, "dense_out_in_map must be int32");
    TORCH_CHECK(
        dense_out_in_map.dim() == 2 && dense_out_in_map.size(1) == kNumLogicalOffsets,
        "dense_out_in_map must be [N, 27]");
    if (dense_masks.defined()) {
        check_cuda_contiguous(dense_masks, "dense_masks");
        TORCH_CHECK(dense_masks.scalar_type() == at::kInt, "dense_masks must be int32");
        TORCH_CHECK(
            dense_masks.dim() == 1 && dense_masks.size(0) == dense_out_in_map.size(0),
            "dense_masks must be [N]");
    }
    TORCH_CHECK(max_bm > 0, "max_bm must be positive");
#endif
    const int n_out = static_cast<int>(dense_out_in_map.size(0));
    auto int_opts = dense_out_in_map.options().dtype(torch::kInt32);
    if (n_out == 0) {
        return make_empty_runtime_tensors(int_opts);
    }

    const int bm = static_cast<int>(max_bm);
    const int template_stride = ((n_out + bm - 1) / bm) * bm;
    if (!template_counts.defined()) {
        template_counts = torch::zeros({kNumTemplates}, int_opts);
    }
    auto template_out_rows = torch::empty({kNumTemplates, template_stride}, int_opts);
    auto input_rows_w1 = torch::empty({kFamilyW1Templates, template_stride, kPayloadWidthW1}, int_opts);
    auto input_rows_w9 = torch::empty({kFamilyW9Templates, template_stride, kPayloadWidthW9}, int_opts);
    auto input_rows_w18 = torch::empty({kFamilyW18Templates, template_stride, kPayloadWidthW18}, int_opts);
    auto input_rows_w27 = torch::empty({kFamilyW27Templates, template_stride, kPayloadWidthW27}, int_opts);

    const dim3 gd((n_out + kBuilderWarpsPerBlock - 1) / kBuilderWarpsPerBlock);
    const char* env_min_t = std::getenv("GTSPARSE_MIN_TEMPLATE");
    const int min_template_id = env_min_t ? std::atoi(env_min_t) : 0;
    build_runtime_from_dense_out_in_map_kernel<<<gd, kBuilderThreads, 0, stream>>>(
        dense_out_in_map.data_ptr<int>(),
        dense_masks.defined() ? dense_masks.data_ptr<int>() : nullptr,
        template_counts.data_ptr<int>(),
        template_out_rows.data_ptr<int>(),
        input_rows_w1.data_ptr<int>(),
        input_rows_w9.data_ptr<int>(),
        input_rows_w18.data_ptr<int>(),
        input_rows_w27.data_ptr<int>(),
        n_out,
        template_stride,
        min_template_id);
    if (sorted) {
        sort_runtime_rows_by_local_mask_(
            template_counts,
            n_out,
            template_stride,
            stream,
            template_out_rows,
            input_rows_w1,
            input_rows_w9,
            input_rows_w18,
            input_rows_w27);
    }

    return finalize_runtime_tensors(
        template_counts,
        template_out_rows,
        input_rows_w1,
        input_rows_w9,
        input_rows_w18,
        input_rows_w27,
        n_out,
        bm,
        stream);
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
    const int64_t hw = static_cast<int64_t>(oH) * static_cast<int64_t>(oW);
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

template <int PayloadWidth>
__device__ __forceinline__ void scatter_dense_reverse_slots(
    const int* __restrict__ payload_base,
    int template_id,
    int out_row,
    int reverse_n_out,
    int* __restrict__ reverse_masks,
    int* __restrict__ dense_out_in_map) {
    const int slot_count = kTemplateSlotCountConst[template_id];
    #pragma unroll
    for (int slot = 0; slot < PayloadWidth; ++slot) {
        if (slot >= slot_count) {
            break;
        }
        const int input_row = payload_base[slot];
        if (input_row < 0 || input_row >= reverse_n_out) {
            continue;
        }
        const int actual_offset = kTemplatePayloadActualOffsetConst[template_id][slot];
        dense_out_in_map[input_row * kNumLogicalOffsets + actual_offset] = out_row;
        atomicOr(reverse_masks + input_row, (1u << actual_offset));
    }
}

static __global__ void build_dense_out_in_map_from_runtime_kernel(
    const int* __restrict__ out_rows,
    const int* __restrict__ input_rows_w1,
    const int* __restrict__ input_rows_w9,
    const int* __restrict__ input_rows_w18,
    const int* __restrict__ input_rows_w27,
    const int* __restrict__ template_ids,
    const int* __restrict__ input_row_offsets,
    const int* __restrict__ padded_row_count_ptr,
    int* __restrict__ reverse_masks,
    int* __restrict__ template_counts,
    int* __restrict__ dense_out_in_map,
    int reverse_n_out,
    int template_stride,
    int max_global_rows) {
    const int global_row = blockIdx.x * blockDim.x + threadIdx.x;
    if (global_row < kNumTemplates) {
        template_counts[global_row] = 0;
    }
    if (global_row >= max_global_rows) {
        return;
    }
    const int padded_row_count =
        padded_row_count_ptr != nullptr ? padded_row_count_ptr[0] : max_global_rows;
    if (global_row >= padded_row_count) {
        return;
    }
    const int template_id = template_ids[global_row];
    if (template_id < 0 || template_id >= kNumTemplates) {
        return;
    }
    const int row_offset = input_row_offsets[global_row];
    if (row_offset < 0 || row_offset >= template_stride) {
        return;
    }
    const int out_row = out_rows[global_row];
    if (template_id == kTemplateCenter) {
        const int* payload_base = input_rows_w1 + row_offset * kPayloadWidthW1;
        scatter_dense_reverse_slots<kPayloadWidthW1>(
            payload_base, template_id, out_row, reverse_n_out, reverse_masks, dense_out_in_map);
        return;
    }
    if (template_id >= kTemplateSkip2Keep0 && template_id <= kTemplateSkip2Keep2) {
        const int* payload_base =
            input_rows_w9 + ((kTemplateLocalIndexConst[template_id] * template_stride) + row_offset) * kPayloadWidthW9;
        scatter_dense_reverse_slots<kPayloadWidthW9>(
            payload_base, template_id, out_row, reverse_n_out, reverse_masks, dense_out_in_map);
        return;
    }
    if (template_id >= kTemplateSkip1Hole0 && template_id <= kTemplateSkip1Hole2) {
        const int* payload_base =
            input_rows_w18 + ((kTemplateLocalIndexConst[template_id] * template_stride) + row_offset) * kPayloadWidthW18;
        scatter_dense_reverse_slots<kPayloadWidthW18>(
            payload_base, template_id, out_row, reverse_n_out, reverse_masks, dense_out_in_map);
        return;
    }
    const int* payload_base = input_rows_w27 + row_offset * kPayloadWidthW27;
    scatter_dense_reverse_slots<kPayloadWidthW27>(
        payload_base, template_id, out_row, reverse_n_out, reverse_masks, dense_out_in_map);
}

}  // namespace gtsparse_row_template_center_last_builder
