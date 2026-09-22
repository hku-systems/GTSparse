#include "api.h"

#include <cstdlib>
#include <cub/cub.cuh>
#include <pybind11/pybind11.h>

#include "builder_common.cuh"

namespace py = pybind11;

namespace {

using namespace gtsparse_row_template_center_last_builder;

// Private full-conv builder for the center-last runtime contract.

static __global__ void enumerate_output_coord_keys_compact_kernel(
    const int* __restrict__ in_coords,
    int64_t* __restrict__ out_keys,
    int* __restrict__ total_candidates,
    int n_in,
    int oD,
    int oH,
    int oW,
    int stride_d,
    int stride_h,
    int stride_w,
    int pad_d,
    int pad_h,
    int pad_w,
    int dil_d,
    int dil_h,
    int dil_w) {
    using BlockScan = cub::BlockScan<int, 256>;
    __shared__ typename BlockScan::TempStorage s_scan;
    __shared__ int s_block_base;
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t local_keys[kNumLogicalOffsets];
    int local_count = 0;

    if (idx < n_in) {
        const int b = in_coords[idx * 4 + 0];
        const int d = in_coords[idx * 4 + 1];
        const int h = in_coords[idx * 4 + 2];
        const int w = in_coords[idx * 4 + 3];

        #pragma unroll
        for (int off = 0; off < kNumLogicalOffsets; ++off) {
            const int rd = off / 9;
            const int rem = off - rd * 9;
            const int rh = rem / 3;
            const int rw = rem - rh * 3;
            int od = d + pad_d - rd * dil_d;
            int oh = h + pad_h - rh * dil_h;
            int ow = w + pad_w - rw * dil_w;
            if (od % stride_d != 0 || oh % stride_h != 0 || ow % stride_w != 0) {
                continue;
            }
            od /= stride_d;
            oh /= stride_h;
            ow /= stride_w;
            if (od < 0 || od >= oD || oh < 0 || oh >= oH || ow < 0 || ow >= oW) {
                continue;
            }
            local_keys[local_count++] = output_coord_linear_key(b, od, oh, ow, oD, oH, oW);
        }
    }

    int local_base = 0;
    int block_count = 0;
    BlockScan(s_scan).ExclusiveSum(local_count, local_base, block_count);
    if (threadIdx.x == 0) {
        s_block_base = block_count > 0 ? atomicAdd(total_candidates, block_count) : 0;
    }
    __syncthreads();
    if (local_count == 0) {
        return;
    }
    const int base = s_block_base + local_base;
    #pragma unroll
    for (int i = 0; i < kNumLogicalOffsets; ++i) {
        if (i >= local_count) {
            break;
        }
        out_keys[base + i] = local_keys[i];
    }
}

static __global__ void build_full_runtime_kernel(
    const int64_t* __restrict__ out_keys,
    CoordHashMap map,
    int* __restrict__ out_coords,
    int* __restrict__ reverse_dense_out_in_map,
    int* __restrict__ reverse_masks,
    int* __restrict__ template_counts,
    int* __restrict__ template_out_rows,
    int* __restrict__ input_rows_w1,
    int* __restrict__ input_rows_w9,
    int* __restrict__ input_rows_w18,
    int* __restrict__ input_rows_w27,
    int n_out,
    int oD,
    int oH,
    int oW,
    int stride_d,
    int stride_h,
    int stride_w,
    int pad_d,
    int pad_h,
    int pad_w,
    int dil_d,
    int dil_h,
    int dil_w,
    int template_stride,
    int min_template_id) {
    const int warp_id = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int row = blockIdx.x * kBuilderWarpsPerBlock + warp_id;
    if (row >= n_out) {
        return;
    }

    __shared__ int s_probed_rows[kBuilderThreads];
    int b = 0;
    int d = 0;
    int h = 0;
    int w = 0;
    if (lane == 0) {
        decode_output_coord_linear_key(out_keys[row], oD, oH, oW, b, d, h, w);
        out_coords[row * 4 + 0] = b;
        out_coords[row * 4 + 1] = d;
        out_coords[row * 4 + 2] = h;
        out_coords[row * 4 + 3] = w;
    }
    b = __shfl_sync(0xffffffffu, b, 0);
    d = __shfl_sync(0xffffffffu, d, 0);
    h = __shfl_sync(0xffffffffu, h, 0);
    w = __shfl_sync(0xffffffffu, w, 0);

    int probed_input_row = -1;
    bool active = false;
    if (lane < kNumLogicalOffsets) {
        const int rd = lane / 9;
        const int rem = lane - rd * 9;
        const int rh = rem / 3;
        const int rw = rem - rh * 3;
        const int nd = d * stride_d + rd * dil_d - pad_d;
        const int nh = h * stride_h + rh * dil_h - pad_h;
        const int nw = w * stride_w + rw * dil_w - pad_w;
        probed_input_row = map.lookup(b, nd, nh, nw);
        active = probed_input_row >= 0;
        if (reverse_dense_out_in_map != nullptr && active) {
            reverse_dense_out_in_map[probed_input_row * kNumLogicalOffsets + lane] = row;
            atomicOr(reverse_masks + probed_input_row, (1u << lane));
        }
    }
    s_probed_rows[threadIdx.x] = probed_input_row;
    __syncwarp();
    const int* warp_probed_rows = s_probed_rows + (threadIdx.x & ~31);

    const unsigned int active_mask = __ballot_sync(0xffffffffu, active);
    int template_id = -1;
    int row_pos = -1;
    if (lane == 0) {
        template_id = classify_template_fast(active_mask, min_template_id);
        row_pos = atomicAdd(template_counts + template_id, 1);
    }
    template_id = __shfl_sync(0xffffffffu, template_id, 0);
    row_pos = __shfl_sync(0xffffffffu, row_pos, 0);

    scatter_payload_row(
        template_id,
        row_pos,
        template_stride,
        row,
        warp_probed_rows,
        template_out_rows,
        input_rows_w1,
        input_rows_w9,
        input_rows_w18,
        input_rows_w27);
}

}  // namespace

std::tuple<
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor>
build_finalize_row_template_center_last_full_runtime_from_coords(
    torch::Tensor in_coords,
    int oD,
    int oH,
    int oW,
    int stride_d,
    int stride_h,
    int stride_w,
    int pad_d,
    int pad_h,
    int pad_w,
    int dil_d,
    int dil_h,
    int dil_w,
    int max_bm,
    torch::Tensor lookup_coord_hashmap,
    bool build_reverse_cache,
    bool sorted) {
#ifndef NDEBUG
    TORCH_CHECK(in_coords.is_cuda(), "in_coords must be a CUDA tensor");
    TORCH_CHECK(in_coords.is_contiguous(), "in_coords must be contiguous");
    TORCH_CHECK(in_coords.scalar_type() == at::kInt, "in_coords must be int32");
    TORCH_CHECK(in_coords.dim() == 2 && in_coords.size(1) == 4, "in_coords must be [N, 4]");
    TORCH_CHECK(oD > 0 && oH > 0 && oW > 0, "output spatial shape must be positive");
    TORCH_CHECK(max_bm > 0, "max_bm must be positive");
#endif

    c10::cuda::CUDAGuard guard(in_coords.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    const int n = static_cast<int>(in_coords.size(0));
    auto int_opts = in_coords.options().dtype(torch::kInt32);
    auto key_opts = in_coords.options().dtype(torch::kInt64);
    if (n == 0) {
        auto empty = make_empty_runtime_tensors(int_opts);
        auto empty_coords = torch::empty({0, 4}, int_opts);
        auto empty_hash = torch::empty({0, 2}, key_opts);
        auto empty_reverse_dense = torch::empty({0, kNumLogicalOffsets}, int_opts);
        auto empty_reverse_masks = torch::empty({0}, int_opts);
        return {
            std::get<0>(empty),
            std::get<1>(empty),
            std::get<2>(empty),
            std::get<3>(empty),
            std::get<4>(empty),
            std::get<5>(empty),
            std::get<6>(empty),
            std::get<7>(empty),
            std::get<8>(empty),
            empty_reverse_dense,
            empty_reverse_masks,
            empty_coords,
            empty_hash};
    }

    auto candidate_keys = torch::empty({static_cast<int64_t>(n) * static_cast<int64_t>(kNumLogicalOffsets)}, key_opts);
    auto num_candidate_keys = torch::zeros({1}, int_opts);
    enumerate_output_coord_keys_compact_kernel<<<(n + 255) / 256, 256, 0, stream>>>(
        in_coords.data_ptr<int>(),
        candidate_keys.data_ptr<int64_t>(),
        num_candidate_keys.data_ptr<int>(),
        n,
        oD,
        oH,
        oW,
        stride_d,
        stride_h,
        stride_w,
        pad_d,
        pad_h,
        pad_w,
        dil_d,
        dil_h,
        dil_w);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    C10_CUDA_CHECK(cudaStreamSynchronize(stream));

    const int candidate_count = num_candidate_keys.item<int>();
    if (candidate_count <= 0) {
        auto empty = make_empty_runtime_tensors(int_opts);
        auto empty_coords = torch::empty({0, 4}, int_opts);
        auto empty_hash = torch::empty({0, 2}, key_opts);
        auto empty_reverse_dense = torch::empty({0, kNumLogicalOffsets}, int_opts);
        auto empty_reverse_masks = torch::empty({0}, int_opts);
        return {
            std::get<0>(empty),
            std::get<1>(empty),
            std::get<2>(empty),
            std::get<3>(empty),
            std::get<4>(empty),
            std::get<5>(empty),
            std::get<6>(empty),
            std::get<7>(empty),
            std::get<8>(empty),
            empty_reverse_dense,
            empty_reverse_masks,
            empty_coords,
            empty_hash};
    }

    auto sorted_keys = torch::empty({candidate_count}, key_opts);
    size_t sort_temp_bytes = 0;
    cub::DeviceRadixSort::SortKeys(
        nullptr,
        sort_temp_bytes,
        candidate_keys.data_ptr<int64_t>(),
        sorted_keys.data_ptr<int64_t>(),
        candidate_count,
        0,
        64,
        stream);
    auto sort_temp = torch::empty({static_cast<int64_t>(sort_temp_bytes)}, in_coords.options().dtype(torch::kUInt8));
    cub::DeviceRadixSort::SortKeys(
        sort_temp.data_ptr(),
        sort_temp_bytes,
        candidate_keys.data_ptr<int64_t>(),
        sorted_keys.data_ptr<int64_t>(),
        candidate_count,
        0,
        64,
        stream);

    auto unique_keys = torch::empty({candidate_count}, key_opts);
    auto num_unique_t = torch::zeros({1}, int_opts);
    size_t unique_temp_bytes = 0;
    cub::DeviceSelect::Unique(
        nullptr,
        unique_temp_bytes,
        sorted_keys.data_ptr<int64_t>(),
        unique_keys.data_ptr<int64_t>(),
        num_unique_t.data_ptr<int>(),
        candidate_count,
        stream);
    auto unique_temp = torch::empty({static_cast<int64_t>(unique_temp_bytes)}, in_coords.options().dtype(torch::kUInt8));
    cub::DeviceSelect::Unique(
        unique_temp.data_ptr(),
        unique_temp_bytes,
        sorted_keys.data_ptr<int64_t>(),
        unique_keys.data_ptr<int64_t>(),
        num_unique_t.data_ptr<int>(),
        candidate_count,
        stream);
    C10_CUDA_CHECK(cudaStreamSynchronize(stream));

    const int n_out = num_unique_t.item<int>();
    if (n_out <= 0) {
        auto empty = make_empty_runtime_tensors(int_opts);
        auto empty_coords = torch::empty({0, 4}, int_opts);
        auto empty_hash = torch::empty({0, 2}, key_opts);
        auto empty_reverse_dense = torch::empty({0, kNumLogicalOffsets}, int_opts);
        auto empty_reverse_masks = torch::empty({0}, int_opts);
        return {
            std::get<0>(empty),
            std::get<1>(empty),
            std::get<2>(empty),
            std::get<3>(empty),
            std::get<4>(empty),
            std::get<5>(empty),
            std::get<6>(empty),
            std::get<7>(empty),
            std::get<8>(empty),
            empty_reverse_dense,
            empty_reverse_masks,
            empty_coords,
            empty_hash};
    }

    const int bm = static_cast<int>(max_bm);
    const int template_stride = ((n_out + bm - 1) / bm) * bm;
    auto out_coords = torch::empty({n_out, 4}, int_opts);
    auto template_counts = torch::zeros({kNumTemplates}, int_opts);
    auto template_out_rows = torch::empty({kNumTemplates, template_stride}, int_opts);
    auto input_rows_w1 = torch::empty({kFamilyW1Templates, template_stride, kPayloadWidthW1}, int_opts);
    auto input_rows_w9 = torch::empty({kFamilyW9Templates, template_stride, kPayloadWidthW9}, int_opts);
    auto input_rows_w18 = torch::empty({kFamilyW18Templates, template_stride, kPayloadWidthW18}, int_opts);
    auto input_rows_w27 = torch::empty({kFamilyW27Templates, template_stride, kPayloadWidthW27}, int_opts);
    auto reverse_dense_out_in_map = torch::empty({0, kNumLogicalOffsets}, int_opts);
    auto reverse_masks = torch::empty({0}, int_opts);
    int* reverse_dense_ptr = nullptr;
    int* reverse_masks_ptr = nullptr;
    if (build_reverse_cache) {
        reverse_dense_out_in_map = torch::empty({n, kNumLogicalOffsets}, int_opts);
        reverse_masks = torch::empty({n}, int_opts);
        C10_CUDA_CHECK(cudaMemsetAsync(
            reverse_dense_out_in_map.data_ptr<int>(),
            0xFF,
            static_cast<size_t>(n) * static_cast<size_t>(kNumLogicalOffsets) * sizeof(int),
            stream));
        C10_CUDA_CHECK(cudaMemsetAsync(
            reverse_masks.data_ptr<int>(),
            0,
            static_cast<size_t>(n) * sizeof(int),
            stream));
        reverse_dense_ptr = reverse_dense_out_in_map.data_ptr<int>();
        reverse_masks_ptr = reverse_masks.data_ptr<int>();
    }

    CoordHashMap map;
    CoordHashMapOwner map_owner;
    if (lookup_coord_hashmap.defined() && lookup_coord_hashmap.numel() > 0) {
        map = view_coord_hashmap(lookup_coord_hashmap);
    } else {
        map_owner = build_coord_hashmap(in_coords, stream);
        map = map_owner.map;
    }

    const dim3 gd((n_out + kBuilderWarpsPerBlock - 1) / kBuilderWarpsPerBlock);
    const char* env_min_t = std::getenv("GTSPARSE_MIN_TEMPLATE");
    const int min_template_id = env_min_t ? std::atoi(env_min_t) : 0;
    build_full_runtime_kernel<<<gd, kBuilderThreads, 0, stream>>>(
        unique_keys.data_ptr<int64_t>(),
        map,
        out_coords.data_ptr<int>(),
        reverse_dense_ptr,
        reverse_masks_ptr,
        template_counts.data_ptr<int>(),
        template_out_rows.data_ptr<int>(),
        input_rows_w1.data_ptr<int>(),
        input_rows_w9.data_ptr<int>(),
        input_rows_w18.data_ptr<int>(),
        input_rows_w27.data_ptr<int>(),
        n_out,
        oD,
        oH,
        oW,
        stride_d,
        stride_h,
        stride_w,
        pad_d,
        pad_h,
        pad_w,
        dil_d,
        dil_h,
        dil_w,
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

    auto finalized = finalize_runtime_tensors(
        template_counts,
        template_out_rows,
        input_rows_w1,
        input_rows_w9,
        input_rows_w18,
        input_rows_w27,
        n_out,
        bm,
        stream);
    auto out_map_owner = build_coord_hashmap(out_coords, stream);
    return {
        std::get<0>(finalized),
        std::get<1>(finalized),
        std::get<2>(finalized),
        std::get<3>(finalized),
        std::get<4>(finalized),
        std::get<5>(finalized),
        std::get<6>(finalized),
        std::get<7>(finalized),
        std::get<8>(finalized),
        reverse_dense_out_in_map,
        reverse_masks,
        out_coords,
        out_map_owner.buckets};
}

void register_gtsparse3d_finalize_row_template_center_last_build_full_runtime_cuda(pybind11::module& m) {
    m.def(
        "gtsparse3d_finalize_row_template_center_last_build_full_runtime_from_coords",
        &build_finalize_row_template_center_last_full_runtime_from_coords,
        py::arg("in_coords"),
        py::arg("oD"),
        py::arg("oH"),
        py::arg("oW"),
        py::arg("stride_d"),
        py::arg("stride_h"),
        py::arg("stride_w"),
        py::arg("pad_d"),
        py::arg("pad_h"),
        py::arg("pad_w"),
        py::arg("dil_d"),
        py::arg("dil_h"),
        py::arg("dil_w"),
        py::arg("max_bm"),
        py::arg("lookup_coord_hashmap") = torch::Tensor(),
        py::arg("build_reverse_cache") = false,
        py::arg("sorted") = false);
}
