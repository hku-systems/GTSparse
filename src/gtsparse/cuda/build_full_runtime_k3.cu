#include "api.h"

#include <cub/cub.cuh>
#include <pybind11/pybind11.h>

#include "builder_k3.cuh"

namespace py = pybind11;

namespace {

using namespace gtsparse_kernel3;
using namespace gtsparse_kernel3_builder;

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
    int dil_d) {
    using BlockScan = cub::BlockScan<int, 256>;
    __shared__ typename BlockScan::TempStorage scan_storage;
    __shared__ int block_base;
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t local_keys[kNumLogicalOffsets];
    int local_count = 0;

    if (row < n_in) {
        const int b = in_coords[row * 4 + 0];
        const int d = in_coords[row * 4 + 1];
        const int h = in_coords[row * 4 + 2];
        const int w = in_coords[row * 4 + 3];
        #pragma unroll
        for (int offset = 0; offset < kNumLogicalOffsets; ++offset) {
            int od = d + pad_d - offset * dil_d;
            int oh = h + pad_h;
            int ow = w + pad_w;
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
    BlockScan(scan_storage).ExclusiveSum(local_count, local_base, block_count);
    if (threadIdx.x == 0) {
        block_base = block_count > 0 ? atomicAdd(total_candidates, block_count) : 0;
    }
    __syncthreads();
    #pragma unroll
    for (int index = 0; index < kNumLogicalOffsets; ++index) {
        if (index < local_count) {
            out_keys[block_base + local_base + index] = local_keys[index];
        }
    }
}

static __global__ void build_full_runtime_kernel(
    const int64_t* __restrict__ out_keys,
    CoordHashMap map,
    int* __restrict__ out_coords,
    int* __restrict__ template_counts,
    int* __restrict__ template_out_rows,
    int* __restrict__ input_rows_w1,
    int* __restrict__ input_rows_w2,
    int* __restrict__ input_rows_w3,
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
    int template_stride) {
    const int warp_id = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int row = blockIdx.x * kBuilderWarpsPerBlock + warp_id;
    if (row >= n_out) {
        return;
    }

    __shared__ int probed_rows[kBuilderThreads];
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

    int input_row = -1;
    bool active = false;
    if (lane < kNumLogicalOffsets) {
        const int nd = d * stride_d + lane * dil_d - pad_d;
        const int nh = h * stride_h - pad_h;
        const int nw = w * stride_w - pad_w;
        input_row = map.lookup(b, nd, nh, nw);
        active = input_row >= 0;
    }
    probed_rows[threadIdx.x] = input_row;
    __syncwarp();
    const int* warp_probed_rows = probed_rows + (threadIdx.x & ~31);

    const unsigned int active_mask = __ballot_sync(0xffffffffu, active);
    int template_id = -1;
    int row_pos = -1;
    if (lane == 0) {
        template_id = classify_template_fast(active_mask);
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
        input_rows_w2,
        input_rows_w3);
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
    torch::Tensor>
build_kernel3_full_runtime_from_coords(
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
    int max_bm,
    torch::Tensor lookup_coord_hashmap) {
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

    auto candidate_keys = torch::empty({static_cast<int64_t>(n) * kNumLogicalOffsets}, key_opts);
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
        dil_d);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    C10_CUDA_CHECK(cudaStreamSynchronize(stream));
    const int candidate_count = num_candidate_keys.item<int>();

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
    auto num_unique = torch::zeros({1}, int_opts);
    size_t unique_temp_bytes = 0;
    cub::DeviceSelect::Unique(
        nullptr,
        unique_temp_bytes,
        sorted_keys.data_ptr<int64_t>(),
        unique_keys.data_ptr<int64_t>(),
        num_unique.data_ptr<int>(),
        candidate_count,
        stream);
    auto unique_temp = torch::empty({static_cast<int64_t>(unique_temp_bytes)}, in_coords.options().dtype(torch::kUInt8));
    cub::DeviceSelect::Unique(
        unique_temp.data_ptr(),
        unique_temp_bytes,
        sorted_keys.data_ptr<int64_t>(),
        unique_keys.data_ptr<int64_t>(),
        num_unique.data_ptr<int>(),
        candidate_count,
        stream);
    C10_CUDA_CHECK(cudaStreamSynchronize(stream));
    const int n_out = num_unique.item<int>();

    const int bm = static_cast<int>(max_bm);
    const int template_stride = ((n_out + bm - 1) / bm) * bm;
    auto out_coords = torch::empty({n_out, 4}, int_opts);
    auto template_counts = torch::zeros({kNumTemplates}, int_opts);
    auto template_out_rows = torch::empty({kNumTemplates, template_stride}, int_opts);
    auto input_rows_w1 = torch::empty({kFamilyW1Templates, template_stride, kPayloadWidthW1}, int_opts);
    auto input_rows_w2 = torch::empty({kFamilyW2Templates, template_stride, kPayloadWidthW2}, int_opts);
    auto input_rows_w3 = torch::empty({kFamilyW3Templates, template_stride, kPayloadWidthW3}, int_opts);

    CoordHashMap map;
    CoordHashMapOwner map_owner;
    if (lookup_coord_hashmap.defined() && lookup_coord_hashmap.numel() > 0) {
        map = view_coord_hashmap(lookup_coord_hashmap);
    } else {
        map_owner = build_coord_hashmap(in_coords, stream);
        map = map_owner.map;
    }

    const dim3 grid((n_out + kBuilderWarpsPerBlock - 1) / kBuilderWarpsPerBlock);
    build_full_runtime_kernel<<<grid, kBuilderThreads, 0, stream>>>(
        unique_keys.data_ptr<int64_t>(),
        map,
        out_coords.data_ptr<int>(),
        template_counts.data_ptr<int>(),
        template_out_rows.data_ptr<int>(),
        input_rows_w1.data_ptr<int>(),
        input_rows_w2.data_ptr<int>(),
        input_rows_w3.data_ptr<int>(),
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
        template_stride);

    auto finalized = finalize_runtime_tensors(
        template_counts,
        template_out_rows,
        input_rows_w1,
        input_rows_w2,
        input_rows_w3,
        n_out,
        bm,
        stream);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
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
        out_coords,
        out_map_owner.buckets};
}

void register_gtsparse_kernel3_cuda(pybind11::module& m) {
    m.def(
        "gtsparse_kernel3_build_full_runtime_from_coords",
        &build_kernel3_full_runtime_from_coords,
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
        py::arg("max_bm"),
        py::arg("lookup_coord_hashmap") = torch::Tensor());
    m.def(
        "gtsparse_kernel3_fp16_forward",
        &kernel3_fp16_forward,
        py::arg("features"),
        py::arg("logical_weight"),
        py::arg("out_rows"),
        py::arg("input_rows_w1"),
        py::arg("input_rows_w2"),
        py::arg("input_rows_w3"),
        py::arg("template_ids"),
        py::arg("input_row_offsets"),
        py::arg("n_out"));
    m.def(
        "gtsparse_kernel3_fp32_forward",
        &kernel3_fp32_forward,
        py::arg("features"),
        py::arg("logical_weight"),
        py::arg("out_rows"),
        py::arg("input_rows_w1"),
        py::arg("input_rows_w2"),
        py::arg("input_rows_w3"),
        py::arg("template_ids"),
        py::arg("input_row_offsets"),
        py::arg("n_out"));
}
