#include "api.h"

#include <cstdlib>
#include <pybind11/pybind11.h>

#include "builder_common.cuh"

namespace py = pybind11;

namespace {

using namespace gtsparse_row_template_center_last_builder;

// Private SubM builder for the center-last runtime contract.

static __global__ void build_subm_runtime_kernel(
    const int* __restrict__ coords,
    CoordHashMap map,
    int* __restrict__ template_counts,
    int* __restrict__ template_out_rows,
    int* __restrict__ input_rows_w1,
    int* __restrict__ input_rows_w9,
    int* __restrict__ input_rows_w18,
    int* __restrict__ input_rows_w27,
    int n,
    int template_stride,
    int min_template_id) {
    const int warp_id = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int row = blockIdx.x * kBuilderWarpsPerBlock + warp_id;
    if (row >= n) {
        return;
    }

    __shared__ int s_probed_rows[kBuilderThreads];
    const int b = coords[row * 4 + 0];
    const int d = coords[row * 4 + 1];
    const int h = coords[row * 4 + 2];
    const int w = coords[row * 4 + 3];

    int probed_input_row = -1;
    bool active = false;
    if (lane < kNumLogicalOffsets) {
        const int rd = lane / 9;
        const int rem = lane - rd * 9;
        const int rh = rem / 3;
        const int rw = rem - rh * 3;
        probed_input_row = map.lookup(b, d + rd - 1, h + rh - 1, w + rw - 1);
        active = probed_input_row >= 0;
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
    torch::Tensor>
build_finalize_row_template_center_last_runtime_from_coords(
    torch::Tensor coords,
    int max_bm,
    torch::Tensor coord_hashmap,
    bool sorted) {
#ifndef NDEBUG
    TORCH_CHECK(coords.is_cuda(), "coords must be a CUDA tensor");
    TORCH_CHECK(coords.is_contiguous(), "coords must be contiguous");
    TORCH_CHECK(coords.scalar_type() == at::kInt, "coords must be int32");
    TORCH_CHECK(coords.dim() == 2 && coords.size(1) == 4, "coords must be [N, 4]");
    TORCH_CHECK(max_bm > 0, "max_bm must be positive");
#endif

    c10::cuda::CUDAGuard guard(coords.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    const int n = static_cast<int>(coords.size(0));
    auto int_opts = coords.options().dtype(torch::kInt32);
    if (n == 0) {
        auto empty = make_empty_runtime_tensors(int_opts);
        auto empty_hash = torch::empty({0, 2}, coords.options().dtype(torch::kInt64));
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
            empty_hash};
    }

    const int bm = static_cast<int>(max_bm);
    const int template_stride = ((n + bm - 1) / bm) * bm;
    auto template_counts = torch::zeros({kNumTemplates}, int_opts);
    auto template_out_rows = torch::empty({kNumTemplates, template_stride}, int_opts);
    auto input_rows_w1 = torch::empty({kFamilyW1Templates, template_stride, kPayloadWidthW1}, int_opts);
    auto input_rows_w9 = torch::empty({kFamilyW9Templates, template_stride, kPayloadWidthW9}, int_opts);
    auto input_rows_w18 = torch::empty({kFamilyW18Templates, template_stride, kPayloadWidthW18}, int_opts);
    auto input_rows_w27 = torch::empty({kFamilyW27Templates, template_stride, kPayloadWidthW27}, int_opts);

    CoordHashMap map;
    CoordHashMapOwner map_owner;
    torch::Tensor runtime_hashmap;
    if (coord_hashmap.defined() && coord_hashmap.numel() > 0) {
        map = view_coord_hashmap(coord_hashmap);
        runtime_hashmap = coord_hashmap;
    } else {
        map_owner = build_coord_hashmap(coords, stream);
        map = map_owner.map;
        runtime_hashmap = map_owner.buckets;
    }

    const dim3 gd((n + kBuilderWarpsPerBlock - 1) / kBuilderWarpsPerBlock);
    const char* env_min_t = std::getenv("GTSPARSE_MIN_TEMPLATE");
    const int min_template_id = env_min_t ? std::atoi(env_min_t) : 0;
    build_subm_runtime_kernel<<<gd, kBuilderThreads, 0, stream>>>(
        coords.data_ptr<int>(),
        map,
        template_counts.data_ptr<int>(),
        template_out_rows.data_ptr<int>(),
        input_rows_w1.data_ptr<int>(),
        input_rows_w9.data_ptr<int>(),
        input_rows_w18.data_ptr<int>(),
        input_rows_w27.data_ptr<int>(),
        n,
        template_stride,
        min_template_id);
    if (sorted) {
        sort_runtime_rows_by_local_mask_(
            template_counts,
            n,
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
        n,
        bm,
        stream);
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
        runtime_hashmap};
}

void register_gtsparse3d_finalize_row_template_center_last_build_runtime_cuda(pybind11::module& m) {
    m.def(
        "gtsparse3d_finalize_row_template_center_last_build_runtime_from_coords",
        &build_finalize_row_template_center_last_runtime_from_coords,
        py::arg("coords"),
        py::arg("max_bm"),
        py::arg("coord_hashmap") = torch::Tensor(),
        py::arg("sorted") = false);
}
