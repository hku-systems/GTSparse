#include "api.h"

#include <pybind11/pybind11.h>

#include "builder_common.cuh"

namespace py = pybind11;

namespace {

using namespace gtsparse_row_template_center_last_builder;

// Private reverse-conv builder for the center-last runtime contract.

static __global__ void build_reverse_runtime_kernel(
    const int* __restrict__ out_coords,
    CoordHashMap map,
    int* __restrict__ template_counts,
    int* __restrict__ template_out_rows,
    int* __restrict__ input_rows_w1,
    int* __restrict__ input_rows_w9,
    int* __restrict__ input_rows_w18,
    int* __restrict__ input_rows_w27,
    int n_out,
    int stride_d,
    int stride_h,
    int stride_w,
    int pad_d,
    int pad_h,
    int pad_w,
    int dil_d,
    int dil_h,
    int dil_w,
    int template_stride) {
    const int warp_id = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int row = blockIdx.x * kBuilderWarpsPerBlock + warp_id;
    if (row >= n_out) {
        return;
    }

    __shared__ int s_probed_rows[kBuilderThreads];
    const int b = out_coords[row * 4 + 0];
    const int d = out_coords[row * 4 + 1];
    const int h = out_coords[row * 4 + 2];
    const int w = out_coords[row * 4 + 3];

    int probed_input_row = -1;
    bool active = false;
    if (lane < kNumLogicalOffsets) {
        const int rd = lane / 9;
        const int rem = lane - rd * 9;
        const int rh = rem / 3;
        const int rw = rem - rh * 3;
        const int nd_num = d + pad_d - rd * dil_d;
        const int nh_num = h + pad_h - rh * dil_h;
        const int nw_num = w + pad_w - rw * dil_w;
        if (nd_num % stride_d == 0 && nh_num % stride_h == 0 && nw_num % stride_w == 0) {
            const int nd = nd_num / stride_d;
            const int nh = nh_num / stride_h;
            const int nw = nw_num / stride_w;
            probed_input_row = map.lookup(b, nd, nh, nw);
            active = probed_input_row >= 0;
        }
    }
    s_probed_rows[threadIdx.x] = probed_input_row;
    __syncwarp();
    const int* warp_probed_rows = s_probed_rows + (threadIdx.x & ~31);

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
build_finalize_row_template_center_last_reverse_runtime_from_coords(
    torch::Tensor lookup_coords,
    torch::Tensor out_coords,
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
    bool sorted) {
#ifndef NDEBUG
    TORCH_CHECK(lookup_coords.is_cuda(), "lookup_coords must be a CUDA tensor");
    TORCH_CHECK(out_coords.is_cuda(), "out_coords must be a CUDA tensor");
    TORCH_CHECK(lookup_coords.is_contiguous(), "lookup_coords must be contiguous");
    TORCH_CHECK(out_coords.is_contiguous(), "out_coords must be contiguous");
    TORCH_CHECK(lookup_coords.scalar_type() == at::kInt, "lookup_coords must be int32");
    TORCH_CHECK(out_coords.scalar_type() == at::kInt, "out_coords must be int32");
    TORCH_CHECK(lookup_coords.dim() == 2 && lookup_coords.size(1) == 4, "lookup_coords must be [N, 4]");
    TORCH_CHECK(out_coords.dim() == 2 && out_coords.size(1) == 4, "out_coords must be [N, 4]");
    TORCH_CHECK(max_bm > 0, "max_bm must be positive");
#endif

    c10::cuda::CUDAGuard guard(out_coords.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    const int n_out = static_cast<int>(out_coords.size(0));
    auto int_opts = out_coords.options().dtype(torch::kInt32);
    if (n_out == 0) {
        auto empty = make_empty_runtime_tensors(int_opts);
        auto empty_hash = torch::empty({0, 2}, out_coords.options().dtype(torch::kInt64));
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
    const int template_stride = ((n_out + bm - 1) / bm) * bm;
    auto template_counts = torch::zeros({kNumTemplates}, int_opts);
    auto template_out_rows = torch::empty({kNumTemplates, template_stride}, int_opts);
    auto input_rows_w1 = torch::empty({kFamilyW1Templates, template_stride, kPayloadWidthW1}, int_opts);
    auto input_rows_w9 = torch::empty({kFamilyW9Templates, template_stride, kPayloadWidthW9}, int_opts);
    auto input_rows_w18 = torch::empty({kFamilyW18Templates, template_stride, kPayloadWidthW18}, int_opts);
    auto input_rows_w27 = torch::empty({kFamilyW27Templates, template_stride, kPayloadWidthW27}, int_opts);

    CoordHashMap map;
    CoordHashMapOwner map_owner;
    if (lookup_coord_hashmap.defined() && lookup_coord_hashmap.numel() > 0) {
        map = view_coord_hashmap(lookup_coord_hashmap);
    } else {
        map_owner = build_coord_hashmap(lookup_coords, stream);
        map = map_owner.map;
    }

    const dim3 gd((n_out + kBuilderWarpsPerBlock - 1) / kBuilderWarpsPerBlock);
    build_reverse_runtime_kernel<<<gd, kBuilderThreads, 0, stream>>>(
        out_coords.data_ptr<int>(),
        map,
        template_counts.data_ptr<int>(),
        template_out_rows.data_ptr<int>(),
        input_rows_w1.data_ptr<int>(),
        input_rows_w9.data_ptr<int>(),
        input_rows_w18.data_ptr<int>(),
        input_rows_w27.data_ptr<int>(),
        n_out,
        stride_d,
        stride_h,
        stride_w,
        pad_d,
        pad_h,
        pad_w,
        dil_d,
        dil_h,
        dil_w,
        template_stride);
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
        out_map_owner.buckets};
}

void register_gtsparse3d_finalize_row_template_center_last_build_reverse_runtime_cuda(pybind11::module& m) {
    m.def(
        "gtsparse3d_finalize_row_template_center_last_build_reverse_runtime_from_coords",
        &build_finalize_row_template_center_last_reverse_runtime_from_coords,
        py::arg("lookup_coords"),
        py::arg("out_coords"),
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
        py::arg("sorted") = false);
}
