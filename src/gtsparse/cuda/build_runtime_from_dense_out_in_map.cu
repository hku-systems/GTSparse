#include "api.h"

#include <pybind11/pybind11.h>

#include "builder_common.cuh"

namespace py = pybind11;

using namespace gtsparse_row_template_center_last_builder;

std::tuple<
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor>
build_finalize_row_template_center_last_runtime_from_dense_out_in_map(
    torch::Tensor dense_out_in_map,
    torch::Tensor dense_masks,
    int max_bm,
    bool sorted) {
    c10::cuda::CUDAGuard guard(dense_out_in_map.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    if (!dense_masks.defined() || dense_masks.numel() == 0) {
        dense_masks = torch::Tensor();
    }
    return build_runtime_from_dense_out_in_map(
        dense_out_in_map,
        dense_masks,
        torch::Tensor(),
        max_bm,
        sorted,
        stream);
}

void register_gtsparse3d_finalize_row_template_center_last_build_runtime_from_dense_out_in_map_cuda(pybind11::module& m) {
    m.def(
        "gtsparse3d_finalize_row_template_center_last_build_runtime_from_dense_out_in_map",
        &build_finalize_row_template_center_last_runtime_from_dense_out_in_map,
        py::arg("dense_out_in_map"),
        py::arg("dense_masks") = torch::Tensor(),
        py::arg("max_bm"),
        py::arg("sorted") = false);
}
