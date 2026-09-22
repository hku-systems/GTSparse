#include "api.h"

#include <pybind11/pybind11.h>

#include "builder_common.cuh"

namespace py = pybind11;

using namespace gtsparse_row_template_center_last_builder;

// Reverse reuse builder from an existing center-last full-runtime metadata pack.

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
build_finalize_row_template_center_last_reverse_runtime_from_full_runtime(
    torch::Tensor forward_out_rows,
    torch::Tensor forward_input_rows_w1,
    torch::Tensor forward_input_rows_w9,
    torch::Tensor forward_input_rows_w18,
    torch::Tensor forward_input_rows_w27,
    torch::Tensor forward_template_ids,
    torch::Tensor forward_input_row_offsets,
    torch::Tensor forward_padded_counts,
    int reverse_n_out,
    int max_bm,
    bool sorted) {
#ifndef NDEBUG
    check_cuda_contiguous(forward_out_rows, "forward_out_rows");
    check_cuda_contiguous(forward_input_rows_w1, "forward_input_rows_w1");
    check_cuda_contiguous(forward_input_rows_w9, "forward_input_rows_w9");
    check_cuda_contiguous(forward_input_rows_w18, "forward_input_rows_w18");
    check_cuda_contiguous(forward_input_rows_w27, "forward_input_rows_w27");
    check_cuda_contiguous(forward_template_ids, "forward_template_ids");
    check_cuda_contiguous(forward_input_row_offsets, "forward_input_row_offsets");
    check_cuda_contiguous(forward_padded_counts, "forward_padded_counts");
    TORCH_CHECK(forward_out_rows.scalar_type() == at::kInt, "forward_out_rows must be int32");
    TORCH_CHECK(forward_template_ids.scalar_type() == at::kInt, "forward_template_ids must be int32");
    TORCH_CHECK(forward_input_row_offsets.scalar_type() == at::kInt, "forward_input_row_offsets must be int32");
    TORCH_CHECK(forward_padded_counts.scalar_type() == at::kInt, "forward_padded_counts must be int32");
    TORCH_CHECK(reverse_n_out >= 0, "reverse_n_out must be non-negative");
    TORCH_CHECK(max_bm > 0, "max_bm must be positive");
#endif

    c10::cuda::CUDAGuard guard(forward_out_rows.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    auto int_opts = forward_out_rows.options().dtype(torch::kInt32);
    if (reverse_n_out == 0) {
        return make_empty_runtime_tensors(int_opts);
    }

    const int max_global_rows = static_cast<int>(forward_template_ids.numel());
    if (max_global_rows <= 0) {
        return build_runtime_from_dense_out_in_map(
            torch::full({reverse_n_out, kNumLogicalOffsets}, -1, int_opts),
            torch::Tensor(),
            torch::Tensor(),
            max_bm,
            sorted,
            stream);
    }

    const int template_stride = static_cast<int>(forward_input_rows_w1.size(1));
    auto reverse_masks = torch::empty({reverse_n_out}, int_opts);
    C10_CUDA_CHECK(cudaMemsetAsync(
        reverse_masks.data_ptr<int>(),
        0,
        static_cast<size_t>(reverse_n_out) * sizeof(int),
        stream));
    auto template_counts = torch::empty({kNumTemplates}, int_opts);
    auto dense_out_in_map = torch::empty({reverse_n_out, kNumLogicalOffsets}, int_opts);
    auto padded_row_count = torch::empty({1}, int_opts);
    build_padded_row_count_kernel<<<1, 32, 0, stream>>>(
        forward_padded_counts.data_ptr<int>(),
        padded_row_count.data_ptr<int>());
    constexpr int kThreads = 256;
    const int build_items = max_global_rows > kNumTemplates ? max_global_rows : kNumTemplates;
    build_dense_out_in_map_from_runtime_kernel<<<(build_items + kThreads - 1) / kThreads, kThreads, 0, stream>>>(
        forward_out_rows.data_ptr<int>(),
        forward_input_rows_w1.data_ptr<int>(),
        forward_input_rows_w9.data_ptr<int>(),
        forward_input_rows_w18.data_ptr<int>(),
        forward_input_rows_w27.data_ptr<int>(),
        forward_template_ids.data_ptr<int>(),
        forward_input_row_offsets.data_ptr<int>(),
        padded_row_count.data_ptr<int>(),
        reverse_masks.data_ptr<int>(),
        template_counts.data_ptr<int>(),
        dense_out_in_map.data_ptr<int>(),
        reverse_n_out,
        template_stride,
        max_global_rows);
    return build_runtime_from_dense_out_in_map(
        dense_out_in_map,
        reverse_masks,
        template_counts,
        max_bm,
        sorted,
        stream);
}

void register_gtsparse3d_finalize_row_template_center_last_build_reverse_from_full_runtime_cuda(pybind11::module& m) {
    m.def(
        "gtsparse3d_finalize_row_template_center_last_build_reverse_runtime_from_full_runtime",
        &build_finalize_row_template_center_last_reverse_runtime_from_full_runtime,
        py::arg("forward_out_rows"),
        py::arg("forward_input_rows_w1"),
        py::arg("forward_input_rows_w9"),
        py::arg("forward_input_rows_w18"),
        py::arg("forward_input_rows_w27"),
        py::arg("forward_template_ids"),
        py::arg("forward_input_row_offsets"),
        py::arg("forward_padded_counts"),
        py::arg("reverse_n_out"),
        py::arg("max_bm"),
        py::arg("sorted") = false);
}
