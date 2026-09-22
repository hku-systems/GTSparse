#pragma once

#include <pybind11/pybind11.h>
#include <torch/extension.h>

torch::Tensor kernel3_fp16_forward(
    torch::Tensor features,
    torch::Tensor logical_weight,
    torch::Tensor out_rows,
    torch::Tensor input_rows_w1,
    torch::Tensor input_rows_w2,
    torch::Tensor input_rows_w3,
    torch::Tensor template_ids,
    torch::Tensor input_row_offsets,
    int64_t n_out);

torch::Tensor kernel3_fp32_forward(
    torch::Tensor features,
    torch::Tensor logical_weight,
    torch::Tensor out_rows,
    torch::Tensor input_rows_w1,
    torch::Tensor input_rows_w2,
    torch::Tensor input_rows_w3,
    torch::Tensor template_ids,
    torch::Tensor input_row_offsets,
    int64_t n_out);

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
    torch::Tensor lookup_coord_hashmap = torch::Tensor());

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
    int64_t n_out);

torch::Tensor kernel9_fp32_forward(
    torch::Tensor features,
    torch::Tensor logical_weight,
    torch::Tensor out_rows,
    torch::Tensor input_rows_w1,
    torch::Tensor input_rows_w4,
    torch::Tensor input_rows_w7,
    torch::Tensor input_rows_w9,
    torch::Tensor template_ids,
    torch::Tensor input_row_offsets,
    int64_t n_out);

torch::Tensor kernel8_fp16_forward(
    torch::Tensor features, torch::Tensor logical_weight, torch::Tensor out_rows,
    torch::Tensor input_rows_w1, torch::Tensor input_rows_w2,
    torch::Tensor input_rows_w4, torch::Tensor input_rows_w8,
    torch::Tensor template_ids, torch::Tensor input_row_offsets, int64_t n_out);

torch::Tensor kernel8_fp32_forward(
    torch::Tensor features, torch::Tensor logical_weight, torch::Tensor out_rows,
    torch::Tensor input_rows_w1, torch::Tensor input_rows_w2,
    torch::Tensor input_rows_w4, torch::Tensor input_rows_w8,
    torch::Tensor template_ids, torch::Tensor input_row_offsets, int64_t n_out);

torch::Tensor finalize_row_template_center_last_fp32_forward(
    torch::Tensor features,
    torch::Tensor logical_weight,
    torch::Tensor out_rows,
    torch::Tensor input_rows_w1,
    torch::Tensor input_rows_w9,
    torch::Tensor input_rows_w18,
    torch::Tensor input_rows_w27,
    torch::Tensor template_ids,
    torch::Tensor input_row_offsets,
    int64_t n_out);

torch::Tensor finalize_row_template_center_last_fp32_sorted_forward(
    torch::Tensor features,
    torch::Tensor logical_weight,
    torch::Tensor out_rows,
    torch::Tensor input_rows_w1,
    torch::Tensor input_rows_w9,
    torch::Tensor input_rows_w18,
    torch::Tensor input_rows_w27,
    torch::Tensor template_ids,
    torch::Tensor input_row_offsets,
    int64_t n_out);

torch::Tensor finalize_row_template_center_last_fp16_forward(
    torch::Tensor features,
    torch::Tensor logical_weight,
    torch::Tensor out_rows,
    torch::Tensor input_rows_w1,
    torch::Tensor input_rows_w9,
    torch::Tensor input_rows_w18,
    torch::Tensor input_rows_w27,
    torch::Tensor template_ids,
    torch::Tensor input_row_offsets,
    int64_t n_out);

torch::Tensor finalize_row_template_center_last_fp16_sorted_forward(
    torch::Tensor features,
    torch::Tensor logical_weight,
    torch::Tensor out_rows,
    torch::Tensor input_rows_w1,
    torch::Tensor input_rows_w9,
    torch::Tensor input_rows_w18,
    torch::Tensor input_rows_w27,
    torch::Tensor template_ids,
    torch::Tensor input_row_offsets,
    int64_t n_out);

torch::Tensor finalize_row_template_center_last_fp16_setting1_forward(
    torch::Tensor features,
    torch::Tensor logical_weight,
    torch::Tensor out_rows,
    torch::Tensor input_rows_w1,
    torch::Tensor input_rows_w9,
    torch::Tensor input_rows_w18,
    torch::Tensor input_rows_w27,
    torch::Tensor template_ids,
    torch::Tensor input_row_offsets,
    int64_t n_out);

torch::Tensor finalize_row_template_center_last_fp16_setting1_sorted_forward(
    torch::Tensor features,
    torch::Tensor logical_weight,
    torch::Tensor out_rows,
    torch::Tensor input_rows_w1,
    torch::Tensor input_rows_w9,
    torch::Tensor input_rows_w18,
    torch::Tensor input_rows_w27,
    torch::Tensor template_ids,
    torch::Tensor input_row_offsets,
    int64_t n_out);

torch::Tensor finalize_row_template_center_last_fp16_setting2_forward(
    torch::Tensor features,
    torch::Tensor logical_weight,
    torch::Tensor out_rows,
    torch::Tensor input_rows_w1,
    torch::Tensor input_rows_w9,
    torch::Tensor input_rows_w18,
    torch::Tensor input_rows_w27,
    torch::Tensor template_ids,
    torch::Tensor input_row_offsets,
    int64_t n_out);

torch::Tensor finalize_row_template_center_last_fp16_setting2_sorted_forward(
    torch::Tensor features,
    torch::Tensor logical_weight,
    torch::Tensor out_rows,
    torch::Tensor input_rows_w1,
    torch::Tensor input_rows_w9,
    torch::Tensor input_rows_w18,
    torch::Tensor input_rows_w27,
    torch::Tensor template_ids,
    torch::Tensor input_row_offsets,
    int64_t n_out);

torch::Tensor finalize_row_template_center_last_fp16_setting3_forward(
    torch::Tensor features,
    torch::Tensor logical_weight,
    torch::Tensor out_rows,
    torch::Tensor input_rows_w1,
    torch::Tensor input_rows_w9,
    torch::Tensor input_rows_w18,
    torch::Tensor input_rows_w27,
    torch::Tensor template_ids,
    torch::Tensor input_row_offsets,
    int64_t n_out);

torch::Tensor finalize_row_template_center_last_fp16_setting3_sorted_forward(
    torch::Tensor features,
    torch::Tensor logical_weight,
    torch::Tensor out_rows,
    torch::Tensor input_rows_w1,
    torch::Tensor input_rows_w9,
    torch::Tensor input_rows_w18,
    torch::Tensor input_rows_w27,
    torch::Tensor template_ids,
    torch::Tensor input_row_offsets,
    int64_t n_out);

torch::Tensor finalize_row_template_center_last_fp32_setting1_forward(
    torch::Tensor features,
    torch::Tensor logical_weight,
    torch::Tensor out_rows,
    torch::Tensor input_rows_w1,
    torch::Tensor input_rows_w9,
    torch::Tensor input_rows_w18,
    torch::Tensor input_rows_w27,
    torch::Tensor template_ids,
    torch::Tensor input_row_offsets,
    int64_t n_out);

torch::Tensor finalize_row_template_center_last_fp32_setting1_sorted_forward(
    torch::Tensor features,
    torch::Tensor logical_weight,
    torch::Tensor out_rows,
    torch::Tensor input_rows_w1,
    torch::Tensor input_rows_w9,
    torch::Tensor input_rows_w18,
    torch::Tensor input_rows_w27,
    torch::Tensor template_ids,
    torch::Tensor input_row_offsets,
    int64_t n_out);

torch::Tensor finalize_row_template_center_last_fp32_setting2_forward(
    torch::Tensor features,
    torch::Tensor logical_weight,
    torch::Tensor out_rows,
    torch::Tensor input_rows_w1,
    torch::Tensor input_rows_w9,
    torch::Tensor input_rows_w18,
    torch::Tensor input_rows_w27,
    torch::Tensor template_ids,
    torch::Tensor input_row_offsets,
    int64_t n_out);

torch::Tensor finalize_row_template_center_last_fp32_setting2_sorted_forward(
    torch::Tensor features,
    torch::Tensor logical_weight,
    torch::Tensor out_rows,
    torch::Tensor input_rows_w1,
    torch::Tensor input_rows_w9,
    torch::Tensor input_rows_w18,
    torch::Tensor input_rows_w27,
    torch::Tensor template_ids,
    torch::Tensor input_row_offsets,
    int64_t n_out);

torch::Tensor finalize_row_template_center_last_fp32_setting3_forward(
    torch::Tensor features,
    torch::Tensor logical_weight,
    torch::Tensor out_rows,
    torch::Tensor input_rows_w1,
    torch::Tensor input_rows_w9,
    torch::Tensor input_rows_w18,
    torch::Tensor input_rows_w27,
    torch::Tensor template_ids,
    torch::Tensor input_row_offsets,
    int64_t n_out);

torch::Tensor finalize_row_template_center_last_fp32_setting3_sorted_forward(
    torch::Tensor features,
    torch::Tensor logical_weight,
    torch::Tensor out_rows,
    torch::Tensor input_rows_w1,
    torch::Tensor input_rows_w9,
    torch::Tensor input_rows_w18,
    torch::Tensor input_rows_w27,
    torch::Tensor template_ids,
    torch::Tensor input_row_offsets,
    int64_t n_out);

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
    torch::Tensor coord_hashmap = torch::Tensor(),
    bool sorted = false);

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
    torch::Tensor lookup_coord_hashmap = torch::Tensor(),
    bool build_reverse_cache = false,
    bool sorted = false);

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
    bool sorted = false);

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
    torch::Tensor lookup_coord_hashmap = torch::Tensor(),
    bool sorted = false);

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
    bool sorted = false);

void register_gtsparse3d_finalize_row_template_center_last_cuda(pybind11::module& m);
void register_gtsparse3d_finalize_row_template_center_last_build_runtime_cuda(pybind11::module& m);
void register_gtsparse3d_finalize_row_template_center_last_build_full_runtime_cuda(pybind11::module& m);
void register_gtsparse3d_finalize_row_template_center_last_build_runtime_from_dense_out_in_map_cuda(pybind11::module& m);
void register_gtsparse3d_finalize_row_template_center_last_build_reverse_runtime_cuda(pybind11::module& m);
void register_gtsparse3d_finalize_row_template_center_last_build_reverse_from_full_runtime_cuda(pybind11::module& m);
void register_gtsparse_kernel3_cuda(pybind11::module& m);
void register_gtsparse_kernel9_cuda(pybind11::module& m);
void register_gtsparse_kernel8_cuda(pybind11::module& m);
