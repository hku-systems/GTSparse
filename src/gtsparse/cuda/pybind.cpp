#include <pybind11/pybind11.h>
void register_gtsparse3d_finalize_row_template_center_last_cuda(pybind11::module &m);
void register_gtsparse3d_finalize_row_template_center_last_build_runtime_cuda(pybind11::module &m);
void register_gtsparse3d_finalize_row_template_center_last_build_full_runtime_cuda(pybind11::module &m);
void register_gtsparse3d_finalize_row_template_center_last_build_runtime_from_dense_out_in_map_cuda(pybind11::module &m);
void register_gtsparse3d_finalize_row_template_center_last_build_reverse_runtime_cuda(pybind11::module &m);
void register_gtsparse3d_finalize_row_template_center_last_build_reverse_from_full_runtime_cuda(pybind11::module &m);
void register_gtsparse_kernel3_cuda(pybind11::module &m);
void register_gtsparse_kernel9_cuda(pybind11::module &m);
void register_gtsparse_kernel8_cuda(pybind11::module &m);


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    register_gtsparse3d_finalize_row_template_center_last_cuda(m);
    register_gtsparse3d_finalize_row_template_center_last_build_runtime_cuda(m);
    register_gtsparse3d_finalize_row_template_center_last_build_full_runtime_cuda(m);
    register_gtsparse3d_finalize_row_template_center_last_build_runtime_from_dense_out_in_map_cuda(m);
    register_gtsparse3d_finalize_row_template_center_last_build_reverse_runtime_cuda(m);
    register_gtsparse3d_finalize_row_template_center_last_build_reverse_from_full_runtime_cuda(m);
    register_gtsparse_kernel3_cuda(m);
    register_gtsparse_kernel9_cuda(m);
    register_gtsparse_kernel8_cuda(m);
}
