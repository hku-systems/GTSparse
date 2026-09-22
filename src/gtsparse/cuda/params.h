#pragma once

#include <cuda_fp16.h>

struct FinalizeRowTemplateCenterLastFP32Params {
    const float* __restrict__ features;
    const float* __restrict__ weight;
    const int* __restrict__ out_rows;
    const int* __restrict__ input_rows_w1;
    const int* __restrict__ input_rows_w9;
    const int* __restrict__ input_rows_w18;
    const int* __restrict__ input_rows_w27;
    const int* __restrict__ template_ids;
    const int* __restrict__ input_row_offsets;
    float* __restrict__ output;
    int n_out;
    int c_in;
    int c_out;
    int padded_rows;
    int template_stride;
};

using FinalizeRowTemplateCenterLastFP32Setting1Params = FinalizeRowTemplateCenterLastFP32Params;
using FinalizeRowTemplateCenterLastFP32Setting2Params = FinalizeRowTemplateCenterLastFP32Params;
using FinalizeRowTemplateCenterLastFP32Setting3Params = FinalizeRowTemplateCenterLastFP32Params;

struct FinalizeRowTemplateCenterLastFP16Params {
    const half* __restrict__ features;
    const half* __restrict__ weight;
    const int* __restrict__ out_rows;
    const int* __restrict__ input_rows_w1;
    const int* __restrict__ input_rows_w9;
    const int* __restrict__ input_rows_w18;
    const int* __restrict__ input_rows_w27;
    const int* __restrict__ template_ids;
    const int* __restrict__ input_row_offsets;
    half* __restrict__ output;
    int n_out;
    int c_in;
    int c_out;
    int padded_rows;
    int template_stride;
};

using FinalizeRowTemplateCenterLastFP16Setting1Params = FinalizeRowTemplateCenterLastFP16Params;
using FinalizeRowTemplateCenterLastFP16Setting2Params = FinalizeRowTemplateCenterLastFP16Params;
using FinalizeRowTemplateCenterLastFP16Setting3Params = FinalizeRowTemplateCenterLastFP16Params;
