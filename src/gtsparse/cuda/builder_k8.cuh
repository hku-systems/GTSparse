#pragma once

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

#include "contract_k8.h"
#include "hashmap.cuh"

namespace gtsparse_kernel8_builder {

using namespace gtsparse_kernel8;
constexpr int kBuilderThreads = 256;
constexpr int kBuilderWarpsPerBlock = 8;

static __device__ __constant__ unsigned int kRejectMask[kNumTemplates] = {
    0xffu,
    0xfeu,0xfdu,0xfbu,0xf7u,0xefu,0xdfu,0xbfu,0x7fu,
    0xf0u,0x0fu,0xccu,0x33u,0xaau,0x55u,0x00u,
};

static __device__ __constant__ int kPayloadOffset[kNumTemplates][8] = {
    {-1,-1,-1,-1,-1,-1,-1,-1},
    {0,-1,-1,-1,-1,-1,-1,-1}, {1,-1,-1,-1,-1,-1,-1,-1},
    {2,-1,-1,-1,-1,-1,-1,-1}, {3,-1,-1,-1,-1,-1,-1,-1},
    {4,-1,-1,-1,-1,-1,-1,-1}, {5,-1,-1,-1,-1,-1,-1,-1},
    {6,-1,-1,-1,-1,-1,-1,-1}, {7,-1,-1,-1,-1,-1,-1,-1},
    {0,1,2,3,-1,-1,-1,-1}, {4,5,6,7,-1,-1,-1,-1},
    {0,1,4,5,-1,-1,-1,-1}, {2,3,6,7,-1,-1,-1,-1},
    {0,2,4,6,-1,-1,-1,-1}, {1,3,5,7,-1,-1,-1,-1},
    {0,1,2,3,4,5,6,7},
};

__device__ __forceinline__ int slot_count(int template_id) {
    if (template_id == kTemplateEmpty) return 0;
    if (template_id < kTemplateFaceBegin) return 1;
    if (template_id < kTemplateFull8) return 4;
    return 8;
}

__device__ __forceinline__ int classify(unsigned int mask) {
    if ((mask & 0xffu) == 0u) return kTemplateEmpty;
    #pragma unroll
    for (int id = kTemplateSingletonBegin; id < kNumTemplates; ++id) {
        if ((mask & kRejectMask[id]) == 0u) return id;
    }
    return kTemplateFull8;
}

template <int Width>
__device__ __forceinline__ void write_payload(
    int template_id, int row_pos, int stride, int out_row,
    const int* __restrict__ probed, int local_id,
    int* __restrict__ template_out_rows, int* __restrict__ payload) {
    const int lane = threadIdx.x & 31;
    if (lane == 0) template_out_rows[template_id * stride + row_pos] = out_row;
    if (lane < Width) {
        payload[((local_id * stride + row_pos) * Width) + lane] =
            probed[kPayloadOffset[template_id][lane]];
    }
}

__device__ __forceinline__ void scatter_payload(
    int template_id, int row_pos, int stride, int out_row,
    const int* __restrict__ probed, int* __restrict__ template_out_rows,
    int* __restrict__ w1, int* __restrict__ w2,
    int* __restrict__ w4, int* __restrict__ w8) {
    const int lane = threadIdx.x & 31;
    if (template_id == kTemplateEmpty) {
        if (lane == 0) template_out_rows[row_pos] = out_row;
    } else if (template_id < kTemplateFaceBegin) {
        write_payload<1>(template_id,row_pos,stride,out_row,probed,template_id-1,template_out_rows,w1);
    } else if (template_id < kTemplateFull8) {
        write_payload<4>(template_id,row_pos,stride,out_row,probed,template_id-kTemplateFaceBegin,template_out_rows,w4);
    } else {
        write_payload<8>(template_id,row_pos,stride,out_row,probed,0,template_out_rows,w8);
    }
}

static __global__ void layout_kernel(const int* counts, int* padded, int* bases, int bm) {
    if (blockIdx.x || threadIdx.x) return;
    int prefix = 0;
    for (int id = kNumTemplates - 1; id >= 0; --id) {
        const int value = ((counts[id] + bm - 1) / bm) * bm;
        padded[id] = value; bases[id] = prefix; prefix += value;
    }
}

struct ScatterParams {
    const int* counts; const int* padded; const int* bases; const int* template_out;
    int* out; int* w1; int* w2; int* w4; int* w8; int* ids; int* offsets;
    int stride; int max_rows;
};

template <int Width>
__device__ __forceinline__ void scatter_family(
    ScatterParams p, int id, int local_id, int* payload) {
    const int count=p.counts[id], padded=p.padded[id], base=p.bases[id];
    for (int row=threadIdx.x; row<padded; row+=blockDim.x) {
        const int global=base+row; p.ids[global]=id; p.offsets[global]=row;
        if (row<count) p.out[global]=p.template_out[id*p.stride+row];
        else {
            p.out[global]=-1;
            int* dst=payload+(local_id*p.stride+row)*Width;
            #pragma unroll
            for(int i=0;i<Width;++i) dst[i]=-1;
        }
    }
}

static __global__ void scatter_kernel(ScatterParams p) {
    const int id=blockIdx.x;
    if(id==kNumTemplates){
        const int total=p.bases[0]+p.padded[0];
        for(int row=total+threadIdx.x;row<p.max_rows;row+=blockDim.x){p.out[row]=-1;p.ids[row]=-1;p.offsets[row]=-1;}
        return;
    }
    if(p.padded[id]<=0)return;
    if(id==kTemplateEmpty){
        const int count=p.counts[id], padded=p.padded[id], base=p.bases[id];
        for(int row=threadIdx.x;row<padded;row+=blockDim.x){
            const int global=base+row;p.ids[global]=id;p.offsets[global]=row;
            p.out[global]=row<count?p.template_out[row]:-1;
        }
    } else if(id<kTemplateFaceBegin) scatter_family<1>(p,id,id-1,p.w1);
    else if(id<kTemplateFull8) scatter_family<4>(p,id,id-kTemplateFaceBegin,p.w4);
    else scatter_family<8>(p,id,0,p.w8);
}

static inline std::tuple<torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,
    torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor>
finalize(const torch::Tensor& counts,const torch::Tensor& template_out,
    const torch::Tensor& w1,const torch::Tensor& w2,const torch::Tensor& w4,const torch::Tensor& w8,
    int n,int bm,cudaStream_t stream){
    auto opts=counts.options();auto padded=torch::empty({kNumTemplates},opts);auto bases=torch::empty({kNumTemplates},opts);
    layout_kernel<<<1,128,0,stream>>>(counts.data_ptr<int>(),padded.data_ptr<int>(),bases.data_ptr<int>(),bm);
    const int max_rows=((n+kNumTemplates*bm+bm-1)/bm)*bm;
    auto out=torch::empty({max_rows},opts),ids=torch::empty({max_rows},opts),offsets=torch::empty({max_rows},opts);
    ScatterParams p{counts.data_ptr<int>(),padded.data_ptr<int>(),bases.data_ptr<int>(),template_out.data_ptr<int>(),
        out.data_ptr<int>(),w1.data_ptr<int>(),w2.data_ptr<int>(),w4.data_ptr<int>(),w8.data_ptr<int>(),
        ids.data_ptr<int>(),offsets.data_ptr<int>(),static_cast<int>(w1.size(1)),max_rows};
    scatter_kernel<<<kNumTemplates+1,256,0,stream>>>(p);
    return {out,w1,w2,w4,w8,ids,offsets,counts,padded};
}

__device__ __forceinline__ int64_t key(int b,int d,int h,int w,int D,int H,int W){return (((static_cast<int64_t>(b)*D+d)*H+h)*W+w);}
__device__ __forceinline__ void decode(int64_t value,int D,int H,int W,int&b,int&d,int&h,int&w){
    const int64_t hw=static_cast<int64_t>(H)*W,dhw=static_cast<int64_t>(D)*hw;
    b=value/dhw;value-=static_cast<int64_t>(b)*dhw;d=value/hw;value-=static_cast<int64_t>(d)*hw;h=value/W;w=value-static_cast<int64_t>(h)*W;
}

}  // namespace gtsparse_kernel8_builder
