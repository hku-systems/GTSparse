#include "api.h"
#include <cub/cub.cuh>
#include <pybind11/pybind11.h>
#include "builder_k8.cuh"

namespace py=pybind11;
namespace {
using namespace gtsparse_kernel8;
using namespace gtsparse_kernel8_builder;

static __global__ void enumerate_kernel(
    const int* coords,
    int64_t* keys,
    int* invalid_count,
    int n,
    int D,
    int H,
    int W) {
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= n) {
        return;
    }
    const int b = coords[row * 4];
    const int d = coords[row * 4 + 1] >> 1;
    const int h = coords[row * 4 + 2] >> 1;
    const int w = coords[row * 4 + 3] >> 1;
    if (d < D && h < H && w < W) {
        keys[row] = key(b, d, h, w, D, H, W);
    } else {
        keys[row] = 0x7fffffffffffffffLL;
        atomicAdd(invalid_count, 1);
    }
}

static __global__ void build_full_kernel(const int64_t* keys,CoordHashMap map,int* out_coords,
    int* counts,int* template_out,int* w1,int* w2,int* w4,int* w8,int n,int D,int H,int W,
    int sd,int sh,int sw,int pd,int ph,int pw,int dd,int dh,int dw,int stride){
    const int warp=threadIdx.x>>5,lane=threadIdx.x&31,row=blockIdx.x*kBuilderWarpsPerBlock+warp;if(row>=n)return;
    __shared__ int probed[kBuilderThreads];int b=0,d=0,h=0,w=0;
    if(lane==0){decode(keys[row],D,H,W,b,d,h,w);out_coords[row*4]=b;out_coords[row*4+1]=d;out_coords[row*4+2]=h;out_coords[row*4+3]=w;}
    b=__shfl_sync(0xffffffffu,b,0);d=__shfl_sync(0xffffffffu,d,0);h=__shfl_sync(0xffffffffu,h,0);w=__shfl_sync(0xffffffffu,w,0);
    int input=-1;bool active=false;if(lane<8){const int rd=lane/4,rh=(lane/2)%2,rw=lane%2;
        input=map.lookup(b,d*sd+rd*dd-pd,h*sh+rh*dh-ph,w*sw+rw*dw-pw);active=input>=0;}
    probed[threadIdx.x]=input;__syncwarp();const int* warp_rows=probed+(threadIdx.x&~31);
    const unsigned mask=__ballot_sync(0xffffffffu,active);int id=-1,pos=-1;
    if(lane==0){id=classify(mask);pos=atomicAdd(counts+id,1);}id=__shfl_sync(0xffffffffu,id,0);pos=__shfl_sync(0xffffffffu,pos,0);
    scatter_payload(id,pos,stride,row,warp_rows,template_out,w1,w2,w4,w8);
}

static __global__ void build_reverse_kernel(const int* targets,CoordHashMap map,int* counts,int* template_out,
    int* w1,int* w2,int* w4,int* w8,int n,int sd,int sh,int sw,int pd,int ph,int pw,int dd,int dh,int dw,int stride){
    const int warp=threadIdx.x>>5,lane=threadIdx.x&31,row=blockIdx.x*kBuilderWarpsPerBlock+warp;if(row>=n)return;
    __shared__ int probed[kBuilderThreads];const int b=targets[row*4],d=targets[row*4+1],h=targets[row*4+2],w=targets[row*4+3];
    int input=-1;bool active=false;if(lane<8){const int rd=lane/4,rh=(lane/2)%2,rw=lane%2;
        const int nd=d+pd-rd*dd,nh=h+ph-rh*dh,nw=w+pw-rw*dw;
        if(nd%sd==0&&nh%sh==0&&nw%sw==0){input=map.lookup(b,nd/sd,nh/sh,nw/sw);active=input>=0;}}
    probed[threadIdx.x]=input;__syncwarp();const int* warp_rows=probed+(threadIdx.x&~31);
    const unsigned mask=__ballot_sync(0xffffffffu,active);int id=-1,pos=-1;
    if(lane==0){id=classify(mask);pos=atomicAdd(counts+id,1);}id=__shfl_sync(0xffffffffu,id,0);pos=__shfl_sync(0xffffffffu,pos,0);
    scatter_payload(id,pos,stride,row,warp_rows,template_out,w1,w2,w4,w8);
}

struct Buffers{torch::Tensor counts,out,w1,w2,w4,w8;};
static Buffers allocate(int n,int bm,const torch::TensorOptions& opts){
    const int stride=((n+bm-1)/bm)*bm;
    return {torch::zeros({kNumTemplates},opts),torch::empty({kNumTemplates,stride},opts),
        torch::empty({kFamilyW1Templates,stride,1},opts),torch::empty({kFamilyW2Templates,stride,2},opts),
        torch::empty({kFamilyW4Templates,stride,4},opts),torch::empty({kFamilyW8Templates,stride,8},opts)};
}
}

std::tuple<torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,
    torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor>
build_kernel8_full_runtime(torch::Tensor coords,int D,int H,int W,int sd,int sh,int sw,int pd,int ph,int pw,
    int dd,int dh,int dw,int bm,torch::Tensor input_hash){
    c10::cuda::CUDAGuard guard(coords.device());cudaStream_t stream=at::cuda::getCurrentCUDAStream();
    const int n=coords.size(0);auto opts=coords.options().dtype(torch::kInt32),key_opts=coords.options().dtype(torch::kInt64);
    auto candidates=torch::empty({n},key_opts),counts=torch::zeros({2},opts);
    enumerate_kernel<<<(n+255)/256,256,0,stream>>>(coords.data_ptr<int>(),candidates.data_ptr<int64_t>(),counts.data_ptr<int>()+1,n,D,H,W);
    auto sorted=torch::empty({n},key_opts);size_t sort_bytes=0;
    cub::DeviceRadixSort::SortKeys(nullptr,sort_bytes,candidates.data_ptr<int64_t>(),sorted.data_ptr<int64_t>(),n,0,64,stream);
    auto sort_temp=torch::empty({static_cast<int64_t>(sort_bytes)},coords.options().dtype(torch::kUInt8));
    cub::DeviceRadixSort::SortKeys(sort_temp.data_ptr(),sort_bytes,candidates.data_ptr<int64_t>(),sorted.data_ptr<int64_t>(),n,0,64,stream);
    auto unique=torch::empty({n},key_opts);size_t unique_bytes=0;
    cub::DeviceSelect::Unique(nullptr,unique_bytes,sorted.data_ptr<int64_t>(),unique.data_ptr<int64_t>(),counts.data_ptr<int>(),n,stream);
    auto unique_temp=torch::empty({static_cast<int64_t>(unique_bytes)},coords.options().dtype(torch::kUInt8));
    cub::DeviceSelect::Unique(unique_temp.data_ptr(),unique_bytes,sorted.data_ptr<int64_t>(),unique.data_ptr<int64_t>(),counts.data_ptr<int>(),n,stream);
    int host_counts[2];
    C10_CUDA_CHECK(cudaMemcpyAsync(host_counts,counts.data_ptr<int>(),sizeof(host_counts),cudaMemcpyDeviceToHost,stream));
    C10_CUDA_CHECK(cudaStreamSynchronize(stream));const int n_out=host_counts[0]-(host_counts[1]>0);
    auto b=allocate(n_out,bm,opts);auto out_coords=torch::empty({n_out,4},opts);CoordHashMap map;CoordHashMapOwner owner;
    if(input_hash.defined()&&input_hash.numel())map=view_coord_hashmap(input_hash);else{owner=build_coord_hashmap(coords,stream);map=owner.map;}
    const int stride=b.w1.size(1);build_full_kernel<<<(n_out+kBuilderWarpsPerBlock-1)/kBuilderWarpsPerBlock,kBuilderThreads,0,stream>>>(
        unique.data_ptr<int64_t>(),map,out_coords.data_ptr<int>(),b.counts.data_ptr<int>(),b.out.data_ptr<int>(),b.w1.data_ptr<int>(),b.w2.data_ptr<int>(),b.w4.data_ptr<int>(),b.w8.data_ptr<int>(),
        n_out,D,H,W,sd,sh,sw,pd,ph,pw,dd,dh,dw,stride);
    auto result=finalize(b.counts,b.out,b.w1,b.w2,b.w4,b.w8,n_out,bm,stream);C10_CUDA_KERNEL_LAUNCH_CHECK();auto out_hash=build_coord_hashmap(out_coords,stream);
    return {std::get<0>(result),std::get<1>(result),std::get<2>(result),std::get<3>(result),std::get<4>(result),std::get<5>(result),std::get<6>(result),std::get<7>(result),std::get<8>(result),out_coords,out_hash.buckets};
}

std::tuple<torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,
    torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor>
build_kernel8_reverse_runtime(torch::Tensor lookup_coords,torch::Tensor target_coords,int sd,int sh,int sw,int pd,int ph,int pw,
    int dd,int dh,int dw,int bm,torch::Tensor lookup_hash){
    c10::cuda::CUDAGuard guard(target_coords.device());cudaStream_t stream=at::cuda::getCurrentCUDAStream();
    const int n=target_coords.size(0);auto opts=target_coords.options().dtype(torch::kInt32);auto b=allocate(n,bm,opts);
    CoordHashMap map;CoordHashMapOwner owner;if(lookup_hash.defined()&&lookup_hash.numel())map=view_coord_hashmap(lookup_hash);else{owner=build_coord_hashmap(lookup_coords,stream);map=owner.map;}
    const int stride=b.w1.size(1);build_reverse_kernel<<<(n+kBuilderWarpsPerBlock-1)/kBuilderWarpsPerBlock,kBuilderThreads,0,stream>>>(
        target_coords.data_ptr<int>(),map,b.counts.data_ptr<int>(),b.out.data_ptr<int>(),b.w1.data_ptr<int>(),b.w2.data_ptr<int>(),b.w4.data_ptr<int>(),b.w8.data_ptr<int>(),
        n,sd,sh,sw,pd,ph,pw,dd,dh,dw,stride);
    auto result=finalize(b.counts,b.out,b.w1,b.w2,b.w4,b.w8,n,bm,stream);C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {std::get<0>(result),std::get<1>(result),std::get<2>(result),std::get<3>(result),std::get<4>(result),std::get<5>(result),std::get<6>(result),std::get<7>(result),std::get<8>(result),target_coords,torch::empty({0},opts)};
}

void register_gtsparse_kernel8_cuda(pybind11::module&m){
    m.def("gtsparse_kernel8_build_full_runtime",&build_kernel8_full_runtime);
    m.def("gtsparse_kernel8_build_reverse_runtime",&build_kernel8_reverse_runtime);
    m.def("gtsparse_kernel8_fp16_forward",&kernel8_fp16_forward);
    m.def("gtsparse_kernel8_fp32_forward",&kernel8_fp32_forward);
}
