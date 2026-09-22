#include "api.h"
#include "contract_k8.h"
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime.h>

namespace {
using namespace gtsparse_kernel8;
struct P{const float*features;const float*weight;const int*out;const int*w1;const int*w2;const int*w4;const int*w8;const int*ids;const int*offsets;float*output;int cin,cout,rows,stride;};
static __device__ __constant__ int kOff[kNumTemplates][8]={
 {-1,-1,-1,-1,-1,-1,-1,-1},{0,-1,-1,-1,-1,-1,-1,-1},{1,-1,-1,-1,-1,-1,-1,-1},{2,-1,-1,-1,-1,-1,-1,-1},{3,-1,-1,-1,-1,-1,-1,-1},{4,-1,-1,-1,-1,-1,-1,-1},{5,-1,-1,-1,-1,-1,-1,-1},{6,-1,-1,-1,-1,-1,-1,-1},{7,-1,-1,-1,-1,-1,-1,-1},
 {0,1,2,3,-1,-1,-1,-1},{4,5,6,7,-1,-1,-1,-1},{0,1,4,5,-1,-1,-1,-1},{2,3,6,7,-1,-1,-1,-1},{0,2,4,6,-1,-1,-1,-1},{1,3,5,7,-1,-1,-1,-1},{0,1,2,3,4,5,6,7}};
__device__ __forceinline__ int slots(int id){if(id==0)return 0;if(id<9)return 1;if(id<15)return 4;return 8;}
__device__ __forceinline__ int width(int id){return slots(id);}
__device__ __forceinline__ const int* rows(const P&p,int id,int offset,int seed){
 if(id==0)return p.w1+offset+seed;if(id<9)return p.w1+((id-1)*p.stride+offset+seed);
 if(id<15)return p.w4+((id-9)*p.stride+offset+seed)*4;return p.w8+(offset+seed)*8;}

__device__ __forceinline__ void tile3(const P&p,int id,int outbase,int rowoff,int bn,float*A,float*B){
 const int tid=threadIdx.x,Cin=p.cin,Cout=p.cout,loops=Cin/32,count=slots(id),pitch=width(id)*16,stride=Cin*Cout;
 float C[64];
 #pragma unroll
 for(int i=0;i<64;++i)C[i]=0;
 float*Ap=A+tid*4,*Ar=A+(tid/16)*32,*Bp=B+tid*4,*Br=B+(tid%16);
 const int loc=outbase+tid/16,col0=bn+tid%16,ca=tid*4%32;const int*rp=rows(p,id,rowoff,tid/8);
 const int first=count?kOff[id][0]:0;const float*wp=p.weight+first*stride+(tid/16)*Cout+bn+tid*4%64;
 int slot=0,ci_tile=0,ci=0;
 for(int loop=0;loop<count*loops;++loop){const float*wci=wp+ci*Cout;__syncthreads();const int*ip=rp;
  #pragma unroll
  for(int r=0;r<8;++r){const int idx=*ip;if(idx>=0)*reinterpret_cast<float4*>(Ap+r*512)=*reinterpret_cast<const float4*>(p.features+idx*Cin+ci+ca);else *reinterpret_cast<float4*>(Ap+r*512)=make_float4(0,0,0,0);ip+=pitch;}
  const float*bp=wci;
  #pragma unroll
  for(int r=0;r<4;++r){*reinterpret_cast<float4*>(Bp+r*512)=*reinterpret_cast<const float4*>(bp);bp+=8*Cout;}
  __syncthreads();
  #pragma unroll
  for(int a=0;a<8;++a){
   #pragma unroll
   for(int b=0;b<4;++b){const int k=a*4+b;
    #pragma unroll
    for(int i=0;i<64;++i)C[i]+=Ar[((i/4)*8)*32+k]*Br[k*64+(i%4)*16];}}
  ++ci_tile;ci+=32;if(ci_tile==loops){ci_tile=0;ci=0;const int prev=slot++;if(slot<count){rp+=1;wp+=(kOff[id][slot]-kOff[id][prev])*stride;}}}
 #pragma unroll
 for(int i=0;i<64;++i){const int out=p.out[loc+(i/4)*8],col=col0+(i%4)*16;if(out>=0&&col<Cout)p.output[out*Cout+col]=C[i];}}

__device__ __forceinline__ void tile1(const P&p,int id,int outbase,int rowoff,int bn,float*A,float*B){
 const int tid=threadIdx.x,Cin=p.cin,Cout=p.cout,loops=Cin/16,count=slots(id),pitch=width(id)*16,stride=Cin*Cout;
 float C[32];
 #pragma unroll
 for(int i=0;i<32;++i)C[i]=0;
 float*Ap=A+tid*4,*Ar=A+(tid/4)*16,*Bp=B+tid*4,*Br=B+tid%4;
 const int loc=outbase+tid/4,col0=bn+tid%4,ca=tid*4%16;const int*rp=rows(p,id,rowoff,tid/4);
 const int first=count?kOff[id][0]:0;const float*wp=p.weight+first*stride+(tid/4)*Cout+bn+tid*4%16;
 int slot=0,ci_tile=0,ci=0;
 for(int loop=0;loop<count*loops;++loop){__syncthreads();const int*ip=rp;
  #pragma unroll
  for(int r=0;r<8;++r){const int idx=*ip;if(idx>=0)*reinterpret_cast<float4*>(Ap+r*256)=*reinterpret_cast<const float4*>(p.features+idx*Cin+ci+ca);else *reinterpret_cast<float4*>(Ap+r*256)=make_float4(0,0,0,0);ip+=pitch;}
  *reinterpret_cast<float4*>(Bp)=*reinterpret_cast<const float4*>(wp+ci*Cout);__syncthreads();
  #pragma unroll
  for(int a=0;a<4;++a){
   #pragma unroll
   for(int b=0;b<4;++b){const int k=a*4+b;
    #pragma unroll
    for(int i=0;i<32;++i)C[i]+=Ar[((i/4)*16)*16+k]*Br[k*16+(i%4)*4];}}
  ++ci_tile;ci+=16;if(ci_tile==loops){ci_tile=0;ci=0;const int prev=slot++;if(slot<count){rp+=1;wp+=(kOff[id][slot]-kOff[id][prev])*stride;}}}
 #pragma unroll
 for(int i=0;i<32;++i){const int out=p.out[loc+(i/4)*16],col=col0+(i%4)*4;if(out>=0)p.output[out*Cout+col]=C[i];}}

__global__ void kernel3(P p){__shared__ float A[4096],B[2048];const int nt=p.cout/64,logical=blockIdx.x,base=logical/nt*128,bn=logical%nt*64,id=p.ids[base];if(id<0)return;tile3(p,id,base,p.offsets[base],bn,A,B);}
__global__ void kernel1(P p){__shared__ float A[2048],B[256];const int nt=p.cout/16,logical=blockIdx.x,base=logical/nt*128,bn=logical%nt*16,id=p.ids[base];if(id<0)return;tile1(p,id,base,p.offsets[base],bn,A,B);}
}

torch::Tensor kernel8_fp32_forward(torch::Tensor f,torch::Tensor w,torch::Tensor out,torch::Tensor w1,torch::Tensor w2,torch::Tensor w4,torch::Tensor w8,torch::Tensor ids,torch::Tensor offsets,int64_t n){
 c10::cuda::CUDAGuard guard(f.device());auto y=torch::empty({n,w.size(1)},f.options());P p{f.data_ptr<float>(),w.data_ptr<float>(),out.data_ptr<int>(),w1.data_ptr<int>(),w2.data_ptr<int>(),w4.data_ptr<int>(),w8.data_ptr<int>(),ids.data_ptr<int>(),offsets.data_ptr<int>(),y.data_ptr<float>(),static_cast<int>(f.size(1)),static_cast<int>(w.size(1)),static_cast<int>(out.size(0)),static_cast<int>(w1.size(1))};
 if(p.cout%64==0){const int grid=p.rows/128*(p.cout/64);kernel3<<<grid,128,0,at::cuda::getCurrentCUDAStream()>>>(p);}else{const int grid=p.rows/128*(p.cout/16);kernel1<<<grid,64,0,at::cuda::getCurrentCUDAStream()>>>(p);}return y;}
