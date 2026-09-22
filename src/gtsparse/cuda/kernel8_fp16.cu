#include "api.h"
#include "contract_k8.h"
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace {
using namespace gtsparse_kernel8;
struct P{const half*features;const half*weight;const int*out;const int*w1;const int*w2;const int*w4;const int*w8;const int*ids;const int*offsets;half*output;int cin,cout,rows,stride;};
static __device__ __constant__ int kOff[kNumTemplates][8]={
 {-1,-1,-1,-1,-1,-1,-1,-1},{0,-1,-1,-1,-1,-1,-1,-1},{1,-1,-1,-1,-1,-1,-1,-1},{2,-1,-1,-1,-1,-1,-1,-1},{3,-1,-1,-1,-1,-1,-1,-1},{4,-1,-1,-1,-1,-1,-1,-1},{5,-1,-1,-1,-1,-1,-1,-1},{6,-1,-1,-1,-1,-1,-1,-1},{7,-1,-1,-1,-1,-1,-1,-1},
 {0,1,2,3,-1,-1,-1,-1},{4,5,6,7,-1,-1,-1,-1},{0,1,4,5,-1,-1,-1,-1},{2,3,6,7,-1,-1,-1,-1},{0,2,4,6,-1,-1,-1,-1},{1,3,5,7,-1,-1,-1,-1},{0,1,2,3,4,5,6,7}};
__device__ __forceinline__ int slots(int id){if(id==0)return 0;if(id<9)return 1;if(id<15)return 4;return 8;}
__device__ __forceinline__ const int* rows(const P&p,int id,int offset,int seed){
 if(id==0)return p.w1+offset+seed;if(id<9)return p.w1+((id-1)*p.stride+offset+seed);
 if(id<15)return p.w4+((id-9)*p.stride+offset+seed)*4;return p.w8+(offset+seed)*8;}
__device__ __forceinline__ void mma(float*c,const half*ah,const half*bh){
#if __CUDA_ARCH__ >= 800
 const unsigned*a=reinterpret_cast<const unsigned*>(ah),*b=reinterpret_cast<const unsigned*>(bh);
 asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13};":"=f"(c[0]),"=f"(c[1]),"=f"(c[2]),"=f"(c[3]):"r"(a[0]),"r"(a[1]),"r"(a[2]),"r"(a[3]),"r"(b[0]),"r"(b[1]),"f"(c[0]),"f"(c[1]),"f"(c[2]),"f"(c[3]));
 asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13};":"=f"(c[4]),"=f"(c[5]),"=f"(c[6]),"=f"(c[7]):"r"(a[0]),"r"(a[1]),"r"(a[2]),"r"(a[3]),"r"(b[2]),"r"(b[3]),"f"(c[4]),"f"(c[5]),"f"(c[6]),"f"(c[7]));
#elif __CUDA_ARCH__ >= 750
 const unsigned*a0=reinterpret_cast<const unsigned*>(ah),*a1=reinterpret_cast<const unsigned*>(ah+4),*b0=reinterpret_cast<const unsigned*>(bh),*b4=reinterpret_cast<const unsigned*>(bh+4),*b2=reinterpret_cast<const unsigned*>(bh+2),*b6=reinterpret_cast<const unsigned*>(bh+6);
 asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32{%0,%1,%2,%3},{%4,%5},{%6},{%7,%8,%9,%10};":"=f"(c[0]),"=f"(c[1]),"=f"(c[2]),"=f"(c[3]):"r"(a0[0]),"r"(a0[1]),"r"(b0[0]),"f"(c[0]),"f"(c[1]),"f"(c[2]),"f"(c[3]));
 asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32{%0,%1,%2,%3},{%4,%5},{%6},{%7,%8,%9,%10};":"=f"(c[4]),"=f"(c[5]),"=f"(c[6]),"=f"(c[7]):"r"(a0[0]),"r"(a0[1]),"r"(b4[0]),"f"(c[4]),"f"(c[5]),"f"(c[6]),"f"(c[7]));
 asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32{%0,%1,%2,%3},{%4,%5},{%6},{%7,%8,%9,%10};":"=f"(c[0]),"=f"(c[1]),"=f"(c[2]),"=f"(c[3]):"r"(a1[0]),"r"(a1[1]),"r"(b2[0]),"f"(c[0]),"f"(c[1]),"f"(c[2]),"f"(c[3]));
 asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32{%0,%1,%2,%3},{%4,%5},{%6},{%7,%8,%9,%10};":"=f"(c[4]),"=f"(c[5]),"=f"(c[6]),"=f"(c[7]):"r"(a1[0]),"r"(a1[1]),"r"(b6[0]),"f"(c[4]),"f"(c[5]),"f"(c[6]),"f"(c[7]));
#endif
}

__device__ __forceinline__ void tile3(const P&p,int id,int outbase,int rowoff,int bn,half*A,half*B){
 const int tid=threadIdx.x,x=tid&31,y=tid>>5,Cin=p.cin,Cout=p.cout,loops=Cin/32,count=slots(id),pitch=count*32,stride=Cin*Cout;
 float C[64];half af[32],bf[16];
 #pragma unroll
 for(int i=0;i<64;++i)C[i]=0;
 const int seed=y*8+x/4;const int*rp=rows(p,id,rowoff,seed);const int first=count?kOff[id][0]:0;
 const half*wp=p.weight+first*stride+bn+((y<<2)+(x>>3))*Cout+(x&7)*8;const half*fp=p.features+(x&3)*8;
 const int reorder=(y%2)*64+x/4,colbase=bn+y/2*32+(x%4)*2;int slot=0,ci_tile=0,ci=0;
 for(int loop=0;loop<count*loops;++loop){const half*fci=fp+ci,*wci=wp+ci*Cout;__syncthreads();
  #pragma unroll
  for(int r=0;r<4;++r){half*dst=A+r*1280+y*320+(x>>2)*40+(x&3)*8;const int idx=rp[r*pitch];if(idx>=0)*reinterpret_cast<uint4*>(dst)=*reinterpret_cast<const uint4*>(fci+idx*Cin);else *reinterpret_cast<uint4*>(dst)=make_uint4(0,0,0,0);}
  #pragma unroll
  for(int r=0;r<2;++r){half*dst=B+r*1152+y*288+(x>>3)*72+(x&7)*8;*reinterpret_cast<uint4*>(dst)=*reinterpret_cast<const uint4*>(wci+r*16*Cout);}
  __syncthreads();
  #pragma unroll
  for(int kh=0;kh<2;++kh){
   #pragma unroll
   for(int mt=0;mt<4;++mt){unsigned addr;asm volatile("{.reg .u64 a;cvta.to.shared.u64 a,%1;cvt.u32.u64 %0,a;}":"=r"(addr):"l"((void*)((A+(y&1)*2560+mt*640+kh*16)+((x&15)*40+(x>>4)*8))));
#if __CUDA_ARCH__ >= 750
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16{%0,%1,%2,%3},[%4];":"=r"(((unsigned*)(af+mt*8))[0]),"=r"(((unsigned*)(af+mt*8))[1]),"=r"(((unsigned*)(af+mt*8))[2]),"=r"(((unsigned*)(af+mt*8))[3]):"r"(addr));
#endif
   }
   #pragma unroll
   for(int np=0;np<2;++np){unsigned addr;asm volatile("{.reg .u64 a;cvta.to.shared.u64 a,%1;cvt.u32.u64 %0,a;}":"=r"(addr):"l"((void*)((B+kh*1152+(y>>1)*32+np*16)+((x&15)*72+(x>>4)*8))));
#if __CUDA_ARCH__ >= 750
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16{%0,%1,%2,%3},[%4];":"=r"(((unsigned*)(bf+np*8))[0]),"=r"(((unsigned*)(bf+np*8))[1]),"=r"(((unsigned*)(bf+np*8))[2]),"=r"(((unsigned*)(bf+np*8))[3]):"r"(addr));
#endif
   }
   #pragma unroll
   for(int mt=0;mt<4;++mt){
    #pragma unroll
    for(int np=0;np<2;++np)mma(C+mt*16+np*8,af+mt*8,bf+np*8);
   }
  }
  ++ci_tile;ci+=32;if(ci_tile==loops){ci_tile=0;ci=0;const int prev=slot++;if(slot<count){rp+=1;wp+=(kOff[id][slot]-kOff[id][prev])*stride;}}}
 #pragma unroll
 for(int mt=0;mt<4;++mt){const int local=reorder+mt*16;
  #pragma unroll
  for(int np=0;np<2;++np){
   #pragma unroll
   for(int i=0;i<8;++i){const int row=local+((i/2)%2)*8,out=p.out[outbase+row],col=colbase+np*16+(i%2)+(i/4)*8;if(out>=0)p.output[out*Cout+col]=__float2half(C[mt*16+np*8+i]);}}}
}

__device__ __forceinline__ void tile(const P&p,int id,int outbase,int rowoff,int bn,half*A,half*B){
 const int tid=threadIdx.x,x=tid&31,y=tid>>5,Cin=p.cin,Cout=p.cout,loops=Cin/32,count=slots(id),pitch=count*16,stride=Cin*Cout;
 float C[32];half af[32],bf[8];
 #pragma unroll
 for(int i=0;i<32;++i)C[i]=0;
 const int seed=y*8+x/4;const int*rp=rows(p,id,rowoff,seed);const int first=count?kOff[id][0]:0;
 const half*wp=p.weight+first*stride+bn+(y*16+x/2)*Cout+(x*8%16);const half*fp=p.features+(x*8%32);
 const int reorder=(y%2)*64+x/4,colbase=bn+(x%4)*2;int slot=0,ci_tile=0,ci=0;
 for(int loop=0;loop<count*loops;++loop){const half*fci=fp+ci,*wci=wp+ci*Cout;__syncthreads();
  #pragma unroll
  for(int r=0;r<8;++r){half*dst=A+r*640+y*320+(x>>2)*40+(x&3)*8;const int idx=rp[r*pitch];if(idx>=0)*reinterpret_cast<uint4*>(dst)=*reinterpret_cast<const uint4*>(fci+idx*Cin);else *reinterpret_cast<uint4*>(dst)=make_uint4(0,0,0,0);}
  half*bdst=B+y*640+(x>>1)*40+(x&1)*8;*reinterpret_cast<uint4*>(bdst)=*reinterpret_cast<const uint4*>(wci);__syncthreads();
  #pragma unroll
  for(int kh=0;kh<2;++kh){
   #pragma unroll
   for(int mt=0;mt<4;++mt){unsigned addr;asm volatile("{.reg .u64 a;cvta.to.shared.u64 a,%1;cvt.u32.u64 %0,a;}":"=r"(addr):"l"((void*)((A+(y&1)*2560+mt*640+kh*16)+((x&15)*40+(x>>4)*8))));
#if __CUDA_ARCH__ >= 750
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16{%0,%1,%2,%3},[%4];":"=r"(((unsigned*)(af+mt*8))[0]),"=r"(((unsigned*)(af+mt*8))[1]),"=r"(((unsigned*)(af+mt*8))[2]),"=r"(((unsigned*)(af+mt*8))[3]):"r"(addr));
#endif
   }
   unsigned addr;asm volatile("{.reg .u64 a;cvta.to.shared.u64 a,%1;cvt.u32.u64 %0,a;}":"=r"(addr):"l"((void*)((B+kh*640)+((x&15)*40+(x>>4)*8))));
#if __CUDA_ARCH__ >= 750
   asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16{%0,%1,%2,%3},[%4];":"=r"(((unsigned*)bf)[0]),"=r"(((unsigned*)bf)[1]),"=r"(((unsigned*)bf)[2]),"=r"(((unsigned*)bf)[3]):"r"(addr));
#endif
   #pragma unroll
   for(int mt=0;mt<4;++mt)mma(C+mt*8,af+mt*8,bf);
  }
  ++ci_tile;ci+=32;if(ci_tile==loops){ci_tile=0;ci=0;const int prev=slot++;if(slot<count){rp+=1;wp+=(kOff[id][slot]-kOff[id][prev])*stride;}}}
 #pragma unroll
 for(int mt=0;mt<4;++mt){const int local=reorder+mt*16;
  #pragma unroll
  for(int i=0;i<8;++i){const int row=local+((i/2)%2)*8,out=p.out[outbase+row],col=colbase+(i%2)+(i/4)*8;if(out>=0)p.output[out*Cout+col]=__float2half(C[mt*8+i]);}}
}
__global__ void kernel2(P p){__shared__ half A[5120],B[1280];const int nt=p.cout/16,logical=blockIdx.x,base=logical/nt*128,bn=logical%nt*16,id=p.ids[base];if(id<0)return;tile(p,id,base,p.offsets[base],bn,A,B);}
__global__ void kernel3(P p){__shared__ half A[5120],B[2304];const int nt=p.cout/64,logical=blockIdx.x,base=logical/nt*128,bn=logical%nt*64,id=p.ids[base];if(id<0)return;tile3(p,id,base,p.offsets[base],bn,A,B);}
}

torch::Tensor kernel8_fp16_forward(torch::Tensor f,torch::Tensor w,torch::Tensor out,torch::Tensor w1,torch::Tensor w2,torch::Tensor w4,torch::Tensor w8,torch::Tensor ids,torch::Tensor offsets,int64_t n){
 c10::cuda::CUDAGuard guard(f.device());auto y=torch::empty({n,w.size(1)},f.options());P p{reinterpret_cast<const half*>(f.data_ptr<at::Half>()),reinterpret_cast<const half*>(w.data_ptr<at::Half>()),out.data_ptr<int>(),w1.data_ptr<int>(),w2.data_ptr<int>(),w4.data_ptr<int>(),w8.data_ptr<int>(),ids.data_ptr<int>(),offsets.data_ptr<int>(),reinterpret_cast<half*>(y.data_ptr<at::Half>()),static_cast<int>(f.size(1)),static_cast<int>(w.size(1)),static_cast<int>(out.size(0)),static_cast<int>(w1.size(1))};
 if(p.cout%64==0){const int grid=p.rows/128*(p.cout/64);kernel3<<<grid,128,0,at::cuda::getCurrentCUDAStream()>>>(p);}else{const int grid=p.rows/128*(p.cout/16);kernel2<<<grid,64,0,at::cuda::getCurrentCUDAStream()>>>(p);}return y;}
