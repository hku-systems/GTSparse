from __future__ import annotations
from dataclasses import dataclass
import math
import torch
import torch.nn as nn
from gtsparse import _C
from gtsparse.sparse3d.sparse_tensor import GTSparseSparseConvTensor
from .metadata import GeometricTemplateMetadata, GeometricTemplateReverseEdge

BM=128

@dataclass(slots=True)
class Kernel8Runtime:
    out_rows:torch.Tensor
    input_rows_w1:torch.Tensor
    input_rows_w2:torch.Tensor
    input_rows_w4:torch.Tensor
    input_rows_w8:torch.Tensor
    template_ids:torch.Tensor
    input_row_offsets:torch.Tensor
    template_counts:torch.Tensor
    padded_counts:torch.Tensor
    out_coords:torch.Tensor
    coord_hashmap:torch.Tensor
    out_spatial:tuple[int,int,int]

@dataclass(slots=True)
class Kernel8ReverseSpec:
    lookup_coords:torch.Tensor
    target_coords:torch.Tensor
    target_spatial:tuple[int,int,int]
    lookup_hashmap:torch.Tensor
    max_bm:int
    def build(self):
        values=_C.gtsparse_kernel8_build_reverse_runtime(
            self.lookup_coords,self.target_coords,2,2,2,0,0,0,1,1,1,self.max_bm,self.lookup_hashmap)
        return Kernel8Runtime(*values,self.target_spatial)

def _conv(features,weight,runtime):
    fn=_C.gtsparse_kernel8_fp16_forward if features.dtype==torch.float16 else _C.gtsparse_kernel8_fp32_forward
    return fn(features,weight,runtime.out_rows,runtime.input_rows_w1,runtime.input_rows_w2,
        runtime.input_rows_w4,runtime.input_rows_w8,runtime.template_ids,runtime.input_row_offsets,runtime.out_coords.size(0))

class GeometricTemplateKernel8Conv3d(nn.Module):
    def __init__(self,in_channels:int,out_channels:int,*,bias:bool=False,max_bm:int=BM)->None:
        super().__init__();self.in_channels=int(in_channels);self.out_channels=int(out_channels)
        self.kernel_size=(2,2,2);self.stride=(2,2,2);self.padding=(0,0,0);self.dilation=(1,1,1);self.transposed=False
        if self.in_channels%16 or self.out_channels%16:raise ValueError("kernel-8 requires channels divisible by 16")
        self.max_bm=int(max_bm);self.weight=nn.Parameter(torch.empty(8,self.in_channels,self.out_channels));self.bias=nn.Parameter(torch.zeros(self.out_channels)) if bias else None;self.runtime_weight_cache=None;self.reset_parameters()
    def reset_parameters(self):
        self.runtime_weight_cache=None;bound=1/math.sqrt(self.in_channels*8);nn.init.uniform_(self.weight,-bound,bound)
        if self.bias is not None:nn.init.uniform_(self.bias,-bound,bound)
    def _runtime_weight(self):
        if self.runtime_weight_cache is None:self.runtime_weight_cache=self.weight.contiguous().view(8*self.in_channels,self.out_channels)
        return self.runtime_weight_cache
    def train(self,mode=True):self.runtime_weight_cache=None;return super().train(mode)
    def _load_from_state_dict(self,*args,**kwargs):self.runtime_weight_cache=None;return super()._load_from_state_dict(*args,**kwargs)
    def build_runtime(self,x:GTSparseSparseConvTensor):
        out_spatial=tuple((int(size)-2)//2+1 for size in x.spatial_shape)
        values=_C.gtsparse_kernel8_build_full_runtime(x.indices,*out_spatial,*self.stride,*self.padding,*self.dilation,self.max_bm,
            torch.empty(0,device=x.indices.device) if x.coord_hashmap is None else x.coord_hashmap)
        runtime=Kernel8Runtime(*values,out_spatial)
        return runtime,out_spatial
    def forward(self,x:GTSparseSparseConvTensor)->GTSparseSparseConvTensor:
        runtime,out_spatial=self.build_runtime(x)
        reverse=Kernel8ReverseSpec(runtime.out_coords,x.indices,tuple(x.spatial_shape),runtime.coord_hashmap,self.max_bm)
        features=_conv(x.features,self._runtime_weight(),runtime)
        if self.bias is not None:features=features+self.bias
        metadata=GeometricTemplateMetadata(reverse_chain=(GeometricTemplateReverseEdge(reverse,x.coord_hashmap),*getattr(x.metadata,"reverse_chain",())))
        return x.replace_sparse(new_features=features,new_coords=runtime.out_coords,new_spatial_shape=out_spatial,new_batch_size=x.batch_size,
            coord_hashmap=runtime.coord_hashmap,metadata=metadata)

class GeometricTemplateKernel8InverseConv3d(nn.Module):
    def __init__(self,in_channels:int,out_channels:int,*,bias:bool=False)->None:
        super().__init__();self.in_channels=int(in_channels);self.out_channels=int(out_channels);self.kernel_size=(2,2,2);self.stride=(2,2,2);self.transposed=True
        if self.in_channels%16 or self.out_channels%16:raise ValueError("kernel-8 requires channels divisible by 16")
        self.weight=nn.Parameter(torch.empty(8,self.in_channels,self.out_channels));self.bias=nn.Parameter(torch.zeros(self.out_channels)) if bias else None;self.runtime_weight_cache=None;self.reset_parameters()
    def reset_parameters(self):
        self.runtime_weight_cache=None;bound=1/math.sqrt(self.in_channels*8);nn.init.uniform_(self.weight,-bound,bound)
        if self.bias is not None:nn.init.uniform_(self.bias,-bound,bound)
    def _runtime_weight(self):
        if self.runtime_weight_cache is None:self.runtime_weight_cache=self.weight.contiguous().view(8*self.in_channels,self.out_channels)
        return self.runtime_weight_cache
    def train(self,mode=True):self.runtime_weight_cache=None;return super().train(mode)
    def _load_from_state_dict(self,*args,**kwargs):self.runtime_weight_cache=None;return super()._load_from_state_dict(*args,**kwargs)
    def forward(self,x:GTSparseSparseConvTensor)->GTSparseSparseConvTensor:
        edge,*remaining=x.metadata.reverse_chain;runtime=edge.runtime.build();features=_conv(x.features,self._runtime_weight(),runtime)
        if self.bias is not None:features=features+self.bias
        return x.replace_sparse(new_features=features,new_coords=runtime.out_coords,new_spatial_shape=runtime.out_spatial,new_batch_size=x.batch_size,
            coord_hashmap=edge.coord_hashmap,metadata=GeometricTemplateMetadata(reverse_chain=tuple(remaining)))
