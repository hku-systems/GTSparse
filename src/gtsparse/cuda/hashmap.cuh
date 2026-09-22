// Packed GPU hash table for coordinate -> row index mapping.
//
// This center-last builder keeps a private copy so its builders stay fully
// decoupled from the older flat-tiled implementation.

#pragma once

#include <cuda_runtime.h>
#include <cstdint>
#include <torch/extension.h>

constexpr int64_t kEmptyHashKey = 0;

inline void check_cuda_contiguous(const torch::Tensor& t, const char* name) {
    TORCH_CHECK(t.is_cuda(), name, " must be a CUDA tensor");
    TORCH_CHECK(t.is_contiguous(), name, " must be contiguous");
}

struct CoordHashMap {
    int64_t* bucket_words;
    int capacity_mask;

    __device__ __forceinline__ static int64_t hash_coords(int b, int d, int h, int w) {
        uint64_t hash = 14695981039346656037ULL;
        const int data[4] = {b, d, h, w};
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            hash ^= static_cast<unsigned int>(data[i]);
            hash *= 1099511628211ULL;
        }
        return static_cast<int64_t>(hash);
    }

    __device__ __forceinline__ int lookup(int b, int d, int h, int w) const {
        const int64_t key = hash_coords(b, d, h, w);
        int slot = static_cast<int>(static_cast<uint64_t>(key) & static_cast<uint64_t>(capacity_mask));
        while (true) {
            const ulonglong2 bucket = reinterpret_cast<const ulonglong2*>(bucket_words)[slot];
            const int64_t cur = static_cast<int64_t>(bucket.x);
            if (cur == key) {
                return static_cast<int>(static_cast<uint32_t>(bucket.y)) - 1;
            }
            if (cur == kEmptyHashKey) {
                return -1;
            }
            slot = (slot + 1) & capacity_mask;
        }
    }
};

struct CoordHashMapOwner {
    CoordHashMap map;
    torch::Tensor buckets;
};

inline int hashmap_capacity_for_rows(int n_rows) {
    int capacity = n_rows * 4;
    if (capacity < 32) {
        capacity = 32;
    }
    int pow2 = 1;
    while (pow2 < capacity) {
        pow2 <<= 1;
    }
    return pow2;
}

inline CoordHashMap view_coord_hashmap(const torch::Tensor& coord_hashmap) {
    check_cuda_contiguous(coord_hashmap, "coord_hashmap");
    TORCH_CHECK(coord_hashmap.dtype() == torch::kInt64, "coord_hashmap must be int64");
    TORCH_CHECK(coord_hashmap.dim() == 2 && coord_hashmap.size(1) == 2, "coord_hashmap must be [capacity, 2]");
    const int capacity = static_cast<int>(coord_hashmap.size(0));
    TORCH_CHECK(
        capacity >= 32 && (capacity & (capacity - 1)) == 0,
        "coord_hashmap capacity must be a power of two >= 32");
    CoordHashMap map;
    map.bucket_words = const_cast<int64_t*>(coord_hashmap.data_ptr<int64_t>());
    map.capacity_mask = capacity - 1;
    return map;
}

static __global__ void insert_coords_kernel(
    int64_t* __restrict__ bucket_words,
    const int* __restrict__ coords,
    int n_rows,
    int capacity_mask) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_rows) {
        return;
    }
    const int b = coords[idx * 4 + 0];
    const int d = coords[idx * 4 + 1];
    const int h = coords[idx * 4 + 2];
    const int w = coords[idx * 4 + 3];
    const int64_t key = CoordHashMap::hash_coords(b, d, h, w);
    int slot = static_cast<int>(static_cast<uint64_t>(key) & static_cast<uint64_t>(capacity_mask));
    while (true) {
        const int64_t prev = static_cast<int64_t>(atomicCAS(
            reinterpret_cast<unsigned long long*>(bucket_words + slot * 2),
            static_cast<unsigned long long>(kEmptyHashKey),
            static_cast<unsigned long long>(key)));
        if (prev == kEmptyHashKey || prev == key) {
            bucket_words[slot * 2 + 1] = static_cast<int64_t>(static_cast<uint32_t>(idx + 1));
            return;
        }
        slot = (slot + 1) & capacity_mask;
    }
}

inline CoordHashMapOwner build_coord_hashmap(const torch::Tensor& coords, cudaStream_t stream) {
    CoordHashMapOwner owner;
    const int n_rows = static_cast<int>(coords.size(0));
    const int capacity = hashmap_capacity_for_rows(n_rows);
    owner.buckets = torch::zeros({capacity, 2}, coords.options().dtype(torch::kInt64));
    owner.map.bucket_words = owner.buckets.data_ptr<int64_t>();
    owner.map.capacity_mask = capacity - 1;
    if (n_rows > 0) {
        constexpr int kThreads = 256;
        insert_coords_kernel<<<(n_rows + kThreads - 1) / kThreads, kThreads, 0, stream>>>(
            owner.map.bucket_words,
            coords.data_ptr<int>(),
            n_rows,
            owner.map.capacity_mask);
    }
    return owner;
}
