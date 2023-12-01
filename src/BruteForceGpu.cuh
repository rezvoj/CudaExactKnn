#pragma once
#include "KnnTrees.cuh"


namespace KnnTrees {

    namespace BruteForceKernels {

        template <uint Dims, uint K, bool HeapNN>
        __forceinline__ __device__
        void knnDevice(
                const uint idx,
                const Array<float, Dims>* data,
                uint size,
                uint64_t* nearestNeighbours,
                Query<Dims, K>& query) {
            fillArray<uint64_t, K>(nearestNeighbours, encodeUFloatInt(query.maxDistance2, -1));
            for (uint index = 0; index < size; ++index) {
                const float distance2 = calcDistance2<Dims>(query.points[idx], data[index]);
                if constexpr (HeapNN)
                    heapConsider<K>(nearestNeighbours, index, distance2);
                else
                    arrayConsider<K>(nearestNeighbours, index, distance2);
            }
            #pragma unroll
            for (uint i = 0; i < K; ++i) {
                const uint64_t neighbour = nearestNeighbours[i];
                query.rIndexes[idx][i] = static_cast<int>(neighbour);
                query.rDistances[idx][i] = decodeEncoded1UFloat(neighbour);
            }
        }

        template <uint Dims, uint K, bool HeapNN>
        __global__
        void knnKernel(const Array<float, Dims>* data, uint size, Query<Dims, K> query) {
            const uint idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx >= query.count) return;
            uint64_t nearestNeighbours[K];
            knnDevice<Dims, K, HeapNN>(idx, data, size, nearestNeighbours, query);
        }

        template <uint Dims, uint K, uint BlockSize, bool HeapNN>
        __global__
        void knnShKernel(const Array<float, Dims>* data, uint size, Query<Dims, K> query) {
            const uint idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx >= query.count) return;
            __shared__ uint64_t sNearestNeighbours[K * BlockSize];
            uint64_t* nearestNeighbours = sNearestNeighbours + threadIdx.x * K;
            knnDevice<Dims, K, HeapNN>(idx, data, size, nearestNeighbours, query);
        }

    }

    template <uint Dims>
    class BruteForceGpu {
    public:
        using Point = Array<float, Dims>;

    private:
        uint mSize;
        Point* mDData;

    public:
        BruteForceGpu(const Point* points, const uint size) : mSize(size) {
            Cuda::check(cudaMalloc(&mDData, sizeof(Point) * mSize));
            Cuda::check(cudaMemcpy(mDData, points, sizeof(Point) * mSize, Cuda::D2D));
            Cuda::check(cudaDeviceSynchronize());
        }

        ~BruteForceGpu() {
            cudaFree(mDData);
        }

        template <uint K, uint BlockSize, bool HeapNN, bool SharedMem>
        void batchKnn(Query<Dims, K>& query) const {
            using namespace BruteForceKernels;
            const uint blockCount = Cuda::blockCount(query.count, BlockSize);
            if constexpr (SharedMem) 
                knnShKernel<Dims, K, BlockSize, HeapNN><<<blockCount, BlockSize>>>(mDData, mSize, query);
            else 
                knnKernel<Dims, K, HeapNN><<<blockCount, BlockSize>>>(mDData, mSize, query);
            Cuda::check(cudaDeviceSynchronize());
        }

    };

}
