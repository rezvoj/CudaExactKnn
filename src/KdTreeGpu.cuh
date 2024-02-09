#pragma once
#include <cub/cub.cuh>
#include <algorithm>
#include <numeric>
#include <cstring>
#include "KnnTrees.cuh"


namespace KnnTrees {

    namespace KdTreeKernels {

        template <uint Dims, uint K, bool HeapNN>
        __forceinline__ __device__
        float considerPointDevice(
                const uint idx,
                float maxDistance2,
                uint64_t* nearestNeighbours,
                const uint nodeIdx, 
                const Array<float, Dims>& point,
                const uint nodeSize,
                const uint maxChildren,
                const IndexPoint<Dims>* indexPoints) {
            const uint idxIdx = maxChildren * (nodeIdx - nodeSize); 
            for (uint offset = 0; offset < maxChildren; ++offset) {
                const IndexPoint<Dims> indexPoint = indexPoints[idxIdx + offset];
                if (indexPoint.index == -1) break;
                float distance2 = calcDistance2<Dims>(point, indexPoint.point);
                if (HeapNN) 
                    maxDistance2 = heapConsider<K>(nearestNeighbours, indexPoint.index, distance2);
                else 
                    maxDistance2 = arrayConsider<K>(nearestNeighbours, indexPoint.index, distance2);
            }
            return maxDistance2;
        }
 
        template <uint Dims, uint AuxDims, uint K, bool HeapNN>
        __forceinline__ __device__
        void knnDevice(
                const uint idx, 
                uint64_t* nearestNeighbours,
                uint* indexStack,
                float* distanceStack,
                Query<Dims, K>& query,
                const uint maxChildren,
                const uint nodeSize,
                const float* nodes,
                const IndexPoint<Dims>* indexPoints) {
            float maxDistance2 = query.maxDistance2;
            fillArray<uint64_t, K>(nearestNeighbours, encodeUFloatInt(maxDistance2, -1));
            const Array<float, Dims> point = query.points[idx];
            indexStack[0] = 0; distanceStack[0] = 0.0f;
            uint stackSize = 1;
            while (stackSize) {
                if (distanceStack[--stackSize] >= maxDistance2) continue;
                uint nodeIdx = indexStack[stackSize];
                uint currDim = (log2Ceil(nodeIdx + 2) - 1) % AuxDims;
                while (nodeIdx < nodeSize) {
                    uint closeChild, farChild;
                    const float distance = point[currDim] - nodes[nodeIdx];
                    const uint leftChild = leftChildOf(nodeIdx);
                    if (distance < 0.0f) {
                        closeChild = leftChild;
                        farChild = leftChild + 1;
                    }
                    else {
                        farChild = leftChild;
                        closeChild = leftChild + 1;
                    }
                    const float distance2 = distance * distance;
                    if (distance2 < maxDistance2) {
                        indexStack[stackSize] = farChild;
                        distanceStack[stackSize] = distance2;
                        stackSize += 1;
                    }
                    nodeIdx = closeChild;
                    currDim = (currDim + 1) % AuxDims;
                } 
                maxDistance2 = considerPointDevice<Dims, K, HeapNN>(
                    idx, maxDistance2, nearestNeighbours, nodeIdx, point,
                    nodeSize, maxChildren, indexPoints
                );
            }
            #pragma unroll
            for (uint i = 0; i < K; ++i) {
                const uint64_t neighbour = nearestNeighbours[i];
                query.rIndexes[idx][i] = static_cast<int>(neighbour);
                query.rDistances[idx][i] = decodeEncoded1UFloat(neighbour);
            }
        }

        template <uint Dims, uint AuxDims, uint K, uint BlockSize, bool HeapNN> 
        __global__
        void knnKernelShSh(
                Query<Dims, K> query,
                const uint maxChildren,
                const uint nodeSize,
                const float* nodes,
                const IndexPoint<Dims>* indexPoints) {
            const uint idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx >= query.count) return;
            constexpr uint STACK_SIZE = sizeof(uint) * 8;
            __shared__ uint sIndexStack[BlockSize * STACK_SIZE];
            __shared__ float sDistanceStack[BlockSize * STACK_SIZE];
            uint* indexStack = sIndexStack + threadIdx.x * STACK_SIZE;
            float* distanceStack = sDistanceStack + threadIdx.x * STACK_SIZE;
            __shared__ uint64_t sNearestNeighbours[BlockSize * K];
            uint64_t* nearestNeighbours = sNearestNeighbours + threadIdx.x * K;
            knnDevice<Dims, AuxDims, K, HeapNN>(
                idx, nearestNeighbours, 
                indexStack, distanceStack,
                query, maxChildren, nodeSize,
                nodes, indexPoints
            );
        }

        template <uint Dims, uint AuxDims, uint K, uint BlockSize, bool HeapNN> 
        __global__
        void knnKernelNshSh(
                Query<Dims, K> query,
                const uint maxChildren,
                const uint nodeSize,
                const float* nodes,
                const IndexPoint<Dims>* indexPoints) {
            const uint idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx >= query.count) return;
            constexpr uint STACK_SIZE = sizeof(uint) * 8;
            __shared__ uint sIndexStack[BlockSize * STACK_SIZE];
            __shared__ float sDistanceStack[BlockSize * STACK_SIZE];
            uint* indexStack = sIndexStack + threadIdx.x * STACK_SIZE;
            float* distanceStack = sDistanceStack + threadIdx.x * STACK_SIZE;
            uint64_t nearestNeighbours[K];
            knnDevice<Dims, AuxDims, K, HeapNN>(
                idx, nearestNeighbours, 
                indexStack, distanceStack,
                query, maxChildren, nodeSize,
                nodes, indexPoints
            );
        }

        template <uint Dims, uint AuxDims, uint K, uint BlockSize, bool HeapNN> 
        __global__
        void knnKernelShNsh(
                Query<Dims, K> query,
                const uint maxChildren,
                const uint nodeSize,
                const float* nodes,
                const IndexPoint<Dims>* indexPoints) {
            const uint idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx >= query.count) return;
            constexpr uint STACK_SIZE = sizeof(uint) * 8;
            uint indexStack[STACK_SIZE];
            float distanceStack[STACK_SIZE];
            __shared__ uint64_t sNearestNeighbours[BlockSize * K];
            uint64_t* nearestNeighbours = sNearestNeighbours + threadIdx.x * K;
            knnDevice<Dims, AuxDims, K, HeapNN>(
                idx, nearestNeighbours, 
                indexStack, distanceStack,
                query, maxChildren, nodeSize,
                nodes, indexPoints
            );
        }

        template <uint Dims, uint AuxDims, uint K, uint BlockSize, bool HeapNN> 
        __global__
        void knnKernelNshNsh(
                Query<Dims, K> query,
                const uint maxChildren,
                const uint nodeSize,
                const float* nodes,
                const IndexPoint<Dims>* indexPoints) {
            const uint idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx >= query.count) return;
            constexpr uint STACK_SIZE = sizeof(uint) * 8;
            uint indexStack[STACK_SIZE];
            float distanceStack[STACK_SIZE];
            uint64_t nearestNeighbours[K];
            knnDevice<Dims, AuxDims, K, HeapNN>(
                idx, nearestNeighbours, 
                indexStack, distanceStack,
                query, maxChildren, nodeSize,
                nodes, indexPoints
            );
        }

        __global__
        void iotaKernel(int* indexes, uint size) {
            const uint idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx >= size) return;
            indexes[idx] = idx;
        }

        template <typename Type>
        __global__
        void fillKernel(Type* data, uint size, const Type value) {
            const uint idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx >= size) return;
            data[idx] = value;
        }

        template <uint Dims>
        __global__
        void fillPointSort(
                uint64_t* pointSort,
                const int* indexes,
                const Array<float, Dims>* points,
                const uint* nodePoints,
                const uint dimension,
                const uint size) {
            const uint idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx >= size) return; 
            const uint nodeIdx = nodePoints[idx];
            const float dimensionValue = points[indexes[idx]][dimension];
            pointSort[idx] = encodeUintSFloat(nodeIdx, dimensionValue);
        }

        template <uint Dims>
        __global__
        void updateNodesKernel(
                float* nodes,
                const int* indexes,
                const Array<float, Dims>* points,
                uint* nodeSizes,
                const uint* nodeScans,
                const uint size,
                const uint nodesBaseIndex,
                const uint dimension) {
            const uint idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx >= size) return;
            const uint nodeIdx = nodesBaseIndex + idx;
            const uint rightHalfSize = nodeSizes[nodeIdx] / 2;
            const uint leftHalfSize = rightHalfSize + nodeSizes[nodeIdx] % 2;
            const uint medianIdx = nodeScans[idx] + leftHalfSize - 1;
            nodes[nodeIdx] = points[indexes[medianIdx]][dimension];
            const uint leftChildIdx = leftChildOf(nodeIdx);
            const uint rightChildIdx = leftChildIdx + 1;
            nodeSizes[leftChildIdx] = leftHalfSize;
            nodeSizes[rightChildIdx] = rightHalfSize;
        }

        __global__
        void updatePointNodesKernel(
                uint* pointNodes,
                const uint size,
                const uint* nodeSizes, 
                const uint* nodeScans,
                const uint nodesBaseIndex) {
            const uint idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx >= size) return;
            const uint nodeIdx = pointNodes[idx];
            const uint nodeScan = nodeScans[nodeIdx - nodesBaseIndex];
            const uint leftHalfSize = nodeSizes[nodeIdx] / 2 + nodeSizes[nodeIdx] % 2;
            const bool isLeftChild = idx < nodeScan + leftHalfSize;
            pointNodes[idx] = leftChildOf(nodeIdx) + (isLeftChild ? 0 : 1);
        }

        template <uint Dims>
        __global__
        void setIndexPointsKernel(
                IndexPoint<Dims>* indexPoints,
                const int* indexes,
                const Array<float, Dims>* points,
                const uint indexGroupSize,
                const uint size,
                const uint* nodeScans,
                const uint* pointNodes,
                const uint nodesBaseIndex) {
            const uint idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx >= size) return;
            const uint nodeIdx = pointNodes[idx];
            const uint nodePos = nodeIdx - nodesBaseIndex;
            const uint nodeScan = nodeScans[nodePos];
            const uint groupIndex = indexGroupSize * nodePos + idx - nodeScan;
            IndexPoint<Dims> indexPoint;
            indexPoint.index = indexes[idx];
            indexPoint.point = points[indexPoint.index];
            indexPoints[groupIndex] = indexPoint;
        }

    }

    template <uint Dims, uint AuxDims, uint CBlockSize>
    class KdTreeGpu {
    public:
        using Point = Array<float, Dims>;

    private:
        uint mDepthSkips;
        uint mNodeDepth;
        uint mMaxChildren;
        uint mIndexSize;
        uint mNodeSize;
        float* mDNodes;
        IndexPoint<Dims>* mDIndexPoints;

    private:
        __forceinline__
        void mSplitPoints(const Point* points, const uint pointSize) {
            using namespace KdTreeKernels;
            int* indexes;
            Cuda::check(cudaMalloc(&indexes, sizeof(int) * pointSize));
            const uint pointBlockCount = Cuda::blockCount(pointSize, CBlockSize);
            iotaKernel<<<pointBlockCount, CBlockSize>>>(indexes, pointSize);
            uint* nodeSizes; uint* nodePoints; uint* nodeSizeScans;
            Cuda::check(cudaMalloc(&nodeSizes, sizeof(uint) * (2 * mNodeSize + 1)));
            Cuda::check(cudaMalloc(&nodeSizeScans, sizeof(uint) * (mNodeSize + 1)));
            Cuda::check(cudaMalloc(&nodePoints, sizeof(uint) * pointSize));
            uint64_t* pointSort; uint64_t* pointSortOut; int* indexesOut;
            Cuda::check(cudaMalloc(&pointSort, sizeof(uint64_t) * pointSize));
            Cuda::check(cudaMalloc(&pointSortOut, sizeof(uint64_t) * pointSize));
            Cuda::check(cudaMalloc(&indexesOut, sizeof(int) * pointSize));
            void* tempData = nullptr; size_t tempSize = 0; size_t scanSize = 0;
            cub::DeviceRadixSort::SortPairs(
                tempData, tempSize, pointSort, pointSortOut, 
                indexes, indexesOut, pointSize
            );
            cub::DeviceScan::ExclusiveSum(
                tempData, scanSize, nodeSizes,
                nodeSizeScans, mNodeSize + 1
            );
            tempSize = std::max(tempSize, scanSize);
            Cuda::check(cudaMalloc(&tempData, tempSize));
            const uint zero = 0;
            Cuda::check(cudaMemcpy(nodeSizes, &pointSize, sizeof(uint), Cuda::H2D));
            Cuda::check(cudaMemcpy(nodeSizeScans, &zero, sizeof(uint), Cuda::H2D));
            fillKernel<uint><<<pointBlockCount, CBlockSize>>>(nodePoints, pointSize, 0);
            uint nodesBaseIndex = 0; uint nodeDepthCount = 1;
            for (uint depth = 0; depth < mNodeDepth; ++depth) {
                const uint dimension = depth % AuxDims;
                fillPointSort<Dims><<<pointBlockCount, CBlockSize>>>(
                    pointSort, indexes, points, nodePoints, 
                    dimension, pointSize
                );
                cub::DeviceRadixSort::SortPairs(
                    tempData, tempSize, pointSort, pointSortOut, 
                    indexes, indexesOut, pointSize
                );
                swap(indexes, indexesOut);
                const uint nodeBlockCount = Cuda::blockCount(nodeDepthCount, CBlockSize);
                updateNodesKernel<Dims><<<nodeBlockCount, CBlockSize>>>(
                    mDNodes, indexes, points, nodeSizes, nodeSizeScans,
                    nodeDepthCount, nodesBaseIndex, dimension
                );
                updatePointNodesKernel<<<pointBlockCount, CBlockSize>>>(
                    nodePoints, pointSize, nodeSizes, nodeSizeScans, nodesBaseIndex
                );
                nodesBaseIndex += nodeDepthCount;
                nodeDepthCount *= 2;
                cub::DeviceScan::ExclusiveSum(
                    tempData, tempSize, nodeSizes + nodesBaseIndex,
                    nodeSizeScans, nodeDepthCount
                );
            }
            IndexPoint<Dims> emptyIndexPoint;
            emptyIndexPoint.index = -1;
            const uint indexesBlockCount = Cuda::blockCount(mIndexSize, CBlockSize);
            fillKernel<IndexPoint<Dims>><<<indexesBlockCount, CBlockSize>>>(
                mDIndexPoints, mIndexSize, emptyIndexPoint
            );
            setIndexPointsKernel<Dims><<<pointBlockCount, CBlockSize>>>(
                mDIndexPoints, indexes, points, mMaxChildren, pointSize,
                nodeSizeScans, nodePoints, nodesBaseIndex
            );
            Cuda::check(cudaFree(indexes));
            Cuda::check(cudaFree(nodeSizes));
            Cuda::check(cudaFree(nodeSizeScans));
            Cuda::check(cudaFree(nodePoints));
            Cuda::check(cudaFree(pointSort));
            Cuda::check(cudaFree(pointSortOut));
            Cuda::check(cudaFree(indexesOut));
            Cuda::check(cudaFree(tempData));
        }

    public:
        KdTreeGpu(const Point* points, const uint size, const uint depthSkips = 0):
                mDepthSkips(std::min(depthSkips, log2Ceil(size) - 1)),
                mNodeDepth(log2Ceil(size) - mDepthSkips),
                mMaxChildren(pow2(mDepthSkips)),
                mIndexSize(pow2(mNodeDepth + mDepthSkips)),
                mNodeSize(pow2(mNodeDepth) - 1) {
            Cuda::check(cudaMalloc(&mDNodes, sizeof(BallNode<Dims>) * mNodeSize));
            Cuda::check(cudaMalloc(&mDIndexPoints, sizeof(IndexPoint<Dims>) * mIndexSize));
            mSplitPoints(points, size);
            Cuda::check(cudaDeviceSynchronize());
        }

        ~KdTreeGpu() {
            cudaFree(mDNodes);
            cudaFree(mDIndexPoints);
        }

    public:
        template <uint K, uint BlockSize, bool HeapNN, bool SharedMemNN, bool SharedMemStack>
        void batchKnn(Query<Dims, K>& query) const {
            const uint blockCount = Cuda::blockCount(query.count, BlockSize);
            if constexpr (SharedMemNN && SharedMemStack)
                KdTreeKernels::knnKernelShSh<Dims, AuxDims, K, BlockSize, HeapNN>
                    <<<blockCount, BlockSize>>>(query, mMaxChildren, mNodeSize, mDNodes, mDIndexPoints);
            else if constexpr (!SharedMemNN && SharedMemStack)
                KdTreeKernels::knnKernelNshSh<Dims, AuxDims, K, BlockSize, HeapNN>
                    <<<blockCount, BlockSize>>>(query, mMaxChildren, mNodeSize, mDNodes, mDIndexPoints);
            else if constexpr (SharedMemNN && !SharedMemStack)
                KdTreeKernels::knnKernelShNsh<Dims, AuxDims, K, BlockSize, HeapNN>
                    <<<blockCount, BlockSize>>>(query, mMaxChildren, mNodeSize, mDNodes, mDIndexPoints);
            else if constexpr (!SharedMemNN && !SharedMemStack)
                KdTreeKernels::knnKernelNshNsh<Dims, AuxDims, K, BlockSize, HeapNN>
                    <<<blockCount, BlockSize>>>(query, mMaxChildren, mNodeSize, mDNodes, mDIndexPoints);
            Cuda::check(cudaGetLastError());
            Cuda::check(cudaDeviceSynchronize());
        }

    };

}
