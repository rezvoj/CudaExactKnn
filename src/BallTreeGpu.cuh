#pragma once
#include <cub/cub.cuh>
#include "KnnTrees.cuh"
#include <numeric>
#include <algorithm>
#include <limits>


namespace KnnTrees {

    namespace BallTreeKernels {

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
        void fillIndexesToDimsKernel(
                float* dataP1, 
                const int* indexes, 
                const Array<float, Dims>* points, 
                const uint size) {
            const uint idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx >= size) return;
            const Array<float, Dims> point = points[indexes[idx]];
            #pragma unroll
            for (uint dimension = 0; dimension < Dims; ++dimension) {
                dataP1[dimension * (size + 1) + idx] = point[dimension];
            }
        }

        template <uint Dims>
        __global__
        void createNodeCentroidsKernel(
                Array<float, Dims>* nodeCentorids,
                const float* centroidScans,
                const uint* nodeSizes,
                const uint* nodeSizeScans,
                const uint nodeBaseIdx,
                const uint nodeSize,
                const uint pointSize) {
            const uint idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx >= nodeSize) return;
            const uint nodeIdx = nodeBaseIdx + idx;
            const uint size = nodeSizes[nodeIdx];
            const uint firstIdx = nodeSizeScans[idx];
            const uint lastIdx = firstIdx + size;
            Array<float, Dims> centroid;
            #pragma unroll
            for (uint dimension = 0; dimension < Dims; ++dimension) {
                const uint offset = dimension * (pointSize + 1);
                const float firstScan = centroidScans[offset + firstIdx];
                const float lastScan = centroidScans[offset + lastIdx];
                centroid[dimension] = (lastScan - firstScan) / size;
            }
            nodeCentorids[nodeIdx] = centroid;
        }

        template <uint Dims>
        __global__
        void fillIndexesToVarianceKernel(
                float* dataP1,
                const int* indexes, 
                const Array<float, Dims>* points,
                const uint* nodePoints,
                const Array<float, Dims>* centroids,
                const uint size) {
            const uint idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx >= size) return;
            const Array<float, Dims> point = points[indexes[idx]];
            const Array<float, Dims> centroid = centroids[nodePoints[idx]];
            #pragma unroll
            for (uint dimension = 0; dimension < Dims; ++dimension) {
                const float diff = point[dimension] - centroid[dimension];
                dataP1[dimension * (size + 1) + idx] = diff * diff;
            }
        }
        
        template <uint Dims>
        __global__
        void pickNodeDimKernel(
                uint* nodeDims,
                const float* distanceScans,
                const uint* nodeSizes,
                const uint* nodeSizeScans,
                const uint nodeBaseIdx,
                const uint nodeSize,
                const uint pointSize) {
            const uint idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx >= nodeSize) return;
            const uint nodeIdx = nodeBaseIdx + idx;
            const uint size = nodeSizes[nodeIdx];
            const uint firstIdx = nodeSizeScans[idx];
            const uint lastIdx = firstIdx + size;
            float maxValue = 0.0f; 
            uint bestDim = 0;
            #pragma unroll
            for (uint dimension = 0; dimension < Dims; ++dimension) {
                const uint offset = dimension * (pointSize + 1);
                float value = distanceScans[offset + lastIdx] - distanceScans[offset + firstIdx];
                if (value > maxValue) {
                    maxValue = value;
                    bestDim = dimension;
                }
            }
            nodeDims[nodeIdx] = bestDim;
        }

        template <uint Dims>
        __global__
        void encodeDimValueToSortKernel(
                uint64_t* pointSortKey,
                const int* indexes,
                const Array<float, Dims>* points,
                const uint* nodePoints,
                const uint* nodeDims,
                const uint size) {
            const uint idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx >= size) return;
            const uint nodeIdx = nodePoints[idx];
            const uint dimension = nodeDims[nodeIdx];
            const float dimValue = points[indexes[idx]][dimension];
            pointSortKey[idx] = encodeUintSFloat(nodeIdx, dimValue);
        }

        __global__
        void updateNodeSizesKernel(
                uint* nodeSizes,
                const uint size,
                const uint nodesBaseIndex) {
            const uint idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx >= size) return;
            const uint nodeIdx = nodesBaseIndex + idx;
            const uint rightHalfSize = nodeSizes[nodeIdx] / 2;
            const uint leftHalfSize = rightHalfSize + nodeSizes[nodeIdx] % 2;
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
        void createNodeCentroidsKernel(
                Array<float, Dims>* nodeCentorids,
                BallNode<Dims>* ballNodes,
                const float* centroidScans,
                const uint* nodeSizes,
                const uint* nodeSizeScans,
                const uint nodeBaseIdx,
                const uint nodeSize,
                const uint pointSize) {
            const uint idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx >= nodeSize) return;
            const uint nodeIdx = nodeBaseIdx + idx;
            const uint size = nodeSizes[nodeIdx];
            const uint firstIdx = nodeSizeScans[idx];
            const uint lastIdx = firstIdx + size;
            Array<float, Dims> centroid;
            #pragma unroll
            for (uint dimension = 0; dimension < Dims; ++dimension) {
                const uint offset = dimension * (pointSize + 1);
                const float firstScan = centroidScans[offset + firstIdx];
                const float lastScan = centroidScans[offset + lastIdx];
                centroid[dimension] = (lastScan - firstScan) / size;
            }
            nodeCentorids[nodeIdx] = centroid;
            const uint parentNodeIdx = parentOf(nodeIdx);
            if (nodeIdx % 2) {
                ballNodes[parentNodeIdx].leftCentroid = centroid;
            }
            else {
                ballNodes[parentNodeIdx].rightCentroid = centroid;
            } 
        }

        template <uint Dims>
        __global__
        void encodeDistanceValueToSortKernel(
                uint64_t* pointSortKey,
                const int* indexes,
                const Array<float, Dims>* points,
                const uint* nodePoints,
                const Array<float, Dims>* centroids,
                const uint size) {
            const uint idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx >= size) return;
            const uint nodeIdx = nodePoints[idx];
            const float distance2 = calcDistance2(centroids[nodeIdx], points[indexes[idx]]);
            pointSortKey[idx] = encodeUintUFloat(nodeIdx, distance2);
        }

        template <uint Dims>
        __global__
        void createNodeRadiusKernel(
                BallNode<Dims>* ballNodes,
                const Array<float, Dims>* centroids,
                const int* indexes,
                const Array<float, Dims>* points,
                const uint* nodeSizes,
                const uint* nodeSizeScans,
                const uint nodeBaseIdx,
                const uint size) {
            const uint idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx >= size) return;
            const uint nodeIdx = nodeBaseIdx + idx;
            const uint nodeSize = nodeSizes[nodeIdx];
            const uint nodeSizeScan = nodeSizeScans[idx];
            const Array<float, Dims> centroid = centroids[nodeIdx];
            const Array<float, Dims> radiusPoint = points[indexes[nodeSizeScan + nodeSize - 1]];
            const float radius = sqrt(calcDistance2(centroid, radiusPoint));
            const uint parentNodeIdx = parentOf(nodeIdx);
            if (nodeIdx % 2) {
                ballNodes[parentNodeIdx].leftRadius = radius;
            }
            else {
                ballNodes[parentNodeIdx].rightRadius = radius;
            }
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

    template <uint Dims, uint CBlockSize>
    class BallTreeGpu {
    public:
        using Point = Array<float, Dims>;

    private:
        uint mDepthSkips;
        uint mNodeDepth;
        uint mMaxChildren;
        uint mIndexSize;
        uint mNodeSize;
        uint mPointSize;
        uint mBinaryNodeSize;
        BallNode<Dims>* mDNodes;
        IndexPoint<Dims>* mDIndexPoints;

    private:
        __forceinline__
        void mSplitPoints(const Point* points, const uint pointSize) {
            using namespace BallTreeKernels;
            int* indexes;
            Cuda::check(cudaMalloc(&indexes, sizeof(int) * pointSize));
            const uint pointBlockCount = Cuda::blockCount(pointSize, CBlockSize);
            iotaKernel<<<pointBlockCount, CBlockSize>>>(indexes, pointSize);
            Point* nodeCentorids; uint* nodeSizes; uint* nodeSizeScans; uint* nodePoints;
            Cuda::check(cudaMalloc(&nodeCentorids, sizeof(Point) * (2 * mNodeSize + 1)));
            Cuda::check(cudaMalloc(&nodeSizes, sizeof(uint) * (2 * mNodeSize + 1)));
            Cuda::check(cudaMalloc(&nodeSizeScans, sizeof(uint) * (mNodeSize + 1)));
            Cuda::check(cudaMalloc(&nodePoints, sizeof(uint) * pointSize));
            float* distCentrScans; uint* nodeDims; 
            uint64_t* pointSort; uint64_t* pointSortOut; int* indexesOut;
            Cuda::check(cudaMalloc(&distCentrScans, sizeof(float) * (pointSize + 1) * Dims));
            Cuda::check(cudaMalloc(&nodeDims, sizeof(uint) * mNodeSize));
            Cuda::check(cudaMalloc(&pointSort, sizeof(uint64_t) * pointSize));
            Cuda::check(cudaMalloc(&pointSortOut, sizeof(uint64_t) * pointSize));
            Cuda::check(cudaMalloc(&indexesOut, sizeof(int) * pointSize));
            void* tempData = nullptr; size_t tempSize = 0; size_t tempSizeNodes = 0; size_t tempSizeSort = 0;
            cub::DeviceScan::ExclusiveSum(tempData, tempSize, distCentrScans, distCentrScans, pointSize + 1);
            cub::DeviceScan::ExclusiveSum(tempData, tempSizeNodes, nodeSizes, nodeSizeScans, mNodeSize + 1);
            cub::DeviceRadixSort::SortPairs(
                tempData, tempSizeSort, pointSort, 
                pointSortOut, indexes, indexesOut, pointSize
            );
            tempSize = std::max(tempSize, std::max(tempSizeNodes, tempSizeSort));
            Cuda::check(cudaMalloc(&tempData, tempSize));
            const uint zero = 0;
            Cuda::check(cudaMemcpy(nodeSizes, &pointSize, sizeof(uint), Cuda::H2D));
            Cuda::check(cudaMemcpy(nodeSizeScans, &zero, sizeof(uint), Cuda::H2D));
            fillKernel<uint><<<pointBlockCount, CBlockSize>>>(nodePoints, pointSize, 0);
            fillIndexesToDimsKernel<Dims><<<pointBlockCount, CBlockSize>>>(
                distCentrScans, indexes, points, pointSize
            );
            #pragma unroll
            for (uint dimension = 0; dimension < Dims; ++dimension) {
                float* dimDistCentrScans = distCentrScans + dimension * (pointSize + 1);
                cub::DeviceScan::ExclusiveSum(
                    tempData, tempSize, dimDistCentrScans, dimDistCentrScans, pointSize + 1
                );
            }
            createNodeCentroidsKernel<Dims><<<1, 32>>>(
                nodeCentorids, distCentrScans, nodeSizes, nodeSizeScans, 0, 1, pointSize
            );
            uint nodesBaseIndex = 0; uint nodeDepthCount = 1;
            for (uint depth = 0; depth < mNodeDepth; ++depth) {
                fillIndexesToVarianceKernel<Dims><<<pointBlockCount, CBlockSize>>>(
                    distCentrScans, indexes, points, nodePoints, nodeCentorids, pointSize
                );
                #pragma unroll
                for (uint dimension = 0; dimension < Dims; ++dimension) { 
                    float* dimDistCentrScans = distCentrScans + dimension * (pointSize + 1);
                    cub::DeviceScan::ExclusiveSum(
                        tempData, tempSize, dimDistCentrScans, dimDistCentrScans, pointSize + 1
                    );
                }
                const uint nodeBlockCount = Cuda::blockCount(nodeDepthCount, CBlockSize);
                pickNodeDimKernel<Dims><<<nodeBlockCount, CBlockSize>>>(
                    nodeDims, distCentrScans, nodeSizes, nodeSizeScans, 
                    nodesBaseIndex, nodeDepthCount, pointSize
                );
                encodeDimValueToSortKernel<Dims><<<pointBlockCount, CBlockSize>>>(
                    pointSort, indexes, points, nodePoints, nodeDims, pointSize
                );
                cub::DeviceRadixSort::SortPairs(
                    tempData, tempSizeSort, pointSort, pointSortOut, 
                    indexes, indexesOut, pointSize
                );
                swap(indexes, indexesOut);
                updateNodeSizesKernel<<<nodeBlockCount, CBlockSize>>>(
                    nodeSizes, nodeDepthCount, nodesBaseIndex
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
                fillIndexesToDimsKernel<Dims><<<pointBlockCount, CBlockSize>>>(
                    distCentrScans, indexes, points, pointSize
                );
                #pragma unroll
                for (uint dimension = 0; dimension < Dims; ++dimension) {
                    float* dimDistCentrScans = distCentrScans + dimension * (pointSize + 1);
                    cub::DeviceScan::ExclusiveSum(
                        tempData, tempSize, dimDistCentrScans, dimDistCentrScans, pointSize + 1
                    );
                }
                const uint nextNodeBlockCount = Cuda::blockCount(nodeDepthCount, CBlockSize);
                createNodeCentroidsKernel<Dims><<<nextNodeBlockCount, CBlockSize>>>(
                    nodeCentorids, mDNodes, distCentrScans, nodeSizes, nodeSizeScans, 
                    nodesBaseIndex, nodeDepthCount, pointSize
                );
                encodeDistanceValueToSortKernel<Dims><<<pointBlockCount, CBlockSize>>>(
                    pointSort, indexes, points, nodePoints, nodeCentorids, pointSize
                );
                cub::DeviceRadixSort::SortPairs(
                    tempData, tempSizeSort, pointSort, pointSortOut, 
                    indexes, indexesOut, pointSize
                );
                swap(indexes, indexesOut);
                createNodeRadiusKernel<Dims><<<nextNodeBlockCount, CBlockSize>>>(
                    mDNodes, nodeCentorids, indexes, points,
                    nodeSizes, nodeSizeScans, nodesBaseIndex, nodeDepthCount
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
            Cuda::check(cudaFree(nodeCentorids));
            Cuda::check(cudaFree(nodeSizes));
            Cuda::check(cudaFree(nodeSizeScans));
            Cuda::check(cudaFree(nodePoints));
            Cuda::check(cudaFree(distCentrScans));
            Cuda::check(cudaFree(nodeDims));
            Cuda::check(cudaFree(pointSort));
            Cuda::check(cudaFree(pointSortOut));
            Cuda::check(cudaFree(indexesOut));
            Cuda::check(cudaFree(tempData));
        }

    public:
        BallTreeGpu(const Point* points, const uint size, const uint depthSkips = 1):
                mDepthSkips(std::min(std::max(depthSkips, 1U), log2Ceil(size) - 1)),
                mNodeDepth(log2Ceil(size) - mDepthSkips),
                mMaxChildren(pow2(mDepthSkips)),
                mIndexSize(pow2(mNodeDepth + mDepthSkips)),
                mNodeSize(pow2(mNodeDepth) - 1) {
            Cuda::check(cudaMalloc(&mDNodes, sizeof(BallNode<Dims>) * mNodeSize));
            Cuda::check(cudaMalloc(&mDIndexPoints, sizeof(IndexPoint<Dims>) * mIndexSize));
            mSplitPoints(points, size);
            Cuda::check(cudaDeviceSynchronize());
        }

        ~BallTreeGpu() {
            cudaFree(mDNodes);
            cudaFree(mDIndexPoints);
        }

    };

}
