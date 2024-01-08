#pragma once
#include "KnnTrees.cuh"
#include <numeric>
#include <algorithm>
#include <limits>


namespace KnnTrees {

    template <uint Dims>
    class BallTree {
    public:
        using Point = Array<float, Dims>;

    private:
        uint mDepthSkips;
        uint mNodeDepth;
        uint mMaxChildren;
        uint mIndexSize;
        uint mNodeSize;
        uint mBinaryNodeSize;
        uint mPointSize;
        BallNode<Dims>* mNodes;
        IndexPoint<Dims>* mIndexPoints;

    private:
        void mConstruct(
                const uint idx,
                int* indexes,
                const Point* points,
                const uint size,
                float* distances) {
            if (idx < mNodeSize) {
                const uint rightCount = size / 2;
                const uint leftCount = rightCount + size % 2;
                int* pivotPoint = indexes + leftCount - 1;
                float dimMeans[Dims]{};
                for (uint i = 0; i < size; ++i) {
                    for (uint dim = 0; dim < Dims; dim++) {
                        dimMeans[dim] += points[indexes[i]][dim];
                    }
                }
                for (uint dim = 0; dim < Dims; dim++) {
                    dimMeans[dim] = dimMeans[dim] / size;
                }
                float dimVariances[Dims]{};
                for (uint i = 0; i < size; i++) {
                    for (uint dim = 0; dim < Dims; ++dim) {
                        float diff = points[indexes[i]][dim] - dimMeans[dim];
                        dimVariances[dim] += diff * diff;
                    }
                }
                for (uint dim = 0; dim < Dims; dim++) {
                    dimVariances[dim] = dimVariances[dim] / size;
                }
                uint bestDim = 0;
                float highestVariance = 0.0f;
                for (uint dim = 0; dim < Dims; ++dim) {
                    if (dimVariances[dim] > highestVariance) {
                        bestDim = dim;
                        highestVariance = dimVariances[dim];
                    }
                }
                const auto sortLambda = [points, bestDim](const int& idx1, const int& idx2) {
                    return points[idx1][bestDim] < points[idx2][bestDim];
                };
                std::nth_element(indexes, pivotPoint, indexes + size, sortLambda);
                int* rightIndexes = indexes + leftCount;
                const uint leftIdx = leftChildOf(idx);
                float dimSums[Dims]{};
                for (uint i = 0; i < leftCount; ++i) {
                    for (uint dim = 0; dim < Dims; dim++) {
                        dimSums[dim] += points[indexes[i]][dim];
                    }
                }
                for (uint dim = 0; dim < Dims; dim++) {
                    mNodes[idx].leftCentroid[dim] = dimSums[dim] / leftCount;
                }
                float maxDist = 0.0f;
                for (uint i = 0; i < leftCount; ++i) {
                    const float distance = calcDistance2(mNodes[idx].leftCentroid, points[indexes[i]]);
                    if (distance > maxDist) {
                        maxDist = distance;
                    }
                }
                mNodes[idx].leftRadius = std::sqrt(maxDist);
                float dimSumsR[Dims]{};
                for (uint i = 0; i < rightCount; ++i) {
                    for (uint dim = 0; dim < Dims; dim++) {
                        dimSumsR[dim] += points[rightIndexes[i]][dim];
                    }
                }
                for (uint dim = 0; dim < Dims; dim++) {
                    mNodes[idx].rightCentroid[dim] = dimSumsR[dim] / rightCount;
                }
                float maxDistR = 0.0f;
                for (uint i = 0; i < rightCount; ++i) {
                    const float distance = calcDistance2(mNodes[idx].rightCentroid, points[rightIndexes[i]]);
                    if (distance > maxDistR) {
                        maxDistR = distance;
                    }
                }
                mNodes[idx].rightRadius = std::sqrt(maxDistR);
                mConstruct(leftIdx, indexes, points, leftCount, distances);
                mConstruct(leftIdx + 1, rightIndexes, points, rightCount, distances);
                return;
            }
            const uint idxToIdx = mMaxChildren * (idx - mNodeSize);
            uint offset = 0;
            while (offset < size) {
                const int indexx = indexes[offset];
                mIndexPoints[idxToIdx + offset].index = indexx;
                mIndexPoints[idxToIdx + offset].point = points[indexx];
                offset += 1;
            }
            while (offset < mMaxChildren) {
                mIndexPoints[idxToIdx + offset].index = -1;
                offset += 1;
            }
        }

    public:
        BallTree(const Point* points, const uint size, const uint depthSkips = 1):
                mDepthSkips(std::min(std::max(depthSkips, 1U), log2Ceil(size) - 1)),
                mNodeDepth(log2Ceil(size) - mDepthSkips),
                mMaxChildren(pow2(mDepthSkips)),
                mIndexSize(pow2(mNodeDepth + mDepthSkips)),
                mNodeSize(pow2(mNodeDepth) - 1),
                mPointSize(size),
                mNodes(new BallNode<Dims>[mNodeSize]),
                mIndexPoints(new IndexPoint<Dims>[mIndexSize]) {
            int* sortIndexes = new int[mPointSize];
            std::iota(sortIndexes, sortIndexes + mPointSize, 0);
            float* distances = new float[mPointSize];
            mConstruct(0, sortIndexes, points, mPointSize, distances);
            delete[] distances;
        }

        ~BallTree() {
            delete[] mNodes;
            delete[] mIndexPoints;
        }

    private:
        template <uint K, bool HeapNN>
        __forceinline__ 
        float mConsider(
                float maxDistance2,
                uint64_t* nearestNeighbours,
                const uint idx,
                const Array<float, Dims>& point) const {
            const uint idxIdx = mMaxChildren * (idx - mNodeSize);
            for (uint offset = 0; offset < mMaxChildren; ++offset) {
                const IndexPoint<Dims> indexPoint = mIndexPoints[idxIdx + offset];
                if (indexPoint.index == -1) break;
                float distance2 = calcDistance2<Dims>(point, indexPoint.point);
                if constexpr (HeapNN) 
                    maxDistance2 = heapConsider<K>(nearestNeighbours, indexPoint.index, distance2);
                else 
                    maxDistance2 = arrayConsider<K>(nearestNeighbours, indexPoint.index, distance2);
            }
            return maxDistance2;
        }

        template <uint K, bool HeapNN>
        void mKnn(
                uint64_t* nearestNeighbours, 
                const Array<float, Dims>& point, 
                float maxDistance2) const {
            constexpr uint STACK_SIZE = sizeof(uint) * 8;
            uint indexStack[STACK_SIZE];
            float distanceStack[STACK_SIZE];
            uint stackSize = 1;
            indexStack[0] = 0; 
            distanceStack[0] = 0.0f;
            while (stackSize) {
                if (distanceStack[--stackSize] >= maxDistance2) continue;
                uint nodeIdx = indexStack[stackSize];
                bool skipNode = false;
                while (nodeIdx < mNodeSize) {
                    BallNode<Dims> node = mNodes[nodeIdx];
                    const float leftDistance = std::sqrt(calcDistance2(node.leftCentroid, point));
                    const float leftBorderDistance = leftDistance - node.leftRadius;
                    const float leftBorderPositiveDistance = max(leftBorderDistance, 0.0f);
                    const float leftBorderDistance2 = leftBorderPositiveDistance * leftBorderPositiveDistance;
                    const float rightDistance = std::sqrt(calcDistance2(node.rightCentroid, point));
                    const float rightBorderDistance = rightDistance - node.rightRadius;
                    const float rightBorderPositiveDistance = max(rightBorderDistance, 0.0f);
                    const float rightBorderDistance2 = rightBorderPositiveDistance * rightBorderPositiveDistance;
                    uint closeChild, farChild;
                    float closeBorderDistance2, farBorderDistance2;
                    const uint leftChild = leftChildOf(nodeIdx);
                    if (leftBorderDistance < rightBorderDistance) {
                        closeChild = leftChild;
                        closeBorderDistance2 = leftBorderDistance2;
                        farChild = leftChild + 1;
                        farBorderDistance2 = rightBorderDistance2;
                    }
                    else {
                        farChild = leftChild;
                        farBorderDistance2 = leftBorderDistance2;
                        closeChild = leftChild + 1;
                        closeBorderDistance2 = rightBorderDistance2;
                    }
                    if (closeBorderDistance2 >= maxDistance2) {
                        skipNode = true;
                        break;
                    }
                    if (farBorderDistance2 < maxDistance2) {
                        indexStack[stackSize] = farChild;
                        distanceStack[stackSize] = farBorderDistance2;
                        stackSize += 1;
                    }
                    nodeIdx = closeChild;
                }
                if (!skipNode) {
                    maxDistance2 = mConsider<K, HeapNN>(
                        maxDistance2, nearestNeighbours, nodeIdx, point
                    );
                }
            }
        }

    public:
        template <uint K, bool HeapNN>
        void batchKnn(Query<Dims, K>& query) const {
            uint64_t nearestNeighbours[K];
            for (uint idx = 0; idx < query.count; ++idx) {
                fillArray<uint64_t, K>(nearestNeighbours, encodeUFloatInt(query.maxDistance2, -1));
                mKnn<K, HeapNN>(nearestNeighbours, query.points[idx], query.maxDistance2);
                for (uint i = 0; i < K; ++i) {
                    const uint64_t neighbour = nearestNeighbours[i];
                    query.rIndexes[idx][i] = static_cast<int>(neighbour);
                    query.rDistances[idx][i] = decodeEncoded1UFloat(neighbour);
                }
            }
        }

    };

}
