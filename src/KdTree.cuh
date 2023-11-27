#pragma once
#include <algorithm>
#include <numeric>
#include <cstring>
#include "KnnTrees.cuh"


namespace KnnTrees {

    template <uint Dims>
    class KdTree {
    public:
        using Point = Array<float, Dims>;

    private:
        uint mDepthSkips;
        uint mNodeDepth;
        uint mMaxChildren;
        uint mIndexSize;
        uint mNodeSize;
        uint mPointSize;
        int* mIndexes;
        float* mNodes;
        Point* mPoints;

    private:
        void mConstruct(const uint idx, int* indexes, const uint size, const uint depth) {
            if (depth < mNodeDepth) {
                const uint rightCount = size / 2;
                const uint leftCount = rightCount + size % 2;
                const uint dimIdx = depth % Dims;
                int* pivotPoint = indexes + leftCount - 1;
                const auto sortLambda = [this, dimIdx](const int& idx1, const int& idx2) {
                    return mPoints[idx1][dimIdx] < mPoints[idx2][dimIdx];
                };
                std::nth_element(indexes, pivotPoint, indexes + size, sortLambda);
                mNodes[idx] = mPoints[*pivotPoint][dimIdx];
                const uint leftIdx = leftChildOf(idx);
                mConstruct(leftIdx, indexes, leftCount, depth + 1);
                mConstruct(leftIdx + 1, indexes + leftCount, rightCount, depth + 1);
                return;
            }
            const uint idxIdx = mMaxChildren * (idx - mNodeSize);
            uint offset = 0;
            while (offset < size) {
                mIndexes[idxIdx + offset] = indexes[offset];
                offset += 1;
            }
            while (offset < mMaxChildren) {
                mIndexes[idxIdx + offset] = -1;
                offset += 1;
            }
        }

    public:
        KdTree(const Point* points, const uint size, const uint depthSkips = 0):
                mDepthSkips(std::min(depthSkips, log2Ceil(size) - 1)),
                mNodeDepth(log2Ceil(size) - mDepthSkips),
                mMaxChildren(pow2(mDepthSkips)),
                mIndexSize(pow2(mNodeDepth + mDepthSkips)),
                mNodeSize(pow2(mNodeDepth) - 1),
                mPointSize(size),
                mIndexes(new int[mIndexSize]),
                mNodes(new float[mNodeSize]),
                mPoints(new Point[mPointSize]) {
            std::memcpy(mPoints, points, sizeof(Point) * mPointSize);
            int* sortIndexes = new int[mPointSize];
            std::iota(sortIndexes, sortIndexes + mPointSize, 0);
            mConstruct(0, sortIndexes, mPointSize, 0);
            delete[] sortIndexes;
        }

        ~KdTree() {
            delete[] mIndexes;
            delete[] mNodes;
            delete[] mPoints;
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
                const int pointIdx = mIndexes[idxIdx + offset];
                if (pointIdx == -1) break;                
                float distance2 = calcDistance2<Dims>(point, mPoints[pointIdx]);
                if constexpr (HeapNN) maxDistance2 = heapConsider<K>(nearestNeighbours, pointIdx, distance2);
                else maxDistance2 = arrayConsider<K>(nearestNeighbours, pointIdx, distance2);
            }
            return maxDistance2;
        }

        template <uint K, bool HeapNN>
        __forceinline__ 
        void mKnn(
                uint64_t* nearestNeighbours, 
                const Array<float, Dims>& point, 
                float maxDistance2) const {
            constexpr uint STACK_SIZE = sizeof(uint) * 8;
            uint indexStack[STACK_SIZE];
            float distanceStack[STACK_SIZE];
            uint depthStack[STACK_SIZE];
            uint stackSize = 1;
            indexStack[0] = 0; 
            distanceStack[0] = 0.0f;
            depthStack[0] = 0;
            while (stackSize) {
                stackSize -= 1;
                if (distanceStack[stackSize] >= maxDistance2) continue;
                uint idx = indexStack[stackSize];
                uint depth = depthStack[stackSize];
                while (depth < mNodeDepth) {
                    const uint dimension = depth % Dims;
                    uint closeChild, farChild;
                    const float distance = point[dimension] - mNodes[idx];
                    if (distance < 0.0f) {
                        closeChild = leftChildOf(idx);
                        farChild = closeChild + 1;
                    }
                    else {
                        farChild = leftChildOf(idx);
                        closeChild = farChild + 1;
                    }
                    const float distance2 = distance * distance;
                    if (distance2 < maxDistance2) {
                        indexStack[stackSize] = farChild;
                        depthStack[stackSize] = depth + 1;
                        distanceStack[stackSize] = distance2;
                        stackSize += 1;
                    }
                    idx = closeChild;
                    depth += 1;
                }
                maxDistance2 = mConsider<K, HeapNN>(
                    maxDistance2, nearestNeighbours, idx, point
                );
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
