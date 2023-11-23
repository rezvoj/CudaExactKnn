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

    };

}
