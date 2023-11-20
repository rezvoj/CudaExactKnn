#pragma once
#include <cstring>
#include "KnnTrees.cuh"


namespace KnnTrees {

    template <uint Dims>
    class BruteForce {
    public:
        using Point = Array<float, Dims>;

    private:
        uint mSize;
        Point* mData;

    public:
        BruteForce(const Point* points, const uint size): 
                mSize(size), mData(new Point[mSize]) {
            std::memcpy(mData, points, sizeof(Point) * mSize);
        }
        
        ~BruteForce() {
            delete[] mData;
        }
 
        template <uint K, bool HeapNN>
        void batchKnn(Query<Dims, K>& query) const {
            uint64_t nearestNeighbours[K];
            for (uint idx = 0; idx < query.count; ++idx) {
                fillArray<uint64_t, K>(nearestNeighbours, encodeUFloatInt(query.maxDistance2, -1));
                for (uint index = 0; index < mSize; ++index) {
                    const float distance2 = calcDistance2<Dims>(query.points[idx], mData[index]);
                    if constexpr (HeapNN) heapConsider<K>(nearestNeighbours, index, distance2);
                    else arrayConsider<K>(nearestNeighbours, index, distance2);
                }
                #pragma unroll
                for (uint i = 0; i < K; ++i) {
                    const uint64_t neighbour = nearestNeighbours[i];
                    query.rIndexes[idx][i] = static_cast<int>(neighbour);
                    query.rDistances[idx][i] = decodeEncoded1UFloat(neighbour);
                }
            }
        }

    };

}
