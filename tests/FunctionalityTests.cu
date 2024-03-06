#include <iostream>
#include "../src/BruteForce.cuh"
#include "../src/KdTree.cuh"
#include "../src/BallTree.cuh"
#include "../src/BruteForceGpu.cuh"
#include "../src/KdTreeGpu.cuh"
#include "../src/BallTreeGpu.cuh"
#include "Tests.cuh"


int main(int argc, char const *argv[]) {
    
    cudaFree(0);
    constexpr uint dims = 4;
    constexpr uint pointCount = 100000;
    constexpr uint qryCount = 50000;
    constexpr uint k = 10;

    // Init data
    KnnTrees::Array<float, dims>* points = KnnTrees::pinnedMalloc<KnnTrees::Array<float, dims>>(pointCount);
    generateRandomPoints<dims>(points, pointCount, 1.0);
    KnnTrees::Array<float, dims>* dPoints = gpuCopy(points, pointCount);
    KnnTrees::Array<float, dims>* queryPoints = KnnTrees::pinnedMalloc<KnnTrees::Array<float, dims>>(qryCount);
    generateRandomPoints<dims>(queryPoints, qryCount, 1.0);
    KnnTrees::Array<float, dims>* dQueryPoints = gpuCopy(queryPoints, qryCount);
    KnnTrees::Query<dims, k> query = {queryPoints, qryCount, 5.9f, nullptr, nullptr};
    query.rIndexes = KnnTrees::pinnedMalloc<KnnTrees::Array<int, k>>(qryCount);
    query.rDistances = KnnTrees::pinnedMalloc<KnnTrees::Array<float, k>>(qryCount);
    KnnTrees::Query<dims, k> dQuery = {dQueryPoints, qryCount, 5.9f, nullptr, nullptr};
    KnnTrees::Cuda::check(cudaMalloc(&dQuery.rIndexes, sizeof(KnnTrees::Array<int, k>) * qryCount));
    KnnTrees::Cuda::check(cudaMalloc(&dQuery.rDistances, sizeof(KnnTrees::Array<float, k>) * qryCount));
    KnnTrees::Array<int, k>* copy;

    // Correct Result
    KnnTrees::BruteForceGpu<dims> bfg(dPoints, pointCount);
    bfg.batchKnn<k, 16, false, false>(dQuery);
    KnnTrees::Array<int, k>* result = cpuCopy(dQuery.rIndexes, qryCount);
    renewOutputGpu(dQuery, qryCount);

    // BruteForce
    KnnTrees::BruteForce<dims> bf(points, pointCount);
    bf.batchKnn<k, false>(query);
    assert(checkQueryErrors(query.rIndexes, result, qryCount) == 0);
    renewOutput(query, qryCount);
    bf.batchKnn<k, true>(query);
    assert(checkQueryErrors(query.rIndexes, result, qryCount) == 0);
    renewOutput(query, qryCount);

    // KdTree
    KnnTrees::KdTree<dims> kd(points, pointCount, 3);
    kd.batchKnn<k, false>(query);
    assert(checkQueryErrors(query.rIndexes, result, qryCount) == 0);
    renewOutput(query, qryCount);
    KnnTrees::KdTree<dims> kd2(points, pointCount, 0);
    kd2.batchKnn<k, true>(query);
    assert(checkQueryErrors(query.rIndexes, result, qryCount) == 0);
    renewOutput(query, qryCount);

    // BallTree
    KnnTrees::BallTree<dims> ball(points, pointCount, 3);
    ball.batchKnn<k, false>(query);
    assert(checkQueryErrors(query.rIndexes, result, qryCount) == 0);
    renewOutput(query, qryCount);
    KnnTrees::BallTree<dims> ball2(points, pointCount, 0);
    ball2.batchKnn<k, true>(query);
    assert(checkQueryErrors(query.rIndexes, result, qryCount) == 0);
    renewOutput(query, qryCount);

    // BruteForceGpu
    bfg.batchKnn<k, 16, true, false>(dQuery);
    copy = cpuCopy(dQuery.rIndexes, qryCount);
    assert(checkQueryErrors(copy, result, qryCount) == 0);
    KnnTrees::pinnedFree(copy);
    renewOutputGpu(dQuery, qryCount);
    bfg.batchKnn<k, 16, false, true>(dQuery);
    copy = cpuCopy(dQuery.rIndexes, qryCount);
    assert(checkQueryErrors(copy, result, qryCount) == 0);
    KnnTrees::pinnedFree(copy);
    renewOutputGpu(dQuery, qryCount);

    // KdTreeGpu
    KnnTrees::KdTreeGpu<dims, dims, 16> kdg(dPoints, pointCount, 0);
    kdg.batchKnn<k, 16, true, false, true>(dQuery);
    copy = cpuCopy(dQuery.rIndexes, qryCount);
    assert(checkQueryErrors(copy, result, qryCount) == 0);
    KnnTrees::pinnedFree(copy);
    renewOutputGpu(dQuery, qryCount);
    KnnTrees::KdTreeGpu<dims, 3, 16> kdg2(dPoints, pointCount, 3);
    kdg2.batchKnn<k, 16, false, true, false>(dQuery);
    copy = cpuCopy(dQuery.rIndexes, qryCount);
    assert(checkQueryErrors(copy, result, qryCount) == 0);
    KnnTrees::pinnedFree(copy);
    renewOutputGpu(dQuery, qryCount);

    // BallTreeGpu
    KnnTrees::BallTreeGpu<dims, 16> ballg(dPoints, pointCount, 0);
    ballg.batchKnn<k, 16, true, true, true>(dQuery);
    copy = cpuCopy(dQuery.rIndexes, qryCount);
    assert(checkQueryErrors(copy, result, qryCount) == 0);
    KnnTrees::pinnedFree(copy);
    renewOutputGpu(dQuery, qryCount);
    KnnTrees::BallTreeGpu<dims, 16> ballg2(dPoints, pointCount, 2);
    ballg2.batchKnn<k, 16, false, false, false>(dQuery);
    copy = cpuCopy(dQuery.rIndexes, qryCount);
    assert(checkQueryErrors(copy, result, qryCount) == 0);
    KnnTrees::pinnedFree(copy);
    renewOutputGpu(dQuery, qryCount);

    KnnTrees::Cuda::check(cudaFree(dQuery.rDistances));
    KnnTrees::Cuda::check(cudaFree(dQuery.rIndexes));
    KnnTrees::pinnedFree(query.rDistances);
    KnnTrees::pinnedFree(query.rIndexes);
    KnnTrees::pinnedFree(queryPoints);
    KnnTrees::Cuda::check(cudaFree(dQueryPoints));
    KnnTrees::pinnedFree(points);
    KnnTrees::Cuda::check(cudaFree(dPoints));

    std::cout << "All tests passed" << std::endl;

    return 0;
}
