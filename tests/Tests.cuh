#pragma once
#include <set>
#include <vector>
#include <random>
#include <float.h>
#include "../src/KnnTrees.cuh"


template <typename Type>
Type* cpuCopy(Type* gpuData, const uint size) {
    Type* cpuData = KnnTrees::pinnedMalloc<Type>(size);
    KnnTrees::Cuda::check(cudaMemcpy(cpuData, gpuData, sizeof(Type) * size, KnnTrees::Cuda::D2H));
    return cpuData;
}

template <typename Type>
Type* gpuCopy(Type* cpuData, const uint size) {
    Type* gpuData;
    KnnTrees::Cuda::check(cudaMalloc(&gpuData, sizeof(Type) * size));
    KnnTrees::Cuda::check(cudaMemcpy(gpuData, cpuData, sizeof(Type) * size, KnnTrees::Cuda::H2D));
    return gpuData;
}

template <uint Dims, uint K>
void renewOutput(KnnTrees::Query<Dims, K>& query, const uint qryCount) {
    auto* oldA = query.rDistances;
    auto* oldB = query.rIndexes;
    query.rIndexes = KnnTrees::pinnedMalloc<KnnTrees::Array<int, K>>(qryCount);
    query.rDistances = KnnTrees::pinnedMalloc<KnnTrees::Array<float, K>>(qryCount);
    KnnTrees::pinnedFree(oldA);
    KnnTrees::pinnedFree(oldB);
}

template <uint Dims, uint K>
void renewOutputGpu(KnnTrees::Query<Dims, K>& dQuery, const uint qryCount) {
    auto* oldA = dQuery.rDistances;
    auto* oldB = dQuery.rIndexes;
    KnnTrees::Cuda::check(cudaMalloc(&dQuery.rIndexes, sizeof(KnnTrees::Array<int, K>) * qryCount));
    KnnTrees::Cuda::check(cudaMalloc(&dQuery.rDistances, sizeof(KnnTrees::Array<float, K>) * qryCount));
    KnnTrees::Cuda::check(cudaFree(oldA));
    KnnTrees::Cuda::check(cudaFree(oldB));
}

template <uint K>
uint checkQueryErrors(
        const KnnTrees::Array<int, K>* test, 
        const KnnTrees::Array<int, K>* correct, 
        const uint qryCount) {
        uint errorCount = 0;
    std::vector<std::set<int>> results(qryCount);
    for (uint i = 0; i < qryCount; i++) {
        results[i] = std::set<int>();
        for (uint j = 0; j < K; j++) {
            results[i].insert(correct[i][j]);
        }
    }
    for (uint i = 0; i < qryCount; i++) {
        uint error = 0;
        for (uint j = 0; j < K; j++) {
            if (results[i].end() == results[i].find(test[i][j])) {
                error = 1;
            }
        }
        errorCount += error;
    }
    return errorCount;
}

template <uint Dims>
void generateRandomPoints(
        KnnTrees::Array<float, Dims>* points, 
        const uint pointCount,
        const float range) {
    std::random_device rd;
    std::mt19937 gen(rd());
    for (uint i = 0; i < pointCount; ++i) {
        for (uint j = 0; j < Dims; ++j) { 
            std::uniform_real_distribution<float> dist(0, range);
            points[i][j] = dist(gen);
        }
    }
}
