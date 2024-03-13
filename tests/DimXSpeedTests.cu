#include <iostream>
#include "../src/BruteForceGpu.cuh"
#include "../src/KdTreeGpu.cuh"
#include "../src/BallTreeGpu.cuh"
#include "Tests.cuh"


int main(int argc, char const *argv[]) {
    
    cudaFree(0);
    constexpr uint qryCount = 10000;
    constexpr uint k = 5;
    constexpr uint Dimensionalities[] = {6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17};


    std::cout << "\nRandom points range limited" << std::endl;
    forLoop<sizeof(Dimensionalities) / sizeof(Dimensionalities[0])>([&](auto idx) {
        constexpr uint dims = Dimensionalities[idx];
        float maxRange = 0.01 * dims;
        std::cout << "Dimensionality: " << dims << std::endl;
        // PointCount increase speed tests
        KnnTrees::Array<float, dims>* queryPoints = KnnTrees::pinnedMalloc<KnnTrees::Array<float, dims>>(qryCount);
        generateRandomPoints<dims>(queryPoints, qryCount, 1.0);
        KnnTrees::Array<float, dims>* dQueryPoints = gpuCopy(queryPoints, qryCount);
        constexpr uint PointCounts[] = {500000};
        forLoop<1>([&](auto i) {
            constexpr uint pointCount = PointCounts[i];
            KnnTrees::Array<float, dims>* points = KnnTrees::pinnedMalloc<KnnTrees::Array<float, dims>>(pointCount);
            generateRandomPoints<dims>(points, pointCount, 1.0);
            KnnTrees::Array<float, dims>* dPoints = gpuCopy(points, pointCount);
            KnnTrees::Query<dims, k> query = {queryPoints, qryCount, maxRange, nullptr, nullptr};
            query.rIndexes = KnnTrees::pinnedMalloc<KnnTrees::Array<int, k>>(qryCount);
            query.rDistances = KnnTrees::pinnedMalloc<KnnTrees::Array<float, k>>(qryCount);
            KnnTrees::Query<dims, k> dQuery = {dQueryPoints, qryCount, maxRange, nullptr, nullptr};
            KnnTrees::Cuda::check(cudaMalloc(&dQuery.rIndexes, sizeof(KnnTrees::Array<int, k>) * qryCount));
            KnnTrees::Cuda::check(cudaMalloc(&dQuery.rDistances, sizeof(KnnTrees::Array<float, k>) * qryCount));
            float kdTime = 0.0f; float ballTime = 0.0f; float bfTime = 0.0f;
            for (uint j = 0; j < 6; j++) {
                // KdTree GPU
                KnnTrees::KdTreeGpu<dims, dims, 32> kdg(dPoints, pointCount, 3);
                startTime();
                kdg.template batchKnn<k, 32, false, false, true>(dQuery);
                if (j != 0) kdTime += getTime();
                // BallTree GPU
                KnnTrees::BallTreeGpu<dims, 32> ballg(dPoints, pointCount, 1);            
                startTime();
                ballg.template batchKnn<k, 16, true, true, true>(dQuery);
                if (j != 0) ballTime += getTime();
                // BruteForce Gpu
                KnnTrees::BruteForceGpu<dims> bfg(dPoints, pointCount);            
                startTime();
                bfg.template batchKnn<k, 32, false, false>(dQuery);
                if (j != 0) bfTime += getTime();
            }
            std::cout << "KdTreeGpu - points: " << pointCount << ": " << (kdTime / 5) << "ms" << std::endl;
            std::cout << "BallTreeGpu - points: " << pointCount << ": " << (ballTime / 5) << "ms" << std::endl;
            std::cout << "BruteForceGpu - points: " << pointCount << ": " << (bfTime / 5) << "ms" << std::endl;
            KnnTrees::Cuda::check(cudaFree(dQuery.rDistances));
            KnnTrees::Cuda::check(cudaFree(dQuery.rIndexes));
            KnnTrees::pinnedFree(query.rDistances);
            KnnTrees::pinnedFree(query.rIndexes);
            KnnTrees::pinnedFree(points);
            KnnTrees::Cuda::check(cudaFree(dPoints));
        });
        KnnTrees::pinnedFree(queryPoints);
        KnnTrees::Cuda::check(cudaFree(dQueryPoints));
    });


    std::cout << "\nRandom points" << std::endl;
    forLoop<sizeof(Dimensionalities) / sizeof(Dimensionalities[0])>([&](auto idx) {
        constexpr uint dims = Dimensionalities[idx];
        std::cout << "Dimensionality: " << dims << std::endl;
        // PointCount increase speed tests
        KnnTrees::Array<float, dims>* queryPoints = KnnTrees::pinnedMalloc<KnnTrees::Array<float, dims>>(qryCount);
        generateRandomPoints<dims>(queryPoints, qryCount, 1.0);
        KnnTrees::Array<float, dims>* dQueryPoints = gpuCopy(queryPoints, qryCount);
        constexpr uint PointCounts[] = {500000};
        forLoop<1>([&](auto i) {
            constexpr uint pointCount = PointCounts[i];
            KnnTrees::Array<float, dims>* points = KnnTrees::pinnedMalloc<KnnTrees::Array<float, dims>>(pointCount);
            generateRandomPoints<dims>(points, pointCount, 1.0);
            KnnTrees::Array<float, dims>* dPoints = gpuCopy(points, pointCount);
            KnnTrees::Query<dims, k> query = {queryPoints, qryCount, 5.9f, nullptr, nullptr};
            query.rIndexes = KnnTrees::pinnedMalloc<KnnTrees::Array<int, k>>(qryCount);
            query.rDistances = KnnTrees::pinnedMalloc<KnnTrees::Array<float, k>>(qryCount);
            KnnTrees::Query<dims, k> dQuery = {dQueryPoints, qryCount, 5.9f, nullptr, nullptr};
            KnnTrees::Cuda::check(cudaMalloc(&dQuery.rIndexes, sizeof(KnnTrees::Array<int, k>) * qryCount));
            KnnTrees::Cuda::check(cudaMalloc(&dQuery.rDistances, sizeof(KnnTrees::Array<float, k>) * qryCount));
            float kdTime = 0.0f; float ballTime = 0.0f; float bfTime = 0.0f;
            for (uint j = 0; j < 6; j++) {
                // KdTree GPU
                KnnTrees::KdTreeGpu<dims, dims, 32> kdg(dPoints, pointCount, 3);
                startTime();
                kdg.template batchKnn<k, 32, false, false, true>(dQuery);
                if (j != 0) kdTime += getTime();
                // BallTree GPU
                KnnTrees::BallTreeGpu<dims, 32> ballg(dPoints, pointCount, 1);            
                startTime();
                ballg.template batchKnn<k, 16, true, true, true>(dQuery);
                if (j != 0) ballTime += getTime();
                // BruteForce Gpu
                KnnTrees::BruteForceGpu<dims> bfg(dPoints, pointCount);            
                startTime();
                bfg.template batchKnn<k, 32, false, false>(dQuery);
                if (j != 0) bfTime += getTime();
            }
            std::cout << "KdTreeGpu - points: " << pointCount << ": " << (kdTime / 5) << "ms" << std::endl;
            std::cout << "BallTreeGpu - points: " << pointCount << ": " << (ballTime / 5) << "ms" << std::endl;
            std::cout << "BruteForceGpu - points: " << pointCount << ": " << (bfTime / 5) << "ms" << std::endl;
            KnnTrees::Cuda::check(cudaFree(dQuery.rDistances));
            KnnTrees::Cuda::check(cudaFree(dQuery.rIndexes));
            KnnTrees::pinnedFree(query.rDistances);
            KnnTrees::pinnedFree(query.rIndexes);
            KnnTrees::pinnedFree(points);
            KnnTrees::Cuda::check(cudaFree(dPoints));
        });
        KnnTrees::pinnedFree(queryPoints);
        KnnTrees::Cuda::check(cudaFree(dQueryPoints));
    });


    return 0;
}
