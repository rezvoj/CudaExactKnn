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
    constexpr uint dims = 5;
    constexpr uint pointCount = 100000;
    constexpr uint qryCount = 10000;
    constexpr uint k = 10;

    // QryCount increase speed tests
    KnnTrees::Array<float, dims>* points = KnnTrees::pinnedMalloc<KnnTrees::Array<float, dims>>(pointCount);
    generateRandomPoints<dims>(points, pointCount, 1.0);
    KnnTrees::Array<float, dims>* dPoints = gpuCopy(points, pointCount);
    constexpr uint QryCounts[] = {1000, 5000, 10000, 25000, 50000, 100000};
    forLoop<6>([&](auto i) {
        constexpr uint qryCount = QryCounts[i];
        KnnTrees::Array<float, dims>* queryPoints = KnnTrees::pinnedMalloc<KnnTrees::Array<float, dims>>(qryCount);
        generateRandomPoints<dims>(queryPoints, qryCount, 1.0);
        KnnTrees::Array<float, dims>* dQueryPoints = gpuCopy(queryPoints, qryCount);
        KnnTrees::Query<dims, k> query = {queryPoints, qryCount, 5.9f, nullptr, nullptr};
        query.rIndexes = KnnTrees::pinnedMalloc<KnnTrees::Array<int, k>>(qryCount);
        query.rDistances = KnnTrees::pinnedMalloc<KnnTrees::Array<float, k>>(qryCount);
        KnnTrees::Query<dims, k> dQuery = {dQueryPoints, qryCount, 5.9f, nullptr, nullptr};
        KnnTrees::Cuda::check(cudaMalloc(&dQuery.rIndexes, sizeof(KnnTrees::Array<int, k>) * qryCount));
        KnnTrees::Cuda::check(cudaMalloc(&dQuery.rDistances, sizeof(KnnTrees::Array<float, k>) * qryCount));
        float kdTime = 0.0f; float ballTime = 0.0f; float bfTime = 0.0f;
        float kdTimeCpu = 0.0f; float ballTimeCpu = 0.0f; float bfTimeCpu = 0.0f;
        for (uint j = 0; j < 6; j++) {
            // KdTree GPU
            KnnTrees::KdTreeGpu<dims, dims, 32> kdg(dPoints, pointCount, 3);
            startTime();
            kdg.batchKnn<k, 32, false, false, true>(dQuery);
            if (j != 0) kdTime += getTime();
            // BallTree GPU
            KnnTrees::BallTreeGpu<dims, 32> ballg(dPoints, pointCount, 1);            
            startTime();
            ballg.batchKnn<k, 16, true, true, true>(dQuery);
            if (j != 0) ballTime += getTime();
            // BruteForce Gpu
            KnnTrees::BruteForceGpu<dims> bfg(dPoints, pointCount);            
            startTime();
            bfg.batchKnn<k, 32, false, false>(dQuery);
            if (j != 0) bfTime += getTime();
            // KdTree
            KnnTrees::KdTree<dims> kd(points, pointCount, 3);
            startTime();
            kd.batchKnn<k, false>(query);
            if (j != 0) kdTimeCpu += getTime();
            // BallTree
            KnnTrees::BallTree<dims> ball(points, pointCount, 1);
            startTime();
            ball.batchKnn<k, false>(query);
            if (j != 0) ballTimeCpu += getTime();
            // BruteForce
            KnnTrees::BruteForce<dims> bf(points, pointCount);
            startTime();
            bf.batchKnn<k, false>(query);
            if (j != 0) bfTimeCpu += getTime();
        }
        std::cout << "KdTreeGpu - qries: " << qryCount << ": " << (kdTime / 5) << "ms" << std::endl;
        std::cout << "BallTreeGpu - qries: " << qryCount << ": " << (ballTime / 5) << "ms" << std::endl;
        std::cout << "BruteForceGpu - qries: " << qryCount << ": " << (bfTime / 5) << "ms" << std::endl;
        std::cout << "KdTree - qries: " << qryCount << ": " << (kdTimeCpu / 5) << "ms" << std::endl;
        std::cout << "BallTree - qries: " << qryCount << ": " << (ballTimeCpu / 5) << "ms" << std::endl;
        std::cout << "BruteForce - qries: " << qryCount << ": " << (bfTimeCpu / 5) << "ms" << std::endl;
        KnnTrees::Cuda::check(cudaFree(dQuery.rDistances));
        KnnTrees::Cuda::check(cudaFree(dQuery.rIndexes));
        KnnTrees::pinnedFree(query.rDistances);
        KnnTrees::pinnedFree(query.rIndexes);
        KnnTrees::pinnedFree(queryPoints);
        KnnTrees::Cuda::check(cudaFree(dQueryPoints));
    });
    KnnTrees::pinnedFree(points);
    KnnTrees::Cuda::check(cudaFree(dPoints));


    // PointCount increase speed tests
    KnnTrees::Array<float, dims>* queryPoints = KnnTrees::pinnedMalloc<KnnTrees::Array<float, dims>>(qryCount);
    generateRandomPoints<dims>(queryPoints, qryCount, 1.0);
    KnnTrees::Array<float, dims>* dQueryPoints = gpuCopy(queryPoints, qryCount);
    constexpr uint PointCounts[] = {10000, 50000, 100000, 250000, 500000, 1000000};
    forLoop<6>([&](auto i) {
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
        float kdTimeCpu = 0.0f; float ballTimeCpu = 0.0f; float bfTimeCpu = 0.0f;
        for (uint j = 0; j < 6; j++) {
            // KdTree GPU
            KnnTrees::KdTreeGpu<dims, dims, 32> kdg(dPoints, pointCount, 3);
            startTime();
            kdg.batchKnn<k, 32, false, false, true>(dQuery);
            if (j != 0) kdTime += getTime();
            // BallTree GPU
            KnnTrees::BallTreeGpu<dims, 32> ballg(dPoints, pointCount, 1);
            startTime();
            ballg.batchKnn<k, 16, true, true, true>(dQuery);
            if (j != 0) ballTime += getTime();
            // BruteForce Gpu
            KnnTrees::BruteForceGpu<dims> bfg(dPoints, pointCount);            
            startTime();
            bfg.batchKnn<k, 32, false, false>(dQuery);
            if (j != 0) bfTime += getTime();
            // KdTree
            KnnTrees::KdTree<dims> kd(points, pointCount, 3);
            startTime();
            kd.batchKnn<k, false>(query);
            if (j != 0) kdTimeCpu += getTime();
            // BallTree
            KnnTrees::BallTree<dims> ball(points, pointCount, 1);
            startTime();
            ball.batchKnn<k, false>(query);
            if (j != 0) ballTimeCpu += getTime();
            // BruteForce
            KnnTrees::BruteForce<dims> bf(points, pointCount);
            startTime();
            bf.batchKnn<k, false>(query);
            if (j != 0) bfTimeCpu += getTime();
        }
        std::cout << "KdTreeGpu - points: " << pointCount << ": " << (kdTime / 5) << "ms" << std::endl;
        std::cout << "BallTreeGpu - points: " << pointCount << ": " << (ballTime / 5) << "ms" << std::endl;
        std::cout << "BruteForceGpu - points: " << pointCount << ": " << (bfTime / 5) << "ms" << std::endl;
        std::cout << "KdTree - points: " << pointCount << ": " << (kdTimeCpu / 5) << "ms" << std::endl;
        std::cout << "BallTree - points: " << pointCount << ": " << (ballTimeCpu / 5) << "ms" << std::endl;
        std::cout << "BruteForce - points: " << pointCount << ": " << (bfTimeCpu / 5) << "ms" << std::endl;
        KnnTrees::Cuda::check(cudaFree(dQuery.rDistances));
        KnnTrees::Cuda::check(cudaFree(dQuery.rIndexes));
        KnnTrees::pinnedFree(query.rDistances);
        KnnTrees::pinnedFree(query.rIndexes);
        KnnTrees::pinnedFree(points);
        KnnTrees::Cuda::check(cudaFree(dPoints));
    });
    KnnTrees::pinnedFree(queryPoints);
    KnnTrees::Cuda::check(cudaFree(dQueryPoints));


    return 0;
}
