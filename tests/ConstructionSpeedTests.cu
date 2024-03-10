#include <iostream>
#include "../src/KdTree.cuh"
#include "../src/BallTree.cuh"
#include "../src/KdTreeGpu.cuh"
#include "../src/BallTreeGpu.cuh"
#include "Tests.cuh"


int main(int argc, char const *argv[]) {
    
    cudaFree(0);
    constexpr uint dims = 3;
    constexpr uint pointCount = 500000;
    std::cout << "Construction time speed test" << std::endl;


    std::cout << "Test with changing point count and dimension " << dims << std::endl;
    constexpr uint PointCounts[] = {100000, 250000, 500000, 1000000, 2500000};
    forLoop<5>([&](auto i) {
        constexpr uint pointCount = PointCounts[i];
        KnnTrees::Array<float, dims>* points = KnnTrees::pinnedMalloc<KnnTrees::Array<float, dims>>(pointCount);
        generateRandomPoints<dims>(points, pointCount, 1.0);
        KnnTrees::Array<float, dims>* dPoints = gpuCopy(points, pointCount);
        float kdTime = 0.0f; float ballTime = 0.0f;
        float kdTimeCpu = 0.0f; float ballTimeCpu = 0.0f;
        for (uint j = 0; j < 5; j++) {
            // KdTree GPU
            startTime();
            KnnTrees::KdTreeGpu<dims, dims, 32> kdg(dPoints, pointCount, 3);
            kdTime += getTime();
            // BallTree GPU
            startTime();
            KnnTrees::BallTreeGpu<dims, 32> ballg(dPoints, pointCount, 1);
            ballTime += getTime();
            // KdTree
            startTime();
            KnnTrees::KdTree<dims> kd(points, pointCount, 3);
            kdTimeCpu += getTime();
            // BallTree
            startTime();
            KnnTrees::BallTree<dims> ball(points, pointCount, 1);
            ballTimeCpu += getTime();
        }
        std::cout << "KdTreeGpu - points: " << pointCount << ": " << (kdTime / 5) << "ms" << std::endl;
        std::cout << "BallTreeGpu - points: " << pointCount << ": " << (ballTime / 5) << "ms" << std::endl;
        std::cout << "KdTree - points: " << pointCount << ": " << (kdTimeCpu / 5) << "ms" << std::endl;
        std::cout << "BallTree - points: " << pointCount << ": " << (ballTimeCpu / 5) << "ms" << std::endl;
        KnnTrees::pinnedFree(points);
        KnnTrees::Cuda::check(cudaFree(dPoints));
    });


    constexpr uint Dimensionalities[] = {3, 5, 7, 10, 15, 20};
    forLoop<6>([&](auto i) {
        constexpr uint dims = Dimensionalities[i];
        KnnTrees::Array<float, dims>* points = KnnTrees::pinnedMalloc<KnnTrees::Array<float, dims>>(pointCount);
        generateRandomPoints<dims>(points, pointCount, 1.0);
        KnnTrees::Array<float, dims>* dPoints = gpuCopy(points, pointCount);
        float kdTime = 0.0f; float ballTime = 0.0f;
        float kdTimeCpu = 0.0f; float ballTimeCpu = 0.0f;
        for (uint j = 0; j < 6; j++) {
            // KdTree GPU
            startTime();
            KnnTrees::KdTreeGpu<dims, dims, 32> kdg(dPoints, pointCount, 3);
            if (j != 0) kdTime += getTime();
            // BallTree GPU
            startTime();
            KnnTrees::BallTreeGpu<dims, 32> ballg(dPoints, pointCount, 1);
            if (j != 0) ballTime += getTime();
            // KdTree
            startTime();
            KnnTrees::KdTree<dims> kd(points, pointCount, 3);
            if (j != 0) kdTimeCpu += getTime();
            // BallTree
            startTime();
            KnnTrees::BallTree<dims> ball(points, pointCount, 1);
            if (j != 0) ballTimeCpu += getTime();
        }
        std::cout << "KdTreeGpu - dims: " << dims << ": " << (kdTime / 5) << "ms" << std::endl;
        std::cout << "BallTreeGpu - dims: " << dims << ": " << (ballTime / 5) << "ms" << std::endl;
        std::cout << "KdTree - dims: " << dims << ": " << (kdTimeCpu / 5) << "ms" << std::endl;
        std::cout << "BallTree - dims: " << dims << ": " << (ballTimeCpu / 5) << "ms" << std::endl;
        KnnTrees::pinnedFree(points);
        KnnTrees::Cuda::check(cudaFree(dPoints));
    });


    return 0;
}
