# CUDA Exact KNN

![License](https://img.shields.io/badge/license-MIT-blue.svg) ![CUDA Version](https://img.shields.io/badge/CUDA-11.0-green)

## Table of Contents

- [Introduction](#introduction)
- [Requirements](#requirements)
- [Usage](#usage)
- [Testing](#testing)
- [License](#license)

## Introduction

This project is a GPU-accelerated library for performing fast and exact k-nearest neighbors (KNN) batch queries on large datasets. Leveraging CUDA for GPU acceleration, the library speeds up both batch queries and data structure construction. It supports multiple KNN algorithms: Brute Force, K-D Tree and Ball Tree, providing flexibility for different use cases and offers performance configurations such as heap-based nearest neighbor tracking and shared memory optimizations. Testing suite is included for performance and correctness verification.

Development on this project is on hold until I have access to a PC with a CUDA-enabled GPU. Then I will work on providing a better API, a proper build system, more sophisticated tests, and possibly adding more algorithms and Python bindings.

## Requirements

- CUDA enabled GPU.
- CUDA 11 toolkit and runtime or higher (or lower perhaps).
- Linux machine since the current setup uses Makefile.

## Usage

### Construction of the Simple Data Structures

##### Brute Force:

```cpp
    KnnTrees::Array<float, DIMENSIONS>* points = ...;
    KnnTrees::BruteForce<DIMENSIONS> bruteForce(points, pointCount);
```

##### K-D Tree:

```cpp
    KnnTrees::Array<float, DIMENSIONS>* points = ...;
    KnnTrees::KdTree<DIMENSIONS> kdTree(points, pointCount, depthSkips);
```

##### Ball Tree:

```cpp
    KnnTrees::Array<float, DIMENSIONS>* points = ...;
    KnnTrees::BallTree<DIMENSIONS> ballTree(points, pointCount, depthSkips);
```

### Construction of the GPU-Accelerated Data Structures

##### Brute Force:

```cpp
    KnnTrees::Array<float, DIMENSIONS>* devicePoints = ...;
    KnnTrees::BruteForceGpu<DIMENSIONS> bruteForceGpu(devicePoints, pointCount);
```

##### K-D Tree:

```cpp
    KnnTrees::Array<float, DIMENSIONS>* devicePoints = ...;
    KnnTrees::KdTreeGpu<DIMENSIONS, SPLIT_DIMENSIONS, BLOCK_SIZE> kdTreeGpu(devicePoints, pointCount, depthSkips);
```

##### Ball Tree:

```cpp
    KnnTrees::Array<float, DIMENSIONS>* devicePoints = ...;
    KnnTrees::BallTreeGpu<DIMENSIONS, BLOCK_SIZE> ballTreeGpu(devicePoints, pointCount, depthSkips);
```

### Batch Query on Simple Data Structures

```cpp
    KnnTrees::Array<float, DIMENSIONS>* queryPoints = ...;
    KnnTrees::Array<int, K>* resultIndexes = ...;
    KnnTrees::Array<float, K>* resultDistances = ...;
    KnnTrees::Query<DIMENSIONS, K> query = {
        queryPoints, queryPointCount, maxRadius, 
        resultIndexes, resultDistances
    };
    simpleDS.batchKnn<K, USE_HEAP_TRACKING>(query);
```

### Batch Query on GPU-Accelerated Data Structures

##### Brute Force:

```cpp
    KnnTrees::Array<float, DIMENSIONS>* deviceQueryPoints = ...;
    KnnTrees::Array<int, K>* deviceResultIndexes = ...;
    KnnTrees::Array<float, K>* deviceResultDistances = ...;
    KnnTrees::Query<DIMENSIONS, K> deviceQuery = {
        deviceQueryPoints, queryPointCount, maxRadius, 
        deviceResultIndexes, deviceResultDistances
    };
    bruteForce.batchKnn<K, BLOCK_SIZE, USE_HEAP_TRACKING, USE_TRACKING_SHMEM>(deviceQuery);
```

##### K-D Tree and Ball Tree:

```cpp
    KnnTrees::Array<float, DIMENSIONS>* deviceQueryPoints = ...;
    KnnTrees::Array<int, K>* deviceResultIndexes = ...;
    KnnTrees::Array<float, K>* deviceResultDistances = ...;
    KnnTrees::Query<DIMENSIONS, K> deviceQuery = {
        deviceQueryPoints, queryPointCount, maxRadius, 
        deviceResultIndexes, deviceResultDistances
    };
    tree.batchKnn<K, BLOCK_SIZE, USE_HEAP_TRACKING, USE_TRACKING_SHMEM, USE_STACK_SHMEM>(deviceQuery);
```

## Testing

The project includes tests to ensure functionality and benchmark performance. Since certain test cases use high amount of shared memory, they might not run on older cards. Each test focuses on a different aspect:

- **FunctionalityTests**: Ensures correctness of the KNN implementations.
- **SpeedupSpeedTests**: Compares the performance of GPU-accelerated algorithms against their CPU counterparts.
- **ConstructionSpeedTests**: Benchmarks the time required to build various data structures.
- **Dim3SpeedTests**: Detailed tests for the the performance in 3-dimensional space.
- **DimXSpeedTests**: Evaluates performance across varying dimensionalities.

Makefile to compile the test binaries is included, you might need to change the GPU architecture. You can compile the tests using the following commands:

```sh
make FunctionalityTests
make SpeedupSpeedTests
make ConstructionSpeedTests
make Dim3SpeedTests
make DimXSpeedTests
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
