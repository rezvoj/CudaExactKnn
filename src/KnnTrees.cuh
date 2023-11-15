#pragma once
#include <stdexcept>


namespace KnnTrees {

    template <typename Type, uint Size>
    struct Array {
        Type arr[Size];
        __forceinline__ __host__ __device__
        Type& operator[](uint index) {
            return arr[index];
        }
        __forceinline__ __host__ __device__
        const Type& operator[](uint index) const {
            return arr[index];
        }
    };

    class CudaException : public std::runtime_error {
        public: CudaException(const cudaError_t error): 
            std::runtime_error(cudaGetErrorString(error)) {}
    };

    template <uint Dims>
    struct IndexPoint {
        int index;
        KnnTrees::Array<float, Dims> point;
    };

    template <uint Dims>
    struct BallNode {
        KnnTrees::Array<float, Dims> leftCentroid;
        float leftRadius;
        KnnTrees::Array<float, Dims> rightCentroid;
        float rightRadius;
    };

    template <typename Type>
    __forceinline__ __host__ __device__
    void swap(Type& first, Type& second) {
        Type temporary = first;
        first = second;
        second = temporary;
    }

    template <typename Type, uint Size>
    __forceinline__ __host__ __device__
    void fillArray(Type* array, const Type& value) {
        #pragma unroll
        for (uint idx = 0; idx < Size; ++idx) {
            array[idx] = value;
        }
    }

    template <typename Type, uint Size>
    __forceinline__ __host__ __device__
    void copyArray(const Type* from, Type* to) {
        #pragma unroll
        for (uint idx = 0; idx < Size; ++idx) {
            to[idx] = from[idx];
        }
    }

    template <uint Dims>
    __forceinline__ __host__ __device__
    float calcDistance2(
            const KnnTrees::Array<float, Dims>& p1,
            const KnnTrees::Array<float, Dims>& p2) {
        float result = 0;
        #pragma unroll
        for (uint dimension = 0; dimension < Dims; ++dimension) {
            const float difference = p1[dimension] - p2[dimension];
            result += difference * difference;
        }
        return result;
    }

    __forceinline__ __host__ __device__
    uint pow2(const uint exponent) {
        return 1U << exponent;
    }

    __forceinline__ __host__ __device__
    uint log2Ceil(const uint input) {
        uint comparator = 2;
        uint result = 1;
        while (comparator < input) {
            comparator *= 2;
            result += 1;
        }
        return result;
    }

    __forceinline__ __host__ __device__
    uint leftChildOf(const uint idx) {
        return 2 * idx + 1;
    }

    __forceinline__ __host__ __device__
    uint parentOf(const uint idx) {
        return (idx - 1) / 2;
    }

    __forceinline__ __host__ __device__
    uint64_t encodeUintUFloat(const uint integer, const float unsignedFloat) {
        union {float input; uint output;} floatAsUint;
        floatAsUint.input = unsignedFloat;
        return (static_cast<uint64_t>(integer) << 32) | static_cast<uint64_t>(floatAsUint.output);
    }

    __forceinline__ __host__ __device__
    uint64_t encodeUintSFloat(const uint integer, const float signedFloat) {
        union {float input; int output;} floatAsUint;
        floatAsUint.input = signedFloat;
        const uint mask = static_cast<uint32_t>(floatAsUint.output >> 31);
        floatAsUint.output = (floatAsUint.output ^ mask) | ((!mask) << 31);
        return (static_cast<uint64_t>(integer) << 32)
            | static_cast<uint64_t>(static_cast<uint32_t>(floatAsUint.output));
    }

    __forceinline__ __host__ __device__
    uint64_t encodeUFloatInt(const float unsignedFloat, const int integer) {
        union {float input; uint output;} floatAsUint;
        floatAsUint.input = unsignedFloat;
        return (static_cast<uint64_t>(static_cast<uint32_t>(floatAsUint.output)) << 32)
            | static_cast<uint64_t>(static_cast<uint32_t>(integer));
    }

    __forceinline__ __host__ __device__
    float decodeEncoded1UFloat(const uint64_t value) {
        const uint floatBits = static_cast<uint>(value >> 32);
        return *reinterpret_cast<const float*>(&floatBits);
    }

};
