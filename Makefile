NVCC_FLAGS = -std=c++17 --gpu-architecture=sm_86 -O3 -use_fast_math

FunctionalityTests:
	nvcc $(NVCC_FLAGS) -o FunctionalityTests tests/FunctionalityTests.cu

SpeedupSpeedTests:
	nvcc $(NVCC_FLAGS) -o SpeedupSpeedTests tests/SpeedupSpeedTests.cu
