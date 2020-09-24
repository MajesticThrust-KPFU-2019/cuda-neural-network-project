/*
 * cuda_utils.cuh
 *
 *  Created on: Sep 24, 2020
 *      Author: timur
 */

#ifndef CUDA_UTILS_CUH_
#define CUDA_UTILS_CUH_

/**
 * Invoke a __global__ function with max occupancy.
 * @param dynamicSMemSize - Per-block dynamic shared memory usage intended, in bytes (see cudaOccupancyMaxPotentialBlockSize)
 * @param blockSizeLimit - The maximum block size func is designed to work with. 0 means no limit ((see cudaOccupancyMaxPotentialBlockSize)
 * @param inputSize - number of elements that can be processed in parallel by a single thread each
 * @param func - the __global__ function in question
 * @param funcArgs - func arguments as a parameter_pack
 */
template<class ...Args>
void cudaInvokeMaxOccupancy(size_t dynamicSMemSize, int blockSizeLimit,
		int inputSize, void (*func)(Args...), Args ... funcArgs) {
	// see https://developer.nvidia.com/blog/cuda-pro-tip-occupancy-api-simplifies-launch-configuration/
	// The launch configurator returned block size
	int blockSize;
	// The minimum grid size needed to achieve the
	// maximum occupancy for a full device launch
	int minGridSize;

	cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, func,
			dynamicSMemSize, blockSizeLimit);

	// The actual grid size needed, based on input size
	// Round up according to array size
	auto gridSize = (inputSize + blockSize - 1) / blockSize;

//	printf("Launching kernel with gridSize=%d, blockSize=%d\n", gridSize,
//			blockSize);

	func<<< gridSize, blockSize >>>(funcArgs...);
}

/**
 * Convert 2-dim index to 1-dim. Used to index matrices stored as 1-dim arrays.
 * @param i
 * @param j
 * @param ld - leading dimension; number of rows for column-major storage
 */
inline unsigned int IDX2C(size_t i, size_t j, size_t ld);

void print_cuda_matrix(const float* dev_ptr, size_t nrows, size_t ncols);

/**
 * Vectorized sigmoid function.
 * @param x - input vector allocated on device
 * @param y - output vector allocated on device
 * @param N - length of each vector
 */
__global__ void sigmoid(const float *x, float *y, size_t N);

/**
 * Vectorized sigmoid derivative function.
 * @param x - input vector allocated on device with precomputed sigmoid values
 * @param y - output vector allocated on device
 * @param N - length of each vector
 */
__global__ void sigmoid_derivative(const float *sig_x, float *y, size_t N);

/**
 * Hadamard product for vectors. y is both an input and an output.
 */
__global__ void vhadamard(const float *x, float *y, size_t N);

/**
 * Performs y = a*x + b, where
 * a - scalar,
 * x - vector with length N, input
 * b - scalar,
 * y - vector with length N, output,
 * N - vector size
 */
__global__ void axpb(const float a, const float *x, const float b, float *y, size_t N);

#endif /* CUDA_UTILS_CUH_ */
