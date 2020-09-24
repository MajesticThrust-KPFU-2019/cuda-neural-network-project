/*
 * cuda_utils.cu
 *
 *  Created on: Sep 24, 2020
 *      Author: timur
 */

#include "utils.hpp"

#include "cuda_helpers/helper_cuda.h"

inline unsigned int IDX2C(unsigned int i, unsigned int j, unsigned int ld) {
	return (((j) * (ld)) + (i));
}

void print_cuda_matrix(const float* dev_ptr, size_t nrows, size_t ncols) {
	auto h_matrix = new float[nrows * ncols];

	cudaMemcpy(h_matrix, dev_ptr, nrows * ncols * sizeof(float),
			cudaMemcpyDeviceToHost);
	getLastCudaError(
			string_format("Unable to copy data from device (%d) to host",
					dev_ptr).c_str());

	for (auto i = 0; i < nrows; i++) {
		for (auto j = 0; j < ncols; j++) {
			auto idx = IDX2C(i, j, nrows);
			printf("%.6f\t", h_matrix[idx]);
		}
		printf("\n");
	}

	delete[] h_matrix;
}

__global__ void sigmoid(const float *x, float *y, size_t N) {
	int tidx = threadIdx.x + blockIdx.x * blockDim.x;
	int stride = blockDim.x * gridDim.x;

	for (; tidx < N; tidx += stride) {
		y[tidx] = 1.0 / (1 + expf(-x[tidx]));
	}
}

__global__ void sigmoid_derivative(const float *sig_x, float *y, size_t N) {
	int tidx = threadIdx.x + blockIdx.x * blockDim.x;
	int stride = blockDim.x * gridDim.x;

	for (; tidx < N; tidx += stride) {
		y[tidx] = sig_x[tidx] * (1 - sig_x[tidx]);
	}
}

__global__ void vhadamard(const float *x, float *y, size_t N) {
	int tidx = threadIdx.x + blockIdx.x * blockDim.x;
	int stride = blockDim.x * gridDim.x;

	for (; tidx < N; tidx += stride) {
		y[tidx] = y[tidx] * x[tidx];
	}
}

__global__ void axpb(const float a, const float *x, const float b, float *y,
		size_t N) {
	int tidx = threadIdx.x + blockIdx.x * blockDim.x;
	int stride = blockDim.x * gridDim.x;

	for (; tidx < N; tidx += stride) {
		y[tidx] = a * x[tidx] + b;
	}
}
