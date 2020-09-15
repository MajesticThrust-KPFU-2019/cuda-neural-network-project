#include "neural_network.cuh"

#include <initializer_list>
#include <vector>
#include <exception>
#include <random>     // mt19937 and uniform_int_distribution
#include <algorithm>  // generate
#include <functional> // bind
#include <time.h>

#include <cuda_runtime.h>
#include <cublas_v2.h>

#include <curand.h>
#include <curand_kernel.h>

#include "cuda_helpers/helper_cuda.h"

#include "utils.hpp"
#include <iostream>

// TODO remove
void print_vector2(const std::vector<float>& v) {
	for (auto o : v) {
		std::cout << o << " ";
	}
	std::cout << std::endl;
}

/**
 * Convert 2-dim index to 1-dim. Used to index matrices stored as 1-dim arrays.
 * @param i
 * @param j
 * @param ld - leading dimension; number of rows for column-major storage
 */
inline unsigned int IDX2C(unsigned int i, unsigned int j, unsigned int ld) {
	return (((j) * (ld)) + (i));
}

const decltype(NeuralNetwork::layer_sizes_)& NeuralNetwork::layer_sizes() const {
	return this->layer_sizes_;
}

void print_cuda_matrix(const float* dev_ptr, size_t nrows, size_t ncols) {
	auto h_matrix = new float[nrows * ncols];

	cudaMemcpy(h_matrix, dev_ptr, nrows * ncols * sizeof(float),
			cudaMemcpyDeviceToHost);
	getLastCudaError("Unable to copy data from device to host");

	for (auto i = 0; i < nrows; i++) {
		for (auto j = 0; j < ncols; j++) {
			auto idx = IDX2C(i, j, nrows);
			printf("%.6f\t", h_matrix[idx]);
		}
		printf("\n");
	}

	delete[] h_matrix;
}

NeuralNetwork::NeuralNetwork(std::initializer_list<unsigned int> layer_sizes) :
		layer_sizes_(layer_sizes) {
	if (this->layer_sizes_.size() < 2) {
		throw std::invalid_argument("Must specify at least 2 layers!");
	}

	// allocate weights
	this->dev_weights = std::vector<float*>(this->layer_sizes_.size() - 1);
	for (auto i = 0; i < this->layer_sizes_.size() - 1; i++) {
		auto prev_ls = this->layer_sizes_[i];
		auto next_ls = this->layer_sizes_[i + 1];

		auto size = next_ls * prev_ls * sizeof(float);
		cudaMalloc((void**) &(this->dev_weights[i]), size);
		getLastCudaError(
				string_format("Error allocating weights %d", i).c_str());
	}

	// allocate biases
	this->dev_biases = std::vector<float*>(this->layer_sizes_.size() - 1);
	for (auto i = 0; i < this->layer_sizes_.size() - 1; i++) {
		auto next_ls = this->layer_sizes_[i + 1];

		auto size = next_ls * sizeof(float);
		cudaMalloc((void**) &(this->dev_biases[i]), size);
		getLastCudaError(
				string_format("Error allocating biases %d", i).c_str());
	}

	// allocate activation vectors
	// skip the input layer
	this->dev_activations = std::vector<float*>(this->layer_sizes_.size() - 1);
	for (auto i = 0; i < this->layer_sizes_.size() - 1; i++) {
		auto layer_size = this->layer_sizes_[i + 1];

		// include trailing 1 for a bias
		auto size = layer_size * sizeof(float);
		cudaMalloc((void**) &(this->dev_activations[i]), size);
		getLastCudaError(
				string_format("Error allocating activation %d", i).c_str());

		// fill with 1's
		cudaMemset(this->dev_activations[i], 1, size);
		getLastCudaError(
				string_format("Error filling activation %d with 1's", i).c_str());
	}

	// allocate context
	auto status = cublasCreate(&this->cublasHandle);
	checkCudaErrors(status);
}

NeuralNetwork::~NeuralNetwork() {
	for (auto const &dev_ptr : this->dev_weights) {
		cudaFree(dev_ptr);
	}

	for (auto const &dev_ptr : this->dev_biases) {
		cudaFree(dev_ptr);
	}

	for (auto const &dev_ptr : this->dev_activations) {
		cudaFree(dev_ptr);
	}

	auto status = cublasDestroy(this->cublasHandle);
	checkCudaErrors(status);
}

void NeuralNetwork::init_random(float min, float max) {
//	std::random_device r;
//	std::mt19937 eng(r()); // a source of random data
	std::mt19937 eng(time(nullptr));

	std::uniform_real_distribution<float> dist(min, max);
	auto gen_f = bind(dist, eng);

	// fill weights in a random fashion
	for (auto i = 0; i < this->layer_sizes_.size() - 1; i++) {
		auto l_prev = this->layer_sizes_[i];
		auto l_next = this->layer_sizes_[i + 1];

		auto weights_size = l_next * l_prev * sizeof(float);
		auto biases_size = l_next * sizeof(float);

		printf(
				"\nFilling layer %d; sizes: [%d, %d]; weights size: %d, biases size: %d\n",
				i, l_prev, l_next, weights_size, biases_size);

		// fill weights with (min .. max] floats
		auto weights = new std::vector<float>(weights_size);
		std::generate(weights->begin(), weights->end(), gen_f);
		printf("Generated weights %d on host:\n", i);
		print_vector2(*weights);

		auto biases = new std::vector<float>(biases_size);
		std::generate(biases->begin(), biases->end(), gen_f);
		printf("Generated biases %d on host:\n", i);
		print_vector2(*biases);

		// copy to device
		cudaMemcpy(this->dev_weights[i], weights->data(), weights_size,
				cudaMemcpyHostToDevice);
		getLastCudaError(
				string_format("Copy random weights %d to device", i).c_str());

		cudaMemcpy(this->dev_biases[i], biases->data(), biases_size,
				cudaMemcpyHostToDevice);
		getLastCudaError(
				string_format("Copy random biases %d to device", i).c_str());

		delete weights;
		delete biases;
	}
}

//	void init_from_data() {
//		// TODO init from data how? what format?
//	}

/**
 * Sigmoid function.
 * @param x - input vector allocated on device
 * @param y - output vector allocated on device
 * @param N - length of each vector
 */
__global__ void sigmoid(const float *x, float *y, unsigned int N) {
	int tidx = threadIdx.x + blockIdx.x * blockDim.x;
	int stride = blockDim.x * gridDim.x;

	for (; tidx < N; tidx += stride) {
		y[tidx] = 1.0 / (1 + expf(-x[tidx]));
	}
}

/**
 * Forward propagation. Fills internal activation vectors.
 *
 * Expects the input to be a vector with the same length as the input layer.
 * Expects the output to have enough allocated space for the output vector.
 */
void NeuralNetwork::predict(float *dev_input, float *dev_output) {
	cublasStatus_t status;

	float *layer_input = dev_input;

	float alpha = 1, beta = 1;

	// propagate forward
	for (auto i = 0; i < this->layer_sizes_.size() - 1; i++) {
		printf("\nIteration %d\n\n", i);

		auto l_prev = this->layer_sizes_[i];
		auto l_next = this->layer_sizes_[i + 1];

		printf("Prev layer - %d, next layer - %d\n\n", l_prev, l_next);

		printf("Weights %d:\n", i);
		print_cuda_matrix(this->dev_weights[i], l_next, l_prev);
		printf("\n");

		printf("Biases %d:\n", i);
		print_cuda_matrix(this->dev_biases[i], l_next, 1);
		printf("\n");

		printf("Input %d:\n", i);
		print_cuda_matrix(layer_input, l_prev, 1);
		printf("\n");

		auto layer_output = this->dev_activations[i];

		// y param in cublasSgemv is both an input (bias) and an output (activation)
		// hence, copy bias into the activation
		cudaMemcpy(layer_output, this->dev_biases[i], l_next * sizeof(float),
				cudaMemcpyDeviceToDevice);
		getLastCudaError(
				string_format("Copying bias %d to activation", i).c_str());

		status = cublasSgemv(this->cublasHandle, CUBLAS_OP_N, l_next, l_prev,
				&alpha, this->dev_weights[i], l_next, layer_input, 1, &beta,
				layer_output, 1);
		checkCudaErrors(status);

		printf("Output %d after Sgemv:\n", i);
		print_cuda_matrix(layer_output, l_next, 1);
		printf("\n");

		// The launch configurator returned block size
		int blockSize;
		// The minimum grid size needed to achieve the
		// maximum occupancy for a full device launch
		int minGridSize;

		cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, sigmoid, 0,
				0);
		// The actual grid size needed, based on input size
		// Round up according to array size
		auto gridSize = (l_next + blockSize - 1) / blockSize;

		sigmoid<<< gridSize, blockSize >>>(layer_output, layer_output, l_next);

		printf("Output %d:\n", i);
		print_cuda_matrix(layer_output, l_next, 1);
		printf("\n");

		layer_input = layer_output;
	}

	// copy final activation to the output
	auto last_layer_size = this->layer_sizes_.back();
	auto last_activation = this->dev_activations.back();
	cudaMemcpy(dev_output, last_activation, last_layer_size * sizeof(float),
			cudaMemcpyDeviceToDevice);
	getLastCudaError("Unable to copy the last ANN activation to the output");
}

// pass dev pointer(s) to the batch, and a dev pointer for output
void NeuralNetwork::train_batch() {
	//
}

