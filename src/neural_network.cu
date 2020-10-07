#include "neural_network.cuh"

#include <initializer_list>
#include <vector>
#include <exception>
#include <random>
#include <algorithm>
#include <functional>
#include <time.h>

#include <cuda_runtime.h>
#include <cublas_v2.h>

#include "cuda_helpers/helper_cuda.h"

#include "utils.hpp"
#include "cuda_utils.cuh"

void print_memory_usage() {
	size_t free_byte, total_byte;
	cudaMemGetInfo(&free_byte, &total_byte);
	getLastCudaError("Error getting memory info");
	printf("Free memory (bytes): %u/%u\n", free_byte, total_byte);
}

const decltype(NeuralNetwork::layer_sizes_)& NeuralNetwork::layer_sizes() const {
	return this->layer_sizes_;
}

NeuralNetwork::NeuralNetwork(std::initializer_list<size_t> layer_sizes) :
		layer_sizes_(layer_sizes) {
	printf("Initializing ANN with layer sizes: ");
	print_vector(this->layer_sizes_);

	if (this->layer_sizes_.size() < 2) {
		throw std::invalid_argument("Must specify at least 2 layers!");
	}

	print_memory_usage();

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

	print_memory_usage();

	// allocate biases
	this->dev_biases = std::vector<float*>(this->layer_sizes_.size() - 1);
	for (auto i = 0; i < this->layer_sizes_.size() - 1; i++) {
		auto next_ls = this->layer_sizes_[i + 1];

		auto size = next_ls * sizeof(float);
		cudaMalloc((void**) &(this->dev_biases[i]), size);
		getLastCudaError(
				string_format("Error allocating biases %d", i).c_str());
	}

	print_memory_usage();

	// allocate activations
	this->dev_activations = std::vector<float*>(this->layer_sizes_.size());
	for (auto i = 0; i < this->layer_sizes_.size(); i++) {
		auto layer_size = this->layer_sizes_[i];

		auto size = layer_size * sizeof(float);
		cudaMalloc((void**) &(this->dev_activations[i]), size);
		getLastCudaError(
				string_format("Error allocating activation %d", i).c_str());
	}

	print_memory_usage();

	// allocate error vectors, skip the input layer
	this->dev_errors = std::vector<float*>(this->layer_sizes_.size() - 1);
	for (auto i = 0; i < this->layer_sizes_.size() - 1; i++) {
		auto layer_size = this->layer_sizes_[i + 1];

		auto size = layer_size * sizeof(float);
		cudaMalloc((void**) &(this->dev_errors[i]), size);
		getLastCudaError(string_format("Error allocating error %d", i).c_str());
	}

	print_memory_usage();

	// allocate context
	printf("Allocating cublas context\n");
	auto status = cublasCreate(&this->cublasHandle);
	checkCudaErrors(status);

	print_memory_usage();
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

	for (auto const &dev_ptr : this->dev_errors) {
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

//		printf(
//				"\nFilling layer %d; sizes: [%d, %d]; weights size: %d, biases size: %d\n",
//				i, l_prev, l_next, weights_size, biases_size);

// fill weights with (min .. max] floats
		auto weights = new std::vector<float>(weights_size);
		std::generate(weights->begin(), weights->end(), gen_f);
//		printf("Generated weights %d on host:\n", i);
//		print_vector(*weights);

		auto biases = new std::vector<float>(biases_size);
//		std::generate(biases->begin(), biases->end(), gen_f);
		std::fill(biases->begin(), biases->end(), 0);
//		printf("Generated biases %d on host:\n", i);
//		print_vector(*biases);

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

void NeuralNetwork::evaluate(const float *dev_input) {
	cublasStatus_t status;

	cudaMemcpy(this->dev_activations[0], dev_input,
			this->layer_sizes_[0] * sizeof(float), cudaMemcpyDeviceToDevice);
	getLastCudaError("Error copying input vector to the first activation");

	float alpha = 1, beta = 0;

	// propagate forward
	for (auto i = 0; i < this->layer_sizes_.size() - 1; i++) {
//		printf("\nIteration %d\n\n", i);

		auto l_prev = this->layer_sizes_[i];
		auto l_next = this->layer_sizes_[i + 1];

		auto layer_input = this->dev_activations[i];
		auto layer_output = this->dev_activations[i + 1];

//		printf("Prev layer - %d, next layer - %d\n\n", l_prev, l_next);

//		printf("Weights %d:\n", i);
//		print_cuda_matrix(this->dev_weights[i], l_next, l_prev);
//		printf("\n");

//		printf("Biases %d:\n", i);
//		print_cuda_matrix(this->dev_biases[i], l_next, 1);
//		printf("\n");

//		printf("Input %d:\n", i);
//		print_cuda_matrix(layer_input, l_prev, 1);
//		printf("\n");

		// y param in cublasSgemv is both an input (bias) and an output (activation)
		// hence, copy bias into the activation
//		cudaMemcpy(layer_output, this->dev_biases[i], l_next * sizeof(float),
//				cudaMemcpyDeviceToDevice);
//		getLastCudaError(
//				string_format("Copying bias %d to activation", i).c_str());

		status = cublasSgemv(this->cublasHandle, CUBLAS_OP_N, l_next, l_prev,
				&alpha, this->dev_weights[i], l_next, layer_input, 1, &beta,
				layer_output, 1);
		checkCudaErrors(status);

//		printf("Output %d after Sgemv:\n", i);
//		print_cuda_matrix(layer_output, l_next, 1);
//		printf("\n");

// apply sigmoid in-place to activation vector
		cudaInvokeMaxOccupancy(0, 0, l_next, sigmoid,
				(const float *) layer_output, layer_output, l_next);

//		printf("Output %d:\n", i);
//		print_cuda_matrix(layer_output, l_next, 1);
//		printf("\n");
	}

	// copy final activation to the output
//	auto last_layer_size = this->layer_sizes_.back();
//	auto last_activation = this->dev_activations.back();
//	cudaMemcpy(dev_output, last_activation, last_layer_size * sizeof(float),
//			cudaMemcpyDeviceToDevice);
//	getLastCudaError("Unable to copy the last ANN activation to the output");
}

void NeuralNetwork::predict(const float *dev_input, float *dev_output) {
	this->evaluate(dev_input);
	cudaMemcpy(dev_output, this->dev_activations.back(),
			this->layer_sizes_.back() * sizeof(float),
			cudaMemcpyDeviceToDevice);
	getLastCudaError("Error copying network output to dev_output");
}

void NeuralNetwork::train(const float* dev_x_train, const float* dev_y_train,
		float learning_rate, float *out_cost) {
	cublasStatus_t status;
	// needed to pass into cublasSgemm as a negative coefficient when updating weights
	learning_rate = -learning_rate;

	// see https://brilliant.org/wiki/backpropagation/
	// ^ The Backpropagation Algorithm paragraph

	int i = this->layer_sizes_.size() - 1 - 1;	// last index - 1

	// save layer outputs
	this->evaluate(dev_x_train);

	auto rnext = this->layer_sizes_[i + 1];
	auto rprev = this->layer_sizes_[i];
//	printf("NN train; rnext = %d, rprev = %d\n", rnext, rprev);
	auto next_activation = this->dev_activations[i + 1];
	auto prev_activation = this->dev_activations[i];

	// TODO biases

	// compute error for the output layer

	// write sigmoid derivative into error vector
	cudaInvokeMaxOccupancy(0, 0, rnext, sigmoid_derivative,
			(const float *) next_activation, this->dev_errors[i], rnext);

//	printf("Last layer sigmoid derivative:\n");
//	print_cuda_matrix(this->dev_errors[i], rnext, 1);

	// compute output delta and overwrite output layer activation
	// (which is no longer needed, as the sigmoid derivative is already computed)
	float alpha = -1;
	status = cublasSaxpy(this->cublasHandle, rnext, &alpha, dev_y_train, 1,
			next_activation, 1);
	checkCudaErrors(status);

	// write MSE error to the method output
	status = cublasSdot(this->cublasHandle, rnext, next_activation, 1,
			next_activation, 1, out_cost);
	checkCudaErrors(status);
	*out_cost /= 2.0 * rnext;

//	printf("MSE error = %f\n", *out_cost);
//	print_cuda_matrix(next_activation, rnext, 1);

	// compute output layer error
	cudaInvokeMaxOccupancy(0, 0, rnext, vhadamard,
			(const float *) next_activation, this->dev_errors[i], rnext);

//	printf("Last layer error:\n");
//	print_cuda_matrix(this->dev_errors[i], rnext, 1);

//	printf("Updating weights\n");
//	printf("Layers:\n");
//	print_vector(this->layer_sizes());
//	printf("Error i+1:\n");
//	print_cuda_matrix(this->dev_errors[i], rnext, 1);
//	printf("Activation i:\n");
//	print_cuda_matrix(prev_activation, 1, rprev);
//	printf("Current weights i\n");
//	print_cuda_matrix(this->dev_weights[i], rnext, rprev);

	// update weights
	float beta = 1;
	status = cublasSgemm(this->cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, rnext,
			rprev, 1, &learning_rate, this->dev_errors[i], rnext,
			prev_activation, 1, &beta, this->dev_weights[i], rnext);
	checkCudaErrors(status);

//	printf("Updated weights\n");
//	print_cuda_matrix(this->dev_weights[i], rnext, rprev);

	// compute errors for hidden layers, update weights
	for (--i; i >= 0; i--) {
		auto rnext2 = this->layer_sizes_[i + 2];
		rnext = this->layer_sizes_[i + 1];
		rprev = this->layer_sizes_[i];
		next_activation = this->dev_activations[i + 1];
		prev_activation = this->dev_activations[i];
//		printf("Backprop i = %d; rnext2 = %d, rnext = %d, rprev = %d\n", i,
//				rnext2, rnext, rprev);

		// compute sigmoid derivative and write it into the output vector
		cudaInvokeMaxOccupancy(0, 0, rnext, sigmoid_derivative,
				(const float *) next_activation, next_activation, rnext);

//		printf("Sigmoid derivative\n");
//		print_cuda_matrix(next_activation, rnext, 1);

//		printf("Error i + 1\n");
//		print_cuda_matrix(this->dev_errors[i + 1], 1, rnext2);

		// write error intermediate into the error vector
		alpha = 1;
		float intermediate_beta = 0;
		status = cublasSgemv(this->cublasHandle, CUBLAS_OP_T, rnext2, rnext,
				&alpha, this->dev_weights[i + 1], rnext2,
				this->dev_errors[i + 1], 1, &intermediate_beta,
				this->dev_errors[i], 1);
		checkCudaErrors(status);

//		printf("Error intermediate\n");
//		print_cuda_matrix(this->dev_errors[i], rnext, 1);

		// compute error
		cudaInvokeMaxOccupancy(0, 0, rnext, vhadamard,
				(const float *) next_activation, this->dev_errors[i], rnext);

//		printf("Error\n");
//		print_cuda_matrix(this->dev_errors[i], rnext, 1);

//		printf("Current weights i\n");
//		print_cuda_matrix(this->dev_weights[i], rnext, rprev);

		// update weights
		status = cublasSgemm(this->cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N,
				rnext, rprev, 1, &learning_rate, this->dev_errors[i], rnext,
				prev_activation, 1, &beta, this->dev_weights[i], rnext);
		checkCudaErrors(status);

//		printf("Updated weights\n");
//		print_cuda_matrix(this->dev_weights[i], rnext, rprev);
	}
}

