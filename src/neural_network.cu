#include "neural_network.cuh"

#include <initializer_list>
#include <vector>
#include <exception>

#include <cuda_runtime.h>
#include "cuda_helpers/helper_cuda.h"

#include "utils.hpp"

void printOnLastCudaError(const char *msg) {
	auto err = cudaGetLastError();
	if (err != cudaSuccess)
		fprintf(stderr, "%s\n%s\n", msg, cudaGetErrorString(err));
}

void printOnLastCudaError(const std::string msg) {
	printOnLastCudaError(msg.c_str());
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

NeuralNetwork::NeuralNetwork(std::initializer_list<unsigned int> layer_sizes) :
		layer_sizes_(layer_sizes) {
	if (this->layer_sizes_.size() < 2) {
		throw std::invalid_argument("Must specify at least 2 layers!");
	}

	// allocate weights + biases
	this->dev_weights = std::vector<float*>(this->layer_sizes_.size() - 1);
	for (auto i = 1; i < this->layer_sizes_.size(); i++) {
		auto prev_ls = this->layer_sizes_[i - 1];
		auto next_ls = this->layer_sizes_[i];
		// include biases in weights
		auto size = next_ls * (prev_ls + 1) + sizeof(float);
		auto w_i = i - 1;
		cudaMalloc((void**) &(this->dev_weights[w_i]), size);
		printOnLastCudaError(string_format("Error allocating weights %d", w_i));
	}

	// allocate activation vectors
	this->dev_activations = std::vector<float*>(this->layer_sizes_.size() - 1);
	for (auto i = 1; i < this->layer_sizes_.size(); i++) {
		auto layer_size = this->layer_sizes_[i];
		auto act_i = i - 1;

		cudaMalloc((void**) &(this->dev_activations[act_i]),
				layer_size * sizeof(float));
		printOnLastCudaError(
				string_format("Error allocating activation %d", act_i));

	}
}

NeuralNetwork::~NeuralNetwork() {
	for (auto const &devPtr : this->dev_weights) {
		cudaFree(devPtr);
	}

	for (auto const &devPtr : this->dev_activations) {
		cudaFree(devPtr);
	}
}

void NeuralNetwork::init_random() {
	// fill weights in a random fashion
}

//	void init_from_data() {
//		// TODO init from data how? what format?
//	}

/**
 * Forward propagation.
 *
 * Expects the input to be a vector with the same length as the input layer.
 * Expects the output to have enough allocated space for the output vector.
 */
void NeuralNetwork::predict(float *&devInput, float *&devOutput) {
	// the predict should copy vector to the device and copy the result from device to host
	// because the result is intended to be consumed from the outside
}

// or maybe just `train`? how should the dataset be loaded into memory?
// it would make sense to pass device pointers there
// this ideally should be a coroutine, but those are not yet supported
// so synchronous code it is
void NeuralNetwork::train_batch() {
	//
}

