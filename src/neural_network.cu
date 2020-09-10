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

NeuralNetwork::NeuralNetwork(std::initializer_list<unsigned int> layer_sizes) :
		layer_sizes(layer_sizes) {
	if (this->layer_sizes.size() < 2) {
		throw std::invalid_argument("Must specify at least 2 layers!");
	}

	// allocate weights + biases
	this->dev_weights = std::vector<float*>();
	// TODO

	// allocate activation vectors
	this->dev_activations = std::vector<float*>();
	for (auto i = 1; i < this->layer_sizes.size(); i++) {
		auto layer_size = this->layer_sizes[i];
		float *devPtr;
		cudaMalloc((void**) &devPtr, layer_size * sizeof(float));
		printOnLastCudaError(
				string_format<int>("Error allocating activation %d", i));
	}
}

NeuralNetwork::~NeuralNetwork() {
	for (auto const &devPtr : this->dev_weights) {
		cudaFree(devPtr);
	}
//		delete this->dev_weights;

	for (auto const &devPtr : this->dev_activations) {
		cudaFree(devPtr);
	}
//		delete this->dev_activations;
}

void NeuralNetwork::init_random() {
	// fill weights in a random fashion
}

//	void init_from_data() {
//		// TODO init from data how? what format?
//	}

void NeuralNetwork::predict(/* input vector - what type? *//* output vector - what type? */) {
	// forward propagation
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

