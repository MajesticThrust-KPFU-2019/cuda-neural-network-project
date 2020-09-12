#include "neural_network.cuh"

#include <initializer_list>
#include <vector>
#include <exception>

#include <cuda_runtime.h>
#include "cuda_helpers/helper_cuda.h"
#include <cublas_v2.h>

#include "utils.hpp"

//void printLastCublasError(cublasStatus_t status, const char *msg) {
//	if (status != CUBLAS_STATUS_SUCCESS)
//		printf(stderr, "%s\n%s\n", msg, _cudaGetErrorEnum(status));
//}

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
		getLastCudaError(
				string_format("Error allocating weights %d", w_i).c_str());
	}

	// allocate activation vectors
	this->dev_activations = std::vector<float*>(this->layer_sizes_.size() - 1);
	for (auto i = 1; i < this->layer_sizes_.size(); i++) {
		auto layer_size = this->layer_sizes_[i];

		// include trailing 1 for a bias
		auto size = (layer_size + 1) * sizeof(float);
		auto act_i = i - 1;
		cudaMalloc((void**) &(this->dev_activations[act_i]), size);
		getLastCudaError(
				string_format("Error allocating activation %d", act_i).c_str());

	}

	// allocate context
	auto status = cublasCreate(&this->cublasHandle);
	checkCudaErrors(status);
}

NeuralNetwork::~NeuralNetwork() {
	for (auto const &devPtr : this->dev_weights) {
		cudaFree(devPtr);
	}

	for (auto const &devPtr : this->dev_activations) {
		cudaFree(devPtr);
	}

	auto status = cublasDestroy(this->cublasHandle);
	checkCudaErrors(status);
}

__global__ void fill_array_randomly() {
	// TODO
}

void NeuralNetwork::init_random() {
	// fill weights in a random fashion
	for (auto i = 0; i < this->layer_sizes_.size(); i++) {
		auto l_prev = this->layer_sizes_[i];
		auto l_next = this->layer_sizes_[i + 1];
	}

}

//	void init_from_data() {
//		// TODO init from data how? what format?
//	}

/**
 * Forward propagation. Fills internal activation vectors.
 *
 * Expects the input to be a vector with the same length as the input layer.
 * Expects the output to have enough allocated space for the output vector.
 */
void NeuralNetwork::predict(float *devInput, float *devOutput) {
	cublasStatus_t status;

	float *layerInput = devInput;
	float *layerOutput = this->dev_activations[0];

	float alpha = 1, beta = 1;

	// propagate forward
	for (auto i = 0; i < this->layer_sizes_.size(); i++) {
		auto l_prev = this->layer_sizes_[i];
		auto l_next = this->layer_sizes_[i + 1];

		status = cublasSgemv(this->cublasHandle, CUBLAS_OP_N, l_next,
				l_prev + 1, &alpha, this->dev_weights[i], l_next, layerInput, 1,
				&beta, layerOutput, 1);
		checkCudaErrors(status);

		layerInput = layerOutput;
		layerOutput = this->dev_activations[i + 1];
	}

	// copy final activation to the output
	auto last_i = this->layer_sizes_.size() - 1;
	cudaMemcpy(devOutput, this->dev_activations[last_i], this->layer_sizes_[last_i], cudaMemcpyDeviceToDevice);
	getLastCudaError("Unable to copy the last ANN activation to the output");
}

// pass dev pointer(s) to the batch, and a dev pointer for output
void NeuralNetwork::train_batch() {
	//
}

