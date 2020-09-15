//============================================================================
// Name        : cuda-neural-network-project.cpp
// Author      : Timur Mukhametulin
// Version     :
// Copyright   : 
// Description : Hello World in C++, Ansi-style
//============================================================================

#include <iostream>
#include <vector>
#include <algorithm>
#include "mnist/mnist_reader.hpp"

#include <cuda_runtime.h>
#include "cuda_helpers/helper_cuda.h"

#include "neural_network.cuh"

int main() {
	auto dataset = mnist::read_dataset<std::vector, std::vector, uint8_t,
			uint8_t>();
	std::cout << "Hello mnist" << std::endl; // prints
	std::cout << "Nbr of training images = " << dataset.training_images.size()
			<< std::endl;
	std::cout << "Nbr of training labels = " << dataset.training_labels.size()
			<< std::endl;
	std::cout << "Nbr of test images = " << dataset.test_images.size()
			<< std::endl;
	std::cout << "Nbr of test labels = " << dataset.test_labels.size()
			<< std::endl;

	const auto input_size = 28 * 28;
	const auto output_size = 10;

	auto ann = NeuralNetwork { input_size, 300, output_size };
	ann.init_random();

	auto input_vector = std::vector<float>(input_size);
	std::fill(input_vector.begin(), input_vector.end(), 1);

	float *dev_input;
	cudaMalloc((void**) &dev_input, input_size * sizeof(float));
	getLastCudaError("Allocate dev input vector");

	float *dev_output;
	cudaMalloc((void**) &dev_output, output_size * sizeof(float));
	getLastCudaError("Allocate dev output vector");

	cudaMemcpy(dev_input, input_vector.data(), input_size * sizeof(float),
			cudaMemcpyHostToDevice);
	getLastCudaError("Copy input vector to device");

	ann.predict(dev_input, dev_output);

	auto output_vector = std::vector<float>(output_size);
	cudaMemcpy(output_vector.data(), dev_output, output_size,
			cudaMemcpyDeviceToHost);

	for (auto o : output_vector) {
		std::cout << o << " ";
	}
	std::cout << std::endl;

	return 0;
}
