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

void print_vector(const std::vector<float>& v) {
	for (auto o : v) {
		std::cout << o << " ";
	}
	std::cout << std::endl;
}

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

//	const auto input_size = 28 * 28;
//	const auto output_size = 10;

	const auto input_size = 4;
	const auto output_size = 2;

//	auto ann = NeuralNetwork { input_size, 300, output_size };
	auto ann = NeuralNetwork { input_size, 3, output_size };
	ann.init_random();

//	auto input_vector = std::vector<float>(input_size);
//	std::fill(input_vector.begin(), input_vector.end(), 0.5);
	auto input_vector = std::vector<float> { 0.4, 0.6, 0.2, 0.9 };
	printf("Input vector\n");
	print_vector(input_vector);

	auto expected_output_vector = std::vector<float>(output_size);
	std::fill(expected_output_vector.begin(), expected_output_vector.end(), 1);
	printf("Expected output vector\n");
	print_vector(expected_output_vector);

	float *dev_input;
	cudaMalloc((void**) &dev_input, input_size * sizeof(float));
	getLastCudaError("Allocate dev input vector");

	float *dev_output;
	cudaMalloc((void**) &dev_output, output_size * sizeof(float));
	getLastCudaError("Allocate dev output vector");

	cudaMemcpy(dev_input, input_vector.data(), input_size * sizeof(float),
			cudaMemcpyHostToDevice);
	getLastCudaError("Copy input vector to device");

	cudaMemcpy(dev_output, expected_output_vector.data(),
			output_size * sizeof(float), cudaMemcpyHostToDevice);
	getLastCudaError("Copy output vector to device");

	float learning_rate = 0.05;
	float error = 0;

//	ann.predict(dev_input, dev_output);

	auto output_vector = std::vector<float>(output_size);

	// iter 1
	ann.train(dev_input, dev_output, learning_rate, &error);
	cudaMemcpy(output_vector.data(), dev_output, output_size * sizeof(float),
			cudaMemcpyDeviceToHost);

	printf("Error: %f\n", error);
	print_vector(output_vector);

	// iter 2
	ann.train(dev_input, dev_output, learning_rate, &error);
	cudaMemcpy(output_vector.data(), dev_output, output_size * sizeof(float),
			cudaMemcpyDeviceToHost);

	printf("Error: %f\n", error);
	print_vector(output_vector);

	// iter 3
	ann.train(dev_input, dev_output, learning_rate, &error);
	cudaMemcpy(output_vector.data(), dev_output, output_size * sizeof(float),
			cudaMemcpyDeviceToHost);

	printf("Error: %f\n", error);
	print_vector(output_vector);

	return 0;
}
