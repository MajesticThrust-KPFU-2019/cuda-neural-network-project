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
//#include "mnist/mnist_reader.hpp"

#include <cuda_runtime.h>
#include <cublas_v2.h>
// include after cuda libs to make checkCudaErrors work with their statuses
#include "cuda_helpers/helper_cuda.h"

#include "neural_network.cuh"
#include "device_dataset.cuh"
#include "genetic-algorithm.cuh"

#include "utils.hpp"

int main_backprop_test() {
//	auto dataset = mnist::read_dataset<std::vector, std::vector, uint8_t,
//			uint8_t>();

	const auto input_size = 4;
	const auto output_size = 2;

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

int main_backprop() {
	auto train_dataset = MnistDeviceDatasetProvider(true);
	auto test_dataset = MnistDeviceDatasetProvider(false);
	auto ann = NeuralNetwork { train_dataset.x_size, 300, train_dataset.y_size };
	ann.init_random();

	const auto epoch_count = 10;
	const float learning_rate = 0.01;

	float* dev_input = 0;
	float* dev_expected_output = 0;
	float error;

	float* dev_actual_output;
	cudaMalloc((void**) &dev_actual_output,
			ann.layer_sizes().back() * sizeof(float));
	getLastCudaError(
			"Error allocating device memory for actual nn output vector");

	cublasHandle_t cuhand;
	auto status = cublasCreate(&cuhand);
	checkCudaErrors(status);

	for (auto epoch_i = 0; epoch_i < epoch_count; epoch_i++) {
		// train using the whole training dataset
		printf("Epoch %d\n", epoch_i);
		train_dataset.init_epoch();

		for (auto point_i = 0; point_i < train_dataset.dataset_size;
				point_i++) {
			train_dataset.get_single_pair(&dev_input, &dev_expected_output);
			ann.train(dev_input, dev_expected_output, learning_rate, &error);

			if (point_i % 1000 == 0) {
				printf("%d: cost = %f\n", point_i, error);
			}
		}

		// evaluate using the whole test dataset
		printf("\nLast cost: %f; evaluating...\n", error);
		test_dataset.init_epoch();

		size_t correct = 0;
		for (auto point_i = 0; point_i < test_dataset.dataset_size; point_i++) {
			test_dataset.get_single_pair(&dev_input, &dev_expected_output);

			ann.predict(dev_input, dev_actual_output);

			int expectedAmax = 0, actualAmax = 0;
			status = cublasIsamax(cuhand, ann.layer_sizes().back(),
					dev_expected_output, 1, &expectedAmax);
			checkCudaErrors(status);
			status = cublasIsamax(cuhand, ann.layer_sizes().back(),
					dev_actual_output, 1, &actualAmax);
			checkCudaErrors(status);
			correct += (expectedAmax == actualAmax) ? 1 : 0;
		}

		auto accuracy = (double) correct / (double) test_dataset.dataset_size;
		printf("Epoch %d accuracy: %.3f (%d / %d correct)\n", epoch_i, accuracy,
				correct, test_dataset.dataset_size);
	}

	status = cublasDestroy(cuhand);
	checkCudaErrors(status);

	return 0;
}

int main_genetic() {
	// population 20, 5% mutations
	auto gen = GeneticAlgorithmManager(20, 0.05);

	// 2% accuracy threshold, 100 generations max
	gen.run(0.02, 100);

	return 0;
}

int main() {
//	return main_backprop();
	return main_genetic();
}
