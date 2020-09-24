/*
 * genetic-algorithm.cu
 *
 *  Created on: Sep 24, 2020
 *      Author: timur
 */

#include "genetic-algorithm.cuh"

//#include <algorithm>
#include <numeric>
#include <random>

#include <cuda_runtime.h>
#include <curand.h>
#include <curand_kernel.h>

#include "cuda_helpers/helper_cuda.h"

#include "utils.hpp"
#include "cuda_utils.cuh"

GeneticAlgorithmManager::GeneticAlgorithmManager(size_t population_size,
		float mutation_rate) :
		population(population_size), mutation_rate(mutation_rate) {
	// TODO somehow reduce the number of data points to speed up calculations
	this->dataset = MnistDeviceDatasetProvider(true);

	std::initializer_list<size_t> ann_init_list = { this->dataset.x_size, 300,
			this->dataset.y_size };

	// allocate chromosomes
	this->chromosome_length = std::accumulate(ann_init_list.begin(),
			ann_init_list.end(), 0);
	cudaMalloc((void**) &this->dev_chromosome_a,
			this->chromosome_length * sizeof(float));
	getLastCudaError("Error allocating chromosome a");
	cudaMalloc((void**) &this->dev_chromosome_b,
			this->chromosome_length * sizeof(float));
	getLastCudaError("Error allocating chromosome b");

	// allocate ann output
	cudaMalloc((void**) &this->dev_nn_output,
			this->dataset.y_size * sizeof(float));
	getLastCudaError("Error allocating neural network output");

	// create initial population with random weights
	curandStatus_t curand_status;
	curandGenerator_t gen;
	curand_status = curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_MTGP32);
	checkCudaErrors(curand_status);

	curand_status = curandSetPseudoRandomGeneratorSeed(gen, time(0));
	checkCudaErrors(curand_status);

	for (auto i = 0; i < population_size; i++) {
		this->population[i] = new NeuralNetwork(ann_init_list);
		curand_status = curandGenerateUniform(gen, this->dev_chromosome_a,
				this->chromosome_length);
		checkCudaErrors(curand_status);

		// convert weights from 0..1 to a -1..+1 range
		cudaInvokeMaxOccupancy(0, 0, this->chromosome_length, axpb, 2.0f,
				(const float *) this->dev_chromosome_a, -1.0f,
				this->dev_chromosome_a, this->chromosome_length);

		this->chromosome_to_nn(this->population[i], this->dev_chromosome_a);
	}

	auto cublas_status = cublasCreate(&this->cublasHandle);
	checkCudaErrors(cublas_status);
}

GeneticAlgorithmManager::~GeneticAlgorithmManager() {
	for (auto nn : this->population) {
		delete nn;
	}

	auto status = cublasDestroy(this->cublasHandle);
	checkCudaErrors(status);
}

/**
 * arg_max with optional index omission
 * makes sure that the omitted index is never the result
 * needed for recomputing most fit index and preventing situations when
 * that index is the same as it was the last time
 */
size_t arg_max(std::vector<float> const& vec, int omit_index = -1) {
	size_t max_i;
	if (omit_index == 0) {
		max_i = 1;
	} else {
		max_i = 0;
	}

	for (auto i = 1; i < vec.size(); i++) {
		if (i != omit_index && vec[max_i] < vec[i]) {
			max_i = i;
		}
	}

	return max_i;
}

size_t arg_min(std::vector<float> const& vec, int omit_index = -1) {
	size_t min_i;
	if (omit_index == 0) {
		min_i = 1;
	} else {
		min_i = 0;
	}

	for (auto i = 1; i < vec.size(); i++) {
		if (i != omit_index && vec[min_i] > vec[i]) {
			min_i = i;
		}
	}

	return min_i;
}

/**
 * Selects a weighted element randomly and returns it's index.
 *
 * Takes a vector of positive weights and a uniform random number in range [0, 1].
 *
 * Optionally omits a single element.
 */
size_t weighted_randarg(std::vector<float> const& weights, float r,
		int omit_index = -1) {
	size_t i;
	float sum = 0;
	for (i = 0; i < weights.size(); i++) {
		if (i != omit_index)
			sum += weights[i];
	}

	float cumsum = 0;
	auto k = r * sum;
	for (i = 0; i < weights.size(); i++) {
		if (i == omit_index)
			continue;

		cumsum += weights[i];
		if (k <= cumsum)
			break;
	}

	return i;
}

void GeneticAlgorithmManager::run(float fitness_threshold,
		size_t max_generations) {
	std::mt19937 gen(time(0));
	std::uniform_real_distribution<float> dist(0.0, 1.0);

	// compute initial fitness
	auto fitness_values = std::vector<float>(this->population.size());
	for (auto i = 0; i < this->population.size(); i++) {
		fitness_values[i] = this->fitness(this->population[i]);
	}

	// tracking fitness threshold
	int prev_top_i = -1;
	int current_top_i = arg_max(fitness_values);

	printf("Fitness values:\n");
	print_vector(fitness_values);
	printf("prev_top = %d, curr_top = %d", prev_top_i, current_top_i);

	// generation loop
	for (auto gen_i = 0; gen_i < max_generations; gen_i++) {
		// selection
		auto select_a = weighted_randarg(fitness_values, dist(gen));
		auto select_b = weighted_randarg(fitness_values, dist(gen), select_a);

		// crossover
		this->crossover(this->population[select_a], this->population[select_b],
				dist(gen));

		// mutation
		this->mutation(this->dev_chromosome_a, this->chromosome_length);
		this->mutation(this->dev_chromosome_b, this->chromosome_length);

		// replace the weakest performers
		auto low_a = arg_min(fitness_values);
		auto low_b = arg_min(fitness_values, low_a);
		this->chromosome_to_nn(this->population[low_a], this->dev_chromosome_a);
		this->chromosome_to_nn(this->population[low_b], this->dev_chromosome_b);

		// recompute fitness
		fitness_values[low_a] = this->fitness(this->population[low_a]);
		fitness_values[low_b] = this->fitness(this->population[low_b]);

		prev_top_i = current_top_i;
		current_top_i = arg_max(fitness_values, prev_top_i);

		printf("Fitness values:\n");
		print_vector(fitness_values);
		printf("prev_top = %d, curr_top = %d", prev_top_i, current_top_i);

		// check top performers difference against the threshold, exit if converged
		float prev_top_fitness = fitness_values[prev_top_i];
		float current_top_fitness = fitness_values[current_top_i];
		if (abs(prev_top_fitness - current_top_fitness) <= fitness_threshold) {
			break;
		}
	}

	// TODO retrieve and copy the neural network via current_top_i
}

void GeneticAlgorithmManager::nn_to_chromosome(NeuralNetwork* nn,
		float* dev_chromosome) {
	size_t offset = 0;
	for (auto i = 0; i < (*nn).layer_sizes().size(); i++) {
		auto size = (*nn).layer_sizes()[i];
		cudaMemcpy(dev_chromosome + offset, (*nn).dev_weights[i],
				size * sizeof(float), cudaMemcpyDeviceToDevice);
		getLastCudaError("Error copying nn weights to chromosome");
	}
}

void GeneticAlgorithmManager::chromosome_to_nn(NeuralNetwork* nn,
		const float* dev_chromosome) {
	size_t offset = 0;
	for (auto i = 0; i < (*nn).layer_sizes().size(); i++) {
		auto size = (*nn).layer_sizes()[i];
		cudaMemcpy((*nn).dev_weights[i], dev_chromosome + offset,
				size * sizeof(float), cudaMemcpyDeviceToDevice);
		getLastCudaError("Error copying chromosome to nn weights");
	}
}

float GeneticAlgorithmManager::fitness(NeuralNetwork* nn) {
	float* dev_input = 0;
	float* dev_expected_output = 0;
	size_t correct = 0;
	for (auto point_i = 0; point_i < this->dataset.dataset_size; point_i++) {
		this->dataset.get_single_pair(&dev_input, &dev_expected_output);

		nn->predict(dev_input, this->dev_nn_output);

		int expectedAmax = 0, actualAmax = 0;

		cublasStatus_t status;
		status = cublasIsamax(this->cublasHandle, nn->layer_sizes().back(),
				dev_expected_output, 1, &expectedAmax);
		checkCudaErrors(status);
		status = cublasIsamax(this->cublasHandle, nn->layer_sizes().back(),
				this->dev_nn_output, 1, &actualAmax);
		checkCudaErrors(status);
		correct += (expectedAmax == actualAmax) ? 1 : 0;
	}

	auto accuracy = (double) correct / (double) this->dataset.dataset_size;

	return accuracy;
}

void GeneticAlgorithmManager::crossover(NeuralNetwork* nn_a,
		NeuralNetwork* nn_b, float r) {
	// perform a single swap up until a single crossover point
	this->nn_to_chromosome(nn_a, this->dev_chromosome_a);
	this->nn_to_chromosome(nn_b, this->dev_chromosome_b);
	size_t len = ceil(r * this->chromosome_length);
	auto status = cublasSswap(this->cublasHandle, len, this->dev_chromosome_a,
			1, this->dev_chromosome_b, 1);
	checkCudaErrors(status);
}

__global__ void mutation_kernel(float* ch, size_t N, float mut_rate,
		unsigned long long seed) {
	int tidx = threadIdx.x + blockIdx.x * blockDim.x;
	int stride = blockDim.x * gridDim.x;

	curandState_t state;
	curand_init(seed, tidx, 0, &state);
	float r = curand_uniform(&state);
	if (r <= mut_rate) {
		for (; tidx < N; tidx += stride) {
			// add uniform random value in range [-1, 1]
			ch[tidx] = ch[tidx] + (-1 + 2 * curand_uniform(&state));
		}
	}
}

void GeneticAlgorithmManager::mutation(float *dev_chromosome, size_t N) {
	cudaInvokeMaxOccupancy(0, 0, N, mutation_kernel, dev_chromosome, N,
			this->mutation_rate, (unsigned long long) time(0));
	cudaDeviceSynchronize();
}
