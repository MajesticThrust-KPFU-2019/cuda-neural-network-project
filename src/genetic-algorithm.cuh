/*
 * genetic-algorithm.cuh
 *
 *  Created on: Sep 24, 2020
 *      Author: timur
 */

#ifndef GENETIC_ALGORITHM_CUH_
#define GENETIC_ALGORITHM_CUH_

class NeuralNetwork;

#include <vector>

#include <cublas_v2.h>

#include "neural_network.cuh"
#include "device_dataset.cuh"

class GeneticAlgorithmManager {
private:
	std::vector<NeuralNetwork*> population;
	float mutation_rate;

	size_t chromosome_length;
	float* dev_chromosome_a;
	float* dev_chromosome_b;

	MnistDeviceDatasetProvider dataset;
	float* dev_nn_output;

	cublasHandle_t cublasHandle;

public:
	/**
	 * Initializes a genetic algorithm for neural networks.
	 * Has a fixed population size.
	 * Stopping condition is either a fitness threshold or max
	 * generations - whichever comes first (specified on algorithm launch).
	 *
	 * @param population size - how many neural networks can exist in a population at once
	 * @param mutation_rate - probability with which each gene mutates
	 */
	GeneticAlgorithmManager(size_t population_size, float mutation_rate);
	~GeneticAlgorithmManager();

	/**
	 * Launches genetic algorithm.
	 *
	 * If the internal state is already dirty, can be used to resume the algorithm with new stopping condition.
	 *
	 * @param fitness_threshold - the minimum difference in fitness between generations that causes the algorithm to stop.
	 * @param max_generations - after how many generations the algorithm should stop
	 */
	void run(float fitness_threshold, size_t max_generations);

private:
	/**
	 * Copies neural network weights into the chromosome array.
	 * Assumes that `dev_chromosome` has enough allocated space for all the weights.
	 * The chromosome length is calculated as the sum of all network's layer sizes.
	 */
	void nn_to_chromosome(NeuralNetwork* nn, float* dev_chromosome);

	/**
	 * Instantiates neural network weights from the chromosome array.
	 * The chromosome length is calculated as the sum of all network's layer sizes.
	 */
	void chromosome_to_nn(NeuralNetwork* nn, const float* dev_chromosome);

	/**
	 * Calculates fitness of given neural network.
	 */
	float fitness(NeuralNetwork* nn);

	/**
	 * Performs a crossover between two neural networks.
	 * Takes two networks and a random value in range [0, 1].
	 * Stores the newly created weights in `dev_chromosome_a` and `dev_chromosome_b`.
	 */
	void crossover(NeuralNetwork* nn_a, NeuralNetwork* nn_b, float r);

	/**
	 * Performs a mutation on a given chromosome.
	 * Takes dev pointer to the chromosome and count of elements.
	 */
	void mutation(float *dev_chromosome, size_t N);
};

#endif /* GENETIC_ALGORITHM_CUH_ */
