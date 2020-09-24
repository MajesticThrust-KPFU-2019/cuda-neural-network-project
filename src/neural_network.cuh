/*
 * neural_network.cuh
 *
 *  Created on: Sep 7, 2020
 *      Author: timur
 */

#ifndef NEURAL_NETWORK_CUH_
#define NEURAL_NETWORK_CUH_

#include <initializer_list>
#include <vector>
#include <cublas_v2.h>

#include "genetic-algorithm.cuh"

/**
 * @brief - A host object that uses CUDA kernels under the hood to speed up the calculations.
 */
class NeuralNetwork {
	// TODO parameterize activation function?
	// can be a enum or an abstract class template parameter with virtual
	// __device__ methods for the function and the derivative
	// the class must be created on the device for the virtual methods to work!
private:
	friend class GeneticAlgorithmManager;

private:
	/** Layer dimensions of the neural network */
	std::vector<size_t> layer_sizes_;

	/**
	 * Weight matrices, column-major, n_layers - 1.
	 *
	 * Weights between layer 0 and 1 have an index of 0, and so on.
	 */
	std::vector<float*> dev_weights;

	/** Bias vectors, n_layers - 1 */
	std::vector<float*> dev_biases;

	/** Activations from the last forward propagation, n_layers. Includes input and output layers. */
	std::vector<float*> dev_activations;

	/** Errors for each layer, n_layers - 1 */
	std::vector<float*> dev_errors;

	/** A vector for intermediate calculations, length = max(layer_sizes) */
	float* dev_intermediate;

	/** cuBLAS context */
	cublasHandle_t cublasHandle;

public:
	const decltype(layer_sizes_)& layer_sizes() const;

private:
	/**
	 * Forward propagation. Fills internal activation vectors. Writes the last layer output to dev_output.
	 *
	 * Expects the input to be a vector with the same length as the input layer.
	 * The output is written to the last pointer in the dev_activations array.
	 *
	 * @param dev_input - device pointer to the input vector
	 */
	void evaluate(const float *dev_input);

public:
	/**
	 * @param layer_sizes - how many layers the network should have, including input and output layers; must be at least 2
	 */
	NeuralNetwork(std::initializer_list<size_t> layer_sizes);
	~NeuralNetwork();

	/**
	 * Initialize neural network weights randomly.
	 *
	 * @param min - minimum value of a single weight
	 * @param max - maximum value of a single weight
	 */
	void init_random(float min = -1, float max = 1);

	/**
	 * Evaluates the network and outputs argmax
	 *
	 * Expects the input to be a vector with the same length as the input layer.
	 * Expects the output to have enough allocated space for the output vector.
	 *
	 * @param dev_input - device pointer to the input vector
	 * @param dev_output - device pointer for the output vector
	 */
	void predict(const float *dev_input, float *dev_output);
	/**
	 * Update network weights with a single data point.
	 *
	 * @param dev_x_train - a device pointer to input
	 * @param dev_y_train - a device pointer to expected (training) output
	 * @param learning_rate
	 * @param out_cost - a reference to write the calculated cost to
	 */
	void train(const float* dev_x_train, const float* dev_y_train,
			float learning_rate, float* out_cost);
};

#endif /* NEURAL_NETWORK_CUH_ */
