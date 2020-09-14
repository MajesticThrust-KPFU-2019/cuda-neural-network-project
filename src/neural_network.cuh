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

/**
 * @brief - A host object that uses CUDA kernels under the hood to speed up the calculations.
 */
class NeuralNetwork {
	// TODO parameterize activation function?
	// can be a enum or an abstract class template parameter with virtual
	// __device__ methods for the function and the derivative
	// the class must be created on the device for the virtual methods to work!

private:
	/** Layer dimensions of the neural network */
	std::vector<unsigned int> layer_sizes_;

	/** Weight matrices, column-major, n_layers - 1 */
	std::vector<float*> dev_weights;

	/** Activations from the last forward propagation, n_layers - 1 */
	std::vector<float*> dev_activations;

	/** cuBLAS context */
	cublasHandle_t cublasHandle;

public:
	const decltype(layer_sizes_)& layer_sizes() const;

public:
	/**
	 * @param layer_sizes - how many layers the network should have, including input and output layers; must be at least 2
	 */
	NeuralNetwork(std::initializer_list<unsigned int> layer_sizes);
	~NeuralNetwork();

	void init_random(float min = -1, float max = 1);
	void predict(float *dev_input, float *dev_output);
	void train_batch();
};

#endif /* NEURAL_NETWORK_CUH_ */
