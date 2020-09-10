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

/**
 * @brief - A host object that uses CUDA kernels under the hood to speed up the calculations.
 */
class NeuralNetwork {
	// TODO parameterize activation function?
	// can be a enum or an abstract class template parameter with virtual
	// __device__ methods for the function and the derivative
	// the class must be created on the device for the virtual methods to work!

private:
	std::vector<unsigned int> layer_sizes;

	/** weight matrices, n_layers - 1 */
	std::vector<float*> dev_weights;

	/** activations from the last forward propagation, n_layers - 1 */
	std::vector<float*> dev_activations;

public:
	/**
	 * @param layer_sizes - how many layers the network should have, including input and output layers; must be at least 2
	 */
	NeuralNetwork(std::initializer_list<unsigned int> layer_sizes);
	~NeuralNetwork();

	void init_random();
	void predict();
	void train_batch();
};

#endif /* NEURAL_NETWORK_CUH_ */
