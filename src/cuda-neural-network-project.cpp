//============================================================================
// Name        : cuda-neural-network-project.cpp
// Author      : Timur Mukhametulin
// Version     :
// Copyright   : 
// Description : Hello World in C++, Ansi-style
//============================================================================

#include <iostream>
#include <vector>
#include <mnist/mnist_reader.hpp>
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

	auto ann = NeuralNetwork { 28*28, 300, 10 };

	return 0;
}
