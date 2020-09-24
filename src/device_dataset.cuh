/*
 * device_dataset.cuh
 *
 *  Created on: Sep 8, 2020
 *      Author: timur
 */

#ifndef DEVICE_DATASET_CUH_
#define DEVICE_DATASET_CUH_

#include <vector>

/**
 * Provides device pointers to training/validation data pairs for mnist dataset.
 *
 * Preprocesses the data in the following way:
 *	- inputs are converted from 0..255 to 0..1 range
 *	- outputs are converted into vectors with length of 10,
 * 	  where all elements are 0 and and the correct label index is 1
 *
 * Performs dataset shuffling for an epoch and allows for sequential point retrieval.
 *
 * Manages memory copying between host and device.
 *
 */
class MnistDeviceDatasetProvider {
private:
	std::vector<float*> dev_xs;
	std::vector<float*> dev_ys;

	/**
	 * Indices for random order access.
	 */
	std::vector<size_t> indices;

	/**
	 * Current iteration number/index in epoch.
	 */
	size_t i;

public:
	size_t x_size;
	size_t y_size;
	size_t dataset_size;

	/**
	 * Loads mnist dataset into device memory.
	 * Loads training dataset by default. If `training` is false, loads the test dataset.
	 */
	MnistDeviceDatasetProvider(bool training = true);
	~MnistDeviceDatasetProvider();

	/**
	 * Initializes an epoch by shuffling internal array and resetting state.
	 */
	void init_epoch();

	/**
	 * Returns a single input and expected output pair.
	 * Expects host pointers to device pointers.
	 * Input has a length of `x_size`.
	 * Output has a length of `y_size`.
	 */
	void get_single_pair(float **out_dev_x, float **out_dev_y);
};

#endif /* DEVICE_DATASET_CUH_ */
