#include "device_dataset.cuh"

#include <random>
#include <algorithm>
#include <numeric>

#include "mnist/mnist_reader.hpp"

#include <cuda_runtime.h>
#include "cuda_helpers/helper_cuda.h"

MnistDeviceDatasetProvider::MnistDeviceDatasetProvider(bool training) {
	auto dataset = mnist::read_dataset<std::vector, std::vector, uint8_t,
			uint8_t>();

	auto& images = training ? dataset.training_images : dataset.test_images;
	auto& labels = training ? dataset.training_labels : dataset.test_labels;

	// mnist dataset specs
	this->x_size = 28 * 28;
	this->y_size = 10;

	this->dataset_size = images.size(); // == labels.size()

	this->indices.resize(this->dataset_size);
	std::iota(this->indices.begin(), this->indices.end(), 0);
	this->i = 0;

	this->dev_xs.resize(this->dataset_size);
	this->dev_ys.resize(this->dataset_size);

	// tmp arrays for data conversion
	float x[this->x_size];
	float y[this->y_size];

	for (auto i = 0; i < this->dataset_size; i++) {
		cudaMalloc((void**) &(this->dev_xs[i]), this->x_size * sizeof(float));
		getLastCudaError("Error allocating memory for x");

		// convert x from uint 0..255 to float 0..1
		for (auto j = 0; j < this->x_size; j++) {
			x[j] = images[i][j] / 255.0;
		}
		cudaMemcpy(this->dev_xs[i], x, this->x_size * sizeof(float),
				cudaMemcpyHostToDevice);
		getLastCudaError("Error copying x to device");

		cudaMalloc((void**) &(this->dev_ys[i]), this->y_size * sizeof(float));
		getLastCudaError("Error allocating memory for y");

		// convert y to vector
		for (auto j = 0; j < this->y_size; j++) {
			y[j] = labels[i] == j ? 1 : 0;
		}
		cudaMemcpy(this->dev_ys[i], y, this->y_size * sizeof(float),
				cudaMemcpyHostToDevice);
		getLastCudaError("Error copying y to device");
	}
}

MnistDeviceDatasetProvider::~MnistDeviceDatasetProvider() {

}

void MnistDeviceDatasetProvider::init_epoch() {
	auto rng = std::default_random_engine { };
	std::shuffle(this->indices.begin(), this->indices.end(), rng);
	this->i = 0;
}

void MnistDeviceDatasetProvider::get_single_pair(float **out_dev_x,
		float **out_dev_y) {
	auto i = this->indices[this->i];

	*out_dev_x = this->dev_xs[i];
	*out_dev_y = this->dev_ys[i];

	this->i++;
}
