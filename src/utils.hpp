/*
 * utils.hpp
 *
 *  Created on: Sep 10, 2020
 *      Author: timur
 */

#ifndef UTILS_HPP_
#define UTILS_HPP_

#include <memory>
#include <string>
#include <stdexcept>
#include <vector>
#include <iostream>
#include <numeric>

#include <cuda_runtime.h>

template<typename ... Args>
std::string string_format(const std::string &format, Args ... args) {
	size_t size = snprintf(nullptr, 0, format.c_str(), args ...) + 1; // Extra space for '\0'
	if (size <= 0) {
		throw std::runtime_error("Error during formatting.");
	}
	std::unique_ptr<char[]> buf(new char[size]);
	snprintf(buf.get(), size, format.c_str(), args ...);
	return std::string(buf.get(), buf.get() + size - 1); // We don't want the '\0' inside
}

template<typename T>
void print_vector(const std::vector<T>& v) {
	for (auto o : v) {
		std::cout << o << " ";
	}
	std::cout << std::endl;
}

template<typename T, typename A>
int arg_max(std::vector<T, A> const& vec) {
	return static_cast<int>(std::distance(vec.begin(),
			max_element(vec.begin(), vec.end())));
}

template<typename T, typename A>
int arg_min(std::vector<T, A> const& vec) {
	return static_cast<int>(std::distance(vec.begin(),
			min_element(vec.begin(), vec.end())));
}

#endif /* UTILS_HPP_ */
