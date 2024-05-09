#pragma once

#include <functional>

namespace sail::cu::gs {

int forward(
	// resource
	std::function<char*(size_t)> geom_buffer,
	std::function<char*(size_t)> tile_buffer,
	std::function<char*(size_t)> image_buffer,
	// params
	const int P,
	const int width, const int height,
	// input
	const float* mean3d,
	// output
	float* out_color,
	// debug
	bool debug = false);

void backward(
	// resource
	char* geom_buffer,
	char* tile_buffer,
	char* image_buffer,
	// params
	const int P,
	// input
	const float* dL_dpix,
	// output
	float* dL_dmean3d,
	// debug
	bool debug);

}// namespace sail::cu::gs