#pragma once
#ifndef CUDA_SH_PROCESSOR_H_INCLUDED
#define CUDA_SH_PROCESSOR_H_INCLUDED

#include <functional>

namespace CudaSHProcessor {
class SHProcessor {
public:
	static void forward(
		std::function<char*(size_t)> geom_buffer,
		// input
		const float* shs,
		const float* dirs,
		// params
		const int P, int D, int M,
		// output
		float* colors);

	static void backward(
		char* geom_buffer,
		// input
		const float* dL_dcolor,
		// params
		const int P, int D, int M,
		const float* shs,
		const float* dirs,
		// output
		float* dL_dshs,
		float* dL_ddirs);
};
}// namespace CudaSHProcessor

#endif// CUDA_SH_PROCESSOR_H_INCLUDED