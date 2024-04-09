#pragma once
#ifndef CUDA_SH_PROCESSOR_BACKWARD_H_INCLUDED
#define CUDA_SH_PROCESSOR_BACKWARD_H_INCLUDED

#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#define GLM_FORCE_CUDA
#include <glm/glm.hpp>

namespace BACKWARD {
void eval_sh(
	int P, int D, int M,
	const float* shs,
	const float* dirs,
	const bool* clamped,
	// input
	const float* dL_dcolor,
	// output
	float* dL_dsh,
	float* dL_ddir);
}// namespace BACKWARD

#endif// CUDA_SH_PROCESSOR_BACKWARD_H_INCLUDED