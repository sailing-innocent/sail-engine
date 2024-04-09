#include "processor_impl.h"
#include <iostream>
#include <fstream>
#include <algorithm>
#include <numeric>
#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cub/cub.cuh>
#include <cub/device/device_radix_sort.cuh>
#define GLM_FORCE_CUDA
#include <glm/glm.hpp>

#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
namespace cg = cooperative_groups;

#include "auxiliary.h"
#include "forward.h"
#include "backward.h"

CudaSHProcessor::GeometryState CudaSHProcessor::GeometryState::fromChunk(char*& chunk, size_t P) {
	GeometryState geom;
	obtain(chunk, geom.clamped, P * 3, 128);
	return geom;
}

void CudaSHProcessor::SHProcessor::forward(
	std::function<char*(size_t)> geom_buffer,
	// input
	const float* shs,
	const float* dirs,
	// params
	const int P, int D, int M,
	// output
	float* colors) {

	size_t chunk_size = required<GeometryState>(P);
	char* chunkptr = geom_buffer(chunk_size);
	GeometryState geom_state = GeometryState::fromChunk(chunkptr, P);
	// Run preprocessing per-Gaussian (transformation, bounding, conversion of SHs to RGB)

	FORWARD::eval_sh(
		P, D, M,
		shs,
		dirs,
		geom_state.clamped,
		colors);
}

void CudaSHProcessor::SHProcessor::backward(
	char* geom_buffer,
	// input
	const float* dL_dcolor,
	// params
	const int P, int D, int M,
	const float* shs,
	const float* dirs,
	// output
	float* dL_dshs,
	float* dL_ddirs) {
	GeometryState geom_state = GeometryState::fromChunk(geom_buffer, P);

	BACKWARD::eval_sh(
		P, D, M,
		shs,
		dirs,
		geom_state.clamped,
		dL_dcolor,
		dL_dshs,
		dL_ddirs);
}