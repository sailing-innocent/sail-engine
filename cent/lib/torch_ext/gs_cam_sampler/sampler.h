#ifndef CUDA_SAMPLER_H_INCLUDED
#define CUDA_SAMPLER_H_INCLUDED

#include <vector>
#include <functional>

namespace CudaSampler {
class Sampler {
public:
	static void markVisible(
		int P,
		float* means3D,
		float* viewmatrix,
		float* projmatrix,
		bool* present);

	static int forward(
		std::function<char*(size_t)> geometryBuffer,
		std::function<char*(size_t)> binningBuffer,
		std::function<char*(size_t)> imageBuffer,
		const int P,
		const float* background,
		const int width, int height,
		const float* means3D,
		const float* colors,
		const float* opacities,
		const float* scales,
		const float scale_modifier,
		const float* rotations,
		const float* cov3D_precomp,
		const float* viewmatrix,
		const float* projmatrix,
		const float tan_fovx, float tan_fovy,
		const bool prefiltered,
		float* out_color,
		int* radii = nullptr,
		bool debug = false);

	static void backward(
		const int P, int R,
		const float* background,
		const int width, int height,
		const float* means3D,
		const float* colors,
		const float* scales,
		const float scale_modifier,
		const float* rotations,
		const float* cov3D_precomp,
		const float* viewmatrix,
		const float* projmatrix,
		const float tan_fovx, float tan_fovy,
		const int* radii,
		char* geom_buffer,
		char* binning_buffer,
		char* image_buffer,
		const float* dL_dpix,
		float* dL_dmean2D,
		float* dL_dconic,
		float* dL_dopacity,
		float* dL_dcolor,
		float* dL_dmean3D,
		float* dL_dcov3D,
		float* dL_dscale,
		float* dL_drot,
		bool debug);
};
};// namespace CudaSampler

#endif