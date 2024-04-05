#pragma once
/**
 * @file demo/diff_pano_sampler.h
 * @brief The CUDA based Differential Panomara Sampler
 * @date 2024-04-05
 * @author sailing-innocent
*/
#include <cstdint>
#include "SailCu/config.h"
#include <span>

namespace sail::cu {

class SAIL_CU_API DiffPanoSampler {
public:
	DiffPanoSampler() = default;
	~DiffPanoSampler() = default;
	void forward(
		// input
		float* d_source_pix,// the source pano image data
		// params
		const int ph,			 // res height h, the total resolution 3, h, 2h
		std::span<float, 3> dir, // direction x,y,z
		const float fov_y,		 // field of view
		const int h, const int w,// the sampled image res 3 x h x w
		// output
		float* d_target_pix) noexcept;
	void forward_py(
		int64_t d_source_pix,
		// params
		const int ph,			 // res height h, the total resolution 3, h, 2h
		std::span<float, 3> dir, // direction x,y,z
		const float fov_y,		 // field of view
		const int h, const int w,// the sampled image res 3 x h x w
		int64_t d_target_pix) noexcept;
	void backward(
		// input
		float* d_dL_d_target_pix,
		// params
		float* d_dL_d_source_pix) noexcept;
	void backward_py(int64_t d_dL_d_target_pix, int64_t d_dL_d_source_pix) noexcept;

private:
	// save for backward
	int m_w;
	int m_h;
};

}// namespace sail::cu