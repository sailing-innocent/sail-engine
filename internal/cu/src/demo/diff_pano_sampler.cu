
/**
 * @file demo/diff_pano_sampler.cu
 * @brief The CUDA based Diff PanoSampler
 * @date 2024-03-20
 * @author sailing-innocent
*/

#include "SailCu/demo/diff_pano_sampler.h"
#include <iostream>
#include <span>

namespace sail::cu {

}// namespace sail::cu

namespace sail::cu {

void DiffPanoSampler::forward(
	float* d_source_pix,
	// params
	const int ph,			 // res height h, the total resolution 3, h, 2h
	std::span<float, 3> dir, // direction x,y,z
	const float fov_y,		 // field of view
	const int h, const int w,// the sampled image res 3 x h x w
	float* d_target_pix) noexcept {
	dim3 block(32, 32);
	dim3 grid((w + block.x - 1) / block.x, (h + block.y - 1) / block.y);
	// sample_forward_kernel<<<grid, block>>>(w, h, d_source_pix, d_target_pix);
}

void DiffPanoSampler::forward_py(
	int64_t d_source_pix,
	// params
	const int ph,			 // res height h, the total resolution 3, h, 2h
	std::span<float, 3> dir, // direction x,y,z
	const float fov_y,		 // field of view
	const int h, const int w,// the sampled image res 3 x h x w
	int64_t d_target_pix) noexcept {
	m_w = w;
	m_h = h;
	forward(
		reinterpret_cast<float*>(d_source_pix),
		ph, dir, fov_y,
		h, w,
		reinterpret_cast<float*>(d_target_pix));
}

void DiffPanoSampler::backward(float* d_dL_d_target_pix, float* d_dL_d_source_pix) noexcept {
	dim3 block(32, 32);
	dim3 grid((m_w + block.x - 1) / block.x, (m_h + block.y - 1) / block.y);
	// sample_backward_kernel<<<grid, block>>>(m_w, m_h, d_dL_d_target_pix, d_dL_d_source_pix);
}

void DiffPanoSampler::backward_py(int64_t d_dL_d_target_pix, int64_t d_dL_d_source_pix) noexcept {
	// std::cout << "backward" << std::endl;
	backward(reinterpret_cast<float*>(d_dL_d_target_pix), reinterpret_cast<float*>(d_dL_d_source_pix));
}

}// namespace sail::cu