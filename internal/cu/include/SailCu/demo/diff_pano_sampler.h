#pragma once
/**
 * @file demo/diff_pano_sampler.h
 * @brief The CUDA based Differential Panomara Sampler
 * @date 2024-04-05
 * @author sailing-innocent
*/
#include <cstdint>
#include "SailCu/config.h"

namespace sail::cu {

class SAIL_CU_API DiffPanoSampler {
public:
	DiffPanoSampler() = default;
	~DiffPanoSampler() = default;
	void forward(
		// input
		float* d_source_pix,
		// params
		int h, int w,
		// output
		float* d_target_pix) noexcept;
	void forward_py(int64_t d_source_pix, int h, int w, int64_t d_target_pix) noexcept;
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