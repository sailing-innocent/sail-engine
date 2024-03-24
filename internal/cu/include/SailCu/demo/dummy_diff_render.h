#pragma once
/**
 * @file demo/dummy_diff_render.h
 * @brief The CUDA based Dummy Diff Renderer
 * @date 2024-03-20
 * @author sailing-innocent
*/
#include <cstdint>
#include "SailCu/config.h"

namespace sail::cu {

class SAIL_CU_API DummyDiffRender {
public:
	DummyDiffRender() = default;
	~DummyDiffRender() = default;
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