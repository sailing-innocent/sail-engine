
/**
 * @file demo/diff_pano_sampler.cu
 * @brief The CUDA based Diff PanoSampler
 * @date 2024-03-20
 * @author sailing-innocent
*/

#include "SailCu/demo/diff_pano_sampler.h"
#include <iostream>

namespace sail::cu {

__global__ void sample_forward_kernel(int W, int H, float* d_spix, float* d_tpix) {
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	if (x >= W || y >= H) return;
	int i = y * W + x;
	for (int c = 0; c < 3; ++c) {
		d_tpix[i + c * W * H] = 1.0f - d_spix[i + c * W * H];
	}
}

__global__ void sample_backward_kernel(int W, int H, float* dL_dtpix, float* dL_dspix) {
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	if (x >= W || y >= H) return;

	int i = y * W + x;
	for (int c = 0; c < 3; ++c) {
		printf("dL_dtpix[%d] = %f\n", i + c * W * H, dL_dtpix[i + c * W * H]);
		dL_dspix[i + c * W * H] = -dL_dtpix[i + c * W * H];
	}
}

}// namespace sail::cu

namespace sail::cu {

void DiffPanoSampler::forward(float* d_source_pix, int h, int w, float* d_target_pix) noexcept {
	dim3 block(32, 32);
	dim3 grid((w + block.x - 1) / block.x, (h + block.y - 1) / block.y);
	sample_forward_kernel<<<grid, block>>>(w, h, d_source_pix, d_target_pix);
}

void DiffPanoSampler::forward_py(int64_t d_source_pix, int h, int w, int64_t d_target_pix) noexcept {
	m_w = w;
	m_h = h;
	forward(reinterpret_cast<float*>(d_source_pix), h, w, reinterpret_cast<float*>(d_target_pix));
}

void DiffPanoSampler::backward(float* d_dL_d_target_pix, float* d_dL_d_source_pix) noexcept {
	dim3 block(32, 32);
	dim3 grid((m_w + block.x - 1) / block.x, (m_h + block.y - 1) / block.y);
	sample_backward_kernel<<<grid, block>>>(m_w, m_h, d_dL_d_target_pix, d_dL_d_source_pix);
}

void DiffPanoSampler::backward_py(int64_t d_dL_d_target_pix, int64_t d_dL_d_source_pix) noexcept {
	// std::cout << "backward" << std::endl;
	backward(reinterpret_cast<float*>(d_dL_d_target_pix), reinterpret_cast<float*>(d_dL_d_source_pix));
}

}// namespace sail::cu