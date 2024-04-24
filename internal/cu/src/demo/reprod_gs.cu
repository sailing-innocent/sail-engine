#include "SailCu/demo/reprod_gs.h"

namespace sail::cu {

__global__ void debug_img_shader(int w, int h, float* d_out) {
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	if (x >= w || y >= h) {
		return;
	}
	int idx = y * w + x;
	d_out[3 * idx + 0] = (float)x / w;
	d_out[3 * idx + 1] = (float)y / h;
	d_out[3 * idx + 2] = 0.0f;
}

void ReprodGs::debug_img(int w, int h, std::span<float> h_out) noexcept {
	float* d_out;
	cudaMalloc(&d_out, 3 * w * h * sizeof(float));
	dim3 block(16, 16);
	dim3 grid((w + block.x - 1) / block.x, (h + block.y - 1) / block.y);
	debug_img_shader<<<grid, block>>>(w, h, d_out);
	cudaMemcpy(h_out.data(), d_out, 3 * w * h * sizeof(float), cudaMemcpyDeviceToHost);
	cudaFree(d_out);
}

}// namespace sail::cu