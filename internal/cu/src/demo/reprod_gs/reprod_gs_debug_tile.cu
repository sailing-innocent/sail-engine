#include "SailCu/demo/reprod_gs.h"
#include <cooperative_groups.h>

namespace cg = cooperative_groups;

namespace sail::cu {

__global__ void debug_tile_shader(int w, int h, int BLOCK_X, int BLOCK_Y, float* d_out) {
	auto block = cg::this_thread_block();
	uint2 pix_min = {block.group_index().x * BLOCK_X, block.group_index().y * BLOCK_Y};
	uint2 pix_max = {min(pix_min.x + BLOCK_X, w), min(pix_min.y + BLOCK_Y, h)};
	uint2 pix = {pix_min.x + block.thread_index().x, pix_min.y + block.thread_index().y};
	uint32_t pix_id = pix.y * w + pix.x;
	float2 pixf = {(float)pix.x, (float)pix.y};
	uint2 grid_id = {block.group_index().x, block.group_index().y};
	bool inside = pix.x < w && pix.y < h;
	// chessboard pattern
	float C[3] = {1.0f, 1.0f, 0.0f};// yellow
	if ((grid_id.x + grid_id.y) % 2 == 0) {
		// blue
		C[0] = 0.0f;
		C[1] = 0.0f;
		C[2] = 1.0f;
	}

	if (inside) {
		for (int ch = 0; ch < 3; ch++)
			d_out[pix_id * 3 + ch] = C[ch];
	}
}

void ReprodGs::debug_tile(int w, int h, std::vector<float> h_out) noexcept {
	float* d_out;
	cudaMalloc(&d_out, 3 * w * h * sizeof(float));
	dim3 block(16, 16);
	dim3 grid((w + block.x - 1) / block.x, (h + block.y - 1) / block.y);
	debug_tile_shader<<<grid, block>>>(w, h, block.x, block.y, d_out);
	cudaMemcpy(h_out.data(), d_out, 3 * w * h * sizeof(float), cudaMemcpyDeviceToHost);
	cudaFree(d_out);
}

}// namespace sail::cu