#include <torch/extension.h>

#include "pano_sampler.h"

__global__ void sample_pano_backward_kernel(
	const float* __restrict__ d_dLdpix,
	const float* __restrict__ d_dirs,
	float* __restrict__ d_dLdpano,
	const int pano_w,
	const int pano_h,
	const int w,
	const int h) {
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	if (x >= w || y >= h) return;
	int pix_id = y * w + x;
	float dir_x = d_dirs[pix_id * 3 + 0];
	float dir_y = d_dirs[pix_id * 3 + 1];
	float dir_z = d_dirs[pix_id * 3 + 2];
	float theta = atan2(dir_x, dir_y);
	float phi = atan2(dir_z, sqrt(dir_x * dir_x + dir_y * dir_y));
	float pano_x = (theta / (2 * M_PI) + 0.5f) * pano_w;
	float pano_y = (phi / M_PI + 0.5f) * pano_h;
	int pano_x0 = floor(pano_x);
	int pano_y0 = floor(pano_y);
	int pano_x1 = ceil(pano_x);
	int pano_y1 = ceil(pano_y);
	pano_x0 = min(max(0, pano_x0), pano_w - 2);
	pano_y0 = min(max(0, pano_y0), pano_h - 2);
	pano_x1 = min(max(1, pano_x1), pano_w - 1);
	pano_y1 = min(max(1, pano_y1), pano_h - 1);

	float dx = pano_x - pano_x0;
	float dy = pano_y - pano_y0;

	for (int c = 0; c < 3; c++) {
		float dLdpix = d_dLdpix[pix_id + c * w * h];
		float dLdv0 = dLdpix * (1 - dy);
		float dLdv1 = dLdpix * dy;
		float dLdv00 = dLdv0 * (1 - dx);
		float dLdv01 = dLdv0 * dx;
		float dLdv10 = dLdv1 * (1 - dx);
		float dLdv11 = dLdv1 * dx;
		atomicAdd(&d_dLdpano[(pano_y0 * pano_w + pano_x0) + c * pano_w * pano_h], dLdv00);
		atomicAdd(&d_dLdpano[(pano_y0 * pano_w + pano_x1) + c * pano_w * pano_h], dLdv01);
		atomicAdd(&d_dLdpano[(pano_y1 * pano_w + pano_x0) + c * pano_w * pano_h], dLdv10);
		atomicAdd(&d_dLdpano[(pano_y1 * pano_w + pano_x1) + c * pano_w * pano_h], dLdv11);
	}
}

__global__ void sample_pano_forward_kernel(
	const float* __restrict__ d_pano,
	const float* __restrict__ d_dirs,
	float* __restrict__ output,
	const int pano_w,
	const int pano_h,
	const int w,
	const int h) {
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	if (x >= w || y >= h) return;
	int pix_id = y * w + x;
	float dir_x = d_dirs[pix_id * 3 + 0];
	float dir_y = d_dirs[pix_id * 3 + 1];
	float dir_z = d_dirs[pix_id * 3 + 2];
	float theta = atan2(dir_x, dir_y);
	float phi = atan2(dir_z, sqrt(dir_x * dir_x + dir_y * dir_y));
	float pano_x = (theta / (2 * M_PI) + 0.5) * pano_w;
	float pano_y = (phi / M_PI + 0.5) * pano_h;
	int pano_x0 = floor(pano_x);
	int pano_y0 = floor(pano_y);
	int pano_x1 = ceil(pano_x);
	int pano_y1 = ceil(pano_y);
	pano_x0 = min(max(0, pano_x0), pano_w - 2);
	pano_y0 = min(max(0, pano_y0), pano_h - 2);
	pano_x1 = min(max(1, pano_x1), pano_w - 1);
	pano_y1 = min(max(1, pano_y1), pano_h - 1);

	float dx = pano_x - pano_x0;
	float dy = pano_y - pano_y0;
	for (int c = 0; c < 3; c++) {
		float v00 = d_pano[(pano_y0 * pano_w + pano_x0) + c * pano_w * pano_h];
		float v01 = d_pano[(pano_y0 * pano_w + pano_x1) + c * pano_w * pano_h];
		float v10 = d_pano[(pano_y1 * pano_w + pano_x0) + c * pano_w * pano_h];
		float v11 = d_pano[(pano_y1 * pano_w + pano_x1) + c * pano_w * pano_h];
		float v0 = v00 * (1 - dx) + v01 * dx;
		float v1 = v10 * (1 - dx) + v11 * dx;
		output[pix_id + c * w * h] = v0 * (1 - dy) + v1 * dy;
	}
}

void SAMPLER::panorama_sampler_forward(
	const float* d_pano,
	const float* d_dirs,
	float* d_output,
	const int w,
	const int h,
	const int pano_w,
	const int pano_h) {
	const dim3 threads(16, 16);
	const dim3 blocks((w + threads.x - 1) / threads.x, (h + threads.y - 1) / threads.y);
	sample_pano_forward_kernel<<<blocks, threads>>>(
		d_pano,
		d_dirs,
		d_output,
		pano_w,
		pano_h,
		w,
		h);
}

void SAMPLER::panorama_sampler_backward(
	const float* d_dLdpix,
	const float* d_dirs,
	float* d_dLdpano,
	const int w,
	const int h,
	const int pano_w,
	const int pano_h) {
	const dim3 threads(16, 16);
	const dim3 blocks((w + threads.x - 1) / threads.x, (h + threads.y - 1) / threads.y);
	sample_pano_backward_kernel<<<blocks, threads>>>(
		d_dLdpix,
		d_dirs,
		d_dLdpano,
		pano_w,
		pano_h,
		w,
		h);
}
