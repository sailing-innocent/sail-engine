/**
 * @file demo/simple2d.cu
 * @brief Some Simple CUDA shader
 * @date 2024-03-20
 * @author sailing-innocent
*/

#include "SailCu/demo/simple2d.h"

namespace sail::cu {

__global__ void sine_wave_shader(float* pixels, float t, unsigned int height, unsigned int width) {
	unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

	// calculate uv coordinate
	float u = x / (float)width;
	float v = y / (float)height;

	u = 2.0f * u - 1;
	v = 2.0f * v - 1;

	// calculate simple sine wave pattern
	float freq = 18.0f;
	// float w = sinf(u * freq + time) * cosf(v * freq + time) * 0.5f;
	float w = sinf(freq * sqrtf(u * u + v * v) - t * 6.0f);
	// write position
	// pixels[y*width + x] = make_float4(u, v, w, 1.0f);
	pixels[8 * (y * width + x) + 0] = u;
	pixels[8 * (y * width + x) + 1] = v;
	pixels[8 * (y * width + x) + 2] = w;
	pixels[8 * (y * width + x) + 3] = 1.0f;
	// generate color
	pixels[8 * (y * width + x) + 4] = w + 0.5f;
	pixels[8 * (y * width + x) + 5] = 0.3f;
	pixels[8 * (y * width + x) + 6] = 0.8f;
	pixels[8 * (y * width + x) + 7] = 1.0f;
}

void Simple2DShader::sine_wave(float* pixels, float t, int height, int width) {
	dim3 blocks(width / 16, height / 16);
	dim3 threads(16, 16);
	sine_wave_shader<<<blocks, threads>>>(pixels, t, height, width);
}

}// namespace sail::cu