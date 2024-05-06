#include "kernel.h"

__global__ void add_kernel(
	const float* a, const float* b, float* c, int N) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < N) {
		c[i] = a[i] + b[i];
	}
}

void dummy_add(const float* a, const float* b, float* c, int N) {
	int blockDim = 256;
	int gridDim = (N + blockDim - 1) / blockDim;
	add_kernel<<<gridDim, blockDim>>>(a, b, c, N);
}