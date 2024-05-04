/**
 * @file dummy.cuh
 * @brief The Dummy Kernel
 * @author sailing-innocent
 * @date 2024-05-05
 */

#include <cuda.h>

namespace sail::cu {

__global__ void cuda_inc_kernel(int* d_array, int N) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < N) {
		d_array[idx] += 1;
	}
}

__global__ void cuda_add_kernel(const int* d_array_a, const int* d_array_b, int* d_array_c, int N) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < N) {
		d_array_c[idx] = d_array_a[idx] + d_array_b[idx];
	}
}

}// namespace sail::cu