/**
 * @file dummy.cu
 * @brief Basic CUDA kernels and api
 * @date 2023-10-04
 * @author sailing-innocent
*/

#include "SailCu/dummy.h"

namespace sail::cu {

__global__ void cuda_add(int* d_array, int N) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < N) {
		d_array[idx] += 1;
	}
}

void cuda_inc(int* d_array, int N) {
	int block_size = 256;
	int grid_size = (N + block_size - 1) / block_size;
	cuda_add<<<grid_size, block_size>>>(d_array, N);
	cudaDeviceSynchronize();
}

}// namespace sail::cu