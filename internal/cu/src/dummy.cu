/**
 * @file dummy.cu
 * @brief Basic CUDA kernels and api
 * @date 2023-10-04
 * @author sailing-innocent
*/

#include "SailCu/dummy.h"
#include "SailCu/kernel/dummy.cuh"
#include <iostream>

namespace sail::cu {

void cuda_inc(int* d_array, int N) {
	int block_size = 256;
	int grid_size = (N + block_size - 1) / block_size;
	cuda_inc_kernel<<<grid_size, block_size>>>(d_array, N);
}

void cuda_add(int* d_array_a, int* d_array_b, int* d_array_c, int N) {
	// std::cout << "cuda_add " << N << std::endl;
	int block_size = 256;
	int grid_size = (N + block_size - 1) / block_size;
	cuda_add_kernel<<<grid_size, block_size>>>(d_array_a, d_array_b, d_array_c, N);
}

}// namespace sail::cu