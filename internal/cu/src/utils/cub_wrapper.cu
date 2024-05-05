/**
 * @file cub_wrapper.cu
 * @brief The CUB Wrapper Implementation
 * @author sailing-innocent
 * @date 2024-05-05
 */

#include "SailCu/utils/cub_warpper.h"

#include <cuda.h>
#include <cub/cub.cuh>
// #include <iostream>

namespace sail::cu {

void cub_inclusive_sum(int* d_in, int* d_out, int N) {
	void* d_temp_storage = nullptr;
	size_t temp_storage_bytes = 0;
	cub::DeviceScan::InclusiveSum(d_temp_storage, temp_storage_bytes, d_in, N);
	cudaDeviceSynchronize();
	// std::cout << "temp_storage_bytes: " << temp_storage_bytes << std::endl;
	cudaMalloc(&d_temp_storage, temp_storage_bytes);
	cub::DeviceScan::InclusiveSum(d_temp_storage, temp_storage_bytes, d_in, d_out, N);
	cudaFree(d_temp_storage);
}

}// namespace sail::cu
