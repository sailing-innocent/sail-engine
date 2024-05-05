#include "test_util.h"
#include "SailCu/utils/cub_warpper.h"
#include <cuda_runtime_api.h>

// WIP: bug in cub_inclusive_sum

namespace sail::test {

int test_inclusive_sum() {
	int N = 1000;
	int* h_array = new int[N];
	int* d_array;
	int* d_array_out;
	cudaMalloc((void**)&d_array, N * sizeof(int));
	cudaMalloc((void**)&d_array_out, N * sizeof(int));
	for (int i = 0; i < N; i++) {
		h_array[i] = i;
	}
	cudaMemcpy(d_array, h_array, N * sizeof(int), cudaMemcpyHostToDevice);

	sail::cu::cub_inclusive_sum(d_array, d_array_out, N);

	cudaMemcpy(h_array, d_array_out, N * sizeof(int), cudaMemcpyDeviceToHost);
	cudaDeviceSynchronize();
	for (int i = 0; i < N; i++) {
		CHECK(h_array[i] == i * (i + 1) / 2);
	}
	cudaFree(d_array);
	cudaFree(d_array_out);
	delete[] h_array;
	return 0;
}

}// namespace sail::test

TEST_SUITE("cub") {
	TEST_CASE("prefix_scan") {
		using namespace sail::test;
		CHECK(test_inclusive_sum() == 0);
	}
}