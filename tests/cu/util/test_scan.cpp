#include "SailCu/primitive/scan.h"
#include "test_util.h"
#include "cuda_runtime.h"
#include <cstddef>
#include <numeric>
#include <vector>

namespace sail::test {

int test_cu_scan() {
	constexpr int N = 1000;
	std::vector<int> h_in_arr(N);
	std::iota(h_in_arr.begin(), h_in_arr.end(), 1);
	std::vector<int> h_out_arr_gt(N);
	std::vector<int> h_out_arr(N);

	// exclusive scan
	std::exclusive_scan(h_in_arr.begin(), h_in_arr.end(), h_out_arr_gt.begin(), 0);

	for (int i = 0; i < N; i++) {
		CHECK(h_out_arr_gt[i] == i * (i + 1) / 2);
	}

	// check allocate memory
	int* d_temp_storage = nullptr;
	int temp_storage_size = 0;
	cu::exclusive_scan(NULL, temp_storage_size, NULL, NULL, 100);
	CHECK(temp_storage_size == 1);// single pass
	cu::exclusive_scan(NULL, temp_storage_size, NULL, NULL, 1000);
	CHECK(temp_storage_size == 4 + 1);// ceil(1000 / 256) + 1
	cu::exclusive_scan(NULL, temp_storage_size, NULL, NULL, 10000);
	CHECK(temp_storage_size == 40 + 1);
	cu::exclusive_scan(NULL, temp_storage_size, NULL, NULL, 100000);
	CHECK(temp_storage_size == 391 + 2 + 1);

	// allocate temp storage size
	cudaMalloc(&d_temp_storage, temp_storage_size * sizeof(int));

	// exclusive scan
	int* d_int_arr;
	cudaMalloc(&d_int_arr, N * sizeof(int));
	cudaMemcpy(d_int_arr, h_in_arr.data(), N * sizeof(int), cudaMemcpyHostToDevice);
	int* d_out_arr;
	cudaMalloc(&d_out_arr, N * sizeof(int));

	cu::exclusive_scan(d_temp_storage, temp_storage_size, d_int_arr, d_out_arr, N);

	// copy out
	cudaMemcpy(h_out_arr.data(), d_out_arr, N * sizeof(int), cudaMemcpyDeviceToHost);

	// check
	for (int i = 0; i < N; i++) {
		CHECK(h_out_arr_gt[i] == h_out_arr[i]);
	}

	return 0;
}

}// namespace sail::test

TEST_SUITE("cu::utils") {
	TEST_CASE("scan") {
		CHECK(sail::test::test_cu_scan() == 0);
	}
}
