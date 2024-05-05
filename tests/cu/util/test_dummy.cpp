#include "SailCu/dummy.h"
#include "test_util.h"
#include "cuda_runtime_api.h"// cuMemcpy, cuMalloc

namespace sail::test {

int test_cuda_inc() {
	int N = 100;
	int* h_array = new int[N];
	int* d_array;
	cudaMalloc((void**)&d_array, N * sizeof(int));
	for (int i = 0; i < N; i++) {
		h_array[i] = i;
	}
	cudaMemcpy(d_array, h_array, N * sizeof(int), cudaMemcpyHostToDevice);
	sail::cu::cuda_inc(d_array, N);
	cudaMemcpy(h_array, d_array, N * sizeof(int), cudaMemcpyDeviceToHost);
	cudaDeviceSynchronize();
	for (int i = 0; i < N; i++) {
		CHECK(h_array[i] == i + 1);
	}
	cudaFree(d_array);
	delete[] h_array;
	return 0;
}

int test_cuda_add() {
	int N = 100;
	int* h_array_a = new int[N];
	int* h_array_b = new int[N];
	int* h_array_c = new int[N];
	int* d_array_a;
	int* d_array_b;
	int* d_array_c;
	cudaMalloc((void**)&d_array_a, N * sizeof(int));
	cudaMalloc((void**)&d_array_b, N * sizeof(int));
	cudaMalloc((void**)&d_array_c, N * sizeof(int));
	for (int i = 0; i < N; i++) {
		h_array_a[i] = i;
		h_array_b[i] = i * i;
	}
	cudaMemcpy(d_array_a, h_array_a, N * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_array_b, h_array_b, N * sizeof(int), cudaMemcpyHostToDevice);
	sail::cu::cuda_add(d_array_a, d_array_b, d_array_c, N);
	cudaMemcpy(h_array_c, d_array_c, N * sizeof(int), cudaMemcpyDeviceToHost);
	cudaDeviceSynchronize();
	for (int i = 0; i < N; i++) {
		CHECK(h_array_c[i] == i + i * i);
	}
	cudaFree(d_array_a);
	cudaFree(d_array_b);
	cudaFree(d_array_c);
	delete[] h_array_a;
	delete[] h_array_b;
	delete[] h_array_c;
	return 0;
}

}// namespace sail::test

TEST_CASE("cuda_dummy") {
	using namespace sail::test;
	CHECK(test_cuda_inc() == 0);
	CHECK(test_cuda_add() == 0);
}