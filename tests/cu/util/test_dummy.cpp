#include "SailCu/dummy.h"
#include "test_util.h"
#include "cuda_runtime_api.h"// cuMemcpy, cuMalloc

namespace sail::test {

int test_cuda_dummy() {
	int N = 10;
	int* h_array = new int[N];
	int* d_array;
	cudaMalloc((void**)&d_array, N * sizeof(int));
	for (int i = 0; i < N; i++) {
		h_array[i] = i;
	}
	cudaMemcpy(d_array, h_array, N * sizeof(int), cudaMemcpyHostToDevice);
	sail::cu::cuda_inc(d_array, N);
	cudaMemcpy(h_array, d_array, N * sizeof(int), cudaMemcpyDeviceToHost);
	for (int i = 0; i < N; i++) {
		CHECK(h_array[i] == i + 1);
	}
	return 0;
}

}// namespace sail::test

TEST_CASE("cuda_dummy") {
	CHECK(sail::test::test_cuda_dummy() == 0);
}