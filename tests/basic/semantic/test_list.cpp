#include "test_util.h"
// test the native C++ list
#include <cmath>
TEST_SUITE("basic::semantic") {
	TEST_CASE("list_add") {
		int N = 1 << 20;// 1M elements
		float* x = new float[N];
		float* y = new float[N];

		for (int i = 0; i < N; i++) {
			x[i] = 1.0f;
			y[i] = 2.0f;
		}

		for (int i = 0; i < N; i++) {
			y[i] = x[i] + y[i];
		}

		// check errors
		float maxError = 0.0f;
		for (int i = 0; i < N; i++) {
			maxError = fmax(maxError, fabs(y[i] - 3.0f));
		}

		CHECK(maxError < 1e-5f);

		// free memory
		delete[] x;
		delete[] y;
	}
}
