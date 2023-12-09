#include "test_util.h"
#include <vector>

TEST_CASE("util") {
	constexpr int N = 10;
	std::vector<float> a(N);
	std::vector<float> b(N);
	std::vector<float> c(N - 1);
	for (int i = 0; i < N; ++i) {
		a[i] = static_cast<float>(i);
		b[i] = static_cast<float>(i);
	}
	CHECK(sail::test::float_span_equal(a, b));
	for (int i = 0; i < N - 1; ++i) {
		c[i] = static_cast<float>(i);
		if (i == 2) {
			b[i] = 3.0f;
		}
	}
	CHECK_FALSE(sail::test::float_span_equal(a, c));
	CHECK_FALSE(sail::test::float_span_equal(a, b));
}