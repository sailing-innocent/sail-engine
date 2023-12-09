#include "test_util.h"

#include <cmath>

TEST_SUITE("basic::semantic") {
	TEST_CASE("cmath_calc") {
		float PI = 3.1415926f;
		REQUIRE(abs(tanf(45.0f / 180.0f * PI) - 1.0f) < 0.001f);
		REQUIRE(abs(tanf(30.0f / 180.0f * PI) - 0.57735026919f) < 0.001f);
	}
}
