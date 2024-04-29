#include "test_util.h"
#include "SailMath/type/vector.hpp"
#include <EASTL/array.h>
#include <EASTL/numeric.h>

namespace sail::test {

template<typename T, size_t N>
int test_vector() {
	eastl::array<T, N> data1;
	eastl::iota(data1.begin(), data1.end(), 0);
	eastl::array<T, N> data2;
	eastl::iota(data2.begin(), data2.end(), 1);

	Vector<T, N> v1(data1);
	Vector<T, N> v2(data2);
	// add
	Vector<T, N> v3 = v1 + v2;
	for (size_t i = 0; i < N; ++i) {
		CHECK(v3[i] == doctest::Approx(data1[i] + data2[i]));
	}
	// minus
	v3 = v1 - v2;
	for (size_t i = 0; i < N; ++i) {
		CHECK(v3[i] == doctest::Approx(data1[i] - data2[i]));
	}
	// scalar multiply
	v3 = v1 * 2;
	for (size_t i = 0; i < N; ++i) {
		CHECK(v3[i] == doctest::Approx(data1[i] * 2));
	}
	// scalar divide
	v3 = v1 / 2;
	for (size_t i = 0; i < N; ++i) {
		CHECK(v3[i] == doctest::Approx(data1[i] / 2));
	}
	// dot product
	T result = dot(v1, v2);
	T expect = eastl::inner_product(data1.begin(), data1.end(), data2.begin(), static_cast<T>(0));
	CHECK(result == doctest::Approx(expect));

	// cross product
	if constexpr (N == 3) {
		Vector<T, N> v4 = cross(v1, v2);
		CHECK(v4[0] == doctest::Approx(data1[1] * data2[2] - data1[2] * data2[1]));
		CHECK(v4[1] == doctest::Approx(data1[2] * data2[0] - data1[0] * data2[2]));
		CHECK(v4[2] == doctest::Approx(data1[0] * data2[1] - data1[1] * data2[0]));
	}

	return 0;
}

}// namespace sail::test

using namespace sail::test;
TEST_SUITE("core::math") {
	TEST_CASE("vector") {
		CHECK(test_vector<float, 3>() == 0);
	}
}