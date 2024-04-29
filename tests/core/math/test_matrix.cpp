#include "test_util.h"

#include "SailMath/type/matrix.hpp"
#include <EASTL/array.h>
#include <EASTL/numeric.h>

namespace sail::test {

template<typename T, size_t M, size_t N>
int test_matrix() {
	eastl::array<eastl::array<T, N>, M> data1;
	for (size_t i = 0; i < M; ++i) {
		eastl::iota(data1[i].begin(), data1[i].end(), i * N);
	}
	eastl::array<eastl::array<T, N>, M> data2;
	for (size_t i = 0; i < M; ++i) {
		eastl::iota(data2[i].begin(), data2[i].end(), (i + 1) * N);
	}

	Matrix<T, M, N> m1(data1);
	Matrix<T, M, N> m2(data2);
	// add
	Matrix<T, M, N> m3 = m1 + m2;
	for (size_t i = 0; i < M; ++i) {
		for (size_t j = 0; j < N; ++j) {
			CHECK(m3[i][j] == doctest::Approx(data1[i][j] + data2[i][j]));
		}
	}
	// minus
	m3 = m1 - m2;
	for (size_t i = 0; i < M; ++i) {
		for (size_t j = 0; j < N; ++j) {
			CHECK(m3[i][j] == doctest::Approx(data1[i][j] - data2[i][j]));
		}
	}
	// scalar multiply
	m3 = m1 * 2;
	for (size_t i = 0; i < M; ++i) {
		for (size_t j = 0; j < N; ++j) {
			CHECK(m3[i][j] == doctest::Approx(data1[i][j] * 2));
		}
	}
	// scalar divide
	m3 = m1 / 2;
	for (size_t i = 0; i < M; ++i) {
		for (size_t j = 0; j < N; ++j) {
			CHECK(m3[i][j] == doctest::Approx(data1[i][j] / 2));
		}
	}
	// matrix multiply
	if constexpr (M == N) {
		Matrix<T, M, N> m4 = m1 * m2;
		for (size_t i = 0; i < M; ++i) {
			for (size_t j = 0; j < N; ++j) {
				T sum = 0;
				for (size_t k = 0; k < N; ++k) {
					sum += data1[i][k] * data2[k][j];
				}
				CHECK(m4[i][j] == doctest::Approx(sum));
			}
		}
	}

	return 0;
}

}// namespace sail::test

using namespace sail::test;
TEST_SUITE("core::math") {
	TEST_CASE("matrix") {
		CHECK(test_matrix<float, 3, 3>() == 0);
	}
}
