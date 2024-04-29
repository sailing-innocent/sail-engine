#include "test_util.h"

/**
 * @file test_rtm_vector.cpp
 * @brief The test suite for RTM math
 * @author sailing-innocent
 * @date 2024-04-29
 */
#include "SailMath/rtm_math.hpp"

namespace sail::test {

int test_rtm_vector() {
	Vec4f v1 = Vec4f{1.0f, 2.0f, 3.0f, 4.0f};
	CHECK(v1.x() == doctest::Approx(1.0f));
	CHECK(v1.y() == doctest::Approx(2.0f));
	CHECK(v1.z() == doctest::Approx(3.0f));
	CHECK(v1.w() == doctest::Approx(4.0f));
	Vec4f v2 = Vec4f{5.0f, 6.0f, 7.0f, 8.0f};
	Vec4f v3 = v1 + v2;
	v3.download();
	CHECK(v3.x() == doctest::Approx(6.0f));
	CHECK(v3.y() == doctest::Approx(8.0f));
	CHECK(v3.z() == doctest::Approx(10.0f));
	CHECK(v3.w() == doctest::Approx(12.0f));

	Vec4f v4 = v1 - v2;
	v4.download();
	CHECK(v4.x() == doctest::Approx(-4.0f));
	CHECK(v4.y() == doctest::Approx(-4.0f));
	CHECK(v4.z() == doctest::Approx(-4.0f));
	CHECK(v4.w() == doctest::Approx(-4.0f));

	Vec4f v5 = v1 * v2;
	v5.download();
	CHECK(v5.x() == doctest::Approx(5.0f));
	CHECK(v5.y() == doctest::Approx(12.0f));
	CHECK(v5.z() == doctest::Approx(21.0f));
	CHECK(v5.w() == doctest::Approx(32.0f));

	Vec4f v6 = v1 / v2;
	v6.download();
	CHECK(v6.x() == doctest::Approx(0.2f));
	CHECK(v6.y() == doctest::Approx(0.3333333333333333f));
	CHECK(v6.z() == doctest::Approx(0.42857142857142855f));
	CHECK(v6.w() == doctest::Approx(0.5f));

	return 0;
}

}// namespace sail::test

TEST_SUITE("core::math") {
	TEST_CASE("rtm_vector") {
		using namespace sail::test;
		REQUIRE(test_rtm_vector() == 0);
	}
}