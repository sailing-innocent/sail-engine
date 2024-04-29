#include "test_util.h"

/**
 * @file test_rtm_matrix.cpp
 * @brief The test suite for RTM math
 * @author sailing-innocent
 * @date 2024-04-29
 */

#include "SailMath/rtm_math.hpp"

namespace sail::test {

int test_rtm_matrix() {
	// row major
	Mat4f m1 = Mat4f{1.0f, 2.0f, 3.0f, 4.0f,
					 5.0f, 6.0f, 7.0f, 8.0f,
					 9.0f, 10.0f, 11.0f, 12.0f,
					 13.0f, 14.0f, 15.0f, 16.0f};
	// fetch
	CHECK(m1(0, 0) == doctest::Approx(1.0f));
	CHECK(m1(0, 1) == doctest::Approx(5.0f));
	CHECK(m1(0, 2) == doctest::Approx(9.0f));
	CHECK(m1(0, 3) == doctest::Approx(13.0f));
	CHECK(m1(1, 0) == doctest::Approx(2.0f));
	CHECK(m1(1, 1) == doctest::Approx(6.0f));
	CHECK(m1(1, 2) == doctest::Approx(10.0f));
	CHECK(m1(1, 3) == doctest::Approx(14.0f));
	CHECK(m1(2, 0) == doctest::Approx(3.0f));
	CHECK(m1(2, 1) == doctest::Approx(7.0f));
	CHECK(m1(2, 2) == doctest::Approx(11.0f));
	CHECK(m1(2, 3) == doctest::Approx(15.0f));
	CHECK(m1(3, 0) == doctest::Approx(4.0f));
	CHECK(m1(3, 1) == doctest::Approx(8.0f));
	CHECK(m1(3, 2) == doctest::Approx(12.0f));
	CHECK(m1(3, 3) == doctest::Approx(16.0f));

	Vec4f v1 = Vec4f{1.0f, 2.0f, 3.0f, 4.0f};// row vector
	Vec4f v2 = v1 * m1;
	v2.download();
	CHECK(v2.x() == doctest::Approx(90.0f));
	CHECK(v2.y() == doctest::Approx(100.0f));
	CHECK(v2.z() == doctest::Approx(110.0f));
	CHECK(v2.w() == doctest::Approx(120.0f));

	Mat4f m2 = Mat4f{1.0f, 2.0f, 1.0f, 2.0f,
					 3.0f, 4.0f, 3.0f, 4.0f,
					 1.0f, 2.0f, 1.0f, 2.0f,
					 3.0f, 4.0f, 3.0f, 4.0f};
	Mat4f m3 = m1 * m2;
	m3.download();
	CHECK(m3(0, 0) == doctest::Approx(22.0f));
	CHECK(m3(0, 1) == doctest::Approx(54.0f));
	CHECK(m3(0, 2) == doctest::Approx(86.0f));
	CHECK(m3(0, 3) == doctest::Approx(118.0f));
	CHECK(m3(1, 0) == doctest::Approx(32.0f));
	CHECK(m3(1, 1) == doctest::Approx(80.0f));
	CHECK(m3(1, 2) == doctest::Approx(128.0f));
	CHECK(m3(1, 3) == doctest::Approx(176.0f));
	CHECK(m3(2, 0) == doctest::Approx(22.0f));
	CHECK(m3(2, 1) == doctest::Approx(54.0f));
	CHECK(m3(2, 2) == doctest::Approx(86.0f));
	CHECK(m3(2, 3) == doctest::Approx(118.0f));
	CHECK(m3(3, 0) == doctest::Approx(32.0f));
	CHECK(m3(3, 1) == doctest::Approx(80.0f));
	CHECK(m3(3, 2) == doctest::Approx(128.0f));
	CHECK(m3(3, 3) == doctest::Approx(176.0f));

	Mat4f m4 = m2 * m1;
	m4.download();
	CHECK(m4(0, 0) == doctest::Approx(46.0f));
	CHECK(m4(0, 1) == doctest::Approx(102.0f));
	CHECK(m4(0, 2) == doctest::Approx(46.0f));
	CHECK(m4(0, 3) == doctest::Approx(102.0f));
	CHECK(m4(1, 0) == doctest::Approx(52.0f));
	CHECK(m4(1, 1) == doctest::Approx(116.0f));
	CHECK(m4(1, 2) == doctest::Approx(52.0f));
	CHECK(m4(1, 3) == doctest::Approx(116.0f));
	CHECK(m4(2, 0) == doctest::Approx(58.0f));
	CHECK(m4(2, 1) == doctest::Approx(130.0f));
	CHECK(m4(2, 2) == doctest::Approx(58.0f));
	CHECK(m4(2, 3) == doctest::Approx(130.0f));
	CHECK(m4(3, 0) == doctest::Approx(64.0f));
	CHECK(m4(3, 1) == doctest::Approx(144.0f));
	CHECK(m4(3, 2) == doctest::Approx(64.0f));
	CHECK(m4(3, 3) == doctest::Approx(144.0f));
	return 0;
}

}// namespace sail::test

TEST_SUITE("core::math") {
	TEST_CASE("rtm_matrix") {
		using namespace sail::test;
		REQUIRE(test_rtm_matrix() == 0);
	}
}