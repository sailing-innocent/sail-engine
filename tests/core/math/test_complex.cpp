#include "test_util.h"
#include "SailMath/type/complex.hpp"

namespace sail::test {

template<typename T>
int test_complex() {
	Complex<T> a(1, 2);
	Complex<T> b(3, 4);
	// add
	Complex<T> c = a + b;
	CHECK(a.real() == doctest::Approx(1));
	CHECK(a.imag() == doctest::Approx(2));
	CHECK(b.real() == doctest::Approx(3));
	CHECK(b.imag() == doctest::Approx(4));
	CHECK(c.real() == doctest::Approx(4));
	CHECK(c.imag() == doctest::Approx(6));
	// minus
	c = a - b;
	CHECK(c.real() == doctest::Approx(-2));
	CHECK(c.imag() == doctest::Approx(-2));
	// multiply
	c = a * b;
	CHECK(c.real() == doctest::Approx(-5));
	CHECK(c.imag() == doctest::Approx(10));
	// divide
	c = a / b;
	CHECK(c.real() == doctest::Approx(0.44));
	CHECK(c.imag() == doctest::Approx(0.08));
	// norm
	CHECK(a.norm() == doctest::Approx(5));
	CHECK(b.norm() == doctest::Approx(25));
	return 0;
}

}// namespace sail::test

TEST_SUITE("core::math") {
	TEST_CASE("complex") {
		using namespace sail::test;
		REQUIRE(test_complex<float>() == 0);
		REQUIRE(test_complex<double>() == 0);
	}
}