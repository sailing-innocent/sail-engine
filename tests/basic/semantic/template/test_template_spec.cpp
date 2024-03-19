#include "test_util.h"

namespace sail::test {

template<int N>
int Factorial() {
	return N * Factorial<N - 1>();
}

// case 1
template<>
int Factorial<0>() {
	return 1;
}

int test_factorial() {
	CHECK(Factorial<5>() == 120);
	CHECK(Factorial<0>() == 1);
	return 0;
}

}// namespace sail::test

TEST_SUITE("semantic::template") {
	TEST_CASE("template_spec") {
		REQUIRE(sail::test::test_factorial() == 0);
	}
}