#include "test_util.h"

#include <tuple>

TEST_SUITE("basic::semantic") {
	TEST_CASE("tuple") {
		std::tuple<int, int> a{1, 2};
		REQUIRE(std::get<0>(a) == 1);
		REQUIRE(std::get<1>(a) == 2);
	}
}
