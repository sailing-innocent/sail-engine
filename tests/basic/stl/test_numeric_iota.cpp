#include "test_util.h"

#include <vector>
#include <numeric>

TEST_SUITE("basic::numeric") {
	TEST_CASE("iota") {
		std::vector<int> a;
		a.resize(3);
		std::iota(a.begin(), a.end(), 0);
		REQUIRE(a[0] == 0);
		REQUIRE(a[1] == 1);
		REQUIRE(a[2] == 2);
	}
}
