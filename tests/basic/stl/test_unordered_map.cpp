// https://en.cppreference.com/w/cpp/container/unordered_map/unordered_map
#include "test_util.h"
#include <unordered_map>

TEST_SUITE("basic::containers") {
	TEST_CASE("unordered_map") {
		std::unordered_map<int, int> m;
		m[1] = 2;
		m[2] = 3;
		REQUIRE(m[1] == 2);
		REQUIRE(m[2] == 3);
	}
}