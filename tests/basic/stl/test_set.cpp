#include "test_util.h"

#include <set>

TEST_SUITE("basic") {
	TEST_CASE("set_insert") {
		std::set<int> s = {1, 2};
		REQUIRE(s.size() == 2);
		s.insert(3);
		REQUIRE(s == std::set<int>{1, 2, 3});
	}
}
