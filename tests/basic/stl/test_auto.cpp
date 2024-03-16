#include "test_util.h"
#include <type_traits>

namespace sail::test {

struct Dummy {
	int var;
};

decltype(auto) func_0(Dummy& d) {
	return d.var;
}

decltype(auto) func_1(Dummy& d) {
	return (d.var);
}

int test_auto() {
	Dummy d{0};
	CHECK(std::is_same_v<decltype(func_0(d)), int>);
	CHECK(std::is_same_v<decltype(func_1(d)), int&>);
	return 0;
}

}// namespace sail::test

TEST_SUITE("basic::semantic") {
	TEST_CASE("cpp11_auto") {
		REQUIRE(sail::test::test_auto() == 0);
	}
}
