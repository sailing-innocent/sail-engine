#include "test_util.h"

namespace sail::test {

int test_template_special() {
	auto _ = 1;
	auto b = 0 ^ _ ^ 0;
	CHECK(b == 0);
	return 0;
}

}// namespace sail::test

TEST_SUITE("semantic::template") {
	TEST_CASE("template_special") {
		REQUIRE(sail::test::test_template_special() == 0);
	}
}