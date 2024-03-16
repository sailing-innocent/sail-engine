#include "test_util.h"

namespace sail::test {

struct A {
	int x;
	int y;
};

struct B : public A {
	int z;
};
}// namespace sail::test

TEST_SUITE("basic::semantic") {
	TEST_CASE("class_inherit") {
		using namespace sail::test;
		A a{0, 1};
		B b{0, 1, 2};
		CHECK(a.x == 0);
		CHECK(a.y == 1);
		CHECK(b.x == 0);
		CHECK(b.y == 1);
		CHECK(b.z == 2);
	}
}
