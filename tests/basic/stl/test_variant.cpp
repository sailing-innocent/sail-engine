#include <test_util.h>

#include <variant>
#include <vector>

namespace sail::test {

struct Circle {
	int id() { return 0; }
};
struct Square {
	int id() { return 1; }
};
struct Triangle {
	int id() { return 2; }
};
using Shape = std::variant<Circle, Square, Triangle>;

struct GenericInvoker {
	template<typename T>
	int operator()(T& shape) {
		return shape.id();
	}
};

int test_variant() {
	std::vector<Shape> shapes{Circle{}, Square{}, Triangle{}};
	std::vector<int> ids{0, 1, 2};

	for (auto i = 0; i < shapes.size(); ++i) {
		CHECK(std::visit(GenericInvoker{}, shapes[i]) == ids[i]);
	}
	return 0;
}

}// namespace sail::test

TEST_SUITE("basic::semantic") {
	TEST_CASE("cpp17_variant") {
		REQUIRE(sail::test::test_variant() == 0);
	}
}