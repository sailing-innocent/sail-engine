#include "test_util.h"
#include <string>
#include <cstddef>
#include <concepts>

namespace sail::test {

template<typename T>
concept always_satisfied = true;

template<typename T>
concept Hashable = requires(T a) {
	{
		std::hash<T>{}(a)
	} -> std::convertible_to<std::size_t>;
};

struct meow {
};

template<Hashable T>
void f(T) {
}

int test_hashable_success() {
	using namespace std::literals;
	using std::operator""s;
	try {
		f("abc"s);
	} catch (...) {
		return 1;
	}
	return 0;
}

}// namespace sail::test

TEST_SUITE("basic::semantic") {
	TEST_CASE("cpp20_concepts") {
		REQUIRE(sail::test::test_hashable_success() == 0);
	}
}
