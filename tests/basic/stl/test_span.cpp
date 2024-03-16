// cpp20 feature span: https://en.cppreference.com/w/cpp/container/span
#include "test_util.h"

#include <algorithm>
#include <span>
#include <array>

namespace sail::test {

int test_span() {
	std::array<int, 5> arr{1, 2, 3, 4, 5};
	std::span<int> sp1(arr);
	std::span<int> sp2(sp1.subspan(1, sp1.size() - 2));// {2, 3, 4}
	REQUIRE(sp2.size() == 3);
	std::transform(sp2.begin(), sp2.end(),
				   sp2.begin(), [](int i) { return i * i; });

	for (auto i = 0; i < 5; i++) {
		if (i > 0 && i < 4) {
			REQUIRE(arr[i] == (i + 1) * (i + 1));
		} else {
			REQUIRE(arr[i] == i + 1);
		}
	}
	return 0;
}

}// namespace sail::test

TEST_SUITE("basic::semantic") {
	TEST_CASE("cpp20_span") {
		REQUIRE(sail::test::test_span() == 0);
	}
}
