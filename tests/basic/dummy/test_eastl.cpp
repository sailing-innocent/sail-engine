#include "test_util.h"

#include <array>

TEST_CASE("eastl_array") {
	std::array<int, 3> test_arr{0, 1, 2};
	for (auto i = 0; i < 3; i++) {
		CHECK(test_arr[i] == i);
	}
}