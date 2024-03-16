#include "test_util.h"

#include <EASTL/array.h>

TEST_CASE("eastl_array") {
	eastl::array<int, 3> test_arr{0, 1, 2};
	for (auto i = 0; i < 3; i++) {
		CHECK(test_arr[i] == i);
	}
}