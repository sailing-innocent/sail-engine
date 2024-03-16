/**
 * @file test/helper/test_reduce.cpp
 * @author sailing-innocent
 * @brief Test Suite for Parallel Reduce
 * @date 2023-12-28
 */

#include "test_util.h"
#include <luisa/runtime/context.h>
#include <luisa/runtime/device.h>
#include <luisa/runtime/stream.h>
#include <luisa/dsl/sugar.h>

#include <numeric>
using namespace luisa;
using namespace luisa::compute;
namespace sail::test {

int test_parallel_reduce(Device& device, int N) {
	luisa::vector<int> input_array(N);
	for (auto i = 0; i < N; ++i) {
		input_array[i] = i;
	}
	int gt = N * (N - 1) / 2;
	int result = std::reduce(input_array.begin(), input_array.end(), 0.0, std::plus<int>());
	CHECK(result == gt);

	return 0;
}

}// namespace sail::test

TEST_SUITE("parallel") {
	TEST_CASE("reduce") {
		Context context{sail::test::argv()[0]};
		auto device = context.create_device("dx");
		REQUIRE(sail::test::test_parallel_reduce(device, 10) == 0);
	}
}