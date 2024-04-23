/**
 * @file test_stride_loop.cpp
 * @author sailing-innocent
 * @brief Tester for Grid Stride Loop
 * @date 2023-09-15
 */

#include "test_util.h"
#include "luisa/luisa-compute.h"
#include "SailInno/helper/grid_stride_loop.h"
#include <vector>

using namespace luisa;
using namespace luisa::compute;

namespace sail::test {

int grid_stride_loop_test(Device& device) {
	std::vector<int> gt, res;
	auto stream = device.create_stream();
	constexpr auto count = 64;
	gt.resize(count);
	res.resize(count);
	for (auto i = 0; i < count; ++i) {
		gt[i] = i;
	}

	auto d_in = device.create_buffer<int>(count);

	auto iota = device.compile<1>([&] {
		set_block_size(64);
		sail::inno::grid_stride_loop(
			count, [&](Int i) {
			d_in->write(i, i);
		});
	});
	stream << iota().dispatch(64) << synchronize()
		   << d_in.copy_to(res.data()) << synchronize();

	for (auto i = 0; i < count; ++i) {
		CHECK(res[i] == gt[i]);
	}

	return 0;
}

}// namespace sail::test

TEST_SUITE("helper") {
	TEST_CASE("grid_stride_loop") {
		Context context{sail::test::argv()[0]};
		Device device = context.create_device("dx");
		REQUIRE(sail::test::grid_stride_loop_test(device) == 0);
	}
}