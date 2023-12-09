/**
 * @file test/render/test_scan.cpp
 * @author sailing-innocent
 * @brief Test Suite for Parallel Scan
 * @date 2023-12-28
 */

#include "test_util.h"
#include <luisa/runtime/context.h>
#include <luisa/runtime/device.h>
#include <luisa/runtime/stream.h>
#include <luisa/dsl/sugar.h>
#include <luisa/vstl/vector.h>

#include <numeric>
#include "SailInno/helper/device_parallel.h"

using namespace luisa;
using namespace luisa::compute;
namespace sail::inno::test {

int test_exclusive_scan(Device& device, int N) {
	luisa::vector<int> input_array(N);
	luisa::vector<int> gt_array(N);
	int init_value = 1;
	for (auto i = 0; i < N; ++i) {
		input_array[i] = i + 1;
		gt_array[i] = i * (i + 1) / 2 + init_value;
	}
	luisa::vector<int> result_array(N);

	std::exclusive_scan(input_array.begin(), input_array.end(), result_array.begin(), init_value);

	for (auto i = 0; i < N; ++i) {
		CHECK(result_array[i] == gt_array[i]);
		result_array[i] = 0;// clear to 0
	}

	auto stream = device.create_stream();
	auto input_buf = device.create_buffer<int>(N);
	auto result_buf = device.create_buffer<int>(N);
	// upload input data
	stream << input_buf.copy_from(input_array.data()) << synchronize();
	// upload init value to result buffer
	stream << result_buf.view(0, 1).copy_from(&init_value) << synchronize();

	DeviceParallel dp{};
	dp.create(device);

	size_t temp_space_size;
	dp.scan_exclusive_sum(temp_space_size, input_buf, result_buf, init_value, N);
	LUISA_INFO("temp_space_size: {}", temp_space_size);
	auto temp_buf = device.create_buffer<int>(temp_space_size);

	CommandList cmdlist;

	dp.scan_exclusive_sum(cmdlist, temp_buf, input_buf, result_buf, init_value, N);

	stream << cmdlist.commit() << synchronize();

	// download result data
	stream << result_buf.copy_to(result_array.data()) << synchronize();

	for (auto i = 0; i < N; ++i) {
		CHECK(result_array[i] == gt_array[i]);
	}
	// for (auto i = 20; i < 40; i++) {
	// 	LUISA_INFO("result_array[{}]: {}", i, result_array[i]);
	// 	LUISA_INFO("gt_array[{}]: {}", i, gt_array[i]);
	// }
	return 0;
}

int test_inclusive_scan(Device& device, int N) {
	luisa::vector<int> input_array(N);
	luisa::vector<int> gt_array(N);
	for (auto i = 0; i < N; ++i) {
		input_array[i] = i;
		gt_array[i] = i * (i + 1) / 2;
	}
	luisa::vector<int> result_array(N);
	std::inclusive_scan(input_array.begin(), input_array.end(), result_array.begin());

	for (auto i = 0; i < N; ++i) {
		CHECK(result_array[i] == gt_array[i]);
		result_array[i] = 0;// clear to 0
	}

	return 0;
}

}// namespace sail::inno::test

TEST_SUITE("parallel") {
	TEST_CASE("scan") {
		Context context{sail::test::argv()[0]};
		auto device = context.create_device("dx");
		REQUIRE(sail::inno::test::test_inclusive_scan(device, 10) == 0);
		REQUIRE(sail::inno::test::test_exclusive_scan(device, 10) == 0);
		// REQUIRE(inno::test::test_exclusive_scan(device, 1000) == 0);
	}
}