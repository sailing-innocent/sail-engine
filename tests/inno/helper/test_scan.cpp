/**
 * @file test/render/test_scan.cpp
 * @author sailing-innocent
 * @brief Test Suite for Parallel Scan
 * @date 2024-04-14
 */

#include "test_util.h"
#include <luisa/runtime/context.h>
#include <luisa/runtime/device.h>
#include <luisa/runtime/stream.h>
#include <luisa/core/clock.h>
#include <luisa/dsl/sugar.h>
#include <luisa/vstl/vector.h>

#include <numeric>
#include <type_traits>
#include "SailInno/helper/device_parallel.h"

using namespace luisa;
using namespace luisa::compute;
namespace sail::inno::test {

template<typename T>
static constexpr bool is_numeric_v = std::is_integral_v<T> || std::is_floating_point_v<T>;

template<typename T>
concept NumericT = is_numeric_v<T>;

template<NumericT T>
int test_exclusive_scan(Device& device, int N) {
	int N_EXP = 100;
	luisa::vector<T> input_array(N);
	luisa::vector<T> gt_array(N);
	int init_value = 0;
	for (auto i = 0; i < N; ++i) {
		input_array[i] = static_cast<T>(1);
		gt_array[i] = i;
	}
	luisa::vector<T> result_array(N);
	// fill 0
	std::fill(result_array.begin(), result_array.end(), static_cast<T>(0));

	Clock clk;
	clk.tic();
	for (auto j = 0; j < N_EXP; ++j) {
		std::exclusive_scan(input_array.begin(), input_array.end(), result_array.begin(), init_value);
	}
	LUISA_INFO("CPU time: {}", clk.toc() / N_EXP);

	for (auto i = 0; i < N; ++i) {
		CHECK(result_array[i] == doctest::Approx(gt_array[i]));
		result_array[i] = static_cast<T>(0);// clear to 0
	}

	auto stream = device.create_stream();
	auto input_buf = device.create_buffer<T>(N);
	auto result_buf = device.create_buffer<T>(N);
	clk.tic();
	// upload input data
	stream << input_buf.copy_from(input_array.data()) << synchronize();
	// upload init value to result buffer
	stream << result_buf.view(0, 1).copy_from(&init_value) << synchronize();

	DeviceParallel dp{};
	dp.create(device);

	size_t temp_space_size;
	dp.scan_exclusive_sum<T>(temp_space_size, input_buf, result_buf, init_value, N);
	LUISA_INFO("temp_space_size: {}", temp_space_size);
	auto temp_buf = device.create_buffer<T>(temp_space_size);

	CommandList cmdlist;
	for (auto j = 0; j < N_EXP; ++j) {
		dp.scan_exclusive_sum<T>(cmdlist, temp_buf, input_buf, result_buf, init_value, N);
		stream << cmdlist.commit();
	}
	stream << synchronize();
	// download result data
	stream << result_buf.copy_to(result_array.data()) << synchronize();

	LUISA_INFO("GPU time: {}", clk.toc() / N_EXP);

	for (auto i = 0; i < N; ++i) {
		CHECK(result_array[i] == gt_array[i]);
	}
	return 0;
}

}// namespace sail::inno::test

TEST_SUITE("parallel_primitive") {
	TEST_CASE("scan") {
		Context context{sail::test::argv()[0]};
		auto device = context.create_device("cuda");
		REQUIRE(sail::inno::test::test_exclusive_scan<int>(device, 10) == 0);
		REQUIRE(sail::inno::test::test_exclusive_scan<int>(device, 1000) == 0);
		REQUIRE(sail::inno::test::test_exclusive_scan<int>(device, 1e6) == 0);
		REQUIRE(sail::inno::test::test_exclusive_scan<float>(device, 10) == 0);
		REQUIRE(sail::inno::test::test_exclusive_scan<float>(device, 1000) == 0);
		REQUIRE(sail::inno::test::test_exclusive_scan<float>(device, 1e6) == 0);
		// REQUIRE(sail::inno::test::test_exclusive_scan<float>(device, 10) == 0);
	}
}