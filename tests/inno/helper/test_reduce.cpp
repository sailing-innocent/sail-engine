/**
 * @file test/render/test_scan.cpp
 * @author sailing-innocent
 * @brief Test Suite for Parallel Reduce
 * @date 2024-14
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
#include <functional>

using namespace luisa;
using namespace luisa::compute;

namespace sail::inno::test {

template<typename T>
static constexpr bool is_numeric_v = std::is_integral_v<T> || std::is_floating_point_v<T>;

template<typename T>
concept NumericT = is_numeric_v<T>;

enum class ReduceType : int {
	SUM = 0,
	MAX = 1,
	MIN = 2
};

// min op
template<typename T>
struct min_op : public std::function<T(T, T)> {
	T operator()(const T& x, const T& y) const {
		return std::min(x, y);
	}
};

// max op
template<typename T>
struct max_op : public std::function<T(T, T)> {
	T operator()(const T& x, const T& y) const {
		return std::max(x, y);
	}
};

// add op
template<typename T>
struct add_op : public std::function<T(T, T)> {
	T operator()(const T& x, const T& y) const {
		return x + y;
	}
};

template<NumericT T>
int test_reduce(Device& device, int N, ReduceType r_type) {
	int N_EXP = 100;
	luisa::vector<T> input_array(N);
	T gt_value = static_cast<T>(0);
	T init_value = static_cast<T>(0);
	for (auto i = 0; i < N; ++i) {
		input_array[i] = static_cast<T>(i);
	}
	gt_value = static_cast<T>(N * (N - 1) / 2);
	if (r_type == ReduceType::MIN) {
		gt_value = static_cast<T>(0);
	} else if (r_type == ReduceType::MAX) {
		gt_value = static_cast<T>(N - 1);
	}
	T result_value = static_cast<T>(0);
	Clock clk;
	clk.tic();
	for (auto j = 0; j < N_EXP; ++j) {
		std::function<T(T, T)> std_op = add_op<T>();
		if (r_type == ReduceType::MIN) {
			std_op = min_op<T>();
		} else if (r_type == ReduceType::MAX) {
			std_op = max_op<T>();
		}

		result_value = std::reduce(input_array.begin(), input_array.end(), init_value, std_op);
	}
	LUISA_INFO("CPU time: {}", clk.toc() / N_EXP);
	CHECK(result_value == doctest::Approx(gt_value));
	result_value = static_cast<T>(0);

	auto stream = device.create_stream();
	auto input_buf = device.create_buffer<T>(N);
	auto result_buf = device.create_buffer<T>(1);
	clk.tic();
	stream << input_buf.copy_from(input_array.data());
	DeviceParallel dp{};
	dp.create(device);
	size_t temp_space_size;
	dp.reduce<T>(temp_space_size, input_buf, result_buf, N);
	auto temp_buf = device.create_buffer<T>(temp_space_size);
	CommandList cmdlist;
	for (auto j = 0; j < N_EXP; ++j) {
		dp.reduce<T>(cmdlist, temp_buf, input_buf, result_buf, N, static_cast<int>(r_type));
		stream << cmdlist.commit();
	}
	stream << synchronize();
	stream << result_buf.copy_to(&result_value) << synchronize();

	LUISA_INFO("GPU time: {}", clk.toc() / N_EXP);
	CHECK(result_value == doctest::Approx(gt_value));
	return 0;
}

}// namespace sail::inno::test

TEST_SUITE("parallel_primitive") {
	TEST_CASE("reduce") {
		using namespace sail::inno::test;
		Context context{sail::test::argv()[0]};
		auto device = context.create_device("cuda");
		REQUIRE(test_reduce<int>(device, 10, ReduceType::SUM) == 0);
		REQUIRE(test_reduce<int>(device, 1000, ReduceType::SUM) == 0);
		REQUIRE(test_reduce<int>(device, 1e5, ReduceType::SUM) == 0);
		REQUIRE(test_reduce<float>(device, 10, ReduceType::SUM) == 0);
		REQUIRE(test_reduce<float>(device, 1000, ReduceType::SUM) == 0);

		REQUIRE(test_reduce<int>(device, 10, ReduceType::MIN) == 0);
		REQUIRE(test_reduce<int>(device, 1000, ReduceType::MIN) == 0);
		REQUIRE(test_reduce<int>(device, 1e5, ReduceType::MIN) == 0);
		REQUIRE(test_reduce<float>(device, 10, ReduceType::MIN) == 0);
		REQUIRE(test_reduce<float>(device, 1000, ReduceType::MIN) == 0);

		REQUIRE(test_reduce<int>(device, 10, ReduceType::MAX) == 0);
		REQUIRE(test_reduce<int>(device, 1000, ReduceType::MAX) == 0);
		REQUIRE(test_reduce<int>(device, 1e5, ReduceType::MAX) == 0);
		REQUIRE(test_reduce<float>(device, 10, ReduceType::MAX) == 0);
		REQUIRE(test_reduce<float>(device, 1000, ReduceType::MAX) == 0);
	}
}