/**
 * @file test/inno/helper/test_buffer_filler.cpp
 * @author sailing-innocent
 * @brief Test Suite for Buffer Filler
 * @date 2023-12-27
 */

#include "test_util.h"

#include "SailInno/helper/buffer_filler.h"

#include <luisa/runtime/context.h>
#include <luisa/runtime/device.h>
#include <luisa/runtime/stream.h>
#include <luisa/dsl/sugar.h>

using namespace luisa;
using namespace luisa::compute;
namespace sail::inno::test {

template<typename T>
int test_buffer_filler(Device& device, T fill_v) {
	constexpr int N = 10;
	inno::BufferFiller bf{};
	auto buffer = device.create_buffer<T>(N);
	auto stream = device.create_stream(StreamTag::GRAPHICS);
	stream << bf.fill(device, buffer, fill_v) << synchronize();
	auto buffer_data = luisa::vector<T>(N);
	stream << buffer.copy_to(buffer_data.data()) << synchronize();

	for (auto i = 0u; i < N; ++i) {
		CHECK(buffer_data[i] == doctest::Approx(fill_v));
	}

	return 0;
}

}// namespace sail::inno::test

TEST_SUITE("helper") {
	TEST_CASE("buffer_filler") {
		Context context{sail::test::argv()[0]};
		auto device = context.create_device("dx");
		REQUIRE(sail::inno::test::test_buffer_filler(device, 1.0f) == 0);
		REQUIRE(sail::inno::test::test_buffer_filler(device, 1) == 0);
		REQUIRE(sail::inno::test::test_buffer_filler(device, 1u) == 0);
	}
}