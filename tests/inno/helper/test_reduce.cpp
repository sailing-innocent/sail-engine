/**
 * @file test/render/test_reduce.cpp
 * @author sailing-innocent
 * @brief Test Suite for Parallel Reduce
 * @date 2024-04-14
 */

#include "SailInno/core/runtime.h"
#include "test_util.h"

#include <luisa/runtime/context.h>
#include <luisa/runtime/device.h>
#include <luisa/runtime/stream.h>
#include "luisa/dsl/sugar.h"
#include "luisa/runtime/command_list.h"
#include "luisa/core/stl/unordered_map.h"

#include <type_traits>

using namespace luisa;
using namespace luisa::compute;
namespace sail::test {

template<typename T>
static constexpr bool is_numeric_v = std::is_integral_v<T> || std::is_floating_point_v<T>;
template<typename T>
concept NumericT = is_numeric_v<T>;

class Dummy : public inno::LuisaModule {
public:
	template<NumericT T>
	void compile(Device& device) {
		luisa::string_view key = Type::of<T>()->description();
		if (!key.empty()) {
			luisa::unique_ptr<Shader<1, Buffer<T>, int>> shad_ptr = nullptr;
			inno::lazy_compile(device, shad_ptr, [](BufferVar<T> buf, Int N) {
				auto idx = dispatch_id().x;
				$if(idx < N) {
					buf->write(idx, buf->read(idx) + static_cast<T>(1));
				};
			});

			shad_map.try_emplace(key, std::move(shad_ptr));
		}
	}

	template<NumericT T>
	void run(CommandList& cmdlist, BufferView<T> buf) {
		luisa::string_view key = Type::of<T>()->description();
		if (!key.empty()) {
			size_t N = buf.size();
			auto* shad_it = shad_map.find(key);
			if (shad_it != shad_map.end()) {
				LUISA_INFO("dispatching fetched! {}", N);
				auto& shad_ptr = shad_it->second;
				cmdlist << (*reinterpret_cast<Shader<1, Buffer<T>, int>*>(&(*shad_ptr)))(buf, N).dispatch(N);
			} else {
				LUISA_INFO("dispatching NOT fetched!");
			}
		}
	}

private:
	// shader
	luisa::unordered_map<luisa::string, luisa::unique_ptr<Resource>> shad_map;
};

template<NumericT T>
int test_reduce(Device& device) {
	constexpr int N = 100;
	auto d_arr = device.create_buffer<T>(N);
	luisa::vector<T> h_arr;
	h_arr.resize(N);
	for (auto i = 0; i < N; i++) {
		h_arr[i] = static_cast<T>(i);
	}

	Dummy dummy{};
	dummy.compile<T>(device);

	auto stream = device.create_stream();
	CommandList cmdlist;
	cmdlist << d_arr.copy_from(h_arr.begin());
	dummy.run<T>(cmdlist, d_arr.view());		// increment
	cmdlist << d_arr.copy_to(h_arr.begin());	// copy back
	stream << cmdlist.commit() << synchronize();// sync

	for (auto i = 0; i < N; i++) {
		CHECK(h_arr[i] == doctest::Approx(static_cast<T>(i + 1)));
	}
	return 0;
}
}// namespace sail::test

TEST_SUITE("parallel_primitive") {
	TEST_CASE("reduce") {
		Context context{sail::test::argv()[0]};
		auto device = context.create_device("dx");
		REQUIRE(sail::test::test_reduce<int>(device) == 0);
		REQUIRE(sail::test::test_reduce<float>(device) == 0);
	}
}