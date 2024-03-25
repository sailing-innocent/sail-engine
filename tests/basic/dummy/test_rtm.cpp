#include "test_util.h"
#include "SailBase/math/vector4f.h"

TEST_CASE("rtm") {
	sail_float4_t a = {1.0f, 2.0f, 3.0f, 4.0f};
	CHECK(a.x == 1.0f);
	CHECK(sizeof(sail_float4_t) == 16);
	rtm::vector4f v1 = sail::math::load(a);
	rtm::vector4f v2 = {9.0f, 3.0f, 3.0f, 1.0f};

	auto v3 = v1 + v2;
	sail_float4_t result;
	sail::math::store(v3, result);
	CHECK(result.x == 10.0f);
	CHECK(result.y == 5.0f);
	CHECK(result.z == 6.0f);
	CHECK(result.w == 5.0f);
}