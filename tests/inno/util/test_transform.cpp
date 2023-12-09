/**
 * @file test/basic/test_transform.cpp
 * @author sailing-innocent
 * @brief Math Transform Tester
 * @date 2023-01-18
 */

#include "test_util.h"
#include "SailInno/util/math/transform.h"
#include <luisa/dsl/sugar.h>
#include <luisa/core/logging.h>

namespace sail::inno::test {

using namespace luisa;
using namespace luisa::compute;

int test_rotate() {
	using namespace inno::math;
	using namespace luisa;
	using namespace luisa::compute;

	auto aa = make_float4(1.0f, 1.0f, 1.0f, 90.0f);
	auto qaa = qvec_from_aa<float4>(aa);
	auto qvec = normalize(qaa);
	auto Rtarget = R_from_qvec<float4, float3x3>(qvec);
	auto q = make_float4(0.0f, 0.0f, 0.0f, 1.0f);
	LUISA_INFO("qvec: {} -> {}", q, qvec);
	auto dLdq = make_float4(0.0f);
	auto max_iter = 10;
	auto alpha = 0.1f;

	for (auto idx = 0; idx < max_iter; idx++) {
		auto R = R_from_qvec<float4, float3x3>(q);
		auto dLdR = R - Rtarget;
		dLdq = R_from_qvec_backward<float3, float4, float3x3>(dLdR, q, R);
		q = q - alpha * dLdq;
		q = normalize(q);
		LUISA_INFO("qvec: {} -> {}", q, qvec);
	}
	// LUISA_INFO("qvec: {} -> {}", q, qvec);

	return 0;
}

}// namespace sail::inno::test

TEST_SUITE("math") {
	TEST_CASE("rotate") {
		REQUIRE(sail::inno::test::test_rotate() == 0);
	}
}