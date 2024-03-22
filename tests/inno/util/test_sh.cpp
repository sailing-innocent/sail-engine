/**
 * @file tests/inno/util.cpp
 * @author sailing-innocent
 * @brief SH Tester
 * @date 2023-01-20
 */

#include "SailInno/util/graphic/sh.h"
#include "test_util.h"
#include <luisa/dsl/sugar.h>

namespace sail::test {

int test_sh() {
	using namespace inno::util;
	using namespace luisa;
	using namespace luisa::compute;

	auto sh_00 = make_float3(0.5f, 0.5f, 0.5f);
	auto sh_10 = make_float3(0.5f, 0.5f, 0.5f);
	auto sh_11 = make_float3(0.5f, 0.5f, 0.5f);
	auto sh_12 = make_float3(0.5f, 0.5f, 0.5f);
	auto sh_20 = make_float3(0.5f, 0.5f, 0.5f);
	auto sh_21 = make_float3(0.5f, 0.5f, 0.5f);
	auto sh_22 = make_float3(0.5f, 0.5f, 0.5f);
	auto sh_23 = make_float3(0.5f, 0.5f, 0.5f);
	auto sh_24 = make_float3(0.5f, 0.5f, 0.5f);
	auto sh_30 = make_float3(0.5f, 0.5f, 0.5f);
	auto sh_31 = make_float3(0.5f, 0.5f, 0.5f);
	auto sh_32 = make_float3(0.5f, 0.5f, 0.5f);
	auto sh_33 = make_float3(0.5f, 0.5f, 0.5f);
	auto sh_34 = make_float3(0.5f, 0.5f, 0.5f);
	auto sh_35 = make_float3(0.5f, 0.5f, 0.5f);
	auto sh_36 = make_float3(0.5f, 0.5f, 0.5f);

	auto sh_00_tgt = make_float3(0.8f, 0.8f, 0.8f);
	auto sh_10_tgt = make_float3(0.8f, 0.8f, 0.8f);
	auto sh_11_tgt = make_float3(0.8f, 0.8f, 0.8f);
	auto sh_12_tgt = make_float3(0.8f, 0.8f, 0.8f);
	auto sh_20_tgt = make_float3(0.8f, 0.8f, 0.8f);
	auto sh_21_tgt = make_float3(0.8f, 0.8f, 0.8f);
	auto sh_22_tgt = make_float3(0.8f, 0.8f, 0.8f);
	auto sh_23_tgt = make_float3(0.8f, 0.8f, 0.8f);
	auto sh_24_tgt = make_float3(0.8f, 0.8f, 0.8f);
	auto sh_30_tgt = make_float3(0.8f, 0.8f, 0.8f);
	auto sh_31_tgt = make_float3(0.8f, 0.8f, 0.8f);
	auto sh_32_tgt = make_float3(0.8f, 0.8f, 0.8f);
	auto sh_33_tgt = make_float3(0.8f, 0.8f, 0.8f);
	auto sh_34_tgt = make_float3(0.8f, 0.8f, 0.8f);
	auto sh_35_tgt = make_float3(0.8f, 0.8f, 0.8f);
	auto sh_36_tgt = make_float3(0.8f, 0.8f, 0.8f);

	auto dL_d_sh00 = make_float3(0.0f, 0.0f, 0.0f);
	auto dL_d_sh10 = make_float3(0.0f, 0.0f, 0.0f);
	auto dL_d_sh11 = make_float3(0.0f, 0.0f, 0.0f);
	auto dL_d_sh12 = make_float3(0.0f, 0.0f, 0.0f);
	auto dL_d_sh20 = make_float3(0.0f, 0.0f, 0.0f);
	auto dL_d_sh21 = make_float3(0.0f, 0.0f, 0.0f);
	auto dL_d_sh22 = make_float3(0.0f, 0.0f, 0.0f);
	auto dL_d_sh23 = make_float3(0.0f, 0.0f, 0.0f);
	auto dL_d_sh24 = make_float3(0.0f, 0.0f, 0.0f);
	auto dL_d_sh30 = make_float3(0.0f, 0.0f, 0.0f);
	auto dL_d_sh31 = make_float3(0.0f, 0.0f, 0.0f);
	auto dL_d_sh32 = make_float3(0.0f, 0.0f, 0.0f);
	auto dL_d_sh33 = make_float3(0.0f, 0.0f, 0.0f);
	auto dL_d_sh34 = make_float3(0.0f, 0.0f, 0.0f);
	auto dL_d_sh35 = make_float3(0.0f, 0.0f, 0.0f);
	auto dL_d_sh36 = make_float3(0.0f, 0.0f, 0.0f);

	auto dir = make_float3(0.48f, 0.64f, 0.6f);
	auto color_tgt = compute_color_from_sh_level_0(sh_00_tgt);

	color_tgt += compute_color_from_sh_level_1(dir, sh_10_tgt, sh_11_tgt, sh_12_tgt);
	color_tgt += 0.5f;

	auto color = make_float3(0.0f, 0.0f, 0.0f);
	auto dL_d_color = make_float3(0.0f, 0.0f, 0.0f);
	auto dL_d_dir = make_float3(0.0f, 0.0f, 0.0f);

	auto alpha = 0.1f;

	for (auto i = 0; i < 900; i++) {
		color = compute_color_from_sh_level_0(sh_00);
		if (i > 300) {
			color = color + compute_color_from_sh_level_1(dir, sh_10, sh_11, sh_12);
		}
		color = color + 0.5f;
		dL_d_color = color - color_tgt;
		// color = color + compute_color_from_sh_level_2(dir, sh_20, sh_21, sh_22, sh_23, sh_24);
		// color = color + compute_color_from_sh_level_3(dir, sh_30, sh_31, sh_32, sh_33, sh_34, sh_35,
		// sh_36);

		// backward
		compute_color_from_sh_level_0_backward(dL_d_color, dL_d_sh00);
		sh_00 = sh_00 - alpha * dL_d_sh00;

		// compute_color_from_sh_level_2_backward(dL_d_color, dir, dL_d_sh20, dL_d_sh21, dL_d_sh22,
		// dL_d_sh23, dL_d_sh24, dL_d_dir); compute_color_from_sh_level_3_backward(dL_d_color, dir,
		// dL_d_sh30, dL_d_sh31, dL_d_sh32, dL_d_sh33, dL_d_sh34, dL_d_sh35, dL_d_sh36, dL_d_dir);

		if (i > 300) {
			compute_color_from_sh_level_1_backward(
				dL_d_color, dir, dL_d_sh10, dL_d_sh11, dL_d_sh12, dL_d_dir);
			sh_10 = sh_10 - alpha * dL_d_sh10;
			sh_11 = sh_11 - alpha * dL_d_sh11;
			sh_12 = sh_12 - alpha * dL_d_sh12;
		}

		// sh_20 = sh_20 - alpha * dL_d_sh20;
		// sh_21 = sh_21 - alpha * dL_d_sh21;
		// sh_22 = sh_22 - alpha * dL_d_sh22;
		// sh_23 = sh_23 - alpha * dL_d_sh23;
		// sh_24 = sh_24 - alpha * dL_d_sh24;
		// sh_30 = sh_30 - alpha * dL_d_sh30;
		// sh_31 = sh_31 - alpha * dL_d_sh31;
		// sh_32 = sh_32 - alpha * dL_d_sh32;
		// sh_33 = sh_33 - alpha * dL_d_sh33;
		// sh_34 = sh_34 - alpha * dL_d_sh34;
		// sh_35 = sh_35 - alpha * dL_d_sh35;
		// sh_36 = sh_36 - alpha * dL_d_sh36;

		if (i % 100 == 0) {
			LUISA_INFO("color: {} -> {}", color, color_tgt);
			LUISA_INFO("sh_00: {} -> {}", sh_00, sh_00_tgt);
			LUISA_INFO("sh_10: {} -> {}", sh_10, sh_10_tgt);
			LUISA_INFO("sh_11: {} -> {}", sh_11, sh_11_tgt);
			LUISA_INFO("sh_12: {} -> {}", sh_12, sh_12_tgt);
			// LUISA_INFO("sh_20: {} -> {}", sh_20, sh_20_tgt);
			// LUISA_INFO("sh_21: {} -> {}", sh_21, sh_21_tgt);
			// LUISA_INFO("sh_22: {} -> {}", sh_22, sh_22_tgt);
			// LUISA_INFO("sh_23: {} -> {}", sh_23, sh_23_tgt);
			// LUISA_INFO("sh_24: {} -> {}", sh_24, sh_24_tgt);
			// LUISA_INFO("sh_30: {} -> {}", sh_30, sh_30_tgt);
			// LUISA_INFO("sh_31: {} -> {}", sh_31, sh_31_tgt);
			// LUISA_INFO("sh_32: {} -> {}", sh_32, sh_32_tgt);
			// LUISA_INFO("sh_33: {} -> {}", sh_33, sh_33_tgt);
			// LUISA_INFO("sh_34: {} -> {}", sh_34, sh_34_tgt);
			// LUISA_INFO("sh_35: {} -> {}", sh_35, sh_35_tgt);
			// LUISA_INFO("sh_36: {} -> {}", sh_36, sh_36_tgt);
		}
	}

	return 0;
}

}// namespace sail::test

TEST_SUITE("inno::util") {
	TEST_CASE("sh") {
		REQUIRE(sail::test::test_sh() == 0);
	}
}
