/**
 * @file package/diff_render/gs/gaussian_splatter_callable.cpp
 * @author sailing-innocent
 * @date 2024-03-05
 * @brief The Gaussian Splatter Callables
 */

#include "SailInno/diff_render/reprod_gs_splatter.h"
#include "SailInno/util/graphic/sh.h"
#include "SailInno/util/math/gaussian.h"

using namespace luisa;
using namespace luisa::compute;

namespace sail::inno::render {

void ReprodGS::compile_callables(Device& device) noexcept {
	using namespace luisa;
	using namespace luisa::compute;
	mp_ndc2pix = luisa::make_unique<Callable<float(float, uint)>>([](Float v, UInt S) {
		return ((v + 1.0f) * S - 1.0f) * 0.5f;
	});
	mp_get_rect = luisa::make_unique<Callable<void(luisa::compute::float2, int, luisa::compute::uint2&, luisa::compute::uint2&, luisa::compute::uint2, luisa::compute::uint2)>>(
		[](
			Float2 p,
			Int max_radius,
			UInt2& rect_min,
			UInt2& rect_max,
			UInt2 blocks, UInt2 grids) {
		// clamp
		rect_min = make_uint2(
			clamp(UInt((p.x - max_radius) / blocks.x), Var(0u), grids.x),
			clamp(UInt((p.y - max_radius) / blocks.y), Var(0u), grids.y));
		rect_max = make_uint2(
			clamp(UInt(p.x + max_radius + blocks.x - 1) / blocks.x, Var(0u), grids.x),
			clamp(UInt(p.y + max_radius + blocks.y - 1) / blocks.y, Var(0u), grids.y));
	});

	mp_compute_cov_3d = luisa::make_unique<Callable<float3x3(float3, float, float4)>>([](Float3 scales, Float scale_modifier, Float4 rot_qvec) {
		Float3 scale = scale_modifier * scales;
		// computer rotation from rot_qvec
		Float r = rot_qvec.x;
		Float x = rot_qvec.y;
		Float y = rot_qvec.z;
		Float z = rot_qvec.w;
		Float4 qvec = make_float4(x, y, z, r);
		Float3x3 cov = math::calc_cov<Float3, Float4, Float3x3>(scale, qvec);
		return cov;
	});

	mp_compute_cov_2d = luisa::make_unique<Callable<float3(float4, float4, float3x3, float4x4)>>(
		[](
			Float4 p_view,
			Float4 camera_primitive,
			Float3x3 cov_3d,
			Float4x4 view_matrix) {
		// Linear Gaussian Transformation
		// $\Sigma_y=A\Sigma_xA^T$

		auto focal_x = camera_primitive.x;
		auto focal_y = camera_primitive.y;
		auto tan_fov_x = camera_primitive.z;
		auto tan_fov_y = camera_primitive.w;
		auto t = p_view.xyz();
		auto limx = 1.3f * tan_fov_x;
		auto limy = 1.3f * tan_fov_y;
		auto txtz = t.x / t.z;
		auto tytz = t.y / t.z;
		t.x = clamp(txtz, -limx, limx) * t.z;
		t.y = clamp(tytz, -limy, limy) * t.z;

		// Float3x3 J = make_float3x3(
		// 	focal_x / t.z, 0.0f, -(focal_x * t.x) / (t.z * t.z),
		// 	0.0f, focal_y / t.z, -(focal_y * t.y) / (t.z * t.z),
		// 	0.0f, 0.0f, 0.0f);

		// consider function p = m(t)
		// $p_x=\frac{f_xt_x}{t_z}$
		// $p_y=\frac{f_yt_y}{t_z}$
		// $p_z=1$
		// Calculate the Jacobian of m(t)
		// J =
		// fx/tz, 0.0,  fx*tx/(tz * tz)
		// 0.0,   fy/tz, fy*ty/(tz * tz)
		// 0.0    0.0,   0.0
		Float3x3 J = make_float3x3(
			focal_x / t.z, 0.0f, 0.0f,
			0.0f, focal_y / t.z, 0.0f,
			-(focal_x * t.x) / (t.z * t.z), -(focal_y * t.y) / (t.z * t.z), 0.0f);

		Float3x3 W = make_float3x3(
			view_matrix[0].xyz(),
			view_matrix[1].xyz(),
			view_matrix[2].xyz());

		Float3x3 T = J * W;
		Float3x3 cov = T * cov_3d * transpose(T);
		// low pass filter
		// auto focal = camera_primitive.y;
		cov[0][0] += 0.3f;
		cov[1][1] += 0.3f;
		return make_float3(cov[0][0], cov[0][1], cov[1][1]);
	});

	mp_compute_color_from_sh = luisa::make_unique<Callable<float3(int, int, int, Buffer<float>, float3, Buffer<float>)>>(
		[&](Int idx, Int deg, Int max_deg, BufferVar<float> means, Float3 campos, BufferVar<float> shs) {
		Int sh_idx_start = idx * (max_deg + 1) * (max_deg + 1) * 3;
		Int feat_dim = (max_deg + 1) * (max_deg + 1);// 1 + 3 + 5 + 7 = 16
		// (N, feat_dim, 3)

		Float3 sh_00 = make_float3(
			shs.read(sh_idx_start + 0 * 3 + 0),
			shs.read(sh_idx_start + 0 * 3 + 1),
			shs.read(sh_idx_start + 0 * 3 + 2));

		// 1
		Float3 result = sh_00;
		// use color when deg == -1 and max_deg == 0
		$if(deg > -1) {
			result = util::compute_color_from_sh_level_0(sh_00);

			$if(deg > 0) {
				Float3 pos = make_float3(means.read(idx * 3 + 0), means.read(idx * 3 + 1), means.read(idx * 3 + 2));
				Float3 dir = normalize(pos - campos);

				// 3
				auto sh_10 = make_float3(shs.read(sh_idx_start + 1 * 3 + 0), shs.read(sh_idx_start + 1 * 3 + 1), shs.read(sh_idx_start + 1 * 3 + 2));
				auto sh_11 = make_float3(shs.read(sh_idx_start + 2 * 3 + 0), shs.read(sh_idx_start + 2 * 3 + 1), shs.read(sh_idx_start + 2 * 3 + 2));
				auto sh_12 = make_float3(shs.read(sh_idx_start + 3 * 3 + 0), shs.read(sh_idx_start + 3 * 3 + 1), shs.read(sh_idx_start + 3 * 3 + 2));

				result = result + util::compute_color_from_sh_level_1(dir, sh_10, sh_11, sh_12);

				$if(deg > 1) {
					// 5
					auto sh_20 = make_float3(shs.read(sh_idx_start + 4 * 3 + 0), shs.read(sh_idx_start + 4 * 3 + 1), shs.read(sh_idx_start + 4 * 3 + 2));
					auto sh_21 = make_float3(shs.read(sh_idx_start + 5 * 3 + 0), shs.read(sh_idx_start + 5 * 3 + 1), shs.read(sh_idx_start + 5 * 3 + 2));
					auto sh_22 = make_float3(shs.read(sh_idx_start + 6 * 3 + 0), shs.read(sh_idx_start + 6 * 3 + 1), shs.read(sh_idx_start + 6 * 3 + 2));
					auto sh_23 = make_float3(shs.read(sh_idx_start + 7 * 3 + 0), shs.read(sh_idx_start + 7 * 3 + 1), shs.read(sh_idx_start + 7 * 3 + 2));
					auto sh_24 = make_float3(shs.read(sh_idx_start + 8 * 3 + 0), shs.read(sh_idx_start + 8 * 3 + 1), shs.read(sh_idx_start + 8 * 3 + 2));

					result = result + util::compute_color_from_sh_level_2(dir, sh_20, sh_21, sh_22, sh_23, sh_24);

					$if(deg > 2) {
						// 7
						auto sh_30 = make_float3(shs.read(sh_idx_start + 9 * 3 + 0), shs.read(sh_idx_start + 9 * 3 + 1), shs.read(sh_idx_start + 9 * 3 + 2));
						auto sh_31 = make_float3(shs.read(sh_idx_start + 10 * 3 + 0), shs.read(sh_idx_start + 10 * 3 + 1), shs.read(sh_idx_start + 10 * 3 + 2));
						auto sh_32 = make_float3(shs.read(sh_idx_start + 11 * 3 + 0), shs.read(sh_idx_start + 11 * 3 + 1), shs.read(sh_idx_start + 11 * 3 + 2));
						auto sh_33 = make_float3(shs.read(sh_idx_start + 12 * 3 + 0), shs.read(sh_idx_start + 12 * 3 + 1), shs.read(sh_idx_start + 12 * 3 + 2));
						auto sh_34 = make_float3(shs.read(sh_idx_start + 13 * 3 + 0), shs.read(sh_idx_start + 13 * 3 + 1), shs.read(sh_idx_start + 13 * 3 + 2));
						auto sh_35 = make_float3(shs.read(sh_idx_start + 14 * 3 + 0), shs.read(sh_idx_start + 14 * 3 + 1), shs.read(sh_idx_start + 14 * 3 + 2));
						auto sh_36 = make_float3(shs.read(sh_idx_start + 15 * 3 + 0), shs.read(sh_idx_start + 15 * 3 + 1), shs.read(sh_idx_start + 15 * 3 + 2));

						result = result + util::compute_color_from_sh_level_3(dir, sh_30, sh_31, sh_32, sh_33, sh_34, sh_35, sh_36);
					};
				};
			};

			result = result + 0.5f;
		};
		result = max(result, 0.0f);
		// result = clamp(result, 0.0f, 1.0f);
		// result = 0.5f * result;
		return result;
	});

	mp_compute_color_from_sh_backward = luisa::make_unique<
		Callable<float3(
			int, int, int,
			Buffer<float>,
			float3,
			Buffer<float>,
			Buffer<float>,
			Buffer<float>)>>(
		[&](
			Int idx, Int deg, Int max_deg,
			BufferVar<float> means,
			Float3 campos,
			BufferVar<float> shs,
			BufferVar<float> dL_d_color_feature,
			BufferVar<float> dL_d_feat) {
		Int feat_dim = (max_deg + 1) * (max_deg + 1);
		Int sh_idx_start = idx * feat_dim * 3;
		Float3 sh_00 = make_float3(
			shs.read(sh_idx_start + 0),
			shs.read(sh_idx_start + 1),
			shs.read(sh_idx_start + 2));
		Float3 result = sh_00;
		// input
		Float3 dL_d_color = make_float3(
			dL_d_color_feature.read(3 * idx + 0),
			dL_d_color_feature.read(3 * idx + 1),
			dL_d_color_feature.read(3 * idx + 2));

		// output
		$if(deg == -1) {
			// use RGB, direct output
			dL_d_feat.write(sh_idx_start + 0, dL_d_color[0]);
			dL_d_feat.write(sh_idx_start + 1, dL_d_color[1]);
			dL_d_feat.write(sh_idx_start + 2, dL_d_color[2]);
		};

		// USE SH
		$if(deg > -1) {
			result = util::compute_color_from_sh_level_0(sh_00);
			Float3 dL_d_sh00 = make_float3(0.0f);
			// $if(idx < 10) {
			// 	// debug dL_d_color
			// 	device_log("dL_d_color: {}", dL_d_color);
			// };
			util::compute_color_from_sh_level_0_backward(dL_d_color, dL_d_sh00);
			dL_d_feat.write(sh_idx_start + 0 * 3 + 0, dL_d_sh00[0]);
			dL_d_feat.write(sh_idx_start + 0 * 3 + 1, dL_d_sh00[1]);
			dL_d_feat.write(sh_idx_start + 0 * 3 + 2, dL_d_sh00[2]);

			$if(deg > 0) {
				Float3 pos = make_float3(means.read(idx * 3 + 0), means.read(idx * 3 + 1), means.read(idx * 3 + 2));
				Float3 dir = normalize(pos - campos);
				Float3 dL_d_dir = make_float3(0.0f);

				auto sh_10 = make_float3(shs.read(sh_idx_start + 1 * 3 + 0), shs.read(sh_idx_start + 4), shs.read(sh_idx_start + 5));
				auto sh_11 = make_float3(shs.read(sh_idx_start + 1 * 3 + 0), shs.read(sh_idx_start + 7), shs.read(sh_idx_start + 8));
				auto sh_12 = make_float3(shs.read(sh_idx_start + 1 * 3 + 0), shs.read(sh_idx_start + 10), shs.read(sh_idx_start + 11));

				Float3 dL_d_sh10 = make_float3(0.0f);
				Float3 dL_d_sh11 = make_float3(0.0f);
				Float3 dL_d_sh12 = make_float3(0.0f);

				result = result + util::compute_color_from_sh_level_1(dir, sh_10, sh_11, sh_12);
				util::compute_color_from_sh_level_1_backward(dL_d_color, dir, dL_d_sh10, dL_d_sh11, dL_d_sh12, dL_d_dir);

				// 1-3
				dL_d_feat.write(sh_idx_start + 1 * 3 + 0, dL_d_sh10[0]);
				dL_d_feat.write(sh_idx_start + 1 * 3 + 1, dL_d_sh10[1]);
				dL_d_feat.write(sh_idx_start + 1 * 3 + 2, dL_d_sh10[2]);
				dL_d_feat.write(sh_idx_start + 2 * 3 + 0, dL_d_sh11[0]);
				dL_d_feat.write(sh_idx_start + 2 * 3 + 1, dL_d_sh11[1]);
				dL_d_feat.write(sh_idx_start + 2 * 3 + 2, dL_d_sh11[2]);
				dL_d_feat.write(sh_idx_start + 3 * 3 + 0, dL_d_sh12[0]);
				dL_d_feat.write(sh_idx_start + 3 * 3 + 1, dL_d_sh12[1]);
				dL_d_feat.write(sh_idx_start + 3 * 3 + 2, dL_d_sh12[2]);

				$if(deg > 1) {
					// 4-8
					auto sh_20 = make_float3(shs.read(sh_idx_start + 4 * 3 + 0), shs.read(sh_idx_start + 4 * 3 + 1), shs.read(sh_idx_start + 4 * 3 + 2));
					auto sh_21 = make_float3(shs.read(sh_idx_start + 5 * 3 + 0), shs.read(sh_idx_start + 5 * 3 + 1), shs.read(sh_idx_start + 5 * 3 + 2));
					auto sh_22 = make_float3(shs.read(sh_idx_start + 6 * 3 + 0), shs.read(sh_idx_start + 6 * 3 + 1), shs.read(sh_idx_start + 6 * 3 + 2));
					auto sh_23 = make_float3(shs.read(sh_idx_start + 7 * 3 + 0), shs.read(sh_idx_start + 7 * 3 + 1), shs.read(sh_idx_start + 7 * 3 + 2));
					auto sh_24 = make_float3(shs.read(sh_idx_start + 8 * 3 + 0), shs.read(sh_idx_start + 8 * 3 + 1), shs.read(sh_idx_start + 8 * 3 + 2));

					result = result + util::compute_color_from_sh_level_2(dir, sh_20, sh_21, sh_22, sh_23, sh_24);

					Float3 dL_d_sh_20 = make_float3(0.0f);
					Float3 dL_d_sh_21 = make_float3(0.0f);
					Float3 dL_d_sh_22 = make_float3(0.0f);
					Float3 dL_d_sh_23 = make_float3(0.0f);
					Float3 dL_d_sh_24 = make_float3(0.0f);

					util::compute_color_from_sh_level_2_backward(
						dL_d_color, dir, dL_d_sh_20, dL_d_sh_21, dL_d_sh_22, dL_d_sh_23, dL_d_sh_24, dL_d_dir);

					// write back to dL_d_feat
					dL_d_feat.write(sh_idx_start + 4 * 3 + 0, dL_d_sh_20[0]);
					dL_d_feat.write(sh_idx_start + 4 * 3 + 1, dL_d_sh_20[1]);
					dL_d_feat.write(sh_idx_start + 4 * 3 + 2, dL_d_sh_20[2]);
					dL_d_feat.write(sh_idx_start + 5 * 3 + 0, dL_d_sh_21[0]);
					dL_d_feat.write(sh_idx_start + 5 * 3 + 1, dL_d_sh_21[1]);
					dL_d_feat.write(sh_idx_start + 5 * 3 + 2, dL_d_sh_21[2]);
					dL_d_feat.write(sh_idx_start + 6 * 3 + 0, dL_d_sh_22[0]);
					dL_d_feat.write(sh_idx_start + 6 * 3 + 1, dL_d_sh_22[1]);
					dL_d_feat.write(sh_idx_start + 6 * 3 + 2, dL_d_sh_22[2]);
					dL_d_feat.write(sh_idx_start + 7 * 3 + 0, dL_d_sh_23[0]);
					dL_d_feat.write(sh_idx_start + 7 * 3 + 1, dL_d_sh_23[1]);
					dL_d_feat.write(sh_idx_start + 7 * 3 + 2, dL_d_sh_23[2]);
					dL_d_feat.write(sh_idx_start + 8 * 3 + 0, dL_d_sh_24[0]);
					dL_d_feat.write(sh_idx_start + 8 * 3 + 1, dL_d_sh_24[1]);
					dL_d_feat.write(sh_idx_start + 8 * 3 + 2, dL_d_sh_24[2]);

					$if(deg > 2) {
						// 9 - 15
						auto sh_30 = make_float3(shs.read(sh_idx_start + 9 * 3 + 0), shs.read(sh_idx_start + 9 * 3 + 1), shs.read(sh_idx_start + 9 * 3 + 2));
						auto sh_31 = make_float3(shs.read(sh_idx_start + 10 * 3 + 0), shs.read(sh_idx_start + 10 * 3 + 1), shs.read(sh_idx_start + 10 * 3 + 2));
						auto sh_32 = make_float3(shs.read(sh_idx_start + 11 * 3 + 0), shs.read(sh_idx_start + 11 * 3 + 1), shs.read(sh_idx_start + 11 * 3 + 2));
						auto sh_33 = make_float3(shs.read(sh_idx_start + 12 * 3 + 0), shs.read(sh_idx_start + 12 * 3 + 1), shs.read(sh_idx_start + 12 * 3 + 2));
						auto sh_34 = make_float3(shs.read(sh_idx_start + 13 * 3 + 0), shs.read(sh_idx_start + 13 * 3 + 1), shs.read(sh_idx_start + 13 * 3 + 2));
						auto sh_35 = make_float3(shs.read(sh_idx_start + 14 * 3 + 0), shs.read(sh_idx_start + 14 * 3 + 1), shs.read(sh_idx_start + 14 * 3 + 2));
						auto sh_36 = make_float3(shs.read(sh_idx_start + 15 * 3 + 0), shs.read(sh_idx_start + 15 * 3 + 1), shs.read(sh_idx_start + 15 * 3 + 2));
						result = result + util::compute_color_from_sh_level_3(dir, sh_30, sh_31, sh_32, sh_33, sh_34, sh_35, sh_36);

						Float3 dL_d_sh_30 = make_float3(0.0f);
						Float3 dL_d_sh_31 = make_float3(0.0f);
						Float3 dL_d_sh_32 = make_float3(0.0f);
						Float3 dL_d_sh_33 = make_float3(0.0f);
						Float3 dL_d_sh_34 = make_float3(0.0f);
						Float3 dL_d_sh_35 = make_float3(0.0f);
						Float3 dL_d_sh_36 = make_float3(0.0f);

						util::compute_color_from_sh_level_3_backward(
							dL_d_color, dir, dL_d_sh_30, dL_d_sh_31, dL_d_sh_32, dL_d_sh_33, dL_d_sh_34, dL_d_sh_35, dL_d_sh_36, dL_d_dir);

						// write back
						dL_d_feat.write(sh_idx_start + 9 * 3 + 0, dL_d_sh_30[0]);
						dL_d_feat.write(sh_idx_start + 9 * 3 + 1, dL_d_sh_30[1]);
						dL_d_feat.write(sh_idx_start + 9 * 3 + 2, dL_d_sh_30[2]);
						dL_d_feat.write(sh_idx_start + 10 * 3 + 0, dL_d_sh_31[0]);
						dL_d_feat.write(sh_idx_start + 10 * 3 + 1, dL_d_sh_31[1]);
						dL_d_feat.write(sh_idx_start + 10 * 3 + 2, dL_d_sh_31[2]);
						dL_d_feat.write(sh_idx_start + 11 * 3 + 0, dL_d_sh_32[0]);
						dL_d_feat.write(sh_idx_start + 11 * 3 + 1, dL_d_sh_32[1]);
						dL_d_feat.write(sh_idx_start + 11 * 3 + 2, dL_d_sh_32[2]);
						dL_d_feat.write(sh_idx_start + 12 * 3 + 0, dL_d_sh_33[0]);
						dL_d_feat.write(sh_idx_start + 12 * 3 + 1, dL_d_sh_33[1]);
						dL_d_feat.write(sh_idx_start + 12 * 3 + 2, dL_d_sh_33[2]);
						dL_d_feat.write(sh_idx_start + 13 * 3 + 0, dL_d_sh_34[0]);
						dL_d_feat.write(sh_idx_start + 13 * 3 + 1, dL_d_sh_34[1]);
						dL_d_feat.write(sh_idx_start + 13 * 3 + 2, dL_d_sh_34[2]);
						dL_d_feat.write(sh_idx_start + 14 * 3 + 0, dL_d_sh_35[0]);
						dL_d_feat.write(sh_idx_start + 14 * 3 + 1, dL_d_sh_35[1]);
						dL_d_feat.write(sh_idx_start + 14 * 3 + 2, dL_d_sh_35[2]);
						dL_d_feat.write(sh_idx_start + 15 * 3 + 0, dL_d_sh_36[0]);
						dL_d_feat.write(sh_idx_start + 15 * 3 + 1, dL_d_sh_36[1]);
						dL_d_feat.write(sh_idx_start + 15 * 3 + 2, dL_d_sh_36[2]);
					};
				};
			};
			result = result + 0.5f;
			result = max(result, 0.0f);
		};
		return result;
		// TODO: d_xyz
	});
}
}// namespace sail::inno::render