/**
 * @file package/diff_gs_projector/projector_shader.cpp
 * @author sailing-innocent
 * @date 2024-03-07
 * @brief The Gaussian Splatter Basic Implement
 */

#include "SailInno/gaussian/diff_gs_projector.h"
#include "SailInno/util/graphic/sh.h"
#include "SailInno/util/math/gaussian.h"

#include <luisa/dsl/sugar.h>

using namespace luisa;
using namespace luisa::compute;

namespace sail::inno::gaussian {

void DiffGaussianProjector::compile(Device& device) noexcept {
	using namespace luisa;
	using namespace luisa::compute;

	mp_compute_cov_3d = luisa::make_unique<Callable<float3x3(float3, float, float4)>>([](Float3 scales, Float scale_modifier, Float4 rot_qvec) {
		Float3 scale = scale_modifier * scales;
		// computer rotation from rot_qvec
		Float r = rot_qvec.x;
		Float x = rot_qvec.y;
		Float y = rot_qvec.z;
		Float z = rot_qvec.w;

		Float4 qvec = make_float4(x, y, z, r);
		Float3x3 cov = math::calc_cov<Float3, Float4, Float3x3>(scale, qvec);

		// TODO
		return cov;
	});

	mp_compute_cov_2d = luisa::make_unique<Callable<float3(float3, float3x3, float4x4)>>(
		[](
			Float3 p_view,
			Float3x3 cov_3d,
			Float4x4 view_matrix) {
		Float3 cov = math::proj_cov3d_to_cov2d_01<Float3, Float3x3, Float4x4>(p_view, cov_3d, view_matrix);
		return cov;
	});

	mp_compute_color_from_sh = luisa::make_unique<Callable<float3(int, int, int, Buffer<float>, float3, Buffer<float>)>>(
		[&](Int idx, Int deg, Int max_deg, BufferVar<float> means, Float3 campos, BufferVar<float> shs) {
		Int sh_idx_start = idx * (max_deg + 1) * (max_deg + 1) * 3;
		Float3 sh_00 = make_float3(shs.read(sh_idx_start + 0), shs.read(sh_idx_start + 1), shs.read(sh_idx_start + 2));

		Float3 result = sh_00;
		// use color when deg == -1 and max_deg == 0
		$if(deg > -1) {
			result = util::compute_color_from_sh_level_0(sh_00);
			$if(deg > 0) {
				Float3 pos = make_float3(means.read(idx * 3 + 0), means.read(idx * 3 + 1), means.read(idx * 3 + 2));
				Float3 dir = normalize(pos - campos);
				auto sh_10 = make_float3(shs.read(sh_idx_start + 3), shs.read(sh_idx_start + 4), shs.read(sh_idx_start + 5));
				auto sh_11 = make_float3(shs.read(sh_idx_start + 6), shs.read(sh_idx_start + 7), shs.read(sh_idx_start + 8));
				auto sh_12 = make_float3(shs.read(sh_idx_start + 9), shs.read(sh_idx_start + 10), shs.read(sh_idx_start + 11));

				result = result + util::compute_color_from_sh_level_1(dir, sh_10, sh_11, sh_12);

				$if(deg > 1) {
					auto sh_20 = make_float3(shs.read(sh_idx_start + 12), shs.read(sh_idx_start + 13), shs.read(sh_idx_start + 14));
					auto sh_21 = make_float3(shs.read(sh_idx_start + 15), shs.read(sh_idx_start + 16), shs.read(sh_idx_start + 17));
					auto sh_22 = make_float3(shs.read(sh_idx_start + 18), shs.read(sh_idx_start + 19), shs.read(sh_idx_start + 20));
					auto sh_23 = make_float3(shs.read(sh_idx_start + 21), shs.read(sh_idx_start + 22), shs.read(sh_idx_start + 23));
					auto sh_24 = make_float3(shs.read(sh_idx_start + 24), shs.read(sh_idx_start + 25), shs.read(sh_idx_start + 26));

					result = result + util::compute_color_from_sh_level_2(dir, sh_20, sh_21, sh_22, sh_23, sh_24);

					$if(deg > 2) {
						auto sh_30 = make_float3(shs.read(sh_idx_start + 27), shs.read(sh_idx_start + 28), shs.read(sh_idx_start + 29));
						auto sh_31 = make_float3(shs.read(sh_idx_start + 30), shs.read(sh_idx_start + 31), shs.read(sh_idx_start + 32));
						auto sh_32 = make_float3(shs.read(sh_idx_start + 33), shs.read(sh_idx_start + 34), shs.read(sh_idx_start + 35));
						auto sh_33 = make_float3(shs.read(sh_idx_start + 36), shs.read(sh_idx_start + 37), shs.read(sh_idx_start + 38));
						auto sh_34 = make_float3(shs.read(sh_idx_start + 39), shs.read(sh_idx_start + 40), shs.read(sh_idx_start + 41));
						auto sh_35 = make_float3(shs.read(sh_idx_start + 42), shs.read(sh_idx_start + 43), shs.read(sh_idx_start + 44));
						auto sh_36 = make_float3(shs.read(sh_idx_start + 45), shs.read(sh_idx_start + 46), shs.read(sh_idx_start + 47));

						result = result + util::compute_color_from_sh_level_3(dir, sh_30, sh_31, sh_32, sh_33, sh_34, sh_35, sh_36);
					};
				};
			};
			result = result + 0.5f;
		};
		result = max(result, 0.0f);
		return result;
	});
	lazy_compile(device, m_forward_preprocess_shader,
				 [&](
					 Int P, Int D, Int M,
					 // input
					 BufferVar<float> means_3d,
					 BufferVar<float> feat_buffer,
					 BufferVar<float> scale_buffer,
					 BufferVar<float> rotq_buffer,
					 // params
					 Float scale_modifier,
					 // output
					 BufferVar<float> means_2d,
					 BufferVar<float> depth_features,
					 BufferVar<float> color_features,
					 BufferVar<float> covs_2d,
					 // camera
					 Float3 cam_pos,
					 Float4x4 view_matrix,
					 Float4x4 proj_matrix) {
		set_block_size(m_blocks.x * m_blocks.y);
		auto idx = dispatch_id().x;
		$if(idx >= static_cast<$uint>(P)) { return; };

		// preprocess SH
		auto color = (*mp_compute_color_from_sh)(static_cast<Int>(idx), D, M, means_3d, cam_pos, feat_buffer);
		// -----------------------------
		// project to screen space
		// -----------------------------
		auto mean_3d = make_float3(means_3d.read(3 * idx + 0), means_3d.read(3 * idx + 1), means_3d.read(3 * idx + 2));
		Float4 p_hom = make_float4(mean_3d, 1.0f);
		Float4 p_view_hom = view_matrix * p_hom;
		Float3 p_view = p_view_hom.xyz();
		// near culling method
		$if(p_view.z <= 0.2f) { return; };
		// calculate 3d covariance
		Float3 s = make_float3(scale_buffer.read(3 * idx + 0), scale_buffer.read(3 * idx + 1), scale_buffer.read(3 * idx + 2));
		Float4 rotq = make_float4(rotq_buffer.read(4 * idx + 0), rotq_buffer.read(4 * idx + 1), rotq_buffer.read(4 * idx + 2), rotq_buffer.read(4 * idx + 3));
		Float3x3 cov_3d = (*mp_compute_cov_3d)(s, scale_modifier, rotq);
		// cov_3d = make_float3x3(1.0f) * 0.0001f;
		// calculate projected covariance 2d
		Float3 cov_2d = (*mp_compute_cov_2d)(p_view, cov_3d, view_matrix);
		covs_2d.write(3 * idx + 0, cov_2d.x);
		covs_2d.write(3 * idx + 1, cov_2d.y);
		covs_2d.write(3 * idx + 2, cov_2d.z);
		means_2d.write(2 * idx + 0, p_view.x / p_view.z);
		means_2d.write(2 * idx + 1, p_view.y / p_view.z);
		color_features.write(3 * idx + 0, color.x);
		color_features.write(3 * idx + 1, color.y);
		color_features.write(3 * idx + 2, color.z);

		Float4 depth_feature = make_float4(0.0f);
		depth_feature.x = p_view.z;
		depth_features.write(idx, depth_feature.x);
	});
}

}// namespace sail::inno::gaussian