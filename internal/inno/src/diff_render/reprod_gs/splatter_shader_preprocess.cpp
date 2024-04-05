/**
 * @file package/diff_render/gs/gaussian_splatter_shader_preprocess.cpp
 * @author sailing-innocent
 * @date 2024-03-06
 * @brief The Gaussian Splatter Shader Preprocess
 */

#include "SailInno/diff_render/reprod_gs_splatter.h"
#include <luisa/dsl/sugar.h>
#include "SailInno/util/math/gaussian.h"
#include "SailInno/util/math/transform.h"
#include "luisa/core/basic_types.h"

using namespace luisa;
using namespace luisa::compute;

namespace sail::inno::render {

void ReprodGS::compile_forward_preprocess_shader(Device& device) noexcept {
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
					 UInt2 resolution,
					 UInt2 grids,
					 // output
					 BufferVar<float> means_2d,
					 BufferVar<float> depth_features,
					 BufferVar<float> color_features,
					 BufferVar<float> conics,
					 BufferVar<uint> tiles_touched,
					 BufferVar<int> radii,
					 // camera
					 Float3 cam_pos,
					 Float4 camera_primitive,
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

		Float4 p_proj_hom = proj_matrix * p_view_hom;
		Float p_w = 1.0f / (p_proj_hom.w + 1e-6f);
		Float3 p_proj = p_proj_hom.xyz() * p_w;

		// color = make_float3(p_proj.x, p_proj.y, p_view.z / 10.0f); // debug xyz color
		// near culling method
		$if(p_view.z <= 0.2f) { return; };

		// calculate 3d covariance
		Float3 s = make_float3(scale_buffer.read(3 * idx + 0), scale_buffer.read(3 * idx + 1), scale_buffer.read(3 * idx + 2));
		Float4 rotq = make_float4(rotq_buffer.read(4 * idx + 0), rotq_buffer.read(4 * idx + 1), rotq_buffer.read(4 * idx + 2), rotq_buffer.read(4 * idx + 3));

		Float3x3 cov_3d = (*mp_compute_cov_3d)(s, scale_modifier, rotq);
		// calculate projected covariance 2d
		Float3 cov_2d = (*mp_compute_cov_2d)(p_view_hom, camera_primitive, cov_3d, view_matrix);

		Float det = cov_2d.x * cov_2d.z - cov_2d.y * cov_2d.y;
		Float inv_det = 1.0f / det;
		Float3 conic = inv_det * make_float3(cov_2d.z, -cov_2d.y, cov_2d.x);// inv: [0][0] [0][1] transpose [1][1] inverse

		Float mid = 0.5f * (cov_2d.x + cov_2d.z);
		Float lambda1 = mid + sqrt(max(0.1f, mid * mid - det));
		Float lambda2 = mid - sqrt(max(0.1f, mid * mid - det));
		Int my_radius = ceil(3.0f * sqrt(max(lambda1, lambda2)));
		UInt2 rect_min, rect_max;

		auto point_image_ndc = make_float2(p_proj.x, p_proj.y);
		auto point_image = make_float2((*mp_ndc2pix)(point_image_ndc.x, resolution.x), (*mp_ndc2pix)(point_image_ndc.y, resolution.y));

		(*mp_get_rect)(point_image, my_radius, rect_min, rect_max, m_blocks, grids);

		// write out

		radii.write(idx, my_radius);
		tiles_touched.write(idx, (rect_max.x - rect_min.x) * (rect_max.y - rect_min.y));

		conics.write(3 * idx + 0, conic.x);
		conics.write(3 * idx + 1, conic.y);
		conics.write(3 * idx + 2, conic.z);

		means_2d.write(2 * idx + 0, point_image.x);
		means_2d.write(2 * idx + 1, point_image.y);

		color_features.write(3 * idx + 0, color.x);
		color_features.write(3 * idx + 1, color.y);
		color_features.write(3 * idx + 2, color.z);

		depth_features.write(idx, p_view.z);
	});
}

void ReprodGS::compile_backward_preprocess_shader(Device& device) noexcept {
	lazy_compile(device, m_backward_preprocess_shader,
				 [&](
					 // input
					 BufferVar<float> dL_d_means_2d,
					 BufferVar<float> dL_d_conic,
					 BufferVar<float> dL_d_color_feature,
					 // output
					 BufferVar<float> dL_d_xyz,
					 BufferVar<float> dL_d_feat,
					 BufferVar<float> dL_d_scale,
					 BufferVar<float> dL_d_rotq,
					 // params
					 Int P, Int D, Int M,
					 UInt2 resolution, UInt2 grids,
					 BufferVar<float> means,
					 BufferVar<float> shs,
					 BufferVar<float> scale_buffer,
					 BufferVar<float> rotq_buffer,
					 BufferVar<float> opacity_features,
					 BufferVar<float> color_feature,
					 BufferVar<float> conics,
					 Float3 campos,
					 Float4 camera_primitive,
					 Float4x4 view_matrix) {
		auto idx = dispatch_id().x;
		$if(idx >= static_cast<$uint>(P)) { return; };
		auto mean_3d = make_float3(means.read(3 * idx + 0), means.read(3 * idx + 1), means.read(3 * idx + 2));
		Float4 p_hom = make_float4(mean_3d, 1.0f);
		Float4 p_view_hom = view_matrix * p_hom;

		(*mp_compute_color_from_sh_backward)(
			// params
			static_cast<Int>(idx), D, M,
			means,
			campos,
			shs,
			// input
			dL_d_color_feature,
			// output
			dL_d_feat);

		// dL_d_conic -> dL_d_cov_2d
		Float3 conic = make_float3(
			conics.read(3 * idx + 0),
			conics.read(3 * idx + 1),
			conics.read(3 * idx + 2));
		// Float opacity = opacity_features.read(idx);
		Float det_inv = (conic.x * conic.z - conic.y * conic.y);
		Float det_inv_2 = det_inv * det_inv;
		Float3 cov_2d = make_float3(conic.z, -conic.y, conic.x) / det_inv;
		Float a = cov_2d.x;
		Float b = cov_2d.y;
		Float c = cov_2d.z;
		// device_log("cov2d: {}", cov_2d);

		Float3 dL_d_con = make_float3(dL_d_conic.read(idx * 3 + 0), dL_d_conic.read(idx * 3 + 1), dL_d_conic.read(idx * 3 + 2));
		Float3 dL_d_cov2d;
		dL_d_cov2d.x = det_inv_2 * (-c * c * dL_d_con.x + 2 * b * c * dL_d_con.y - b * b * dL_d_con.z);
		dL_d_cov2d.z = det_inv_2 * (-b * b * dL_d_con.x + 2 * a * b * dL_d_con.y - a * a * dL_d_con.z);
		dL_d_cov2d.y = det_inv_2 * 2 * (b * c * dL_d_con.x - (a * c + b * b) * dL_d_con.y + a * b * dL_d_con.z);
		// $if(idx < 10) {
		// 	device_log("dL_d_cov2d: {}", dL_d_cov2d);
		// };

		// dL_d_cov_3d -> dL_d_cov3d
		Float3x3 dL_d_cov3d = make_float3x3(0.0f);
		// dL_d_cov3d[0][0] = dL_d_cov2d.x;
		// dL_d_cov3d[1][1] = dL_d_cov2d.z;
		inno::math::proj_cov3d_to_cov2d_backward<Float3, Float4, Float3x3, Float4x4>(dL_d_cov2d, dL_d_cov3d, p_view_hom, camera_primitive, view_matrix);

		// debug dL_d_cov3d
		// $if(idx < 10) {
		// 	device_log("dL_d_cov3d: {}", dL_d_cov3d);
		// };
		Float3 dL_ds = make_float3(0.0f, 0.0f, 0.0f);
		Float4 dL_dqvec = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
		Float3 scale = make_float3(scale_buffer.read(3 * idx + 0), scale_buffer.read(3 * idx + 1), scale_buffer.read(3 * idx + 2));

		// F**K U
		// careful: rxyz -> xyzw
		Float4 qvec = make_float4(
			rotq_buffer.read(4 * idx + 1),
			rotq_buffer.read(4 * idx + 2),
			rotq_buffer.read(4 * idx + 3),
			rotq_buffer.read(4 * idx + 0));

		// debug scale and qvec
		// $if(idx < 10) {
		// 	device_log("scale: {}", scale);
		// 	device_log("qvec: {}", qvec);
		// };

		inno::math::calc_cov_backward<Float3, Float4, Float3x3>(dL_d_cov3d, dL_ds, dL_dqvec, scale, qvec);
		// Float3x3 R = sail::inno::math::R_from_qvec<Float4, Float3x3>(qvec);
		// // debug R
		// $if(idx < 10) {
		// 	device_log("R: {}", R);
		// };

		// write out
		dL_d_scale.write(3 * idx + 0, dL_ds.x);
		dL_d_scale.write(3 * idx + 1, dL_ds.y);
		dL_d_scale.write(3 * idx + 2, dL_ds.z);

		// careful: xyzw -> rxyz
		dL_d_rotq.write(4 * idx + 0, dL_dqvec.w);
		dL_d_rotq.write(4 * idx + 1, dL_dqvec.x);
		dL_d_rotq.write(4 * idx + 2, dL_dqvec.y);
		dL_d_rotq.write(4 * idx + 3, dL_dqvec.z);
		// dL_d_xyz.write(3 * idx + 0, dL_d_cov3d[0][0]);
		// dL_d_xyz.write(3 * idx + 1, dL_d_cov3d[1][1]);
		// dL_d_xyz.write(3 * idx + 2, dL_d_cov3d[2][2]);
	});
}

}// namespace sail::inno::render
