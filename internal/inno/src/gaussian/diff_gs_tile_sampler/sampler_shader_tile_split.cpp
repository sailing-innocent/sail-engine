/**
 * @file packages/gaussian/diff_gs_tile_sampler/sampler_shader_tile_split.cpp
 * @author sailing-innocent
 * @date 2024-03-18
 * @brief Tile Split Shader Forward and Backward
 */

#include "SailInno/gaussian/diff_gs_tile_sampler.h"
#include "SailInno/util/graphic/image.h"
#include "luisa/dsl/var.h"
#include <luisa/dsl/sugar.h>
#include <luisa/dsl/printer.h>

using namespace luisa;
using namespace luisa::compute;

namespace sail::inno::gaussian {

void DiffGaussianTileSampler::compile_tile_split_shader(Device& device) noexcept {
	lazy_compile(device, m_forward_tile_split_shader,
				 [&](
					 Int P,
					 UInt2 resolution,
					 UInt2 grids,
					 Float fov_rad,
					 // input
					 BufferVar<float> means_2d,
					 BufferVar<float> covs_2d,
					 // output
					 BufferVar<float> conics,
					 BufferVar<float> means_2d_res,
					 BufferVar<uint> tiles_touched,
					 BufferVar<int> radii) {
		set_block_size(m_blocks.x * m_blocks.y);
		auto idx = dispatch_id().x;
		$if(idx >= static_cast<$uint>(P)) { return; };
		// -----------------------------
		// allocate to tile space
		// TODO: Culling
		// -----------------------------
		// invert covariance
		// det(M) = M[0][0] * M[1][1] - M[0][1] * M[1][0]

		// Linear Transformation for 2D Gaussian
		auto wf = Float(resolution.x);
		auto hf = Float(resolution.y);

		auto aspect = wf / hf;
		auto fy = 1.0f * tan(fov_rad * 0.5f);
		auto fx = fy * aspect;

		auto point_image_2d = make_float2(means_2d.read(2 * idx + 0), means_2d.read(2 * idx + 1));
		// assume edge [-tan(fov/2), tan(fov/2)]
		// zoom to ndc
		auto point_image_ndc = make_float2(
			point_image_2d.x / fx,
			point_image_2d.y / fy);// -> [-1, 1]
		// zoom to pixel
		auto point_image = make_float2(util::ndc2pix<Float>(point_image_ndc.x, wf), util::ndc2pix<Float>(point_image_ndc.y, hf));
		// -> [0, w] [0, h]

		Float3 cov_2d = make_float3(
			covs_2d.read(3 * idx + 0),
			covs_2d.read(3 * idx + 1),
			covs_2d.read(3 * idx + 2));
		// assume edge [-tan(fov/2), tan(fov/2)]

		// zoom to pixel
		auto sx = wf / fx / 2.0f;
		auto sy = hf / fy / 2.0f;
		cov_2d = make_float3(cov_2d.x * sx * sx, cov_2d.y * sx * sy, cov_2d.z * sy * sy);
		// J^TXJ = [xxa xyb; xyb yyc]

		// now everything locates in pixel space

		// get radius = 3 \sigma
		Float det = cov_2d.x * cov_2d.z - cov_2d.y * cov_2d.y;
		Float inv_det = 1.0f / det;
		Float3 conic = inv_det * make_float3(cov_2d.z, -cov_2d.y, cov_2d.x);// inv: [0][0] [0][1] transpose [1][1] inverse

		Float mid = 0.5f * (cov_2d.x + cov_2d.z);
		Float lambda1 = mid + sqrt(max(0.1f, mid * mid - det));
		Float lambda2 = mid - sqrt(max(0.1f, mid * mid - det));
		Int my_radius = ceil(3.0f * sqrt(max(lambda1, lambda2)));
		// 迷惑行为

		// get rect
		UInt2 rect_min, rect_max;
		(*mp_get_rect)(point_image, my_radius, rect_min, rect_max, m_blocks, grids);

		// write radius
		radii.write(idx, my_radius);
		auto n_tiles_touched = (rect_max.x - rect_min.x) * (rect_max.y - rect_min.y);
		// device_log("n_tiles_touched: {}", n_tiles_touched);
		tiles_touched.write(idx, n_tiles_touched);
		// write conic
		conics.write(3 * idx + 0, conic.x);
		conics.write(3 * idx + 1, conic.y);
		conics.write(3 * idx + 2, conic.z);
		// update means2d res
		means_2d_res.write(2 * idx + 0, point_image.x);
		means_2d_res.write(2 * idx + 1, point_image.y);
	});

	// backward
	lazy_compile(device, m_backward_tile_split_shader,
				 [&](
					 Int P,
					 // input
					 BufferVar<float> dL_d_conic,
					 // params
					 UInt2 resolution,
					 Float fov_rad,
					 BufferVar<float> covs_2d,
					 // output
					 BufferVar<float> dL_d_cov_2d) {
		set_block_size(m_blocks.x * m_blocks.y);
		auto idx = dispatch_id().x;
		$if(idx >= static_cast<$uint>(P)) { return; };

		auto wf = Float(resolution.x);
		auto hf = Float(resolution.y);
		auto fy = hf * tan(fov_rad * 0.5f) / 2.0f;
		auto fx = fy * wf / hf;
		// J
		// transform to pixel space
		Float3 cov_2d = make_float3(
			covs_2d.read(3 * idx + 0) * fx * fx,
			covs_2d.read(3 * idx + 1) * fx * fy,
			covs_2d.read(3 * idx + 2) * fy * fy);

		Float det = cov_2d.x * cov_2d.z - cov_2d.y * cov_2d.y;
		// Float det_inv = conic.x * conic.z - conic.y * conic.y;
		Float det_inv_2 = 1 / (det * det + 1e-6f);
		Float a = cov_2d.x;
		Float b = cov_2d.y;
		Float c = cov_2d.z;

		// conic in pixel spacee
		Float3 dL_d_con = make_float3(dL_d_conic.read(idx * 3 + 0), dL_d_conic.read(idx * 3 + 1), dL_d_conic.read(idx * 3 + 2));
		Float3 dL_d_cov2d;
		// dc_a/da= - c * c
		// dc_b/da= + c * b
		// dc_c/da= - b * b
		// dc_a / dc = - b * b
		// dc_b / dc =  a * b
		// dc_c / dc = - a * a
		dL_d_cov2d.x = det_inv_2 * (-c * c * dL_d_con.x + b * c * dL_d_con.y - b * b * dL_d_con.z);
		dL_d_cov2d.z = det_inv_2 * (-b * b * dL_d_con.x + a * b * dL_d_con.y - a * a * dL_d_con.z);
		dL_d_cov2d.y = det_inv_2 * 2 * (b * c * dL_d_con.x - (a * c + b * b) * dL_d_con.y + a * b * dL_d_con.z);
		// dL_d_cov2d = dL_d_con;
		// back to cov_2d
		dL_d_cov_2d.write(3 * idx + 0, dL_d_cov2d.x * fx * fx);
		dL_d_cov_2d.write(3 * idx + 1, dL_d_cov2d.y * fx * fy);
		dL_d_cov_2d.write(3 * idx + 2, dL_d_cov2d.z * fy * fy);
	});
}

}// namespace sail::inno::gaussian