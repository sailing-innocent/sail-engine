/**
 * @file packages/gaussian/diff_gs_tile_sampler/sampler_shader_tile_split.cpp
 * @author sailing-innocent
 * @date 2024-03-18
 * @brief Tile Split Shader Forward and Backward
 */

#include "SailInno/gaussian/diff_gs_tile_sampler.h"
#include "SailInno/util/graphic/image.h"
#include <luisa/dsl/sugar.h>

using namespace luisa;
using namespace luisa::compute;

namespace sail::inno::gaussian {

void DiffGaussianTileSampler::compile_tile_split_shader(Device& device) noexcept {
	lazy_compile(device, m_forward_tile_split_shader,
				 [&](
					 Int P,
					 UInt2 resolution,
					 UInt2 grids,
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
		// -----------------------------
		// invert covariance
		// det(M) = M[0][0] * M[1][1] - M[0][1] * M[1][0]

		// Linear Transformation for 2D Gaussian
		Float3 cov_2d = make_float3(
			covs_2d.read(3 * idx + 0) * resolution.x * resolution.x * 0.25f,
			covs_2d.read(3 * idx + 1) * resolution.x * resolution.y * 0.25f,
			covs_2d.read(3 * idx + 2) * resolution.y * resolution.y * 0.25f);

		auto point_image_ndc = make_float2(means_2d.read(2 * idx + 0), means_2d.read(2 * idx + 1));
		auto point_image = make_float2(util::ndc2pix<Float>(point_image_ndc.x, resolution.x), util::ndc2pix<Float>(point_image_ndc.y, resolution.y));

		// get radius = 3 \sigma
		Float det = cov_2d.x * cov_2d.z - cov_2d.y * cov_2d.y;
		Float inv_det = 1.0f / det;
		Float3 conic = inv_det * make_float3(cov_2d.z, -cov_2d.y, cov_2d.x);// inv: [0][0] [0][1] transpose [1][1] inverse
		Float mid = 0.5f * (cov_2d.x + cov_2d.z);
		Float lambda1 = mid + sqrt(max(0.1f, mid * mid - det));
		Float lambda2 = mid - sqrt(max(0.1f, mid * mid - det));
		Int my_radius = ceil(3.0f * sqrt(max(lambda1, lambda2)));

		// get rect
		UInt2 rect_min, rect_max;
		(*mp_get_rect)(point_image, my_radius, rect_min, rect_max, m_blocks, grids);

		// write radius
		radii.write(idx, my_radius);
		tiles_touched.write(idx, (rect_max.x - rect_min.x) * (rect_max.y - rect_min.y));
		// write conic
		conics.write(3 * idx + 0, conic.x);
		conics.write(3 * idx + 1, conic.y);
		conics.write(3 * idx + 2, conic.z);
		// update means2d res
		means_2d_res.write(2 * idx + 0, point_image.x);
		means_2d_res.write(2 * idx + 1, point_image.y);
	});

	// backward
}

}// namespace sail::inno::gaussian