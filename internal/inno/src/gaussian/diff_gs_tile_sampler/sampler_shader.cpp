/**
 * @file packages/gaussian/diff_gs_tile_sampler/sampler_shader.cpp
 * @author sailing-innocent
 * @date 2024-03-07
 * @brief Tile Based Sampler for Discretely Sampling a list of standard Gaussian
 */

#include "SailInno/gaussian/diff_gs_tile_sampler.h"
#include "SailInno/util/graphic/image.h"
#include <luisa/dsl/sugar.h>

using namespace luisa;
using namespace luisa::compute;

namespace sail::inno::gaussian {

void DiffGaussianTileSampler::compile(Device& device) noexcept {
	mp_ndc2pix = luisa::make_unique<Callable<float(float, uint)>>([](Float v, UInt S) {
		return ((v + 1.0f) * S - 1.0f) * 0.5f;
	});

	mp_get_rect = luisa::make_unique<Callable<void(luisa::compute::float2, int, luisa::compute::uint2&, luisa::compute::uint2&, luisa::compute::uint2, luisa::compute::uint2)>>([](Float2 p, Int max_radius, UInt2& rect_min, UInt2& rect_max, UInt2 blocks, UInt2 grids) {
		// clamp
		rect_min = make_uint2(
			clamp(UInt((p.x - max_radius) / blocks.x), Var(0u), grids.x),
			clamp(UInt((p.y - max_radius) / blocks.y), Var(0u), grids.y));
		rect_max = make_uint2(
			clamp(UInt(p.x + max_radius + blocks.x - 1) / blocks.x, Var(0u), grids.x),
			clamp(UInt(p.y + max_radius + blocks.y - 1) / blocks.y, Var(0u), grids.y));
	});
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

		// Linear Transformation for Gaussian
		Float3 cov_2d = make_float3(
			covs_2d.read(3 * idx + 0) * resolution.x * resolution.x * 0.25f,
			covs_2d.read(3 * idx + 1) * resolution.x * resolution.y * 0.25f,
			covs_2d.read(3 * idx + 2) * resolution.y * resolution.y * 0.25f);

		Float det = cov_2d.x * cov_2d.z - cov_2d.y * cov_2d.y;
		Float inv_det = 1.0f / det;
		Float3 conic = inv_det * make_float3(cov_2d.z, -cov_2d.y, cov_2d.x);// inv: [0][0] [0][1] transpose [1][1] inverse

		Float mid = 0.5f * (cov_2d.x + cov_2d.z);
		Float lambda1 = mid + sqrt(max(0.1f, mid * mid - det));
		Float lambda2 = mid - sqrt(max(0.1f, mid * mid - det));

		// 3 \sigma as affected zone
		Int my_radius = ceil(3.0f * sqrt(max(lambda1, lambda2)));

		UInt2 rect_min, rect_max;

		auto point_image_ndc = make_float2(means_2d.read(2 * idx + 0), means_2d.read(2 * idx + 1));
		auto point_image = make_float2(util::ndc2pix<Float>(point_image_ndc.x, resolution.x), util::ndc2pix<Float>(point_image_ndc.y, resolution.y));

		(*mp_get_rect)(point_image, my_radius, rect_min, rect_max, m_blocks, grids);
		// write means_2d
		radii.write(idx, my_radius);
		tiles_touched.write(idx, (rect_max.x - rect_min.x) * (rect_max.y - rect_min.y));
		// write conic
		conics.write(3 * idx + 0, conic.x);
		conics.write(3 * idx + 1, conic.y);
		conics.write(3 * idx + 2, conic.z);

		// update means2d res
		means_2d_res.write(2 * idx + 0, point_image.x);
		means_2d_res.write(2 * idx + 1, point_image.y);
		//  means_2d_res.write(2 * idx + 0, point_image_ndc.x);
		//  means_2d_res.write(2 * idx + 1, point_image_ndc.y);
	});

	lazy_compile(device, m_copy_with_keys_shader,
				 [&](
					 Int P,
					 BufferVar<float> points_xy,
					 BufferVar<uint> offsets,
					 BufferVar<int> radii,
					 BufferVar<float> depth_features,
					 BufferVar<ulong> keys_unsorted,
					 BufferVar<uint> values_unsorted,
					 UInt2 blocks,
					 UInt2 grids) {
		auto idx = dispatch_id().x;
		$if(idx >= static_cast<$uint>(P)) { return; };
		auto radius = radii.read(idx);
		$if(radius > 0) {
			// generate key/value
			$uint off = 0u;
			$if(idx >= 1u) {
				off = offsets.read(idx - 1);
			};
			Float2 point_xy = make_float2(points_xy.read(2 * idx + 0), points_xy.read(2 * idx + 1));
			UInt2 rect_min, rect_max;

			(*mp_get_rect)(point_xy, radius, rect_min, rect_max, blocks, grids);
			$for(i, rect_min.x, rect_max.x) {
				$for(j, rect_min.y, rect_max.y) {
					ULong key = ULong(i + j * grids.x);
					key <<= 32ull;
					// TODO: directly bit-and float depth
					auto depth = depth_features.read(idx);
					key |= ULong(depth.as<UInt>());
					keys_unsorted.write(off, key);
					values_unsorted.write(off, idx);
					off = off + 1u;
				};
			};
		};
	});
	lazy_compile(device, m_get_ranges_shader, [&](Int L, BufferVar<ulong> point_list_keys, BufferVar<uint> ranges) {
		auto idx = dispatch_id().x;
		$if(idx >= L) { return; };
		ULong key = point_list_keys.read(idx);
		UInt curr_tile = UInt(key >> 32ull);
		UInt prev_tile = 0u;
		$if(idx > 0u) {
			prev_tile = UInt(point_list_keys.read(idx - 1) >> 32ull);
			$if(curr_tile != prev_tile) {
				ranges.write(2 * prev_tile + 1u, idx);
				ranges.write(2 * curr_tile + 0u, idx);
			};
		};
		$if(idx == L - 1) {
			ranges.write(2 * curr_tile + 1u, UInt(L));
		};
	});

	lazy_compile(device, m_forward_render_shader,
				 [&](
					 UInt2 resolution,
					 BufferVar<float> target_img,
					 UInt2 grids,
					 BufferVar<uint> ranges,
					 BufferVar<uint> point_list,
					 BufferVar<float> means_2d_res,
					 BufferVar<float> features,// 4 * features
					 BufferVar<float> conic,
					 BufferVar<uint> n_contrib,
					 BufferVar<float> accum_alpha) {
		set_block_size(m_blocks);
		auto xy = dispatch_id().xy();
		auto w = resolution.x;
		auto h = resolution.y;
		auto thread_idx = thread_id().x + thread_id().y * block_size().x;

		Bool inside = Bool(xy.x < resolution.x) & Bool(xy.y < resolution.y);
		Bool done = !inside;
		auto tile_xy = block_id();

		auto pix_id = xy.x + resolution.x * xy.y;
		auto pix_f = Float2(
			static_cast<Float>(xy.x),
			static_cast<Float>(xy.y));

		Int range_start = (Int)ranges.read(2 * (tile_xy.x + tile_xy.y * grids.x) + 0u);
		Int range_end = (Int)ranges.read(2 * (tile_xy.x + tile_xy.y * grids.x) + 1u);

		// background color
		Float3 color = make_float3(1.0f, 1.0f, 1.0f);
		// debug grid
		$if((tile_xy.x + tile_xy.y) % 2 == 0) {
			color = make_float3(0.0f, 0.0f, 0.0f);
		};

		// make rounds
		// round step = shared_mem_size = block_size = block_x * block_y
		const Int round_step = Int(m_shared_mem_size);

		const Int rounds = ((range_end - range_start + round_step - 1) / round_step);
		Int todo = range_end - range_start;

		Shared<int>* collected_ids = new Shared<int>(m_shared_mem_size);
		Shared<float2>* collected_means = new Shared<float2>(m_shared_mem_size);
		Shared<float4>* collected_conic_opacity = new Shared<float4>(m_shared_mem_size);

		Float T = 1.0f;
		Float3 C = make_float3(0.0f, 0.0f, 0.0f);
		UInt contributor = 0u;
		UInt last_contributor = 0u;

		$for(i, rounds) {
			// require __syncthreads_count(done) to accelerate
			Int progress = i * round_step + thread_idx;
			$if(progress + range_start < range_end) {
				Int coll_id = point_list.read(progress + range_start);
				collected_ids->write(thread_idx, coll_id);
				Float2 means = make_float2(
					means_2d_res.read(2 * coll_id + 0),
					means_2d_res.read(2 * coll_id + 1));
				collected_means->write(thread_idx, means);
				Float opacity = features.read(4 * coll_id + 3);

				Float4 conic_opacity = make_float4(
					conic.read(3 * coll_id + 0),
					conic.read(3 * coll_id + 1),
					conic.read(3 * coll_id + 2),
					opacity);
				collected_conic_opacity->write(thread_idx, conic_opacity);
			};
			sync_block();

			// iterate over the current batch

			$for(j, min(round_step, todo)) {
				$if(done) { $break; };
				contributor = contributor + 1u;

				Float2 mean = collected_means->read(j);
				Float2 d = mean - pix_f;
				Float4 con_o = collected_conic_opacity->read(j);

				Float power = -0.5f * (con_o.x * d.x * d.x + con_o.z * d.y * d.y) - con_o.y * d.x * d.y;
				$if(power > 0.0f) { $continue; };
				Float alpha = min(0.99f, con_o.w * exp(power));
				$if(alpha < 1.0f / 255.0f) { $continue; };
				Float test_T = T * (1.0f - alpha);
				$if(test_T < 0.0001f) {
					done = true;
					$continue;
				};
				auto id = collected_ids->read(j);
				Float3 feat = make_float3(
					features->read(4 * id + 0),
					features->read(4 * id + 1),
					features->read(4 * id + 2));
				//  feat = make_float3(1.0f, 0.0f, 0.0f);
				C = C + T * alpha * feat;
				T = test_T;

				last_contributor = contributor;
			};

			todo = todo - round_step;
		};

		$if(inside) {
			color = color * T + C;
			// todo: collect final T and last contributor
			$for(i, 0, 3) {
				target_img.write(pix_id + i * h * w, min(1.0f, color[i]));
			};
			accum_alpha.write(pix_id, T);
			n_contrib.write(pix_id, last_contributor);
		};
	});
}

}// namespace sail::inno::gaussian