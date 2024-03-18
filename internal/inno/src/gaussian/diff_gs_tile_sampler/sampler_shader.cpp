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
	mp_get_rect = luisa::make_unique<Callable<void(luisa::compute::float2, int, luisa::compute::uint2&, luisa::compute::uint2&, luisa::compute::uint2, luisa::compute::uint2)>>([](Float2 p, Int max_radius, UInt2& rect_min, UInt2& rect_max, UInt2 blocks, UInt2 grids) {
		// clamp
		rect_min = make_uint2(
			clamp(UInt((p.x - max_radius) / blocks.x), Var(0u), grids.x),
			clamp(UInt((p.y - max_radius) / blocks.y), Var(0u), grids.y));
		rect_max = make_uint2(
			clamp(UInt(p.x + max_radius + blocks.x - 1) / blocks.x, Var(0u), grids.x),
			clamp(UInt(p.y + max_radius + blocks.y - 1) / blocks.y, Var(0u), grids.y));
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

	compile_tile_split_shader(device);
	compile_render_shader(device);
}

}// namespace sail::inno::gaussian