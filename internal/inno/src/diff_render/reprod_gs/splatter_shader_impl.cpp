/**
 * @file package/diff_render/gs/gaussian_splatter_shader.cpp
 * @author sailing-innocent
 * @date 2023-12-27
 * @brief The Gaussian Splatter Shader
 */

#include "SailInno/diff_render/reprod_gs_splatter.h"
#include <luisa/dsl/sugar.h>
#include "SailInno/util/math/gaussian.h"

using namespace luisa;
using namespace luisa::compute;

namespace sail::inno::render {

void ReprodGS::compile_copy_with_keys_shader(Device& device) noexcept {
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
			$for(j, rect_min.y, rect_max.y) {
				$for(i, rect_min.x, rect_max.x) {
					ULong key = ULong(i + j * grids.x);
					key <<= 32ull;
					// TODO: directly bit-and float depth
					auto depth = depth_features.read(idx);

					key |= ULong(depth.as<UInt>()) & 0x00000000FFFFFFFFull;
					keys_unsorted.write(off, key);
					values_unsorted.write(off, idx);
					off = off + 1u;
				};
			};
		};
	});
}
void ReprodGS::compile_get_ranges_shader(Device& device) noexcept {
	lazy_compile(device, m_get_ranges_shader, [&](Int L, BufferVar<ulong> point_list_keys, BufferVar<uint> ranges) {
		auto idx = dispatch_id().x;
		$if(idx >= L) { return; };
		ULong key = point_list_keys.read(idx);
		UInt curr_tile = UInt(key >> 32ull);
		UInt prev_tile = 0u;
		$if(idx == 0u) {
			ranges.write(2 * curr_tile + 0u, 0u);
		}
		$else {
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
}

}// namespace sail::inno::render
