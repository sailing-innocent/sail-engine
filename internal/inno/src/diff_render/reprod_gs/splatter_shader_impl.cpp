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
}
void ReprodGS::compile_get_ranges_shader(Device& device) noexcept {
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
}
void ReprodGS::compile_forward_render_shader(Device& device) noexcept {
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
		//  $if((tile_xy.x + tile_xy.y) % 2 == 0)
		//  {
		//      color = make_float3(0.0f, 0.0f, 0.0f);
		//  };

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

void ReprodGS::compile_backward_render_shader(Device& device) noexcept {
	lazy_compile(device, m_backward_render_shader,
				 [&](
					 // input
					 BufferVar<float> dL_d_pix,
					 // output
					 BufferVar<float> dL_d_means_2d,// 2 * P
					 BufferVar<float> dL_d_conic,	// 3 * P
					 BufferVar<float> dL_d_color_feature,
					 BufferVar<float> dL_d_opacity,// 4 * P
					 // params
					 UInt2 resolution,
					 UInt2 grids,
					 BufferVar<float> result_img,
					 BufferVar<float2> means_2d,
					 BufferVar<uint> ranges,
					 BufferVar<uint> point_list,
					 BufferVar<float4> features,
					 BufferVar<float4> conic_opacity,
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
		Float3 bg_color = make_float3(1.0f, 1.0f, 1.0f);
		//  $if((tile_xy.x + tile_xy.y) % 2 == 0)
		//  {
		//      color = make_float3(0.0f, 0.0f, 0.0f);
		//  };
		// make rounds
		// round step = shared_mem_size = block_size = block_x * block_y
		const Int round_step = Int(m_shared_mem_size);

		const Int rounds = ((range_end - range_start + round_step - 1) / round_step);
		Int todo = range_end - range_start;

		Shared<uint>* collected_ids = new Shared<uint>(m_shared_mem_size);
		Shared<float2>* collected_means = new Shared<float2>(m_shared_mem_size);
		Shared<float4>* collected_conic_opacity = new Shared<float4>(m_shared_mem_size);
		Shared<float>* collected_color = new Shared<float>(m_shared_mem_size * 3);

		Float T_final = 0.0f;
		$if(inside) {
			T_final = accum_alpha.read(pix_id);
		};
		Float T = T_final;

		UInt contributor = todo;
		UInt last_contributor = 0u;
		$if(inside) {
			last_contributor = n_contrib.read(pix_id);
		};

		Float3 dLdpix = make_float3(
			dL_d_pix.read(pix_id + w * h * 0),
			dL_d_pix.read(pix_id + w * h * 1),
			dL_d_pix.read(pix_id + w * h * 2));

		Float3 accum_rec = make_float3(0.0f);
		Float last_alpha = 0.0f;
		Float3 last_color = make_float3(0.0f);
		// Traverse all Gaussians
		$for(i, rounds) {
			sync_block();
			Int progress = i * round_step + thread_idx;
			$if(progress + range_start < range_end) {
				//  Int coll_id = point_list.read(progress + range_start);
				Int coll_id = point_list.read(range_end - 1 - progress);
				collected_ids->write(thread_idx, coll_id);
				collected_means->write(thread_idx, means_2d.read(coll_id));
				collected_conic_opacity->write(thread_idx, conic_opacity.read(coll_id));
				auto color = features.read(coll_id);
				$for(ch, 3) {
					collected_color->write(3 * thread_idx + ch, color[ch]);
				};
			};
			sync_block();
			// iterate over the Gaussians of current batch

			$for(j, min(round_step, todo)) {
				contributor = contributor - 1u;
				$if(contributor >= last_contributor) {
					// no contribution to color, pass
					$continue;
				};
				// calcuate alpha
				Float2 mean = collected_means->read(j);
				Float2 d = mean - pix_f;
				Float4 con_o = collected_conic_opacity->read(j);

				Float power = -0.5f * (con_o.x * d.x * d.x + con_o.z * d.y * d.y) - con_o.y * d.x * d.y;
				$if(power > 0.0f) { $continue; };

				Float G = exp(power);
				Float alpha = min(0.99f, con_o.w * G);
				$if(alpha < 1.0f / 255.0f) { $continue; };

				T = T / (1.0f - alpha);// backward calc weight

				Float d_ch_d_color = alpha * T;
				//  Float d_ch_d_color = 1.0f;
				Float dL_dalpha = 0.0f;
				UInt global_id = collected_ids->read(j);
				$for(ch, 3) {
					Float c = collected_color->read(3 * j + ch);
					accum_rec[ch] = last_alpha * last_color[ch] + (1.0f - last_alpha) * accum_rec[ch];
					last_color[ch] = c;
					//  Float dL_d_ch = 1.0f;
					Float dL_d_ch = dLdpix[ch];
					dL_dalpha += (c - accum_rec[ch]) * dL_d_ch;
					// atomic add to dL_dcolor
					Float dL_d_color_feat = d_ch_d_color * dL_d_ch;
					dL_d_color_feature.atomic(4 * global_id + ch)
						.fetch_add(dL_d_color_feat);
				};

				dL_dalpha = dL_dalpha * T;
				last_alpha = alpha;

				Float bg_dot_pixel = 0;
				$for(k, 3) {
					bg_dot_pixel += bg_color[k] * dLdpix[k];
				};
				dL_dalpha += (-T_final / (1.0f - alpha)) * bg_dot_pixel;

				Float dL_dG = dL_dalpha * con_o.w;
				Float gdx = G * d.x;
				Float gdy = G * d.y;

				// backward conic
				dL_d_conic.atomic(global_id * 3 + 0).fetch_add(-0.5f * gdx * d.x * dL_dG);
				dL_d_conic.atomic(global_id * 3 + 1).fetch_add(-0.5f * gdx * d.y * dL_dG);
				dL_d_conic.atomic(global_id * 3 + 2).fetch_add(gdy * d.y * dL_dG);

				// TODO backward means 2D
				// backward for opacity
				dL_d_opacity.atomic(global_id).fetch_add(G * dL_dalpha);
			};

			todo = todo - round_step;
		};
	});
}

}// namespace sail::inno::render
