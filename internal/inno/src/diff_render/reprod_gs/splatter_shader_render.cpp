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

void ReprodGS::compile_forward_render_shader(Device& device) noexcept {
	lazy_compile(device, m_forward_render_shader,
				 [&](
					 UInt2 resolution,
					 BufferVar<float> target_img,
					 // output
					 UInt2 grids,
					 BufferVar<uint> ranges,
					 BufferVar<uint> point_list,
					 BufferVar<float> means_2d,
					 BufferVar<float> conic,		   // 3 * P
					 BufferVar<float> opacity_features,// P
					 BufferVar<float> color_features,  // 3 * P
					 // save for backward
					 BufferVar<uint> n_contrib,
					 BufferVar<float> accum_alpha) {
		set_block_size(m_blocks);
		auto xy = dispatch_id().xy();
		auto w = resolution.x;
		auto h = resolution.y;

		auto thread_idx = thread_id().x + thread_id().y * block_size().x;
		// auto thread_idx = thread_id().y + thread_id().x * block_size().y;

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
		// debug grid
		// $if((tile_xy.x + tile_xy.y) % 2 == 0) {
		// 	color = make_float3(0.0f, 0.0f, 0.0f);
		// };

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
			// $if(done) {
			// 	$continue;
			// };
			sync_block();

			Int progress = i * round_step + thread_idx;

			$if(progress + range_start < range_end) {
				Int coll_id = point_list.read(progress + range_start);
				collected_ids->write(thread_idx, coll_id);
				Float2 means = make_float2(
					means_2d.read(2 * coll_id + 0),
					means_2d.read(2 * coll_id + 1));
				collected_means->write(thread_idx, means);
				Float opacity = opacity_features.read(coll_id);
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
				// $if(done) { $break; };
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
					color_features->read(3 * id + 0),
					color_features->read(3 * id + 1),
					color_features->read(3 * id + 2));
				// feat = make_float3(0.2, 0.2f, 0.2f);
				C = C + T * alpha * feat;
				T = test_T;
				last_contributor = contributor;
			};

			todo = todo - round_step;
		};

		$if(inside) {
			auto color = bg_color * T + C;
			$for(i, 0, 3) {
				target_img.write(pix_id + i * h * w, color[i]);
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
					 BufferVar<float> dL_d_means_2d,	 // 2 * P
					 BufferVar<float> dL_d_conic,		 // 3 * P
					 BufferVar<float> dL_d_color_feature,// 3 * P
					 BufferVar<float> dL_d_opacity,		 // P
					 // params
					 UInt2 resolution,
					 UInt2 grids,
					 BufferVar<float> result_img,
					 BufferVar<uint> ranges,
					 BufferVar<uint> point_list,
					 BufferVar<float> means_2d,
					 BufferVar<float> conic,
					 BufferVar<float> opacity_features,
					 BufferVar<float> color_features,
					 BufferVar<uint> n_contrib,
					 BufferVar<float> accum_alpha) {
		set_block_size(m_blocks);
		auto xy = dispatch_id().xy();
		auto w = resolution.x;
		auto h = resolution.y;
		Float ddelx_dx = 0.5f * w;
		Float ddely_dy = 0.5f * w;

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
		$if((tile_xy.x + tile_xy.y) % 2 == 0) {
			bg_color = make_float3(0.0f, 0.0f, 0.0f);
		};
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
				collected_means->write(thread_idx, make_float2(
													   means_2d.read(2 * coll_id + 0),
													   means_2d.read(2 * coll_id + 1)));

				Float opacity = opacity_features.read(coll_id);
				collected_conic_opacity->write(thread_idx, make_float4(
															   conic.read(3 * coll_id + 0),
															   conic.read(3 * coll_id + 1),
															   conic.read(3 * coll_id + 2), opacity));

				$for(ch, 3) {
					auto n = color_features.read(3 * coll_id + ch);
					collected_color->write(3 * thread_idx + ch, n);
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

					dL_d_color_feature.atomic(3 * global_id + ch)
						.fetch_add(dL_d_color_feat);
				};

				dL_dalpha = dL_dalpha * T;
				last_alpha = alpha;

				Float bg_dot_pixel = 0;
				$for(k, 3) {
					bg_dot_pixel += bg_color[k] * dLdpix[k];
				};
				dL_dalpha += (-T_final / (1.0f - alpha)) * bg_dot_pixel;
				// backward for opacity
				dL_d_opacity.atomic(global_id).fetch_add(G * dL_dalpha);

				Float dL_dG = dL_dalpha * con_o.w;
				Float gdx = G * d.x;
				Float gdy = G * d.y;
				auto dG_ddelx = -gdx * con_o.x - gdy * con_o.y;
				auto dG_ddely = -gdy * con_o.z - gdx * con_o.y;

				// backward conic
				dL_d_conic.atomic(global_id * 3 + 0).fetch_add(-0.5f * gdx * d.x * dL_dG);
				dL_d_conic.atomic(global_id * 3 + 1).fetch_add(-0.5f * gdx * d.y * dL_dG);
				dL_d_conic.atomic(global_id * 3 + 2).fetch_add(-0.5f * gdy * d.y * dL_dG);
				dL_d_means_2d.atomic(global_id * 2 + 0).fetch_add(dL_dG * dG_ddelx * ddelx_dx);
				dL_d_means_2d.atomic(global_id * 2 + 1).fetch_add(dL_dG * dG_ddely * ddely_dy);
			};

			todo = todo - round_step;
		};
	});
}
}// namespace sail::inno::render