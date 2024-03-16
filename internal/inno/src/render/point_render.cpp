/**
 * @file render/point_render.h
 * @author sailing-innocent
 * @date 2023/12/27
 * @brief The Point Visualize Render Implementation
 */

#include "SailInno/render/point_render.h"
#include <luisa/dsl/sugar.h>

using namespace luisa;
using namespace luisa::compute;

// API

namespace sail::inno::render {

void PointRender::render(CommandList& cmdlist, ImageView<float> target_img, int width, int height, int P, BufferView<float> xyz, BufferView<float> color, float4x4 view_matrix, float4x4 proj_matrix) {
	cmdlist << (*ms_render_img)(target_img, width, height, P, xyz, color, view_matrix, proj_matrix).dispatch(P);
}

void PointRender::render(CommandList& cmdlist, BufferView<float> target_img, int width, int height, int P, BufferView<float> xyz, BufferView<float> color, float4x4 view_matrix, float4x4 proj_matrix) {
	LUISA_INFO("render to buffer with {} points", P);
	cmdlist << (*ms_render_buffer)(target_img, width, height, P, xyz, color, view_matrix, proj_matrix).dispatch(P);
}

}// namespace sail::inno::render

// Core

namespace sail::inno::render {

void PointRender::compile(Device& device, int stride) {
	auto T = [](Float3 a0, Float4x4 proj_view) noexcept {
		auto a_hom = make_float4(a0, 1.f);
		auto a = proj_view * a_hom;
		return make_float2(a.x, a.y) / a.w;
	};

	// auto T = [](Float3 a0, Float4x4 proj_view) noexcept {
	// 	return make_float2(a0.x, a0.y);
	// };

	lazy_compile(device, ms_render_img, [&T, stride](ImageFloat img, Int w, Int h, Int P, BufferFloat xyz, BufferFloat color, Float4x4 view_matrix, Float4x4 proj_matrix) {
		auto p = dispatch_id().x;
		$if(p < P) { return; };
		auto origin_pos = make_float4(
			xyz.read(stride * p + 0),
			xyz.read(stride * p + 1),
			xyz.read(stride * p + 2), 1.0f);
		auto base_pos = view_matrix * origin_pos;
		base_pos = proj_matrix * base_pos;
		base_pos = base_pos / base_pos.w;
		// auto base_color = origin_pos;
		auto base_color = make_float4(
			color.read(stride * p + 0),
			color.read(stride * p + 1),
			color.read(stride * p + 2), 1.0f);
		// splat 3x3
		auto wf = static_cast<Float>(w);
		auto hf = static_cast<Float>(h);

		for (auto i = -2; i <= 2; i++) {
			for (auto j = -2; j <= 2; j++) {
				auto pos = make_int2(static_cast<Int>(base_pos.x * wf), static_cast<Int>(base_pos.y * hf)) +
						   make_int2(i, j);

				auto hix = static_cast<Int>(wf * 0.5f);
				auto hiy = static_cast<Int>(hf * 0.5f);
				$if(pos.x >= -hix & pos.x < hix & pos.y >= -hiy & pos.y < hiy) {
					img.write(
						make_uint2(static_cast<UInt>(pos.x + hix), static_cast<UInt>(hiy - pos.y - 1)),
						base_color);
				};
			}
		}
	});

	lazy_compile(device, ms_render_buffer, [&stride](BufferFloat img, Int w, Int h, Int P, BufferFloat xyz, BufferFloat color, Float4x4 view_matrix, Float4x4 proj_matrix) {
		auto p = dispatch_id().x;
		$if(p < P) { return; };
		auto origin_pos = make_float4(
			xyz.read(stride * p + 0),
			xyz.read(stride * p + 1),
			xyz.read(stride * p + 2), 1.0f);
		auto base_pos = view_matrix * origin_pos;
		base_pos = proj_matrix * base_pos;
		base_pos = base_pos / base_pos.w;
		// auto base_color = origin_pos;
		auto base_color = make_float4(
			color.read(stride * p + 0),
			color.read(stride * p + 1),
			color.read(stride * p + 2), 1.0f);
		// splat 3x3
		auto wf = static_cast<Float>(w);
		auto hf = static_cast<Float>(h);

		for (auto i = -2; i <= 2; i++) {
			for (auto j = -2; j <= 2; j++) {
				auto pos = make_int2(static_cast<Int>(base_pos.x * wf), static_cast<Int>(base_pos.y * hf)) +
						   make_int2(i, j);

				auto hix = static_cast<Int>(wf * 0.5f);
				auto hiy = static_cast<Int>(hf * 0.5f);
				$if(pos.x >= -hix & pos.x < hix & pos.y >= -hiy & pos.y < hiy) {
					auto idx = (pos.x + hix) + (hiy - pos.y - 1) * w;
					img.write(idx * 3 + 0, base_color.x);
					img.write(idx * 3 + 1, base_color.y);
					img.write(idx * 3 + 2, base_color.z);
				};
			}
		}
	});
}

}// namespace sail::inno::render