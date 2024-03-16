#pragma once
/**
 * @file render/point_render.h
 * @author sailing-innocent
 * @date 2023-12-27
 * @brief The Point Visualize Render
 */

#include "SailInno/core/runtime.h"
#include <luisa/runtime/buffer.h>

namespace sail::inno::render {

class SAIL_INNO_API PointRender : public LuisaModule {
public:
	PointRender() = default;
	~PointRender() = default;
	// render to image
	void render(CommandList& cmdlist, ImageView<float> target_img, int width, int height, int P, BufferView<float> xyz, BufferView<float> color, float4x4 view_matrix, float4x4 proj_matrix);
	// render to buffer
	void render(CommandList& cmdlist, BufferView<float> target_img, int width, int height, int P, BufferView<float> xyz, BufferView<float> color, float4x4 view_matrix, float4x4 proj_matrix);

	void compile(Device& device, int stride = 3);

protected:
	U<Shader<1, Image<float>, int, int, int, Buffer<float>, Buffer<float>, float4x4, float4x4>> ms_render_img;
	U<Shader<1, Buffer<float>, int, int, int, Buffer<float>, Buffer<float>, float4x4, float4x4>> ms_render_buffer;
};

}// namespace sail::inno::render