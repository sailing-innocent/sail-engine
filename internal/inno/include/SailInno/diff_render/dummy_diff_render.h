#pragma once
/**
 * @file app/dummy_diff_render.h
 * @author sailing-innocent
 * @date 2023-12-29
 * @brief the dummy diff render
 */

#include "SailInno/core/runtime.h"
#include <luisa/runtime/buffer.h>

namespace sail::inno::render {

class DummyDiffRender : public LuisaModule {
public:
	DummyDiffRender() = default;
	~DummyDiffRender() = default;
	// API
	void create(Device& device) noexcept;
	void forward_impl(
		CommandList& cmdlist,
		int width, int height,
		BufferView<float> source_img_buffer,
		BufferView<float> target_img_buffer) noexcept;

	void backward_impl(
		CommandList& cmdlist,
		int width, int height,
		BufferView<float> dL_dtpix,
		BufferView<float> dL_dspix) noexcept;

private:
	void compile(Device& device) noexcept;

private:
	// shaders
	U<Shader<1,
			 int, int,	   // w,h
			 Buffer<float>,// source_img_buf
			 Buffer<float> // target_img_buf
			 >>
		ms_forward;
	U<Shader<1,
			 int, int,	   // w,h
			 Buffer<float>,// dL_dtpix
			 Buffer<float> // dL_dspix
			 >>
		ms_backward;
};

}// namespace sail::inno::render