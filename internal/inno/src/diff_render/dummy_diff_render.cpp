/**
 * @file diff_render/dummy_render.cpp
 * @brief the implementation of dummy differential renderer
 * @author sailing-innocent
 * @date 2023-11-02
 */

#include "SailInno/diff_render/dummy_diff_render.h"
#include <luisa/dsl/sugar.h>

using namespace luisa;
using namespace luisa::compute;

// API

namespace sail::inno::render {

void DummyDiffRender::create(Device& device) noexcept {
	compile(device);
}

void DummyDiffRender::forward_impl(
	CommandList& cmdlist,
	int width, int height,
	BufferView<float> source_img_buffer,
	BufferView<float> target_img_buffer) noexcept {
	cmdlist << (*ms_forward)(width, height, source_img_buffer, target_img_buffer).dispatch(height * width);
}

void DummyDiffRender::backward_impl(
	CommandList& cmdlist,
	int width, int height,
	BufferView<float> dL_dtpix,
	BufferView<float> dL_dspix) noexcept {
	cmdlist << (*ms_backward)(width, height, dL_dtpix, dL_dspix).dispatch(height * width);
}

}// namespace sail::inno::render

// Core

namespace sail::inno::render {

void DummyDiffRender::compile(Device& device) noexcept {
	lazy_compile(device, ms_forward, [](Int height, Int width, BufferFloat source_img, BufferFloat target_img) {
		set_block_size(64);
		auto idx = dispatch_x();
		$if(idx >= width * height) { return; };
		auto r = source_img.read(idx * 3 + 0);
		auto g = source_img.read(idx * 3 + 1);
		auto b = source_img.read(idx * 3 + 2);
		target_img.write(idx * 3 + 0, 1.0f - r);
		target_img.write(idx * 3 + 1, 1.0f - g);
		target_img.write(idx * 3 + 2, 1.0f - b);
	});

	lazy_compile(device, ms_backward, [](Int height, Int width, BufferFloat dL_dtpix, BufferFloat dL_dspix) {
		set_block_size(64);
		auto idx = dispatch_x();
		$if(idx >= width * height) { return; };
		auto dL_dtr = dL_dtpix.read(idx * 3 + 0);
		auto dL_dtg = dL_dtpix.read(idx * 3 + 1);
		auto dL_dtb = dL_dtpix.read(idx * 3 + 2);

		dL_dspix.write(idx * 3 + 0, -dL_dtr);
		dL_dspix.write(idx * 3 + 1, -dL_dtg);
		dL_dspix.write(idx * 3 + 2, -dL_dtb);
	});
}

}// namespace sail::inno::render