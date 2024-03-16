#include "SailInno/render/dummy_render.h"
#include <luisa/dsl/sugar.h>
#include <luisa/dsl/resource.h>

namespace sail::inno::render {

using namespace luisa;
using namespace luisa::compute;

void DummyRender::render(CommandList& cmdlist, BufferView<float> target_img, int height, int width) noexcept {
	LUISA_INFO("DummyRender::render {} x {}", height, width);
	cmdlist << (*ms_render)(target_img, height, width).dispatch(width * height);
}

void DummyRender::compile(Device& device) noexcept {
	LUISA_INFO("DummyRender::compile");
	lazy_compile(device, ms_render, [&](BufferFloat target_img, Int height, Int width) {
		set_block_size(64);
		auto idx = dispatch_x();
		$if(idx >= width * height) { return; };
		target_img.write(idx * 3 + 0, 1.0f);
		target_img.write(idx * 3 + 1, 0.0f);
		target_img.write(idx * 3 + 2, 0.0f);
	});
}

}// namespace sail::inno::render