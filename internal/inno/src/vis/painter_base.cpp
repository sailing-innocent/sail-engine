/**
 * @file packages/painter/painter_base.cpp
 * @author sailing-innocent
 * @date 2023/12/27
 * @brief The Painter Base class, Impl
*/

#include "SailInno/vis/painter_base.h"
#include <luisa/dsl/sugar.h>

using namespace luisa;
using namespace luisa::compute;
namespace sail::inno {

void PainterBase::create(Device& device) noexcept {
	compile(device);
}

void PainterBase::paint(CommandList& cmdlist, ImageView<float> out_img, int w, int h) noexcept {
	cmdlist << (*ms_clear)(out_img, w, h, m_clear_color).dispatch(w, h);
}

void PainterBase::paint_sync(Device& device, Stream& stream, ImageView<float> out_img, int w, int h) noexcept {
	stream << (*ms_clear)(out_img, w, h, m_clear_color).dispatch(w, h);
}

}// namespace sail::inno

// Core

namespace sail::inno {

void PainterBase::compile(Device& device) noexcept {
	lazy_compile(device, ms_clear, [&](ImageFloat img, Int w, Int h, Float3 color) {
		set_block_size(block_size);
		auto xy = dispatch_id().xy();
		$if(xy.x >= w | xy.y >= h) { return; };
		img.write(xy, make_float4(color, 1.0f));
	});
}

}// namespace sail::inno