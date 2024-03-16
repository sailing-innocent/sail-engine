#pragma once

/**
 * @file packages/painter/painter_base.h
 * @author sailing-innocent
 * @date 2023/12/27
 * @brief The Painter Base class, Pure Color
 */

#include <luisa/runtime/device.h>
#include <luisa/runtime/buffer.h>
#include <luisa/runtime/stream.h>
#include <luisa/dsl/sugar.h>

#include "SailInno/core/runtime.h"

namespace sail::inno {

class SAIL_INNO_API PainterBase : public LuisaModule {
public:
	PainterBase() = default;
	uint2 block_size = {16, 16};
	virtual ~PainterBase() = default;
	virtual void create(Device& device) noexcept;
	virtual void paint(CommandList& cmdlist, ImageView<float> out_img, int w, int h) noexcept;
	virtual void paint_sync(Device& device, Stream& stream, ImageView<float> out_img, int w, int h) noexcept;
	void update_clear_color(float3 color) noexcept { m_clear_color = color; }

protected:
	virtual void compile(Device& device) noexcept;
	U<Shader<2, Image<float>, int, int, float3>> ms_clear;
	float3 m_clear_color;
};

}// namespace sail::inno