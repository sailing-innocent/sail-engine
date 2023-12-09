#pragma once

/**
 * @file render/dummy_render.h
 * @author sailing-innocent
 * @date 2023/12/26
 * @brief The Dummy Render Implementation
*/

#include "SailInno/core/runtime.h"

namespace sail::inno::render {

class SAIL_INNO_API DummyRender : public LuisaModule {
public:
	DummyRender() = default;
	~DummyRender() = default;
	void render(CommandList& cmdlist, BufferView<float> target_img, int height, int width);
	void compile(Device& device);

protected:
	U<Shader<1, Buffer<float>, int, int>> ms_render;
};

}// namespace sail::inno::render