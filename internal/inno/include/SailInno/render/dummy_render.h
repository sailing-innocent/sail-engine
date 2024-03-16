#pragma once

/**
 * @file render/dummy_render.h
 * @author sailing-innocent
 * @date 2023/12/26
 * @brief The Dummy Render Implementation
*/

#include "SailInno/core/runtime.h"

namespace sail::inno::render {

class SAIL_INNO_API DummyRender {
	template<typename T>
	using Buffer = luisa::compute::Buffer<T>;
	template<typename T>
	using BufferView = luisa::compute::BufferView<T>;
	template<size_t I, typename... Ts>
	using Shader = luisa::compute::Shader<I, Ts...>;
	using Device = luisa::compute::Device;
	using CommandList = luisa::compute::CommandList;

public:
	DummyRender() = default;
	~DummyRender() = default;
	void render(CommandList& cmdlist, BufferView<float> target_img, int height, int width);
	void compile(Device& device);

protected:
	U<Shader<1, Buffer<float>, int, int>> ms_render;
};

}// namespace sail::inno::render