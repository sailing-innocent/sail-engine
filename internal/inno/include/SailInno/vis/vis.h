#pragma once
/**
 * @file visualizer/base_visualizer.h
 * @author sailing-innocent
 * @date 2023/04/09
 * @brief the base visualizer
*/

#include <luisa/runtime/buffer.h>
#include <luisa/runtime/device.h>
#include <luisa/runtime/stream.h>
#include <luisa/runtime/context.h>
#include <luisa/runtime/swapchain.h>
#include <luisa/gui/window.h>

#include "SailInno/core/runtime.h"
#include "SailInno/vis/painter_base.h"

namespace sail::inno {

class SAIL_INNO_API Visualizer {
	using Device = luisa::compute::Device;
	using Stream = luisa::compute::Stream;
	using CommandList = luisa::compute::CommandList;

public:
	Visualizer() = default;
	~Visualizer() = default;
	void create(Device& device, Stream& stream) noexcept;
	void vis(CommandList& cmdlist) noexcept;
	void vis_sync(Device& device, Stream& stream) noexcept;

	luisa::string name = "BasicVisualizer";
	luisa::uint2 resolution = {1024u, 768u};

	bool running() {
		// if window not init, return false
		if (!mp_window) {
			return false;
		}
		return !mp_window->should_close();
	}
	void present_and_poll_events(Stream& stream) noexcept {
		stream << mp_swapchain->present(m_display_img);
		mp_window->poll_events();
	}
	void add_painter(S<PainterBase> p_painter) noexcept {
		mp_painters.push_back(p_painter);
	}

protected:
	U<luisa::compute::Swapchain> mp_swapchain;
	U<luisa::compute::Window> mp_window;
	luisa::compute::Image<float> m_display_img;
	luisa::vector<S<PainterBase>> mp_painters;
};

}// namespace sail::inno