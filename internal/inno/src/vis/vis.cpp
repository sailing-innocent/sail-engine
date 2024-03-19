/**
 * @file package/vis/vis.cpp
 * @author sailing-innocent
 * @date 2023-12-27
 * @brief Visualizer impl
*/

#include "SailInno/vis/vis.h"
#include "luisa/runtime/rhi/resource.h"

using namespace luisa;
using namespace luisa::compute;

namespace sail::inno {

void Visualizer::create(Device& device, Stream& stream) noexcept {
	mp_window = luisa::make_unique<Window>(name, resolution);
	SwapchainOption option{
		.window = mp_window->native_handle(),
		.display = mp_window->native_display(),
		.size = resolution};
	mp_swapchain = luisa::make_unique<Swapchain>(device.create_swapchain(
		stream, option));
	m_display_img = device.create_image<float>(mp_swapchain->backend_storage(), resolution.x, resolution.y);
}

void Visualizer::vis(CommandList& cmdlist) noexcept {
	for (auto& painter : mp_painters) {
		painter->paint(cmdlist, m_display_img, resolution.x, resolution.y);
	}
}

void Visualizer::vis_sync(Device& device, Stream& stream) noexcept {
	for (auto& painter : mp_painters) {
		painter->paint_sync(device, stream, m_display_img, resolution.x, resolution.y);
	}
}

}// namespace sail::inno
