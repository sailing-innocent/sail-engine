/**
 * @file imgui_window.cpp
 * @brief ImGuiWindow impl
 * @author sailing-innocent
 * @date 2024-05-19
 */
#include <mutex>
#include <random>

#if defined(_WIN32)
#define GLFW_EXPOSE_NATIVE_WIN32
#elif defined(LUISA_PLATFORM_APPLE)
#define GLFW_EXPOSE_NATIVE_COCOA
#else
#define GLFW_EXPOSE_NATIVE_X11// TODO: other window compositors
#endif

#ifndef GLFW_INCLUDE_NONE
#define GLFW_INCLUDE_NONE
#endif

#include <GLFW/glfw3.h>
#include <GLFW/glfw3native.h>

#include <imgui.h>
#include <imgui_impl_glfw.h>

#ifdef Bool// good job!
#undef Bool
#endif

#ifdef True// better!
#undef True
#endif

#ifdef False// best!
#undef False
#endif

#ifdef Always// ...
#undef Always
#endif

#ifdef None// speechless
#undef None
#endif

#include <luisa/core/logging.h>
#include <luisa/core/stl/queue.h>
#include <luisa/core/stl/vector.h>
#include <luisa/core/stl/map.h>
#include <luisa/runtime/device.h>
#include <luisa/runtime/shader.h>
#include <luisa/runtime/stream.h>
#include <luisa/runtime/buffer.h>
#include <luisa/runtime/image.h>
#include <luisa/runtime/bindless_array.h>
#include <luisa/runtime/swapchain.h>
#include <luisa/runtime/rtx/accel.h>
#include <luisa/gui/window.h>
#include <luisa/backends/ext/raster_ext.hpp>
#include <luisa/runtime/raster/raster_shader.h>
#include <luisa/dsl/sugar.h>

#include "SailInno/util/imgui_window.h"

// Core Impl
namespace sail::inno {
namespace detail {
struct GUIObjectData {
	uint tex_id;
	std::array<float, 2> min_view;
	std::array<float, 2> max_view;
};
struct GUIVertex {
	std::array<float, 2> pos;
	std::array<float, 2> uv;
	uint32_t col;
};
static_assert(sizeof(GUIVertex) == sizeof(ImDrawVert) && alignof(GUIVertex) == alignof(ImDrawVert));

[[nodiscard]] inline auto glfw_window_native_handle(GLFWwindow* window) noexcept {
#if defined(_WIN32)
	return reinterpret_cast<uint64_t>(glfwGetWin32Window(window));
#elif defined(LUISA_PLATFORM_APPLE)
	return reinterpret_cast<uint64_t>(glfwGetCocoaWindow(window));
#else
	return reinterpret_cast<uint64_t>(glfwGetX11Window(window));
#endif
}
}// namespace detail

class ImGuiWindow::Impl {
	class ContextGuard {};
	Device& m_device;
	Stream& m_stream;
	luisa::compute::Window m_main_window;

public:
	explicit Impl(Device& device, Stream& stream, luisa::string name, luisa::string_view shader_path, const Config& config) noexcept
		: m_device{device},
		  m_stream{stream},
		  m_main_window{name, config.size.x, config.size.y, config.resizable, config.full_screen} {
		// raster state
		// rebuild_swapchain_if_changed
		// clear shader
	}
	~Impl() noexcept {}

private:
	// rebuild swapchain if changed
	// on_imgui_create_window
	// on_imgui_destroy_window
	// on_imgui_set_window_size
	// on_imgui_render_window

public:
	[[nodiscard]] auto handle() const noexcept { return m_main_window.window(); }
	[[nodiscard]] auto& window() noexcept { return m_main_window; }
	[[nodiscard]] bool should_close() const noexcept {
		return static_cast<bool>(glfwWindowShouldClose(handle()));
	}
	[[nodiscard]] bool set_should_close(bool b) noexcept {
		glfwSetWindowShouldClose(handle(), b);
	}
	// register textures
	// unregister textures
private:
	// creat font texture

public:
	// lifecycle
	// prepare frame
	// render frame
private:
	bool m_is_inside_frame = false;
	ImGuiContext* m_old_ctx{nullptr};
};

}// namespace sail::inno

// API
namespace sail::inno {

ImGuiWindow::ImGuiWindow(Device& device, Stream& stream, luisa::string name, luisa::string_view shader_path, const Config& config) noexcept
	: ImGuiWindow{} {
	// create
}
ImGuiWindow::~ImGuiWindow() noexcept = default;

GLFWwindow* ImGuiWindow::handle() const noexcept {
	LUISA_ASSERT(mp_impl != nullptr, "ImGuiWindow is not created.");
	return mp_impl->handle();
}

void ImGuiWindow::create(Device& device, Stream& stream,
						 luisa::string name,
						 luisa::string_view shader_path,
						 const Config& config) noexcept {

	destroy();
	mp_impl = luisa::make_unique<Impl>(device, stream, std::move(name), shader_path, config);
}
void ImGuiWindow::destroy() noexcept {}

}// namespace sail::inno
