/**
 * @file imgui_window.cpp
 * @brief The Implementation of ImGUI Window
 * @author sailing-innocent
 * @date 2024-05-01
 */

#if defined(_WIN32)
#define GLFW_EXPOSE_NATIVE_WIN32
#endif
#ifndef GLFW_INCLUDE_NONE
#define GLFW_INCLUDE_NONE
#endif
#include <GLFW/glfw3.h>
#include <GLFW/glfw3native.h>
#include <imgui.h>
#include <imgui_impl_glfw.h>

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
#include <luisa/backends/ext/raster_ext.hpp>
#include <luisa/runtime/raster/raster_shader.h>
#include <luisa/dsl/sugar.h>

#include "SailInno/window/imgui_window.h"

namespace sail::inno {

namespace detail {
struct GUIObjectData {};
struct GUIVertex {};
[[nodiscard]] inline auto glfw_window_native_handle(GLFWwindow* window) noexcept {
#if defined(_WIN32)
	return reinterpret_cast<uint64_t>(glfwGetWin32Window(window));
#endif
}
}// namespace detail

// -----------------------
// class ImGuiWindow::Impl
// The Implementation
// -----------------------
class ImGuiWindow::Impl {
	class CtxGaurd {};
	Window m_main_window;

public:
	Impl() = default;
	~Impl() noexcept {}
	// lifecycle
	[[nodiscard]] bool should_close() const noexcept {
		return static_cast<bool>(glfwWindowShouldClose(m_main_window.window()));
	}
	void set_should_close(bool b) noexcept {
		glfwSetWindowShouldClose(m_main_window.window(), b);
	}
	void prepare_frame() noexcept {
		glfwPollEvents();
	}
	void render_frame() noexcept {
		// imgui render
	}

private:
	void _draw() noexcept {}
	void _render() noexcept {}
};

// -----------------------
// class ImGuiWindow
// The API Definition
// -----------------------

ImGuiWindow::ImGuiWindow() noexcept {}
ImGuiWindow::~ImGuiWindow() noexcept = default;

bool ImGuiWindow::should_close() const noexcept {
	LUISA_ASSERT(mp_impl, "ImGuiWindow is not created");
	return mp_impl->should_close();
}

void ImGuiWindow::set_should_close(bool b) noexcept {
	LUISA_ASSERT(mp_impl, "ImGuiWindow is not created");
	mp_impl->set_should_close(b);
}

void ImGuiWindow::prepare_frame() noexcept {
	LUISA_ASSERT(mp_impl, "ImGuiWindow is not created");
	mp_impl->prepare_frame();
}

void ImGuiWindow::render_frame() noexcept {
	LUISA_ASSERT(mp_impl, "ImGuiWindow is not created");
	mp_impl->render_frame();
}

}// namespace sail::inno