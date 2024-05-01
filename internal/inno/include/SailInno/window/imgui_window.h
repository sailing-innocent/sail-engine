#pragma once
/**
 * @file imgui_window.h
 * @brief The ImGUI Window 
 * @author sailing-innocent
 * @date 2024-05-01
 */

#include "SailInno/core/runtime.h"
#include <luisa/runtime/device.h>
#include <luisa/gui/window.h>

namespace sail::inno {
using namespace luisa;
using namespace luisa::compute;

class SAIL_INNO_API ImGuiWindow {
public:
	struct Config {};

private:
	class ContextGaurd {};

public:
	class Impl;

private:
	unique_ptr<Impl> mp_impl;

public:
	ImGuiWindow() noexcept;
	~ImGuiWindow() noexcept;
	// delete copy
	ImGuiWindow(const ImGuiWindow&) = delete;
	ImGuiWindow& operator=(const ImGuiWindow&) = delete;
	// keep move
	ImGuiWindow(ImGuiWindow&&) noexcept;
	ImGuiWindow& operator=(ImGuiWindow&&) noexcept;

	// lifecycle
	void create() noexcept {}
	void destroy() noexcept {}

	// context

	// resource
	[[nodiscard]] GLFWwindow* handle() const noexcept;
	[[nodiscard]] Window& window() noexcept;
	[[nodiscard]] Window const& window() const noexcept;

	[[nodiscard]] bool valid() const noexcept { return mp_impl != nullptr; }
	[[nodiscard]] operator bool() const noexcept { return valid(); }
	[[nodiscard]] bool should_close() const noexcept;
	void set_should_close(bool b = true) noexcept;

	void prepare_frame() noexcept;// call event handle, imgui new frame etc. and makes the context current
	void render_frame() noexcept; // calls imgui render, swapbuffer
};

}// namespace sail::inno