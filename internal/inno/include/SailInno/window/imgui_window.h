#pragma once
/**
 * @file imgui_window.h
 * @brief The ImGUI Window 
 * @author sailing-innocent
 * @date 2024-05-01
 */

#include "SailInno/core/runtime.h"
#include "luisa/vstl/config.h"
#include <luisa/runtime/device.h>
#include <luisa/gui/window.h>

struct GLFWwindow;
struct ImGuiContext;

namespace sail::inno {
using namespace luisa;
using namespace luisa::compute;

class SAIL_INNO_API ImGuiWindow {
public:
	struct Config {
		uint2 size = {1280, 720};
		bool resizable = true;
		bool fullscreen = false;

		[[nodiscard]] static Config default_config() noexcept {
			return {};
		}
	};

private:
	class ContextGaurd {
		ImGuiWindow* mp_self;

	public:
		// delete copy and move
		ContextGaurd(const ContextGaurd&) = delete;
		ContextGaurd& operator=(const ContextGaurd&) = delete;
		ContextGaurd(ContextGaurd&&) = delete;
		ContextGaurd& operator=(ContextGaurd&&) = delete;

		explicit ContextGaurd(ImGuiWindow* self) noexcept : mp_self{self} {
			mp_self->push_context();
		}
		~ContextGaurd() noexcept {
			mp_self->pop_context();
		}
	};

public:
	class Impl;

private:
	unique_ptr<Impl> mp_impl;

public:
	ImGuiWindow(Device& device, Stream& stream) noexcept;
	~ImGuiWindow() noexcept;
	// delete copy
	ImGuiWindow(const ImGuiWindow&) = delete;
	ImGuiWindow& operator=(const ImGuiWindow&) = delete;
	// keep move
	ImGuiWindow(ImGuiWindow&&) noexcept;
	ImGuiWindow& operator=(ImGuiWindow&&) noexcept;

	// lifecycle
	void create(Device& device, Stream& stream) noexcept;
	void destroy() noexcept;

	// context
	[[nodiscard]] ImGuiContext* context() const noexcept;
	void push_context() noexcept;
	void pop_context() noexcept;

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