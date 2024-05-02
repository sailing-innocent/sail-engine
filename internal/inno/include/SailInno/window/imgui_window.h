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

// IOService

struct GLFWwindow;
struct ImGuiContext;

namespace luisa::compute {
class Swapchain;

template<typename T>
class Image;

class Sampler;

}// namespace luisa::compute

namespace sail::inno {
using namespace luisa;
using namespace luisa::compute;

class SAIL_INNO_API ImGuiWindow {
public:
	struct Config {
		uint2 size{1280u, 720u};
		bool resizable{true};
		bool fullscreen{false};
		bool hdr{false};
		bool vsync{false};
		uint back_buffer_count{2u};

		[[nodiscard]] static Config default_config() noexcept { return {}; }
	};

private:
	class ContextGuard {
		ImGuiWindow* mp_self;

	public:
		// delete copy and move
		ContextGuard(const ContextGuard&) = delete;
		ContextGuard& operator=(const ContextGuard&) = delete;
		ContextGuard(ContextGuard&&) = delete;
		ContextGuard& operator=(ContextGuard&&) = delete;

		explicit ContextGuard(ImGuiWindow* self) noexcept : mp_self{self} {
			mp_self->push_context();
		}
		~ContextGuard() noexcept {
			mp_self->pop_context();
		}
	};

public:
	class Impl;

private:
	unique_ptr<Impl> mp_impl;

public:
	ImGuiWindow(
		Device& device,
		Stream& stream,
		// IOService
		luisa::string name,
		luisa::filesystem::path const& shader_dir,
		const Config& config = Config::default_config()) noexcept;
	~ImGuiWindow() noexcept;
	// delete copy
	ImGuiWindow(const ImGuiWindow&) = delete;
	ImGuiWindow& operator=(const ImGuiWindow&) = delete;
	// keep move
	ImGuiWindow(ImGuiWindow&&) noexcept;
	ImGuiWindow& operator=(ImGuiWindow&&) noexcept;

	// lifecycle
	void create(
		Device& device,
		Stream& stream,
		// IOService
		luisa::string name,
		luisa::filesystem::path const& shader_dir,
		const Config& config = Config::default_config()) noexcept;
	void destroy() noexcept;

	// context
	[[nodiscard]] ImGuiContext* context() const noexcept;
	void push_context() noexcept;
	void pop_context() noexcept;

	// resource
	[[nodiscard]] GLFWwindow* handle() const noexcept;
	[[nodiscard]] Window& window() noexcept;
	[[nodiscard]] Window const& window() const noexcept;
	[[nodiscard]] Swapchain& swapchain() const noexcept;
	[[nodiscard]] Image<float>& framebuffer() const noexcept;

	[[nodiscard]] bool valid() const noexcept { return mp_impl != nullptr; }
	[[nodiscard]] operator bool() const noexcept { return valid(); }
	[[nodiscard]] bool should_close() const noexcept;
	void set_should_close(bool b = true) noexcept;

	void prepare_frame() noexcept;// call event handle, imgui new frame etc. and makes the context current
	void render_frame() noexcept; // calls imgui render, swapbuffer
								  // register textures
	template<typename F>
	decltype(auto) with_context(F&& f) noexcept {
		ContextGuard g{this};
		return luisa::invoke(std::forward<F>(f));
	}

	template<typename F>
	void with_frame(F&& f) noexcept {
		prepare_frame();
		luisa::invoke(std::forward<F>(f));
		render_frame();
	}
};

}// namespace sail::inno