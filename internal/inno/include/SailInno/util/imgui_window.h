#pragma once
/**
 * @file imgui_window.h
 * @brief The IMGUI Window
 * @author sailing-innocent
 * @date 2024-05-19
 */

#include "SailInno/config.h"
#include <luisa/luisa-compute.h>

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
		uint2 size = {1280u, 720u};
		bool resizable = true;
		bool full_screen = false;
		[[nodiscard]] static Config make_default() noexcept { return {}; }
	};

private:
	class ContextGuard {};

public:
	class Impl;
	ImGuiWindow() = default;
	explicit ImGuiWindow(Device& device, Stream& stream, luisa::string name, luisa::string_view shader_path, const Config& config = Config::make_default()) noexcept;
	~ImGuiWindow() noexcept;
	// delete copy
	ImGuiWindow(const ImGuiWindow&) = delete;
	ImGuiWindow& operator=(const ImGuiWindow&) = delete;
	// default move
	ImGuiWindow(ImGuiWindow&&) noexcept = default;
	ImGuiWindow& operator=(ImGuiWindow&&) noexcept = default;

private:
	luisa::unique_ptr<Impl> mp_impl;

public:
	// life cycle
	void create(Device& device, Stream& stream,
				luisa::string name,
				luisa::string_view shader_path,
				const Config& config = Config::make_default()) noexcept;
	void destroy() noexcept;
	// getter
	[[nodiscard]] GLFWwindow* handle() const noexcept;
};

}// namespace sail::inno