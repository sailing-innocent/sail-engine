#pragma once
#include "SailInno/core/runtime.h"

#include <luisa/runtime/device.h>

namespace sail::inno {
class IOService;
struct GLFWwindow;
struct ImGuiContext;
}// namespace sail::inno

namespace luisa::compute {
class SwapChain;
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
		uint back_buffers_count{2u};
		[[nodiscard]] static Config default_config() noexcept { return {}; }
	};

private:
	class ContextGuard {
	private:
		ImGuiWindow* _self;

	public:
	};

public:
	class Impl;

private:
	luisa::unique_ptr<Impl> _pimpl;

public:
	ImGuiWindow() noexcept = default;

	void create() noexcept;
	void destroy() noexcept;
};

}// namespace sail::inno