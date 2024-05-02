/**
 * @file imgui_window.cpp
 * @brief The Implementation of ImGUI Window
 * @author sailing-innocent
 * @date 2024-05-01
 */

#include "imgui_internal.h"
#include <mutex>
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

struct GUIObjectData {
	uint tex_id;
	std::array<float, 2> pos;
	std::array<float, 2> size;
};

struct GUIVertex {
	std::array<float, 2> pos;
	std::array<float, 2> uv;
	uint32_t col;
};

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
	Device& m_device;
	Stream& m_stream;
	// IOService
	Config m_config;
	ImGuiContext* mp_context;
	// manage window and swapchain
	Window m_main_window;
	Swapchain m_swapchain;
	Image<float> m_main_framebuffer;
	// Image<float> m_font_texture;
	// BindlessArray m_texture_array;
	vstd::HashMap<GLFWwindow*, Swapchain> m_platform_swapchains;
	vstd::HashMap<GLFWwindow*, Image<float>> m_platform_framebuffers;
	RasterState m_raster_state;
	MeshFormat m_mesh_format;
	// rendering
	Shader2D<Image<float>, float3> m_clear_shader;
	RasterShader<float4x4, BindlessArray, Buffer<detail::GUIObjectData>> m_draw_shader;
	uint64_t m_frames_idx{};
	luisa::vector<detail::GUIObjectData> m_host_obj_data;
	luisa::vector<RasterMesh> m_draw_meshes;
	// io timeline

	class CtxGuard {
	private:
		ImGuiContext* mp_curr_ctx;
		ImGuiContext* mp_prev_ctx;

	public:
		explicit CtxGuard(ImGuiContext* ctx) noexcept
			: mp_curr_ctx(ctx),
			  mp_prev_ctx(ImGui::GetCurrentContext()) {
			ImGui::SetCurrentContext(mp_curr_ctx);
		}

		~CtxGuard() noexcept {
			auto* curr_ctx = ImGui::GetCurrentContext();
			LUISA_ASSERT(curr_ctx == mp_curr_ctx, "ImGui context is not restored correctly");
			ImGui::SetCurrentContext(mp_prev_ctx);
		}

		CtxGuard(const CtxGuard&) = delete;
		CtxGuard& operator=(const CtxGuard&) = delete;
		CtxGuard(CtxGuard&&) = delete;
		CtxGuard& operator=(CtxGuard&&) = delete;
	};

private:
	template<typename F>
	decltype(auto) _with_context(F&& f) noexcept {
		CtxGuard guard{mp_context};
		return luisa::invoke(std::forward<F>(f));
	}

	// callbacks
	void _rebuild_swapchain_if_changed(GLFWwindow* w, Swapchain& sc, Image<float>& fb) {
		auto fw = 0, fh = 0;
		glfwGetFramebufferSize(w, &fw, &fh);
		auto size = make_uint2(fw, fh);
		if (any(size != 0u)) {
			auto native_handle = detail::glfw_window_native_handle(w);
			sc = m_device.create_swapchain(
				m_stream, SwapchainOption{
							  .display = native_handle,
							  .window = native_handle,
							  .size = size,
							  .wants_hdr = m_config.hdr,
							  .wants_vsync = m_config.vsync,
							  .back_buffer_count = m_config.back_buffer_count,
						  });
		}
	}
	void _on_imgui_create_window(ImGuiViewport* vp) noexcept {
		auto* glfw_window = static_cast<GLFWwindow*>(vp->PlatformHandle);
		LUISA_ASSERT(glfw_window != nullptr && glfw_window != m_main_window.window(),
					 "Invalid GLFW window.");

		Swapchain sc;
		Image<float> fb;
		_rebuild_swapchain_if_changed(glfw_window, sc, fb);
		m_platform_swapchains.emplace(glfw_window, std::move(sc));
		m_platform_framebuffers.emplace(glfw_window, std::move(fb));
	}
	void _on_imgui_destroy_window(ImGuiViewport* vp) noexcept {
		// sync
		if (
			auto* glfw_window = static_cast<GLFWwindow*>(vp->PlatformHandle);
			glfw_window != m_main_window.window()) {
			// _platform_swapchains.remove(glfw_window);
			// _platform_framebuffers.remove(glfw_window);
		}
	}

public:
	Impl(
		Device& device,
		Stream& stream,
		luisa::string name,
		luisa::filesystem::path const& shader_path,
		const Config& config) noexcept
		: m_device{device},
		  m_stream{stream},
		  m_config{config},
		  mp_context{[] {
		IMGUI_CHECKVERSION();
		return ImGui::CreateContext();
	}()},
		  m_main_window{name, config.size.x, config.size.y, config.resizable, config.fullscreen} {
		m_raster_state = RasterState{
			.cull_mode = CullMode::None,
			.blend_state = BlendState{
				.enable_blend = true,
				.prim_op = BlendWeight::PrimAlpha,
				.img_op = BlendWeight::OneMinusPrimAlpha}};
		VertexAttribute vert_attribs[] = {
			{VertexAttributeType::Position, PixelFormat::RG32F},
			{VertexAttributeType::UV0, PixelFormat::RG32F},
			{VertexAttributeType::Color, PixelFormat::RGBA8UNorm},
		};
		m_mesh_format.emplace_vertex_stream(vert_attribs);
		// m_draw_shader = m_device.load_raster_shader<float4x4, BindlessArray, Buffer<detail::GUIObjectData>>(luisa::to_string(shader_path / "test_imgui.bin"));
		// LUISA_ASSERT(m_draw_shader, "Invalid raster shader.");

		// initialize GLFW
		static std::once_flag once_flag;
		std::call_once(once_flag, [] {
			// set error callback
			glfwSetErrorCallback([](int error, const char* description) noexcept {
				if (error != GLFW_NO_ERROR) [[likely]] {
					LUISA_ERROR("GLFW Error (code = 0x{:08x}): {}.", error, description);
				}
			});
			if (!glfwInit()) [[unlikely]] {
				LUISA_ERROR_WITH_LOCATION("Failed to initialize GLFW.");
			}
		});
		// create main swapchain
		_rebuild_swapchain_if_changed(m_main_window.window(), m_swapchain, m_main_framebuffer);
		// TODO: install user callbacks

		// imgui config
		_with_context([this] {
			auto io = ImGui::GetIO();
			io.ConfigFlags |= ImGuiConfigFlags_ViewportsEnable;// Enable Multi-Viewport / Platform Windows
			// styles
			// register GLFW window
			ImGui_ImplGlfw_InitForOther(m_main_window.window(), true);

			// register renderer
			io.BackendRendererUserData = this;
			io.BackendRendererName = "imgui_impl_luisa";
			io.BackendFlags |= ImGuiBackendFlags_RendererHasVtxOffset;
			io.BackendFlags |= ImGuiBackendFlags_RendererHasViewports;

			if (io.ConfigFlags & ImGuiConfigFlags_ViewportsEnable) [[likely]] {
				auto& platform_io = ImGui::GetPlatformIO();
				static constexpr auto imgui_get_this = [] {
					return ImGui::GetCurrentContext() ?
							   static_cast<Impl*>(ImGui::GetIO().BackendRendererUserData) :
							   nullptr;
				};
				platform_io.Renderer_CreateWindow = [](ImGuiViewport* vp) {
					if (auto* self = imgui_get_this()) {
						self->_on_imgui_create_window(vp);
					}
				};
				platform_io.Renderer_DestroyWindow = [](ImGuiViewport* vp) {
					if (auto* self = imgui_get_this()) {
						self->_on_imgui_destroy_window(vp);
					}
				};
			}
		});

		// create texture array
		// create shaders
	}
	~Impl() noexcept {}

	// resource
	[[nodiscard]] auto context() const noexcept { return mp_context; }
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

ImGuiWindow::ImGuiWindow(
	Device& device,
	Stream& stream,
	luisa::string name,
	luisa::filesystem::path const& shader_dir,
	const Config& config) noexcept {
	this->create(device, stream, name, shader_dir, config);
}
ImGuiWindow::~ImGuiWindow() noexcept = default;

void ImGuiWindow::create(
	Device& device,
	Stream& stream,
	luisa::string name,
	luisa::filesystem::path const& shader_dir,
	const Config& config) noexcept {
	this->destroy();
	mp_impl = luisa::make_unique<Impl>(
		device, stream, name, shader_dir, config);
}

void ImGuiWindow::destroy() noexcept {
	mp_impl = nullptr;
}

ImGuiContext* ImGuiWindow::context() const noexcept {
	LUISA_ASSERT(mp_impl, "ImGuiWindow not created.");
	return mp_impl->context();
}

void ImGuiWindow::push_context() noexcept {
	LUISA_ASSERT(mp_impl, "ImGuiWindow not created.");
	// auto& stack = detail::imgui_context_stack();
	// auto curr_ctx = ImGui::GetCurrentContext();
	// stack.emplace_back(curr_ctx);
	// auto ctx = mp_impl->context();
	// ImGui::SetCurrentContext(ctx);
	// detail::imgui_context_stack().emplace_back(ctx);
}

void ImGuiWindow::pop_context() noexcept {
	LUISA_ASSERT(mp_impl, "ImGuiWindow not created.");
	// if (auto& stack = detail::imgui_context_stack();
	// 	!stack.empty() && stack.back() == _impl->context()) {
	// 	stack.pop_back();
	// 	auto ctx = stack.empty() ? nullptr : stack.back();
	// 	ImGui::SetCurrentContext(ctx);
	// } else {
	// 	LUISA_WARNING_WITH_LOCATION("Invalid ImGui context stack.");
	// }
}

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