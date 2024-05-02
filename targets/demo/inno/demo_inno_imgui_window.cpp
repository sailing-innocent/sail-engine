/**
 * @file demo_inno_imgui_window.cpp
 * @brief The Demo of using LuisaCompute's embedded ImGuiWindow
 * @author sailing-innocent
 * @date 2024-05-01
 */

#include "SailInno/window/imgui_window.h"

using namespace sail::inno;
using namespace luisa;
using namespace luisa::compute;

int main(int args, char* argv[]) {
	Context context{argv[0]};
	DeviceConfig config{
		.inqueue_buffer_limit = false};
	Device device = context.create_device(
		"dx", &config);
	Stream stream = device.create_stream(StreamTag::GRAPHICS);

	ImGuiWindow::Config window_cfg{

	};
	ImGuiWindow window{
		device,
		stream,
		// io
		"ImGui Window",
		"shader_path",
		window_cfg};

	// render pipeline
	// window set callback

	while (!window.should_close()) {
		window.prepare_frame();
		// render
		window.render_frame();
	}

	return 0;
}