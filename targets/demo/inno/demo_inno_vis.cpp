/**
 * @file demo_vis.cpp
 * @author sailing-innocent
 * @date 2023-12-27
 * @brief Inno Pure Painter Visualizer Demo
 */

#include <luisa/runtime/context.h>
#include <luisa/runtime/device.h>
#include <luisa/runtime/stream.h>

#include "SailInno/vis/vis.h"
#include "SailInno/vis/painter_base.h"

using namespace luisa;
using namespace luisa::compute;

namespace sail::inno::test {

int test_pure_painter(Device& device) {
	auto stream = device.create_stream(StreamTag::GRAPHICS);
	auto mp_vis = luisa::make_unique<Visualizer>();
	mp_vis->create(device, stream);
	auto mp_painter = luisa::make_unique<PainterBase>();
	mp_painter->create(device);
	mp_painter->update_clear_color(make_float3(1.0f, 0.0f, 1.0f));
	mp_vis->add_painter(std::move(mp_painter));
	auto i = 0;
	while (mp_vis->running()) {
		CommandList cmdlist;
		mp_vis->vis(cmdlist);
		stream << cmdlist.commit() << synchronize();
		mp_vis->present_and_poll_events(stream);
		if (i > 300) {
			break;
		}
		i = i + 1;
	}

	return 0;
}

}// namespace sail::inno::test

int main(int argc, char* argv[]) {
	Context context{argv[0]};
	auto device = context.create_device("dx");
	return sail::inno::test::test_pure_painter(device);
	// return 0;
}