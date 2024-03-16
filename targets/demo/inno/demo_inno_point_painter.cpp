/**
 * @file demo_inno_point_painter
 * @author sailing-innocent
 * @date 2023-12-27
 * @brief Point Painter Test Suite
 */

#include <luisa/runtime/context.h>
#include <luisa/runtime/device.h>
#include <luisa/runtime/stream.h>

#include "SailInno/vis/vis.h"
#include "SailInno/vis/point_painter.h"
#include "SailInno/util/scene/points.h"
#include "SailInno/util/camera.h"

using namespace luisa;
using namespace luisa::compute;

namespace sail::inno::test {

int test_point_painter(Device& device) {
	auto stream = device.create_stream(StreamTag::GRAPHICS);

	int N = 1000;
	PointsScene scene{N};
	scene.create(device);
	CommandList cmdlist;
	scene.init(cmdlist);
	uint w = 2048u;
	uint h = 768u;
	stream << cmdlist.commit() << synchronize();
	Camera cam{
		make_float3(5.0f, 5.0f, 5.0f),// eye
		make_float3(0.5f, 0.5f, 0.5f),// target
		make_float3(0.0f, 0.0f, 1.0f),// up
	};
	cam.set_aspect_ratio(static_cast<float>(w) / static_cast<float>(h));
	// Camera cam;

	auto mp_vis = luisa::make_unique<Visualizer>();
	mp_vis->name = "Point Painter";
	mp_vis->resolution = make_uint2(w, h);
	mp_vis->create(device, stream);
	auto mp_painter = luisa::make_unique<PointPainter>();
	mp_painter->create(device);
	mp_painter->update_clear_color(make_float3(0.0f, 0.0f, 0.0f));
	mp_painter->update_point(scene.num_points(), scene.xyz(), scene.color());
	mp_painter->update_camera(cam);

	mp_vis->add_painter(std::move(mp_painter));

	while (mp_vis->running()) {
		mp_vis->vis(cmdlist);
		stream << cmdlist.commit() << synchronize();
		mp_vis->present_and_poll_events(stream);
	}

	return 0;
}

}// namespace sail::inno::test

int main(int argc, char* argv[]) {
	Context context{argv[0]};
	auto device = context.create_device("dx");
	return sail::inno::test::test_point_painter(device);
	// return 0;
}
