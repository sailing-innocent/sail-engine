/**
 * @file demo_inno_fluid_sph
 * @author sailing-innocent
 * @date 2024-03-16
 * @brief Fluid SPH
 */

// TODO: Algorithm Migrate
// WCSPH
// PCISPH
#include <luisa/runtime/context.h>
#include <luisa/runtime/device.h>
#include <luisa/runtime/stream.h>

#include "SailInno/vis/vis.h"
#include "SailInno/vis/point_painter.h"
#include "SailInno/util/scene/points.h"
#include "SailInno/util/camera.h"
#include "SailInno/solver/sph/builder.h"
#include "SailInno/solver/sph/fluid_particles.h"
#include "SailInno/solver/sph/solver.h"

using namespace luisa;
using namespace luisa::compute;

namespace sail::inno::test {

int test_fluid_sph(Device& device) {
	Stream stream = device.create_stream(StreamTag::GRAPHICS);

	// parameters
	constexpr float dt = 0.001f;
	constexpr float dx = 0.005f;

	inno::sph::SPHSolver solver{};
	auto config =
		inno::sph::SPHSolverConfig{
			.model_kind = inno::sph::SPHModelKind::WCSPH};
	solver.config(config);
	solver.create(device);

	inno::sph::SPHFluidBuilder builder{solver};
	auto fluid_data = builder.grid(
		make_float3(0.25f, 0.25f, 0.5f),
		make_float3(0.5f),
		dx * 2);

	builder.place_particles(fluid_data);

	CommandList cmdlist;
	solver.compile(device);
	solver.init_upload(device, cmdlist);

	stream << cmdlist.commit() << synchronize();

	constexpr uint w = 1024u;
	constexpr uint h = 768u;
	// camera
	inno::Camera cam{
		make_float3(2.0f, 2.0f, 2.0f),
		make_float3(0.5f, 0.5f, 0.5f),
		make_float3(0.0f, 0.0f, 1.0f),
	};
	cam.set_aspect_ratio(static_cast<float>(w) / static_cast<float>(h));

	// visualizer
	auto mp_vis = luisa::make_unique<Visualizer>();
	mp_vis->name = "SPH Visualizer";
	mp_vis->resolution = make_uint2(w, h);
	mp_vis->create(device, stream);

	auto mp_painter = luisa::make_unique<PointPainter>();
	mp_painter->m_stride = 4;// float3 stride 4
	mp_painter->create(device);
	mp_painter->update_clear_color(make_float3(0.0f, 0.0f, 0.0f));
	mp_painter->update_point(
		solver.particles().size(),
		solver.particles().m_d_pos.view().as<float>(),
		solver.particles().m_d_pos.view().as<float>());

	// mp_painter->update_point(scene.num_points(), scene.xyz(), scene.color());
	mp_painter->update_camera(cam);
	mp_vis->add_painter(std::move(mp_painter));

	while (mp_vis->running()) {
		// physics step
		solver.step(cmdlist);
		stream << cmdlist.commit() << synchronize();
		// render step
		mp_vis->vis(cmdlist);
		stream << cmdlist.commit() << synchronize();
		// present
		mp_vis->present_and_poll_events(stream);
	}
	// final sync
	stream << synchronize();

	return 0;
}
}// namespace sail::inno::test

int main(int argc, char* argv[]) {
	Context context{argv[0]};
	auto device = context.create_device("dx");
	return sail::inno::test::test_fluid_sph(device);
}
