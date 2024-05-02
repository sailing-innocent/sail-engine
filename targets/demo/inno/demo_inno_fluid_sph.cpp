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
#include "SailInno/util/camera.h"

#include "SailInno/solver/csigsph/fluid_builder.h"
#include "SailInno/solver/csigsph/fluid_particles.h"
#include "SailInno/solver/csigsph/solver.h"

using namespace luisa;
using namespace luisa::compute;

namespace sail::inno::test {

int test_fluid_sph(Device& device) {
	Stream stream = device.create_stream(StreamTag::GRAPHICS);
	// parameters
	constexpr float dt = 0.002f;
	constexpr float dx = 0.005f;
	constexpr int model_kind = 1;
	constexpr float alpha = 0.004;
	constexpr float stiffB = 100.0f;
	constexpr int support_size = 4;

	inno::csigsph::SPHSolver solver{};
	auto config = inno::csigsph::SPHSolverConfig{
		.world_size = 1.0f,
		.sph_model_kind = model_kind,
		.least_iter = 3};

	auto param = inno::csigsph::SPHParam{
		.delta_time = dt,
		.bound_kind = inno::csigsph::SPHBoundKind::WATERFALL,
		.dx = dx,
		.h_fac = dx * support_size,
		.alpha = alpha,
		.stiffB = stiffB};
	solver.config(config);
	solver.param(param);
	// generate initial fluid data
	inno::csigsph::FluidBuilder builder{solver};
	auto fluid_data = builder.grid(
		make_float3(0.25f, 0.25f, 0.5f),
		make_float3(0.5f),
		dx * 2);
	builder.push_particle(fluid_data);

	CommandList cmdlist;
	solver.create(device);
	solver.compile(device);
	LUISA_INFO("Compile done");
	solver.init_upload(device, cmdlist);
	LUISA_INFO("Upload done");
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

	auto mp_vis = luisa::make_unique<Visualizer>();
	mp_vis->name = "SPH Visualizer";
	mp_vis->resolution = make_uint2(w, h);
	mp_vis->create(device, stream);
	auto mp_painter = luisa::make_unique<PointPainter>();
	mp_painter->m_stride = 4;// float3 stride 4
	mp_painter->create(device);
	mp_painter->update_clear_color(make_float3(0.0f, 0.0f, 0.0f));
	auto point_num = solver.particles().size();
	mp_painter->update_point(
		point_num,
		solver.particles().m_pos.view(0, point_num).as<float>(),
		solver.particles().m_pos.view(0, point_num).as<float>());
	mp_painter->update_camera(cam);
	mp_vis->add_painter(std::move(mp_painter));

	LUISA_INFO("Start running");
	while (mp_vis->running()) {
		// physics step
		// includes accel rebuild and allocate
		solver.step(device, cmdlist);
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
