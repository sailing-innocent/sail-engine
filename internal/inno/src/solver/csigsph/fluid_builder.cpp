/**
 * @file fluid_builder.cpp
 * @brief The Fluid Builder
 * @author sailing-innocent
 * @date 2024-05-02
 */
#include "SailInno/solver/csigsph/fluid_builder.h"
#include <fstream>
namespace sail::inno::csigsph {

void Fluid::to_csv(const std::string& path) const noexcept {
	std::ofstream of;
	of.open(path);
	for (const auto& pos : h_pos) {
		of << pos[0] << "," << pos[1] << "," << pos[2] << std::endl;
	}
	of.close();
}

Fluid FluidBuilder::grid(
	const luisa::float3& bottom_left_pos,
	const luisa::float3& grid_size,
	const float& dx2) noexcept {
	// create particles
	Fluid particle;
	particle.h_pos.clear();
	int n_length = grid_size[0] / dx2;
	int n_width = grid_size[1] / dx2;
	int n_height = grid_size[2] / dx2;
	for (int dx = 0; dx < n_length; ++dx) {
		for (int dy = 0; dy < n_width; ++dy) {
			for (int dz = 0; dz < n_height; ++dz) {
				particle.h_pos.push_back(bottom_left_pos + luisa::make_float3(dx * dx2, dy * dx2, dz * dx2));
			}
		}
	}

	return particle;
}

void FluidBuilder::push_particle(Fluid& fluid) {
	auto& particles = m_solver.particles();
	particles.push_particles(fluid.h_pos);
}

void FluidBuilder::place_particle(Fluid& fluid) {
	auto& particles = m_solver.particles();
	particles.place_particles(fluid.h_pos);
}

void FluidBuilder::download(luisa::compute::CommandList& cmdlist, Fluid& fluid) noexcept {
	auto& particles = m_solver.particles();
	auto view = particles.m_pos.view(0, fluid.h_pos.size());
	cmdlist << view.copy_to(fluid.h_pos.data());
}
}// namespace sail::inno::csigsph