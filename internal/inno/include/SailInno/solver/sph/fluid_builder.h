#pragma once
/**
 * @file fluid_builder.h
 * @brief The Fluid Builder Header
 * @author sailing-innocent
 * @date 2024-05-02
 */
#include <luisa/luisa-compute.h>
#include "SailInno/solver/sph/solver.h"

namespace sail::inno::sph {

class SAIL_INNO_API Fluid {
public:
	friend class FluidBuilder;
	// particle pos
	luisa::vector<luisa::float3> h_pos;
	void to_csv(const std::string& path) const noexcept;
};

class SAIL_INNO_API FluidBuilder {
public:
	FluidBuilder(SPHSolver& solver) noexcept : m_solver{solver} {}
	Fluid grid(const luisa::float3& bottom_left_pos, const luisa::float3& grid_size, const float& dx2) noexcept;
	void push_particle(Fluid& fluid);
	void place_particle(Fluid& fluid);
	void download(luisa::compute::CommandList& cmdlist, Fluid& fluid) noexcept;

private:
	SPHSolver& m_solver;
};
}// namespace sail::inno::sph
