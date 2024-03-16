#pragma once

/**
 * @file source/package/solver/fluid/sph/builder.h
 * @author sailing-innocent
 * @date 2023-02-22
 * @brief (temp) SPH Fluid Builder
 */

#include "SailBase/config.h"
#include <luisa/core/basic_types.h>
#include <luisa/core/stl/vector.h>

namespace sail::inno::sph {

class SPHSolver;

class SAIL_INNO_API SPHFluidData {
public:
	luisa::vector<luisa::float3> h_pos;
	// void to_csv(const std::string& path) noexcept;
};

class SAIL_INNO_API SPHFluidBuilder {
public:
	SPHFluidBuilder(SPHSolver& solver) noexcept
		: m_solver(solver){};
	SPHFluidData grid(const luisa::float3& bottom_left,
					  const luisa::float3& grid_size,
					  const float dx2) noexcept;

	void push_particles(const SPHFluidData& data) noexcept;
	void place_particles(const SPHFluidData& data) noexcept;

private:
	SPHSolver& m_solver;
};

};// namespace sail::inno::sph