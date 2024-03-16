#pragma once

/**
 * @file source/package/solver/fluid/sph/bounding.h
 * @author sailing-innocent
 * @date 2023-02-24
 * @brief (temp) SPH Fluid Bounding
 */

#include <luisa/core/basic_types.h>
#include "sph_executor.h"
#include "fluid_particles.h"

namespace sail::inno::sph {
class SPHSolver;

class SAIL_INNO_API Bounding : public SPHExecutor {
	friend class SPHSolver;
	template<size_t I, typename... Ts>
	using Shader = luisa::compute::Shader<I, Ts...>;
	using Int = luisa::compute::Int;
	using Float = luisa::compute::Float;

public:
	Bounding(SPHSolver& solver) noexcept;
	void solve(CommandList& cmdlist) noexcept;

private:
	// lifecycle
	// only friend class can call these methods
	void create(Device& device) noexcept;
	void compile(Device& device) noexcept;
	void reset() noexcept;

private:
	// shaders
	U<Shader<1, int, float, float>> ms_bounding_cube;
};

}// namespace sail::inno::sph
