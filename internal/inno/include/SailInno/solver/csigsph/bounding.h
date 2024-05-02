#pragma once
/**
 * @file bounding.h
 * @brief The Bounding
 * @author sailing-innocent
 * @date 2024-05-02
 */

#include <luisa/luisa-compute.h>
#include "SailInno/solver/csigsph/sph_executor.h"
#include "SailInno/solver/csigsph/fluid_particles.h"
#include "SailInno/helper/buffer_filler.h"
#include "SailInno/helper/device_parallel.h"

namespace sail::inno::csigsph {
class SPHSolver;
// Boundary restrictions
class SAIL_INNO_API Bounding : public SPHExecutor {
	template<typename T>
	using U = luisa::unique_ptr<T>;

	template<typename T>
	using Buffer = luisa::compute::Buffer<T>;

	friend class SPHSolver;

public:
	Bounding(SPHSolver& solver) noexcept;
	void solve(luisa::compute::CommandList& cmdlist) noexcept;

private:
	void create() noexcept;
	void compile(Device& device) noexcept;
	void reset() noexcept;
	// size_t m_size = 0;
	// size_t m_capacity = 0;
	U<Shader<1, int, float, float>> bounding_cube;
	U<Shader<1, int, float, float>> bounding_sphere;
	U<Shader<1, int, float, float>> bounding_waterfall;
	U<Shader<1, int, float, float>> bounding_heightmap;
};
}// namespace sail::inno::csigsph