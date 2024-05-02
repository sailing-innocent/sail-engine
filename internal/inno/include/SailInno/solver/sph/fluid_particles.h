#pragma once

/**
 * @file source/package/solver/fluid/sph/fluid_particles.h
 * @author sailing-innocent
 * @date 2023-02-22
 * @brief SPH Fluid Particles
 */

#include <luisa/core/basic_types.h>
#include "sph_executor.h"

namespace sail::inno::sph {
class SPHSolver;
// class Neighbor;
// class BaseSPH;
// class WCSPH;
// class PCISPH;

class SAIL_INNO_API SPHFluidParticles : public SPHExecutor {
	friend class SPHSolver;

public:
	SPHFluidParticles(SPHSolver& solver) noexcept;

	Buffer<int> m_d_id;
	Buffer<luisa::float3> m_d_pos;
	Buffer<luisa::float3> m_d_vel;

	luisa::vector<luisa::float3> m_h_pos;
	luisa::vector<int> m_h_id;

	size_t size() const noexcept { return m_size; }
	size_t max_size() const noexcept { return m_max_size; }

	// place and push particles
	int place_particles(const luisa::vector<luisa::float3>& host_pos) noexcept;
	int push_particles(const luisa::vector<luisa::float3>& host_pos) noexcept;

private:
	void create(Device& device) noexcept;
	void init_upload(Device& device, CommandList& cmdlist) noexcept;
	void reset(Device& device, CommandList& cmdlist) noexcept;
	void allocate(Device& device, size_t size) noexcept;

	size_t m_size;
	size_t m_max_size;
};

}// namespace sail::inno::sph