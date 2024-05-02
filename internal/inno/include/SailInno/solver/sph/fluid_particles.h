#pragma once
/**
 * @author Oncle-Ha
 * @date 2023-04-06
 */

#include <luisa/luisa-compute.h>
#include "SailInno/solver/sph/sph_executor.h"

#include "SailInno/solver/sph/model/wcsph.h"
#include "SailInno/solver/sph/model/pcisph.h"

namespace sail::inno::sph {
class Neighbor;
class SPHSolver;
class BaseSPH;
class WCSPH;
class PCISPH;

class FluidParticles : public SPHExecutor {
	template<typename T>
	using U = luisa::unique_ptr<T>;

	template<typename T>
	using Buffer = luisa::compute::Buffer<T>;

	friend class SPHSolver;
	friend class Neighbor;
	friend class BaseSPH;
	friend class WCSPH;
	friend class PCISPH;

public:
	FluidParticles(SPHSolver& solver) noexcept;

	Buffer<int> m_id;
	Buffer<luisa::float3> m_pos;
	Buffer<luisa::float3> m_vel;

	luisa::vector<luisa::float3> m_h_pos;
	luisa::vector<int> m_h_id;

	auto size() const noexcept { return m_size; }
	auto max_size() const noexcept { return m_capacity; }

	int place_particles(const luisa::vector<luisa::float3>& host_pos) noexcept;
	int push_particles(const luisa::vector<luisa::float3>& host_pos) noexcept;

private:
	void create(Device& device) noexcept;
	void init_upload(Device& device, CommandList& cmdlist) noexcept;
	void reset(Device& device, CommandList& cmdlist) noexcept;
	void allocate(luisa::compute::Device& device, size_t size) noexcept;

	size_t m_size = 0;
	size_t m_capacity = 0;
};
}// namespace sail::inno::sph