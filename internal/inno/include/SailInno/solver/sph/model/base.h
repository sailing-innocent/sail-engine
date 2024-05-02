#pragma once
/**
 * @file sphbase.h
 * @brief The SPHBase
 * @author sailing-innocent
 * @date 2024-05-02
 */
#include <luisa/luisa-compute.h>
#include "SailInno/solver/sph/sph_executor.h"
namespace sail::inno::sph {
class Neighbor;
class SPHSolver;

class BaseSPH : public SPHExecutor {
	friend class SPHSolver;
	friend class Neighbor;

public:
	BaseSPH(SPHSolver& solver) noexcept;

	Buffer<float> m_rho;
	Buffer<float> m_pres;
	Buffer<float> m_corrected_pres;
	Buffer<luisa::float3> m_delta_vel_vis;
	Buffer<luisa::float3> m_delta_vel_pres;
	Buffer<float> m_pres_factor;

	float m_mass;
	float m_kpci;
	auto size() const noexcept { return m_size; }
	virtual void before_iter(luisa::compute::CommandList& cmdlist) noexcept;

protected:
	virtual void create(Device& device) noexcept;
	virtual void compile(Device& device) noexcept;
	void allocate(Device& device, size_t size) noexcept;
	void init_mass() noexcept;
	void init_kpci() noexcept;
	void init_cubic() noexcept;
	void reset() noexcept;

	virtual void iteration(luisa::compute::CommandList& cmdlist) noexcept;
	virtual void predict(luisa::compute::CommandList& cmdlist) noexcept;
	virtual void after_iter(luisa::compute::CommandList& cmdlist) noexcept;

	size_t m_size = 0;
	size_t m_capacity = 0;

	UCallable<float(float3, float)> smoothKernel;
	UCallable<float3(float3, float)> smoothGrad;

	U<Shader<1, float, float, float, float, float, float, int, float>> neighborSearch_Rho;
	U<Shader<1, float, float, float, float, float, float, float3, int, float>> neighborSearch_Vis;
	U<Shader<1, int, float, float>> updateStates;
};
}// namespace sail::inno::sph