#pragma once
/**
 * @file wcsph.h
 * @brief The WCSPH implementation
 * @author sailing-innocent
 * @date 2024-05-02
 */

#include "base.h"

namespace sail::inno::sph {

class WCSPH : public BaseSPH {
	template<typename T>
	using U = luisa::unique_ptr<T>;

	template<typename T>
	using Buffer = luisa::compute::Buffer<T>;

	friend class SPHSolver;
	friend class Neighbor;

public:
	WCSPH(SPHSolver& solver) noexcept;

	auto size() const noexcept { return m_size; }

protected:
	void create(Device& device) noexcept override;
	void compile(Device& device) noexcept override;
	void allocate(luisa::compute::Device& device, size_t size) noexcept;

	void iteration(luisa::compute::CommandList& cmdlist) noexcept override;

	U<Shader<1, int, float, float, float, float, float>> updatePres;
	U<Shader<1, float, float, int, float>> neighborSearch_Pres;
	U<Shader<1, int, float, float, float, float, float, float3>> forceSearch_Force;
	U<Shader<1, int, float, float>> forceSearch_Rho;
};

}// namespace sail::inno::sph