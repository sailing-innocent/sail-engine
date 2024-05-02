#pragma once
/**
 * @file pcisph.h
 * @brief The PCISPH
 * @author sailing-innocent
 * @date 2024-05-02
 */

#include "base.h"

namespace sail::inno::sph {

class PCISPH : public BaseSPH {
	template<typename T>
	using U = luisa::unique_ptr<T>;

	template<typename T>
	using Buffer = luisa::compute::Buffer<T>;

	friend class SPHSolver;
	friend class Neighbor;

public:
	PCISPH(SPHSolver& solver) noexcept;

	Buffer<luisa::float3> m_predicted_pos;
	// Buffer<luisa::float3> m_predicted_vel;
	auto size() const noexcept { return m_size; }

protected:
	void create(Device& device) noexcept override;
	void compile(Device& device) noexcept override;

	void allocate(luisa::compute::Device& device, size_t size) noexcept;

	void predict(luisa::compute::CommandList& cmdlist) noexcept override;
	void iteration(luisa::compute::CommandList& cmdlist) noexcept override;
	void after_iter(luisa::compute::CommandList& cmdlist) noexcept override;
	// Shaders
	U<Shader<1, int, float>> predictPosAndVel;
	U<Shader<1, float, float, float, float, int, float>> neighborSearch_TmpRho;
	U<Shader<1, float, float, int, float>> neighborSearch_CorPres;
};
}// namespace sail::inno::sph