#pragma once
/**
 * @file source/package/solver/fluid/sph/model/dummy_sph.h
 * @author sailing-innocent
 * @date 2023-02-23
 * @brief Dummy SPH Fluid, let particles move sine wave
 */

#include "base_sph.h"
namespace sail::inno::sph {

class SAIL_INNO_API DummySPH : public BaseSPH {
public:
	DummySPH(SPHSolver& solver) noexcept
		: BaseSPH(solver){};
	void create(Device& device) noexcept override{};
	void compile(Device& device) noexcept override;
	void iteration(CommandList& cmdlist) noexcept override;
};
}// namespace sail::inno::sph
