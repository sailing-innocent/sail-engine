#pragma once
/**
 * @file source/package/solver/fluid/sph/model/wc_sph.h
 * @author sailing-innocent
 * @date 2023-02-23
 * @brief WCSPH
 */
#include "base_sph.h"
namespace sail::inno::sph {

class SAIL_INNO_API WCSPH : public BaseSPH {
public:
	WCSPH(SPHSolver& solver) noexcept;
	// create keep
	// allocate nothing
	void compile(Device& device) noexcept override;
	void iteration(CommandList& cmdlist) noexcept override;

private:
	// shaders
};

}// namespace sail::inno::sph