#pragma once
/**
 * @file source/package/solver/fluid/sph/model/pci_sph.h
 * @author sailing-innocent
 * @date 2023-02-23
 * @brief WCSPH
 */
#include "base_sph.h"

namespace sail::inno::sph {

class SAIL_INNO_API PCISPH : public BaseSPH {
public:
	PCISPH(SPHSolver& solver) noexcept
		: BaseSPH(solver){};
	// create keep
	// allocate nothing
	void compile(Device& device) noexcept override;
	void iteration(CommandList& cmdlist) noexcept override;
};

}// namespace sail::inno::sph