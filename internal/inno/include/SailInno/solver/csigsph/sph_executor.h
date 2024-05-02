#pragma once
/**
 * @file sph_executor.h
 * @brief The SPH Executor Header
 * @author sailing-innocent
 * @date 2024-05-02
 */

#include "SailInno/core/runtime.h"
#include <luisa/luisa-compute.h>

namespace sail::inno::csigsph {

class SPHSolver;
class SAIL_INNO_API SPHExecutor : public LuisaModule {
public:
	SPHExecutor(SPHSolver& solver) noexcept;

protected:
	const SPHSolver& solver() const noexcept;
	SPHSolver& solver() noexcept;

private:
	SPHSolver& m_solver;
};
}// namespace sail::inno::csigsph
