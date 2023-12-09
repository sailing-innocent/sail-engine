#pragma once

/**
 * @file source/package/solver/fluid/sph/sph_executor.h
 * @author sailing-innocent
 * @date 2023-02-22
 * @brief (temp) General SPH Executor for data share
 */
#include "SailInno/core/runtime.h"

#include <luisa/core/basic_types.h>
#include <luisa/runtime/buffer.h>
#include <luisa/runtime/stream.h>
#include <luisa/runtime/device.h>
#include <luisa/runtime/context.h>

namespace sail::inno::sph {

class SPHSolver;

class SAIL_INNO_API SPHExecutor : public LuisaModule {
public:
	SPHExecutor(SPHSolver& solver) noexcept;
	const SPHSolver& solver() const noexcept;
	SPHSolver& solver() noexcept;

private:
	SPHSolver& m_solver;
};

}// namespace sail::inno::sph