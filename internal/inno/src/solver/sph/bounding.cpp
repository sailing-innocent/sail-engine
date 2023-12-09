/**
 * @file source/package/solver/fluid/sph/bounding.cpp
 * @author sailing-innocent
 * @date 2023-02-24
 * @brief (impl) SPH Fluid Bounding
 */

#include "SailInno/solver/sph/bounding.h"
#include "SailInno/solver/sph/solver.h"
#include "SailInno/solver/sph/sph_executor.h"
#include "SailInno/solver/sph/fluid_particles.h"

using namespace luisa;
using namespace luisa::compute;

// API implementation
namespace sail::inno::sph {

Bounding::Bounding(SPHSolver& solver) noexcept
	: SPHExecutor(solver) {
}

void create(Device& device) noexcept {
}

void solve(CommandList& cmdlist) noexcept {
}

void reset() noexcept {
}

}// namespace sail::inno::sph

// Core implementation
namespace inno::sph {

void compile(Device& device) noexcept {}

}// namespace inno::sph