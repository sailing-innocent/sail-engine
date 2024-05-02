/**
 * @file source/package/solver/fluid/sph/model/base_sph.cpp
 * @author sailing-innocent
 * @date 2023-02-23
 * @brief (impl) SPH Fluid Model Base
 */

#include "SailInno/solver/sph/model/base_sph.h"
#include "SailInno/solver/sph/solver.h"
#include "SailInno/solver/sph/fluid_particles.h"
#include "SailInno/helper/grid_stride_loop.h"

namespace sail::inno::sph {

BaseSPH::BaseSPH(SPHSolver& solver) noexcept
	: SPHExecutor(solver) {
}

void BaseSPH::create(Device& device) noexcept {};

void BaseSPH::iteration(CommandList& cmdlist) noexcept {
	// mp_solver->step(cmdlist);
}

}// namespace sail::inno::sph

// Core Ipml

namespace sail::inno::sph {

void BaseSPH::compile(Device& device) noexcept {
	using namespace luisa;
	using namespace luisa::compute;
	const size_t n_blocks = solver().config().n_blocks;
	const size_t n_threads = solver().config().n_threads;

	auto& particles = solver().particles();

	lazy_compile(device, ms_update_state, [&particles, &n_blocks](Int count, Float delta_time, Float rate) {
		set_block_size(n_blocks);
		grid_stride_loop(count, [&particles, &delta_time](Int p) noexcept {
			Float3 x = particles.m_d_pos->read(p);
			Float3 v = particles.m_d_vel->read(p);
			// update v
			// update x
			x = x + v * delta_time;
		});
	});
}

}// namespace sail::inno::sph