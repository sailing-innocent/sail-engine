/**
 * @file source/package/solver/fluid/sph/model/wc_sph.cpp
 * @author sailing-innocent
 * @date 2023-02-23
 * @brief WCSPH implementation
 */

// TODO Algorithm Migrate

#include "SailInno/solver/sph/model/wc_sph.h"
#include "SailInno/solver/sph/model/base_sph.h"
#include "SailInno/solver/sph/model/../solver.h"
#include "SailInno/solver/sph/model/../fluid_particles.h"

#include <luisa/core/logging.h>
// API implementation
namespace sail::inno::sph {

WCSPH::WCSPH(SPHSolver& solver) noexcept
	: BaseSPH(solver) {
	LUISA_INFO("WCSPH created.");
}

void WCSPH::iteration(CommandList& cmdlist) noexcept {
	auto n_particles = solver().particles().size();
	auto n_threads = solver().config().n_threads;

	// params
	auto delta_time = solver().param().delta_time;
	auto rate = solver().param().collision_rate;

	cmdlist << (*ms_update_state)(n_particles, delta_time, rate).dispatch(n_particles);
}

}// namespace sail::inno::sph

// Core implementation

namespace sail::inno::sph {

void WCSPH::compile(Device& device) noexcept {
	using namespace luisa;
	using namespace luisa::compute;
	const size_t n_blocks = solver().config().n_blocks;
	const size_t n_threads = solver().config().n_threads;

	auto& particles = solver().particles();

	// $x = -\ddot(x)$
	lazy_compile(device, ms_update_state, [&particles, &n_blocks](Int count, Float delta_time, Float rate) {
		set_block_size(n_blocks);
		grid_stride_loop(count, [&particles, &delta_time](Int p) noexcept {
			Float3 x = particles.m_d_pos->read(p);
			Float3 v = particles.m_d_vel->read(p);

			// update v
			v.z = v.z - 0.01f * delta_time * x.z;
			// update x
			x = x + v * delta_time;

			particles.m_d_pos->write(p, x);
			particles.m_d_vel->write(p, v);
		});
	});
}

}// namespace sail::inno::sph
