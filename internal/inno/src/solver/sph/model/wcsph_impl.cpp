#include "SailInno/solver/sph/model/wcsph.h"
#include "SailInno/solver/sph/solver.h"

namespace sail::inno::sph {

//WCSPH
WCSPH::WCSPH(SPHSolver& solver) noexcept : BaseSPH(solver) {
}

void WCSPH::create(Device& device) noexcept {
	BaseSPH::create(device);
}

void WCSPH::allocate(luisa::compute::Device& device, size_t size) noexcept {
}

void WCSPH::iteration(CommandList& cmdlist) noexcept {
	auto n_particles = m_size;
	auto num_thread = solver().neighbor().m_num_thread_up;
	auto n_grids = solver().neighbor().m_num_grids;
	auto cell_size = solver().neighbor().m_cell_size;

	auto h_fac = solver().param().h_fac;
	auto alpha = solver().param().alpha;
	auto stiffB = solver().param().stiffB;
	auto gamma = solver().param().gamma;
	auto rho_0 = solver().param().rho_0;
	auto gravity = solver().param().gravity;
	auto delta_time = solver().param().delta_time;
	auto rate = solver().param().collision_rate;
	cmdlist << (*neighborSearch_Rho)(m_mass, h_fac, alpha, stiffB, gamma, rho_0, n_grids, cell_size).dispatch(num_thread)
			<< (*updatePres)(n_particles, h_fac, alpha, stiffB, gamma, rho_0).dispatch(n_particles)
			<< (*neighborSearch_Vis)(m_mass, h_fac, alpha, stiffB, gamma, rho_0, gravity, n_grids, cell_size).dispatch(num_thread)
			<< (*neighborSearch_Pres)(m_mass, h_fac, n_grids, cell_size).dispatch(num_thread)
			<< (*updateStates)(n_particles, delta_time, rate).dispatch(n_particles);
}
}// namespace sail::inno::sph