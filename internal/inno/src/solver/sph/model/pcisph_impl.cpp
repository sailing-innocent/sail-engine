#include "SailInno/solver/sph/model/pcisph.h"
#include "SailInno/solver/sph/solver.h"
namespace sail::inno::sph {

PCISPH::PCISPH(SPHSolver& solver) noexcept : BaseSPH(solver) {
}

void PCISPH::create(Device& device) noexcept {
	BaseSPH::create(device);
	int num_particles = m_size;
	allocate(device, num_particles);
}

void PCISPH::allocate(luisa::compute::Device& device, size_t size) noexcept {
	m_predicted_pos = device.create_buffer<luisa::float3>(size);
	// m_predicted_vel = device.create_buffer<luisa::float3>(size);
}

void PCISPH::iteration(luisa::compute::CommandList& cmdlist) noexcept {
	auto n_particles = m_size;
	auto num_thread = solver().neighbor().m_num_thread_up;
	auto n_grids = solver().neighbor().m_num_grids;
	auto cell_size = solver().neighbor().m_cell_size;
	// LUISA_INFO("PCISPH Cell Size:{}", cell_size);
	auto h_fac = solver().param().h_fac;
	auto rho_0 = solver().param().rho_0;
	auto delta_time = solver().param().delta_time;

	cmdlist << (*predictPosAndVel)(n_particles, delta_time).dispatch(n_particles)
			<< (*neighborSearch_TmpRho)(m_mass, m_kpci, h_fac, rho_0, n_grids, cell_size).dispatch(num_thread)
			<< (*neighborSearch_CorPres)(m_mass, h_fac, n_grids, cell_size).dispatch(num_thread);
}

void PCISPH::predict(luisa::compute::CommandList& cmdlist) noexcept {
	auto num_thread = solver().neighbor().m_num_thread_up;
	auto n_grids = solver().neighbor().m_num_grids;
	auto cell_size = solver().neighbor().m_cell_size;
	// LUISA_INFO("PCISPH Cell Size:{}", cell_size);
	auto h_fac = solver().param().h_fac;
	auto alpha = solver().param().alpha;
	auto stiffB = solver().param().stiffB;
	auto gamma = solver().param().gamma;
	auto rho_0 = solver().param().rho_0;
	auto gravity = solver().param().gravity;
	cmdlist << (*neighborSearch_Rho)(m_mass, h_fac, alpha, stiffB, gamma, rho_0, n_grids, cell_size).dispatch(num_thread)
			<< (*neighborSearch_Vis)(m_mass, h_fac, alpha, stiffB, gamma, rho_0, gravity, n_grids, cell_size).dispatch(num_thread);
}

void PCISPH::after_iter(luisa::compute::CommandList& cmdlist) noexcept {
	auto n_particles = m_size;
	auto delta_time = solver().param().delta_time;
	auto rate = solver().param().collision_rate;

	cmdlist << (*updateStates)(n_particles, delta_time, rate).dispatch(n_particles);
}
}// namespace sail::inno::sph