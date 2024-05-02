/**
 * @file source/package/solver/fluid/sph/fluid_particles.cpp
 * @author sailing-innocent
 * @date 2023-02-22
 * @brief (impl) SPH Fluid Particles
 */
#include "SaiLInno/solver/sph/fluid_particles.h"
#include "SailInno/solver/sph/solver.h"
#include <luisa/core/logging.h>
#include <malloc.h>

namespace sail::inno::sph {

SPHFluidParticles::SPHFluidParticles(SPHSolver& solver) noexcept
	: SPHExecutor(solver) {}

void SPHFluidParticles::create(Device& device) noexcept {
	m_max_size = solver().config().n_capacity;
	allocate(device, m_max_size);
}

int SPHFluidParticles::place_particles(const luisa::vector<luisa::float3>& host_pos) noexcept {
	// clear and place
	m_size = 0;
	m_h_pos.clear();
	m_h_id.clear();
	for (size_t i = 0; i < host_pos.size(); i++) {
		m_h_pos.push_back(host_pos[i]);
		m_h_id.push_back(i);
	}
	m_size = host_pos.size();
	LUISA_INFO("SPHFluidParticles::place_particles: size = {}", m_size);
	return m_size;
}

int SPHFluidParticles::push_particles(const luisa::vector<luisa::float3>& host_pos) noexcept {
	// append to tail
	for (size_t i = 0; i < host_pos.size(); i++) {
		m_h_pos.push_back(host_pos[i]);
		m_h_id.push_back(i + m_size);
	}
	m_size += host_pos.size();
	return m_size;
}

void SPHFluidParticles::allocate(Device& device, size_t size) noexcept {
	m_d_pos = device.create_buffer<luisa::float3>(size);
	m_d_vel = device.create_buffer<luisa::float3>(size);
	m_d_id = device.create_buffer<int>(size);
}

void SPHFluidParticles::init_upload(Device& device, CommandList& cmdlist) noexcept {
	using namespace luisa;
	using namespace luisa::compute;
	LUISA_ASSERT(m_max_size >= m_size, "The size of particle must smaller than capacity.");
	LUISA_ASSERT(m_h_pos.size() == m_size, "The size of pos must equal to m_size.");
	LUISA_ASSERT(m_h_id.size() == m_size, "The size of id must equal to m_size.");
	reset(device, cmdlist);
}

void SPHFluidParticles::reset(Device& device, CommandList& cmdlist) noexcept {
	using namespace luisa;
	using namespace luisa::compute;
	LUISA_ASSERT(m_max_size >= m_size, "The size of particle must smaller than capacity.");
	cmdlist << m_d_pos.view(0, m_size).copy_from(m_h_pos.data())
			<< m_d_id.view(0, m_size).copy_from(m_h_id.data())
			<< solver().filler().fill(device, m_d_vel.view(0, m_size), make_float3(0.0f));// clear vel
}

}// namespace sail::inno::sph