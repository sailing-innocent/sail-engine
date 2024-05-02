/**
 * @file solver.cpp
 * @brief The Solver Implementation
 * @author sailing-innocent
 * @date 2024-05-02
 */
#include "SailInno/solver/sph/solver.h"
#include "SailInno/solver/sph/fluid_particles.h"
#include "SailInno/solver/sph/neighbor.h"
#include "SailInno/solver/sph/bounding.h"
#include "SailInno/solver/sph/sph_executor.h"

#include <luisa/core/clock.h>

namespace sail::inno::sph {

SPHSolver::SPHSolver() noexcept {
	m_particles = luisa::make_unique<FluidParticles>(*this);
	m_neighbor = luisa::make_unique<Neighbor>(*this);
	m_bounding = luisa::make_unique<Bounding>(*this);
}

SPHSolver::~SPHSolver() noexcept {
	m_particles = nullptr;
	m_neighbor = nullptr;
	m_bounding = nullptr;
}

void SPHSolver::config(const SPHSolverConfig& config) noexcept {
	m_config = config;
	m_filler = luisa::make_unique<BufferFiller>();
	m_device_parallel = luisa::make_unique<DeviceParallel>();
	if (m_config.sph_model_kind == 0) {
		m_sphmodel = luisa::make_unique<WCSPH>(*this);
	} else if (m_config.sph_model_kind == 1) {
		m_sphmodel = luisa::make_unique<PCISPH>(*this);
	} else {
		LUISA_ERROR_WITH_LOCATION("SPH model kind not supported");
	}
}

void SPHSolver::create(Device& device) noexcept {
	m_particles->create(device);
	m_neighbor->create(device);
	m_sphmodel->create(device);
	m_device_parallel->create(device);
}
void SPHSolver::compile(Device& device) noexcept {
	m_neighbor->compile(device);
	m_sphmodel->compile(device);
	m_bounding->compile(device);
}

void SPHSolver::init_upload(Device& device, CommandList& cmdlist) noexcept {
	m_particles->init_upload(device, cmdlist);
}

void SPHSolver::param(const SPHParam& param) noexcept {
	m_param = param;
}

void SPHSolver::reset(Device& device, CommandList& cmdlist) noexcept {
	m_particles->reset(device, cmdlist);
	m_sphmodel->reset();
	m_neighbor->reset();
}

void SPHSolver::step(Device& device, CommandList& cmdlist) noexcept {
	luisa::Clock clock;
	for (size_t i = 0; i < m_config.max_iter; ++i) {
		m_neighbor->solve(device, cmdlist);

		if (m_config.sph_model_kind == 1) {// PCISPH
			// LUISA_INFO("Choose PCISPH");
			m_sphmodel->predict(cmdlist);
			for (size_t j = 0; j < m_config.least_iter; ++j) {
				m_sphmodel->iteration(cmdlist);
			}
			m_sphmodel->after_iter(cmdlist);
		} else if (m_config.sph_model_kind == 0) {// WCSPH
			// LUISA_INFO("Choose WCSPH");
			m_sphmodel->iteration(cmdlist);
		}
		m_bounding->solve(cmdlist);
	}
}

void SPHSolver::setup_iteration(Device& device, CommandList& cmdlist) noexcept {
	m_neighbor->solve(device, cmdlist);
	m_sphmodel->predict(cmdlist);
}

void SPHSolver::iteration(CommandList& cmdlist) noexcept {
	m_sphmodel->iteration(cmdlist);
}

void SPHSolver::finish_iteration(CommandList& cmdlist) noexcept {
	m_sphmodel->after_iter(cmdlist);
	m_bounding->solve(cmdlist);
}

}// namespace sail::inno::sph