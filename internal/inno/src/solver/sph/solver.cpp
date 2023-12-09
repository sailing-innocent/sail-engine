/**
 * @file source/package/solver/fluid/sph/sph.cpp
 * @author sailing-innocent
 * @date 2023-02-22
 * @brief (impl) Fluid Solver
 */

#include "SailInno/solver/sph/solver.h"
#include "SailInno/solver/sph/fluid_particles.h"
#include "SailInno/solver/sph/neighbor.h"
#include "SailInno/solver/sph/bounding.h"
#include "SailInno/solver/sph/model/base_sph.h"
#include "SailInno/solver/sph/model/dummy_sph.h"
#include "SailInno/solver/sph/model/wc_sph.h"
#include "SailInno/solver/sph/model/pci_sph.h"

#include <luisa/core/clock.h>

namespace sail::inno::sph {

SPHSolver::SPHSolver() noexcept {
	mp_particles = luisa::make_unique<SPHFluidParticles>(*this);
	mp_buffer_filler = luisa::make_unique<BufferFiller>();
}

SPHSolver::~SPHSolver() noexcept {
}

void SPHSolver::config(const SPHSolverConfig& config) noexcept {
	m_config = config;

	if (m_config.model_kind == SPHModelKind::DUMMY) {
		mp_sph_model = luisa::make_unique<DummySPH>(*this);
	} else if (m_config.model_kind == SPHModelKind::WCSPH) {
		mp_sph_model = luisa::make_unique<WCSPH>(*this);
	} else if (m_config.model_kind == SPHModelKind::PCISPH) {
		mp_sph_model = luisa::make_unique<PCISPH>(*this);
	}
}

void SPHSolver::create(Device& device) noexcept {
	mp_particles->create(device);
	mp_sph_model->create(device);
}

void SPHSolver::compile(Device& device) noexcept {
	mp_sph_model->compile(device);
}

void SPHSolver::init_upload(Device& device, CommandList& cmdlist) noexcept {
	mp_particles->init_upload(device, cmdlist);
}

void SPHSolver::reset(CommandList& cmdlist) noexcept {
}

void SPHSolver::step(CommandList& cmdlist) noexcept {
	luisa::Clock clock;

	for (size_t i = 0; i < m_config.max_iter; i++) {
		if (m_config.model_kind == SPHModelKind::DUMMY) {
			mp_sph_model->iteration(cmdlist);
		} else if (m_config.model_kind == SPHModelKind::WCSPH) {
			mp_sph_model->iteration(cmdlist);
		} else if (m_config.model_kind == SPHModelKind::PCISPH) {
			mp_sph_model->iteration(cmdlist);
		}
	}
}

}// namespace sail::inno::sph