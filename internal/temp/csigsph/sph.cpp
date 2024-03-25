/**
 * @file sph.cpp
 * @author Oncle-Ha
 * @brief SPH solver
 * @date 2023-04-06
 */
#include "sph.h"
#include "core/package/package.h"
#include "fluid_particles.h"
#include "neighbor.h"
#include "bounding.h"
#include "sph.h"
#include "sph_executor.h"

#include <luisa/core/clock.h>

namespace inno::csigsph {

SPHSolver::SPHSolver() noexcept {
    m_particles = luisa::make_unique<FluidParticles>(*this);
    m_neighbor = luisa::make_unique<Neighbor>(*this);
    m_bounding = luisa::make_unique<Bounding>(*this);
}

SPHSolver::~SPHSolver() noexcept {
}

void SPHSolver::config(const SPHSolverConfig &config) noexcept {
    m_config = config;
    m_filler = PackageGlobal::require<BufferFiller>();
    m_device_parallel = PackageGlobal::require<primitive::DeviceParallel>();

    if (m_config.sph_model_kind == 0)
        m_sphmodel = luisa::make_unique<WCSPH>(*this);
    else if (m_config.sph_model_kind == 1)
        m_sphmodel = luisa::make_unique<PCISPH>(*this);
    else
        LUISA_ERROR_WITH_LOCATION("SPH model kind not supported");
}

void SPHSolver::create() noexcept {
    // m_stream = device().create_stream(luisa::compute::StreamTag::GRAPHICS);
    m_particles->create();
    m_neighbor->create();
    m_sphmodel->create();
}
void SPHSolver::compile() noexcept {
    m_neighbor->compile();
    m_sphmodel->compile();
    m_bounding->compile();
}

void SPHSolver::init_upload(CommandList &cmdlist) noexcept {
    m_particles->init_upload(cmdlist);
}

void SPHSolver::param(const SPHParam &param) noexcept {
    m_param = param;
}

void SPHSolver::reset(CommandList &cmdlist) noexcept {
    m_particles->reset(cmdlist);
    m_sphmodel->reset();
    m_neighbor->reset();
}

void SPHSolver::step(CommandList &cmdlist) noexcept {
    luisa::Clock clock;
    for (size_t i = 0; i < m_config.max_iter; ++i) {
        // LUISA_INFO("iter: {}", i);
        m_neighbor->solve(cmdlist);
        // if (m_stream) {
        //     stream() << [&] { clock.tic(); };
        //     stream() << cmdlist.commit();
        //     stream() << [&] { LUISA_INFO("step: neighbor time = {}", clock.toc()); };
        // }

        if (m_config.sph_model_kind == 1) {// PCISPH
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
        // if (m_stream) {
        //     stream() << [&] { clock.tic(); };
        //     stream() << cmdlist.commit();
        //     stream() << [&] { LUISA_INFO("step: solve time = {}", clock.toc()); };
        // }
    }
}

void SPHSolver::setup_iteration(CommandList &cmdlist) noexcept {
    m_neighbor->solve(cmdlist);
    m_sphmodel->predict(cmdlist);
}

void SPHSolver::iteration(CommandList &cmdlist) noexcept {
    m_sphmodel->iteration(cmdlist);
}

void SPHSolver::finish_iteration(CommandList &cmdlist) noexcept {
    m_sphmodel->after_iter(cmdlist);
    m_bounding->solve(cmdlist);
}

}// namespace inno::csigsph