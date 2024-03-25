/**
 * @author Oncle-Ha
 * @date 2023-04-06
 */

#include "fluid_particles.h"
#include "sph.h"

namespace inno::csigsph {

void FluidParticles::create() noexcept {
    m_capacity = solver().config().n_capacity;
    LUISA_INFO("Size: {} {}", m_size, m_capacity);
    allocate(solver().device(), m_capacity);
}

FluidParticles::FluidParticles(SPHSolver &solver) noexcept : SPHExecutor{solver} {
}

// add
int FluidParticles::push_particles(const luisa::vector<luisa::float3> &host_pos) noexcept {
    for (size_t i = 0; i < host_pos.size(); i++) {
        m_h_pos.push_back(host_pos[i]);
        m_h_id.push_back(m_size + i);
    }
    m_size += host_pos.size();
    return m_size;
}

// replace
int FluidParticles::place_particles(const luisa::vector<luisa::float3> &host_pos) noexcept {
    m_size = 0;
    m_h_pos.clear();
    m_h_id.clear();
    for (size_t i = 0; i < host_pos.size(); i++) {
        m_h_pos.push_back(host_pos[i]);
        m_h_id.push_back(i);
    }
    m_size = m_h_pos.size();
    LUISA_INFO("Place particles: {}", m_size);
    return m_size;
}

void FluidParticles::allocate(luisa::compute::Device &device, size_t size) noexcept {
    m_id = device.create_buffer<int>(size);
    m_pos = device.create_buffer<luisa::float3>(size);
    m_vel = device.create_buffer<luisa::float3>(size);
}

void FluidParticles::init_upload(luisa::compute::CommandList &cmdlist) noexcept {
    using namespace luisa;
    using namespace luisa::compute;
    auto &filler = solver().filler();
    LUISA_ASSERT(m_capacity >= m_size, "The size of particle must smaller than capacity.");
    LUISA_ASSERT(m_h_pos.size() == m_size, "The size of pos must equal to m_size.");
    LUISA_ASSERT(m_h_id.size() == m_size, "The size of id must equal to m_size.");

    cmdlist << m_pos.view(0, m_size).copy_from(m_h_pos.data())
            << m_id.view(0, m_size).copy_from(m_h_id.data())
            << filler.fill(m_vel.view(0, m_size), make_float3(0.0f));// clear vel
}

void FluidParticles::reset(luisa::compute::CommandList &cmdlist) noexcept {
    using namespace luisa;
    using namespace luisa::compute;
    auto &filler = solver().filler();
    LUISA_ASSERT(m_capacity >= m_size, "The size of particle must smaller than capacity.");
    cmdlist << m_pos.view(0, m_size).copy_from(m_h_pos.data())
            << m_id.view(0, m_size).copy_from(m_h_id.data())
            << filler.fill(m_vel.view(0, m_size), make_float3(0.0f));// clear vel
}
}// namespace inno::csigsph