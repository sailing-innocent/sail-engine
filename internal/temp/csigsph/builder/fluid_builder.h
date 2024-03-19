#pragma once
/**
 * @author Oncle-Ha
 * @date 2023-04-08
 */
#include <luisa/runtime/buffer.h>
#include <luisa/runtime/device.h>
#include <luisa/runtime/stream.h>
#include <luisa/runtime/context.h>

#include <luisa/core/stl/vector.h>
#include "../sph.h"

namespace inno::csigsph {
class Fluid {
public:
    friend class FluidBuilder;
    // particle pos
    luisa::vector<luisa::float3> h_pos;
    void to_csv(const std::string &path) const noexcept;
};

class FluidBuilder {
public:
    FluidBuilder(SPHSolver &solver) noexcept : m_solver{solver} {}
    Fluid grid(const luisa::float3 &bottom_left_pos, const luisa::float3 &grid_size, const float &dx2) noexcept;
    void push_particle(Fluid &fluid);
    void place_particle(Fluid &fluid);
    void download(luisa::compute::CommandList &cmdlist, Fluid &fluid) noexcept;

private:
    SPHSolver &m_solver;
};
}// namespace inno::csigsph
