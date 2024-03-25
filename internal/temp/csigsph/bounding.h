#pragma once
/**
 * @author Oncle-Ha
 * @date 2023-05-26
 */

#include <luisa/core/basic_types.h>
#include <luisa/runtime/buffer.h>
#include <luisa/runtime/device.h>
#include <luisa/core/logging.h>

#include "core/lazy/lazy.h"

#include "sph_executor.h"
#include "fluid_particles.h"

#include "packages/buffer/buffer_filler.h"
#include "packages/parallel/device_parallel.h"

namespace inno::csigsph {
class SPHSolver;

// Boundary restrictions
class Bounding : public SPHExecutor {
    template<typename T>
    using U = luisa::unique_ptr<T>;

    template<typename T>
    using Buffer = luisa::compute::Buffer<T>;

    friend class SPHSolver;

public:
    Bounding(SPHSolver &solver) noexcept;
    // auto size() const noexcept { return m_size; }
    void solve(luisa::compute::CommandList &cmdlist) noexcept;

private:
    void create() noexcept;

    void compile() noexcept;
    void reset() noexcept;

    // void allocate(luisa::compute::Device &device, size_t size) noexcept;

    // size_t m_size = 0;
    // size_t m_capacity = 0;

    Global1D<int, float, float>::UShader bounding_cube;
    Global1D<int, float, float>::UShader bounding_sphere;
    Global1D<int, float, float>::UShader bounding_waterfall;
    Global1D<int, float, float>::UShader bounding_heightmap;
};
}// namespace inno::csigsph