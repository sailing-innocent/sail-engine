#pragma once
/**
 * @author Oncle-Ha
 * @date 2023-04-07
 */

#include <luisa/core/basic_types.h>
#include <luisa/runtime/buffer.h>
#include <luisa/runtime/device.h>
#include <luisa/core/logging.h>

#include "core/lazy/lazy.h"
#include "sph_executor.h"

namespace inno::csigsph {
class Neighbor;
class SPHSolver;

class BaseSPH : public SPHExecutor {
    template<typename T>
    using U = luisa::unique_ptr<T>;

    template<typename T>
    using Buffer = luisa::compute::Buffer<T>;

    friend class SPHSolver;
    friend class Neighbor;

public:
    BaseSPH(SPHSolver &solver) noexcept;

    Buffer<float> m_rho;
    Buffer<float> m_pres;
    Buffer<float> m_corrected_pres;
    Buffer<luisa::float3> m_delta_vel_vis;
    Buffer<luisa::float3> m_delta_vel_pres;
    Buffer<float> m_pres_factor;

    float m_mass;
    float m_kpci;
    auto size() const noexcept { return m_size; }
    virtual void before_iter(luisa::compute::CommandList &cmdlist) noexcept;

protected:
    virtual void create() noexcept;

    virtual void compile() noexcept;

    void allocate(luisa::compute::Device &device, size_t size) noexcept;
    void init_mass() noexcept;
    void init_kpci() noexcept;
    void init_cubic() noexcept;
    void reset() noexcept;

    virtual void iteration(luisa::compute::CommandList &cmdlist) noexcept;
    virtual void predict(luisa::compute::CommandList &cmdlist) noexcept;
    virtual void after_iter(luisa::compute::CommandList &cmdlist) noexcept;

    size_t m_size = 0;
    size_t m_capacity = 0;

    UCallable<float(float3, float)> smoothKernel;
    UCallable<float3(float3, float)> smoothGrad;

    Global1D<float, float, float, float, float, float, int, float>::UShader neighborSearch_Rho;
    Global1D<float, float, float, float, float, float, float3, int, float>::UShader neighborSearch_Vis;

    Global1D<int, float, float>::UShader updateStates;
    // Global1D<int>::UShader updateGravity;
};

class WCSPH : public BaseSPH {
    template<typename T>
    using U = luisa::unique_ptr<T>;

    template<typename T>
    using Buffer = luisa::compute::Buffer<T>;

    friend class SPHSolver;
    friend class Neighbor;

public:
    WCSPH(SPHSolver &solver) noexcept;

    auto size() const noexcept { return m_size; }

protected:
    void create() noexcept override;
    ;

    void compile() noexcept override;
    ;

    void allocate(luisa::compute::Device &device, size_t size) noexcept;

    void iteration(luisa::compute::CommandList &cmdlist) noexcept override;

    Global1D<int, float, float, float, float, float>::UShader updatePres;
    Global1D<float, float, int, float>::UShader neighborSearch_Pres;

    Global1D<int, float, float, float, float, float, float3>::UShader forceSearch_Force;
    Global1D<int, float, float>::UShader forceSearch_Rho;
};

class PCISPH : public BaseSPH {
    template<typename T>
    using U = luisa::unique_ptr<T>;

    template<typename T>
    using Buffer = luisa::compute::Buffer<T>;

    friend class SPHSolver;
    friend class Neighbor;

public:
    PCISPH(SPHSolver &solver) noexcept;

    Buffer<luisa::float3> m_predicted_pos;
    // Buffer<luisa::float3> m_predicted_vel;

    auto size() const noexcept { return m_size; }

protected:
    void create() noexcept override;

    void compile() noexcept override;

    void allocate(luisa::compute::Device &device, size_t size) noexcept;

    void predict(luisa::compute::CommandList &cmdlist) noexcept override;
    void iteration(luisa::compute::CommandList &cmdlist) noexcept override;
    void after_iter(luisa::compute::CommandList &cmdlist) noexcept override;
// Shaders
    Global1D<int, float>::UShader predictPosAndVel;
    Global1D<float, float, float, float, int, float>::UShader neighborSearch_TmpRho;
    Global1D<float, float, int, float>::UShader neighborSearch_CorPres;
};

}// namespace inno::csigsph