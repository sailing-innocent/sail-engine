/**
 * @author Oncle-Ha
 * @date 2023-04-06
 */
#pragma once
#include <luisa/runtime/buffer.h>
#include <luisa/runtime/device.h>
#include <luisa/runtime/stream.h>
#include <luisa/runtime/context.h>

namespace inno::csigsph {
class SPHSolver;
class SPHExecutor {
public:
    SPHExecutor(SPHSolver &solver) noexcept;

protected:
    template<typename T>
    using Buffer = luisa::compute::Buffer<T>;

    template<typename T>
    using BufferView = luisa::compute::BufferView<T>;

    using CommandList = luisa::compute::CommandList;

    using Command = luisa::compute::Command;

    using Stream = luisa::compute::Stream;

    using Device = luisa::compute::Device;

    using Synchronize = luisa::compute::Stream::Synchronize;

    using float2 = luisa::float2;
    using float3 = luisa::float3;
    using float4 = luisa::float4;

    using float2x2 = luisa::float2x2;
    using float3x3 = luisa::float3x3;
    using float4x4 = luisa::float4x4;

    using int2 = luisa::int2;
    using int3 = luisa::int3;
    using int4 = luisa::int4;

    using uint = luisa::uint;
    using uint2 = luisa::uint2;
    using uint3 = luisa::uint3;
    using uint4 = luisa::uint4;

    using bool2 = luisa::bool2;
    using bool3 = luisa::bool3;
    using bool4 = luisa::bool4;

    const SPHSolver &solver() const noexcept;
    SPHSolver &solver() noexcept;

private:
    SPHSolver &m_solver;
};
}// namespace inno::csigsph
