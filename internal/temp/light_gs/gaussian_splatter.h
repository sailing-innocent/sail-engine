#pragma once
/**
 * @file packages/render/light_gs/gaussian_splatter.h
 * @author sailing-innocent
 * @date 2024/01/02
 * @brief the shaded gaussian splatter
 */

#include "package/diff_render/gs/gaussian_splatter.h"

namespace inno::render
{

class LightGaussianSplatter : public GaussianSplatter
{
    template <typename T>
    using Buffer = luisa::compute::Buffer<T>;
    template <typename T>
    using BufferView = luisa::compute::BufferView<T>;
    template <typename T>
    using Image = luisa::compute::Image<T>;
    template <typename T>
    using ImageView = luisa::compute::ImageView<T>;
    template <size_t I, typename... Ts>
    using Shader      = luisa::compute::Shader<I, Ts...>;
    using Device      = luisa::compute::Device;
    using CommandList = luisa::compute::CommandList;
    using float2      = luisa::float2;
    using float3      = luisa::float3;
    using float4      = luisa::float4;
    using float3x3    = luisa::float3x3;
    using float4x4    = luisa::float4x4;
    using uint        = luisa::uint;
    using uint2       = luisa::uint2;
    using ulong       = luisa::ulong;
    using Stream      = luisa::compute::Stream;

public:
    virtual void forward_impl(
    Device& device,
    Stream& stream,
    int height, int width,
    BufferView<float> target_img_buffer, // hwc
    int num_gaussians, int sh_deg, int max_sh_deg,
    BufferView<float> xyz_buffer,
    BufferView<float> feature_buffer, // for color
    BufferView<float> opacity_buffer,
    BufferView<float> scale_buffer,
    BufferView<float> rotq_buffer,
    float             scale_modifier,
    Camera&           cam) override;

protected:
    // override method
    virtual void compile(Device& device) noexcept override;
    virtual void compile_forward_preprocess_shader(Device& device) noexcept override;
    virtual void compile_forward_render_shader(Device& device) noexcept override;
    virtual void compile_forward_shade_shader(Device& device) noexcept;
    U<Shader<1,
             int,                           // P
             int,                           // D
             int,                           // M
             Buffer<float>,                 // xyz
             float3,                        // cam_pos
             Buffer<float>,                 // feat buffer
             Buffer<float>,                 // d_color buffer
             Buffer<luisa::compute::float4> // color_features
             >>
    m_mix_color_shader;

    U<Shader<2,
             uint2,          // resolution
             uint2,          // grids
             Buffer<uint>,   // ranges
             Buffer<uint>,   // point_list
             Buffer<float2>, // means_2d
             Buffer<float>,  // dcolor
             Buffer<float4>  // conic_opacity
             >>
    m_forward_shade_shader;
};

} // namespace inno::render