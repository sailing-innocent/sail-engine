/**
 * @file package/diff_render/light_gs/gaussian_splatter_forward.cpp
 * @author sailing-innocent
 * @date 2024-01-03
 * @brief The Gaussian Splatter Lightable Forward Implement
 */

#include "gaussian_splatter.h"
#include <luisa/backends/ext/cuda/lcub/device_scan.h>
#include <luisa/backends/ext/cuda/lcub/device_radix_sort.h>
#include <luisa/dsl/sugar.h>
#include "util/misc/bit_helper.h"
#include "util/scene/camera.h"

// #include "util/scene/camera.h"

using namespace luisa;
using namespace luisa::compute;

// API

namespace inno::render
{

void LightGaussianSplatter::forward_impl(
Device& device,
Stream& stream,
int height, int width,
BufferView<float> target_img_buffer, // hwc
int               num_gaussians,
int sh_deg, int max_sh_deg,
BufferView<float> xyz_buffer,
BufferView<float> feature_buffer, // for color
BufferView<float> opacity_buffer,
BufferView<float> scale_buffer,
BufferView<float> rotq_buffer,
float             scale_modifier,
Camera&           cam)
{
    using namespace inno::util;
    // update camera
    mp_camera = luisa::make_shared<Camera>(cam);
    // ---------------------------------
    // Prepare
    // ---------------------------------
    int  bit        = get_higher_msb(m_blocks.x * m_blocks.y);
    auto resolution = luisa::make_uint2(width, height);
    m_grids         = luisa::make_uint2(
    (unsigned int)((width + m_blocks.x - 1u) / m_blocks.x),
    (unsigned int)((height + m_blocks.y - 1u) / m_blocks.y));

    if (m_num_gaussians != num_gaussians)
    {
        // if num_gaussians changed, reallocate buffer
        geom_state->allocate(device, static_cast<size_t>(num_gaussians));
        m_num_gaussians = num_gaussians;
    }

    if ((m_resolution.x != width) || (m_resolution.y != height))
    {
        // resolution changed, reallocate image buffer
        img_state->allocate(device, static_cast<size_t>(width * height));
        m_resolution = luisa::make_uint2(width, height);
    }

    CommandList cmdlist;
    // clear geometry state
    geom_state->clear(device, cmdlist, *mp_buffer_filler);
    // ---------------------------------------
    // ---
    // Shading
    // ---
    // ---------------------------------
    inno::Camera light_camera{
        luisa::make_float3(3.0f, 3.0f, 3.0f), // position
        luisa::make_float3(0.0f, 0.0f, 0.0f), // target
        luisa::make_float3(0.0f, 0.0f, 1.0f), // up
        Camera::CoordType::FlipY,
        static_cast<float>(width) / static_cast<float>(height), // aspect
    };
    auto light_camera_primitive = light_camera.camera_primitive(width, height);

    cmdlist
    << (*m_forward_preprocess_shader)(
       num_gaussians, sh_deg, max_sh_deg,
       xyz_buffer,
       feature_buffer,
       opacity_buffer,
       scale_buffer,
       rotq_buffer,
       scale_modifier,
       resolution,
       m_grids,
       geom_state->means_2d,
       geom_state->tiles_touched,
       geom_state->radii,
       geom_state->depth_features,
       geom_state->color_features,
       geom_state->conic_opacity,
       light_camera.pos(),
       light_camera_primitive,
       light_camera.view_matrix(),
       light_camera.proj_matrix())
       .dispatch(num_gaussians);

    cmdlist << luisa::compute::cuda::lcub::DeviceScan::InclusiveSum(
    geom_state->scan_temp_storage,
    geom_state->tiles_touched,
    geom_state->point_offsets, num_gaussians);

    int num_lighten = 0;
    cmdlist << geom_state->point_offsets.view(num_gaussians - 1, 1).copy_to(&num_lighten);

    stream << cmdlist.commit() << synchronize();

    if (num_lighten > 0)
    {
        LUISA_INFO("Forward Shading: {} points lighten.", num_lighten);
        // allocate tiles
        tile_state->allocate(device, static_cast<size_t>(num_lighten));
        tile_state->clear(device, cmdlist, *mp_buffer_filler);
        // duplicate keys
        cmdlist << (*m_copy_with_keys_shader)(
                   num_gaussians,
                   geom_state->means_2d,
                   geom_state->point_offsets,
                   geom_state->radii,
                   geom_state->depth_features,
                   0, // camera depth idx
                   tile_state->point_list_keys_unsorted,
                   tile_state->point_list_unsorted,
                   m_blocks, m_grids)
                   .dispatch(num_gaussians);

        cmdlist << luisa::compute::cuda::lcub::DeviceRadixSort::SortPairs(
        tile_state->sort_temp_storage,
        tile_state->point_list_keys_unsorted,
        tile_state->point_list_keys,
        tile_state->point_list_unsorted,
        tile_state->point_list, num_lighten);

        img_state->clear(device, cmdlist, *mp_buffer_filler);
        // get range
        cmdlist << (*m_get_ranges_shader)(
                   num_lighten,
                   tile_state->point_list_keys,
                   img_state->ranges)
                   .dispatch(num_lighten);

        // Shade
        // calc each point's the lighten coefficients to d_color
        cmdlist << (*m_forward_shade_shader)(
                   resolution,
                   m_grids,
                   img_state->ranges,
                   tile_state->point_list,
                   geom_state->means_2d,
                   geom_state->d_color,
                   geom_state->conic_opacity)
                   .dispatch(resolution);

        // mix color
        // mix the d_color and feature_buffer to color_features
        cmdlist << (*m_mix_color_shader)(
                   num_gaussians,
                   sh_deg, max_sh_deg,
                   xyz_buffer,
                   mp_camera->pos(),
                   feature_buffer,
                   geom_state->d_color, geom_state->color_features)
                   .dispatch(num_gaussians);
    }

    stream << cmdlist.commit() << synchronize();

    // ---------------------------------------
    // ---
    // Rendering
    // ---
    // ---------------------------------

    // ---------------------------------
    // Forward Preprocess
    // ---------------------------------
    cmdlist << (*m_forward_preprocess_shader)(
               num_gaussians, sh_deg, max_sh_deg,
               xyz_buffer,
               feature_buffer,
               opacity_buffer,
               scale_buffer,
               rotq_buffer,
               scale_modifier,
               resolution,
               m_grids,
               geom_state->means_2d,
               geom_state->tiles_touched,
               geom_state->radii,
               geom_state->depth_features,
               geom_state->color_features,
               geom_state->conic_opacity,
               mp_camera->pos(),
               mp_camera->camera_primitive(width, height),
               mp_camera->view_matrix(),
               mp_camera->proj_matrix())
               .dispatch(num_gaussians);

    // ---------------------------------
    // Deal with Keys
    // ---------------------------------
    cmdlist << luisa::compute::cuda::lcub::DeviceScan::InclusiveSum(
    geom_state->scan_temp_storage,
    geom_state->tiles_touched,
    geom_state->point_offsets, num_gaussians);

    int num_rendered = 0;
    cmdlist << geom_state->point_offsets.view(num_gaussians - 1, 1).copy_to(&num_rendered);

    stream << cmdlist.commit() << synchronize();
    if (num_rendered <= 0) { return; }

    LUISA_INFO("Forward Rendering: {} points rendered.", num_rendered);
    // allocate tiles
    tile_state->allocate(device, static_cast<size_t>(num_rendered));
    tile_state->clear(device, cmdlist, *mp_buffer_filler);
    // duplicate keys
    cmdlist << (*m_copy_with_keys_shader)(
               num_gaussians,
               geom_state->means_2d,
               geom_state->point_offsets,
               geom_state->radii,
               geom_state->depth_features,
               0, // camera depth idx
               tile_state->point_list_keys_unsorted,
               tile_state->point_list_unsorted,
               m_blocks, m_grids)
               .dispatch(num_gaussians);

    cmdlist << luisa::compute::cuda::lcub::DeviceRadixSort::SortPairs(
    tile_state->sort_temp_storage,
    tile_state->point_list_keys_unsorted,
    tile_state->point_list_keys,
    tile_state->point_list_unsorted,
    tile_state->point_list, num_rendered);

    img_state->clear(device, cmdlist, *mp_buffer_filler);
    // get range
    cmdlist << (*m_get_ranges_shader)(
               num_rendered,
               tile_state->point_list_keys,
               img_state->ranges)
               .dispatch(num_rendered);

    // ---------------------------------
    // Final Render
    // ---------------------------------
    cmdlist << (*m_forward_render_shader)(
               resolution,
               target_img_buffer,
               m_grids,
               img_state->ranges,
               tile_state->point_list,
               geom_state->means_2d,
               geom_state->color_features,
               geom_state->conic_opacity,
               img_state->n_contrib,
               img_state->accum_alpha)
               .dispatch(resolution);
    stream << cmdlist.commit() << synchronize();
}

} // namespace inno::render