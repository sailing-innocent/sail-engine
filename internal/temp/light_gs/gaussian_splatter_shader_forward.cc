/**
 * @file package/diff_render/light_gs/gaussian_splatter_shader_forward.cpp
 * @author sailing-innocent
 * @date 2023-01-03
 * @brief The Light Gaussian Splatter Basic Implement
 */

#include "gaussian_splatter.h"
#include <luisa/dsl/sugar.h>

using namespace luisa;
using namespace luisa::compute;

namespace inno::render
{

void LightGaussianSplatter::compile_forward_shade_shader(Device& device) noexcept
{
    // Captures:
    //  m_blocks
    lazy_compile(device, m_mix_color_shader, [&](Int P, Int D, Int M, BufferVar<float> means_3d, Float3 cam_pos, BufferVar<float> feat_buffer, BufferVar<float> d_color, BufferVar<float4> color_features) {
        set_block_size(m_blocks.x * m_blocks.y);
        auto idx = dispatch_id().x;
        $if(idx >= static_cast<$uint>(P)) { return; };

        auto color  = (*mp_compute_color_from_sh)(static_cast<Int>(idx), D, M, means_3d, cam_pos, feat_buffer);
        auto albedo = make_float4(color, 1.0f);

        auto   light = make_float4(d_color.read(idx * 3 + 0), d_color.read(idx * 3 + 1), d_color.read(idx * 3 + 2), 1.0f);
        Float4 mixed = make_float4(0.0f);
        $for(k, 3ull)
        {
            // mixed[k] = albedo[k] * light[k];
            // mixed[k] = albedo[k];
            mixed[k] = light[k];
        };
        // float s = 1.0f;
        // auto mixed = s * albedo + (1.0f - s) * light;
        color_features.write(idx, mixed);
    });

    // Captures
    //  m_blocks
    lazy_compile(device, m_forward_shade_shader, [&](UInt2 resolution, UInt2 grids, BufferVar<uint> ranges, BufferVar<uint> point_list, BufferVar<float2> means_2d, BufferVar<float> d_color, BufferVar<float4> conic_opacity) {
        set_block_size(m_blocks);
        auto xy         = dispatch_id().xy();
        auto thread_idx = thread_id().x + thread_id().y * block_size().x;
        Bool inside     = Bool(xy.x < resolution.x) & Bool(xy.y < resolution.y);
        $if(!inside) { return; };
        Bool  done         = !inside;
        auto  tile_xy      = block_id();
        Float width_coeff  = 1.0f / (Float)resolution.x;
        Float height_coeff = 1.0f / (Float)resolution.y;
        auto  pix_id       = xy.x + resolution.x * xy.y;
        auto  pix_f        = Float2(
        static_cast<Float>(xy.x),
        static_cast<Float>(xy.y));
        Int range_start = (Int)ranges.read(2 * (tile_xy.x + tile_xy.y * grids.x) + 0u);
        Int range_end   = (Int)ranges.read(2 * (tile_xy.x + tile_xy.y * grids.x) + 1u);

        Float3 light_color = make_float3(10.0f, 2.0f, 2.0f);
        // make rounds
        const Int round_step = Int(m_shared_mem_size);
        const Int rounds     = ((range_end - range_start + round_step - 1) / round_step);
        Int       todo       = range_end - range_start;
        $if(todo <= 0) { return; };

        Shared<int>*    collected_ids           = new Shared<int>(m_shared_mem_size);
        Shared<float2>* collected_means         = new Shared<float2>(m_shared_mem_size);
        Shared<float4>* collected_conic_opacity = new Shared<float4>(m_shared_mem_size);

        Float T = 1.0f;
        $for(i, rounds)
        {
            Int progress = i * round_step + thread_idx;
            // fetch idx and params to calc weight
            $if(progress + range_start < range_end)
            {
                UInt coll_id = point_list.read(progress + range_start);
                collected_ids->write(thread_idx, coll_id);
                collected_means->write(thread_idx, means_2d.read(coll_id));
                collected_conic_opacity->write(thread_idx, conic_opacity.read(coll_id));
            };
            sync_block();
            // iterate over the current batch
            $for(j, min(round_step, todo))
            {
                $if(done) { $break; };
                Float2 xy    = collected_means->read(j);
                Float2 d     = pix_f - xy;
                Float4 con_o = collected_conic_opacity->read(j);
                Float  power = -0.5f * (con_o.x * d.x * d.x + con_o.z * d.y * d.y) - con_o.y * d.x * d.y;
                $if(power > 0.0f) { $continue; };
                Float weight  = min(0.99f, exp(power));
                Float alpha   = min(0.99f, con_o.w * weight);
                auto  coll_id = collected_ids->read(j);
                // local light color
                auto color = T * alpha * light_color;

                $for(k, 3)
                {
                    d_color.atomic(3 * coll_id + k).fetch_add(1000.0f * width_coeff * height_coeff * weight * color[k]);
                };
                Float test_T = T * (1.0f - alpha);
            };
            todo = todo - round_step;
        };
    });
}

void LightGaussianSplatter::compile_forward_preprocess_shader(Device& device) noexcept
{
    lazy_compile(device, m_forward_preprocess_shader,
                 [&](
                 Int P, Int D, Int M,
                 BufferVar<float>  means_3d,
                 BufferVar<float>  feat_buffer,
                 BufferVar<float>  opacity_buffer,
                 BufferVar<float>  scale_buffer,
                 BufferVar<float>  rotq_buffer,
                 Float             scale_modifier,
                 UInt2             resolution,
                 UInt2             grids,
                 BufferVar<float2> mean_2d,
                 BufferVar<uint>   tiles_touched,
                 BufferVar<int>    radii,
                 BufferVar<float4> depth_features,
                 BufferVar<float4> color_features,
                 BufferVar<float4> conic_opacity,
                 Float3            cam_pos,
                 Float4            camera_primitive,
                 Float4x4          view_matrix,
                 Float4x4          proj_matrix) {
                     set_block_size(m_blocks.x * m_blocks.y);
                     auto idx = dispatch_id().x;
                     $if(idx >= static_cast<$uint>(P)) { return; };

                     auto mean_3d = make_float3(means_3d.read(3 * idx + 0), means_3d.read(3 * idx + 1), means_3d.read(3 * idx + 2));

                     // auto color = (*mp_compute_color_from_sh)(static_cast<Int>(idx), D, M, means_3d, cam_pos, feat_buffer);
                     // color_features.write(idx, make_float4(color, 1.0f));

                     Float4 p_hom      = make_float4(mean_3d, 1.0f);
                     Float4 p_view_hom = view_matrix * p_hom;
                     Float4 p_proj_hom = proj_matrix * p_view_hom;
                     Float3 p_view     = p_view_hom.xyz();
                     Float  p_w        = 1.0f / (p_proj_hom.w + 1e-6f);
                     Float3 p_proj     = p_proj_hom.xyz() * p_w;
                     // near culling method
                     $if(p_view.z <= 0.2f) { return; };
                     // calculate 3d covariance
                     Float3   s      = make_float3(scale_buffer.read(3 * idx + 0), scale_buffer.read(3 * idx + 1), scale_buffer.read(3 * idx + 2));
                     Float4   rotq   = make_float4(rotq_buffer.read(4 * idx + 0), rotq_buffer.read(4 * idx + 1), rotq_buffer.read(4 * idx + 2), rotq_buffer.read(4 * idx + 3));
                     Float3x3 cov_3d = (*mp_compute_cov_3d)(s, scale_modifier, rotq);
                     // cov_3d = make_float3x3(1.0f) * 0.0001f;

                     // calculate projected covariance 2d
                     Float3 cov_2d = (*mp_compute_cov_2d)(p_view_hom, camera_primitive, cov_3d, view_matrix);

                     // invert covariance
                     // det(M) = M[0][0] * M[1][1] - M[0][1] * M[1][0]
                     Float  det     = cov_2d.x * cov_2d.z - cov_2d.y * cov_2d.y;
                     Float  inv_det = 1.0f / det;
                     Float3 conic   = inv_det * make_float3(cov_2d.z, -cov_2d.y, cov_2d.x);

                     auto point_image_ndc = make_float2(p_proj.x, p_proj.y);

                     Float4 depth_feature = make_float4(0.0f);
                     depth_feature.x      = abs(p_view.z);
                     depth_features.write(idx, depth_feature);
                     // ndc to pixel coordinate
                     auto point_image = make_float2((*mp_ndc2pix)(point_image_ndc.x, resolution.x), (*mp_ndc2pix)(point_image_ndc.y, resolution.y));
                     // extent on screen space
                     Float mid       = 0.5f * (cov_2d.x + cov_2d.z);
                     Float lambda1   = mid + sqrt(max(0.1f, mid * mid - det));
                     Float lambda2   = mid - sqrt(max(0.1f, mid * mid - det));
                     Int   my_radius = ceil(3.0f * sqrt(max(lambda1, lambda2)));
                     UInt2 rect_min, rect_max;
                     (*mp_get_rect)(point_image, my_radius, rect_min, rect_max, m_blocks, grids);
                     // write means_2d
                     mean_2d.write(idx, point_image);
                     radii.write(idx, my_radius);
                     auto opacity = opacity_buffer.read(idx);
                     conic_opacity.write(idx, make_float4(conic, opacity));
                     tiles_touched.write(idx, (rect_max.x - rect_min.x) * (rect_max.y - rect_min.y));
                 });
}

void LightGaussianSplatter::compile_forward_render_shader(Device& device) noexcept
{
    lazy_compile(device, m_forward_render_shader,
                 [&](
                 UInt2             resolution,
                 BufferVar<float>  target_img,
                 UInt2             grids,
                 BufferVar<uint>   ranges,
                 BufferVar<uint>   point_list,
                 BufferVar<float2> means_2d,
                 BufferVar<float4> features,
                 BufferVar<float4> conic_opacity,
                 BufferVar<uint>   n_contrib,
                 BufferVar<float>  accum_alpha) {
                     set_block_size(m_blocks);
                     auto xy         = dispatch_id().xy();
                     auto w          = resolution.x;
                     auto h          = resolution.y;
                     auto thread_idx = thread_id().x + thread_id().y * block_size().x;
                     Bool inside     = Bool(xy.x < resolution.x) & Bool(xy.y < resolution.y);
                     Bool done       = !inside;
                     auto tile_xy    = block_id();

                     auto pix_id = xy.x + resolution.x * xy.y;
                     auto pix_f  = Float2(
                     static_cast<Float>(xy.x),
                     static_cast<Float>(xy.y));

                     Int range_start = (Int)ranges.read(2 * (tile_xy.x + tile_xy.y * grids.x) + 0u);
                     Int range_end   = (Int)ranges.read(2 * (tile_xy.x + tile_xy.y * grids.x) + 1u);

                     // background color
                     Float3 color = make_float3(1.0f, 1.0f, 1.0f);

                     // make rounds
                     const Int round_step = Int(m_shared_mem_size);
                     const Int rounds     = ((range_end - range_start + round_step - 1) / round_step);
                     Int       todo       = range_end - range_start;

                     Shared<int>*    collected_ids           = new Shared<int>(m_shared_mem_size);
                     Shared<float2>* collected_means         = new Shared<float2>(m_shared_mem_size);
                     Shared<float4>* collected_conic_opacity = new Shared<float4>(m_shared_mem_size);
                     Shared<float3>* collected_colors        = new Shared<float3>(m_shared_mem_size);

                     Float  T = 1.0f;
                     Float3 C = make_float3(0.0f, 0.0f, 0.0f);

                     $for(i, rounds)
                     {
                         // require __syncthreads_count(done) to accelerate
                         Int progress = i * round_step + thread_idx;
                         $if(progress + range_start < range_end)
                         {
                             Int coll_id = point_list.read(progress + range_start);
                             collected_ids->write(thread_idx, coll_id);
                             collected_means->write(thread_idx, means_2d.read(coll_id));
                             collected_conic_opacity->write(thread_idx, conic_opacity.read(coll_id));
                             collected_colors->write(thread_idx, features.read(coll_id).xyz());
                         };
                         sync_block();

                         // iterate over the current batch

                         $for(j, min(round_step, todo))
                         {
                             $if(done) { $break; };
                             Float2 mean  = collected_means->read(j);
                             Float2 d     = mean - pix_f;
                             Float4 con_o = collected_conic_opacity->read(j);
                             Float  power = -0.5f * (con_o.x * d.x * d.x + con_o.z * d.y * d.y) - con_o.y * d.x * d.y;
                             $if(power > 0.0f) { $continue; };
                             Float alpha = min(0.99f, con_o.w * exp(power));
                             $if(alpha < 1.0f / 255.0f) { $continue; };
                             Float test_T = T * (1.0f - alpha);
                             $if(test_T < 0.0001f)
                             {
                                 done = true;
                                 $continue;
                             };
                             C = C + T * alpha * collected_colors->read(j);
                             T = test_T;
                         };

                         todo = todo - round_step;
                     };
                     color = color * T + C;
                     $if(inside)
                     {
                         // todo: collect final T and last contributor
                         $for(i, 0, 3)
                         {
                             target_img.write(pix_id + i * h * w, min(1.0f, color[i]));
                         };
                     };
                 });
}

} // namespace inno::render
