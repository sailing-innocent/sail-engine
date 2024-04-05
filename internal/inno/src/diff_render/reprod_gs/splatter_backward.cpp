/**
 * @file package/diff_render/gaussian_splatter_impl.cpp
 * @author sailing-innocent
 * @date 2023/12/28
 * @brief The Gaussian Splatter Basic Backward Pass Implement
 */

#include "SailInno/diff_render/reprod_gs_splatter.h"
#include <luisa/dsl/sugar.h>

using namespace luisa;
using namespace luisa::compute;

// API

namespace sail::inno::render {

void ReprodGS::backward_impl(
	Device& device,
	Stream& stream,
	// input
	BufferView<float> dL_d_pix,
	// output
	BufferView<float> dL_d_xyz,
	BufferView<float> dL_d_feature,// (feat_dim + 1) * (feat_dim + 1) * 3
	BufferView<float> dL_d_opacity,
	BufferView<float> dL_d_scale,
	BufferView<float> dL_d_rotq,
	BufferView<float> dL_d_means_2d,
	// params
	BufferView<float> target_img_buffer,// hwc
	BufferView<float> xyz_buffer,
	BufferView<float> feature_buffer,// for color
	BufferView<float> opacity_buffer,
	BufferView<float> scale_buffer,
	BufferView<float> rotq_buffer) {
	// LUISA_INFO("ReprodGS::backward_impl");

	CommandList cmdlist;

	mp_buffer_filler->fill(device, geom_state->dL_d_color_feature, 0.0f);
	mp_buffer_filler->fill(device, geom_state->dL_d_conic, 0.0f);

	cmdlist << (*m_backward_render_shader)(
				   // input
				   dL_d_pix,
				   // output
				   dL_d_means_2d,
				   geom_state->dL_d_conic,
				   geom_state->dL_d_color_feature,
				   dL_d_opacity,
				   // params
				   m_resolution,
				   m_grids,
				   target_img_buffer,
				   img_state->ranges,
				   tile_state->point_list,
				   geom_state->means_2d,
				   geom_state->conic,
				   geom_state->opacity_features,
				   geom_state->color_features,
				   img_state->n_contrib,
				   img_state->accum_alpha)
				   .dispatch(m_resolution);

	// LUISA_INFO("backward preprocess with {} ", m_num_gaussians);
	cmdlist << (*m_backward_preprocess_shader)(
				   // input
				   dL_d_means_2d,
				   geom_state->dL_d_conic,
				   geom_state->dL_d_color_feature,
				   // output
				   dL_d_xyz,
				   dL_d_feature,
				   dL_d_scale,
				   dL_d_rotq,
				   // params
				   m_num_gaussians, m_sh_deg, m_max_sh_deg,
				   m_resolution, m_grids,
				   xyz_buffer,
				   feature_buffer,
				   scale_buffer,
				   rotq_buffer,
				   geom_state->opacity_features,
				   geom_state->color_features,
				   geom_state->conic,
				   // camera
				   mp_camera->pos(),
				   mp_camera->camera_primitive(m_resolution.x, m_resolution.y),
				   mp_camera->view_matrix())
				   .dispatch(m_num_gaussians);

	stream << cmdlist.commit() << synchronize();
}

}// namespace sail::inno::render