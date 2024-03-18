/**
 * @file packages/gaussian/diff_gs_tile_sampler/sampler_forward.cpp
 * @author sailing-innocent
 * @date 2024-03-07
 * @brief Tile Based Sampler for Discretely Sampling a list of standard Gaussian
 */

#include "SailInno/gaussian/diff_gs_tile_sampler.h"

namespace sail::inno::gaussian {
void DiffGaussianTileSampler::backward_impl(
	Device& device,
	CommandList& cmdlist,
	// params all saved
	// input
	BufferView<float> dL_dpix,
	// output
	BufferView<float> dL_d_means_2d,
	BufferView<float> dL_d_covs_2d,
	BufferView<float> dL_d_color_features) {

	// auto dL_d_means_2d_res = device.create_buffer<float>(m_num_gaussians * 2);
	// auto dL_d_conic = device.create_buffer<float>(m_num_gaussians * 3);

	LUISA_INFO("DiffGSTileSampler::backward_impl with {}, {}, {}", m_num_gaussians, m_resolution.x, m_resolution.y);

	// clear grad
	cmdlist << mp_buffer_filler->fill(device, dL_d_means_2d, 0.0f);
	cmdlist << mp_buffer_filler->fill(device, dL_d_covs_2d, 0.0f);
	cmdlist << mp_buffer_filler->fill(device, dL_d_color_features, 0.0f);
	// clear temp grad
	// cmdlist << mp_buffer_filler->fill(device, dL_d_means_2d_res, 0.0f);
	// cmdlist << mp_buffer_filler->fill(device, dL_d_conic, 0.0f);

	cmdlist << (*m_backward_render_shader)(
				   m_resolution,
				   //input
				   dL_dpix,
				   // params
				   m_grids,
				   img_state->ranges,
				   tile_state->point_list,
				   geom_state->means_2d_res,
				   geom_state->color_features,
				   geom_state->conic,
				   img_state->n_contrib,
				   img_state->accum_alpha,
				   // output
				   dL_d_covs_2d,
				   dL_d_color_features)
				   .dispatch(m_resolution);
}

}// namespace sail::inno::gaussian