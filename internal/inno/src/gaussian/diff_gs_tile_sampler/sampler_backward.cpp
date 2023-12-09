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
	// input
	BufferView<float> dL_dpix,
	// params
	BufferView<float> covs_2d,
	BufferView<float> opacity_features,
	BufferView<float> color_features,
	// output
	BufferView<float> dL_d_means_2d,
	BufferView<float> dL_d_covs_2d,
	BufferView<float> dL_d_opacity_features,
	BufferView<float> dL_d_color_features) {

	cmdlist << (*m_backward_render_shader)(
				   m_resolution,
				   //input
				   dL_dpix,
				   // params
				   m_grids,
				   img_state->ranges,
				   tile_state->point_list,
				   geom_state->means_2d_res,
				   opacity_features,
				   color_features,
				   geom_state->conic,
				   img_state->n_contrib,
				   img_state->accum_alpha,
				   // output
				   dL_d_means_2d,
				   geom_state->dL_d_conic,
				   dL_d_opacity_features,
				   dL_d_color_features)
				   .dispatch(m_resolution);

	cmdlist << (*m_backward_tile_split_shader)(
				   m_num_gaussians,
				   geom_state->dL_d_conic,
				   m_resolution,
				   covs_2d,
				   dL_d_covs_2d)
				   .dispatch(m_num_gaussians);
}

}// namespace sail::inno::gaussian