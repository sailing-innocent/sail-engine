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
	BufferView<float> dL_dmeans_2d,
	BufferView<float> dL_dcovs_2d,
	BufferView<float> dL_dcolor_features) {
}

}// namespace sail::inno::gaussian