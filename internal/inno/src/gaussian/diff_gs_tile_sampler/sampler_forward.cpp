/**
 * @file packages/gaussian/diff_gs_tile_sampler/sampler_forward.cpp
 * @author sailing-innocent
 * @date 2024-03-07
 * @brief Tile Based Sampler for Discretely Sampling a list of standard Gaussian
 */

#include "SailInno/gaussian/diff_gs_tile_sampler.h"
#include "SailInno/util/misc/bit_helper.h"

#include <luisa/dsl/sugar.h>
#include <luisa/backends/ext/cuda/lcub/device_scan.h>
#include <luisa/backends/ext/cuda/lcub/device_radix_sort.h>

using namespace luisa;
using namespace luisa::compute;

namespace sail::inno::gaussian {

void DiffGaussianTileSampler::forward_impl(
	Device& device,
	Stream& stream,
	// params
	int num_gaussians,
	int height, int width,
	// input
	BufferView<float> means_2d,		   // P * 2
	BufferView<float> covs_2d,		   // P * 3
	BufferView<float> depth_features,  // P * 1
	BufferView<float> opacity_features,// P * 1
	BufferView<float> color_features,  // P * 3
	// output
	BufferView<float> target_img_buffer) {
	m_grids = luisa::make_uint2(
		(unsigned int)((width + m_blocks.x - 1u) / m_blocks.x),
		(unsigned int)((height + m_blocks.y - 1u) / m_blocks.y));

	if (m_num_gaussians != num_gaussians) {
		// if num_gaussians changed, reallocate buffer
		geom_state->allocate(device, num_gaussians);
		m_num_gaussians = num_gaussians;
	}
	if ((m_resolution.x != width) || (m_resolution.y != height)) {
		// resolution changed, reallocate image buffer
		img_state->allocate(device, width * height);
		m_resolution = luisa::make_uint2(width, height);
	}
	CommandList cmdlist;
	cmdlist << (*m_forward_tile_split_shader)(
				   num_gaussians,
				   m_resolution,
				   m_grids,
				   // input
				   means_2d,
				   covs_2d,
				   // output
				   geom_state->conic,
				   geom_state->means_2d_res,
				   geom_state->tiles_touched,
				   geom_state->radii)
				   .dispatch(num_gaussians);

	// copy color_features
	cmdlist << geom_state->color_features.copy_from(color_features);
	// copy opacity features
	cmdlist << geom_state->opacity_features.copy_from(opacity_features);

	cmdlist << luisa::compute::cuda::lcub::DeviceScan::InclusiveSum(
		geom_state->scan_temp_storage,
		geom_state->tiles_touched,
		geom_state->point_offsets, num_gaussians);

	int num_rendered;
	cmdlist << geom_state->point_offsets.view(num_gaussians - 1, 1).copy_to(&num_rendered);
	stream << cmdlist.commit() << synchronize();

	// LUISA_INFO("num_rendered: {}", num_rendered);
	if (num_rendered <= 0) { return; }
	tile_state->allocate(device, static_cast<size_t>(num_rendered));
	tile_state->clear(device, cmdlist, *mp_buffer_filler);
	// duplicate keys
	cmdlist << (*m_copy_with_keys_shader)(
				   num_gaussians,
				   geom_state->means_2d_res,
				   geom_state->point_offsets,
				   geom_state->radii,
				   depth_features,
				   tile_state->point_list_keys_unsorted,
				   tile_state->point_list_unsorted,
				   m_blocks, m_grids)
				   .dispatch(num_gaussians);

	int bit = inno::util::get_higher_msb(m_blocks.x * m_blocks.y);
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

	cmdlist << (*m_forward_render_shader)(
				   m_resolution,
				   target_img_buffer,
				   m_grids,
				   img_state->ranges,
				   tile_state->point_list,
				   geom_state->means_2d_res,
				   geom_state->conic,
				   geom_state->opacity_features,
				   geom_state->color_features,
				   // save for backward
				   img_state->n_contrib,
				   img_state->accum_alpha)
				   .dispatch(m_resolution);
	stream << cmdlist.commit() << synchronize();
}
}// namespace sail::inno::gaussian