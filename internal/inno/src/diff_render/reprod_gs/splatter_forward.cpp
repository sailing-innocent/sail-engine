/**
 * @file package/diff_render/gaussian_splatter_impl.cpp
 * @author sailing-innocent
 * @date 2023/12/28
 * @brief The Gaussian Splatter Basic Forward Pass Implement
 */

#include "SailInno/diff_render/reprod_gs_splatter.h"
#include <luisa/backends/ext/cuda/lcub/device_scan.h>
#include <luisa/backends/ext/cuda/lcub/device_radix_sort.h>
#include <luisa/dsl/sugar.h>
#include "SailInno/util/misc/bit_helper.h"

using namespace luisa;
using namespace luisa::compute;

// API

namespace sail::inno::render {

void ReprodGS::gaussian_proj_impl(
	Device& device,
	CommandList& cmdlist,
	int num_gaussians, int sh_deg, int max_sh_deg,
	float scale_modifier,
	// input
	BufferView<float> xyz_buffer,
	BufferView<float> feature_buffer,// for color
	BufferView<float> scale_buffer,
	BufferView<float> rotq_buffer,
	Camera& cam) {
	// clear geometry state
	geom_state->clear(device, cmdlist, *mp_buffer_filler);
	// calculate camera primitive
	cmdlist << (*m_forward_preprocess_shader)(
				   num_gaussians, sh_deg, max_sh_deg,
				   // input
				   xyz_buffer,
				   feature_buffer,
				   scale_buffer,
				   rotq_buffer,
				   // params
				   scale_modifier,
				   // output
				   geom_state->means_2d,
				   geom_state->depth_features,
				   geom_state->color_features,
				   geom_state->covs_2d,
				   // camera
				   mp_camera->pos(),
				   mp_camera->camera_primitive(m_resolution.x, m_resolution.y),
				   mp_camera->view_matrix(),
				   mp_camera->proj_matrix())
				   .dispatch(num_gaussians);
}

void ReprodGS::forward_impl(
	Device& device,
	Stream& stream,
	int height, int width,
	// output
	BufferView<float> target_img_buffer,// hwc // output
	BufferView<int> radii,
	// params
	int num_gaussians,
	int sh_deg,
	int max_sh_deg,
	// input
	BufferView<float> xyz_buffer,
	BufferView<float> feature_buffer,// for color
	BufferView<float> opacity_buffer,
	BufferView<float> scale_buffer,
	BufferView<float> rotq_buffer,
	float scale_modifier,
	Camera& cam) {
	m_grids = luisa::make_uint2(
		(unsigned int)((width + m_blocks.x - 1u) / m_blocks.x),
		(unsigned int)((height + m_blocks.y - 1u) / m_blocks.y));
	mp_camera = luisa::make_shared<Camera>(cam);
	// save for backward
	m_sh_deg = sh_deg;
	m_max_sh_deg = max_sh_deg;
	m_scale_modifier = scale_modifier;
	if (m_num_gaussians != num_gaussians) {
		// if num_gaussians changed, reallocate buffer
		geom_state->allocate(device, static_cast<size_t>(num_gaussians));
		m_num_gaussians = num_gaussians;
	}
	if ((m_resolution.x != width) || (m_resolution.y != height)) {
		// resolution changed, reallocate image buffer
		img_state->allocate(device, width * height);
		m_resolution = luisa::make_uint2(width, height);
	}

	CommandList cmdlist;

	gaussian_proj_impl(
		device, cmdlist,
		num_gaussians, sh_deg, max_sh_deg,
		scale_modifier,
		xyz_buffer, feature_buffer,
		scale_buffer, rotq_buffer, cam);

	cmdlist << geom_state->opacity_features.copy_from(opacity_buffer);
	cmdlist << (*m_forward_tile_split_shader)(
				   num_gaussians,
				   m_resolution,
				   m_grids,
				   // input
				   geom_state->means_2d,
				   geom_state->covs_2d,
				   // output
				   geom_state->conic,
				   geom_state->means_2d_res,
				   geom_state->tiles_touched,
				   radii)
				   .dispatch(num_gaussians);

	// device scan
	cmdlist << luisa::compute::cuda::lcub::DeviceScan::InclusiveSum(
		geom_state->scan_temp_storage,
		geom_state->tiles_touched,
		geom_state->point_offsets, num_gaussians);

	int num_rendered;
	cmdlist << geom_state->point_offsets.view(num_gaussians - 1, 1).copy_to(&num_rendered);
	stream << cmdlist.commit() << synchronize();

	if (num_rendered <= 0) { return; }
	tile_state->allocate(device, static_cast<size_t>(num_rendered));
	tile_state->clear(device, cmdlist, *mp_buffer_filler);
	// duplicate keys
	cmdlist << (*m_copy_with_keys_shader)(
				   num_gaussians,
				   geom_state->means_2d_res,
				   geom_state->point_offsets,
				   radii,
				   geom_state->depth_features,
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

}// namespace sail::inno::render