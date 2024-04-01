/**
 * @file package/diff_render/gaussian_splatter_impl.cpp
 * @author sailing-innocent
 * @date 2023-12-27
 * @brief The Gaussian Splatter Basic Implement
 */

#include "SailInno/diff_render/reprod_gs_splatter.h"
#include <luisa/dsl/sugar.h>
#include <luisa/backends/ext/cuda/lcub/device_scan.h>
#include <luisa/backends/ext/cuda/lcub/device_radix_sort.h>

#include "SailINno/util/math/gaussian.h"

using namespace luisa;
using namespace luisa::compute;

namespace sail::inno::render {

void ReprodGS::create(Device& device) noexcept {
	mp_buffer_filler = luisa::make_unique<BufferFiller>();
	geom_state = luisa::make_unique<GeometryState>();
	tile_state = luisa::make_unique<TileState>();
	img_state = luisa::make_unique<ImageState>();
	compile(device);
}

}// namespace sail::inno::render

// State Management

namespace sail::inno::render {

void ReprodGS::GeometryState::allocate(Device& device, size_t size) {
	if (size == 0) { return; }
	means_2d = device.create_buffer<float>(size * 2);
	depth_features = device.create_buffer<float>(size);
	opacity_features = device.create_buffer<float>(size);
	color_features = device.create_buffer<float>(3 * size);
	conic = device.create_buffer<float>(size * 3);

	tiles_touched = device.create_buffer<uint>(size);
	point_offsets = device.create_buffer<uint>(size);
	// allocate scan temp storage
	luisa::compute::cuda::lcub::DeviceScan::InclusiveSum(scan_temp_storage_size, tiles_touched, point_offsets, size);
	scan_temp_storage = device.create_buffer<int>(scan_temp_storage_size);
}

void ReprodGS::GeometryState::clear(Device& device, CommandList& cmdlist, BufferFiller& filler) {
	// only necessary to clear
	cmdlist << filler.fill(device, means_2d, 0.0f);
	cmdlist << filler.fill(device, depth_features, 0.0f);
	cmdlist << filler.fill(device, color_features, 0.0f);
	cmdlist << filler.fill(device, tiles_touched, 0u);
	cmdlist << filler.fill(device, point_offsets, 0u);
}

void ReprodGS::TileState::allocate(Device& device, size_t size) {
	point_list_keys = device.create_buffer<ulong>(size);
	point_list_keys_unsorted = device.create_buffer<ulong>(size);
	point_list_unsorted = device.create_buffer<uint>(size);
	point_list = device.create_buffer<uint>(size);
	// allocate sort temp storage
	luisa::compute::cuda::lcub::DeviceRadixSort::SortPairs(sort_temp_storage_size, point_list_keys_unsorted, point_list_keys, point_list_unsorted, point_list, size);
	sort_temp_storage = device.create_buffer<int>(sort_temp_storage_size);

	if (use_shade) {
		point_list_keys_shade = device.create_buffer<ulong>(size);
	}
}

void ReprodGS::TileState::clear(Device& device, CommandList& cmdlist, BufferFiller& filler) {
	cmdlist << filler.fill(device, point_list_keys_unsorted, 0ull);
	cmdlist << filler.fill(device, point_list_unsorted, 0u);
	cmdlist << filler.fill(device, point_list_keys, 0ull);
	cmdlist << filler.fill(device, point_list, 0u);
	if (use_shade) {
		cmdlist << filler.fill(device, point_list_keys_shade, 0ull);
	}
}

void ReprodGS::ImageState::allocate(Device& device, size_t size) {
	if (size == 0) { return; }
	ranges = device.create_buffer<uint>(size * 2);
	n_contrib = device.create_buffer<uint>(size);
	accum_alpha = device.create_buffer<float>(size);
}

void ReprodGS::ImageState::clear(Device& device, CommandList& cmdlist, BufferFiller& filler) {
	cmdlist << filler.fill(device, ranges, 0u);
	cmdlist << filler.fill(device, n_contrib, 0u);
	cmdlist << filler.fill(device, accum_alpha, 0.f);
}

}// namespace sail::inno::render

// Core

namespace sail::inno::render {

void ReprodGS::compile(Device& device) noexcept {
	compile_callables(device);
	compile_forward_preprocess_shader(device);
	compile_backward_preprocess_shader(device);
	compile_copy_with_keys_shader(device);
	compile_get_ranges_shader(device);
	compile_forward_render_shader(device);
	compile_backward_render_shader(device);
}

}// namespace sail::inno::render