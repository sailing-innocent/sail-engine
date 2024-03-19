/**
 * @file packages/gaussian/diff_gs_tile_sampler.cpp
 * @author sailing-innocent
 * @date 2024-03-07
 * @brief Tile Based Sampler for Discretely Sampling a list of standard Gaussian
 */

#include "SailInno/gaussian/diff_gs_tile_sampler.h"

#include <luisa/dsl/sugar.h>
#include <luisa/backends/ext/cuda/lcub/device_scan.h>
#include <luisa/backends/ext/cuda/lcub/device_radix_sort.h>

using namespace luisa;
using namespace luisa::compute;

// API
namespace sail::inno::gaussian {

void DiffGaussianTileSampler::create(Device& device) noexcept {
	mp_buffer_filler = luisa::make_shared<BufferFiller>();
	geom_state = luisa::make_unique<GeometryState>();
	tile_state = luisa::make_unique<TileState>();
	img_state = luisa::make_unique<ImageState>();
	compile(device);
}

void DiffGaussianTileSampler::create(Device& device, S<BufferFiller> buffer_filler) noexcept {
	mp_buffer_filler = buffer_filler;
	geom_state = luisa::make_unique<GeometryState>();
	tile_state = luisa::make_unique<TileState>();
	img_state = luisa::make_unique<ImageState>();
	compile(device);
}

void DiffGaussianTileSampler::GeometryState::allocate(Device& device, size_t size) {
	if (size == 0) { return; }
	means_2d_res = device.create_buffer<float>(size * 2);
	radii = device.create_buffer<int>(size);
	color_features = device.create_buffer<float>(size * 3);
	opacity_features = device.create_buffer<float>(size);
	conic = device.create_buffer<float>(size * 3);
	tiles_touched = device.create_buffer<uint>(size);
	point_offsets = device.create_buffer<uint>(size);
	dL_d_conic = device.create_buffer<float>(size * 3);
	dL_d_means_2d = device.create_buffer<float>(size * 4);
	// allocate scan temp storage
	luisa::compute::cuda::lcub::DeviceScan::InclusiveSum(scan_temp_storage_size, tiles_touched, point_offsets, size);
	scan_temp_storage = device.create_buffer<int>(scan_temp_storage_size);
}

void DiffGaussianTileSampler::GeometryState::clear(Device& device, CommandList& cmdlist, BufferFiller& filler) {
	cmdlist << filler.fill(device, means_2d_res, 0.0f);
	cmdlist << filler.fill(device, radii, 0);
	cmdlist << filler.fill(device, color_features, 0.0f);
	cmdlist << filler.fill(device, opacity_features, 0.0f);
	cmdlist << filler.fill(device, conic, 0.0f);
	cmdlist << filler.fill(device, tiles_touched, 0u);
	cmdlist << filler.fill(device, point_offsets, 0u);
	cmdlist << filler.fill(device, dL_d_conic, 0.0f);
	cmdlist << filler.fill(device, dL_d_means_2d, 0.0f);
}

void DiffGaussianTileSampler::TileState::allocate(Device& device, size_t size) {
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

void DiffGaussianTileSampler::TileState::clear(Device& device, CommandList& cmdlist, BufferFiller& filler) {
	cmdlist << filler.fill(device, point_list_keys_unsorted, 0ull);
	cmdlist << filler.fill(device, point_list_unsorted, 0u);
	cmdlist << filler.fill(device, point_list_keys, 0ull);
	cmdlist << filler.fill(device, point_list, 0u);
	if (use_shade) {
		cmdlist << filler.fill(device, point_list_keys_shade, 0ull);
	}
}

void DiffGaussianTileSampler::ImageState::allocate(Device& device, size_t size) {
	if (size == 0) { return; }
	ranges = device.create_buffer<uint>(size * 2);
	n_contrib = device.create_buffer<uint>(size);
	accum_alpha = device.create_buffer<float>(size);
}

void DiffGaussianTileSampler::ImageState::clear(Device& device, CommandList& cmdlist, BufferFiller& filler) {
	cmdlist << filler.fill(device, ranges, 0u);
	cmdlist << filler.fill(device, n_contrib, 0u);
	cmdlist << filler.fill(device, accum_alpha, 0.f);
}

}// namespace sail::inno::gaussian