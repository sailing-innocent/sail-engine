/**
 * @file app/GS/diff_gs_tile_sampler_app.cpp
 * @author sailing-innocent
 * @date 2024-03-08
 * @brief The Diff GS Tile Sampler
 */

#include "SailInno/app/diff_gs_tile_sampler_app.h"
#include "SailInno/gaussian/diff_gs_tile_sampler.h"
#include <cstdint>

using namespace luisa;
using namespace luisa::compute;

namespace sail::inno::app {

void DiffGSTileSamplerApp::create(luisa::string& cwd, luisa::string& device_name) {
	LUISA_INFO("DiffGSTillSamplerApp::create");
	Context context{cwd.c_str()};
	mp_device = luisa::make_shared<Device>(context.create_device(device_name.c_str()));
	mp_stream = luisa::make_shared<Stream>(mp_device->create_stream());
	mp_sampler = luisa::make_unique<gaussian::DiffGaussianTileSampler>();
	mp_sampler->create(*mp_device);
}

void DiffGSTileSamplerApp::forward(
	// params
	int num_gaussians, int width, int height,
	// input
	int64_t means_2d, int64_t covs_2d, int64_t depth_features, int64_t color_features,
	// output
	int64_t target_img_buffer) {
	LUISA_INFO("DiffGSTileSamplerApp::forward");
	// input
	Buffer<float> means_2d_buf = mp_device->import_external_buffer<float>((void*)means_2d, num_gaussians * 2);
	Buffer<float> covs_2d_buf = mp_device->import_external_buffer<float>((void*)covs_2d, num_gaussians * 3);
	Buffer<float> depth_features_buf = mp_device->import_external_buffer<float>((void*)depth_features, num_gaussians * 1);
	Buffer<float> color_features_buf = mp_device->import_external_buffer<float>((void*)color_features, num_gaussians * 4);

	// output
	Buffer<float> target_img_buf = mp_device->import_external_buffer<float>((void*)target_img_buffer, width * height * 3);

	mp_sampler->forward_impl(
		*mp_device, *mp_stream,
		num_gaussians, width, height,
		means_2d_buf, covs_2d_buf, depth_features_buf, color_features_buf,
		target_img_buf);
}

void DiffGSTileSamplerApp::backward() {
	LUISA_INFO("DiffGSTileSamplerApp::backward");
	// mp_sampler->backward_impl(*mp_device, *mp_stream);
}

}// namespace sail::inno::app
