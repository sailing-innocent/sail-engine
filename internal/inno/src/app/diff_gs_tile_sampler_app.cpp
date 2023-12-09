/**
 * @file app/GS/diff_gs_tile_sampler_app.cpp
 * @author sailing-innocent
 * @date 2024-03-08
 * @brief The Diff GS Tile Sampler
 */

#include "SailInno/app/diff_gs_tile_sampler_app.h"
#include "SailInno/gaussian/diff_gs_tile_sampler.h"
#include "luisa/runtime/buffer.h"
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
	int num_gaussians, int height, int width,
	// input
	int64_t means_2d, int64_t covs_2d, int64_t depth_features, int64_t opacity_features, int64_t color_features,
	// output
	int64_t target_img_buffer) {
	// LUISA_INFO("DiffGSTileSamplerApp::forward");
	// input
	Buffer<float> means_2d_buf = mp_device->import_external_buffer<float>((void*)means_2d, num_gaussians * 2);
	Buffer<float> covs_2d_buf = mp_device->import_external_buffer<float>((void*)covs_2d, num_gaussians * 3);
	Buffer<float> depth_features_buf = mp_device->import_external_buffer<float>((void*)depth_features, num_gaussians * 1);
	Buffer<float> opacity_features_buf = mp_device->import_external_buffer<float>((void*)opacity_features, num_gaussians * 1);
	Buffer<float> color_features_buf = mp_device->import_external_buffer<float>((void*)color_features, num_gaussians * 3);

	// output
	Buffer<float> target_img_buf = mp_device->import_external_buffer<float>((void*)target_img_buffer, width * height * 3);

	// save for background
	m_num_gaussians = num_gaussians;
	m_height = height;
	m_width = width;

	mp_sampler->forward_impl(
		*mp_device, *mp_stream,
		num_gaussians, height, width,
		means_2d_buf,
		covs_2d_buf,
		depth_features_buf,
		opacity_features_buf,
		color_features_buf,
		target_img_buf);
}

void DiffGSTileSamplerApp::backward(
	// input
	int64_t dL_dpix,
	// params
	int64_t covs_2d,
	int64_t opacity_features,
	int64_t color_features,
	// output
	int64_t dL_d_means_2d,
	int64_t dL_d_covs_2d,
	int64_t dL_d_opacity_features,
	int64_t dL_d_color_features) {
	// LUISA_INFO("DiffGSTileSamplerApp::backward with {}, {}, {}", m_num_gaussians, m_height, m_width);
	// input
	Buffer<float> dL_dpix_buf = mp_device->import_external_buffer<float>((void*)dL_dpix, m_height * m_width * 3);
	// output
	Buffer<float> dL_d_means_2d_buf = mp_device->import_external_buffer<float>((void*)dL_d_means_2d, m_num_gaussians * 2);
	Buffer<float> dL_d_covs_2d_buf = mp_device->import_external_buffer<float>((void*)dL_d_covs_2d, m_num_gaussians * 3);
	Buffer<float> dL_d_opacity_features_buf = mp_device->import_external_buffer<float>((void*)dL_d_opacity_features, m_num_gaussians * 1);
	Buffer<float> dL_d_color_features_buf = mp_device->import_external_buffer<float>((void*)dL_d_color_features, m_num_gaussians * 3);

	// params
	Buffer<float> covs_2d_buf = mp_device->import_external_buffer<float>((void*)covs_2d, m_num_gaussians * 3);
	Buffer<float> opacity_features_buf = mp_device->import_external_buffer<float>((void*)opacity_features, m_num_gaussians * 1);
	Buffer<float> color_features_buf = mp_device->import_external_buffer<float>((void*)color_features, m_num_gaussians * 3);

	CommandList cmdlist;
	mp_sampler->backward_impl(
		*mp_device, cmdlist,
		// input
		dL_dpix_buf,
		// params
		covs_2d_buf,
		opacity_features_buf,
		color_features_buf,
		// output
		dL_d_means_2d_buf,
		dL_d_covs_2d_buf,
		dL_d_opacity_features_buf,
		dL_d_color_features_buf);

	(*mp_stream) << cmdlist.commit() << synchronize();
}

}// namespace sail::inno::app
