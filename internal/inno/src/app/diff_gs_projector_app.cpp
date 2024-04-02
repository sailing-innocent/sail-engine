/**
 * @file app/diff_render/diff_gaussian_projector_app.cpp
 * @author sailing-innocent
 * @date 2024-03-07
 * @brief the gaussian scene render app impl
 */

#include "SailInno/app/diff_gs_projector_app.h"

#include "SailInno/util/misc/mat_helper.h"
#include "SailInno/util/camera.h"

using namespace luisa;
using namespace luisa::compute;

namespace sail::inno::app {

void DiffGSProjectorApp::create(luisa::string& cwd, luisa::string& device_name) {
	LUISA_INFO("DiffGSProjectorApp::create");
	Context context{cwd.c_str()};
	mp_device = luisa::make_shared<Device>(context.create_device(device_name.c_str()));
	mp_stream = luisa::make_shared<Stream>(mp_device->create_stream());
	mp_projector = luisa::make_unique<gaussian::DiffGaussianProjector>();
	mp_projector->create(*mp_device);
}

void DiffGSProjectorApp::forward(
	int num_gaussians, int sh_deg, int max_sh_deg, float scale_modifier,
	// input
	int64_t xyz,
	int64_t feature,
	int64_t scale,
	int64_t rotq,
	// output
	int64_t means_2d,
	int64_t covs_2d,
	int64_t depth_features,
	int64_t color_features,
	// camera
	std::array<float, 3> cam_pos, std::array<float, 16> view_matrix_arr) {
	CommandList cmdlist;
	auto view_matrix = arr16_mat44(view_matrix_arr);
	auto cam_pos_float = make_float3(cam_pos[0], cam_pos[1], cam_pos[2]);
	Camera cam{cam_pos_float};
	cam._external_matrix = true;
	cam._view_matrix = view_matrix;
	// input
	Buffer<float> xyz_buf = mp_device->import_external_buffer<float>((void*)xyz, num_gaussians * 3);
	Buffer<float> feature_buf = mp_device->import_external_buffer<float>((void*)feature, num_gaussians * 3);
	Buffer<float> scale_buf = mp_device->import_external_buffer<float>((void*)scale, num_gaussians * 3);
	Buffer<float> rotq_buf = mp_device->import_external_buffer<float>((void*)rotq, num_gaussians * 4);

	// output
	Buffer<float> means_2d_buf = mp_device->import_external_buffer<float>((void*)means_2d, num_gaussians * 2);
	Buffer<float> depth_features_buf = mp_device->import_external_buffer<float>((void*)depth_features, num_gaussians * 1);
	Buffer<float> color_features_buf = mp_device->import_external_buffer<float>((void*)color_features, num_gaussians * 3);
	Buffer<float> covs_2d_buf = mp_device->import_external_buffer<float>((void*)covs_2d, num_gaussians * 3);

	mp_projector->forward_impl(
		*mp_device, cmdlist,
		num_gaussians, sh_deg, max_sh_deg, scale_modifier,
		xyz_buf,
		feature_buf,
		scale_buf,
		rotq_buf,
		cam,
		means_2d_buf,
		depth_features_buf,
		color_features_buf,
		covs_2d_buf);
	(*mp_stream) << cmdlist.commit();
}

void DiffGSProjectorApp::backward() {
	LUISA_INFO("DiffGSProjectorApp::backward");
}

void DiffGSProjectorApp::sync() noexcept {
	(*mp_stream) << synchronize();
}

}// namespace sail::inno::app
