/**
 * @file app/diff_render/gaussian_splatter_app.cpp
 * @author sailing-innocent
 * @date 2023-12-27
 * @brief the gaussian scene render app impl
 */
#include "SailInno/app/reprod_gs_app.h"

#include <luisa/runtime/buffer.h>

#include "SailInno/util/misc/mat_helper.h"
#include "SailInno/util/camera.h"

using namespace luisa;
using namespace luisa::compute;

namespace sail::inno::app {

void ReprodGSApp::create(luisa::string& cwd, luisa::string& device_name) {
	LUISA_INFO("ReprodGSApp::create");
	Context context{cwd.c_str()};
	mp_device = luisa::make_shared<Device>(context.create_device(device_name.c_str()));
	mp_stream = luisa::make_shared<Stream>(mp_device->create_stream());
	mp_render = luisa::make_unique<render::ReprodGS>();
	mp_render->create(*mp_device);
}

void ReprodGSApp::forward(
	int height, int width,
	int64_t target_img,
	int P, int sh_deg, int max_sh_deg,
	int64_t xyz, int64_t feat, int64_t opacity, int64_t scales, int64_t rotqs, float scale_modifier,
	std::array<float, 3> cam_pos, float fov_rad, std::array<float, 16> view_matrix_arr, std::array<float, 16> proj_matrix_arr) {
	m_width = width;
	m_height = height;
	m_P = P;
	m_sh_deg = sh_deg;
	m_max_sh_deg = max_sh_deg;

	// LUISA_INFO("ReprodGSApp::forward construct buffer");
	Buffer<float> xyz_buf = mp_device->import_external_buffer<float>((void*)xyz, P * 3);
	int feat_dim = (max_sh_deg + 1) * (max_sh_deg + 1) * 3;
	Buffer<float> feat_buf = mp_device->import_external_buffer<float>((void*)feat, P * feat_dim);
	Buffer<float> opacity_buf = mp_device->import_external_buffer<float>((void*)opacity, P);

	Buffer<float> target_img_buf = mp_device->import_external_buffer<float>((void*)target_img, width * height * 3);

	Buffer<float> scale_buf = mp_device->import_external_buffer<float>((void*)scales, P * 3);
	Buffer<float> rotq_buf = mp_device->import_external_buffer<float>((void*)rotqs, P * 4);

	auto view_matrix = arr16_mat44(view_matrix_arr);
	auto proj_matrix = arr16_mat44(proj_matrix_arr);
	auto cam_pos_float3 = make_float3(cam_pos[0], cam_pos[1], cam_pos[2]);
	// LUISA_INFO("ReprodGSApp::forward");
	Camera cam{cam_pos_float3};
	cam._external_matrix = true;
	cam._view_matrix = view_matrix;
	cam._proj_matrix = proj_matrix;
	cam.set_aspect_ratio((float)width / (float)height);
	cam.set_fov_rad(fov_rad);

	mp_render->forward_impl(
		*mp_device,
		*mp_stream,
		height, width,
		target_img_buf.view(),
		P, sh_deg, max_sh_deg,
		xyz_buf.view(), feat_buf.view(), opacity_buf.view(),
		scale_buf.view(), rotq_buf.view(),
		scale_modifier,
		cam);
}

void ReprodGSApp::backward(
	// input
	int64_t dL_d_pix,
	// output
	int64_t dL_d_xyz,
	int64_t dL_d_feature,
	int64_t dL_d_opacity,
	int64_t dL_d_scale,
	int64_t dL_d_rotq,
	// params
	int64_t target_img_buffer,// hwc
	int64_t xyz_buffer,
	int64_t feature_buffer,// for color
	int64_t opacity_buffer,
	int64_t scale_buffer,
	int64_t rotq_buffer) {

	Buffer<float> xyz_buf = mp_device->import_external_buffer<float>((void*)xyz_buffer, m_P * 3);
	Buffer<float> dL_d_xyz_buf = mp_device->import_external_buffer<float>((void*)dL_d_xyz, m_P * 3);

	int feat_dim = (m_max_sh_deg + 1) * (m_max_sh_deg + 1) * 3;
	Buffer<float> feat_buf = mp_device->import_external_buffer<float>((void*)feature_buffer, m_P * feat_dim);
	Buffer<float> dL_d_feat_buf = mp_device->import_external_buffer<float>((void*)dL_d_feature, m_P * feat_dim);

	Buffer<float> opacity_buf = mp_device->import_external_buffer<float>((void*)opacity_buffer, m_P);
	Buffer<float> dL_d_op_buf = mp_device->import_external_buffer<float>((void*)dL_d_opacity, m_P);

	Buffer<float> target_img_buf = mp_device->import_external_buffer<float>((void*)target_img_buffer, m_width * m_height * 3);
	Buffer<float> dL_d_pix_buf = mp_device->import_external_buffer<float>((void*)dL_d_pix, m_width * m_height * 3);

	Buffer<float> scale_buf = mp_device->import_external_buffer<float>((void*)scale_buffer, m_P * 3);
	Buffer<float> dL_d_scale_buf = mp_device->import_external_buffer<float>((void*)dL_d_scale, m_P * 3);
	Buffer<float> rotq_buf = mp_device->import_external_buffer<float>((void*)rotq_buffer, m_P * 4);
	Buffer<float> dL_d_rotq_buf = mp_device->import_external_buffer<float>((void*)dL_d_rotq, m_P * 4);

	// LUISA_INFO("ReprodGSApp::backward");
	mp_render->backward_impl(
		(*mp_device),
		(*mp_stream),
		// input
		dL_d_pix_buf.view(),
		// output
		dL_d_xyz_buf.view(),
		dL_d_feat_buf.view(),
		dL_d_op_buf.view(),
		dL_d_scale_buf.view(),
		dL_d_rotq_buf.view(),
		// params
		target_img_buf.view(),// hwc
		xyz_buf.view(),
		feat_buf.view(),// for color
		opacity_buf.view(),
		scale_buf.view(),
		rotq_buf.view());
}

}// namespace sail::inno::app